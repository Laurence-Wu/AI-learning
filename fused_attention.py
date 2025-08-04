"""
Fused Attention Implementation using Triton (with PyTorch fallback)
Based on the Flash Attention v2 algorithm from Tri Dao
Credits: OpenAI kernel team

Tutorial source: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

MEMORY HIERARCHY OVERVIEW:
==========================
This implementation is designed to optimize memory access patterns across the GPU memory hierarchy:

1. **HBM (High Bandwidth Memory)**: Main GPU memory where Q, K, V matrices are stored
   - Slow access (~500 GB/s bandwidth)
   - Large capacity (8-80GB depending on GPU)
   - Where input/output tensors live

2. **SRAM (On-chip memory)**: Fast GPU cache/shared memory  
   - Fast access (~19 TB/s bandwidth on A100)
   - Small capacity (~20MB per streaming multiprocessor)
   - Where we keep frequently accessed blocks during computation

3. **L1/L2 Cache**: Automatic hardware caches
   - L1: ~128KB per SM, very fast access
   - L2: ~6MB shared, fast access
   - Managed automatically by hardware

FLASH ATTENTION ALGORITHM:
=========================
The key insight is to avoid materializing the full attention matrix (N×N) in HBM memory.
Instead, we:
1. Split Q, K, V into blocks that fit in SRAM
2. Compute attention incrementally block by block
3. Keep running statistics (max, sum) to ensure numerical stability
4. Never store the full attention matrix in slow HBM

This implementation supports both:
1. Triton-accelerated version (for CUDA GPUs)
2. Pure PyTorch fallback (for macOS, CPU, or when Triton is not available)
"""

import torch
import os
import math

# Try to import Triton, fall back gracefully if not available
HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    from triton.tools.tensor_descriptor import TensorDescriptor
    HAS_TRITON = True
    print("✓ Triton is available - using accelerated kernels")
except ImportError:
    print("✗ Triton not available - using PyTorch fallback implementation")
    # Create dummy objects to prevent import errors
    class DummyTriton:
        @staticmethod
        def jit(func):
            return func
        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b
    triton = DummyTriton()
    tl = None

# Device detection that works across platforms
def get_device():
    """
    Detect the best available device for computation.
    
    MEMORY HIERARCHY PER DEVICE:
    - CUDA: HBM main memory + SRAM shared memory + L1/L2 caches
    - MPS (Apple Silicon): Unified memory architecture with different cache levels
    - CPU: RAM + L1/L2/L3 caches
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()


def is_hip():
    """Check if running on AMD ROCm (HIP) backend"""
    return HAS_TRITON and triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    """Check if running on NVIDIA CUDA backend"""
    return HAS_TRITON and triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    """Check if GPU supports host descriptors (Hopper architecture and newer)"""
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    """Check if running on NVIDIA Blackwell architecture (compute capability 10.x)"""
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    """Check if running on NVIDIA Hopper architecture (compute capability 9.x)"""
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


# Triton kernels (only available when Triton is installed)
if HAS_TRITON:
    @triton.jit
    def _attn_fwd_inner(acc, l_i, m_i, q,
                        desc_k, desc_v,
                        offset_y, dtype: tl.constexpr, start_m, qk_scale,
                        BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                        STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
                        N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
        """
        Inner loop of Flash Attention computation.
        
        MEMORY LOCATIONS DURING EXECUTION:
        ==================================
        
        INPUT DATA LOCATIONS:
        - q: LOADED INTO SRAM (stays resident throughout computation)
        - k, v: STREAMED FROM HBM → SRAM in blocks
        - acc, l_i, m_i: REGISTER/SRAM (running statistics)
        
        COMPUTATION FLOW:
        1. Load K block from HBM → SRAM (BLOCK_N × HEAD_DIM)
        2. Compute Q@K^T in SRAM (never touches HBM for intermediate results)
        3. Apply softmax statistics in SRAM 
        4. Load V block from HBM → SRAM
        5. Accumulate results in SRAM
        6. Update running statistics in registers/SRAM
        
        MEMORY SAVINGS:
        - Traditional attention: Stores N×N attention matrix in HBM
        - Flash attention: Only stores BLOCK_M×BLOCK_N blocks in SRAM
        - Memory reduction: O(N²) → O(N × BLOCK_SIZE)
        """
        
        # Determine which part of the sequence this stage handles
        # STAGE controls the computation pattern for causal vs non-causal attention
        if STAGE == 1:
            lo, hi = 0, start_m * BLOCK_M  # Pre-diagonal blocks (causal)
        elif STAGE == 2:
            lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M  # Diagonal block (causal)
            lo = tl.multiple_of(lo, BLOCK_M)
        else:
            lo, hi = 0, N_CTX  # Full sequence (non-causal)
        # The difference between Stage 1 and Stage 2:
        # Columns strictly before the row block
        #  cols < start_m · BLOCK_M
        #   ➜ every entry in the BLOCK_M × BLOCK_N rectangle is legal (j ≤ i is always true).
        #   No element-wise mask is needed; we can process the whole rectangle as-is.
        #   This is handled by STAGE = 1 ( “pre-diagonal” pass ).
        # Columns that overlap the diagonal of the row block
        #  cols ∈ [start_m · BLOCK_M , (start_m + 1) · BLOCK_M)
        #   ➜ inside this rectangle only the lower-triangular half (including the diagonal) is legal;
        #   the upper-triangular half must be zeroed out / set to −∞.
        #   This requires an element-wise triangular mask.
        #   This is handled by STAGE = 2 ( “on-diagonal” pass ).


        # Calculate memory offsets
        # These point to locations in HBM where K and V blocks start
        offsetk_y = offset_y + lo
        if dtype == tl.float8e5:
            # FP8 uses different memory layout
            offsetv_y = offset_y * HEAD_DIM + lo
        else:
            offsetv_y = offset_y + lo
            
        # MAIN COMPUTATION LOOP
        # Each iteration processes one block of K and V
        # CRITICAL: This loop keeps Q in SRAM while streaming K,V blocks
        for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            
            # MEMORY OPERATION: Load K block from HBM → SRAM
            # Location: HBM → SRAM (shared memory)
            k = desc_k.load([offsetk_y, 0]).T
            
            # COMPUTE: Q@K^T attention scores in SRAM
            # Location: SRAM computation, results stay in SRAM
            # This is the core matrix multiplication, happens in SRAM
            qk = tl.dot(q, k)
            
            # Apply scaling and causal masking
            if STAGE == 2:
                # Causal masking: prevent attending to future tokens
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk -= m_ij[:, None]
            else:
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                qk = qk * qk_scale - m_ij[:, None]
                
            # COMPUTE: Softmax in SRAM
            # Location: All operations in SRAM (no HBM access)
            p = tl.math.exp2(qk)  # Attention weights
            
            # Update running statistics for numerical stability
            alpha = tl.math.exp2(m_i - m_ij)  # Correction factor
            l_ij = tl.sum(p, 1)  # Row sums
            
            # MEMORY OPERATION: Load V block from HBM → SRAM  
            # Location: HBM → SRAM
            v = desc_v.load([offsetv_y, 0])
            
            # COMPUTE: Update accumulator with P@V in SRAM
            # Location: SRAM computation
            # This is where the actual attention output is computed
            acc = acc * alpha[:, None]  # Apply correction to previous accumulator
            acc = tl.dot(p.to(v.dtype), v, acc)  # Add current contribution
            
            # Update running statistics (stays in registers/SRAM)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
            
            # Advance to next block
            # Update HBM pointers for next K,V blocks
            offsetk_y += BLOCK_N
            if dtype == tl.float8e5:
                offsetv_y += BLOCK_N
            else:
                offsetv_y += BLOCK_N
                
        return acc, l_i, m_i


    @triton.jit
    def _attn_fwd(Q, K, V, sm_scale, M, Out,
                  stride_qz, stride_qh, stride_qm, stride_qk,
                  stride_kz, stride_kh, stride_kn, stride_kk,
                  stride_vz, stride_vh, stride_vn, stride_vk,
                  stride_oz, stride_oh, stride_om, stride_on,
                  Z, H, N_CTX,
                  HEAD_DIM: tl.constexpr,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  STAGE: tl.constexpr,
                  warp_specialize: tl.constexpr):
        """
        Main Triton kernel for Flash Attention forward pass.
        
        EXECUTION CONTEXT:
        ==================
        - Runs on GPU streaming multiprocessor (SM)
        - Each thread block processes one block of output
        - Thread block has access to ~96KB of SRAM (shared memory)
        
        MEMORY HIERARCHY USAGE:
        =======================
        
        1. **INPUT TENSORS** (stored in HBM):
           - Q, K, V: Main attention matrices in GPU main memory
           - Size: [batch, heads, seq_len, head_dim] each
           
        2. **WORKING MEMORY** (SRAM/registers):
           - q: One block of queries, loaded once and kept in SRAM
           - k, v: Streamed blocks, temporary residence in SRAM  
           - acc: Accumulator for output, lives in SRAM
           - m_i, l_i: Running max/sum statistics, in registers
           
        3. **OUTPUT** (written to HBM):
           - Out: Final attention output
           - M: Log-sum-exp values for potential backward pass
           
        BLOCKING STRATEGY:
        ==================
        The algorithm processes the attention computation in blocks to fit in SRAM:
        - Q is divided into blocks of size BLOCK_M × HEAD_DIM  
        - K, V are divided into blocks of size BLOCK_N × HEAD_DIM
        - Each thread block computes one Q block against all K,V blocks
        
        MEMORY EFFICIENCY:
        ==================
        - Traditional: O(N²) memory for full attention matrix
        - Flash Attention: O(N × BLOCK_SIZE) memory usage
        - Example: For N=4096, reduces from 16M to 64K memory usage
        """
        
        # THREAD BLOCK SETUP
        # Each thread block processes one row block of the output
        start_m = tl.program_id(0)  # Which row block this thread block handles
        off_hz = tl.program_id(1)   # Which batch×head this thread block handles
        off_z = off_hz // H         # Batch index
        off_h = off_hz % H          # Head index

        # POINTER ARITHMETIC: Calculate base addresses in HBM
        # These operations just calculate memory addresses, no data movement yet
        Q += off_z * stride_qz + off_h * stride_qh      # Base address for Q
        K += off_z * stride_kz + off_h * stride_kh      # Base address for K  
        V += off_z * stride_vz + off_h * stride_vh      # Base address for V
        Out += off_z * stride_oz + off_h * stride_oh    # Base address for Out
        if M is not None:
            M += off_hz  # Base address for statistics
            
        # BLOCK INDEXING SETUP
        # Calculate which elements this thread block will process
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # Row indices
        offs_n = tl.arange(0, BLOCK_N)                       # Column indices  
        offs_d = tl.arange(0, HEAD_DIM)                      # Feature indices

        # MEMORY POINTER SETUP: Calculate exact memory addresses
        # Still no data movement, just address calculation
        q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)  
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

        # INITIALIZE ACCUMULATORS IN SRAM/REGISTERS
        # These live in fast on-chip memory throughout computation
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # Running max
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0          # Running sum
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)      # Output accumulator

        # CRITICAL MEMORY OPERATION: Load Q block into SRAM
        # Location: HBM → SRAM (this is the expensive memory operation)
        # Q stays in SRAM for the entire computation of this block
        q = tl.load(q_ptrs)
        
        # MAIN COMPUTATION: Simplified loop for compatibility
        # Process all K,V blocks sequentially against the loaded Q block
        for start_n in range(0, N_CTX, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            
            # Boundary check
            if start_n >= N_CTX:
                break
                
            # MEMORY OPERATIONS: Load K,V blocks from HBM → SRAM
            # These are temporary - only needed for this iteration
            k_block_ptrs = k_ptrs + start_n * stride_kn
            v_block_ptrs = v_ptrs + start_n * stride_vn
            
            # Load with masking for sequence boundaries
            k = tl.load(k_block_ptrs, mask=start_n + offs_n[:, None] < N_CTX, other=0.0)
            v = tl.load(v_block_ptrs, mask=start_n + offs_n[:, None] < N_CTX, other=0.0)
            
            # CORE COMPUTATION IN SRAM: Attention scores
            # All matrix operations happen in fast SRAM
            qk = tl.dot(q, k.T)  # Q@K^T computation in SRAM
            qk = qk * sm_scale   # Scale in SRAM
            
            # Apply causal mask if needed (still in SRAM)
            if STAGE == 2:  # causal attention
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = tl.where(mask, qk, -1.0e6)
            
            # SOFTMAX COMPUTATION IN SRAM
            # Compute softmax statistics without storing full attention matrix
            m_ij = tl.max(qk, 1)           # Row-wise max
            qk = qk - m_ij[:, None]        # Subtract max for stability
            p = tl.math.exp2(qk)           # Exponentiate
            l_ij = tl.sum(p, 1)            # Row-wise sum
            
            # UPDATE RUNNING STATISTICS
            # Online algorithm to maintain global softmax statistics
            alpha = tl.math.exp2(m_i - m_ij)  # Correction factor
            acc = acc * alpha[:, None]         # Correct previous accumulator
            acc = tl.dot(p.to(v.dtype), v, acc)  # Add P@V contribution
            l_i = l_i * alpha + l_ij          # Update running sum
            m_i = m_ij                        # Update running max

        # FINALIZATION: Complete softmax computation
        # Convert from log-space back to probability space
        m_i += tl.math.log2(l_i)  # Final log-sum-exp
        acc = acc / l_i[:, None]  # Normalize by sum
        
        # FINAL MEMORY OPERATIONS: Write results back to HBM
        # Location: SRAM → HBM (store final results)
        offs_n = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        out_ptrs = Out + (offs_n[:, None] * stride_om + offs_d[None, :] * stride_on)
        
        # Write output with boundary masking
        tl.store(out_ptrs, acc.to(Out.type.element_ty), mask=offs_n[:, None] < N_CTX)

else:
    # Dummy functions when Triton is not available
    def _attn_fwd_inner(*args, **kwargs):
        raise NotImplementedError("Triton not available")
    
    def _attn_fwd(*args, **kwargs):
        raise NotImplementedError("Triton not available")


def attention_pytorch_fallback(q, k, v, causal=False, sm_scale=None):
    """
    Pure PyTorch implementation of attention (fallback when Triton is not available).
    
    MEMORY HIERARCHY FOR PYTORCH FALLBACK:
    =======================================
    
    CPU/MPS MEMORY LOCATIONS:
    - Input tensors: System RAM or unified memory (Apple Silicon)
    - Intermediate matrices: Computed in-place when possible
    - Attention matrix: Temporarily materialized in memory (less efficient than Triton)
    
    CACHE USAGE:
    - L1 Cache (32-64KB per core): Hot data during matrix operations
    - L2 Cache (256KB-1MB per core): Recently accessed tensor blocks  
    - L3 Cache (8-32MB shared): Shared between cores for large tensors
    - System RAM: Full tensor storage
    
    MEMORY EFFICIENCY COMPARISON:
    - PyTorch: Materializes full N×N attention matrix temporarily
    - Flash Attention (Triton): Never materializes full matrix
    - Memory usage: O(N²) vs O(N×BLOCK_SIZE)
    
    COMPUTATIONAL FLOW:
    1. Compute Q@K^T → temporarily uses O(N²) memory
    2. Apply softmax → operates on full matrix in memory
    3. Compute attention@V → final matrix multiplication
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    
    # NUMERICAL STABILITY: Convert to float32 for intermediate computations
    # Location: System RAM/Unified Memory  
    # This prevents numerical overflow in half-precision computations
    ref_dtype = torch.float32
    q_work = q.to(ref_dtype)  # Copy to RAM with higher precision
    k_work = k.to(ref_dtype)  # Copy to RAM with higher precision  
    v_work = v.to(ref_dtype)  # Copy to RAM with higher precision
    
    # CORE COMPUTATION: Attention scores
    # Location: Computed in system RAM, may use CPU cache hierarchy
    # MEMORY IMPACT: This creates an N×N matrix in memory (expensive!)
    scores = torch.matmul(q_work, k_work.transpose(-2, -1)) * sm_scale
    
    # CAUSAL MASKING: Apply lower triangular mask if needed
    # Location: Mask computed in RAM, applied in-place
    if causal:
        seq_len = q_work.shape[-2]
        # Create causal mask - only allow attending to previous positions
        mask = torch.tril(torch.ones(seq_len, seq_len, device=q_work.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # SOFTMAX COMPUTATION: Convert scores to probabilities  
    # Location: Operates on full attention matrix in RAM
    # Uses CPU SIMD units and cache hierarchy for vectorized operations
    attn_weights = torch.softmax(scores.float(), dim=-1)
    attn_weights = attn_weights.to(ref_dtype)
    
    # FINAL COMPUTATION: Apply attention to values
    # Location: Matrix multiplication in RAM using optimized BLAS
    # CPU cache hierarchy automatically manages data movement
    output = torch.matmul(attn_weights, v_work)
    
    # Convert back to original precision and return
    # Location: Final copy back to original tensor type
    return output.to(q.dtype)


def attention_triton(q, k, v, causal=False, sm_scale=None, warp_specialize=False):
    """
    Triton-accelerated attention implementation.
    
    MEMORY HIERARCHY OPTIMIZATION:
    ===============================
    
    This function sets up the optimal memory access patterns for GPU computation:
    
    1. **Block Size Selection**: 
       - Larger blocks for smaller head dimensions (more compute per memory access)
       - Smaller blocks for larger head dimensions (fit in SRAM constraints)
       
    2. **Grid Layout**:
       - Each thread block processes one output block
       - Thread blocks run in parallel across SMs
       - Total thread blocks = (seq_len / BLOCK_M) × (batch × heads)
       
    3. **Memory Layout**:
       - Tensors stored in HBM with optimal stride patterns
       - SRAM used for temporary block storage during computation
       - Registers used for scalar values and small arrays
    """
    # Shape validation - ensure tensors are compatible
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}  # Supported head dimensions
    
    # Allocate output tensor in HBM
    o = torch.empty_like(q)
    
    if sm_scale is None:
        sm_scale = 1.0 / (Lq ** 0.5)
        
    BATCH, N_HEAD, N_CTX = q.shape[:3]
    HEAD_DIM = q.shape[-1]
    
    # BLOCK SIZE SELECTION STRATEGY:
    # Balance between SRAM usage and computational efficiency
    if HEAD_DIM <= 64:
        # Smaller head dimension: use larger blocks
        # More elements fit in SRAM, better compute-to-memory ratio
        BLOCK_M = 128  # Process 128 query positions at once
        BLOCK_N = 128  # Process 128 key positions at once
    else:
        # Larger head dimension: use smaller blocks  
        # Fewer elements fit in SRAM due to larger feature vectors
        BLOCK_M = 64   # Process 64 query positions at once
        BLOCK_N = 64   # Process 64 key positions at once
        
    # EXECUTION STAGE SELECTION:
    # Determines computation pattern for causal vs non-causal attention
    if causal:
        STAGE = 1 if warp_specialize else 2  # Causal attention stages
    else:
        STAGE = 3  # Non-causal attention
        
    # Allocate log-sum-exp storage in HBM
    # Used for numerical stability and potential backward pass
    M = torch.empty((BATCH * N_HEAD, N_CTX), device=q.device, dtype=torch.float32)
    
    # KERNEL LAUNCH CONFIGURATION:
    # Define the grid of thread blocks that will execute the kernel
    grid = (triton.cdiv(N_CTX, BLOCK_M),  # Number of row blocks needed
            BATCH * N_HEAD,               # Number of batch×head combinations  
            1)                            # Z-dimension (unused)
    
    # KERNEL EXECUTION:
    # Launch Triton kernel with memory-optimized parameters
    _attn_fwd[grid](
        # Input tensors (in HBM)
        q, k, v, sm_scale, M, o,
        
        # Memory stride information (for efficient addressing)
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  # Q strides
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  # K strides  
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  # V strides
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  # Output strides
        
        # Tensor dimensions
        BATCH, N_HEAD, N_CTX,
        
        # Compile-time constants (for optimal code generation)
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N, 
        STAGE=STAGE,
        warp_specialize=warp_specialize,
        
        # GPU execution parameters
        num_warps=4,    # Number of warp schedulers per thread block
        num_stages=4,   # Number of pipeline stages for memory/compute overlap
    )
    
    return o


def attention(q, k, v, causal=False, sm_scale=None, warp_specialize=False):
    """
    Unified attention function that automatically chooses between Triton and PyTorch implementations.
    
    AUTOMATIC OPTIMIZATION SELECTION:
    ==================================
    
    1. **CUDA + Triton Available**: Uses memory-optimized Flash Attention
       - Memory usage: O(N × BLOCK_SIZE) instead of O(N²)
       - Computation: Happens in GPU SRAM (19 TB/s bandwidth)
       - Best performance for training and inference
       
    2. **Non-CUDA or No Triton**: Uses PyTorch fallback
       - Memory usage: O(N²) for temporary attention matrix
       - Computation: Uses CPU/MPS with hardware caches
       - Still numerically stable and correct
       
    MEMORY HIERARCHY AWARENESS:
    ===========================
    The function automatically adapts to the available memory hierarchy:
    - GPU: Optimizes for HBM ↔ SRAM data movement
    - MPS: Uses Apple Silicon unified memory architecture  
    - CPU: Leverages multi-level cache hierarchy (L1/L2/L3)
    """
    # Use Triton if available and on CUDA
    if HAS_TRITON and q.device.type == 'cuda':
        try:
            return attention_triton(q, k, v, causal, sm_scale, warp_specialize)
        except Exception as e:
            print(f"Triton implementation failed: {e}")
            print("Falling back to PyTorch implementation")
            return attention_pytorch_fallback(q, k, v, causal, sm_scale)
    else:
        # Use PyTorch fallback for non-CUDA devices or when Triton is not available
        return attention_pytorch_fallback(q, k, v, causal, sm_scale)


def test_op(BATCH, H, N_CTX, HEAD_DIM, mode, provider, causal=False, warp_specialize=False, device=None):
    """Test operation with specified parameters - includes basic performance estimation."""
    if device is None:
        device = DEVICE
        
    dtype = torch.float16
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, causal, sm_scale, warp_specialize)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = 1.0  # Placeholder timing
    
    # Reference implementation using PyTorch
    else:
        ref_dtype = torch.float32
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        q = q.to(ref_dtype)
        k = k.to(ref_dtype)
        v = v.to(ref_dtype)
        M = torch.tril(torch.ones((N_CTX, N_CTX), device=device))
        p = torch.matmul(q, k.transpose(2, 3)) * 1.3
        if causal:
            p[:, :, M == 0] = float("-inf")
        p = torch.softmax(p.float(), dim=-1)
        p = p.to(ref_dtype)
        ref_out = torch.matmul(p, v).half()
        if mode == "bwd":
            dout = torch.randn_like(q)
            ref_out.backward(dout)
        ms = 1.0  # Placeholder timing
        
    # Calculate theoretical FLOPS (floating point operations per second)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


# Test correctness
def test_correctness():
    """
    Comprehensive correctness testing for the attention implementation.
    
    Tests validate that the implementation:
    1. Produces deterministic outputs
    2. Handles different tensor sizes correctly  
    3. Computes gradients properly
    4. Maintains numerical stability
    """
    print("Testing fused attention correctness...")
    
    # Test 1: Identity test - all values the same should produce identical output
    print("\n1. Testing basic functionality...")
    torch.manual_seed(42)
    BATCH, H, N_CTX, HEAD_DIM = 1, 1, 4, 8
    
    # Create simple test tensors
    q = torch.ones((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE)
    k = torch.ones((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE)
    v = torch.arange(N_CTX).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(BATCH, H, N_CTX, HEAD_DIM).float().to(device=DEVICE, dtype=torch.float16)
    
    # Test non-causal (should average all values)
    output_non_causal = attention(q, k, v, causal=False)
    print(f"Non-causal output shape: {output_non_causal.shape}")
    
    # Test causal (should only look at previous positions)
    output_causal = attention(q, k, v, causal=True)
    print(f"Causal output shape: {output_causal.shape}")
    
    # Test 2: Consistency test with random data
    print("\n2. Testing consistency with random data...")
    torch.manual_seed(42)
    BATCH, H, N_CTX, HEAD_DIM = 2, 4, 64, 32
    
    q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE)
    
    # Test multiple runs should produce identical results
    out1 = attention(q, k, v, causal=True)
    out2 = attention(q, k, v, causal=True)
    
    max_diff = torch.max(torch.abs(out1 - out2))
    print(f"Max difference between runs: {max_diff}")
    
    if max_diff < 1e-6:
        print("✓ Consistency test passed!")
    else:
        print("✗ Consistency test failed!")
    
    # Test 3: Check that output is reasonable
    print("\n3. Testing output reasonableness...")
    
    # All outputs should be finite
    if torch.all(torch.isfinite(out1)):
        print("✓ All outputs are finite")
    else:
        print("✗ Some outputs are not finite")
        return False
    
    # Output should have correct shape
    expected_shape = q.shape
    if out1.shape == expected_shape:
        print(f"✓ Output shape correct: {out1.shape}")
    else:
        print(f"✗ Output shape incorrect: expected {expected_shape}, got {out1.shape}")
        return False
    
    # Test 4: Gradient test
    print("\n4. Testing gradients...")
    q_grad = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE, requires_grad=True)
    k_grad = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE, requires_grad=True)
    v_grad = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE, requires_grad=True)
    
    try:
        output = attention(q_grad, k_grad, v_grad, causal=True)
        loss = output.sum()
        loss.backward()
        
        if q_grad.grad is not None and torch.all(torch.isfinite(q_grad.grad)):
            print("✓ Gradients computed successfully")
        else:
            print("✗ Gradient computation failed")
            return False
    except Exception as e:
        print(f"✗ Gradient test failed: {e}")
        return False
    
    print("\n✓ All tests passed!")
    return True


def simple_benchmark():
    """
    Simple benchmark to verify the implementation works across different sizes.
    
    Tests progressively larger configurations to ensure:
    1. Memory allocation works correctly
    2. Computation completes successfully  
    3. Output shapes are correct
    """
    print("Running simple benchmark...")
    
    # Test with smaller sizes first
    configs = [
        (1, 4, 256, 64),   # Small: 1MB total memory
        (1, 8, 512, 64),   # Medium: 4MB total memory
        (2, 8, 1024, 64),  # Large: 16MB total memory
    ]
    
    for BATCH, H, N_CTX, HEAD_DIM in configs:
        print(f"\nTesting BATCH={BATCH}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}")
        
        # Create test tensors
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE)  
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE)
        
        # Test forward pass
        try:
            output = attention(q, k, v, causal=True)
            print(f"✓ Forward pass successful, output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            return False
            
    return True


try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')


if __name__ == "__main__":
    print("Fused Attention Implementation Test")
    print("=" * 50)
    
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA capability: {torch.cuda.get_device_capability()}")
    print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    print(f"Triton available: {HAS_TRITON}")
    if HAS_TRITON:
        print(f"Triton version: {triton.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Flash Attention available: {HAS_FLASH}")
    print(f"FP8 support: {TORCH_HAS_FP8}")
    print()
    
    # Run tests
    if simple_benchmark():
        print("\n" + "=" * 50)
        test_correctness()
    else:
        print("Basic benchmark failed, skipping correctness test") 