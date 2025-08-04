"""
Fused RoPE Attention Implementation using Triton
Adapted from the Triton implementation of Flash Attention v2
Source: https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py

This implements a windowed attention mechanism with Rotary Position Embedding (RoPE).
The attention can switch between two different RoPE encodings based on distance from the diagonal.
"""

import torch
import triton
import triton.language as tl
import torch.utils.benchmark as benchmark

@triton.jit
def _fwd_kernel(
    Q1, Q2, K1, K2, V, sm_scale,
    L,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
):
    """
    Forward kernel for windowed RoPE attention.
    
    This kernel implements attention with:
    1. Two sets of Q/K matrices (Q1/K1 and Q2/K2) with different RoPE encodings
    2. A window parameter that determines which encoding to use based on distance
    3. Causal masking support
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh
    
    # Create block pointers for Q1, Q2 (queries with different RoPE encodings)
    Q1_block_ptr = tl.make_block_ptr(
        base=Q1 + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    Q2_block_ptr = tl.make_block_ptr(
        base=Q2 + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    # Create block pointers for K1, K2 (keys with different RoPE encodings)
    K1_block_ptr = tl.make_block_ptr(
        base=K1 + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    K2_block_ptr = tl.make_block_ptr(
        base=K2 + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    
    # Create block pointer for V (values)
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    # Initialize offsets and accumulators
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Initialize statistics for online softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # running sum
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)  # accumulator
    
    # Scale factor for attention (log_2(e) for 2^x instead of exp)
    qk_scale = sm_scale * 1.44269504
    
    # Load queries: they stay in SRAM throughout computation
    q1 = tl.load(Q1_block_ptr)
    q1 = (q1 * qk_scale).to(tl.float16)
    q2 = tl.load(Q2_block_ptr)
    q2 = (q2 * qk_scale).to(tl.float16)
    
    # Main attention computation loop
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    
    for start_n in range(lo, hi, BLOCK_N):
        # Initialize attention scores
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
        
        # Apply causal masking if needed
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        
        # Choose which Q/K pair to use based on distance from diagonal
        if start_n <= start_m * BLOCK_M - WINDOW - BLOCK_N or start_n >= (start_m + 1) * BLOCK_M + WINDOW:
            # Far from diagonal: use Q2/K2 (different RoPE encoding)
            k2 = tl.load(K2_block_ptr)
            v = tl.load(V_block_ptr)
            qk += tl.dot(q2, k2, out_dtype=tl.float16)
        elif start_n > (start_m + 1) * BLOCK_M - WINDOW and start_n < start_m * BLOCK_M + WINDOW - BLOCK_N:
            # Close to diagonal: use Q1/K1 (standard RoPE encoding)
            k1 = tl.load(K1_block_ptr)
            v = tl.load(V_block_ptr)
            qk += tl.dot(q1, k1, out_dtype=tl.float16)
        else:
            # Transition region: blend both encodings based on distance
            k1 = tl.load(K1_block_ptr)
            k2 = tl.load(K2_block_ptr)
            v = tl.load(V_block_ptr)
            qk1 = tl.dot(q1, k1, out_dtype=tl.float16)
            qk2 = tl.dot(q2, k2, out_dtype=tl.float16)
            qk += tl.where(tl.abs(offs_m[:, None] - (start_n + offs_n[None, :])) < WINDOW, qk1, qk2)
        
        # Online softmax computation
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        
        # Update accumulator and statistics
        acc_scale = l_i * 0 + alpha  # workaround for compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        
        # Update running statistics
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        
        # Advance block pointers
        K1_block_ptr = tl.advance(K1_block_ptr, (0, BLOCK_N))
        K2_block_ptr = tl.advance(K2_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    
    # Finalize and write output
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    
    # Write output
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(tl.float16))

@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    """Preprocessing for backward pass - compute delta values."""
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    
    # Load output and gradient
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    
    # Compute delta = sum(o * do) for each row
    delta = tl.sum(o * do, axis=1)
    
    # Write delta values
    tl.store(Delta + off_m, delta)

@triton.jit
def _bwd_kernel(
    Q1, Q2, K1, K2, V, sm_scale, Out, DO,
    DQ1, DQ2, DK1, DK2, DV,
    L, D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H, N_CTX,
    num_block_q, num_block_kv,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW: tl.constexpr,
):
    """Backward kernel for windowed RoPE attention."""
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    qk_scale = sm_scale * 1.44269504
    
    # Offset all pointers for current batch/head
    Q1 += off_z * stride_qz + off_h * stride_qh
    Q2 += off_z * stride_qz + off_h * stride_qh
    K1 += off_z * stride_kz + off_h * stride_kh
    K2 += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ1 += off_z * stride_qz + off_h * stride_qh
    DQ2 += off_z * stride_qz + off_h * stride_qh
    DK1 += off_z * stride_kz + off_h * stride_kh
    DK2 += off_z * stride_kz + off_h * stride_kh
    DV += off_z * stride_vz + off_h * stride_vh
    
    # Loop over key/value blocks
    for start_n in range(0, num_block_kv):
        if CAUSAL:
            lo = tl.math.max(start_n * BLOCK_N, 0)
        else:
            lo = 0
        
        # Initialize offsets
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        
        # Setup pointers
        q1_ptrs = Q1 + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        q2_ptrs = Q2 + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k1_ptrs = K1 + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        k2_ptrs = K2 + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq1_ptrs = DQ1 + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq2_ptrs = DQ2 + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        
        # Pointers to statistics
        D_ptrs = D + off_hz * N_CTX
        l_ptrs = L + off_hz * N_CTX
        
        # Initialize gradient accumulators
        dk1 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk2 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        
        # Load K, V blocks (stay in SRAM)
        k1 = tl.load(k1_ptrs)
        k2 = tl.load(k2_ptrs)
        v = tl.load(v_ptrs)
        
        # Loop over query blocks
        for start_m in range(lo, num_block_q * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            
            # Load queries
            q1 = tl.load(q1_ptrs)
            q2 = tl.load(q2_ptrs)
            
            # Recompute attention scores (same logic as forward)
            if CAUSAL:
                qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), float(0.), float("-inf"))
            else:
                qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            
            # Choose Q/K pair based on distance (same as forward)
            if start_m >= (start_n + 1) * BLOCK_N + WINDOW or start_m <= start_n * BLOCK_N - WINDOW - BLOCK_M:
                q2 = tl.load(q2_ptrs)
                qk += tl.dot(q2, tl.trans(k2))
            elif start_m > (start_n + 1) * BLOCK_N - WINDOW and start_m < start_n * BLOCK_N + WINDOW - BLOCK_M:
                q1 = tl.load(q1_ptrs)
                qk += tl.dot(q1, tl.trans(k1))
            else:
                q1 = tl.load(q1_ptrs)
                q2 = tl.load(q2_ptrs)
                qk1 = tl.dot(q1, tl.trans(k1))
                qk2 = tl.dot(q2, tl.trans(k2))
                qk += tl.where(tl.abs(offs_m_curr[:, None] - offs_n[None, :]) < WINDOW, qk1, qk2)

            qk *= qk_scale
            l_i = tl.load(l_ptrs + offs_m_curr)
            p = tl.math.exp2(qk - l_i[:, None])
            
            # Compute gradients
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(p.to(Q1.dtype.element_ty)), do)
            
            # Compute dp
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, tl.trans(v))
            
            # Compute ds
            ds = p * dp * sm_scale
            
            # Update gradients based on which Q/K pair was used
            if start_m >= (start_n + 1) * BLOCK_N + WINDOW or start_m <= start_n * BLOCK_N - WINDOW - BLOCK_M:
                dk2 += tl.dot(tl.trans(ds.to(Q1.dtype.element_ty)), q2)
                dq2 = tl.load(dq2_ptrs)
                dq2 += tl.dot(ds.to(Q1.dtype.element_ty), k2)
                tl.store(dq2_ptrs, dq2)
            elif start_m > (start_n + 1) * BLOCK_N - WINDOW and start_m < start_n * BLOCK_N + WINDOW - BLOCK_M:
                dk1 += tl.dot(tl.trans(ds.to(Q1.dtype.element_ty)), q1)
                dq1 = tl.load(dq1_ptrs)
                dq1 += tl.dot(ds.to(Q1.dtype.element_ty), k1)
                tl.store(dq1_ptrs, dq1)
            else:
                mask = (tl.abs(offs_m_curr[:, None] - offs_n[None, :]) < WINDOW)
                ds1 = tl.where(mask, ds, float(0.))
                ds2 = tl.where(mask, float(0.), ds)
                dk1 += tl.dot(tl.trans(ds1.to(Q1.dtype.element_ty)), q1)
                dk2 += tl.dot(tl.trans(ds2.to(Q1.dtype.element_ty)), q2)
                dq1 = tl.load(dq1_ptrs)
                dq2 = tl.load(dq2_ptrs)
                dq1 += tl.dot(ds1.to(Q1.dtype.element_ty), k1)
                dq2 += tl.dot(ds2.to(Q1.dtype.element_ty), k2)
                tl.store(dq1_ptrs, dq1)
                tl.store(dq2_ptrs, dq2)
            
            # Advance pointers
            dq1_ptrs += BLOCK_M * stride_qm
            dq2_ptrs += BLOCK_M * stride_qm
            q1_ptrs += BLOCK_M * stride_qm
            q2_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        
        # Write back gradients
        dk1_ptrs = DK1 + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        dk2_ptrs = DK2 + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        tl.store(dk1_ptrs, dk1)
        tl.store(dk2_ptrs, dk2)
        tl.store(dv_ptrs, dv)

class _attention(torch.autograd.Function):
    """PyTorch autograd function wrapper for the Triton RoPE attention kernels."""
    
    @staticmethod
    def forward(ctx, q1, q2, k1, k2, v, causal, sm_scale, window):
        # Validate input shapes
        Lq, Lk, Lv = q1.shape[-1], k1.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        
        o = torch.empty_like(q1)
        
        # Block size configuration
        BLOCK_M = 128
        BLOCK_N = 64 if Lk <= 64 else 32
        num_stages = 4 if Lk <= 64 else 3
        num_warps = 4
        
        # Grid configuration
        grid = (triton.cdiv(q1.shape[2], BLOCK_M), q1.shape[0] * q1.shape[1], 1)
        
        # Statistics tensor for softmax
        L = torch.empty((q1.shape[0] * q1.shape[1], q1.shape[2]), device=q1.device, dtype=torch.float32)
        
        # Launch forward kernel
        _fwd_kernel[grid](
            q1, q2, k1, k2, v, sm_scale,
            L, o,
            q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
            k1.stride(0), k1.stride(1), k1.stride(2), k1.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q1.shape[0], q1.shape[1], q1.shape[2],
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
            IS_CAUSAL=causal, WINDOW=window,
            num_warps=num_warps, num_stages=num_stages
        )

        # Save for backward
        ctx.save_for_backward(q1, q2, k1, k2, v, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.window = window
        return o

    @staticmethod
    def backward(ctx, do):
        BLOCK = 128
        q1, q2, k1, k2, v, o, L = ctx.saved_tensors
        do = do.contiguous()
        
        # Initialize gradient tensors
        dq1 = torch.zeros_like(q1, dtype=torch.float32)
        dq2 = torch.zeros_like(q2, dtype=torch.float32)
        dk1 = torch.empty_like(k1)
        dk2 = torch.empty_like(k2)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)
        
        # Preprocess for backward
        _bwd_preprocess[(triton.cdiv(q1.shape[2], BLOCK) * ctx.grid[1], )](
            o, do, delta,
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )
        
        # Launch backward kernel
        _bwd_kernel[(ctx.grid[1],)](
            q1, q2, k1, k2, v, ctx.sm_scale,
            o, do, dq1, dq2, dk1, dk2, dv,
            L, delta,
            q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
            k1.stride(0), k1.stride(1), k1.stride(2), k1.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q1.shape[0], q1.shape[1], q1.shape[2],
            triton.cdiv(q1.shape[2], BLOCK), triton.cdiv(k1.shape[2], BLOCK),
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=8,
            CAUSAL=ctx.causal, WINDOW=ctx.window,
            num_stages=1,
        )
        return dq1, dq2, dk1, dk2, dv, None, None, None

# Export the attention function
triton_attention = _attention.apply

def test_rope_attention():
    """Test the RoPE attention implementation."""
    print("Testing RoPE Attention Implementation")
    print("=" * 50)
    
    # Test configuration
    Z = 1          # batch size
    H = 8          # number of heads
    N_CTX = 1024   # sequence length
    D_HEAD = 64    # head dimension
    WINDOW = 256   # window size for RoPE switching
    sm_scale = 0.5 # attention scale
    
    print(f"Configuration:")
    print(f"  Batch size: {Z}")
    print(f"  Heads: {H}")
    print(f"  Sequence length: {N_CTX}")
    print(f"  Head dimension: {D_HEAD}")
    print(f"  Window size: {WINDOW}")
    print(f"  Scale: {sm_scale}")
    print()
    
    # Create test tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    q1 = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    q2 = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    k1 = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    k2 = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
    
    # Test forward pass
    print("Testing forward pass...")
    try:
        output = triton_attention(q1, q2, k1, k2, v, False, sm_scale, WINDOW)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Test backward pass
    print("Testing backward pass...")
    try:
        grad = torch.randn_like(output)
        output.backward(grad)
        print("✓ Backward pass successful")
        print(f"  q1.grad shape: {q1.grad.shape if q1.grad is not None else 'None'}")
        print(f"  q2.grad shape: {q2.grad.shape if q2.grad is not None else 'None'}")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        return False
    
    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    test_rope_attention()