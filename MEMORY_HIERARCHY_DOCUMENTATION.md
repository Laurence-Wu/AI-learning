# Memory Hierarchy Documentation for Fused Attention

## Overview

The `fused_attention.py` implementation now includes comprehensive comments explaining **where code and data reside** throughout the computation, with detailed memory hierarchy analysis for different platforms.

## Memory Hierarchy Levels Explained

### 1. **GPU Memory Hierarchy (CUDA with Triton)**

#### **HBM (High Bandwidth Memory)**
- **Location**: Main GPU memory 
- **Bandwidth**: ~500 GB/s (slow)
- **Capacity**: 8-80GB depending on GPU
- **Usage**: 
  - Storage of input tensors Q, K, V
  - Final output tensor storage
  - Long-term data residence

#### **SRAM (Shared Memory/On-chip)**  
- **Location**: On-chip shared memory per streaming multiprocessor
- **Bandwidth**: ~19 TB/s on A100 (very fast)
- **Capacity**: ~20MB per SM (small but fast)
- **Usage**:
  - Q block loaded once, stays resident
  - K, V blocks streamed temporarily
  - Attention computation workspace
  - Running statistics (max, sum)

#### **L1/L2 Cache (Hardware Managed)**
- **L1 Cache**: ~128KB per SM, automatic hardware management
- **L2 Cache**: ~6MB shared across SMs
- **Usage**: Automatic caching of frequently accessed data

### 2. **Apple Silicon (MPS) Memory Hierarchy**

#### **Unified Memory Architecture**
- **Location**: Shared between CPU and GPU
- **Bandwidth**: High bandwidth, low latency
- **Usage**: PyTorch operations with automatic memory management

### 3. **CPU Memory Hierarchy**

#### **Cache Levels**
- **L1 Cache**: 32-64KB per core (hot data during matrix ops)
- **L2 Cache**: 256KB-1MB per core (recent tensor blocks)
- **L3 Cache**: 8-32MB shared (large tensor portions)
- **System RAM**: Full tensor storage

## Code Location Analysis

### Flash Attention Algorithm (Triton)

```python
# MEMORY FLOW DOCUMENTATION:

# 1. SETUP PHASE (Address calculation only)
Q += off_z * stride_qz + off_h * stride_qh      # HBM base address
K += off_z * stride_kz + off_h * stride_kh      # HBM base address  
V += off_z * stride_vz + off_h * stride_vh      # HBM base address

# 2. INITIALIZATION (Fast on-chip memory)
m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # REGISTERS/SRAM
l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0          # REGISTERS/SRAM
acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)      # SRAM

# 3. CRITICAL MEMORY OPERATION (Expensive but done once)
q = tl.load(q_ptrs)  # HBM → SRAM (stays resident)

# 4. MAIN LOOP (Per K,V block)
for start_n in range(0, N_CTX, BLOCK_N):
    # MEMORY: Load K,V blocks (temporary residence)
    k = tl.load(k_block_ptrs)  # HBM → SRAM (temporary)
    v = tl.load(v_block_ptrs)  # HBM → SRAM (temporary)
    
    # COMPUTE: All operations in SRAM
    qk = tl.dot(q, k.T)        # SRAM computation
    p = tl.math.exp2(qk)       # SRAM computation
    acc = tl.dot(p, v, acc)    # SRAM computation
    
    # K,V blocks discarded, only running stats kept

# 5. FINAL OUTPUT (Write back to main memory)
tl.store(out_ptrs, acc)  # SRAM → HBM
```

### PyTorch Fallback Algorithm

```python
# MEMORY FLOW FOR PYTORCH FALLBACK:

# 1. INPUT PROCESSING (System RAM/Unified Memory)
q_work = q.to(ref_dtype)  # Copy to RAM with higher precision
k_work = k.to(ref_dtype)  # Copy to RAM with higher precision
v_work = v.to(ref_dtype)  # Copy to RAM with higher precision

# 2. ATTENTION MATRIX COMPUTATION (Temporary O(N²) memory usage)
scores = torch.matmul(q_work, k_work.transpose(-2, -1))  # RAM computation
# ⚠️  This materializes full N×N attention matrix in memory

# 3. SOFTMAX (Operates on full matrix in cache hierarchy)
attn_weights = torch.softmax(scores.float(), dim=-1)  # L1/L2/L3 cache usage

# 4. FINAL COMPUTATION (Optimized BLAS)
output = torch.matmul(attn_weights, v_work)  # Cache-optimized matrix mult
```

## Memory Efficiency Comparison

| Aspect | Traditional Attention | Flash Attention (Triton) | PyTorch Fallback |
|--------|---------------------|-------------------------|------------------|
| **Memory Usage** | O(N²) | O(N × BLOCK_SIZE) | O(N²) |
| **Peak Memory** | Full attention matrix | Small blocks only | Full attention matrix |
| **Example (N=4096)** | 16M elements | 64K elements | 16M elements |
| **Memory Reduction** | Baseline | 99.6% reduction | Same as baseline |
| **Compute Location** | HBM | SRAM | RAM + Caches |

## Platform-Specific Optimizations

### **CUDA GPUs** (Triton Available)
- **Strategy**: Block-wise computation in SRAM
- **Memory Pattern**: HBM → SRAM → Compute → HBM
- **Optimization**: Minimize HBM access, maximize SRAM reuse

### **Apple Silicon** (MPS)
- **Strategy**: Unified memory architecture leverage
- **Memory Pattern**: Unified memory with optimized access patterns
- **Optimization**: Leverage high-bandwidth unified memory

### **CPU Systems**
- **Strategy**: Cache hierarchy optimization
- **Memory Pattern**: RAM → L3 → L2 → L1 → Compute
- **Optimization**: BLAS-optimized matrix operations

## Key Memory Insights

### **Why Flash Attention is Memory Efficient**

1. **Block Processing**: Never materializes full O(N²) attention matrix
2. **Online Algorithm**: Maintains running statistics instead of storing intermediate results  
3. **SRAM Reuse**: Keeps frequently accessed data in fast memory
4. **Minimal HBM Access**: Reduces expensive memory transfers

### **Memory Access Patterns**

```
Traditional Attention:
Q[N×D] × K[N×D]ᵀ = Attention[N×N]  ← Stored in HBM (expensive!)
Attention[N×N] × V[N×D] = Output[N×D]

Flash Attention:
For each block:
  Q_block[M×D] × K_block[N×D]ᵀ = Scores[M×N]  ← Only in SRAM
  Softmax(Scores) × V_block[N×D] → Running_Sum  ← Accumulate in SRAM
Never store full attention matrix!
```

## Performance Implications

### **Memory Bandwidth Utilization**
- **HBM**: 500 GB/s (minimize usage)
- **SRAM**: 19,000 GB/s (maximize usage)  
- **Efficiency Gain**: 38x faster memory access for frequently used data

### **Cache Hierarchy Benefits** (CPU/MPS)
- **L1 Hit**: ~1-2 cycles
- **L2 Hit**: ~10-20 cycles
- **L3 Hit**: ~40-75 cycles
- **RAM Access**: ~200-300 cycles

## Testing Memory Hierarchy

The implementation includes tests that validate memory hierarchy usage:

1. **Consistency Tests**: Ensure identical outputs across memory hierarchies
2. **Size Scaling**: Test progressively larger tensors to stress memory systems
3. **Platform Adaptation**: Automatic selection of optimal implementation
4. **Gradient Flow**: Validate memory efficiency during backpropagation

## Practical Benefits

### **Training Large Models**
- **Memory Savings**: Can train larger models on same hardware
- **Speed Improvement**: Faster training due to reduced memory transfers
- **Stability**: Better numerical stability across platforms

### **Inference Optimization**
- **Reduced Latency**: Fewer memory operations mean faster inference
- **Batch Processing**: Can handle larger batch sizes
- **Edge Deployment**: More efficient for resource-constrained devices

This comprehensive documentation ensures users understand exactly where their computations happen and why the implementation is efficient across different hardware platforms. 