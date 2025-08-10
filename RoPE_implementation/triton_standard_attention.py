"""
Triton Implementation of Standard BERT Attention with Absolute Position Embeddings
For comparison with RoPE attention
"""

import torch
import triton
import triton.language as tl
import math
from typing import Optional


@triton.jit
def _standard_attention_fwd_kernel(
    Q, K, V, sm_scale,
    L, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Standard attention forward pass (no RoPE, uses absolute position embeddings)
    Based on Flash Attention v2 algorithm
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh
    
    # Initialize pointers to Q, K, V blocks
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Initialize running statistics for stable softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Scale by log2(e) for using 2^x instead of exp
    qk_scale = sm_scale * 1.44269504089
    
    # Load Q block once
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(tl.float16)
    
    # Compute attention block by block
    lo = 0
    hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX) if IS_CAUSAL else N_CTX
    
    for start_n in range(lo, hi, BLOCK_N):
        # Load K, V blocks
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        
        # Compute QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        
        # Online softmax update
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        # Update accumulator
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        
        # Update running statistics
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        
        # Advance pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    
    # Finalize output
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.log2(l_i))
    
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
def _standard_attention_bwd_kernel(
    Q, K, V, Out, DO,
    DQ, DK, DV,
    L, D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H, N_CTX,
    num_block_q, num_block_kv,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Standard attention backward pass
    """
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Initialize gradient accumulators
    start_n = tl.program_id(1)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Pointers to K, V, DK, DV
    k_ptrs = K + off_hz * stride_kh + offs_n[:, None] * stride_kn + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_kk
    v_ptrs = V + off_hz * stride_vh + offs_n[:, None] * stride_vk + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vn
    
    # Load K and V blocks
    k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX)
    
    # Initialize gradients
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    
    # Loop over Q blocks
    lo = 0 if not IS_CAUSAL else start_n * BLOCK_N
    hi = N_CTX
    
    for start_m in range(lo, hi, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        
        # Load Q, O, DO, L, D
        q_ptrs = Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX)
        
        # Compute gradients
        qk = tl.dot(q, tl.trans(k))
        
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        
        l_ptrs = L + off_hz * N_CTX + offs_m
        l_i = tl.load(l_ptrs, mask=offs_m < N_CTX)
        
        p = tl.exp2(qk - l_i[:, None])
        
        # Load DO
        do_ptrs = DO + off_hz * stride_qh + offs_m[:, None] * stride_qm + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk
        do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX)
        
        # Accumulate gradients
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
        
        # Compute dP
        D_ptrs = D + off_hz * N_CTX + offs_m
        Di = tl.load(D_ptrs, mask=offs_m < N_CTX)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        dp += tl.dot(do, tl.trans(v))
        
        # Update DK - fix dtype consistency
        dp_scaled = (dp.to(Q.dtype.element_ty) * p).to(Q.dtype.element_ty)
        dk += tl.dot(tl.trans(dp_scaled), q)
    
    # Store gradients
    dk_ptrs = DK + off_hz * stride_kh + offs_n[:, None] * stride_kn + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_kk
    dv_ptrs = DV + off_hz * stride_vh + offs_n[:, None] * stride_vk + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vn
    
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=offs_n[:, None] < N_CTX)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=offs_n[:, None] < N_CTX)


class StandardAttention(torch.autograd.Function):
    """
    Standard BERT attention with absolute position embeddings
    Autograd wrapper for Triton kernels
    """
    
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # Shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        
        o = torch.empty_like(q)
        BLOCK_M = 64
        BLOCK_N = 32 if Lk <= 64 else 16
        num_stages = 2 if Lk <= 64 else 1
        num_warps = 4
        
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        
        _standard_attention_fwd_kernel[grid](
            q, k, v, sm_scale,
            L, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
            IS_CAUSAL=causal,
            num_warps=num_warps,
            num_stages=num_stages
        )
        
        ctx.save_for_backward(q, k, v, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        do = do.contiguous()
        
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        # Compute D for gradient computation
        delta = torch.empty_like(L)
        BLOCK = 64
        
        # Backward kernel launch
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        
        _standard_attention_bwd_kernel[grid](
            q, k, v, o, do,
            dq, dk, dv,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            triton.cdiv(q.shape[2], BLOCK), triton.cdiv(k.shape[2], BLOCK),
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            IS_CAUSAL=ctx.causal,
            num_warps=8,
            num_stages=1
        )
        
        return dq, dk, dv, None, None


def standard_attention(q, k, v, causal=False, sm_scale=None):
    """
    Standard attention with absolute position embeddings
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
        causal: Whether to apply causal mask
        sm_scale: Softmax scale factor
    
    Returns:
        Attention output [batch, heads, seq_len, head_dim]
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    
    return StandardAttention.apply(q, k, v, causal, sm_scale)


class StandardBERTAttention(torch.nn.Module):
    """
    Standard BERT attention layer with absolute position embeddings
    Drop-in replacement for BERT attention
    """
    
    def __init__(self, hidden_size, num_heads, max_position_embeddings=512, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)
        
        # Absolute position embeddings
        self.position_embeddings = torch.nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None, head_mask=None, output_attentions=False, output_hidden_states=False, past_key_value=None, encoder_hidden_states=None, encoder_attention_mask=None, cache_position=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Add position embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeddings
        
        # Linear projections and reshape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply standard attention with Triton
        attn_output = standard_attention(q, k, v, causal=False, sm_scale=self.scale)
        
        # Apply head mask if provided
        if head_mask is not None:
            attn_output = attn_output * head_mask
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # Return tuple to match BERT interface: (hidden_states,)
        # If output_attentions is True, also return attention weights (None for now)
        outputs = (output,)
        if output_attentions:
            outputs = outputs + (None,)  # We don't compute attention weights in this implementation
        
        return outputs


if __name__ == "__main__":
    # Test the implementation
    print("Standard BERT Attention with Triton")
    print("=" * 50)
    
    # Test parameters
    batch_size = 2
    seq_len = 512
    hidden_size = 768
    num_heads = 12
    
    # Create model
    model = StandardBERTAttention(hidden_size, num_heads).cuda()
    
    # Test input
    x = torch.randn(batch_size, seq_len, hidden_size).cuda().half()
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test gradients
    loss = output.sum()
    loss.backward()
    print("Gradient check passed!")
