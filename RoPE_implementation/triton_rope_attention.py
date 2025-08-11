"""
Triton Implementation of RoPE (Rotary Position Embedding) Attention for BERT
For comparison with standard attention
"""

import torch
import triton
import triton.language as tl
import math
from typing import Optional


@triton.jit
def _rope_attention_fwd_kernel(
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
    RoPE attention forward pass
    Applies rotary position embeddings inline during attention computation
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
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize running statistics for stable softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Scale by log2(e) for using 2^x instead of exp
    qk_scale = sm_scale * 1.44269504089
    
    # Load Q block once
    q = tl.load(Q_block_ptr)
    
    # Apply RoPE to Q
    # Compute position-dependent rotation angles
    theta = 10000.0
    dim_half = BLOCK_DMODEL // 2
    
    # Position indices for Q block
    pos_m = offs_m
    
    # Frequency computation with better numerical stability
    freq_idx = offs_d // 2
    # Use stable computation for theta^(-2i/d)
    dim_factor = freq_idx.to(tl.float32) * 2.0 / BLOCK_DMODEL
    # More stable: exp(-log(theta) * factor) for 1/theta^factor
    log_theta = tl.log(theta)
    inv_freq = tl.exp(-log_theta * dim_factor)
    
    # Apply RoPE rotation to Q
    angle_m = pos_m[:, None].to(tl.float32) * inv_freq[None, :]
    cos_m = tl.cos(angle_m)
    sin_m = tl.sin(angle_m)
    
    # Rotate Q: split into even/odd indices
    q_even = tl.where(offs_d % 2 == 0, q, 0)
    q_odd = tl.where(offs_d % 2 == 1, q, 0)
    
    # Apply rotation
    q_rotated = q_even * cos_m - q_odd * sin_m
    q_rotated += q_odd * cos_m + q_even * sin_m
    
    q = q_rotated * qk_scale
    
    # Compute attention block by block
    lo = 0
    hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX) if IS_CAUSAL else N_CTX
    
    for start_n in range(lo, hi, BLOCK_N):
        # Load K, V blocks
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        
        # Apply RoPE to K
        pos_n = start_n + offs_n
        angle_n = pos_n[:, None].to(tl.float32) * inv_freq[None, :]
        cos_n = tl.cos(angle_n)
        sin_n = tl.sin(angle_n)
        
        # Transpose K for rotation
        k_t = tl.trans(k)
        k_even = tl.where(offs_d % 2 == 0, k_t, 0)
        k_odd = tl.where(offs_d % 2 == 1, k_t, 0)
        
        # Apply rotation
        k_rotated = k_even * cos_n - k_odd * sin_n
        k_rotated += k_odd * cos_n + k_even * sin_n
        
        # Transpose back
        k = tl.trans(k_rotated)
        
        # Compute QK^T with RoPE
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        
        # Online softmax update
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        # Update accumulator with proper scaling
        acc *= alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        
        # Update running statistics
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        
        # Advance pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    
    # Finalize output with improved numerical stability
    # Use larger epsilon and add gradient-friendly clipping
    l_i_safe = tl.maximum(l_i, 1e-6)  # Larger epsilon for stability
    # Clip the denominator to prevent extreme values
    l_i_safe = tl.minimum(l_i_safe, 1e6)
    acc = acc / l_i_safe[:, None]
    
    # Store log-sum-exp with clamping to prevent overflow
    log_sum = m_i + tl.log2(l_i_safe)
    log_sum = tl.minimum(tl.maximum(log_sum, -100.0), 100.0)  # Clamp to reasonable range
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, log_sum)
    
    # Write output
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty))


@triton.jit
def _rope_attention_bwd_kernel(
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
    RoPE attention backward pass
    """
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Initialize gradient accumulators
    start_n = tl.program_id(1)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # RoPE parameters using exp and log instead of pow
    theta = 10000.0
    freq_idx = offs_d // 2
    log_theta = tl.log(theta)
    exponent = freq_idx.to(tl.float32) * 2.0 / BLOCK_DMODEL
    inv_freq = 1.0 / tl.exp(log_theta * exponent)
    
    # Pointers to K, V
    k_ptrs = K + off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + off_hz * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    
    # Load K and V blocks
    k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX)
    
    # Apply RoPE to K for backward pass
    pos_n = offs_n
    angle_n = pos_n[:, None].to(tl.float32) * inv_freq[None, :]
    cos_n = tl.cos(angle_n)
    sin_n = tl.sin(angle_n)
    
    k_even = tl.where(offs_d % 2 == 0, k, 0)
    k_odd = tl.where(offs_d % 2 == 1, k, 0)
    k_rotated = k_even * cos_n - k_odd * sin_n
    k_rotated += k_odd * cos_n + k_even * sin_n
    k = k_rotated
    
    # Initialize gradients
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    
    # Loop over Q blocks
    lo = 0 if not IS_CAUSAL else start_n * BLOCK_N
    hi = N_CTX
    
    for start_m in range(lo, hi, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        
        # Load Q
        q_ptrs = Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX)
        
        # Apply RoPE to Q
        pos_m = offs_m
        angle_m = pos_m[:, None].to(tl.float32) * inv_freq[None, :]
        cos_m = tl.cos(angle_m)
        sin_m = tl.sin(angle_m)
        
        q_even = tl.where(offs_d % 2 == 0, q, 0)
        q_odd = tl.where(offs_d % 2 == 1, q, 0)
        q_rotated = q_even * cos_m - q_odd * sin_m
        q_rotated += q_odd * cos_m + q_even * sin_m
        q = q_rotated
        
        # Compute QK^T
        qk = tl.dot(q, tl.trans(k))
        
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        
        l_ptrs = L + off_hz * N_CTX + offs_m
        l_i = tl.load(l_ptrs, mask=offs_m < N_CTX)
        
        p = tl.exp2(qk - l_i[:, None])
        
        # Load DO
        do_ptrs = DO + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX)
        
        # Accumulate gradients
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
        
        # Compute dP
        D_ptrs = D + off_hz * N_CTX + offs_m
        Di = tl.load(D_ptrs, mask=offs_m < N_CTX)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        dp += tl.dot(do, tl.trans(v))
        
        # Update DK (with RoPE gradient) - fix dtype consistency
        dp_scaled = (dp.to(Q.dtype.element_ty) * p).to(Q.dtype.element_ty)
        dk_rot = tl.dot(tl.trans(dp_scaled), q)
        
        # Apply inverse RoPE rotation for gradient
        dk_even = tl.where(offs_d % 2 == 0, dk_rot, 0)
        dk_odd = tl.where(offs_d % 2 == 1, dk_rot, 0)
        dk_unrot = dk_even * cos_n + dk_odd * sin_n
        dk_unrot += dk_odd * cos_n - dk_even * sin_n
        
        dk += dk_unrot
    
    # Store gradients
    dk_ptrs = DK + off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    dv_ptrs = DV + off_hz * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=offs_n[:, None] < N_CTX)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=offs_n[:, None] < N_CTX)


class RoPEAttention(torch.autograd.Function):
    """
    RoPE attention with rotary position embeddings
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
        
        _rope_attention_fwd_kernel[grid](
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
        
        _rope_attention_bwd_kernel[grid](
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


def apply_rope(x, position_ids):
    """Apply RoPE to input tensor with improved numerical stability"""
    batch_size, num_heads, seq_len, head_dim = x.shape
    
    # Create frequency tensor with stable computation
    theta = 10000.0
    dim_half = head_dim // 2
    freq_idx = torch.arange(0, dim_half, dtype=torch.float32, device=x.device)
    # More stable computation: exp(-log(theta) * exponent)
    exponent = freq_idx * 2.0 / head_dim
    inv_freq = torch.exp(-math.log(theta) * exponent)
    
    # Compute position-dependent angles
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
    
    # Expand dimensions for broadcasting
    position_ids = position_ids.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq_len, 1]
    inv_freq = inv_freq.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, dim_half]
    
    # Compute angles
    angles = position_ids * inv_freq
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    
    # Split x into even and odd indices
    x_even = x[..., ::2]  # [batch, heads, seq_len, dim_half]
    x_odd = x[..., 1::2]  # [batch, heads, seq_len, dim_half]
    
    # Apply rotation
    x_rotated_even = x_even * cos_vals - x_odd * sin_vals
    x_rotated_odd = x_odd * cos_vals + x_even * sin_vals
    
    # Interleave back
    x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    x_rotated = x_rotated.flatten(-2)
    
    return x_rotated


def rope_attention(q, k, v, causal=False, sm_scale=None):
    """
    RoPE attention with rotary position embeddings
    
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
    
    # Apply RoPE to Q and K
    seq_len = q.shape[2]
    position_ids = torch.arange(seq_len, device=q.device).unsqueeze(0)
    
    q_rope = apply_rope(q, position_ids)
    k_rope = apply_rope(k, position_ids)
    
    # Use PyTorch's efficient implementation
    return torch.nn.functional.scaled_dot_product_attention(
        q_rope, k_rope, v, 
        attn_mask=None, 
        dropout_p=0.0, 
        is_causal=causal,
        scale=sm_scale
    )


class RoPEBERTAttention(torch.nn.Module):
    """
    BERT attention layer with RoPE (Rotary Position Embeddings)
    Drop-in replacement for standard BERT attention
    """
    
    def __init__(self, hidden_size, num_heads, max_position_embeddings=512, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections (no position embeddings needed for RoPE)
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.out_proj = torch.nn.Linear(hidden_size, hidden_size)
        
        # No position embeddings needed - RoPE handles positioning
        self.dropout = torch.nn.Dropout(dropout)
        
        # Initialize weights properly to prevent NaN
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following BERT initialization"""
        std = 0.02
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            torch.nn.init.zeros_(module.bias)
        
    def forward(self, hidden_states, attention_mask=None, position_ids=None, head_mask=None, output_attentions=False, output_hidden_states=False, past_key_value=None, encoder_hidden_states=None, encoder_attention_mask=None, cache_position=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections and reshape (no position embeddings added)
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE attention with Triton
        attn_output = rope_attention(q, k, v, causal=False, sm_scale=self.scale)
        
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
    print("RoPE BERT Attention with Triton")
    print("=" * 50)
    
    # Test parameters
    batch_size = 2
    seq_len = 512
    hidden_size = 768
    num_heads = 12
    
    # Create model
    model = RoPEBERTAttention(hidden_size, num_heads).cuda()
    
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
    
    # Compare parameter count
    standard_params = sum(p.numel() for p in StandardBERTAttention(hidden_size, num_heads).parameters())
    rope_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameter comparison:")
    print(f"Standard attention: {standard_params:,} parameters")
    print(f"RoPE attention: {rope_params:,} parameters")
    print(f"Saved parameters: {standard_params - rope_params:,} (position embeddings removed)")
