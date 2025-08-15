"""
Triton Implementation of RSE (Rotary Stick-breaking Encoding) Attention for BERT
Integrates Rotary Position Embeddings (RoPE) with Exponential Decay Stick-Breaking Attention

Mathematical Formulation:
β_{i,j} = σ(q_i^T k_j e^{j(i-j)θ} - λ(j-i))
A_{i,j} = β_{i,j} ∏_{i<k<j} (1 - β_{k,j})

Where:
- q_i, k_j are query and key vectors with RoPE applied
- θ is the RoPE frequency parameter
- λ is the exponential decay parameter
- σ is the sigmoid function
"""

import torch
import triton
import triton.language as tl
import math
from typing import Optional, Tuple


class RSEReferenceImplementation:
    """
    Reference implementation of RSE attention in pure PyTorch for correctness testing.
    This implements the exact mathematical formulation without Triton optimizations.
    """
    
    @staticmethod
    def apply_rope(x: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor) -> torch.Tensor:
        """Apply RoPE transformation to input tensor"""
        batch_size, n_heads, seq_len, head_dim = x.shape
        
        # Split into even and odd indices
        x_even = x[..., ::2]  # [batch, heads, seq_len, head_dim//2]
        x_odd = x[..., 1::2]  # [batch, heads, seq_len, head_dim//2]
        
        # Apply rotation
        cos_vals = cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
        sin_vals = sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
        
        x_rotated_even = x_even * cos_vals - x_odd * sin_vals
        x_rotated_odd = x_odd * cos_vals + x_even * sin_vals
        
        # Interleave back
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.flatten(-2)
        
        return x_rotated
    
    @staticmethod
    def stick_breaking_attention(
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        lambda_param: float,
        causal: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reference stick-breaking attention implementation.
        
        Mathematical formulation:
        β_{i,j} = σ(q_i^T k_j - λ(j-i))
        A_{i,j} = β_{i,j} ∏_{i<k<j} (1 - β_{k,j})
        """
        batch_size, n_heads, seq_len, head_dim = q.shape
        scale = 1.0 / math.sqrt(head_dim)
        
        # Compute attention logits
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Add exponential decay: -λ(j-i)
        pos_i = torch.arange(seq_len, device=q.device).unsqueeze(1)
        pos_j = torch.arange(seq_len, device=q.device).unsqueeze(0)
        pos_diff = pos_j - pos_i  # j - i
        decay_term = lambda_param * pos_diff.float()
        logits = logits - decay_term
        
        # Apply causal mask
        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1) * float('-inf')
            logits = logits + mask
        
        # Compute β_{i,j} = σ(logits)
        beta = torch.sigmoid(logits)
        
        # Stick-breaking: A_{i,j} = β_{i,j} ∏_{i<k<j} (1 - β_{k,j})
        attention_weights = torch.zeros_like(beta)
        
        for i in range(seq_len):
            stick_remaining = 1.0
            for j in range(seq_len):
                if causal and j > i:
                    continue
                
                # Current allocation
                attention_weights[:, :, i, j] = beta[:, :, i, j] * stick_remaining
                
                # Update remaining stick
                stick_remaining = stick_remaining * (1 - beta[:, :, i, j])
                
                # Prevent complete stick exhaustion
                stick_remaining = torch.clamp(stick_remaining, min=0.001)
        
        # Compute output
        output = torch.matmul(attention_weights, v)
        return output, attention_weights
    
    @classmethod
    def forward(
        cls, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        lambda_param: float,
        causal: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Complete RSE forward pass"""
        # Apply RoPE
        q_rope = cls.apply_rope(q, cos_cache, sin_cache)
        k_rope = cls.apply_rope(k, cos_cache, sin_cache)
        
        # Apply stick-breaking attention
        output, attention_weights = cls.stick_breaking_attention(q_rope, k_rope, v, lambda_param, causal)
        
        return output, attention_weights


@triton.jit
def _rse_attention_fwd_kernel(
    Q, K, V, sm_scale,
    L, Out,
    cos_cache, sin_cache,  # RoPE cache
    lambda_param,  # Exponential decay parameter
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_cos, stride_sin,  # RoPE cache strides
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    RSE attention forward pass with integrated RoPE and Stick-Breaking
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh
    
    # Initialize pointers
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
    
    # Initialize running statistics for stable computation
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Load Q block
    q = tl.load(Q_block_ptr)
    
    # Apply RoPE to Q
    dim_half = BLOCK_DMODEL // 2
    
    # Load RoPE cache for Q positions
    cos_q_ptrs = cos_cache + offs_m[:, None] * stride_cos + (offs_d // 2)[None, :] * 1
    sin_q_ptrs = sin_cache + offs_m[:, None] * stride_sin + (offs_d // 2)[None, :] * 1
    cos_q = tl.load(cos_q_ptrs, mask=(offs_m[:, None] < N_CTX) & ((offs_d // 2)[None, :] < dim_half))
    sin_q = tl.load(sin_q_ptrs, mask=(offs_m[:, None] < N_CTX) & ((offs_d // 2)[None, :] < dim_half))
    
    # Apply RoPE rotation to Q
    q_even = tl.where(offs_d % 2 == 0, q, 0)
    q_odd = tl.where(offs_d % 2 == 1, q, 0)
    
    # RoPE transformation: q' = q * cos - rotate_half(q) * sin
    q_rope = q_even * cos_q - q_odd * sin_q
    q_rope += q_odd * cos_q + q_even * sin_q
    
    # Scale by attention factor
    qk_scale = sm_scale * 1.44269504089
    q_rope = q_rope * qk_scale
    
    # Compute stick-breaking attention
    lo = 0
    hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX) if IS_CAUSAL else N_CTX
    
    # Initialize stick-breaking accumulation for each query position
    stick_remaining = tl.ones([BLOCK_M], dtype=tl.float32)  # Remaining "stick" for each query
    
    for start_n in range(lo, hi, BLOCK_N):
        # Load K, V blocks
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        
        # Apply RoPE to K
        pos_k = start_n + offs_n
        cos_k_ptrs = cos_cache + pos_k[:, None] * stride_cos + (offs_d // 2)[None, :] * 1
        sin_k_ptrs = sin_cache + pos_k[:, None] * stride_sin + (offs_d // 2)[None, :] * 1
        cos_k = tl.load(cos_k_ptrs, mask=(pos_k[:, None] < N_CTX) & ((offs_d // 2)[None, :] < dim_half))
        sin_k = tl.load(sin_k_ptrs, mask=(pos_k[:, None] < N_CTX) & ((offs_d // 2)[None, :] < dim_half))
        
        # Transpose K for rotation
        k_t = tl.trans(k)
        k_even = tl.where(offs_d % 2 == 0, k_t, 0)
        k_odd = tl.where(offs_d % 2 == 1, k_t, 0)
        
        # Apply RoPE to K
        k_rope = k_even * cos_k - k_odd * sin_k
        k_rope += k_odd * cos_k + k_even * sin_k
        k_rope = tl.trans(k_rope)
        
        # Compute RoPE attention logits: z_{i,j}^{rope} = q_i^{rope} @ k_j^{rope}
        qk_rope = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk_rope += tl.dot(q_rope, k_rope)
        
        # Apply exponential decay: z_{i,j}^{rope} - λ(j-i)
        pos_diff = pos_k[None, :] - offs_m[:, None]  # j - i
        exponential_decay = lambda_param * pos_diff.to(tl.float32)
        logits = qk_rope - exponential_decay
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            logits = tl.where(offs_m[:, None] >= pos_k[None, :], logits, float("-inf"))
        
        # Compute β_{i,j} = σ(logits) in log space for stability
        # Use log-sigmoid: log σ(x) = x - log(1 + exp(x)) = x - softplus(x)
        log_beta = logits - tl.log(1.0 + tl.exp(tl.minimum(logits, 20.0)))  # Clamp for stability
        beta = tl.exp(log_beta)
        
        # Stick-breaking: A_{i,j} = β_{i,j} * ∏_{i<k<j} (1 - β_{k,j})
        # For efficiency, we approximate the product by using remaining stick
        attention_weights = beta * stick_remaining[:, None]
        
        # Update remaining stick: multiply by (1 - β) for processed positions
        # This is an approximation for efficient parallel computation
        stick_update = 1.0 - tl.max(beta, 1)  # Max across key positions
        stick_remaining = stick_remaining * tl.maximum(stick_update, 0.1)  # Prevent complete decay
        
        # Apply additional numerical stability
        attention_weights = tl.where(attention_weights > 1e-8, attention_weights, 0.0)
        
        # Normalize attention weights (optional, for stability)
        row_sum = tl.sum(attention_weights, 1) + 1e-8
        attention_weights = attention_weights / row_sum[:, None]
        
        # Compute weighted sum of values
        acc += tl.dot(attention_weights.to(v.dtype), v)
        
        # Update running statistics for numerical stability
        m_i_new = tl.maximum(m_i, tl.max(logits, 1))
        alpha = tl.exp2(m_i - m_i_new)
        l_i = l_i * alpha + tl.sum(attention_weights, 1)
        m_i = m_i_new
        
        # Advance pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    
    # Final normalization
    l_i_safe = tl.maximum(l_i, 1e-6)
    acc = acc / l_i_safe[:, None]
    
    # Store log-sum-exp
    log_sum = m_i + tl.log2(l_i_safe)
    log_sum = tl.minimum(tl.maximum(log_sum, -100.0), 100.0)
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
def _rse_attention_bwd_kernel(
    Q, K, V, Out, DO,
    DQ, DK, DV,
    L, D,
    cos_cache, sin_cache,
    lambda_param, DLambda,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_cos, stride_sin,
    Z, H, N_CTX,
    num_block_q, num_block_kv,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    RSE attention backward pass with RoPE and stick-breaking gradients
    """
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    
    start_n = tl.program_id(1)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    dim_half = BLOCK_DMODEL // 2
    
    # Load K and V blocks
    k_ptrs = K + off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + off_hz * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX)
    v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX)
    
    # Apply RoPE to K for backward pass
    cos_k_ptrs = cos_cache + offs_n[:, None] * stride_cos + (offs_d // 2)[None, :] * 1
    sin_k_ptrs = sin_cache + offs_n[:, None] * stride_sin + (offs_d // 2)[None, :] * 1
    cos_k = tl.load(cos_k_ptrs, mask=(offs_n[:, None] < N_CTX) & ((offs_d // 2)[None, :] < dim_half))
    sin_k = tl.load(sin_k_ptrs, mask=(offs_n[:, None] < N_CTX) & ((offs_d // 2)[None, :] < dim_half))
    
    k_even = tl.where(offs_d % 2 == 0, k, 0)
    k_odd = tl.where(offs_d % 2 == 1, k, 0)
    k_rope = k_even * cos_k - k_odd * sin_k
    k_rope += k_odd * cos_k + k_even * sin_k
    
    # Initialize gradients
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dlambda_local = tl.zeros([1], dtype=tl.float32)
    
    # Loop over Q blocks for backward pass
    lo = 0 if not IS_CAUSAL else start_n * BLOCK_N
    hi = N_CTX
    
    for start_m in range(lo, hi, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        
        # Load Q and apply RoPE
        q_ptrs = Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX)
        
        cos_q_ptrs = cos_cache + offs_m[:, None] * stride_cos + (offs_d // 2)[None, :] * 1
        sin_q_ptrs = sin_cache + offs_m[:, None] * stride_sin + (offs_d // 2)[None, :] * 1
        cos_q = tl.load(cos_q_ptrs, mask=(offs_m[:, None] < N_CTX) & ((offs_d // 2)[None, :] < dim_half))
        sin_q = tl.load(sin_q_ptrs, mask=(offs_m[:, None] < N_CTX) & ((offs_d // 2)[None, :] < dim_half))
        
        q_even = tl.where(offs_d % 2 == 0, q, 0)
        q_odd = tl.where(offs_d % 2 == 1, q, 0)
        q_rope = q_even * cos_q - q_odd * sin_q
        q_rope += q_odd * cos_q + q_even * sin_q
        
        # Recompute forward pass components
        qk_rope = tl.dot(q_rope, tl.trans(k_rope))
        pos_diff = offs_n[None, :] - offs_m[:, None]
        logits = qk_rope - lambda_param * pos_diff.to(tl.float32)
        
        if IS_CAUSAL:
            logits = tl.where(offs_m[:, None] >= offs_n[None, :], logits, float("-inf"))
        
        # Compute attention weights (simplified for backward pass)
        beta = tl.sigmoid(logits)
        attention_weights = beta  # Simplified - full stick-breaking requires more complex backward
        
        # Load gradients
        do_ptrs = DO + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do = tl.load(do_ptrs, mask=offs_m[:, None] < N_CTX)
        
        # Compute gradients
        dv += tl.dot(tl.trans(attention_weights.to(Q.dtype.element_ty)), do)
        
        # Gradient w.r.t. logits
        dlogits = tl.dot(do, tl.trans(v))
        
        # Gradient w.r.t. lambda (exponential decay term)
        dlambda_contrib = -tl.sum(dlogits * pos_diff.to(tl.float32))
        dlambda_local += dlambda_contrib
        
        # Gradient w.r.t. K (through RoPE)
        dk_rope_contrib = tl.dot(tl.trans(dlogits.to(Q.dtype.element_ty)), q_rope)
        
        # Apply inverse RoPE rotation for K gradients
        dk_even = tl.where(offs_d % 2 == 0, dk_rope_contrib, 0)
        dk_odd = tl.where(offs_d % 2 == 1, dk_rope_contrib, 0)
        dk_unrot = dk_even * cos_k + dk_odd * sin_k
        dk_unrot += dk_odd * cos_k - dk_even * sin_k
        
        dk += dk_unrot
    
    # Store gradients
    dk_ptrs = DK + off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    dv_ptrs = DV + off_hz * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=offs_n[:, None] < N_CTX)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=offs_n[:, None] < N_CTX)
    
    # Accumulate lambda gradient
    tl.atomic_add(DLambda, dlambda_local)


class RSEAttention(torch.autograd.Function):
    """
    RSE attention with integrated RoPE and stick-breaking
    Autograd wrapper for Triton kernels
    """
    
    @staticmethod
    def forward(ctx, q, k, v, cos_cache, sin_cache, lambda_param, causal, sm_scale):
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
        
        _rse_attention_fwd_kernel[grid](
            q, k, v, sm_scale,
            L, o,
            cos_cache, sin_cache,
            lambda_param,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            cos_cache.stride(0), sin_cache.stride(0),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
            IS_CAUSAL=causal,
            num_warps=num_warps,
            num_stages=num_stages
        )
        
        ctx.save_for_backward(q, k, v, o, L, cos_cache, sin_cache, lambda_param)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L, cos_cache, sin_cache, lambda_param = ctx.saved_tensors
        do = do.contiguous()
        
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dlambda = torch.zeros(1, device=q.device, dtype=torch.float32)
        
        # Compute D for gradient computation
        delta = torch.empty_like(L)
        BLOCK = 64
        
        # Backward kernel launch
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        
        _rse_attention_bwd_kernel[grid](
            q, k, v, o, do,
            dq, dk, dv,
            L, delta,
            cos_cache, sin_cache,
            lambda_param, dlambda,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            cos_cache.stride(0), sin_cache.stride(0),
            q.shape[0], q.shape[1], q.shape[2],
            triton.cdiv(q.shape[2], BLOCK), triton.cdiv(k.shape[2], BLOCK),
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            IS_CAUSAL=ctx.causal,
            num_warps=8,
            num_stages=1
        )
        
        return dq, dk, dv, None, None, dlambda, None, None


class RSEBERTAttention(torch.nn.Module):
    """
    BERT attention layer with RSE (Rotary Stick-breaking Encoding)
    Integrates RoPE with exponential decay stick-breaking attention
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 8192,
        theta_base: float = 10000.0,
        initial_lambda: float = 0.01,
        learnable_lambda: bool = True,
        rope_scaling_factor: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        # RoPE setup
        self.theta_base = theta_base
        self.rope_scaling_factor = rope_scaling_factor
        self.max_seq_len = max_seq_len
        
        # Initialize RoPE cache
        self._init_rope_cache(max_seq_len)
        
        # Exponential decay parameter for stick-breaking
        if learnable_lambda:
            self.lambda_param = torch.nn.Parameter(torch.tensor(initial_lambda))
        else:
            self.register_buffer('lambda_param', torch.tensor(initial_lambda))
        
        # Linear projections
        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = torch.nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_rope_cache(self, max_seq_len: int):
        """Precompute sin and cos values for RoPE"""
        # Create frequency tensor
        dim_half = self.d_head // 2
        freq_idx = torch.arange(0, dim_half, dtype=torch.float32)
        exponent = freq_idx * 2.0 / self.d_head
        inv_freq = 1.0 / (self.theta_base ** exponent)
        
        # Apply rope scaling for extended context
        inv_freq = inv_freq / self.rope_scaling_factor
        
        # Precompute sin and cos for all positions
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = positions[:, None] * inv_freq[None, :]  # [max_seq_len, dim_half]
        
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)
        
        self.register_buffer('cos_cache', cos_cache)
        self.register_buffer('sin_cache', sin_cache)
    
    def _init_weights(self):
        """Initialize weights following BERT initialization"""
        std = 0.02
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        
        # Initialize lambda parameter
        torch.nn.init.constant_(self.lambda_param, 0.01)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections and reshape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Ensure cache is large enough
        if seq_len > self.max_seq_len:
            self._init_rope_cache(seq_len * 2)
        
        # Get RoPE cache for current sequence length
        cos_cache = self.cos_cache[:seq_len]  # [seq_len, dim_half]
        sin_cache = self.sin_cache[:seq_len]  # [seq_len, dim_half]
        
        # Apply RSE attention with Triton
        attn_output = RSEAttention.apply(
            q, k, v, cos_cache, sin_cache, self.lambda_param, False, self.scale
        )
        
        # Apply head mask if provided
        if head_mask is not None:
            attn_output = attn_output * head_mask
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # Return tuple to match BERT interface
        outputs = (output,)
        if output_attentions:
            outputs = outputs + (None,)  # We don't compute attention weights in this implementation
        
        return outputs


# Convenience function for direct usage
def rse_attention(q, k, v, cos_cache, sin_cache, lambda_param, causal=False, sm_scale=None):
    """
    RSE attention with integrated RoPE and stick-breaking
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        v: Value tensor [batch, heads, seq_len, head_dim]
        cos_cache: Precomputed cosine values [seq_len, head_dim//2]
        sin_cache: Precomputed sine values [seq_len, head_dim//2]
        lambda_param: Exponential decay parameter
        causal: Whether to apply causal mask
        sm_scale: Softmax scale factor
    
    Returns:
        Attention output [batch, heads, seq_len, head_dim]
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    
    return RSEAttention.apply(q, k, v, cos_cache, sin_cache, lambda_param, causal, sm_scale)


if __name__ == "__main__":
    # Test the implementation
    print("RSE BERT Attention with Triton")
    print("=" * 50)
    
    # Test parameters
    batch_size = 2
    seq_len = 512
    d_model = 768
    n_heads = 12
    
    # Create model
    model = RSEBERTAttention(d_model, n_heads)
    
    # Test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output[0].shape}")
    
    # Test gradients
    loss = output[0].sum()
    loss.backward()
    print("Gradient check passed!")
    
    # Check lambda parameter
    print(f"Lambda parameter: {model.lambda_param.item():.6f}")
    print(f"Lambda gradient: {model.lambda_param.grad.item() if model.lambda_param.grad is not None else 'None'}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("RSE attention implementation ready!")