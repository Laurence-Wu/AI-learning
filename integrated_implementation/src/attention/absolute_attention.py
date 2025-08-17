"""
Triton Implementation of Absolute Positional Encoding Attention for BERT
Traditional sinusoidal position embeddings added to input embeddings
"""

import torch
import triton
import triton.language as tl
import math
from typing import Optional


def calculate_optimal_block_sizes(head_dim, max_shared_memory=100000):
    """
    Calculate optimal block sizes based on shared memory constraints.
    
    Args:
        head_dim: Dimension of attention heads
        max_shared_memory: Maximum shared memory in bytes (default: ~100KB for safety)
    
    Returns:
        tuple: (BLOCK_M, BLOCK_N) optimal block sizes
    """
    # Account for float32 (4 bytes) and some overhead
    bytes_per_element = 4
    safety_factor = 0.8  # Use 80% of available memory for safety
    
    effective_memory = int(max_shared_memory * safety_factor)
    
    # Memory needed: BLOCK_M * head_dim + BLOCK_N * head_dim + BLOCK_M * BLOCK_N
    # Simplify: assume BLOCK_M ≈ BLOCK_N for balanced computation
    # Memory ≈ 2 * BLOCK * head_dim + BLOCK²
    
    # Start with reasonable values and adjust down if needed
    possible_blocks = [128, 64, 32, 16]
    
    for block_size in possible_blocks:
        memory_needed = (2 * block_size * head_dim + block_size * block_size) * bytes_per_element
        if memory_needed <= effective_memory:
            # For rectangular blocks, use slightly different M and N
            block_m = min(block_size, 64)  # Limit BLOCK_M for efficiency
            block_n = min(block_size, 64)  # Limit BLOCK_N for efficiency
            return block_m, block_n
    
    # Fallback to very small blocks
    return 16, 16


@triton.jit
def _absolute_attention_fwd_kernel(
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
    Absolute position encoding attention forward pass
    Standard attention with position embeddings already added to Q and K
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
    
    # Initialize accumulator and max tracker
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Load Q
    q = tl.load(Q_block_ptr)
    
    # Compute attention scores block by block
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load K and compute QK^T
        k = tl.load(K_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        
        # Scale attention scores
        qk *= sm_scale
        
        # Compute new max
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Compute exponentials
        p = tl.exp(qk - m_i_new[:, None])
        alpha = tl.exp(m_i - m_i_new)
        
        # Update accumulators
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        # Load V and update output accumulator
        v = tl.load(V_block_ptr)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        
        # Update trackers
        l_i = l_i_new
        m_i = m_i_new
        
        # Advance pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    
    # Normalize output
    acc = acc / l_i[:, None]
    
    # Store output
    out_offset = off_hz * stride_oh
    Out_block_ptr = tl.make_block_ptr(
        base=Out + out_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(Out_block_ptr, acc.to(Out.dtype.element_ty))
    
    # Store normalization factor for backward pass
    l_offset = off_hz * N_CTX + offs_m
    tl.store(L + l_offset, m_i + tl.log(l_i), mask=offs_m < N_CTX)


class AbsoluteBERTAttention(torch.nn.Module):
    """BERT Attention with Absolute Positional Encoding (traditional sinusoidal)"""
    
    def __init__(self, hidden_size, num_heads, max_position_embeddings, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = torch.nn.Linear(hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = torch.nn.Dropout(dropout)
        
        # Create sinusoidal position embeddings
        self.max_position_embeddings = max_position_embeddings
        self.register_buffer("position_embeddings", self._create_sinusoidal_embeddings(
            max_position_embeddings, hidden_size
        ))
    
    def _create_sinusoidal_embeddings(self, max_seq_length, hidden_size):
        """Create sinusoidal position embeddings"""
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * 
                           -(math.log(10000.0) / hidden_size))
        
        pe = torch.zeros(max_seq_length, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        cache_position=None,
    ):
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Add position embeddings to input
        positions = self.position_embeddings[:seq_length].unsqueeze(0).expand(batch_size, -1, -1)
        hidden_states_with_pos = hidden_states + positions.to(hidden_states.device, hidden_states.dtype)
        
        # Compute Q, K, V with proper dtype handling
        query_layer = self.transpose_for_scores(self.query(hidden_states_with_pos))
        key_layer = self.transpose_for_scores(self.key(hidden_states_with_pos))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Prepare for Triton kernel
        batch_size, num_heads, seq_len, head_dim = query_layer.shape
        
        # Reshape for Triton kernel
        q = query_layer.reshape(batch_size * num_heads, seq_len, head_dim)
        k = key_layer.reshape(batch_size * num_heads, seq_len, head_dim)
        v = value_layer.reshape(batch_size * num_heads, seq_len, head_dim)
        
        # Prepare output tensor
        o = torch.empty_like(q)
        
        # Calculate optimal block sizes based on hardware constraints
        BLOCK_M, BLOCK_N = calculate_optimal_block_sizes(head_dim)
        BLOCK_DMODEL = head_dim
        
        # Debug info (can be removed in production)
        if batch_size == 1 and num_heads == 1:  # Only print once to avoid spam
            print(f"Using block sizes: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, head_dim={head_dim}")
        
        # Prepare workspace for L
        L = torch.empty((batch_size * num_heads, seq_len), 
                       device=q.device, dtype=torch.float32)
        
        # Grid configuration
        grid = lambda META: (
            triton.cdiv(seq_len, META['BLOCK_M']),
            batch_size * num_heads,
        )
        
        # Scale factor
        sm_scale = 1.0 / math.sqrt(head_dim)
        
        # Launch kernel with error handling
        try:
            _absolute_attention_fwd_kernel[grid](
                q, k, v, sm_scale,
                L, o,
                q.stride(0), q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(0), v.stride(1), v.stride(2),
                o.stride(0), o.stride(0), o.stride(1), o.stride(2),
                batch_size * num_heads, num_heads, seq_len,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL=BLOCK_DMODEL,
                BLOCK_N=BLOCK_N,
                IS_CAUSAL=False,
            )
        except triton.runtime.errors.OutOfResources as e:
            print(f"Triton kernel failed with memory error: {e}")
            print(f"Attempted block sizes: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, head_dim={head_dim}")
            print("Falling back to PyTorch implementation...")
            
            # Fallback to PyTorch scaled dot-product attention
            o = torch.nn.functional.scaled_dot_product_attention(
                query_layer, key_layer, value_layer,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=sm_scale
            ).reshape(batch_size * num_heads, seq_len, head_dim)
        
        # Reshape output back
        context_layer = o.reshape(batch_size, num_heads, seq_len, head_dim)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, None) if output_attentions else (context_layer,)
        
        if past_key_value is not None:
            outputs = outputs + (past_key_value,)
        
        return outputs


def absolute_attention(q, k, v, causal=False, sm_scale=None):
    """
    Absolute position attention function for use in CLM models
    
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
    
    # Use PyTorch's efficient implementation with absolute position handling
    # For absolute attention, we rely on the position embeddings added at the model level
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, 
        attn_mask=None, 
        dropout_p=0.0, 
        is_causal=causal,
        scale=sm_scale
    )