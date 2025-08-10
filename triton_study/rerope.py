# Adapted from the triton implementation of flash-attention v2
# https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py
import time
import torch
import torch.utils.benchmark as benchmark
import triton
import triton.language as tl


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
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh
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
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q1 = tl.load(Q1_block_ptr)
    q1 = (q1 * qk_scale).to(tl.float16)
    q2 = tl.load(Q2_block_ptr)
    q2 = (q2 * qk_scale).to(tl.float16)
    # loop over k, v and update accumulator
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        if start_n <= start_m * BLOCK_M - WINDOW - BLOCK_N or start_n >= (start_m + 1) * BLOCK_M + WINDOW:
            k2 = tl.load(K2_block_ptr)
            v = tl.load(V_block_ptr)
            qk += tl.dot(q2, k2, out_dtype=tl.float16)
        elif start_n > (start_m + 1) * BLOCK_M - WINDOW and start_n < start_m * BLOCK_M + WINDOW - BLOCK_N:
            k1 = tl.load(K1_block_ptr)
            v = tl.load(V_block_ptr)
            qk += tl.dot(q1, k1, out_dtype=tl.float16)
        else:
            k1 = tl.load(K1_block_ptr)
            k2 = tl.load(K2_block_ptr)
            v = tl.load(V_block_ptr)
            qk1 = tl.dot(q1, k1, out_dtype=tl.float16)
            qk2 = tl.dot(q2, k2, out_dtype=tl.float16)
            qk += tl.where(tl.abs(offs_m[:, None] - (start_n + offs_n[None, :])) < WINDOW, qk1, qk2)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(tl.float16), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K1_block_ptr = tl.advance(K1_block_ptr, (0, BLOCK_N))
        K2_block_ptr = tl.advance(K2_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.log2(l_i))
    # write back O
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
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel(
    Q1, Q2, K1, K2, V, sm_scale, Out, DO,
    DQ1, DQ2, DK1, DK2, DV,
    L,
    D,
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
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    qk_scale = sm_scale * 1.44269504
    # offset pointers for batch/head
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
    for start_n in range(0, num_block_kv):
        if CAUSAL:
            lo = tl.maximum(start_n * BLOCK_N, 0)
        else:
            lo = 0
        # initialize row/col offsets
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        q1_ptrs = Q1 + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        q2_ptrs = Q2 + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k1_ptrs = K1 + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        k2_ptrs = K2 + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq1_ptrs = DQ1 + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq2_ptrs = DQ2 + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        l_ptrs = L + off_hz * N_CTX
        # initialize dk amd dv
        dk1 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk2 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        k1 = tl.load(k1_ptrs)
        k2 = tl.load(k2_ptrs)
        v = tl.load(v_ptrs)
        # loop over rows
        for start_m in range(lo, num_block_q * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            q1 = tl.load(q1_ptrs)
            q2 = tl.load(q2_ptrs)
            # recompute p = softmax(qk, dim=-1).T
            if CAUSAL:
                qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), float(0.), float("-inf"))
            else:
                qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
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
            p = tl.exp2(qk - l_i[:, None])
            # compute dv
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(p.to(Q1.dtype.element_ty)), do)
            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, tl.trans(v))
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            # compute dk = dot(ds.T, q)
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
            # increment pointers
            dq1_ptrs += BLOCK_M * stride_qm
            dq2_ptrs += BLOCK_M * stride_qm
            q1_ptrs += BLOCK_M * stride_qm
            q2_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        # write-back
        dk1_ptrs = DK1 + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        dk2_ptrs = DK2 + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        tl.store(dk1_ptrs, dk1)
        tl.store(dk2_ptrs, dk2)
        tl.store(dv_ptrs, dv)


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q1, q2, k1, k2, v, causal, sm_scale, window):
        # shape constraints
        Lq, Lk, Lv = q1.shape[-1], k1.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q1)
        BLOCK_M = 64   # Reduced from 128 for shared memory limits
        BLOCK_N = 32 if Lk <= 64 else 16  # Reduced for RTX 4070 Laptop
        num_stages = 2 if Lk <= 64 else 1  # Reduced stages
        num_warps = 4
        grid = (triton.cdiv(q1.shape[2], BLOCK_M), q1.shape[0] * q1.shape[1], 1)
        L = torch.empty((q1.shape[0] * q1.shape[1], q1.shape[2]), device=q1.device, dtype=torch.float32)
        _fwd_kernel[grid](
            q1, q2, k1, k2, v, sm_scale,
            L,
            o,
            q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
            k1.stride(0), k1.stride(1), k1.stride(2), k1.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q1.shape[0], q1.shape[1], q1.shape[2],
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
            IS_CAUSAL=causal, WINDOW=window,
            num_warps=num_warps,
            num_stages=num_stages)

        ctx.save_for_backward(q1, q2, k1, k2, v, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.window = window
        return o

    @staticmethod
    def backward(ctx, do):
        BLOCK = 64  # Reduced for shared memory limits
        q1, q2, k1, k2, v, o, L = ctx.saved_tensors
        do = do.contiguous()
        dq1 = torch.zeros_like(q1, dtype=torch.float32)
        dq2 = torch.zeros_like(q2, dtype=torch.float32)
        dk1 = torch.empty_like(k1)
        dk2 = torch.empty_like(k2)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)
        _bwd_preprocess[(triton.cdiv(q1.shape[2], BLOCK) * ctx.grid[1], )](
            o, do,
            delta,
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )
        _bwd_kernel[(ctx.grid[1],)](
            q1, q2, k1, k2, v, ctx.sm_scale,
            o, do,
            dq1, dq2, dk1, dk2, dv,
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

triton_attention = _attention.apply


def torch_attention(q1, q2, k1, k2, v, causal, sm_scale, window):
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p1 = torch.matmul(q1, k1.transpose(2, 3)) * sm_scale
    p2 = torch.matmul(q2, k2.transpose(2, 3)) * sm_scale
    if causal:
        p1[:, :, M == 0] = float("-inf")
        p2[:, :, M == 0] = float("-inf")
    x = torch.arange(N_CTX, dtype=torch.int, device="cuda")
    M2 = ((x[:, None] - x[None, :]).abs() < window)[None, None, :]
    p = torch.where(M2, p1, p2)
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    return ref_out


Z = 1
H = 8 #32
N_CTX = 2048 #8192
# currently backward is VERY slow for d_head = 128
# https://github.com/openai/triton/issues/1975
D_HEAD = 64
WINDOW = 512 #2048
sm_scale = 0.5

q1 = torch.empty((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
q2 = torch.empty((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
k1 = torch.empty((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
k2 = torch.empty((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=torch.float16, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
grad = torch.randn_like(q1)

torch_output = torch_attention(q1, q2, k1, k2, v, False, sm_scale, WINDOW)
torch_output.backward(grad)
torch_dv, v.grad = v.grad.clone(), None
torch_dk1, k1.grad = k1.grad.clone(), None
torch_dk2, k2.grad = k2.grad.clone(), None
torch_dq1, q1.grad = q1.grad.clone(), None
torch_dq2, q2.grad = q2.grad.clone(), None
triton_output = triton_attention(q1, q2, k1, k2, v, False, sm_scale, WINDOW)
triton_output.backward(grad)
triton_dv, v.grad = v.grad.clone(), None
triton_dk1, k1.grad = k1.grad.clone(), None
triton_dk2, k2.grad = k2.grad.clone(), None
triton_dq1, q1.grad = q1.grad.clone(), None
triton_dq2, q2.grad = q2.grad.clone(), None
assert torch.allclose(torch_output, triton_output, atol=2e-2, rtol=0)
assert torch.allclose(torch_dv, triton_dv, atol=1e-2, rtol=0)
assert torch.allclose(torch_dk1, triton_dk1, atol=1e-2, rtol=0)
assert torch.allclose(torch_dk2, triton_dk2, atol=1e-2, rtol=0)
assert torch.allclose(torch_dq1, triton_dq1, atol=1e-2, rtol=0)
assert torch.allclose(torch_dq2, triton_dq2, atol=1e-2, rtol=0)


def f(fn, q1, q2, k1, k2, v, sm_scale, window, grad):
    q1.grad, q2.grad, k1.grad, k2.grad, v.grad = None, None, None, None, None
    out = fn(q1, q2, k1, k2, v, True, sm_scale, window)
    out.backward(grad, retain_graph=True)

t0 = benchmark.Timer(
    stmt='f(fn, q1, q2, k1, k2, v, sm_scale, window, grad)',
    globals={'f': f, 'fn': torch_attention, 'q1': q1, 'q2': q2, 'k1': k1, 'k2': k2, 'v': v, 'sm_scale': sm_scale, 'window': WINDOW, 'grad': grad},
    num_threads=torch.get_num_threads())

t1 = benchmark.Timer(
    stmt='f(fn, q1, q2, k1, k2, v, sm_scale, window, grad)',
    globals={'f': f, 'fn': triton_attention, 'q1': q1, 'q2': q2, 'k1': k1, 'k2': k2, 'v': v, 'sm_scale': sm_scale, 'window': WINDOW, 'grad': grad},
    num_threads=torch.get_num_threads())

print(t0.timeit(10))
print(t1.timeit(10))