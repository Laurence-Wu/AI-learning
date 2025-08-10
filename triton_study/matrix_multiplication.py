
import torch
import triton
import triton.language as tl
from typing import List


def get_autotune_configs():
    """
    Extensive autotune configuration based on official Triton tutorial.
    Tests different block sizes, stages, and warps for optimal performance.
    """
    configs = []
    for BLOCK_SIZE_M in [16, 32, 64, 128]:
        for BLOCK_SIZE_N in [32, 64, 128, 256]:
            for BLOCK_SIZE_K in [32, 64]:
                for GROUP_SIZE_M in [8]:
                    for num_stages in [2, 3, 4, 5]:
                        for num_warps in [2, 4, 8]:
                            configs.append(
                                triton.Config({
                                    'BLOCK_SIZE_M': BLOCK_SIZE_M,
                                    'BLOCK_SIZE_N': BLOCK_SIZE_N,
                                    'BLOCK_SIZE_K': BLOCK_SIZE_K,
                                    'GROUP_SIZE_M': GROUP_SIZE_M,
                                }, num_stages=num_stages, num_warps=num_warps)
                            )
    # Filter configurations to avoid excessive compilation time
    return configs[:60]  # Use first 60 configs for reasonable compile time


@triton.autotune(configs=get_autotune_configs(), key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Program ID and grid calculation (optimized block ordering)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Calculate offsets with modulo for boundary handling (official tutorial pattern)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointer arithmetic for A and B blocks
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator in FP32 for precision
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main compute loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A and B blocks with proper masking
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # Accumulate using tensor cores
        accumulator = tl.dot(a, b, accumulator)
        # Advance pointers for next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Optional activation function (fuseable in FP32)
    if ACTIVATION == "leaky_relu":
        accumulator = tl.where(accumulator >= 0, accumulator, 0.01 * accumulator)

    # Convert to output dtype
    c = accumulator.to(tl.float16)

    # Write output block with boundary checking
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor, activation: str = "") -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous() and b.is_contiguous(), "Inputs must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
    )
    return c


def _run_self_test():
    if not torch.cuda.is_available():
        print("CUDA device not available; Triton matmul test skipped.")
        return
    device = torch.device('cuda')
    torch.manual_seed(0)
    M, N, K = 1024, 1024, 1024
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    c_triton = matmul(a, b)
    c_torch = torch.matmul(a, b)
    max_abs_diff = (c_triton - c_torch).abs().max().item()
    print(f"Max abs diff vs torch: {max_abs_diff:.4e}")


if __name__ == "__main__":
    _run_self_test()

    # Optional: benchmarking with graph output
    try:
        import matplotlib.pyplot as plt
        import triton.testing as testing

        if not torch.cuda.is_available():
            print("CUDA not available; skipping benchmark plot.")
        else:
            device = torch.device('cuda')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

            sizes: List[int] = [256 * i for i in range(2, 33)]  # 512 .. 8192
            tflops_torch: List[float] = []
            tflops_triton: List[float] = []
            for n in sizes:
                M = N = K = n
                a = torch.randn((M, K), device=device, dtype=torch.float16)
                b = torch.randn((K, N), device=device, dtype=torch.float16)

                # Warmup
                for _ in range(3):
                    _ = torch.matmul(a, b)
                    _ = matmul(a, b)
                torch.cuda.synchronize()

                quantiles = [0.5, 0.2, 0.8]
                ms_torch, _, _ = testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
                ms_triton, _, _ = testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)

                to_tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
                tflops_torch.append(to_tflops(ms_torch))
                tflops_triton.append(to_tflops(ms_triton))

            plt.figure(figsize=(8, 5))
            plt.plot(sizes, tflops_torch, label='torch.matmul (cuBLAS)', color='green')
            plt.plot(sizes, tflops_triton, label='Triton matmul', color='blue')
            plt.xlabel('Matrix size N (square M=N=K)')
            plt.ylabel('TFLOPS (median)')
            plt.title('Matmul Performance Trend (FP16)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            out_path = '/home/yoyo/Coding/AI-learning/triton_study/matmul_bench_square.png'
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            print(f'Saved benchmark plot to {out_path}')
    except Exception as e:
        print(f"Benchmark plotting skipped: {e}")

    # Optional: Run official Triton benchmark pattern
    print("\n" + "="*50)
    print("OPTIONAL: Running official Triton benchmark pattern")
    print("="*50)
    try:
        import triton.testing as testing
        
        # Define the benchmark using Triton's framework
        configs = [
            testing.Benchmark(
                x_names=["M", "N", "K"],
                x_vals=[128 * i for i in range(2, 9)],  # Smaller range for faster execution
                line_arg="provider",
                line_vals=["torch", "triton"],
                line_names=["torch.matmul", "Triton"],
                styles=[("green", "-"), ("blue", "-")],
                ylabel="TFLOPS",
                plot_name="matmul-performance-fp16-optimized",
                args={},
            )
        ]

        @testing.perf_report(configs)
        def benchmark_official(M, N, K, provider):
            if not torch.cuda.is_available():
                return 0, 0, 0
            
            device = torch.device('cuda')
            a = torch.randn((M, K), device=device, dtype=torch.float16)
            b = torch.randn((K, N), device=device, dtype=torch.float16)
            
            quantiles = [0.5, 0.2, 0.8]
            if provider == "torch":
                ms, min_ms, max_ms = testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
            elif provider == "triton":
                ms, min_ms, max_ms = testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
            else:
                return 0, 0, 0
            
            perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
            return perf(ms), perf(max_ms), perf(min_ms)

        print("Running official benchmark (this may take a few minutes)...")
        benchmark_official.run(show_plots=False, print_data=True)
        
    except Exception as e:
        print(f"Official benchmark skipped: {e}")

