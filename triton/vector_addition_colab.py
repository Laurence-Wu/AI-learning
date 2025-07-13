import torch
import triton
import pdb
import triton.language as tl
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@triton.jit
def add_kernel(x_ptr,y_ptr,output_ptr,n_element,
                                BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0,BLOCK_SIZE)
        mask = offsets < n_element
        x = tl.load(x_ptr + offsets, mask = mask)
        y = tl.load(y_ptr + offsets, mask = mask)
        output = x + y
        tl.store(output_ptr + offsets,output,mask = mask)

def add(x:torch.Tensor,y:torch.Tensor):
        output = torch.empty_like(x)
        n_elements = output.numel()
        grid = lambda meta:(triton.cdiv(n_elements,meta['BLOCK_SIZE']),)
        add_kernel[grid](x,y,output,n_elements,BLOCK_SIZE = 1024)
        return output

#now test the bench mark
@triton.testing.perf_report(
        triton.testing.Benchmark(
                x_names = ['size'],
                x_vals = [2**i for i in range(12,28,1)],
                x_log = True,  #The axis is logarithmic
                line_arg = 'provider',
                line_vals=['triton','torch'],
                line_names=['Triton','Torch'],
                styles=[('blue','-'),('green','-')],
                ylabel = 'GB/s',
                plot_name = "add performance",
                args={}
        )
)

def benchmark(size,provider):
        x = torch.rand(size,dtype = torch.float32).to(DEVICE)
        y = torch.rand(size,dtype = torch.float32).to(DEVICE)
        quantiles = [0.5,0.2,0.8]
        if provider == 'torch':
                ms,min_ms,max_ms = triton.testing.do_bench(lambda: x+y,quantiles = quantiles)
        if provider == 'triton':
                ms,min_ms,max_ms = triton.testing.do_bench(lambda: add(x,y),quantiles = quantiles)
        gbps = lambda ms: 3*x.numel()*x.element_size()*1e-9 / (ms * 1e-3)
        return gbps(ms),gbps(max_ms),gbps(min_ms)

if __name__ == "__main__":
        benchmark.run(print_data=True,show_plots=True)
