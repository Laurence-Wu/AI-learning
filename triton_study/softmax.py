import torch
import triton
import triton.language as tl
from triton.runtime import driver
DEVICE = triton.runtime.driver.active.get_active_torch_device()

#check if the gpu using is the HIP (heterogeneous-compute interface for Portability) and return true if runing on AMD GPU and false otherwise
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

#check if the AMD GPU has the CDNA architecture
def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942')
  
def native_softmax(x):
	x_max = x.max(dim = 1)[0]
	#here the x.max return a tuple
	z = x - x_max[:,None] #this operation is the same as unsqeeze(-1)
	#x_max was with shape (2,1) and now its (2,x.column)
	numerator = torch.exp(z)
	denominator = numerator.sum(dim = 1)
	#base on the second dimension,that's the elemental summation
	ret = numerator / denominator[:,None]
	return ret



@triton.jit
def softmax_kernel(output_ptr,input_ptr,input_row_stride,output_row_stride,n_rows,n_cols,num_stages:tl.constexpr,BLOCK_SIZE:tl.constexpr):
    row_start = tl.program_id(axis = 0)
    row_step = tl.num_programs(axis = 0)
    for row_idx in tl.range(row_start,n_rows,row_step,num_stages = num_stages):
		# The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
		
		col_offsets = tl.arange(0,BLOCK_SIZE)
		input_ptrs = row_start_ptr + col_offsets
		#make sure the offsets is in a safe place
		mask = col_offsets < n_cols
		row = tl.load(input_ptrs,mask = mask,other=-float('inf'))
		# subtract the row with the max
		row_minus_max = row - tl.max(row,axis = 0)
		# the exponentiation in triton is fast but approximate		
		numerator = tl.exp(row_minus_max)
		denominator = tl.sum(numerator,axis = 0)
		softmax_output = numerator / denominator
		
		output_row_start_ptr = output_ptr + row_idx * output_row_stride
		output_ptrs = output_row_start_ptr + col_offsets
		tl.store(output_ptrs, softmax_output,mask = mask)

properties = driver.active.utils.get_device_properties(DEVICE.index) 
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()

def softmax(x):
	n_rows,n_cols = x.shape
	
	BLOCK_SIZE = triton.next_power_of_2(n_cols)
    #here we use is to ask the compiler to use more threads per row to increase the processing speed
	num_warps = 8

	# the num_stages is the number of buffers to be staged before the GPU can handle it
	# if there's only one buffer then it just no piplining in the shared memeory
	# if there's two then its double buffer that is while the gpu is computing, buffer B can simultaneously load information. This will cause stall in the pipeline
	# if there's two or more then the the stalling issue will be allieviated
	num_stages = 4 if SIZE_SMEM > 200000 else 2
    
    #create a space for the final output	
	y = torch.empty_like(x)
    #pre-compilation for inspecting the resources
	kernel = softmax_kernel.warmup(y,x,x.stride(0),y.stride(0),n_rows,n_cols,BLOCK_SIZE = BLOCK_SIZE,num_stages,num_warps,grid=(1,))

	kernel._init_handles()
	n_regs = kernel.n_regs
	size_smem = kernel.metadata.shared
    
    #calculate the max number of programs that can run on a SM
	if is_hip():
		NUM_GPRS = NUM_REGS
		if is_cdna():
			NUM_GPRS = NUM_REGS * 2
		MAX_NUM_THREADS = properties["max_thread_per_sm"]
		max_num_waves = MAX_NUM_THREADS // WARP_SIZE
		occupancy = min((NUM_GPRS //WARP_SIZE) //n_regs, max_num_waves) // num_warps
	else:
		occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
	occupancy = min(occupancy , SIZE_SMEM // size_smem)
	num_programs = NUM_SM * occupancy
	
	num_programs = min(num_programs,n_rows)
	
	#create the programs
	kernel[(num_programs,1,1)](y,x,x.stride(x),y.stride(0),n_rows,n_cols,BLOCK_SIZE,num_stages,num_warps = num_warps)
	return y

@triton.testing.perf_report(
	triton.testing.Benchmark(
		x_names = ['N'], # arguments names to use as an x-axis for the plot
		x_vals = [128 * i for i in range(2,100)],
		line_arg = 'provider',
		line_vals = ['triton','torch'],
		line_names = ["triton","torch"],
		styles = [('blue','-'),('green','-')],
		ylabel = "GB/s",
		plot_name = "softmax-proformance",
		args={'M':4096},
	)
)
def benchmark(M,N,provider):
	x = torch.rand(M,N,device = DEVICE,dtype = torch.float32)
	stream = getattr(torch,DEVICE.type).Stream()
	getattr(torch,DEVICE.type).set_stream(stream)
	if provider == 'torch':
		ms = triton.testing.do_bench(lambda: torch.softmax(x,axis = -1))
	if provider == 'triton':
		ms = triton.testing.do_bench(lambda: softmax(x))
	gbps = lambda ms : 2*x.numel() * x.element_size * 1e-9 / (ms * 1e-3)
	return gbps(ms)

if __name__ == "__main__":
	benchmark.run(show_plots= True,print_data = True)

import torch
import triton
import triton.language as tl
from triton.runtime import driver
# Corrected line to get the active device
DEVICE = triton.runtime.driver.active.get_current_device()

#check if the gpu using is the HIP (heterogeneous-compute interface for Portability) and return true if runing on AMD GPU and false otherwise
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

#check if the AMD GPU has the CDNA architecture
def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942')

def native_softmax(x):
        x_max = x.max(dim = 1)[0]
        #here the x.max return a tuple
        z = x - x_max[:,None] #this operation is the same as unsqeeze(-1)
        #x_max was with shape (2,1) and now its (2,x.column)
        numerator = torch.exp(z)
        denominator = numerator.sum(dim = 1)
        #base on the second dimension,that's the elemental summation
        ret = numerator / denominator[:,None]
        return ret



@triton.jit
def softmax_kernel(output_ptr,input_ptr,input_row_stride,output_row_stride,n_rows,n_cols,num_stages:tl.constexpr,BLOCK_SIZE:tl.constexpr):
    row_start = tl.program_id(axis = 0)
    row_step = tl.num_programs(axis = 0)
    for row_idx in tl.range(row_start,n_rows,row_step,num_stages = num_stages):
      # The stride represents how much we need to increase the pointer to advance 1 row
      row_start_ptr = input_ptr + row_idx * input_row_stride

      col_offsets = tl.arange(0,BLOCK_SIZE)
      input_ptrs = row_start_ptr + col_offsets
      #make sure the offsets is in a safe place
      mask = col_offsets < n_cols
      row = tl.load(input_ptrs,mask = mask,other=-float('inf'))
      # subtract the row with the max
      row_minus_max = row - tl.max(row,axis = 0)
      # the exponentiation in triton is fast but approximate
      numerator = tl.exp(row_minus_max)
      denominator = tl.sum(numerator,axis = 0)
      softmax_output = numerator / denominator

      output_row_start_ptr = output_ptr + row_idx * output_row_stride
      output_ptrs = output_row_start_ptr + col_offsets
      tl.store(output_ptrs, softmax_output,mask = mask)

properties = driver.active.utils.get_device_properties(DEVICE)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()

def softmax(x):
        n_rows,n_cols = x.shape

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
    #here we use is to ask the compiler to use more threads per row to increase the processing speed
        num_warps = 8

        # the num_stages is the number of buffers to be staged before the GPU can handle it
        # if there's only one buffer then it just no piplining in the shared memeory
        # if there's two then its double buffer that is while the gpu is computing, buffer B can simultaneously load information. This will cause stall in the pipeline
        # if there's two or more then the the stalling issue will be allieviated
        num_stages = 4 if SIZE_SMEM > 200000 else 2

    #create a space for the final output
        y = torch.empty_like(x)
    #pre-compilation for inspecting the resources
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))

        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared

    #calculate the max number of programs that can run on a SM
        if is_hip():
                NUM_GPRS = NUM_REGS
                if is_cdna():
                        NUM_GPRS = NUM_REGS * 2
                MAX_NUM_THREADS = properties["max_thread_per_sm"]
                max_num_waves = MAX_NUM_THREADS // WARP_SIZE
                occupancy = min((NUM_GPRS //WARP_SIZE) //n_regs, max_num_waves) // num_warps
        else:
                occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy , SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy

        num_programs = min(num_programs,n_rows)

        #create the programs
        kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols)
        return y

@triton.testing.perf_report(
        triton.testing.Benchmark(
                x_names = ['N'], # arguments names to use as an x-axis for the plot
                x_vals = [128 * i for i in range(2,100)],
                line_arg = 'provider',
                line_vals = ['triton','torch'],
                line_names = ["triton","torch"],
                styles = [('blue','-'),('green','-')],
                ylabel = "GB/s",
                plot_name = "softmax-proformance",
                args={'M':4096},
        )
)
def benchmark(M,N,provider):
        x = torch.rand(M,N,device = DEVICE,dtype = torch.float32)
        stream = getattr(torch, x.device.type).Stream()
        getattr(torch, x.device.type).set_stream(stream)
        if provider == 'torch':
                ms = triton.testing.do_bench(lambda: torch.softmax(x,axis = -1))
        if provider == 'triton':
                ms = triton.testing.do_bench(lambda: softmax(x))
        gbps = lambda ms : 2*x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms)

if __name__ == "__main__":
        benchmark.run(show_plots= True,print_data = True)	
	

	
