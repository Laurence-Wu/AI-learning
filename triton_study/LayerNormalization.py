



import torch

import triton
import triton.language as tl

try:
    import apex
    HAP_APEX = True
except ModuleNotFoundError:
    HAS_APEX = False

DEVICE = triton.tuntime.driver.active.get_current_device()


@triton.jit
def _layer_norm_fwd_fused(
    X,  #pointer to the input
    Y,  #pointer to the output
    W,  #pointer to the weights
    B,  #pointer to the bases



    )





