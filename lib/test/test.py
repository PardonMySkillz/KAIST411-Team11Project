from .base import *
from ..util import RunMode

import sys

validations = [
    test_leaky_relu, 
    test_batch_norm, 
    test_conv2d, 
    test_conv2d_stride2, 
    test_conv2d_no_bias, 
    test_max_pool2d, 
    test_pad
]

performance = {
    "stress_leaky_relu": stress_leaky_relu, 
    "stress_batch_norm": stress_batch_norm, 
    "stress_conv2d": stress_conv2d, 
    "stress_maxpool_2d": stress_maxpool_2d, 
    "stress_pad": stress_pad,
    "test_leaky_relu": test_leaky_relu, 
    "test_batch_norm": test_batch_norm, 
    "test_conv2d": test_conv2d, 
    "test_conv2d_stride2": test_conv2d_stride2, 
    "test_conv2d_no_bias": test_conv2d_no_bias, 
    "test_max_pool2d": test_max_pool2d, 
    "test_pad": test_pad
}

if __name__ == "__main__":
    args = sys.argv[1:]
    
    if args[0] == 'C':
        rm = RunMode.C
    elif args[0] == 'CUDA':
        rm = RunMode.CUDA
    elif args[0] == 'CUDAOptimized':
        rm = RunMode.CUDAOptimized
    else:
        print("unsupported runmode")
        exit(1)
    
    if args[1] == 'validations':
        for test in validations:
            test(rm)
    elif args[1] in performance.keys():
        performance[args[1]](rm)
    else:
        print("unsupported test")
        exit(1)