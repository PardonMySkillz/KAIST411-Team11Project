from .base import Functional, c_dll, cu_dll, cuo_dll
import torch
import numpy as np
from ctypes import *

class LeakyReLU(Functional):
    """
        Implement the LeakyReLU with C and CUDA backend.
        Under C or CUDA RunMode conditions, the array has a format below.
        Array: {
            'pointer'   : C pointer that can directly passed to c functions,
            'shape'     : python tuple of array
        }
    
        Inputs:
        - activation     : An array containing input data, of shape (N, c, h, w)
        - negative_slope : A python integer of LeakyReLU parameter
        
        Returns an array output:
        - out: output, of shape (N, c, h, w)
    """
    def torch(self, activation, negative_slope):
        return torch.nn.functional.leaky_relu(activation, negative_slope=negative_slope)

    def c(self, activation, negative_slope):
        activation_ptr = activation['pointer']
        activation_shape = activation['shape']
        activation_c = cast(activation_ptr, POINTER(c_float))
        height, width = activation_shape

        c_dll.leaky_relu.restype = POINTER(c_float)

        output_ptr = c_dll.leaky_relu(activation_c, c_int32(height), c_int32(width), c_float(negative_slope))

        return output_ptr, (height, width)


    def cuda(self, activation, negative_slope):
        activation_ptr = activation['pointer']
        activation_shape = activation['shape']
        activation_c = cast(activation_ptr, POINTER(c_float))
        height, width = activation_shape

        print("negav slope", negative_slope)

        cu_dll.leaky_relu.restype = POINTER(c_float)

        output_ptr = cu_dll.leaky_relu(activation_c, c_int32(height), c_int32(width), c_float(negative_slope))

        return output_ptr, activation_shape


    def cuda_optimized(self, activation, negative_slope):
        # TODO
        pass
