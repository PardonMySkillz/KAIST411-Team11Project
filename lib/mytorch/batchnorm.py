from .base import Functional, c_dll, cu_dll, cuo_dll
import torch
import numpy as np
from ctypes import *

class BatchNorm2d(Functional):
    """
        Implement the BatchNorm2d with C and CUDA backend.
        Note that this function runs in restricted conditions.
        The eps value of this function is fixed to 1e-5.
        Also, under C or CUDA RunMode conditions, the array has a format below.
        Array: {
            'pointer'   : C pointer that can directly passed to c functions,
            'shape'     : python tuple of array
        }
    
        Inputs:
        - activation   : An array containing input data, of shape (N, c, h, w)
        - running_mean : An array of running mean, of shape (c,)
        - running_var  : An array of running var, of shape (c,)
        - weight       : An array of gamma, of shape (c,)
        - bias         : An array of beta, of shape (c,)

        Returns an array output:
        - out: output, of shape (N, c, h, w)
    """
    def torch(self, activation, running_mean, running_var, weight, bias):
        return torch.nn.functional.batch_norm(activation, running_mean, running_var, weight, bias)

    def c(self, activation, running_mean, running_var, weight, bias):
        activation_ptr = activation['pointer']
        activation_shape = activation['shape']
        activation_c = cast(activation_ptr, POINTER(c_float))
        print(bias["shape"])  
        batch_size, channel, height, width = activation_shape
        weight_array = cast(weight['pointer'], POINTER(c_float))
        bias_array = cast(bias['pointer'], POINTER(c_float))
        running_mean_array = cast(running_mean['pointer'], POINTER(c_float))
        running_var_array = cast(running_var['pointer'], POINTER(c_float))

        c_dll.batch_norm.restype = POINTER(c_float)
        output_ptr = c_dll.batch_norm(
            activation_c, c_int32(batch_size), c_int32(channel), c_int32(height), c_int32(width),
            running_mean_array, running_var_array,
            weight_array, bias_array
        )
    
        return output_ptr, activation_shape
    
    def cuda(self, activation, running_mean, running_var, weight, bias):
        activation_ptr = activation['pointer']
        activation_shape = activation['shape']
        activation_c = cast(activation_ptr, POINTER(c_float))
        print(bias["shape"])  
        batch_size, channel, height, width = activation_shape
        weight_array = cast(weight['pointer'], POINTER(c_float))
        bias_array = cast(bias['pointer'], POINTER(c_float))
        running_mean_array = cast(running_mean['pointer'], POINTER(c_float))
        running_var_array = cast(running_var['pointer'], POINTER(c_float))
        cu_dll.batch_norm.restype = POINTER(c_float)
        output_ptr = cu_dll.batch_norm(
            activation_c, c_int32(batch_size), c_int32(channel), c_int32(height), c_int32(width),
            running_mean_array, running_var_array,
            weight_array, bias_array
        )
    
        return output_ptr, activation_shape
    
    def cuda_optimized(self, activation, running_mean, running_var, weight, bias):
        # TODO
        pass