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

    
        batch_size, channel, height, width = activation_shape
        weight_array = np.ascontiguousarray(weight)
        bias_array = np.ascontiguousarray(bias)
        running_mean_array = np.ascontiguousarray(running_mean)
        running_var_array = np.ascontiguousarray(running_var)

        size = batch_size * channel * height * width

        output_c = (c_float * size)()

        c_dll.batch_norm.restype = POINTER(c_float)
        output_ptr = c_dll.batch_norm(
            activation_c, c_int32(batch_size), c_int32(channel), c_int32(height), c_int32(width),
            running_mean_array.ctypes.data_as(POINTER(c_float)), running_var_array.ctypes.data_as(POINTER(c_float)),
            weight_array.ctypes.data_as(POINTER(c_float), bias_array.ctypes.data(POINTER(c_float)))
        )
        
        c_dll.free(output)
        memmove(output_c, output_ptr, size * sizeof(c_float))
        
        output = {
            'pointer': cast(output_c, c_void_p).value,
            'shape': (batch_size, channel, height, width)
        }
        return output
    
    def cuda(self, activation, running_mean, running_var, weight, bias):
        batch_size, channel, height, width = activation.shape[0], activation.shape[1], activation.shape[2], activation.shape[3]
        input_array = np.ascontiguousarray(activation)
        weight_array = np.ascontiguousarray(weight)
        bias_array = np.ascontiguousarray(bias)
        running_mean_array = np.ascontiguousarray(running_mean)
        running_var_array = np.ascontiguousarray(running_var)
        cu_dll.batch_norm.restype = POINTER(c_float)
        output = cu_dll.batch_norm(
            input_array.ctypes.data_as(POINTER(c_float)), c_int32(batch_size), c_int32(channel), c_int32(height), c_int32(width),
            running_mean_array.ctypes.data_as(POINTER(c_float)), running_var_array.ctypes.data_as(POINTER(c_float)),
            weight_array.ctypes.data_as(POINTER(c_float), bias_array.ctypes.data(POINTER(c_float)))
        )
        arr_output = np.ctypeslib.as_array(output, (batch_size*channel*height*width, 1))
        arr_copied = np.copy(arr_output)
        cu_dll.free(output)
        return torch.from_numpy(arr_copied.reshape((batch_size,channel, height, width)))
    
    def cuda_optimized(self, activation, running_mean, running_var, weight, bias):
        # TODO
        pass