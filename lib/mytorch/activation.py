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
        batch_size, channel, height, width = activation.shape[0], activation.shape[1], activation.shape[2], activation.shape[3]
        input_array = np.ascontiguousarray(activation)
        c_dll.leaky_relu.restype = POINTER(c_float)
        output = c_dll.leaky_relu(batch_size, input_array.ctypes.data_as(POINTER(c_float)), c_int32(channel), c_int32(height), c_int32(width), c_int32(negative_slope))
        arr_output = np.ctypeslib.as_array(output, (batch_size*channel*height*width, 1))
        arr_copied = np.copy(arr_output)
        c_dll.free(output)
        return torch.from_numpy(arr_copied.reshape((batch_size,channel, height, width)))
    
    def cuda(self, activation, negative_slope):
        # TODO
        pass

    def cuda_optimized(self, activation, negative_slope):
        # TODO
        pass
