from .base import Functional, c_dll, cu_dll, cuo_dll
import torch
import numpy as np
from ctypes import *

class MaxPool2d(Functional):
    """
        TODO
        Implement the MaxPool2d with C and CUDA backend.
        Note that this function runs in restricted conditions.
        Kernel is a 1d variable, unlike original MaxPool2d, which can take a tuple input.
        Stride is a 1d variable, unlike original MaxPool2d, which can take a tuple input.
        Padding options are restricted. You should instead use a proper pad operation before MaxPool2d.
        Also, under C or CUDA RunMode conditions, the array has a format below.
        Array: {
            'pointer'   : C pointer that can directly passed to c functions,
            'shape'     : python tuple of array
        }
    
        Inputs:
        - activation : An array containing input data, of shape (N, ic, h, w)
        - kernel     : A python integer of kernel size
        - stride     : A python integer of stride

        Returns an array output:
        - out: output, of shape (N, oc, oh, ow)
               Since the operation's ceil_mode is fixed to False, the expected height and width is as below
               oh = (h-k)//s + 1
               ow = (w-k)//s + 1
    """
    def torch(self, activation, kernel, stride):
        return torch.nn.functional.max_pool2d(activation, kernel, stride=stride)

    def c(self, activation, kernel, stride):
        batch_size, in_channels, input_height, input_width = activation['shape']
        output_height = (input_height - kernel) // stride + 1
        output_width = (input_width - kernel) // stride + 1

        input_ptr = cast(activation['pointer'], POINTER(c_float))
        output = c_dll.max_pool2d(batch_size, input_ptr, c_int32(in_channels), c_int32(input_height), c_int32(input_width),
                                  c_int32(kernel),c_int32(kernel), c_int32(stride))
        
        return output, (batch_size, in_channels, output_height, output_width)
        
    
    def cuda(self, activation, kernel, stride):
        # TODO
        pass
    
    def cuda_optimized(self, activation, kernel, stride):
        # TODO
        pass