from .base import Functional, c_dll, cu_dll, cuo_dll
import torch
import numpy as np
import ctypes

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
        activation_ptr = activation['pointer']
        activation_shape = activation['shape']
        activation_c = ctypes.cast(activation_ptr, ctypes.POINTER(ctypes.c_float))

        N, ic, h, w = activation_shape
        kernel_width = kernel
        kernel_height = kernel
        input_height = h
        input_width = w

        output_height = (input_height-kernel_height) // stride +1
        output_width = (input_width-kernel_width) // stride +1
        output_size = output_height*output_width

        
        c_dll.max_pool2d.restype = ctypes.POINTER(ctypes.c_float)
        c_dll.max_pool2d.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)

        output_ptr = c_dll.max_pool2d (
            activation_c,
            input_height,
            input_width,
            kernel_width,
            kernel_height,
            stride
        )
        output = []
        output_shape = (N, ic, output_height, output_width)
        output.append(output_ptr)
        output.append(output_shape)
        
        return output
        
    
    def cuda(self, activation, kernel, stride):
        # TODO
        pass
    
    def cuda_optimized(self, activation, kernel, stride):
        # TODO
        pass