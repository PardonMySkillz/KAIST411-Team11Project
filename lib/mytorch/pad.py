from .base import Functional, c_dll, cu_dll, cuo_dll
import torch
from ctypes import *

class Pad(Functional):
    """
        Implement the Pad with C and CUDA backend.
        Note that this function runs in restricted conditions.
        For example, the tuple size of pad is fixed to 4, unlike original pad, which can take a variable sized tuple input.
        Also, under C or CUDA RunMode conditions, the array has a format below.
        Array: {
            'pointer'   : C pointer that can directly passed to c functions,
            'shape'     : python tuple of array
        }
    
        Inputs:
        - activation : An array containing input data, of shape (N, c, h, w)
        - pad        : A python tuple consisted of (left, right, top, bottom)
        - value      : A python float value to be filled
        
        Returns an array output:
        - out: output, of shape (N, oc, oh, ow)
               the expected height and width is at below
               oh = h+top+bottom
               ow = w+left+right
    """
    def torch(self, activation, pad, value):
        return torch.nn.functional.pad(activation, pad, value=value)

    def c(self, activation, pad, value):
        # TODO
        pass
    
    def cuda(self, activation, pad, value):
        # TODO
        pass
    
    def cuda_optimized(self, activation, pad, value):
        # TODO
        pass