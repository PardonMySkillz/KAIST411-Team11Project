from .base import Functional, c_dll, cu_dll, cuo_dll
import torch
from ctypes import *
import numpy as np
class Conv2d(Functional):
    """
        Implement the Conv2d with C and CUDA backend.
        Note that this function runs in restricted conditions.
        Stride is a 1d variable, unlike original Conv2d, which can take a tuple input.
        Padding options are restricted. You should instead use a proper pad operation before conv2d.
        Also, under C or CUDA RunMode conditions, the array has a format below.
        Array: {
            'pointer'   : C pointer that can directly passed to c functions,
            'shape'     : python tuple of array
        }
    
        Inputs:
        - activation : An array containing input data, of shape (N, ic, h, w)
        - weight     : An array of weights, of shape (oc, ic, kh, kw)
        - bias       : None value, or an array of biases, of shape (oc)
        - stride     : A python integer of convolution stride

        Returns an array output:
        - out: output, of shape (N, oc, oh, ow)
               the expected height and width is at below
               oh = (h-kh)//s + 1
               ow = (w-kw)//s + 1
    """
    def torch(self, activation, weight, bias, stride):
        return torch.nn.functional.conv2d(activation, weight, bias=bias, stride=stride, padding='valid')

    def c(self, activation, weight, bias, stride):
        
        # TODO
        activation_pointer = activation["pointer"]
        weight_pointer = weight["pointer"]
        bias_pointer = bias["pointer"] if bias is not None else None
        batch_size, input_channel, input_height, input_width = activation["shape"][0], activation["shape"][1], activation["shape"][2], activation["shape"][3]
        output_channel, kernel_height, kernel_width = weight["shape"][0], weight["shape"][2], weight["shape"][3]
        output_height, output_width = (input_height - kernel_height) // stride + 1, (input_width - kernel_width) // stride + 1
        output = c_dll.conv2d(c_int32(batch_size), activation_pointer, c_int32(input_channel), c_int32(input_height), c_int32(input_width),
                 weight_pointer, bias_pointer, c_int32(kernel_height), c_int32(kernel_width), 
                 c_int32(output_channel), c_int32(output_height), c_int32(output_width), c_int32(stride))
        return (output, (batch_size,output_channel,output_height,output_width))

    def cuda(self, activation, weight, bias, stride):
        # TODO
        activation_pointer = activation["pointer"]
        weight_pointer = weight["pointer"]
        bias_pointer = bias["pointer"] if bias is not None else None
        batch_size, input_channel, input_height, input_width = activation["shape"][0], activation["shape"][1], activation["shape"][2], activation["shape"][3]
        output_channel, kernel_height, kernel_width = weight["shape"][0], weight["shape"][2], weight["shape"][3]
        output_height, output_width = (input_height - kernel_height) // stride + 1, (input_width - kernel_width) // stride + 1
        output = cu_dll.conv2d(c_int32(batch_size), activation_pointer, c_int32(input_channel), c_int32(input_height), c_int32(input_width),
                 weight_pointer, bias_pointer, c_int32(kernel_height), c_int32(kernel_width), 
                 c_int32(output_channel), c_int32(output_height), c_int32(output_width), c_int32(stride))
        return (output, (batch_size,output_channel,output_height,output_width))
    
    def cuda_optimized(self, activation, weight, bias, stride):
        # TODO
        activation_pointer = activation["pointer"]
        weight_pointer = weight["pointer"]
        bias_pointer = bias["pointer"] if bias is not None else None
        batch_size, input_channel, input_height, input_width = activation["shape"][0], activation["shape"][1], activation["shape"][2], activation["shape"][3]
        output_channel, kernel_height, kernel_width = weight["shape"][0], weight["shape"][2], weight["shape"][3]
        output_height, output_width = (input_height - kernel_height) // stride + 1, (input_width - kernel_width) // stride + 1
        output = cuo_dll.conv2d(c_int32(batch_size), activation_pointer, c_int32(input_channel), c_int32(input_height), c_int32(input_width),
                 weight_pointer, bias_pointer, c_int32(kernel_height), c_int32(kernel_width), 
                 c_int32(output_channel), c_int32(output_height), c_int32(output_width), c_int32(stride))
        
        return (output, (batch_size,output_channel,output_height,output_width))
