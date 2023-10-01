from ..util import RunMode, timer
from ctypes import *

c_dll = cdll.LoadLibrary('./build/lib_c_functions.so')
c_dll.leaky_relu.restype = POINTER(c_float)
c_dll.conv2d.restype = POINTER(c_float)
c_dll.batch_norm.restype = POINTER(c_float)
c_dll.max_pool2d.restype = POINTER(c_float)
c_dll.pad.restype = POINTER(c_float)

cu_dll = cdll.LoadLibrary('./build/lib_cuda_functions.so')
cu_dll.np2cuda.restype = POINTER(c_float)
cu_dll.cuda2np.restype = POINTER(c_float)
cu_dll.leaky_relu.restype = POINTER(c_float)
cu_dll.conv2d.restype = POINTER(c_float)
cu_dll.batch_norm.restype = POINTER(c_float)
cu_dll.max_pool2d.restype = POINTER(c_float)
cu_dll.pad.restype = POINTER(c_float)

cuo_dll = cdll.LoadLibrary('./build/lib_cuda_functions_optimized.so')
# I have no idea why these lines exist and commented, they were in Kim's branch as comments and nothing in mine, I'm just adding them, maybe they are usefull idk - Aziz (image of the situation https://i.imgur.com/piXPcuN.png)
# cuo_dll.np2cuda.restype = POINTER(c_float)
# cuo_dll.cuda2np.restype = POINTER(c_float)
cuo_dll.leaky_relu.restype = POINTER(c_float)
cuo_dll.conv2d.restype = POINTER(c_float)
cuo_dll.batch_norm.restype = POINTER(c_float)
cuo_dll.max_pool2d.restype = POINTER(c_float)
cuo_dll.pad.restype = POINTER(c_float)

class Functional:
    _next_id = 0
    def get_next_id(self):
        return type(self)._next_id
    def set_next_id(self, val):
        type(self)._next_id = val
    next_id = property(get_next_id, set_next_id)

    def __init__(self, mode, label=None):
        self.mode = mode
        self.label = label
        self.id = self.next_id
        self.next_id = self.next_id+1
    
    def __repr__(self):
        return f"{self.label if self.label else (type(self).__name__+str(self.id))} in {self.mode}"

    @timer
    def __call__(self, *args, **kwargs):
        if self.mode is RunMode.TORCH:
            return self.torch(*args, **kwargs)
        elif self.mode is RunMode.C:
            return self.c(*args, **kwargs)
        elif self.mode is RunMode.CUDA:
            return self.cuda(*args, **kwargs)
        elif self.mode is RunMode.CUDAOptimized:
            return self.cuda_optimized(*args, **kwargs)
        else:
            print(f"mode {self.mode} is not supported.")
            exit(1)
