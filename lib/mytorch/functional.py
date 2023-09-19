from ..util import get_run_mode, layer_preproc
from .conv import Conv2d
from .activation import LeakyReLU
from .batchnorm import BatchNorm2d
from .pooling import MaxPool2d
from .pad import Pad

@layer_preproc
def conv2d(activation, kernel, bias, stride, label=None):
    f = Conv2d(get_run_mode(), label)
    return f(activation, kernel, bias, stride)

@layer_preproc
def leaky_relu(activation, negative_slope, label=None):
    f = LeakyReLU(get_run_mode(), label)
    return f(activation, negative_slope)

@layer_preproc
def batch_norm(activation, mean, var, gamma, beta, label=None):
    f = BatchNorm2d(get_run_mode(), label)
    return f(activation, mean, var, gamma, beta)

@layer_preproc
def max_pool2d(activation, kernel, stride, label=None):
    f = MaxPool2d(get_run_mode(), label)
    return f(activation, kernel, stride)

@layer_preproc
def pad(input, pad, value, label=None):
    f = Pad(get_run_mode(), label)
    return f(input, pad, value)