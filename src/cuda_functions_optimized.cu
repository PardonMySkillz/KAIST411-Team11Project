#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CEIL_DIV(X, Y) (((X)+(Y)-1)/(Y))

extern "C"{

// TODOs
// implement CUDA optimized functions whose functionality complies with restricted PyTorch functions
__global__ void _leaky_relu(){}
float* leaky_relu(){}

__global__ void _batch_norm(){}
float* batch_norm(){}

__global__ void _conv2d(){}
float* conv2d(){}

__global__ void _max_pool2d(){}
float* max_pool2d(){}

__global__ void _pad(){}
float* pad(){}

}