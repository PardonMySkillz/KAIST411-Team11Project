#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CEIL_DIV(X, Y) (((X)+(Y)-1)/(Y))

extern "C"{

// TODOs
// implement CUDA optimized functions whose functionality complies with restricted PyTorch functions
__global__ void _leaky_relu(float* input, float* output, int height, int width, float negative_slope){


    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    
    output[index] = input[index] < 0 ?  negative_slope * input[index] : input[index];

}
float* leaky_relu(float* input, int height, int width, float negative_slope){

    float *device_input, *device_output;
    unsigned long size =  height * width;

    cudaMalloc((void **) &device_input, size * sizeof(float));
    cudaMalloc((void **) &device_output, size * sizeof(float));

    cudaMemcpy(device_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(device_output, 0, size * sizeof(float));

    // dim3 threadsPerBlock(16);
    // dim3 numBlock((height * width +threadsPerBlock.x - 1)/ threadsPerBlock.x);

    // _leaky_relu<<<numBlock, threadsPerBlock>>>(device_input, device_output, height, width, negative_slope);

    dim3 threadsPerBlock(height * width);

    _leaky_relu<<<1, threadsPerBlock>>>(device_input, device_output, height, width, negative_slope);

    cudaDeviceSynchronize();

    cudaFree(device_input);

    return device_output;

}

__global__ void _batch_norm(float* input, float* output, int batch_size, int channels, int height, int width, float* running_mean, float* running_var, float* weight, float* bias){

    uint batch = blockIdx.x;
    uint channel = blockIdx.y;

    uint row = threadIdx.x;
    uint col = threadIdx.y;

    uint io_index = batch * channels * height * width + channel * height * width + row * width + col;

    float e = 1e-5;
    output[io_index] = weight[channel] * ((input[io_index] - running_mean[channel]) / sqrt(running_var[channel] + e)) + bias[channel];

}
float* batch_norm(float* input, int batch_size, int channels, int height, int width, float* running_mean, float* running_var, float* weight, float* bias){
    float* output, *device_input, *device_output;
    float *d_running_mean, *d_running_var, *d_weight, *d_bias;
    unsigned long io_size = batch_size * channels * height * width * sizeof(float);
    unsigned long mv_size = channels*sizeof(float);


    output = (float*)malloc(io_size);

    cudaMalloc((void**) &device_input, io_size);
    cudaMalloc((void**) &device_output, io_size);
    cudaMemcpy(device_input, input, io_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_running_mean, mv_size);
    cudaMalloc((void**) &d_running_var, mv_size);
    cudaMalloc((void**) &d_weight, mv_size);
    cudaMalloc((void**) &d_bias, mv_size);
    cudaMemcpy(d_running_mean, running_mean, mv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_running_var, running_var, mv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, mv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, mv_size, cudaMemcpyHostToDevice);

    dim3 numBlocks(batch_size, channels);
    dim3 threadsPerBlock(height, width);

    _batch_norm<<<numBlocks, threadsPerBlock>>>(device_input, device_output, batch_size, channels, height, width, d_running_mean, d_running_var, d_weight, d_bias);

    cudaDeviceSynchronize();

    cudaFree(device_input);
    cudaFree(d_running_mean);
    cudaFree(d_running_var);
    cudaFree(d_bias);
    cudaFree(d_weight);

    return device_output;

}

__global__ void _conv2d(){}
float* conv2d(){}

__global__ void _max_pool2d(){}
float* max_pool2d(){}

__global__ void _pad(){}
float* pad(){}

}