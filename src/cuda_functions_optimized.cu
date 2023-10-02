#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_profiler_api.h>

#define CEIL_DIV(X, Y) (((X)+(Y)-1)/(Y))

#include <time.h>
unsigned long long get_time_ns()
{
    struct timespec ts;

    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) == -1)
    {
        printf("clock_gettime error\n");
        exit(-1);
    }

    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

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

__global__ void _max_pool2d(int batch_size, float* input, int input_channel, int input_height, int input_width, int kernel_height, int kernel_width, int stride, float* output){
    // Commented out for bugs - Aziz
    // const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    // const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    
    
    // if (col < input_width && row < input_height) {
    //     for (int b = 0; b < batch_size; b++) {
    //         for (int c=0; c < input_channel; c++) {                
    //             int start_row = row * stride;
    //             int start_col = col * stride;

    //             float max_value = input[(b * input_channel * input_height * input_width) + (c * input_height * input_width) + (start_row * input_width) + start_col];
    //             __shared__ float shared_input[kernel_height * kernel_width]; //shared memory for input
    //             shared_input[threadIdx.y * kernel_width + threadIdx.x] = input[(b * input_channel * input_height * input_width) + (c * input_height * input_width) + ((start_row + threadIdx.y) * input_width) + (start_col + threadIdx.x)];
    //             __syncthreads();

    //             for (int i = 0; i < kernel_height; i++) {
    //                 for (int j = 0; j < kernel_width; j++) {
    //                     float curr_value = input[(b * input_channel * input_height * input_width) + (c * input_height * input_width) + ((start_row + i) * input_width) + (start_col + j)];
    //                     if (curr_value > max_value) {
    //                         max_value = curr_value;
    //                     }
    //                 }
    //             }
    //             output[(b * input_channel * input_height * input_width) + (c * input_height * input_width) + (row * input_width) + col] = max_value;

    //         }
    //     }
    // }
}
float* max_pool2d(int batch_size, float* input, int input_channel, int input_height, int input_width, int kernel_height, int kernel_width, int stride){
    float* output, *device_output, *device_input;
    int size = batch_size * input_channel * input_height * input_width;
    cudaMalloc((void**)&device_input, size * sizeof(float));
    cudaMalloc((void**)&device_output, size * sizeof(float));

    cudaMemcpy(device_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(device_output, 0, size * sizeof(float));
    dim3 threadPerBlock(32, 32);
    dim3 numBlocks((input_width + threadPerBlock.x - 1) / threadPerBlock.x, (input_height + threadPerBlock.y - 1) / threadPerBlock.y);
    _max_pool2d<<<numBlocks, threadPerBlock>>>(batch_size, device_input, input_channel, input_height, input_width, kernel_height, kernel_width, stride, device_output);
    
    output = (float*)malloc(size * sizeof(float));
    cudaMemcpy(output, device_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_input);
    cudaFree(device_output);
    return output;
}

__global__ void _pad(float *input, float *output, int size, int height, int width, int left, int right, int top, int bottom, int sz2d, float padding)
{
    int new_height = height + top + bottom;
    int new_width = width + left + right;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) return;
    int bc = idx / sz2d;
    int off = idx % sz2d;
    int y = (off / new_width) - top;
    int x = (off % new_width) - left;

    float *bci = input + bc * height * width;
    float *bco = output + bc * new_height * new_width + new_width * top + left;

    if (x < 0 || width <= x || y < 0 || height <= y)
        output[idx] = padding;
    else
        output[idx] = bci[width * y + x];
}

float *pad(float *input_ptr, int batch_size, int channels, int height, int width, int left, int right, int top, int bottom, float padding)
{
    float *d_input = input_ptr, *d_output;
    int new_height = height + top + bottom;
    int new_width = width + left + right;

    // int input_size = batch_size * channels * height * width;
    int output_size = batch_size * channels * new_height * new_width;

    int batchannels = batch_size * channels;

#ifdef DEBUG
    printf("Allocating %f MB...\n", (float)(output_size) / 1e6 * sizeof(float));
#endif

    cudaMalloc((void **)&d_output, sizeof(float) * output_size);

    // Fill array with padding
    int block_size = 1024;
    int num_blocks = CEIL_DIV(output_size, block_size);
    
    _pad<<<num_blocks, block_size>>>(d_input, d_output, output_size, height, width, left, right, top, bottom, new_height * new_width, padding);

    cudaDeviceSynchronize();

    return d_output;
}
}