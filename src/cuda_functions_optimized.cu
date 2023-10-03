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
__global__ void _leaky_relu(float* input, float* output, int size, float negative_slope){

    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < size)
        output[index] = input[index] * (input[index] < 0 ? negative_slope: 1);
}
float* leaky_relu(float* input, int height, int width, float negative_slope){

    float *device_output;
    unsigned long size = height * width;

    cudaMalloc((void **) &device_output, size * sizeof(float));

    int threadsPerBlock = 1024;

    _leaky_relu<<<CEIL_DIV(size, threadsPerBlock), threadsPerBlock>>>(input, device_output, size, negative_slope);

    return device_output;

}

__global__ void _batch_norm(float *input, float *output, int channels, int sz2d, float *running_mean, float *running_var, float *weight, float *bias)
{

    uint b = blockIdx.x;
    uint c = blockIdx.y;

    uint idx = blockIdx.z * blockDim.x + threadIdx.x;

    uint io_index = b * channels * sz2d + c * sz2d + idx;

    float e = 1e-5;
    if (idx < sz2d)
        output[io_index] = weight[c] * ((input[io_index] - running_mean[c]) / __fsqrt_rd(running_var[c] + e)) + bias[c];
}

float* batch_norm(float* input, int batch_size, int channels, int height, int width, float* running_mean, float* running_var, float* weight, float* bias){
    float *device_output;
    int sz2d = height * width;
    unsigned long io_size = batch_size * channels * sz2d * sizeof(float);

    cudaMalloc((void**) &device_output, io_size);

    int threadsPerBlock = 128;
    dim3 numBlocks(batch_size, channels, CEIL_DIV(sz2d, threadsPerBlock));

    _batch_norm<<<numBlocks, threadsPerBlock>>>(input, device_output, channels, sz2d, running_mean, running_var, weight, bias);

    cudaDeviceSynchronize();

    return device_output;

}

__global__ void _conv2d(){}
float* conv2d(){}

__global__ void _max_pool2d_naive(float* input, int input_height, int input_width, int kernel, int stride, float* output, int output_height, int output_width) {
    int batchannel = blockIdx.y;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(idx >= output_height * output_width) return;

    // division and modulo are one instruction (can be mentioned as optimization)
    int yo = idx / output_width;
    int xo = idx % output_width;
    
    int yi = yo * stride;
    int xi = xo * stride;

    float *imagi = input + batchannel * input_height * input_width;
    float* imago = output + batchannel * output_height * output_width;

    float max_val = imagi[input_width * yi + xi];
    for(int i=yi; i < kernel + yi; i++)
        for(int j=xi; j < kernel + xi; j++)
            max_val = fmaxf(max_val, imagi[input_width * i + j]);
    
    imago[idx] = max_val;
}

#define MAX_POOL_1D_BLOCK 256
#define MAX_POOL_1D_WIDTH 1024
__global__ void _max_pool2d_1d(float* input, int input_height, int input_width, int kernel, int stride, float* output, int output_height, int output_width) {
    __shared__ float smem[MAX_POOL_1D_WIDTH];
    int batchannel = blockIdx.y;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // int slice = kernel / stride;

    if(idx >= output_height * output_width) return;

    // division and modulo are one instruction (can be mentioned as optimization)
    int yo = idx / output_width;
    int xo = idx % output_width;
    
    int yi = yo * stride;
    int xi = xo * stride;

    float *imagi = input + batchannel * input_height * input_width;
    float* imago = output + batchannel * output_height * output_width;

    for(int xii = xi; xii < input_width; xii += blockDim.x * stride) {
        float max_val = __FLT_MIN__;
        int e = min(stride + xii, input_width);
        for (int i = yi; i < kernel + yi; i++)
            for (int j = xii; j < e; j++)
                max_val = fmaxf(max_val, imagi[input_width * i + j]);
        smem[xii / stride] = max_val;
    }

    __syncthreads();

    for(int xii = xi; xii < input_width; xii += blockDim.x * stride) {
        float max_val = __FLT_MIN__;
        int e = (kernel + xii) / stride;
        for(int j=xii / stride; j < e; j++)
            max_val = fmaxf(max_val, smem[j]);
        
        imago[output_width * xo + yo] = max_val;
    }
}

float* max_pool2d(int batch_size, float* d_input, int input_channel, int input_height, int input_width, int kernel, int _deprecated, int stride){
    float *d_output;

    int output_height = (input_height - kernel) / stride + 1;
    int output_width = (input_width - kernel) / stride + 1;
    int output_size = batch_size * input_channel * output_height * output_width;

    cudaMalloc((void**)&d_output, output_size * sizeof(float));

    // Smallest ow * oh = 32 * 32

    // Most of these conditions are for simplification, they can be relaxed if there are relevant checks in the kernel
    printf("%d %d %d\n", stride < kernel, kernel % stride == 0, output_width % stride == 0);
    if(stride < kernel && kernel % stride == 0) { // 1D SHARED MEMORY KERNEL
        printf("Using 1D max_pool...\n");
        int blockSize = min(MAX_POOL_1D_BLOCK, output_width);
        dim3 numBlocks(output_height, batch_size * input_channel, 1);
        _max_pool2d_1d<<<numBlocks, blockSize>>>(d_input, input_height, input_width, kernel, stride, d_output, output_height, output_width);
    } else { // No need for shared memory if it will not be used
        int blockSize = 256;
        dim3 numBlocks(CEIL_DIV(output_height * output_width, blockSize), batch_size * input_channel, 1);
        
        _max_pool2d_naive<<<numBlocks, blockSize>>>(d_input, input_height, input_width, kernel, stride, d_output, output_height, output_width);
    }
    cudaDeviceSynchronize();

    cudaFree(d_input);
    return d_output;
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