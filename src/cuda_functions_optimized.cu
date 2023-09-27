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

__global__ void _max_pool2d(int batch_size, float* input, int input_channel, int input_height, int input_width, int kernel_height, int kernel_width, int stride, float* output){
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    
    
    if (col < input_width && row < input_height) {
        for (int b = 0; b < batch_size; b++) {
            for (int c=0; c < input_channel; c++) {                
                int start_row = row * stride;
                int start_col = col * stride;

                float max_value = input[(b * input_channel * input_height * input_width) + (c * input_height * input_width) + (start_row * input_width) + start_col];
                __shared__ float shared_input[kernel_height * kernel_width]; //shared memory for input
                shared_input[threadIdx.y * kernel_width + threadIdx.x] = input[(b * input_channel * input_height * input_width) + (c * input_height * input_width) + ((start_row + threadIdx.y) * input_width) + (start_col + threadIdx.x)];
                __syncthreads();

                for (int i = 0; i < kernel_height; i++) {
                    for (int j = 0; j < kernel_width; j++) {
                        float curr_value = input[(b * input_channel * input_height * input_width) + (c * input_height * input_width) + ((start_row + i) * input_width) + (start_col + j)];
                        if (curr_value > max_value) {
                            max_value = curr_value;
                        }
                    }
                }
                output[(b * input_channel * input_height * input_width) + (c * input_height * input_width) + (row * input_width) + col] = max_value;

            }
        }
    }
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

__global__ void _pad(){}
float* pad(){}

}