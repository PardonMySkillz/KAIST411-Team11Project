#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CEIL_DIV(X, Y) (((X)+(Y)-1)/(Y))

extern "C"{

float* np2cuda(float* input, int size){
    float* output;

    cudaMalloc((void**)&output, sizeof(float) * size);
	cudaMemcpy(output, input, sizeof(float) * size, cudaMemcpyHostToDevice);

    return output;
}

float* cuda2np(float* input, int size){
    float* output = (float*) malloc(sizeof(float) * size);
    cudaMemcpy(output, input, sizeof(float) * size, cudaMemcpyDeviceToHost);
    
    return output;
}

void cuda_free(float* input){
    cudaFree(input);
}

void c_free(float* input){
    free(input);
}

void block_cpu(){
    cudaEvent_t block;
    cudaEventCreateWithFlags(&block, cudaEventBlockingSync);
    cudaEventRecord(block);
    cudaEventSynchronize(block);
    cudaEventDestroy(block);
}


// TODOs
// implement functions whose functionality complies with restricted PyTorch functions
// There are two type of functions to implement for a single operation:
//   CUDA function
//   Interface function that calls CUDA function
// Note that interface function gets the float pointer already malloced at GPU
__global__ void _leaky_relu(float* input, float* output, int batch_size, int channels, int height, int width, int negative_slope){

    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(input[index] < 0) {
        output[index] = negative_slope * input[index];
    } else {
        output[index] = input[index];
    }

}
float* leaky_relu(float* input, int batch_size, int channels, int height, int width, int negative_slope){

    float* output, *device_input, *device_output;
    unsigned long size = batch_size * channels * height * width;

    cudaMalloc((void **) device_input, size * sizeof(float));
    cudaMalloc((void **) device_output, size * sizeof(float));

    cudaMemcpy(device_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(device_output, 0, size * sizeof(float));

    dim3 numBlocks(batch_size, channels);
    dim3 threadsPerBlock(height, width);

    _leaky_relu<<<numBlocks, threadsPerBlock>>>(device_input, device_output, batch_size, channels, height, width, negative_slope);

    cudaMemcpy(output, device_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    cuda_free(device_input);
    cuda_free(device_output);

    return output;

}

__global__ void _batch_norm(){}
float* batch_norm(){}

__global__ void _conv2d(){}
float* conv2d(){}

__global__ void _max_pool2d(float* input, int input_height, int input_width, int kernel_width, int kernel_height, int stride, float* output, int output_width, int output_height){
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < output_width && row < output_height) {
        int start_row = row * stride;
        int start_col = col * stride;
        float max_value = input[start_row * input_width + start_col];
        for (int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_width; j++) {
                float curr_value = input[(start_row+i)*input_width+(start_col+j)];
                if (curr_value > max_value) {
                    max_value = curr_value;
                }
            }
        }
        output[row*output_width+col] = max_value;
    }
    
}
float* max_pool2d(float* input, int input_height, int input_width, int kernel_width, int kernel_height, int stride){
    float* output, *device_input, *device_output;
    int output_height = floor((input_height-kernel_height)/stride) +1;
    int output_width = floor((input_width-kernel_width)/stride) +1;
    int output_size = output_height*output_width;
    int input_size = input_height* input_width*sizeof(float);

    cudaMalloc((void**)&device_input, input_size);
    cudaMalloc((void**)&device_output, output_size * sizeof(float));

    cudaMemcpy(device_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemset(device_output, 0, output_size*sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((output_width + threadsPerBlock.x - 1) / threadsPerBlock.x, (output_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    _max_pool2d<<<numBlocks, threadsPerBlock>>>(device_input, input_height, input_width, kernel_height, kernel_width, stride, device_output, output_width, output_height);
    cudaMemcpy(output, device_output,output_size, cudaMemcpyDeviceToHost);
    cudaFree(device_input);
    cudaFree(device_output);
    return output;
}

__global__ void _pad(){}
float* pad(){}

}