#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

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
__global__ void _leaky_relu(float* input, float* output, int height, int width, float negative_slope){

    uint index = threadIdx.x;
    
    output[index] = input[index] < 0 ? input[index] * negative_slope : input[index];

}
float* leaky_relu(float* input, int height, int width, float negative_slope){

    float *device_input, *device_output;
    unsigned long size = height * width;

    cudaMalloc((void **) &device_input, size * sizeof(float));
    cudaMalloc((void **) &device_output, size * sizeof(float));

    cudaMemcpy(device_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(device_output, 0, size * sizeof(float));

    dim3 threadsPerBlock(height * width, 1, 1);

    _leaky_relu<<<1, threadsPerBlock>>>(device_input, device_output, height, width, negative_slope);

    cudaDeviceSynchronize();

    cuda_free(device_input);

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
    cudaDeviceSynchronize();
    cudaMemcpy(output, device_output,output_size, cudaMemcpyDeviceToHost);
    cudaFree(device_input);
    cudaFree(device_output);
    return output;
}

__global__ void _pad(float *input, float* output, int batch_size, int channels, int height, int width, int left, int right, int top, int bottom, float padding) {
    int new_height = height + top + bottom;
    int new_width = width + left + right;

    float *ptri = input;
    float *ptro = output;
    for (int b = 0; b < batch_size; b++)
        for (int c = 0; c < channels; c++)
        {
            // Pad the top
            for (int i = 0; i < top * new_width; i++)
                ptro[i] = padding;

            // Pad the middle
            for (int i = 0; i < height; i++)
            {
                // Left
                for (int j = 0; j < left; j++, ptro++)
                    *ptro = padding;
                //
                for (int j = 0; j < height; j++, ptri++, ptro++)
                    *ptro = *ptri;
                // Right
                for (int j = 0; j < right; j++, ptro++)
                    *ptro = padding;
            }

            // Pad the end
            for (int i = 0; i < bottom * new_width; i++, ptro++)
                *ptro = padding;
        }
}

float *pad(float *input_ptr, int batch_size, int channels, int height, int width, int left, int right, int top, int bottom, float padding)
{
    float *d_input, *d_output;
    int new_height = height + top + bottom;
    int new_width = width + left + right;

    int input_size = sizeof(float) * batch_size * channels * height * width;
    int output_size = sizeof(float) * batch_size * channels * new_height * new_width;
    printf("%d %d\n", input_size, output_size);
    cudaMalloc((void **)&d_input, input_size);
    cudaMalloc((void **)&d_output, output_size);

    cudaMemcpy(d_input, input_ptr, input_size, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, output_size);

    printf("ttt\n");
    _pad<<<1,1>>>(d_input, d_output, batch_size, channels, height, width,
            left, right, top, bottom, padding);

    cudaDeviceSynchronize();
    printf("ttt\n");
    cudaFree(d_input);

    return d_output;
}

}