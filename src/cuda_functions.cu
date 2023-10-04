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
__global__ void _leaky_relu(float* input, float* output, int size, float negative_slope){

    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < size) output[index] = input[index] < 0 ? input[index] * negative_slope : input[index];

}
float* leaky_relu(float* input, int height, int width, float negative_slope){

    float *device_input, *device_output;
    unsigned long size = height * width;

    cudaMalloc((void **) &device_input, size * sizeof(float));
    cudaMalloc((void **) &device_output, size * sizeof(float));

    cudaMemcpy(device_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(device_output, 0, size * sizeof(float));

    int threadsPerBlock = 256;

    _leaky_relu<<<CEIL_DIV(size, threadsPerBlock), threadsPerBlock>>>(device_input, device_output, size, negative_slope);

    return device_output;

}

__global__ void _batch_norm(float* input, float* output, int channels, int height, int width, float* running_mean, float* running_var, float* weight, float* bias){

    uint b = blockIdx.x;
    uint c = blockIdx.y;

    uint idx = blockIdx.z * 128 + threadIdx.x;

    uint io_index = b * channels * height * width + c * height * width + idx;

    float e = 1e-5;
    if(idx < width * height)
        output[io_index] = weight[c] * ((input[io_index] - running_mean[c]) / sqrt(running_var[c] + e)) + bias[c];

}
float* batch_norm(float* input, int batch_size, int channels, int height, int width, float* running_mean, float* running_var, float* weight, float* bias){
    float *device_output;
    unsigned long io_size = batch_size * channels * height * width * sizeof(float);

    cudaMalloc((void**) &device_output, io_size);

    int threadsPerBlock = 128;
    dim3 numBlocks(batch_size, channels, CEIL_DIV(height * width, threadsPerBlock));

    _batch_norm<<<numBlocks, threadsPerBlock>>>(input, device_output, channels, height, width, running_mean, running_var, weight, bias);

    cudaDeviceSynchronize();

    return device_output;

}

__global__ void _conv2d(int batch_size, float* input, int input_channels, int input_height, int input_width,
                              float* weight, float* bias, int kernel_height, int kernel_width,
                              int output_channel, int output_height, int output_width, int stride, float* output) {
    int batch = blockIdx.x;
    int out_c = threadIdx.x;
    for (int out_h = 0; out_h < output_height; out_h++){
        for (int out_w = 0; out_w < output_width; out_w++){
            int i_h_start = out_h * stride;
            int i_w_start = out_w * stride;
            float result = 0.0;
            if (bias != NULL) {
                result += bias[out_c];
            }
            for (int kernel_h = 0; kernel_h < kernel_height; kernel_h++) {
                for (int kernel_w = 0; kernel_w < kernel_width; kernel_w++) {
                    for (int in_c = 0; in_c < input_channels; in_c++) {
                        int i_h = i_h_start + kernel_h;
                        int i_w = i_w_start + kernel_w;
                        if (i_h >= 0 && i_h < input_height && i_w >= 0 && i_w < input_width) {
                            int input_index = batch * input_channels * input_height * input_width + in_c * input_height * input_width + i_h * input_width + i_w;
                            int kernel_index = out_c * input_channels * kernel_height * kernel_width + in_c * kernel_height * kernel_width + kernel_h * kernel_width + kernel_w;
                            result += input[input_index] * weight[kernel_index];

                        }
                    }
                }
            }
            int output_index = batch * output_channel * output_height * output_width + out_c * output_height * output_width + out_h * output_width + out_w;
            output[output_index] = result;
            
        }
    }
}
float* conv2d(int batch_size, float* input, int input_channels, int input_height, int input_width,
              float* weight, float* bias, int kernel_height, int kernel_width,
              int output_channel, int output_height, int output_width, int stride){
    int output_size = batch_size * output_channel * output_height * output_width;
    float* d_output;
    cudaMalloc((void**)&d_output, output_size * sizeof(float));

    _conv2d<<<batch_size, output_channel>>>(batch_size, input, input_channels, input_height, input_width, 
                    weight, bias, kernel_height, kernel_width,
                    output_channel, output_height, output_width, stride, d_output);
    cudaDeviceSynchronize();
    return d_output;
}


__global__ void _max_pool2d(float* input, int input_height, int input_width, int kernel, int stride, float* output, int output_height, int output_width) {
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

float* max_pool2d(int batch_size, float* d_input, int input_channel, int input_height, int input_width, int kernel_height, int kernel_width, int stride){
    float *d_output;

    int output_height = (input_height - kernel_height) / stride + 1;
    int output_width = (input_width - kernel_width) / stride + 1;
    int output_size = batch_size * input_channel * output_height * output_width;

    cudaMalloc((void**)&d_output, output_size * sizeof(float));

    // Smallest ow * oh = 32 * 32
    int blockSize = 256;
    dim3 numBlocks(CEIL_DIV(output_height * output_width, blockSize), batch_size * input_channel, 1);

    _max_pool2d<<<numBlocks, blockSize>>>(d_input, input_height, input_width, kernel_height, stride, d_output, output_height, output_width);
    cudaDeviceSynchronize();

    cudaFree(d_input);
    return d_output;
}

// Unused
__global__ void _pad(float *input, float* output, int batch_size, int channels, int height, int width, int left, int right, int top, int bottom, float padding) {
    // int new_height = height + top + bottom;
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

__global__ void _pad_fill(float* arr, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < size; i += gridDim.x * blockDim.x)
        arr[i] = value;
}

float *pad(float *input_ptr, int batch_size, int channels, int height, int width, int left, int right, int top, int bottom, float padding)
{
    float *d_output;
    int new_height = height + top + bottom;
    int new_width = width + left + right;

    // int input_size = batch_size * channels * height * width;
    int output_size =  batch_size * channels * new_height * new_width;

    cudaMalloc((void **)&d_output, sizeof(float) * output_size);

    int blockSize = 256;
    int numBlocks = (output_size + blockSize - 1) / blockSize;
    _pad_fill<<<numBlocks, blockSize>>>(d_output, output_size, padding);
    cudaDeviceSynchronize();
    
    
    for (int b = 0; b < batch_size; b++)
        for (int c = 0; c < channels; c++)
        {
            int old_offset
                = b * channels * height * width 
                + c * height * width;
            int new_offset 
                = b * channels * new_height * new_width 
                + c * new_height * new_width
                + top * new_width;
            for(int i = 0; i < height; i++)
                cudaMemcpyAsync(d_output + new_offset + i * new_width + left, input_ptr + old_offset + i * width, width * sizeof(float), cudaMemcpyDeviceToDevice);
        }

    cudaDeviceSynchronize();

    return d_output;
}

}