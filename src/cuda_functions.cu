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

__global__ void _conv2d(int batch_size, float* input, int input_channels, int input_height, int input_width,
                              float* weight, float* bias, int kernel_height, int kernel_width,
                              int output_channel, int output_height, int output_width, int stride, float* output) {
    int batch = blockIdx.x;
    // int out_c = blockIdx.y;
    // int out_h = threadIdx.x;
    // int out_w = threadIdx.y;
    int out_c = threadIdx.x;
    // for (int batch = 0; batch < batch_size; batch++){
    // for (int out_c = 0; out_c < output_channel; out_c++){
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
            // if (output_index < 10){
            //     printf("index: %d; result: %f", output_index, result);
            // }    
            
        }
    }
    // }
    // }
}
float* conv2d(int batch_size, float* input, int input_channels, int input_height, int input_width,
              float* weight, float* bias, int kernel_height, int kernel_width,
              int output_channel, int output_height, int output_width, int stride){
    int output_size = batch_size * output_channel * output_height * output_width;
    float* d_input;
    float* d_weight;
    float* d_bias;
    float* d_output;
    // for (int i = 0; i < 10; i++){
    //     printf("aaaaaaaaa %f\n", input[i]);
    // }
    // Allocate device memory
    cudaMalloc((void**)&d_input, input_channels * input_height * input_width * batch_size * sizeof(float));
    cudaMalloc((void**)&d_weight, input_channels * kernel_height * kernel_width * output_channel * sizeof(float));
    cudaMalloc((void**)&d_bias, output_channel * sizeof(float));
    cudaMalloc((void**)&d_output, output_size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, input, input_channels * input_height * input_width * batch_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, input_channels * kernel_height * kernel_width * output_channel * sizeof(float), cudaMemcpyHostToDevice);
    if (bias != NULL){
    cudaMemcpy(d_bias, bias, output_channel * sizeof(float), cudaMemcpyHostToDevice);
    }
    else{
        d_bias = NULL;
    }
    // Configure grid and block dimensions
    dim3 grid(batch_size, output_channel);
    dim3 block(output_height, output_width);
    
    // Launch the CUDA kernel
    _conv2d<<<batch_size, output_channel>>>(batch_size, d_input, input_channels, input_height, input_width, 
                    d_weight, d_bias, kernel_height, kernel_width,
                    output_channel, output_height, output_width, stride, d_output);
    
    // Allocate memory for the output on the host
    float* output = (float*)malloc(output_size * sizeof(float));
    if (output == NULL) {
        // Handle memory allocation error
        return NULL;
    }

    // Copy the result from device to host
    // cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight);
    // cudaFree(d_output);
    // for (int i = 0; i < 10; i++){
    //     printf("aaaaaaaaa %f\n", output[i]);
    // }
    return d_output;


}


__global__ void _max_pool2d(int batch_size, float* input, int input_channel, int input_height, int input_width, int kernel_height, int kernel_width, int stride, float* output, int output_height, int output_width) {
    //Most intuitive: <<<batch_size, inputChannel>>
    int b = blockIdx.x;
    int c = threadIdx.x;
    printf("b: %d, c: %d\n", b, c);
    for (int h = 0; h < output_height; h++) {
        for (int w = 0; w < output_width; w++) {
            int i_h_start = h * stride;
            int i_w_start = w * stride;
            float result = 0.0;
            for (int kernel_h = 0; kernel_h < kernel_height; kernel_h++) {
                for (int kernel_w = 0; kernel_w < kernel_width; kernel_w++) {
                    for (int in_c = 0; in_c < input_channel; in_c++) {
                        int i_h = i_h_start + kernel_h;
                        int i_w = i_w_start + kernel_w;
                        if (i_h >= 0 && i_h < input_height && i_w >= 0 && i_w < input_width) {
                            int input_index = b * input_channel * input_height * input_width + in_c * input_height * input_width + i_h * input_width + i_w;
                            if (input[input_index] > result) {
                                result = input[input_index];
                            }
                        }
                    }
                }
            }
            output[b*input_channel*output_height*output_width + c*output_height*output_width + h*output_width + w] = result;
        }
    }
}

float* max_pool2d(int batch_size, float* input, int input_channel, int input_height, int input_width, int kernel_height, int kernel_width, int stride){
    float* d_input, *d_output;

    int output_height = CEIL_DIV(input_height - kernel_height + 1, stride);
    int output_width = CEIL_DIV(input_width - kernel_width + 1, stride);
    int output_size = batch_size * input_channel * output_height * output_width;

    cudaMalloc((void**)&d_input, input_channel * input_height * input_width * batch_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, input, input_channel * input_height * input_width * batch_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(batch_size, input_channel);
    dim3 block(output_height, output_width);

    _max_pool2d<<<batch_size, input_channel>>>(batch_size, d_input, input_channel, input_height, input_width, kernel_height, kernel_width, stride, d_output, output_height, output_width);

    

    cudaFree(d_input);
    return d_output;
}

// Unused
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

__global__ void _pad_fill(float* arr, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < size; i += gridDim.x * blockDim.x)
        arr[i] = value;
}

float *pad(float *input_ptr, int batch_size, int channels, int height, int width, int left, int right, int top, int bottom, float padding)
{
    float *d_input, *d_output;
    int new_height = height + top + bottom;
    int new_width = width + left + right;

    int input_size = batch_size * channels * height * width;
    int output_size =  batch_size * channels * new_height * new_width;
    // cudaMalloc((void **)&d_input, input_size);
    cudaMalloc((void **)&d_output, sizeof(float) * output_size);

    // cudaMemcpy(d_input, input_ptr, input_size, cudaMemcpyHostToDevice);
    // cudaMemset(d_output, 0, output_size);

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
                cudaMemcpy(d_output + new_offset + i * new_width + left, input_ptr + old_offset + i * width, input_size, cudaMemcpyHostToDevice);
        }

    // _pad<<<1,1>>>(d_input, d_output, batch_size, channels, height, width, left, right, top, bottom, padding);

    cudaDeviceSynchronize();

    return d_output;
}

}