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
__global__ void _leaky_relu(int batch_size, int channels, int height, int width, int negative_slope){}
float* leaky_relu(int batch_size, int channels, int height, int width, int negative_slope){

}

__global__ void _batch_norm(){}
float* batch_norm(){}

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


__global__ void _max_pool2d(){}
float* max_pool2d(){}

__global__ void _pad(){}
float* pad(){}

}