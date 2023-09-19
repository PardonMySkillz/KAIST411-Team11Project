#include <stdio.h>
#include <stdlib.h>

void hello_world(){
    printf("hello world from c_funcsssstions.c!\n");
}

int add(int a, int b){
    return a + b;
}

int* add_vec1(int* a, int* b, int size){
    int* c = malloc(sizeof(int)*size);
    for (int i = 0; i < size; ++i){
        c[i] = a[i]+b[i];
    }
    return c;
}

void free_p(void* a){
    free(a);
}

float* conv2d(int batch_size, float* input, int input_channels, int input_height, int input_width,
              float* weight, int kernel_height, int kernel_width,
              int output_channel, int output_height, int output_width, int stride) {
	// Compute the output dimensions

	int output_size = batch_size * output_channel * output_height * output_width;
	float* output = (float*)malloc(output_size * sizeof(float));
	if (output == NULL) {
		// Handle memory allocation error
		return NULL;
	}

	// Initialize output to zeros
	for (int i = 0; i < output_size; i++) {
		output[i] = 0.0;
	}

    // Perform the convolution
	for (int batch = 0; batch < batch_size; batch++){
	    for (int out_c = 0; out_c < output_channel; out_c++) {
            for (int out_h = 0; out_h < output_height; out_h++) {
                for (int out_w = 0; out_w < output_width; out_w++) {
                    int i_h_start = out_h * stride;
                    int i_w_start = out_w * stride;

                    for (int kernel_h = 0; kernel_h < kernel_height; kernel_h++) {
                        for (int kernel_w = 0; kernel_w < kernel_width; kernel_w++) {
                            for (int in_c = 0; in_c < input_channels; in_c++) {
                                int i_h = i_h_start + kernel_h;
                                int i_w = i_w_start + kernel_w;
                                if (i_h >= 0 && i_h < input_height && i_w >= 0 && i_w < input_width) {
                                    int input_index = batch*input_channels*input_height*input_width + in_c*input_height*input_width + i_h*input_width + i_w;
                                    int kernel_index = out_c*input_channels*kernel_height*kernel_width + in_c*kernel_height*kernel_width + kernel_h*kernel_width + kernel_w;
                                    int output_index = batch*output_channel*output_height*output_width + out_c*output_height*output_width + out_h*output_width + out_w;
                                    output[output_index] += input[input_index] * weight[kernel_index];
                                }
                            }
                        }
                    }
                }
            }
	    }
	}
  
    return output;
}
