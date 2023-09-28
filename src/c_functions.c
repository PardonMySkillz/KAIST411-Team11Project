#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void c_free(void *ptr){
    free(ptr);
}

// TODOs
// implement a simple C functions whose functionality complies with restricted PyTorch functions
float* leaky_relu(int batch_size, float* input, int channels, int height, int width, int negative_slope){
    int input_size, output_size;
    input_size = output_size = batch_size * channels * height * width;
    float* output = (float*)malloc(output_size * sizeof(float));

    if(output == NULL) {
        return NULL;
    }

    memset(output, 0.0, sizeof(float) * output_size);

    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int index = b * channels * height * width + c * height * width + h * width + w;
                    float input_elem = input[index];
                    output[index] = input_elem < 0 ? negative_slope * input_elem : input_elem;
                }
            }
        }
    }

    return output;

}

float* batch_norm(float* input, int batch_size, int channels, int height, int width, float* running_mean, float* running_variable, float* weight, float* bias){

    int size = batch_size * channels * height * width;
    float* output = (float*)malloc(size * sizeof(float));

    if(output == NULL) return NULL;

    memset(output, 0 , size * sizeof(float));
    float e = 1e-5;
    for (int batch = 0; batch < batch_size; batch ++) {
        for (int channel = 0; channel < channels; channel ++) {
            for (int i = 0; i < width; i++) {
                for (int j = 0; j < height; j++) {
                    int io_index = batch * channels * width * height + channel * height * width + i * width + j;
                    int mv_index = batch * channels * width  + channel * width + i * width + j;
                    output[io_index] = weight[mv_index] * ((input[io_index] - running_mean[mv_index]) / (running_variable[mv_index] + e)) + bias[mv_index];
                }
                
            }
        }
    }

    return output;

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

void* max_pool2d(float* input, int input_height, int input_width, int kernel_width, int kernel_height, int stride) {
    int output_height = floor((input_height-kernel_height)/stride) +1;
    int output_width = floor((input_width-kernel_width)/stride) +1;
    int output_size = output_height*output_width;
    float* output = (float*)malloc(output_size * sizeof(float));
    if (output == NULL) {
        return NULL;
    }
    for (int row =0; row < output_height; row++) {
        for (int col =0; col < output_width; col++) {
            int start_row = row *stride;
            int start_col = col * stride;
            float max_value = input[start_row * input_width + start_col];
            for (int i = 0; i < kernel_height; i ++) {
                for (int j =0; j < kernel_width; j++) {
                    float curr_value = input[(start_row+i)*input_width+(start_col+j)];
                    if (curr_value > max_value) {
                        max_value = curr_value;
                    }
                }
            }
            output[row*output_width+col] = max_value;
        }
    }
    return output;
}

float* pad(){}
