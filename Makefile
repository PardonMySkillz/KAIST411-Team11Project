# Define the directory path variable
SRC_DIR := ./src
BUILD_DIR := ./build

$(shell mkdir -p $(BUILD_DIR))

c: $(SRC_DIR)/c_functions.c
	gcc -Ofast -shared -fPIC -o $(BUILD_DIR)/lib_c_functions.so $^

cuda: $(SRC_DIR)/cuda_functions.cu
	nvcc -Xcompiler -fPIC -shared -O3 -o $(BUILD_DIR)/lib_cuda_functions.so $^

cuda_optimized: $(SRC_DIR)/cuda_functions_optimized.cu
	nvcc -Xcompiler -fPIC -shared -O3 -o $(BUILD_DIR)/lib_cuda_functions_optimized.so $^

clean:
	rm -rf $(BUILD_DIR)

all: c cuda cuda_optimized
