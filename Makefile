# Define the directory path variable
SRC_DIR := ./src
BUILD_DIR := ./build

$(shell mkdir -p $(BUILD_DIR))

all: c cuda cuda_optimized

c: $(SRC_DIR)/c_functions.c
	gcc -Ofast -shared -fPIC -o $(BUILD_DIR)/lib_c_functions.so $^

cuda: $(SRC_DIR)/cuda_functions.cu
	nvcc -Xcompiler -fPIC -shared -O3 --ptxas-options=-v -o $(BUILD_DIR)/lib_cuda_functions.so $^

cuda_optimized: $(SRC_DIR)/cuda_functions_optimized.cu
	nvcc -Xcompiler -fPIC -shared -O3 --ptxas-options=-v -o $(BUILD_DIR)/lib_cuda_functions_optimized.so $^

clean:
	rm -rf $(BUILD_DIR)

# Validation
valc: c
	python3 -m lib.test.test C validations

valcu: cuda
	python3 -m lib.test.test CUDA validations

valcuo: cuda_optimized
	python3 -m lib.test.test CUDAOptimized validations

valall: valc valcu valcuo

# Stress
sssc: c
	python3 -m lib.test.test C stress

ssscu: cuda
	python3 -m lib.test.test CUDA stress

ssscuo: cuda_optimized
	python3 -m lib.test.test CUDAOptimized stress

sssall: sssc ssscu ssscuo


