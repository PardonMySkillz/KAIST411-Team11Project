hello: hello.cu
	nvcc hello.cu -o hello

vec_add_naive: vec_add_naive.cu
	nvcc vec_add_naive.cu -o vec_add_naive

vec_add_threads: vec_add_threads.cu
	nvcc vec_add_threads.cu -o vec_add_threads

all: hello vec_add vec_add_naive vec_add_threads

clean: 
	rm hello vec_add vec_add_naive vec_add_threads 
