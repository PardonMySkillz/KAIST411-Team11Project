#include<stdio.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    printf("Hello World from Host!\n");
    cuda_hello<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
