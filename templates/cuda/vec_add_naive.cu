#include<stdio.h>
#include<stdlib.h>

#define N 4096*4096

__global__ void cuda_vec_mul(float* out, float* a, float* b, int n){
	for(int i=0; i<n; ++i){
		out[i] = a[i]*b[i];
	}
}

int main() {
	float *a, *b, *out;
	size_t bytes = sizeof(float) * N;

	a = (float*) malloc(bytes);
	b = (float*) malloc(bytes);
	out = (float*) malloc(bytes);

	for(int i = 0; i < N; ++i){
		a[i] = 1.0;
		b[i] = 2.0;
	}
	
	// cudaMalloc(void **devPtr, size_t count);
	// cudaFree(void *devPtr);
	// cudaMemcpy(void *dst, void *src, size_t count, cudaMemcpyKind kind);
	// kind canbe cudaMemcpyHostToDevice or cudaMemcpyDeviceToHost
	
	float *da, *db, *dout;
	
	cudaMalloc((void**)&da, bytes);
	cudaMalloc((void**)&db, bytes);
	cudaMalloc((void**)&dout, bytes);

	cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice);
	cuda_vec_mul<<<1,1>>>(dout, da, db, N);

	cudaMemcpy(out, dout, bytes, cudaMemcpyDeviceToHost);
	
	printf("%f\n", out[3]); // expect 2.

	cudaFree(da);
	cudaFree(db);
	cudaFree(dout);

	free(a);
	free(b);
	free(out);
}
