#include <math.h>
#include <stdio.h>
#include <chrono>
#include <utils.hpp>

__global__ void kernelsqrt(float* dev_input, float* dev_output){
	dev_output[threadIdx.x] = sqrtf(dev_input[threadIdx.x]);
}

int main(void)
{
	float input[DIM];
	float output[DIM];
	setupArray(input, DIM, MAX);

	float *dev_input, *dev_output;
	cudaMalloc(&dev_input, sizeof(float) * DIM);
	cudaMalloc(&dev_output, sizeof(float) * DIM);
	cudaMemcpy(dev_input, input, sizeof(float) * DIM, cudaMemcpyHostToDevice);	

	auto start = std::chrono::high_resolution_clock::now();
	kernelsqrt<<<1,DIM>>>(dev_input, dev_output);
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();

	cudaMemcpy(output, dev_output, sizeof(float) * DIM, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	#ifdef DEBUG
	displayResults(input, output, DIM);
	#endif
	double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-6;
	printf("Time to generate on CPU with multithreads: %.4lf ms\n", time_taken);
	return 0;
}