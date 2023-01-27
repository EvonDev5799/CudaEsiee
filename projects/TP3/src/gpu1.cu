#include <math.h>
#include <stdio.h>
#include <chrono>
#include <utils.hpp>

__global__ void kernelsqrt(float* dev_input, float* dev_output){
	dev_output[blockIdx.x] = sqrtf(dev_input[blockIdx.x]);
}

int main(void)
{
	float input[SIZE];
	float output[SIZE];
	setupArray(input, SIZE, MAX);

	float *dev_input, *dev_output;
	cudaMalloc(&dev_input, sizeof(float) * SIZE);
	cudaMalloc(&dev_output, sizeof(float) * SIZE);
	cudaMemcpy(dev_input, input, sizeof(float) * SIZE, cudaMemcpyHostToDevice);	

	auto start = std::chrono::high_resolution_clock::now();
	kernelsqrt<<<SIZE,1>>>(dev_input, dev_output);
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();

	cudaMemcpy(output, dev_output, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	#ifdef DEBUG
	displayResults(input, output, SIZE);
	#endif
	double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-6;
	printf("Time to generate on GPU with multibloc: %.4lf ms\n", time_taken);
	return 0;
}