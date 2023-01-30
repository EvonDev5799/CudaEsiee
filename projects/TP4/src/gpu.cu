#include <math.h>
#include <stdio.h>
#include <chrono>
#include <utils.hpp>

__global__ void kerneladd(float* dev_input1, float* dev_input2, float* dev_output){
	dev_output[blockIdx.x] = dev_input1[blockIdx.x] + dev_input2[blockIdx.x];
}

int main(void)
{
	float input1[DIM];
	float input2[DIM];
	float output[DIM];	
	setupArray(input1, DIM, MAX);
	setupArray(input2, DIM, MAX);

	float *dev_input1,*dev_input2, *dev_output;
	cudaMalloc(&dev_input1, sizeof(float) * DIM);
	cudaMalloc(&dev_input2, sizeof(float) * DIM);
	cudaMalloc(&dev_output, sizeof(float) * DIM);
	cudaMemcpy(dev_input1, input1, sizeof(float) * DIM, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_input2, input2, sizeof(float) * DIM, cudaMemcpyHostToDevice);
	
	auto start = std::chrono::high_resolution_clock::now();
	kerneladd<<<DIM,1>>>(dev_input1, dev_input2, dev_output);
	auto end = std::chrono::high_resolution_clock::now();

	cudaMemcpy(output, dev_output, sizeof(float) * DIM, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	#ifdef DEBUG
	displayResults2(input1, input2, output, DIM);
	#endif
	double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-6;
	printf("Time to compute on gpu: %.4lf ms\n", time_taken);

	cudaFree(dev_input1);
	cudaFree(dev_input2);
	cudaFree(dev_output);
}