#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>

#define SIZE 1000
#define MAX 100000

__global__ void kerneladd(float* dev_input1, float* dev_input2, float* dev_output){
	dev_output[blockIdx.x] = dev_input1[blockIdx.x] + dev_input2[blockIdx.x];
}

float randomFloat(float max)
{
	return ( ((float)rand()) / ((float)RAND_MAX) )*max;
}

void setupArray(float* array, int size, float max)
{
	static int init = 0;
	if (!init){
		srand(time(NULL));
		init = 1;
	}
	for (int i = 0; i < size; i++)
	{
		array[i] = randomFloat(max);
	}
}

void displayResults(float* input1, float* input2, float* output, int size)
{
	for (int i = 0; i < size; i++)
	{
		if(i%10 == 0) {
			printf("\n");			
		}
		printf("%f + %f = %f\n", input1[i],input2[i], output[i]);
	}
}

int main(void)
{
	float input1[SIZE];
	float input2[SIZE];
	float output[SIZE];	
	setupArray(input1, SIZE, MAX);
	setupArray(input2, SIZE, MAX);

	float *dev_input1,*dev_input2, *dev_output;
	cudaMalloc(&dev_input1, sizeof(float) * SIZE);
	cudaMalloc(&dev_input2, sizeof(float) * SIZE);
	cudaMalloc(&dev_output, sizeof(float) * SIZE);
	cudaMemcpy(dev_input1, input1, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_input2, input2, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
	
	auto start = std::chrono::high_resolution_clock::now();
	kerneladd<<<SIZE,1>>>(dev_input1, dev_input2, dev_output);
	auto end = std::chrono::high_resolution_clock::now();

	cudaMemcpy(output, dev_output, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	displayResults(input1, input2, output, SIZE);

	double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-6;
	printf("Time to generate: %lf ms\n", time_taken);
}