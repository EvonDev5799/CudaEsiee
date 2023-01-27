#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>

#define SIZE 1000
#define MAX 100000

__global__ void kernelsqrt(float* dev_input, float* dev_output){
	dev_output[blockIdx.x] = sqrtf(dev_input[blockIdx.x]);
}

float randomFloat(float max)
{
	return ( ((float)rand()) / ((float)RAND_MAX) )*max;
}

void setupArray(float* array, int size, float max)
{
	srand(time(NULL));
	for (int i = 0; i < size; i++)
	{
		array[i] = randomFloat(max);
	}
}

void displayResults(float* input, float* output, int size)
{
	for (int i = 0; i < size; i++)
	{
		if(i%10 == 9) {
			printf("\n");			
		}
		printf("%f %f\n", input[i], output[i]);
	}
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
	auto end = std::chrono::high_resolution_clock::now();

	cudaMemcpy(output, dev_output, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	displayResults(input, output, SIZE);

	double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-6;
	printf("Time to generate: %lf ms\n", time_taken);
	return 0;
}