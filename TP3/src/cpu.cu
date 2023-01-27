#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>

#define SIZE 1000
#define MAX 100000

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
	
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < SIZE; i++)
		output[i]= sqrtf(input[i]);
	auto end = std::chrono::high_resolution_clock::now();

	displayResults(input, output, SIZE);

	double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-6;
	printf("Time to generate: %lf ms\n", time_taken);
}