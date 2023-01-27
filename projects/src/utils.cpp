#include "../includes/utils.hpp"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

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

void displayResults2(float* input1, float* input2, float* output, int size)
{
	for (int i = 0; i < size; i++)
	{
		if(i%10 == 9) {
			printf("\n");			
		}
		printf("%f + %f = %f\n", input1[i],input2[i], output[i]);
	}
}