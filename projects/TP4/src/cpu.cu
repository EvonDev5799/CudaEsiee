#include <math.h>
#include <stdio.h>
#include <chrono>
#include <utils.hpp>

int main(void)
{
	float input1[SIZE];
	float input2[SIZE];
	float output[SIZE];	
	setupArray(input1, SIZE, MAX);
	setupArray(input2, SIZE, MAX);
	
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < SIZE; i++)
		output[i]= input1[i] + input2[i];
	auto end = std::chrono::high_resolution_clock::now();
	#ifdef DEBUG
	displayResults2(input1, input2, output, SIZE);
	#endif
	double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-6;
	printf("Time to generate: %.4lf ms\n", time_taken);
}