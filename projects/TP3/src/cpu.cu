#include <math.h>
#include <stdio.h>
#include <chrono>
#include <utils.hpp>

int main(void)
{
	float input[SIZE];
	float output[SIZE];	
	setupArray(input, SIZE, MAX);
	
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < SIZE; i++)
		output[i]= sqrtf(input[i]);
	auto end = std::chrono::high_resolution_clock::now();
	#ifdef DEBUG
	displayResults(input, output, SIZE);
	#endif
	double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-6;
	printf("Time to generate with CPU: %.4lf ms\n", time_taken);
}