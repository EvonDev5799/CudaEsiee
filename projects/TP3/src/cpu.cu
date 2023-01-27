#include <math.h>
#include <stdio.h>
#include <chrono>
#include <utils.hpp>

int main(void)
{
	float input[DIM];
	float output[DIM];	
	setupArray(input, DIM, MAX);
	
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < DIM; i++)
		output[i]= sqrtf(input[i]);
	auto end = std::chrono::high_resolution_clock::now();
	#ifdef DEBUG
	displayResults(input, output, DIM);
	#endif
	double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-6;
	printf("Time to generate with CPU: %.4lf ms\n", time_taken);
}