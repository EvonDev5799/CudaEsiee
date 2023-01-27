#include <stdio.h>

__global__ void cuda_hello(){
	printf("Hello World du GPU\n");
}

int main(void) {
	printf("Hello World du CPU\n");
	cuda_hello<<<1,1>>>();
	return EXIT_SUCCESS;
}