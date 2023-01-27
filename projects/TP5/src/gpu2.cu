#include <stdio.h>
#include <chrono>
#include <utils.hpp>

struct cuComplex {
	float r;
	float i;
	__device__ cuComplex( float a, float b ) : r(a), i(b) {}

	__device__ float magnitude2( void ) {
		return r * r + i * i;
	}

	__device__ cuComplex operator*(const cuComplex& a) {
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}

	__device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r+a.r, i+a.i);
	}
};

__device__ int julia( int x, int y ) {
	const float scale = 1.5;
	float jx = scale * (float)(DIM/2 - x)/(DIM/2);
	float jy = scale * (float)(DIM/2 - y)/(DIM/2);
	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);
	int i = 0;
	for (i=0; i<200; i++) {
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}
	return 1;
}


__global__ void kernel( unsigned char *ptr, int height, int width ) {
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int size = height * width;
	if(offset < size) {
		int y = offset/width;
		int x = offset - y * width;
		int juliaValue = julia( x, y );
		ptr[offset] = juliaValue;
	}
}

int main( void ) {	
	double time_taken;
	std::chrono::steady_clock::time_point start, end;

	int pixelQuantity = DIM * DIM;
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, 0);
	int capacity = prop.maxThreadsPerBlock;
	int blocQuantity = (pixelQuantity + capacity - 1)/capacity;
	unsigned char * data = (unsigned char*) malloc(pixelQuantity * sizeof(unsigned char));
	unsigned char * dev_data;
	cudaMalloc( (void**)&dev_data, pixelQuantity * sizeof(unsigned char));

	start = std::chrono::high_resolution_clock::now();
	kernel<<<blocQuantity,capacity>>>(dev_data, DIM, DIM);
	cudaDeviceSynchronize();
	end = std::chrono::high_resolution_clock::now();
	time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-6;
	printf("Time to compute with max thread capacity and multibloc: %.4lf ms\n", time_taken);
	
	start = std::chrono::high_resolution_clock::now();
	cudaMemcpy( data, dev_data, DIM * DIM * sizeof(unsigned char), cudaMemcpyDeviceToHost );
	cudaDeviceSynchronize();
	end = std::chrono::high_resolution_clock::now();
	time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-6;
	printf("Time to transfer data: %.4lf ms\n", time_taken);
	cudaFree(dev_data);
	
	start = std::chrono::high_resolution_clock::now();
	rgb_data* pixels = (rgb_data*) malloc(DIM * DIM * sizeof(rgb_data));
	for (int x = 0; x < DIM; x++) {
		for (int y = 0; y < DIM; y++) {
			int a = y * DIM + x;

			if (data[a] == 0) {
				pixels[a] = black();
			} else {
				pixels[a] = white();
			}
		}
	}
	free(data);
	end = std::chrono::high_resolution_clock::now();
	time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-6;
	printf("Time to transform data: %.4lf ms\n", time_taken);
	
	start = std::chrono::high_resolution_clock::now();
	save_bitmap("julia2.bmp", DIM, DIM, 96, pixels);
	free(pixels);
	end = std::chrono::high_resolution_clock::now();
	time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-6;
	printf("Time to save data to file: %.4lf ms\n", time_taken);
}


