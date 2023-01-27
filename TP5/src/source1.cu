#include <stdio.h>
#include <windows.h>
#include <fileapi.h>
#include <chrono>
#define DIM 10000

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


__global__ void kernel( unsigned char *ptr ) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;
	int juliaValue = julia( x, y );
	ptr[offset] = juliaValue;
}

struct rgb_data {
    unsigned char r, g, b;
	rgb_data(unsigned char r,unsigned char g,unsigned char b) : r(r), g(g), b(b) {};
	rgb_data():r(0), g(0), b(0) {};
};

rgb_data black() {
	return rgb_data(0,0,0);
}

rgb_data white() {
	return rgb_data(255,255,255);
}

// modified from https://ricardolovelace.com/blog/creating-bitmap-images-with-c-on-windows/ 
void save_bitmap(const char* file_name, int width, int height, int dpi, rgb_data *pixel_data) {
	int paddingSize = ( 4 - (width * sizeof(rgb_data)) ) % 4;
	int lineSize = width * sizeof(rgb_data) + paddingSize;
	int buffsize = lineSize*height;	
	int file_size = 54 + buffsize;

	HANDLE hFile;
	int ppm = dpi * 39.375;

	BITMAPFILEHEADER bf;
	BITMAPINFOHEADER bi;
	
	bf.bfType			= 0x4D42;
	bf.bfSize			= file_size;
	bf.bfReserved1		= 0;
	bf.bfReserved2		= 0;
	bf.bfOffBits		= 54;

	bi.biSize			= sizeof(bi);
	bi.biWidth			= width;
	bi.biHeight			= height;
	bi.biPlanes			= 1;
	bi.biBitCount		= 24;
	bi.biCompression	= 0;
	bi.biSizeImage		= file_size;
	bi.biXPelsPerMeter	= ppm;
	bi.biYPelsPerMeter	= ppm;
	bi.biClrUsed		= 0;
	bi.biClrImportant	= 0;

	hFile = CreateFileA(file_name, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	WriteFile(hFile, &bf, sizeof(bf), NULL, NULL);
	WriteFile(hFile, &bi, sizeof(bi), NULL, NULL);
	
	char* buff = (char*) calloc(buffsize, 1);

	for (int y = 0; y < height; y++) {
		char* ptr = buff + lineSize*y;
		for (int x = 0; x < width; x++) {
			int a = y * width + x;
			rgb_data RGB = pixel_data[a];

			ptr[0] = RGB.b;
			ptr[1] = RGB.g;
			ptr[2] = RGB.r;
			ptr += sizeof(rgb_data);
		}
	}
	
	WriteFile(hFile, buff, buffsize, NULL, NULL);
	free(buff);
	CloseHandle(hFile);
}

int main( void ) {
	long long time_taken;
	std::chrono::steady_clock::time_point start, end;

	unsigned char * data = (unsigned char*) malloc(DIM * DIM * sizeof(unsigned char));
	unsigned char * dev_data;
	cudaMalloc( (void**)&dev_data, DIM * DIM * sizeof(unsigned char));
	dim3 grid(DIM,DIM);

	start = std::chrono::high_resolution_clock::now();
	kernel<<<grid,1>>>(dev_data);
	end = std::chrono::high_resolution_clock::now();
	time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	printf("Time to compute: %lld microseconds\n", time_taken);
	
	start = std::chrono::high_resolution_clock::now();
	cudaMemcpy( data, dev_data, DIM * DIM * sizeof(unsigned char), cudaMemcpyDeviceToHost );
	cudaDeviceSynchronize();
	cudaFree(dev_data);
	end = std::chrono::high_resolution_clock::now();
	time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	printf("Time to transfer data: %lld microseconds\n", time_taken);
	
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
	time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	printf("Time to transform data: %lld microseconds\n", time_taken);
	
	start = std::chrono::high_resolution_clock::now();
	save_bitmap("julia.bmp", DIM, DIM, 96, pixels);
	free(pixels);
	end = std::chrono::high_resolution_clock::now();
	time_taken = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	printf("Time to save data to file: %lld microseconds\n", time_taken);

	
}


