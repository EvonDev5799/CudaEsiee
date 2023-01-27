#include <stdio.h>
#include <windows.h>
#include <fileapi.h>
#include <chrono>

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

__device__ unsigned char greyscale(rgb_data pixel)
{
	unsigned int temp = pixel.r;
	temp += pixel.g;
	temp += pixel.b;
	return (unsigned char)(temp/3);
}

char greyToChar(unsigned char c) {
	if (c < 32)
		return 'a';
	if (c < 64)
		return 'b';
	if (c < 96)
		return 'c';
	if (c < 128)
		return 'd';
	if (c < 160)
		return 'e';
	if (c < 192)
		return 'f';
	if (c < 224)
		return 'g';
	return 'h';
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
	
	bf.bfType			= 0x4D42u;
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
	
	unsigned char* buff = (unsigned char*) calloc(buffsize, 1);

	for (int y = 0; y < height; y++) {
		unsigned char* ptr = buff + lineSize*y;
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

void open_bitmap(const char* bmpname, rgb_data** ptrOutput, int* ptrWidth, int * ptrHeight)
{
	HANDLE hFile;
	BYTE* buff;
	BITMAPFILEHEADER bf;
	BITMAPINFOHEADER bi;
	hFile = CreateFileA(bmpname, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	ReadFile(hFile, &bf, sizeof(bf), NULL, NULL);
	ReadFile(hFile, &bi, sizeof(bi), NULL, NULL);

	int paddingSize = ( 4 - (bi.biWidth * sizeof(rgb_data)) ) % 4;
	int lineSize = bi.biWidth * sizeof(rgb_data) + paddingSize;
	int buffsize = lineSize*bi.biHeight;

	buff = (BYTE*)malloc(buffsize);
	rgb_data* output = (rgb_data*)malloc(bi.biWidth*bi.biHeight* sizeof(rgb_data));

	ReadFile(hFile, buff, buffsize, NULL, NULL);
	
	int a = 0;
	rgb_data* ptr = (rgb_data*)buff;
	for(int y = 0; y < bi.biHeight; y++)
	{
		for(int x = 0; x < bi.biWidth; x++)
		{
			output[a].b = (*ptr).r;
			output[a].g = (*ptr).g;
			output[a].r = (*ptr).b;
			a++;
			ptr++;
		}
		ptr = (rgb_data*)(((BYTE*)ptr) + paddingSize);
	}
	free(buff);

	*ptrOutput = output;
	*ptrWidth = bi.biWidth;
	*ptrHeight = bi.biHeight;
	CloseHandle(hFile);
}

__device__ int getOffset(int x, int y, int width)
{
	return x + y * width;
}

__global__ void kernel(rgb_data* dev_input, rgb_data* dev_output, int* dev_gx, int dim_g)
{
	int sum = 1020;
	for(int yoff = 0; yoff < dim_g; yoff++)
		for(int xoff = 0; xoff < dim_g; xoff++)
		{
			int gxVal = dev_gx[getOffset(xoff,yoff,dim_g)];
			unsigned char inVal = greyscale(dev_input[getOffset(blockIdx.x + xoff, blockIdx.y + yoff, gridDim.x + 2*(dim_g/2))]);
			sum += gxVal * inVal;
		}
	unsigned char value = (sum*255)/2040;
	dev_output[getOffset(blockIdx.x + dim_g/2, blockIdx.y + dim_g/2, gridDim.x + 2*(dim_g/2))].r = value;
	dev_output[getOffset(blockIdx.x + dim_g/2, blockIdx.y + dim_g/2, gridDim.x + 2*(dim_g/2))].g = value;
	dev_output[getOffset(blockIdx.x + dim_g/2, blockIdx.y + dim_g/2, gridDim.x + 2*(dim_g/2))].b = value;
}

int main(void)
{
	int height, width;
	rgb_data *input, *output;
	rgb_data *dev_input, *dev_output;
	int* dev_gx;
	int dim_g = 3;
	int gx[3][3] =
	{
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1},
	};

	open_bitmap("res/source.bmp", &input, &width, &height);
	int buffer_size = width*height*sizeof(rgb_data);

	cudaMalloc(&dev_gx, sizeof(gx));
	cudaMalloc(&dev_input, buffer_size);	
	cudaMalloc(&dev_output, buffer_size);

	cudaMemcpy(dev_input, input, buffer_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_gx, gx, sizeof(gx), cudaMemcpyHostToDevice);

	free(input);
	dim3 grid(width - 2*(dim_g/2), height - 2*(dim_g/2));

	kernel<<<grid, 1>>>(dev_input,dev_output,dev_gx, dim_g);

	cudaFree(dev_gx);
	output = (rgb_data*) malloc(buffer_size);
	cudaMemcpy(output, dev_output, buffer_size, cudaMemcpyDeviceToHost);

	save_bitmap("res/gpu.bmp", width, height, 96, output);
	cudaFree(dev_input);
	cudaFree(dev_output);
	free(output);
}