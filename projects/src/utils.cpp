#include "../includes/utils.hpp"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <Windows.h>

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

rgb_data black() {
	return rgb_data(0,0,0);
}

rgb_data white() {
	return rgb_data(255,255,255);
}

unsigned char greyscale(rgb_data pixel)
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

void logPixels(rgb_data* data, int width, int height)
{
	for (int y = 0; y < height; y++)
	{
		for(int x = 0; x < width; x++)
			printf("%c", greyToChar(greyscale(data[x + y*width])));
		printf("\n");
	}
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

int getOffset(int x, int y, int width)
{
	return x + y * width;
}