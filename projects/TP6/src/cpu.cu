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
	return x + y*width;
}

unsigned char gx(int x, int y, int width, rgb_data* input, int* gX, int gDim)
{

	int sum = 1020;
	for(int yoff = 0; yoff < gDim; yoff++)
		for(int xoff = 0; xoff < gDim; xoff++)
			sum += gX[getOffset(xoff,yoff,gDim)] * greyscale(input[getOffset(x - gDim/2 + xoff, y - gDim/2+ yoff, width)]);

	// variable number put as const
	//               v
	return(unsigned char) ((sum*255)/2040);
}

int main(void)
{
	int gDim = 3;
	int gX[3][3] =
	{
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1},
	};
	int height, width;
	rgb_data *input, *output;
	open_bitmap("res/source.bmp", &input, &width, &height);
	output = (rgb_data*) calloc(width*height*sizeof(rgb_data), 1);

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++)
		{
			unsigned char value = gx(x,y,width, input, (int*)gX, gDim);
			output[getOffset(x, y , width)].r = value;
			output[getOffset(x, y , width)].g = value;
			output[getOffset(x, y , width)].b = value;
		}
	}

	save_bitmap("res/cpu.bmp", width, height, 96, output);
	free(input);
	free(output);
}