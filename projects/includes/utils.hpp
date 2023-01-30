#pragma once

#define DIM 1000
#define MAX 100000

float randomFloat(float max);
void setupArray(float* array, int size, float max);
void displayResults(float* input, float* output, int size);
void displayResults2(float* input1, float* input2, float* output, int size);

struct rgb_data {
    unsigned char r, g, b;
	rgb_data(unsigned char r,unsigned char g,unsigned char b) : r(r), g(g), b(b) {};
	rgb_data():r(0), g(0), b(0) {};
};

rgb_data black();
rgb_data white();
unsigned char greyscale(rgb_data pixel);
char greyToChar(unsigned char c);
void logPixels(rgb_data* data, int width, int height);
bool save_bitmap(const char* file_name, int width, int height, int dpi, rgb_data *pixel_data);
bool open_bitmap(const char* bmpname, rgb_data** ptrOutput, int* ptrWidth, int * ptrHeight);
int getOffset(int x, int y, int width);