#include <stdio.h>
#include <chrono>
#include <utils.hpp>

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
	open_bitmap("bmp/source.bmp", &input, &width, &height);
	output = (rgb_data*) calloc(width*height*sizeof(rgb_data), 1);

	auto start = std::chrono::high_resolution_clock::now();

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++)
		{
			unsigned char value = gx(x,y,width, input, (int*)gX, gDim);
			output[getOffset(x, y , width)].r = value;
			output[getOffset(x, y , width)].g = value;
			output[getOffset(x, y , width)].b = value;
		}
	}

	auto end = std::chrono::high_resolution_clock::now();
	double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-6;
	printf("Time to filter on CPU: %.4lf ms\n", time_taken);

	save_bitmap("bmp/cpu.bmp", width, height, 96, output);

	free(input);
	free(output);
}