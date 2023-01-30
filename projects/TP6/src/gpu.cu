#include <stdio.h>
#include <chrono>
#include <utils.hpp>

__device__ unsigned char greyscaleD(rgb_data pixel)
{
	unsigned int temp = pixel.r;
	temp += pixel.g;
	temp += pixel.b;
	return (unsigned char)(temp/3);
}

__device__ int getOffsetD(int x, int y, int width)
{
	return x + y * width;
}

__global__ void kernel(rgb_data* dev_input, rgb_data* dev_output, int* dev_gx, int dim_g)
{
	int sum = 1020;
	for(int yoff = 0; yoff < dim_g; yoff++)
		for(int xoff = 0; xoff < dim_g; xoff++)
		{
			int gxVal = dev_gx[getOffsetD(xoff,yoff,dim_g)];
			unsigned char inVal = greyscaleD(dev_input[getOffsetD(blockIdx.x + xoff, blockIdx.y + yoff, gridDim.x + 2*(dim_g/2))]);
			sum += gxVal * inVal;
		}
	unsigned char value = (sum*255)/2040;
	dev_output[getOffsetD(blockIdx.x + dim_g/2, blockIdx.y + dim_g/2, gridDim.x + 2*(dim_g/2))].r = value;
	dev_output[getOffsetD(blockIdx.x + dim_g/2, blockIdx.y + dim_g/2, gridDim.x + 2*(dim_g/2))].g = value;
	dev_output[getOffsetD(blockIdx.x + dim_g/2, blockIdx.y + dim_g/2, gridDim.x + 2*(dim_g/2))].b = value;
}

int main(int argc, char* argv[])
{
	if (argc != 3) {
		printf("how to call : %s input_file_path output_file_path\n", argv[0]);
		return;
	}

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

	open_bitmap(argv[1], &input, &width, &height);
	int buffer_size = width*height*sizeof(rgb_data);

	cudaMalloc(&dev_gx, sizeof(gx));
	cudaMalloc(&dev_input, buffer_size);	
	cudaMalloc(&dev_output, buffer_size);

	cudaMemcpy(dev_input, input, buffer_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_gx, gx, sizeof(gx), cudaMemcpyHostToDevice);

	dim3 grid(width - 2*(dim_g/2), height - 2*(dim_g/2));

	auto start = std::chrono::high_resolution_clock::now();

	kernel<<<grid, 1>>>(dev_input,dev_output,dev_gx, dim_g);
	cudaDeviceSynchronize();

	auto end = std::chrono::high_resolution_clock::now();
	double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-6;
	printf("Time to generate: %.4lf ms\n", time_taken);

	output = (rgb_data*) malloc(buffer_size);
	cudaMemcpy(output, dev_output, buffer_size, cudaMemcpyDeviceToHost);
	save_bitmap(argv[2], width, height, 96, output);

	free(input);
	free(output);

	cudaFree(dev_gx);
	cudaFree(dev_input);
	cudaFree(dev_output);
}