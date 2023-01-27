#include <windows.h>
#include <fileapi.h>

struct rgb_data {
    char r, g, b;
};

struct rgb_data black() {
	struct rgb_data r = {0, 0, 0};
	return r;
}

struct rgb_data white() {
	struct rgb_data r = {253, 253, 253};
	return r;
}

// modified from https://ricardolovelace.com/blog/creating-bitmap-images-with-c-on-windows/ 
void save_bitmap(const char *file_name, int width, int height, int dpi, struct rgb_data *pixel_data) {
	int paddingSize = ( 4 - (width * sizeof(struct rgb_data)) ) % 4;
	char paddingData[3] = {0, 0, 0};
	// create a file object that we will use to write our image
	HANDLE hFile; 
	// we want to know how many pixels to reserve
	int image_size = width * height;
	// a byte is 4 bits but we are creating a 24 bit image so we can represent a pixel with 3
	// our final file size of our image is the width * height * 4 + size of bitmap header
	int file_size = 54 + 3 * image_size + paddingSize * height;
	// pixels per meter https://www.wikiwand.com/en/Dots_per_inch
	int ppm = dpi * 39.375;

	// bitmap file header (14 bytes)
	// we could be savages and just create 2 array but since this is for learning lets
	// use structs so it can be parsed by someone without having to refer to the spec

	// since we have a non-natural set of bytes, we must explicitly tell the
	// compiler to not pad anything, on gcc the attribute alone doesn't work so
	// a nifty trick is if we declare the smallest data type last the compiler
	// *might* ignore padding, in some cases we can use a pragma or gcc's
	// __attribute__((__packed__)) when declaring the struct
	// we do this so we can have an accurate sizeof() which should be 14, however
	// this won't work here since we need to order the bytes as they are written
	BITMAPFILEHEADER bf;

	// bitmap image header (40 bytes)
	BITMAPINFOHEADER bi;

	// if you are on Windows you can include <windows.h>
	// and make use of the BITMAPFILEHEADER and BITMAPINFOHEADER structs
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

	hFile = CreateFile(file_name, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	WriteFile(hFile, &bf, sizeof(bf), NULL, NULL);
	WriteFile(hFile, &bi, sizeof(bi), NULL, NULL);

	// write out pixel data, one last important this to know is the ordering is backwards
	// we have to go BGR as opposed to RGB

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int a = y * width + x;
			struct rgb_data BGR = pixel_data[a];

			unsigned char red   = (BGR.r);
			unsigned char green = (BGR.g);
			unsigned char blue  = (BGR.b);

			// if you don't follow BGR image will be flipped!
			unsigned char color[3] = {
				blue, green, red
			};

			WriteFile(hFile, &color, sizeof(color), NULL, NULL);
		}
		WriteFile(hFile, &paddingData, paddingSize, NULL, NULL);
	}

	CloseHandle(hFile);
}

int main(void) {
	int width  = 15,
    height = 15,
    dpi = 96;

	struct rgb_data* pixels = malloc(width * height * sizeof(struct rgb_data));

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int a = y * width + x;

			if (a%2) {
				pixels[a] = white();
			} else {
				pixels[a] = black();
			}
		}
	}

	save_bitmap("black_border.bmp", width, height, dpi, pixels);
	free(pixels);
}