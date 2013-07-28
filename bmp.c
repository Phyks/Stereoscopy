#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//Note : BMP files are written from left to right but from bottom to top !

struct file_header
{
	char signature[3];
	int size;
	int reserved;
	int offset;
};

struct img_header
{
	int headerSize;
	int width;
	int height;
	int planes;
	int depth;
	int compress;
	int imgSize;
	int resX;
	int resY;
	int colors;
	int importantColors;
};

struct file_header get_file_header(FILE* file)
{
	struct file_header fh={"", 0, 0, 0};
	
	fread(&fh.signature, 2, 1, file);
	fread(&fh.size, 4, 1, file);
	fread(&fh.reserved, 4, 1, file);
	fread(&fh.offset, 4, 1, file);
	
	return fh;
}

struct img_header get_img_header(FILE* file)
{
	struct img_header ih={0,0,0,0,0,0,0,0,0,0,0};
	
	fread(&ih.headerSize, 4, 1, file);
	fread(&ih.width, 4, 1, file);
	fread(&ih.height, 4, 1, file);
	fread(&ih.planes, 2, 1, file);
	fread(&ih.depth, 2, 1, file);
	fread(&ih.compress, 4, 1, file);
	fread(&ih.imgSize, 4, 1, file);
	fread(&ih.resX, 4, 1, file);
	fread(&ih.resY, 4, 1, file);
	fread(&ih.colors, 4, 1, file);
	fread(&ih.importantColors, 4, 1, file);
	
	return ih;
}

int main (int argc, char* argv[])
{
	if(argc < 2)
	{
		fprintf(stderr, "Error : you must specify a bmp image to open.\n");
		return EXIT_FAILURE;
	}
	
	//All the variables we'll use
	FILE *file;
	const char* filename;
	filename = argv[1];
	struct file_header fileHeader;
	struct img_header imageHeader;
	int x, y, yImage;
	unsigned char ***image;
	
	file = fopen(filename, "rb"); //First, open the file in binary mode
	
	if(file == NULL)
	{
		fprintf(stderr, "Error : can't open the bmp image.\n");
		return EXIT_FAILURE;
	}
	
	fileHeader = get_file_header(file); //Get the file header and check it is a bmp file
	
	if(strcmp(fileHeader.signature, "BM") != 0 && strcmp(fileHeader.signature, "BA") != 0 && strcmp(fileHeader.signature, "CI") != 0 && strcmp(fileHeader.signature, "CP") != 0 && strcmp(fileHeader.signature, "IC") != 0 && strcmp(fileHeader.signature, "PT") != 0)
	{
		fprintf(stderr, "Error : This file is not a valid BMP image.\n");
		return EXIT_FAILURE;
	}
	
	imageHeader = get_img_header(file); //Get the image header
	
	if(imageHeader.compress != 0)
	{
		fprintf(stderr, "Error : The BMP file is compressed. This program can't open such files.\n");
		return EXIT_FAILURE;
	}
	
	if(imageHeader.depth != 24) //If it is not a "true-color" RGB bmp file (ie 24 bits per pixel)
	{
		fprintf(stderr, "Error : This BMP is not a standard true-color BMP file. It may be a 256 colors BMP file for example.\n");
		return EXIT_FAILURE;
	}
	
	//Allocation dynamique pour l'image
	if((image = malloc(sizeof(*image) * imageHeader.height)) == NULL)
	{
		perror("malloc:");
		return EXIT_FAILURE;
	}
	for(y = 0; y <= imageHeader.height; y++)
	{
		if((image[y] = malloc(sizeof(**image) * imageHeader.width)) == NULL)
		{
			perror("malloc:");
			return EXIT_FAILURE;
		}
	}
	
	for(y = 0; y <= imageHeader.height; y++)
	{
		for(x = 0; x <= imageHeader.width; x++)
		{
			if((image[y][x] = malloc(sizeof(***image) * 3)) == NULL) //image[y][x][R,G,B]
			{
				perror("malloc:");
				return EXIT_FAILURE;
			}
			image[y][x][0] = 0;
			image[y][x][1] = 0;
			image[y][x][2] = 0;
		}
	}
	
	fseek(file, fileHeader.offset, SEEK_SET); //We don't get all the possible headers, so go to the start of the image informations
	
	for(y = 0; y <= imageHeader.height; y++) //Get all the values for all the pixels in the image
	{
		for(x = 0; x <= imageHeader.width; x++)
		{
			yImage = imageHeader.height - y; //Due to the fact that BMP file are written from bottom to top
			fread(&image[yImage][x][2], 1, 1, file);
			fread(&image[yImage][x][1], 1, 1, file);
			fread(&image[yImage][x][0], 1, 1, file);
		}
	}
	
	
	printf("Output is : (Coordinates of the pixel) : value of each channel");
	for(y = 0; y <= imageHeader.height; y++)
	{
		for(x = 0; x <= imageHeader.width; x++)
		{
			printf("(%d,%d) : R = %d,G = %d,B = %d\n", x, y, image[y][x][0], image[y][x][1], image[y][x][2]);
		}
	}
	
	free(image);
	fclose(file);
	
	return EXIT_SUCCESS;
}
