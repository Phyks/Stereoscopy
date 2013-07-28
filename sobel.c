unsigned char** Sobel(unsigned char*** image, int width, int height)
//cf. http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html#sobel-derivatives and https://fr.wikipedia.org/wiki/Algorithme_de_Sobel 
{
	unsigned char** sobel;
	int x,y,i;
	unsigned char tempx, tempy;
	
	if((sobel = malloc(sizeof(*sobel) * height)) == NULL)
	{
		perror("malloc:");
		return NULL;
	}
	for(y = 0; y <= height; y++)
	{
		if((sobel[y] = malloc(sizeof(**sobel) * width)) == NULL)
		{
			perror("malloc:");
			return NULL;
		}
		for(x = 0; x <= width; x++)
		{
			sobel[y][x] = 0;
		}
	}
	
	for(y = 1; y < height; y++)
	{
		for(x = 1; x < width; x++)
		{
			tempx = -image[y-1][x-1]+image[y-1][x+1]-2*image[y][x-1]+2*image[y][x+1]-image[y+1][x-1]+image[y+1][x+1];
			tempy = -image[y-1][x-1]-2*image[y-1][x]-image[y-1][x+1]+image[y+1][x-1]+2*image[y+1][x]+image[y+1][x+1];
			sobel[y][x] = (int) floor(sqrt(tempx^2 + tempy^2));
		}
	}
	
	return sobel;
}

unsigned char*** add2images(unsigned char*** image1, unsigned char*** image2, int width, int height, int nChannels)
{
	unsigned char*** sum;
	int x,y,i;
	
	if((sum = malloc(sizeof(*sum) * height)) == NULL)
	{
		perror("malloc:");
		return NULL;
	}
	for(y = 0; y <= height; y++)
	{
		if((sum[y] = malloc(sizeof(**sum) * width)) == NULL)
		{
			perror("malloc:");
			return NULL;
		}
	}
	
	for(y = 0; y <= height; y++)
	{
		for(x = 0; x <= width; x++)
		{
			if((sum[y][x] = malloc(sizeof(***sum) * nChannels)) == NULL) //sum[y][x][R,G,B]
			{
				perror("malloc:");
				return NULL;
			}
			for(i = 0; i < nChannels; i++)
			{
				sum[y][x] = (unsigned char) ((image1[y][x][i] + image2[y][x][i])/2);
			}
		}
	}
	
	return sum;
}
