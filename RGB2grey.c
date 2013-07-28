unsigned char** RGB2grey(unsigned char*** image, int width, int height)
{
	unsigned char** grey;
	int x,y;
	
	if((grey = malloc(sizeof(*grey) * height)) == NULL)
	{
		perror("malloc:");
		return NULL;
	}
	for(y = 0; y <= height; y++)
	{
		if((grey[y] = malloc(sizeof(**grey) * width)) == NULL)
		{
			perror("malloc:");
			return NULL;
		}
	}
	
	for(y = 0; y <= height; y++)
	{
		for(x = 0; x <= width; x++)
		{
			grey[y][x] = (unsigned char) (0.2125*image[y][x][0] + 0.7154*image[y][x][1] + 0.0721*image[y][x][2]);
			//Formula found here : https://fr.wikipedia.org/wiki/Niveau_de_gris
		}
	}
	
	return grey;
}
