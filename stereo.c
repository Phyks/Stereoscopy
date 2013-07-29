/* 
--------------------------------------------------------------------------------------------------------------------
------- STEREOSCOPY DISTANCE MEASUREMENT ---------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------

Please, send me an email if you use or modify this program, just to let me know if this program is useful to anybody or how did you improve it :) You can also send me an email to tell me how lame it is ! :)

TLDR; I don't give a damn to anything you can do using this code. It would just be nice to quote where the original code comes from.

--------------------------------------------------------------------------------
"THE NO-ALCOHOL BEER-WARE LICENSE" (Revision 42):
Phyks (webmaster@phyks.me) wrote this file. As long as you retain this notice you
can do whatever you want with this stuff (and you can also do whatever you want
with this stuff without retaining it, but that's not cool...). If we meet some
day, and you think this stuff is worth it, you can buy me a beer soda
in return.
Phyks
---------------------------------------------------------------------------------

--------------------------------------------------------------------------------------------------------------------

This program computes the mean distance between a stereoscopic camera and an object. 
We assume both cameras have the same specs (which are the diameter of the camera's field stop and the focal length).

The code is commented but for more details about the algorithm, please refer to the following article :
http://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1064&context=ecetr

The distance estimation is just an application of the algorithm described in the previous article. To compute the 
displacement vector between the two images, we study the distances between subsets of the images. This is just like if
we had a mask that we could place over the left image and then, we could search the best matching part of the right image.

We are in a three dimensional space (RGB) so we just compute the euclidian distances between each corresponding pixels of
the subsets and then, we compute the mean distance between the two subsets.

Another option is to use FFT transform and phase correlation algorithm. More information cn be found here :
http://docs.opencv.org/doc/tutorials/imgproc/histograms/template_matching/template_matching.html and http://en.wikipedia.org/wiki/Phase_correlation

The sobel argument makes the program perform a Sobel edge detection first.

Notes : 
- We use the openCV library to easily load and go through the images. OpenCV is a C library for computer vision licensed
under a BSD License (http://opensource.org/licenses/bsd-license.php)
- We use FFTW library to perform FFT. FFTW is a C library to compute FFT licensed under the GNU General Public License (GPL, see http://www.fftw.org/doc/License-and-Copyright.html)
- We compute both X and Y displacements but we'll focus on X displacements to determine the mean distance of the object 
as the cameras are supposed to be on the same horizontal plane (Y displacements = vertical displacements are negligible)
- X corresponds to the abscissa and Y to the ordinate of the pixel in the image. They both goes from 0 to ...

Pixels are numeroted this way :
(0,0)	(1,0)	(2,0)	...		(Width,0)
(1,0)	(1,1)	(2,1)	...		(Width,1)
 ...	 ...	 ...	...			...
(0,Height)		 ...			(Width, Height)
--------------------------------------------------------------------------------------------------------------------
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <math.h>
#include <getopt.h>
#include <time.h>
#include <pthread.h>
#include <fftw3.h>

#ifndef M_PI
 #define M_PI 3.14159265358979323846
#endif 

// Structure to store the coordinates of the best matching subsets
struct minXY_struct
{
	float min;
    int X;
    int Y;
};

struct data_find_common
{
	const IplImage* img2;
	const uchar* p1;
	int widthStep1;
	int X_1;
	int Y_1;
	int subset_size;
	int X_start;
	int Y_start;
	int X_end;
	int Y_end;
	struct minXY_struct* temp;
};

/*
--------------------------------------------------------------------------------------------------------------------
------- Find common ------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------
This function takes 10 parameters :
- The image to search in
- The pointer to the pixel in the left-hand corner of the subset we are working on (and its human-readable 
coordinates X_1 and Y_1)
- The size of the subsets we consider
- The coordinates of the part of the image to search in

This function will search the best matching subset in the right image to the current subset in the left image.

It will go through the image img2 (aka the right image) and for each pixel, will compute the mean RGB distance between :
- The subset we are working on in the left image
- A subset of the same size in img2 with the current pixel as the left-hand corner

As the images have been opened in color mode, we are working in a three dimensional space (RGB) and we just compute the 
euclidian distance pixel to pixel using the colors as components. Then, we compute the mean of this distance over the subset.

This function returns a structure with the X and Y coordinates of the left-hand corner of the best matching subset.
--------------------------------------------------------------------------------------------------------------------
*/

static void *find_common(void *data)
{
	struct data_find_common *data_args = data;
	
	//Variables to explore the right image
	uchar *p2, *line2;
	//And the subsets
	int i = 0, j = 0, k = 0;
	
	//To store the distance between the two subsets (distance) and the distances pixel to pixel (distance_temp)
	float distance = 0, distance_temp = 0;
	//Human readables coordinates corresponding to p2
	int X_temp = data_args->X_start, Y_temp = data_args->Y_start;
	
	//The structure it will return. We start with a maximal distance in the left-hand corner of the image
	struct minXY_struct minXY={255, 0, 0};
	
	//We explore the image line by line from left to right
	//Note : we only explore an area with a height equal to the quarter of th total height of the image because the vertical displacement is supposed to be close to 0
	for (line2 = (uchar*) data_args->img2->imageData + data_args->Y_start*data_args->img2->widthStep;
	line2 <= (uchar*) data_args->img2->imageData + data_args->Y_end*data_args->img2->widthStep;
	line2 += data_args->img2->widthStep)
	{
		for (p2 = line2 + data_args->X_start*data_args->img2->nChannels; p2 <= line2 + data_args->X_end*data_args->img2->nChannels; p2 += data_args->img2->nChannels)
		{
			distance = 0;
			//For each pixels, we calculate the mean distance for the subset_size pixel box around it
			for(i = 0; i<data_args->subset_size; i++) //i = iterate through the pixels in a line
			{
				for(j = 0; j<data_args->subset_size; j++) //j = iterate through the lines
				{
					distance_temp = 0;
					
					for(k = 0; k < data_args->img2->nChannels; k++) //k = iterate through the channels
					{
						distance_temp += pow(*(p2 + i*data_args->img2->nChannels + k + j*data_args->img2->widthStep) - *(data_args->p1 + i*data_args->img2->nChannels + k + j*data_args->widthStep1),2); //Works because img1 and img2 have the same number of channels
					}
					distance += sqrt(distance_temp);
				}
			}
			distance /= pow(data_args->subset_size,2); //Distance is the mean distance over the subset
			
			//And if it is better than the previous min, we store the new coordinates
			//We prefer a smaller distance
			// But at equal distance, we prefer the closer pixels and the minimal vertical displacement (as it is supposed to be 0 because cameras are on a same horizontal plane)
			if(distance < minXY.min || (distance == minXY.min && (sqrt(pow(X_temp-data_args->X_1, 2) + pow(Y_temp-data_args->Y_1, 2)) < sqrt(pow(minXY.X-data_args->X_1, 2) + pow(minXY.Y-data_args->Y_1, 2)) || abs(Y_temp - data_args->Y_1) < abs(minXY.Y - data_args->Y_1))))
			{
				minXY.X = X_temp;
				minXY.Y = Y_temp;
				minXY.min = distance;
			}
			
			X_temp++;
		}
		X_temp = data_args->X_start;
		Y_temp++;
	}
	
	//Store the informations (coordinates and minimum) in a structure
	*data_args->temp = minXY;
	
	return NULL;
}

/*
--------------------------------------------------------------------------------------------------------------------
------- Compute mean distance --------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------

This function takes 6 parameters :
- The abscissas of the pixels in the left and in the right image
- The width of the two images
- The distance between the cameras and the angle theta computed previously

We compute the mean distance of the given object by the algorithm described in the article above and return it or 
return -1 if there was an error.
--------------------------------------------------------------------------------------------------------------------
*/

float compute_mean_distance(int X_1, int X_2, int widthL, int widthR, float theta, float deltaX)
{
	float alpha_1, alpha_2;
	
	//If the object is located between two cameras
	if(X_1 > widthL / 2 && X_2 < widthR / 2)
	{
		alpha_1 = atan((X_1 - (widthL / 2))*tan(theta)/(widthL / 2));
		alpha_2 = atan(((widthR / 2) - X_2)*tan(theta)/(widthR / 2));
		
		return tan(M_PI/2 - alpha_1)*tan(M_PI/2 - alpha_2)*deltaX / (tan(M_PI/2 - alpha_1) + tan(M_PI/2 - alpha_2));
	}
	
	//If the object is on the left side of both cameras
	if(X_1 < widthL / 2 && X_2 < widthR / 2)
	{
		alpha_1 = atan(((widthL / 2) - X_1)*tan(theta)/(widthL / 2));
		alpha_2 = atan(((widthR / 2) - X_2)*tan(theta)/(widthR / 2));
		
		return sin(M_PI/2 - alpha_1)*sin(M_PI/2 - alpha_2)*deltaX / sin(alpha_2 - alpha_1);
	}
	
	//If the object is on the right side of both cameras
	if(X_1 > widthL / 2 && X_2 > widthR / 2)
	{
		alpha_1 = atan((X_1 - (widthL / 2))*tan(theta)/(widthL / 2));
		alpha_2 = atan((X_2 - (widthR / 2))*tan(theta)/(widthR / 2));
		
		return sin(M_PI/2 - alpha_1)*sin(M_PI/2 - alpha_2)*deltaX / sin(alpha_1 - alpha_2);
	}
	
	//If the object is exactly in front of the left camera
	if(X_1 == widthL / 2)
	{
		alpha_2 = atan(((widthR / 2) - X_2)*tan(theta)/(widthR / 2));
		
		return tan(M_PI/2 - alpha_2)*deltaX;
	}
	
	//If the object is exactly in front of the right camera
	if(X_2 == widthR / 2)
	{
		alpha_1 = atan((X_1 - (widthL / 2))*tan(theta)/(widthL / 2));
		
		return tan(M_PI/2 - alpha_1)*deltaX;
	}
	
	return -1;
}

/*
--------------------------------------------------------------------------------------------------------------------
------- Print help message -----------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------
*/

void print_help(FILE* stream)
{
	fprintf(stream, "\nThis program computes the mean distance between a stereoscopic camera and an object. We assume both cameras have the same specs (which are the diameter of the camera's field stop and the focal length).\n");
	fprintf(stream, "\nFor more details about the algorithm, please refer to the following article : http://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1064&context=ecetr\n");
	fprintf(stream, "\nThe distance estimation is just an application of the algorithm described in the previous article. To compute the displacement vector between the two images, we study the distances between subsets of the images. This is just like ifwe had a mask that we could place over the left image and then, we could search the best matching part of the right image.\n");
	fprintf(stream, "\nNotes : \n");
	fprintf(stream, "- We use the openCV library to easily load and go through the images. OpenCV is a C library for computer vision licensed under a BSD License (http://opensource.org/licenses/bsd-license.php)\n");
	fprintf(stream, "- We compute both X and Y displacements but we'll focus on X displacements to determine the mean distance of the object as the cameras are supposed to be on the same horizontal plane (Y displacements = vertical displacements are negligible)\n");
	fprintf(stream, "- X corresponds to the abscissa and Y to the ordinate of the pixel in the image. They both goes from 0 to ...\n");
	fprintf(stream, "\nPixels are numeroted this way :\n");
	fprintf(stream, "(0,0)\t(1,0)\t(2,0)\t...\t(Width,0)\n");
	fprintf(stream, "(1,0)\t(1,1)\t(2,1)\t...\t(Width,1)\n");
	fprintf(stream, " ...\t ...\t ...\t...\t  ...\n");
	fprintf(stream, "(0,Height)\t ...\t\t(Width, Height)\n");
	fprintf(stream, "\n======================================\n");
	fprintf(stream, "\n Usage : \n");
	fprintf(stream, "\t -l (--left) \t Path to the left image\n");
	fprintf(stream, "\t -r (--right) \t Path to the right image\n");
	fprintf(stream, "\t -D (--diameter) (float) Diameter of the camera's field stop\n");
	fprintf(stream, "\t -f (--focal) \t (float) Focal length of the camera\n");
	fprintf(stream, "\t -d (--delta) \t (float) Distance between the two cameras\n");
	fprintf(stream, "\t -s (--size) \t [Optional] (int) Size of the subsets used, default is 50\n");
	fprintf(stream, "\t -x (--abscissa) [Optional] (int) Abscissa of the pixel to compute the mean distance\n");
	fprintf(stream, "\t -y (--ordinate) [Optional] (int) Ordinate of the pixel to compute the mean distance\n");
	fprintf(stream, "\t -c (--choose) \t [Optional] To display the image and select which pixel you want to work with (same as -x and -y but useful if you don't know the coordinates of the pixel)\n");
	fprintf(stream, "\t -o (--output) \t [Optional] Path to a file to store the output\n");
	fprintf(stream, "\t -S (--sobel) \t [Optional] Run a Sobel edges detection algorithm on both left and right images first to work only on edges. This option won't have any effect if working in single pixel mode.\n");
	fprintf(stream, "\t -F (--fft) \t [Optional] Use cross-correlation and fft based method (faster but a little bit less accurate).\n");
	fprintf(stream, "\t -H (--hamming) \t [Optional] Use an hamming window to improve FFT (but slightly slower).\n");
	fprintf(stream, "\t -h (--help) \t Display this error message\n");
}

/*
--------------------------------------------------------------------------------------------------------------------
------- Min -----------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------
Return the min of a and b
*/

int minimum(int a, int b)
{
	if(a < b)
		return a;
	else
		return b;
}

/*
--------------------------------------------------------------------------------------------------------------------
------- Max --------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------
Return the max of a and b
*/

int maximum(int a, int b)
{
	if(a > b)
		return a;
	else
		return b;
}

/*
--------------------------------------------------------------------------------------------------------------------
------- Mouse Handler ----------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------
Handle the left button click and get the coordinates of the current pixel
*/

void mouseHandler(int event, int x, int y, int flags, void* param)
{	 
	if(event == CV_EVENT_LBUTTONDOWN) //If left click
	{
		int** input = param;
		
		*input[0] = x; //Get the current coordinates
		*input[1] = y;
		cvDestroyAllWindows(); //And close the window
	}
}

/*
--------------------------------------------------------------------------------------------------------------------
------- Find_FFT ---------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------
Find a template (tpl) in an image (src)

Note : template and src must be the same size
*/

struct Peak
{
    CvPoint pt;
    double  maxval;
};

/*
	Old version using OpenCV
---------------------------------
struct Peak Find_FFT(IplImage* src, IplImage* tpl)
{
    CvSize imgSize = cvSize(src->width, src->height); //Size of the src image
    CvMat tmp;
    
    //src and tpl must be the same size
    assert(src->width == tpl->width);
    assert(src->height == tpl->height);
        
    // Allocate floating point frames used for DFT (real, imaginary and complex) 
	IplImage* realInput = cvCreateImage( imgSize, IPL_DEPTH_64F, 1 ); 
	IplImage* imaginaryInput = cvCreateImage( imgSize, IPL_DEPTH_64F, 1 ); 
	IplImage* complexInput = cvCreateImage( imgSize, IPL_DEPTH_64F, 2 );

	//Find best size for DFT
	int nDFTHeight= cvGetOptimalDFTSize( imgSize.height ); 
	int nDFTWidth= cvGetOptimalDFTSize( imgSize.width ); 
	CvSize dftSize = cvSize(nDFTWidth, nDFTHeight);

	//Images that will store DFT
	CvMat* src_DFT = cvCreateMat( nDFTHeight, nDFTWidth, CV_64FC2 ); 
	CvMat* tpl_DFT = cvCreateMat( nDFTHeight, nDFTWidth, CV_64FC2 );

	//Images used to compute modulus
	IplImage* imageRe = cvCreateImage( dftSize, IPL_DEPTH_64F, 1 );
	IplImage* imageIm = cvCreateImage( dftSize, IPL_DEPTH_64F, 1 );
	IplImage* imageImMag = cvCreateImage( dftSize, IPL_DEPTH_64F, 1 ); 
	IplImage* imageMag = cvCreateImage( dftSize, IPL_DEPTH_64F, 1 ); 

    // Processing of src 
    cvScale(src,realInput,1.0,0); //Convert it to CV_32F (float)
    cvZero(imaginaryInput); 
    cvMerge(realInput,imaginaryInput,NULL,NULL,complexInput);
    cvGetSubRect(src_DFT,&tmp,cvRect(0,0,src->width,src->height)); 
    cvCopy(complexInput,&tmp,NULL);
    if (src_DFT->cols>src->width)
    { 
        cvGetSubRect(src_DFT,&tmp,cvRect(src->width,0,src_DFT->cols-src->width,src->height)); 
        cvZero(&tmp); 
    } 
    cvDFT(src_DFT,src_DFT,CV_DXT_FORWARD,complexInput->height); //Process DFT 

    // Processing of tpl
    cvScale(tpl,realInput,1.0,0); 
    cvMerge(realInput,imaginaryInput,NULL,NULL,complexInput); 
    cvGetSubRect(tpl_DFT,&tmp,cvRect(0,0,tpl->width,tpl->height)); 
    cvCopy(complexInput,&tmp,NULL); 
    if (tpl_DFT->cols>tpl->width) 
    { 
        cvGetSubRect(tpl_DFT,&tmp,cvRect(tpl->width,0,tpl_DFT->cols-tpl->width,tpl->height)); 
        cvZero( &tmp ); 
    } 
    cvDFT(tpl_DFT,tpl_DFT,CV_DXT_FORWARD,complexInput->height);

    // Multiply spectrums of the scene and the model (use CV_DXT_MUL_CONJ to get correlation instead of convolution) 
    cvMulSpectrums(src_DFT,tpl_DFT,src_DFT,CV_DXT_MUL_CONJ); 

    // Split Fourier in real and imaginary parts 
    cvSplit(src_DFT,imageRe,imageIm,0,0); 

    // Compute the magnitude of the spectrum components: Mag = sqrt(Re^2 + Im^2) 
    cvPow( imageRe, imageMag, 2.0 ); 
    cvPow( imageIm, imageImMag, 2.0 ); 
    cvAdd( imageMag, imageImMag, imageMag, NULL ); 
    cvPow( imageMag, imageMag, 0.5 ); 

    // Normalize correlation (Divide real and imaginary components by magnitude) 
    cvDiv(imageRe,imageMag,imageRe,1.0); 
    cvDiv(imageIm,imageMag,imageIm,1.0); 
    cvMerge(imageRe,imageIm,NULL,NULL,src_DFT); 

    //Inverse dft 
    cvDFT( src_DFT, src_DFT, CV_DXT_INVERSE_SCALE, complexInput->height ); 
    cvSplit( src_DFT, imageRe, imageIm, 0, 0 ); 

	//Find the peak (greatest magnitude)
    double minval = 0.0; 
    double maxval = 0.0; 
    CvPoint minloc; 
    CvPoint maxloc; 
    cvMinMaxLoc(imageRe,&minval,&maxval,&minloc,&maxloc,NULL); 

    int x=maxloc.x; // log range 
    //if (x>(imageRe->width/2)) 
    //        x = x-imageRe->width; // positive or negative values 
    int y=maxloc.y; // angle 
    //if (y>(imageRe->height/2)) 
    //        y = y-imageRe->height; // positive or negative values 

	struct Peak pk;
	pk.maxval= maxval;
	pk.pt=cvPoint(x,y);

	cvReleaseImage(&realInput);
	cvReleaseImage(&imaginaryInput);
	cvReleaseImage(&complexInput);
	cvReleaseImage(&imageRe);
	cvReleaseImage(&imageIm);
	cvReleaseImage(&imageImMag);
	cvReleaseImage(&imageMag);
	
	cvReleaseMat(&src_DFT);
	cvReleaseMat(&tpl_DFT);

	return pk;
}
-----------------------------------------*/

/* New version, using FFTW (multithreaded)
------------------------------------------
*/

struct Peak Find_FFT(IplImage* src, IplImage* tpl, int hamming)
{
	int i, j, k = 0;
	double tmp; //To store the modulus temporarily

	//src and tpl must be the same size
	assert(src->width == tpl->width);
	assert(src->height == tpl->height);

	// Get image properties
	int width    = src->width;
	int height   = src->height;
	int step     = src->widthStep;
	int fft_size = width * height;
	
	fftw_init_threads(); //Initialize FFTW for multithreading with a max number of 2 threads (more is not efficient)
	fftw_plan_with_nthreads(2);
	
	//Allocate arrays for FFT of src and tpl
	fftw_complex *src_spatial = ( fftw_complex* )fftw_malloc( sizeof( fftw_complex ) * width * height );
	fftw_complex *src_freq = ( fftw_complex* )fftw_malloc( sizeof( fftw_complex ) * width * height );
	
	fftw_complex *tpl_spatial = ( fftw_complex* )fftw_malloc( sizeof( fftw_complex ) * width * height );
	fftw_complex *tpl_freq = ( fftw_complex* )fftw_malloc( sizeof( fftw_complex ) * width * height );

	fftw_complex *res_spatial = ( fftw_complex* )fftw_malloc( sizeof( fftw_complex ) * width * height ); //Result = Cross correlation
	fftw_complex *res_freq = ( fftw_complex* )fftw_malloc( sizeof( fftw_complex ) * width * height );

	// Setup pointers to images
	uchar *src_data = (uchar*) src->imageData;
	uchar *tpl_data = (uchar*) tpl->imageData;

	// Fill the structure that will be used by fftw
	for(i = 0; i < height; i++)
	{
		for(j = 0 ; j < width ; j++, k++)
		{
			src_spatial[k][0] = (double) src_data[i * step + j];
			src_spatial[k][1] =  0.0;

			tpl_spatial[k][0] = (double) tpl_data[i * step + j];
			tpl_spatial[k][1] =  0.0;
		}
	}
	
	// Hamming window to improve FFT (but slightly slower to compute)
	if(hamming == 1)
	{
		double omega = 2.0*M_PI/(fft_size-1);
		double A= 0.54;
		double B= 0.46;
		for(i=0,k=0;i<height;i++)
		{
			for(j=0;j<width;j++,k++)
			{
			    src_spatial[k][0]= (src_spatial[k][0])*(A-B*cos(omega*k));
			    tpl_spatial[k][0]= (tpl_spatial[k][0])*(A-B*cos(omega*k));
			}
		}
	}

	// Setup FFTW plans
	fftw_plan plan_src = fftw_plan_dft_2d(height, width, src_spatial, src_freq, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan plan_tpl = fftw_plan_dft_2d(height, width, tpl_spatial, tpl_freq, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan plan_res = fftw_plan_dft_2d(height, width, res_freq,  res_spatial,  FFTW_BACKWARD, FFTW_ESTIMATE);

	// Execute the FFT of the images
	fftw_execute(plan_src);
	fftw_execute(plan_tpl);

	// Compute the cross-correlation
	for(i = 0; i < fft_size ; i++ )
	{
		res_freq[i][0] = tpl_freq[i][0] * src_freq[i][0] + tpl_freq[i][1] * src_freq[i][1];
		res_freq[i][1] = tpl_freq[i][0] * src_freq[i][1] - tpl_freq[i][1] * src_freq[i][0];
		
		tmp = sqrt(pow(res_freq[i][0], 2.0) + pow(res_freq[i][1], 2.0));

		res_freq[i][0] /= tmp;
		res_freq[i][1] /= tmp;
	}

	// Get the phase correlation array = compute inverse fft
	fftw_execute(plan_res);

	// Find the peak
	struct Peak pk;	
	IplImage* peak_find = cvCreateImage(cvSize(tpl->width,tpl->height ), IPL_DEPTH_64F, 1);
	double  *peak_find_data = (double*) peak_find->imageData;
	
	for( i = 0 ; i < fft_size ; i++ )
	{
        peak_find_data[i] = res_spatial[i][0] / (double) fft_size;
    }
    
    CvPoint minloc, maxloc;
    double  minval, maxval;
    
    cvMinMaxLoc(peak_find, &minval, &maxval, &minloc, &maxloc, 0);
    
	pk.pt = maxloc;
	pk.maxval = maxval;

	// Clear memory
	fftw_destroy_plan(plan_src);
	fftw_destroy_plan(plan_tpl);
	fftw_destroy_plan(plan_res);
	fftw_free(src_spatial);
	fftw_free(tpl_spatial);
	fftw_free(src_freq);
	fftw_free(tpl_freq);
	fftw_free(res_spatial);
	fftw_free(res_freq);
	cvReleaseImage(&peak_find);
	
	fftw_cleanup_threads(); //Cleanup everything else related to FFTW
	
	return pk;
}


/*
--------------------------------------------------------------------------------------------------------------------
------- Main function ----------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------
*/

int main (int argc, char* argv[])
{
	//---------------------------
	//Initialization of variables
	//---------------------------
	
	//To handle the options
	const char* const short_options = "hl:r:D:f:d:s:x:y:o:cSFH";
	const struct option long_options[] = {
		{ "help",   0, NULL,  'h' },
		{ "left", 1, NULL,  'l' },
		{ "right", 1, NULL,  'r' },
		{ "diameter", 1, NULL,  'D' },
		{ "focal", 1, NULL,  'f' },
		{ "delta", 1, NULL,  'd' },
		{ "size", 1, NULL,  's' },
		{ "abscissa", 1, NULL,  'x' },
		{ "ordinate", 1, NULL,  'y' },
		{ "output", 1, NULL,  'o' },
		{ "choose",   0, NULL,  'c' },
		{ "sobel",   0, NULL,  'S' },
		{"fft", 0, NULL, 'F'},
		{"hamming", 0, NULL, 'H'},
		{ NULL,     0, NULL,  0   }
	};
	int next_option = 0;
	
	//Threads
	pthread_t thread1, thread2, thread3, thread4;
	struct data_find_common data1, data2, data3, data4;
	
	//To store the images
	IplImage* imgL = NULL;
	IplImage* imgR = NULL; 
	uchar *p1, *line1;
	
	//Max X and Y for the images (to avoid computing it many times)
	int X_maxL, Y_maxL, X_maxR, Y_maxR;
	
	//Path to images
	const char* src_pathL = NULL; //Const ?
	const char* src_pathR = NULL;
	
	//Matrices
	int array_size = 0; //Size of the matrices we'll create
	int **displacements; //To store the displacements for every subset
	float *mean_distance;
	
	//To store the parameters
	float D = 0, f = 0, deltaX = 0;
	int X_input = -1, Y_input = -1, c = 0, sobel = 0, fft = 0, hamming = 0;
	int CVLOAD = CV_LOAD_IMAGE_COLOR;
	FILE *output_file = stdout; //By default, the output will be stored in stdout
	const char* output = NULL;
	//The size of the subset, 50 pixels by default.
	int subset_size = 50;
	
	//theta = half angle of view of the cameras, alphas cf. article
	float theta;
	
	//To iterate and compute the mean distances
	int X_1, Y_1, X_2; 
	int j=0, key, char_temp;
	float min;
	CvMat tmp, tmp2;
	//Temporary structures to store what find_common returns
	struct minXY_struct temp1={255, 0, 0};
	struct minXY_struct temp2={255, 0, 0};
	struct minXY_struct temp3={255, 0, 0};
	struct minXY_struct temp4={255, 0, 0};
	
	//Images for sobel
	IplImage* imgL_gray = NULL;
	IplImage* imgL_Sobelx = NULL;
	IplImage* imgL_Sobely = NULL;
	IplImage* imgL_Sobel = NULL;
	IplImage* imgR_gray = NULL;
	IplImage* imgR_Sobelx = NULL;
	IplImage* imgR_Sobely = NULL;
	IplImage* imgR_Sobel = NULL;

	//Image for FFT
	IplImage* template = NULL;
	
	//Time
	clock_t start=clock();

	//----------------------------------------------------------
	//Store the options and the parameters in corresponding vars
	//----------------------------------------------------------
	
	do {
		next_option = getopt_long(argc, argv, short_options, long_options, NULL);

		switch(next_option)
		{
			case 'h':
				print_help(stdout);
				return EXIT_SUCCESS;

			case 'l':
				src_pathL = optarg;
				break;
			
			case 'r':
				src_pathR = optarg;
				break;
				
			case 'D':
				D = atof(optarg);
				break;
				
			case 'f':
				f = atof(optarg);
				break;
			
			case 'd':
				deltaX = atof(optarg);
				break;
				
			case 's':
				subset_size = atoi(optarg);
				break;
				
			case 'x':
				X_input = atoi(optarg);
				break;
				
			case 'y':
				Y_input = atoi(optarg);
				break;
				
			case 'o':
				output = optarg;
				break;
			
			case 'c':
				c = 1;
				break;
				
			case 'S':
				sobel = 1;
				break;
				
			case 'F':
				fft = 1;
				CVLOAD = CV_LOAD_IMAGE_GRAYSCALE;
				break;
				
			case 'H':
				hamming = 1;
				break;

			case -1:  // End of arguments list
				break;

			default:  // Unexpected behavior
				return EXIT_FAILURE;
		}

		} while(next_option != -1);


	//If not enough arguments -> error + help message
	if(src_pathL == NULL || src_pathR == NULL || D == 0 || f == 0 || deltaX == 0)
	{
		fprintf(stderr, "Not enough arguments. You must provide the paths to the two images, the diameter, the focal length and the distance between the two cameras. Please refer to the help message below for more details.\n");
		fprintf(stderr, "======================================\n");
		fprintf(stderr, "Help :\n");
		print_help(stderr);
		return EXIT_FAILURE;
	}
	
	//If the output must be written in a file, open it
	if(output != NULL)
	{
		output_file = fopen(output, "w");
	}

	//------------------------------------------------
	//Load the images and define the needed parameters
	//------------------------------------------------

	//We load the images in color mode even if they are greyscale to avoid problems with images with different color modes for the two images
	if (!(imgL = cvLoadImage (src_pathL, CVLOAD)))
	{
		fprintf (stderr, "couldn't open image file: %s\n",src_pathL);
		return EXIT_FAILURE;
	}

	if (!(imgR = cvLoadImage (src_pathR, CVLOAD)))
	{
		fprintf (stderr, "couldn't open image file: %s\n", src_pathR);
		return EXIT_FAILURE;
	}

	//Check that the images have 3 channels and 8 bits depth (to store values in char)
	if(CVLOAD == CV_LOAD_IMAGE_COLOR)
	{
		assert (imgL->depth == IPL_DEPTH_8U && imgL->nChannels == 3);
		assert (imgR->depth == IPL_DEPTH_8U && imgR->nChannels == 3);
	}
	else
	{
		assert (imgL->depth == IPL_DEPTH_8U && imgL->nChannels == 1);
		assert (imgR->depth == IPL_DEPTH_8U && imgR->nChannels == 1);
	}

	fprintf(output_file, "Now working with %d*%d subset. \n", subset_size, subset_size);
	
	theta = atan(D/(2*f));
	
	//Last coordinates we can study (due to the size of the subset)
	X_maxL = imgL->width - subset_size;
	Y_maxL = imgL->imageSize/imgL->widthStep - subset_size;
	X_maxR = imgR->width - subset_size;
	Y_maxR = imgR->imageSize/imgR->widthStep - subset_size;
	
	//-------------------------------------------------------------------------------------------
	//Test that there's something (interesting) to do, ie that subset_size < imgSize / 2 and same for Y
	//-------------------------------------------------------------------------------------------
	if(floor(imgL->width/2) < subset_size || floor((imgL->imageSize / imgL->widthStep) / 2) < subset_size)
	{
		fprintf(stderr, "Error : subset is greater than the half of the image. Please choose a smaller subset size.\n");
		
		cvReleaseImage(&imgL);
		cvReleaseImage(&imgR);
		
		return EXIT_FAILURE;
	}
	
	//---------------------------------------------------------
	//Get the coordinates of the pixels if argument "-c" passed
	//---------------------------------------------------------
	
	while(c != 0)
	{
		int *input[2] = {&X_input, &Y_input};
		
		cvNamedWindow("Select the pixel to work with", 1); //Create a window, display the image and add the mouse handler
		cvSetMouseCallback("Select the pixel to work with", mouseHandler, (void*) &input);
		cvShowImage("Select the pixel to work with", imgL);
		cvWaitKey(0);
		
		printf("Now working with the pixel : (%d,%d).\n", X_input, Y_input);
		
		printf("Is it ok ? [y/N] \n");
		
		do
		{
			key = fgetc(stdin);
			
			while (char_temp != '\n' && char_temp != EOF)
			{
				char_temp = getchar();
			}
		}while(key != 89 && key != 121 && key != 110 && key != 78 && key != 10);
		
		if(key == 89 || key == 121 || key == 10)
		{
			c = 0;
		}
	}
	
	//-----------------------------------------------
	//Applicate an edge detection algorithm if needed
	//-----------------------------------------------
	
	if(sobel != 0 && X_input < 0 && Y_input < 0)
	{
		//First, apply a gaussian blur
		cvSmooth(imgL, imgL, CV_GAUSSIAN, 3, 3, 0, 0);
		cvSmooth(imgR, imgR, CV_GAUSSIAN, 3, 3, 0, 0);

		//Convert images to greyscale
		imgL_gray = cvCreateImage(cvGetSize(imgL),imgL->depth,1);
		imgL_Sobelx = cvCreateImage(cvGetSize(imgL_gray),imgL_gray->depth,1);
		imgL_Sobely = cvCreateImage(cvGetSize(imgL_gray),imgL_gray->depth,1);
		imgL_Sobel = cvCreateImage(cvGetSize(imgL_gray),imgL_gray->depth,1);
		imgR_gray = cvCreateImage(cvGetSize(imgL),imgL->depth,1);
		imgR_Sobelx = cvCreateImage(cvGetSize(imgL_gray),imgL_gray->depth,1);
		imgR_Sobely = cvCreateImage(cvGetSize(imgL_gray),imgL_gray->depth,1);
		imgR_Sobel = cvCreateImage(cvGetSize(imgL_gray),imgL_gray->depth,1);
		
		cvCvtColor(imgL, imgL_gray, CV_RGB2GRAY);
		cvCvtColor(imgR, imgR_gray, CV_RGB2GRAY);
			
		//Gradient X
		cvSobel(imgL_gray, imgL_Sobelx, 1, 0, 3);
		cvSobel(imgR_gray, imgR_Sobelx, 1, 0, 3);
		
		//Gradient Y
		cvSobel(imgL_gray, imgL_Sobely, 0, 1, 3);
		cvSobel(imgR_gray, imgR_Sobely, 0, 1, 3);
		
		//Add the two images
		cvAdd(imgL_Sobelx, imgL_Sobely, imgL_Sobel, NULL);
		cvAdd(imgR_Sobelx, imgR_Sobely, imgR_Sobel, NULL);
		
		//Show the images
		printf("The images after edge detection will be shown in a new window.\n");
		printf("Press ENTER to continue");
		
		while(getchar() != '\n'){}
		cvStartWindowThread();

		cvNamedWindow("Edge detection of the left image", CV_WINDOW_AUTOSIZE );
		cvShowImage("Edge detection of the left image", imgL_Sobel);
		cvWaitKey(0);
		cvDestroyWindow("Edge detection of the left image");
		
		cvNamedWindow("Edge detection of the right image", CV_WINDOW_AUTOSIZE );
		cvShowImage("Edge detection of the right image", imgR_Sobel);
		cvWaitKey(0);
		cvDestroyWindow("Edge detection of the right image");
		
		cvReleaseImage(&imgL_Sobelx);
		cvReleaseImage(&imgL_Sobely);
		cvReleaseImage(&imgR_Sobelx);
		cvReleaseImage(&imgR_Sobely);
		
		assert (imgL_Sobel->depth == IPL_DEPTH_8U && imgL_Sobel->nChannels == 1);
		assert (imgR_Sobel->depth == IPL_DEPTH_8U && imgR_Sobel->nChannels == 1);
	}
	
	//-------------------------------------------------------------------------------------------
	//Definition of 2 vectors X and Y to store the coordinates of the displacement between images
	//-------------------------------------------------------------------------------------------
	if(X_input >= 0 && Y_input >= 0)
	{
		if(X_input <= X_maxL && Y_input <= Y_maxL)
		{
			fprintf(output_file, "Working with the %dx%d pixel\n", X_input, Y_input);
			array_size = 1; //If we enter a specific pixel, matrices become float
		}
		else //If x or y is too large, inform the user
		{
			fprintf(stderr, "Error : x or y is out of the image. Please note that x and y must be such as x<%d and y <%d\n", X_maxL, Y_maxL);
			
			cvReleaseImage(&imgL);
			cvReleaseImage(&imgR);
			
			return EXIT_FAILURE;
		}
	}
	else
	{
		array_size = floor(imgL->imageSize / imgL->widthStep / subset_size) * floor(imgL->width / subset_size);
	}
		
	//Allocate memory for all the matrices (and initialize them)
	if((displacements = malloc(sizeof(*displacements) * 2)) == NULL)
	{
		perror("malloc:");
		
		cvReleaseImage(&imgL);
		cvReleaseImage(&imgR);
		
		if(sobel != 0)
		{
			cvReleaseImage(&imgL_Sobel);
			cvReleaseImage(&imgR_Sobel);
		}
		
		return EXIT_FAILURE;
	}
	if((displacements[0] = malloc(sizeof(**displacements) * array_size)) == NULL)
	{
		perror("malloc:");
		
		cvReleaseImage(&imgL);
		cvReleaseImage(&imgR);
		
		if(sobel != 0)
		{
			cvReleaseImage(&imgL_Sobel);
			cvReleaseImage(&imgR_Sobel);
		}
		
		return EXIT_FAILURE;
	}
	if((displacements[1] = malloc(sizeof(**displacements) * array_size)) == NULL)
	{
		perror("malloc:");
		
		cvReleaseImage(&imgL);
		cvReleaseImage(&imgR);
		
		if(sobel != 0)
		{
			cvReleaseImage(&imgL_Sobel);
			cvReleaseImage(&imgR_Sobel);
		}
		
		return EXIT_FAILURE;
	}
	
	if((mean_distance = malloc(sizeof(*mean_distance) * array_size)) == NULL)
	{
		perror("malloc:");
		
		cvReleaseImage(&imgL);
		cvReleaseImage(&imgR);
		free(displacements);
		
		if(sobel != 0)
		{
			cvReleaseImage(&imgL_Sobel);
			cvReleaseImage(&imgR_Sobel);
		}
		
		return EXIT_FAILURE;
	}

	for(j = 0; j < array_size; j++)
	{
		displacements[0][j] = 0;
		displacements[1][j] = 0;
		mean_distance[j] = 0;
	}

	//--------------------------------------------------------------------
	//Call find_common to fill the matrices and compute the mean distances
	//--------------------------------------------------------------------
	fprintf(output_file, "Output is : (X-axis, Y-axis) : (Displacement on X-axis, Displacement on Y-axis) -> Mean Distance\n\n");
	//If we work with a specific pixel
	if(X_input >= 0 && Y_input >= 0 && X_input <= X_maxL && Y_input <= Y_maxL)
	{
		//Get the pointer to the pixel
		p1 = (uchar*) imgL->imageData + Y_input*imgL->widthStep + X_input*imgL->nChannels;
		
		if(fft == 1) //If using FFT
		{
			template = cvCreateImage(cvSize(imgR->width, imgR->height), IPL_DEPTH_8U, 1);
			
			cvZero(template);
			cvGetSubRect(template,&tmp,cvRect(0,0,subset_size,subset_size)); 
			cvGetSubRect(imgL,&tmp2,cvRect(X_input,Y_input,subset_size,subset_size)); 
		  	cvCopy(&tmp2,&tmp,NULL);
		  	
			struct Peak pk = Find_FFT(imgR, template, hamming);
			
			X_2 = pk.pt.x;
			displacements[0][0] = pk.pt.x - X_input; //(= DeltaX)
			displacements[1][0] = pk.pt.y - Y_input; //(=DeltaY)
			
			cvReleaseImage(&template);
		}
		else
		{		
			//Compute the displacement
			// 1 2
			// 3 4
			//4 threads
		
			data1.img2 = imgR;
			data1.p1 = p1;
			data1.widthStep1 = imgL->widthStep;
			data1.X_1 = X_input;
			data1.Y_1 = Y_input;
			data1.subset_size = subset_size;
			data1.X_start = 0;
			data1.Y_start = maximum(0, Y_input - (int) floor(imgR->imageSize/8/imgR->widthStep));
			data1.X_end = floor(X_maxR/2);
			data1.Y_end = Y_input;
			data1.temp = &temp1;
			pthread_create (&thread1, NULL, find_common, &data1);
		
			data2.img2 = imgR;
			data2.p1 = p1;
			data2.widthStep1 = imgL->widthStep;
			data2.X_1 = X_input;
			data2.Y_1 = Y_input;
			data2.subset_size = subset_size;
			data2.X_start = (int) floor(X_maxR/2) + 1;
			data2.Y_start = maximum(0, Y_input - (int) floor(imgR->imageSize/8/imgR->widthStep));
			data2.X_end = X_maxR;
			data2.Y_end = Y_input;
			data2.temp = &temp2;
			pthread_create (&thread2, NULL, find_common, &data2);
		
			data3.img2 = imgR;
			data3.p1 = p1;
			data3.widthStep1 = imgL->widthStep;
			data3.X_1 = X_input;
			data3.Y_1 = Y_input;
			data3.subset_size = subset_size;
			data3.X_start = 0;
			data3.Y_start = Y_input + 1;
			data3.X_end = floor(X_maxR/2);
			data3.Y_end = minimum(Y_maxR, Y_input + (int) floor(imgR->imageSize/8/imgR->widthStep));
			data3.temp = &temp3;
			pthread_create (&thread3, NULL, find_common, &data3);
		
			data4.img2 = imgR;
			data4.p1 = p1;
			data4.widthStep1 = imgL->widthStep;
			data4.X_1 = X_input;
			data4.Y_1 = Y_input;
			data4.subset_size = subset_size;
			data4.X_start = floor(X_maxR/2) + 1;
			data4.Y_start = Y_input + 1;
			data4.X_end = X_maxR;
			data4.Y_end = minimum(Y_maxR, Y_input + (int) floor(imgR->imageSize/8/imgR->widthStep));
			data4.temp = &temp4;
			pthread_create (&thread4, NULL, find_common, &data4);
		
			//Wait until threads finish
			pthread_join(thread1, NULL);
		
			//Compute all the results together
			min = temp1.min;
			displacements[0][0] = temp1.X - X_input; //(= DeltaX)
			displacements[1][0] = temp1.Y - Y_input; //(=DeltaY)
			X_2 = temp1.X;
		
			pthread_join(thread2, NULL);
			if(temp2.min < min)
			{
				min = temp2.min;
				displacements[0][0] = temp2.X - X_input; //(= DeltaX)
				displacements[1][0] = temp2.Y - Y_input; //(=DeltaY)
				X_2 = temp2.X;
			}
		
			pthread_join(thread3, NULL);
			if(temp3.min < min)
			{
				min = temp3.min;
				displacements[0][0] = temp3.X - X_input; //(= DeltaX)
				displacements[1][0] = temp3.Y - Y_input; //(=DeltaY)
				X_2 = temp3.X;
			}
		
			pthread_join(thread4, NULL);
			if(temp4.min < min)
			{
				min = temp4.min;
				displacements[0][0] = temp4.X - X_input; //(= DeltaX)
				displacements[1][0] = temp4.Y - Y_input; //(=DeltaY)
				X_2 = temp4.X;
			}
		}

		//Compute mean distance
		mean_distance[0] = compute_mean_distance(X_input, X_2, imgL->width, imgR->width, theta, deltaX);
	
	
		if(mean_distance[0] < 0) //If there's an error (mean_distance < 0 is absurd)
		{
			fprintf(stderr, "An error occurred, negative mean_distance found. Dump :\nCoordinates = (%d, %d) ; Displacement = (%d, %d); Computed value of mean distance : %f\n", X_input, Y_input, displacements[0][0], displacements[1][0], mean_distance[0]);
			
			cvReleaseImage(&imgL);
			cvReleaseImage(&imgR);
			free(displacements);
			free(mean_distance);
			
			if(output != NULL)
				fclose(output_file);
			
			return EXIT_FAILURE;
		}
		else //Else, print the result
		{
			fprintf(output_file, "(%d, %d) : (%d, %d) -> %f\n", X_input, Y_input, displacements[0][0], displacements[1][0], mean_distance[0]);
		}
	}
	//If we work with the entire image
	else
	{
		j = 0; //We use j to go through the matrices
		
		//Explore the image line by line
		for (line1 = (uchar*) imgL->imageData;
			line1 <= (uchar*) imgL->imageData + Y_maxL*imgL->widthStep;
			line1 += imgL->widthStep*subset_size)
		{
			for (p1 = line1; p1 <= line1 + X_maxL*imgL->nChannels; p1 += imgL->nChannels*subset_size)
			{
				//Get the coordinates corresponding to p1
				//(p1 = imageData + X_1 + iChannel + Y_1*widthStep) where iChannel is in {0,1,..nChannels}
				X_1 = floor(((p1 - (uchar*) imgL->imageData) % imgL->widthStep) / imgL->nChannels);
				Y_1 = floor((p1 - (uchar*) imgL->imageData ) / imgL->widthStep);
		
				if(fft == 1) //If using FFT
				{
					template = cvCreateImage(cvSize(imgR->width, imgR->height), IPL_DEPTH_8U, 1);
			
					cvZero(template);
					cvGetSubRect(template,&tmp,cvRect(0,0,subset_size,subset_size)); 
					cvGetSubRect(imgL,&tmp2,cvRect(X_1,Y_1,subset_size,subset_size)); 
				  	cvCopy(&tmp2,&tmp,NULL);
				  	
					struct Peak pk = Find_FFT(imgR, template, hamming);
			
					X_2 = pk.pt.x;
					displacements[0][0] = pk.pt.x - X_1; //(= DeltaX)
					displacements[1][0] = pk.pt.y - Y_1; //(=DeltaY)

					cvReleaseImage(&template);
				}
				else
				{		
					//Compute the displacement
					// 1 2
					// 3 4
					//4 threads
		
					data1.img2 = imgR;
					data1.p1 = p1;
					data1.widthStep1 = imgL->widthStep;
					data1.X_1 = X_1;
					data1.Y_1 = Y_1;
					data1.subset_size = subset_size;
					data1.X_start = 0;
					data1.Y_start = maximum(0, Y_1 - (int) floor(imgR->imageSize/8/imgR->widthStep));
					data1.X_end = floor(X_maxR/2);
					data1.Y_end = Y_1;
					data1.temp = &temp1;
					pthread_create (&thread1, NULL, find_common, &data1);
		
					data2.img2 = imgR;
					data2.p1 = p1;
					data2.widthStep1 = imgL->widthStep;
					data2.X_1 = X_1;
					data2.Y_1 = Y_1;
					data2.subset_size = subset_size;
					data2.X_start = (int) floor(X_maxR/2) + 1;
					data2.Y_start = maximum(0, Y_1 - (int) floor(imgR->imageSize/8/imgR->widthStep));
					data2.X_end = X_maxR;
					data2.Y_end = Y_1;
					data2.temp = &temp2;
					pthread_create (&thread2, NULL, find_common, &data2);
		
					data3.img2 = imgR;
					data3.p1 = p1;
					data3.widthStep1 = imgL->widthStep;
					data3.X_1 = X_1;
					data3.Y_1 = Y_1;
					data3.subset_size = subset_size;
					data3.X_start = 0;
					data3.Y_start = Y_1 + 1;
					data3.X_end = floor(X_maxR/2);
					data3.Y_end = minimum(Y_maxR, Y_1 + (int) floor(imgR->imageSize/8/imgR->widthStep));
					data3.temp = &temp3;
					pthread_create (&thread3, NULL, find_common, &data3);
		
					data4.img2 = imgR;
					data4.p1 = p1;
					data4.widthStep1 = imgL->widthStep;
					data4.X_1 = X_1;
					data4.Y_1 = Y_1;
					data4.subset_size = subset_size;
					data4.X_start = floor(X_maxR/2) + 1;
					data4.Y_start = Y_1 + 1;
					data4.X_end = X_maxR;
					data4.Y_end = minimum(Y_maxR, Y_1 + (int) floor(imgR->imageSize/8/imgR->widthStep));
					data4.temp = &temp4;
					pthread_create (&thread4, NULL, find_common, &data4);
		
					//Wait until threads finish
					pthread_join(thread1, NULL);
		
					//Compute all the results together
					min = temp1.min;
					displacements[0][j] = temp1.X - X_1; //(= DeltaX)
					displacements[1][j] = temp1.Y - Y_1; //(=DeltaY)
					X_2 = temp1.X;
		
					pthread_join(thread2, NULL);
					if(temp2.min < min)
					{
						min = temp2.min;
						displacements[0][j] = temp2.X - X_1; //(= DeltaX)
						displacements[1][j] = temp2.Y - Y_1; //(=DeltaY)
						X_2 = temp2.X;
					}
		
					pthread_join(thread3, NULL);
					if(temp3.min < min)
					{
						min = temp3.min;
						displacements[0][j] = temp3.X - X_1; //(= DeltaX)
						displacements[1][j] = temp3.Y - Y_1; //(=DeltaY)
						X_2 = temp3.X;
					}
		
					pthread_join(thread4, NULL);
					if(temp4.min < min)
					{
						min = temp4.min;
						displacements[0][j] = temp4.X - X_1; //(= DeltaX)
						displacements[1][j] = temp4.Y - Y_1; //(=DeltaY)
						X_2 = temp4.X;
					}
				}

				//Compute mean distance
				mean_distance[j] = compute_mean_distance(X_1, X_2, imgL->width, imgR->width, theta, deltaX);
	
	
				if(mean_distance[j] < 0) //If there's an error (mean_distance < 0 is absurd)
				{
					fprintf(stderr, "An error occurred, negative mean_distance found. Dump :\nCoordinates = (%d, %d) ; Displacement = (%d, %d); Computed value of mean distance : %f\n", X_1, Y_1, displacements[0][j], displacements[1][j], mean_distance[j]);
					fprintf(output_file, "Error, absurd mean distance : (%d, %d) : (%d, %d) -> %f\n", X_1, Y_1, displacements[0][j], displacements[1][j], mean_distance[j]);
				}
				else //Else, print the result
				{
					fprintf(output_file, "(%d, %d) : (%d, %d) -> %f\n", X_1, Y_1, displacements[0][j], displacements[1][j], mean_distance[j]);
				}
			
				j++;
			}
		}
	}
	

	//-----------
	//Free memory
	//-----------
	if(sobel != 0)
	{
		cvReleaseImage(&imgL_Sobel);
		cvReleaseImage(&imgR_Sobel);
	}
	cvReleaseImage(&imgL);
	cvReleaseImage(&imgR);
	free(displacements);
	free(mean_distance);
	
	clock_t end=clock();

	fprintf(output_file, "\nCalcul des distances terminé en %d microsecondes.\n", (int) (end - start));
	
	if(output != NULL)
	{
		fprintf(stdout, "\nCalcul des distances terminé en %d microsecondes.\n", (int) (end - start));
		fclose(output_file);
	}
		

	return EXIT_SUCCESS; //And exit ;)
}

//Almost there

//Just some comments to make it ...

//What ? You'll see soon...




// Yeah ! Here we go ! 1337 lines of code !
