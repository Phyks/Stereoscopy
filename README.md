Stereoscopy
===========

A program (in C) to find the distance to an object using two images and computing distances accross these images (thanks to a stereoscopy algorithm).


License
=======
Please, send me an email if you use or modify this program, just to let me know if this program is useful to anybody or how did you improve it :) You can also send me an email to tell me how lame it is ! :)

TLDR; I don't give a damn to anything you can do using this code. It would just be nice to
quote where the original code comes from.


* --------------------------------------------------------------------------------
* "THE NO-ALCOHOL BEER-WARE LICENSE" (Revision 42):
* Phyks (webmaster@phyks.me) wrote this file. As long as you retain this notice you
* can do whatever you want with this stuff (and you can also do whatever you want
* with this stuff without retaining it, but that's not cool...). If we meet some 
* day, and you think this stuff is worth it, you can buy me a <del>beer</del> soda 
* in return.
* 																	Phyks
* ---------------------------------------------------------------------------------

Usage
=====
This program computes the mean distance between a stereoscopic camera and an object. We assume both cameras have the same specs (which are the diameter of the camera's field stop and the focal length).

For more details about the algorithm, please refer to the following article : http://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1064&context=ecetr

The distance estimation is just an application of the algorithm described in the previous article. To compute the displacement vector between the two images, we study the distances between subsets of the images. This is just like ifwe had a mask that we could place over the left image and then, we could search the best matching part of the right image.

Notes : 
- We use the openCV library to easily load and go through the images. OpenCV is a C library for computer vision licensed under a BSD License (http://opensource.org/licenses/bsd-license.php)
- We compute both X and Y displacements but we'll focus on X displacements to determine the mean distance of the object as the cameras are supposed to be on the same horizontal plane (Y displacements = vertical displacements are negligible)
- X corresponds to the abscissa and Y to the ordinate of the pixel in the image. They both goes from 0 to ...

Pixels are numeroted this way :
(0,0)  (1,0)	(2,0)	...	(Width,0)
(1,0)	(1,1)	(2,1)	...	(Width,1)
 ...	 ...	 ...	...	  ...
(0,Height)	 ...		(Width, Height)


*	 -l (--left) 	 Path to the left image
*	 -r (--right) 	 Path to the right image
*	 -D (--diameter) (float) Diameter of the camera's field stop
*	 -f (--focal) 	 (float) Focal length of the camera
*	 -d (--delta) 	 (float) Distance between the two cameras
*	 -s (--size) 	 [Optional] (int) Size of the subsets used, default is 50
*	 -x (--abscissa) [Optional] (int) Abscissa of the pixel to compute the mean distance
*	 -y (--ordinate) [Optional] (int) Ordinate of the pixel to compute the mean distance
*	 -c (--choose) 	 [Optional] To display the image and select which pixel you want to work with (same as -x and -y but useful if you don't know the coordinates of the pixel)
*	 -o (--output) 	 [Optional] Path to a file to store the output
*	 -S (--sobel) 	 [Optional] Run a Sobel edges detection algorithm on both left and right images first to work only on edges. This option won't have any effect if working in single pixel mode.
*	 -F (--fft) 	 [Optional] Use cross-correlation and fft based method (faster but a little bit less accurate).
*	 -H (--hamming) 	 [Optional] Use an hamming window to improve FFT (but slightly slower).
*	 -h (--help) 	 Display this error message


