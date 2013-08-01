Stereoscopy
===========

A program (in C) to find the distance to an object using two images and computing distances accross these images (thanks to a stereoscopy algorithm).

## Usage
This program computes the mean distance between a stereoscopic camera and an object. We assume both cameras have the same specs (which are the diameter of the field stop of the cameras and the focal length).

The idea behind this software is very simple. First, it tries to determine the distance in pixels between the same element through the two photos (either by using a FFT-based approach or by comparing patterns from the two pictures). Once it gets the distance in pixels through the two pictures, it can easily convert it to a real distance between the camera and the object, using the specifications of the camera.

For more details about the way I convert a distance in pixels to a real (perpendicular) distance in centimeter, please refer to the following article (I just used the formulas in this paper without major modifications) : http://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1064&context=ecetr

This software was tested successfully (_ie_ giving a result correct according to the incertitude on the camera specs) with a standard compact camera, taking two pictures separated by a few centimeters (to mimmic our eyes).

### Notes : 

* I use the OpenCV library to easily load and go through the images. OpenCV is a C library for computer vision licensed under a BSD License ( http://opensource.org/licenses/bsd-license.php )
* The software computes both X and Y displacements but will focus on X displacements to determine the mean distance of the object as the cameras are supposed to be on the same horizontal plane (Y displacements are vertical displacements and should be negligible). It is possible to take into account a vertical displacement with minor modification to the code (you should just find a correct formula in 3D :)
* X corresponds to the abscissa and Y to the ordinate of the pixel in the image. They both goes from 0 to the size of your image.
* It has been successfully tested on a **GNU/Linux** system. I think it could work on Windows with minor or no modifications but haven't try.

Pixels are numeroted this way :
<pre>
(0,0)		(1,0)		(2,0)		...		(Width,0)
(1,0)		(1,1)		(2,1)		...		(Width,1)
 ...    	  ...		  ...		 ...		   ...
(0,Height)    ...		  ...		 ... 	 (Width, Height)
</pre>

## Compiling

You must have opencv libs installed on your system. 

## Command line arguments

<dl>
<dt>-l (--left)</dt><dd>Path to the left image</dd>
<dt>-r (--right)</dt><dd>Path to the right image</dd>
<dt>-D (--diameter)</dt><dd>(float) Diameter of the camera's field stop</dd>
<dt>-f (--focal)</dt><dd>(float) Focal length of the camera</dd>
<dt>-d (--delta)</dt><dd>(float) Distance between the two cameras</dd>
<dt>-s (--size)</dt><dd>[Optional] (int) Size of the subsets used, default is 50</dd>
<dt>-x (--abscissa)</dt><dd>[Optional] (int) Abscissa of the pixel to compute the mean distance</dd>
<dt>-y (--ordinate)</dt><dd>[Optional] (int) Ordinate of the pixel to compute the mean distance</dd>
<dt>-c (--choose)</dt><dd>[Optional] To display the image and select which pixel you want to work with (same as -x and -y but useful if you don't know the coordinates of the pixel)</dd>
<dt>-o (--output)</dt><dd>[Optional] Path to a file to store the output</dd>
<dt>-S (--sobel)</dt><dd>[Optional] Run a Sobel edges detection algorithm on both left and right images first to work only on edges. This option won't have any effect if working in single pixel mode.</dd>
<dt>-F (--fft)</dt><dd>[Optional] Use cross-correlation and fft based method (faster but a little bit less accurate).</dd>
<dt>-H (--hamming)</dt><dd>[Optional] Use an hamming window to improve FFT (but slightly slower).</dd>
<dt>-h (--help)</dt><dd>Display a help message</dd>
</dl>

## License
Please, send me an email if you use or modify this program, just to let me know if this program is useful to anybody or how did you improve it :) You can also send me an email to tell me how lame it is ! :)

### TLDR; 
I don't give a damn to anything you can do using this code. It would just be nice to
quote where the original code comes from.


--------------------------------------------------------------------------------
"THE NO-ALCOHOL BEER-WARE LICENSE" (Revision 42) :

    Phyks (phyks@phyks.me) wrote this file. As long as you retain this notice you
    can do whatever you want with this stuff (and you can also do whatever you want
    with this stuff without retaining it, but that's not cool...). If we meet some 
    day, and you think this stuff is worth it, you can buy me a <del>beer</del> soda 
    in return.
                                                                     Phyks
---------------------------------------------------------------------------------
