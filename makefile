all: stereo bmp

stereo : stereo.o
	gcc stereo.o -o stereo `pkg-config opencv --libs` -lm -lpthread -lfftw3 -lfftw3_threads

stereo.o : stereo.c
	gcc -c stereo.c -Wall `pkg-config opencv --cflags` -lm -lpthread -lfftw3 -lfftw3_threads
	
bmp : bmp.c
	gcc -o bmp bmp.c
