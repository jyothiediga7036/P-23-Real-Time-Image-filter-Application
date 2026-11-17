#ifndef GRAYSCALE_H
#define GRAYSCALE_H

#include <cuda_runtime.h>

launchGrayKernel(unsigned char *d_input,unsigned char *d_output,int width,int height, dim3 grid, dim3 block);

#endif
