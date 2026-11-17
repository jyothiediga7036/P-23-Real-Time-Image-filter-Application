#ifndef INVERTION_H
#define INVERTION_H

#include <cuda_runtime.h>

launchInvertKernel(unsigned char *d_input, int width,int height, dim3 grid, dim3 block);

#endif
