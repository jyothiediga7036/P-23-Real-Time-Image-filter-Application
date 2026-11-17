#ifndef BOXBLUR_H
#define BOXBLUR_H

#include <cuda_runtime.h>

void launchBoxBlur(uchar3 *input, uchar3 *output, int width, int height,dim3 grid,dim3 block);

#endif
