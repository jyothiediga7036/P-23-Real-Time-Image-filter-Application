#include "invertion.h"

__global__ void invert_kernel(unsigned char *d_input,int width,int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = (y * width + x) * 3;

        d_input[index]     = 255 - d_input[index];
        d_input[index + 1] = 255 - d_input[index + 1];
        d_input[index + 2] = 255 - d_input[index + 2];
    }
}

void launchInvertKernel(unsigned char *d_input, int width,int height, dim3 grid, dim3 block)
{
    invert_kernel<<<grid, block>>>(d_input, width, height);
}
