#include "grayscale.h"

__global__ void gray_kernel(unsigned char *d_input,unsigned char *d_output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int index = (y * width + x) * 3;
        int gray_index = y * width + x;

        // OpenCV frame is BGR: B=0.114, G=0.587, R=0.299.
        d_output[gray_index] =0.114f * d_input[index] +0.587f * d_input[index + 1] +0.299f * d_input[index + 2];
    }
}

void launchGrayKernel(unsigned char *d_input,unsigned char *d_output,int width,int height, dim3 grid, dim3 block)
{
    gray_kernel<<<grid, block>>>(d_input, d_output, width, height);
}
