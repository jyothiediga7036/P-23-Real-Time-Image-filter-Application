#include "boxblur.h"

__global__ void boxBlur(uchar3 *input,uchar3 *output,int width,int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int sumB = 0;
    int sumG = 0;
    int sumR = 0;
    int count = 0;

    for (int dy = -3; dy <= 3; dy++)
    {
        for (int dx = -3; dx <= 3; dx++)
        {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                uchar3 pixel = input[ny * width + nx];
                sumB += pixel.x;
                sumG += pixel.y;
                sumR += pixel.z;
                count++;
            }
        }
    }

    uchar3 result;
    result.x = sumB / count;
    result.y = sumG / count;
    result.z = sumR / count;

    output[y * width + x] = result;
}

void launchBoxBlur(uchar3 *input,uchar3 *output,int width,int height, dim3 grid,dim3 block)
{
    boxBlur<<<grid, block>>>(input, output, width, height);
}
