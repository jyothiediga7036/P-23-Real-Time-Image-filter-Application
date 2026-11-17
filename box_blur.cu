#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

// CUDA Box Blur Kernel
__global__ void box_blur(unsigned char* input,unsigned char* output,int width,int height)
{
    int b_sum, g_sum, r_sum, count;

    // Calculate pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        b_sum = 0;
        g_sum = 0;
        r_sum = 0;
        count = 0;

        int dx, dy, nx, ny;

        // Read 3x3 neighbourhood
        for (dy = -3; dy <= 3; dy++)
        {
            for (dx = -3; dx <= 3; dx++)
            {
                nx = x + dx;
                ny = y + dy;

                // Check image boundaries
                if ((nx >= 0 && nx < width) && (ny >= 0 && ny < height))
                {
                    int index = (ny * width + nx)*3;
                    unsigned char b = input[index];
                    unsigned char g = input[index + 1];
                    unsigned char r = input[index + 2];

                    b_sum += b;
                    g_sum += g;
                    r_sum += r;
                    count++;
                }
            }
        }
        // Calculate output index
        int output_index = (y * width + x) * 3;

        // Calculate average
        output[output_index]     = b_sum / count;
        output[output_index + 1] = g_sum / count;
        output[output_index + 2] = r_sum / count;
    }
}
int main()
{
    // 1. Open USB camera
    VideoCapture cap(0, CAP_V4L2);

    if (!cap.isOpened())
    {
        cout << "Camera opening failed!" << endl;
        return -1;
    }
    cout << "Camera opened successfully!" << endl;

    Mat frame,box;

    // 2. Capture first frame
    if (!cap.read(frame))
    {
        cout << "Frame capture failed!" << endl;
        return -1;
    }
    int width = frame.cols;
    int height = frame.rows;

    size_t size = width * height * 3 * sizeof(unsigned char);

    cout << "Width  : " << width << endl;
    cout << "Height : " << height << endl;

    // 3. Create output image
    box.create(height, width, CV_8UC3);

    // 4. Allocate GPU memory
    unsigned char *d_input, *d_output;

    cudaError_t err;
    err = cudaMalloc((void**)&d_input, size);
    if (err != cudaSuccess)
    {
        cout << cudaGetErrorString(err) << endl;
        return -1;
    }

    err = cudaMalloc((void**)&d_output, size);

    if (err != cudaSuccess)
    {
        cout << cudaGetErrorString(err) << endl;
        cudaFree(d_input);
        return -1;
    }

    // 5. Configure CUDA blocks and threads
    dim3 block(16, 16);

    dim3 grid((width + block.x - 1) / block.x,(height + block.y - 1) / block.y);

    // 6. Live camera processing loop
    while (true)
    {
        // Capture live frame
        if (frame.empty())
            break;

        // CPU to GPU copy
        err = cudaMemcpy(d_input,frame.data,size,cudaMemcpyHostToDevice);

        if (err != cudaSuccess)
        {
            cout << cudaGetErrorString(err) << endl;
            break;
        }

        // Launch CUDA kernel
        box_blur<<<grid, block>>>(d_input,d_output,width,height);

        // Check kernel launch
        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            cout << cudaGetErrorString(err) << endl;
            break;
        }

        // Wait for GPU processing
        err = cudaDeviceSynchronize();

        if (err != cudaSuccess)
        {
            cout << cudaGetErrorString(err) << endl;
            break;
        }

        // GPU to CPU copy
        err = cudaMemcpy(box.data,d_output,size,cudaMemcpyDeviceToHost);

        if (err != cudaSuccess)
        {
            cout << cudaGetErrorString(err) << endl;
            break;
        }

        // Display original and blurred frames
        imshow("Original Camera",frame);
        imshow("CUDA Box Blur", box);

        // Exit when user presses q
        if (waitKey(1) == 'q')
            break;
            
    }

    // 7. Release resources
    cudaFree(d_input);
    cudaFree(d_output);

    cap.release();
    destroyAllWindows();

    cout << "Camera processing completed!" << endl;

    return 0;
}







/*#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;

__global__ void boxBlur(
    uchar3 *input,
    uchar3 *output,
    int width,
    int height)
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

            if (nx >= 0 && nx < width &&
                ny >= 0 && ny < height)
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

int main()
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cout << "Camera not opened" << endl;
        return -1;
    }

    cv::Mat frame;
    cv::Mat box;

    while (true)
    {
        if (!cap.read(frame))
        {
            cout << "Frames are not captured" << endl;
            break;
        }

        int width = frame.cols;
        int height = frame.rows;

        size_t size = width * height * sizeof(uchar3);

        // CPU input memory
        unsigned char *h_input = frame.data;

        // GPU memory
        uchar3 *d_input3;
        uchar3 *d_output3;

        cudaMalloc(&d_input3, size);
        cudaMalloc(&d_output3, size);

        // Copy camera frame from CPU to GPU
        cudaMemcpy(d_input3, h_input, size,
                   cudaMemcpyHostToDevice);

        dim3 block(16, 16);

        dim3 grid(
            (width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y
        );

        // Launch CUDA kernel
        boxBlur<<<grid, block>>>(
            d_input3, d_output3, width, height);

        cudaDeviceSynchronize();

        // Create output Mat
        box.create(height, width, CV_8UC3);

        // Copy processed frame from GPU to CPU
        cudaMemcpy(box.data, d_output3, size,
                   cudaMemcpyDeviceToHost);

        cv::imshow("Original Frame", frame);
        cv::imshow("CUDA Box Blur", box);

        cudaFree(d_input3);
        cudaFree(d_output3);

        if (cv::waitKey(1) == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}*/
