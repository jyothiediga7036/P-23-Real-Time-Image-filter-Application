#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

// Each CUDA thread processes one BGR pixel.
// dx/dy from -3 to +3 means a 7x7 box-blur neighbourhood.
__global__ void box_blur(const uchar3 *input,uchar3 *output,int width,int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int b_sum = 0;
    int g_sum = 0;
    int r_sum = 0;
    int count = 0;

    for (int dy = -3; dy <= 3; ++dy)
    {
        for (int dx = -3; dx <= 3; ++dx)
        {
            int nx = x + dx;
            int ny = y + dy;

            // Ignore neighbours outside the image.
            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                uchar3 pixel = input[ny * width + nx];
                b_sum += pixel.x;
                g_sum += pixel.y;
                r_sum += pixel.z;
                ++count;
            }
        }
    }

    uchar3 result;
    result.x = static_cast<unsigned char>(b_sum / count);
    result.y = static_cast<unsigned char>(g_sum / count);
    result.z = static_cast<unsigned char>(r_sum / count);

    output[y * width + x] = result;
}

static bool checkCuda(cudaError_t error, const char *operation)
{
    if (error == cudaSuccess)
        return true;

    cerr << operation << " failed: " << cudaGetErrorString(error) << endl;
    return false;
}

int main()
{
    // Open the USB camera through V4L2.
    VideoCapture cap(0, CAP_V4L2);

    if (!cap.isOpened())
    {
        cerr << "Camera opening failed!" << endl;
        return -1;
    }

    cout << "Camera opened successfully." << endl;
    cout << "Press q or ESC to quit." << endl;

    Mat frame, blurred;

    // Capture the first frame to get the image dimensions.
    if (!cap.read(frame) || frame.empty())
    {
        cerr << "Initial frame capture failed!" << endl;
        cap.release();
        return -1;
    }

    if (frame.type() != CV_8UC3)
    {
        cerr << "Expected an 8-bit, 3-channel BGR frame." << endl;
        cap.release();
        return -1;
    }

    const int width = frame.cols;
    const int height = frame.rows;
    const size_t size = static_cast<size_t>(width) * height * sizeof(uchar3);

    cout << "Width  : " << width << endl;
    cout << "Height : " << height << endl;
    cout << "Bytes  : " << size << endl;

    blurred.create(height, width, CV_8UC3);

    uchar3 *d_input = nullptr;
    uchar3 *d_output = nullptr;

    if (!checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_input), size),
                   "cudaMalloc input"))
    {
        cap.release();
        return -1;
    }

    if (!checkCuda(cudaMalloc(reinterpret_cast<void **>(&d_output), size),
                   "cudaMalloc output"))
    {
        cudaFree(d_input);
        cap.release();
        return -1;
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    bool running = true;

    while (running)
    {
        // The first frame was already captured; subsequent frames are read here.
        if (frame.empty())
            break;

        // Camera frames are normally continuous. Clone only if the Mat has row padding.
        if (!frame.isContinuous())
            frame = frame.clone();

        // CPU (OpenCV) -> GPU copy.
        cudaError_t err = cudaMemcpy(d_input, frame.data, size,
                                     cudaMemcpyHostToDevice);
        if (!checkCuda(err, "cudaMemcpy HostToDevice"))
            break;

        // Launch one thread per output pixel.
        box_blur<<<grid, block>>>(d_input, d_output, width, height);

        err = cudaGetLastError();
        if (!checkCuda(err, "box_blur kernel launch"))
            break;

        // Wait for the kernel and detect execution errors.
        err = cudaDeviceSynchronize();
        if (!checkCuda(err, "cudaDeviceSynchronize"))
            break;

        // GPU -> CPU copy.
        err = cudaMemcpy(blurred.data, d_output, size,
                         cudaMemcpyDeviceToHost);
        if (!checkCuda(err, "cudaMemcpy DeviceToHost"))
            break;

        imshow("Original Camera", frame);
        imshow("CUDA Box Blur - 7x7", blurred);

        int key = waitKey(1);
        if (key == 'q' || key == 27)
            running = false;
        else if (running && !cap.read(frame))
        {
            cerr << "Frame capture failed." << endl;
            break;
        }
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cap.release();
    destroyAllWindows();

    cout << "Box-blur filter test completed." << endl;
    return 0;
}
