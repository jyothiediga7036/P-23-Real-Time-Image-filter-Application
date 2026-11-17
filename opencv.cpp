#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

#include "grayscale.h"
#include "boxblur.h"
#include "invertion.h"

using namespace std;

int main()
{
    //string pipeline ="v4l2src device=/dev/video0 ! ""video/x-raw,width=640,height=480,framerate=30/1 ! ""videoconvert ! appsink";

    //cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    // Alternative webcam opening method:
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cout << "Camera not opened" << endl;
        return -1;
    }

    cout << "Camera opened successfully" << endl;
    cout << "Press g = Grayscale, i = Invert, b = Box Blur, o = Original, q/ESC = Exit" << endl;

    cv::Mat frame, gray, invert, box;
    char key = 'o';

    while (true)
    {
        if (!cap.read(frame))
        {
            cout << "Frames are not available" << endl;
            break;
        }

        int width = frame.cols;
        int height = frame.rows;

        // This project uses 3-channel BGR frames.
        if (frame.type() != CV_8UC3)
        {
            cout << "Expected a 8-bit, 3-channel BGR frame" << endl;
            break;
        }

        size_t size = frame.rows * frame.cols * frame.elemSize();
        size_t gray_size = frame.rows * frame.cols;

        unsigned char *h_input = frame.data;
        unsigned char *d_input = nullptr;
        unsigned char *d_output = nullptr;
        uchar3 *d_blur_output = nullptr;

        cudaError_t err;

        err = cudaMalloc((void **)&d_input, size);
        if (err != cudaSuccess)
        {
            cout << "Input memory allocation failure: "<< cudaGetErrorString(err) << endl;
            return-1;
        }

        err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            cout << "Input memory copy failure: "<< cudaGetErrorString(err) << endl;
            cudaFree(d_input);
            return-1;
        }

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,(height + block.y - 1) / block.y);

        // Show the original frame by default, or when 'o' is pressed.
        cv::imshow("Original Camera", frame);

        if (key == 'g')
        {
            err = cudaMalloc((void **)&d_output, gray_size);
            if (err != cudaSuccess)
            {
                cout << "Gray output allocation failure: "<< cudaGetErrorString(err) << endl;
                cudaFree(d_input);
                break;
            }

            gray.create(height, width, CV_8UC1);
            launchGrayKernel(d_input, d_output, width, height, grid, block);

            err = cudaGetLastError();
            if (err == cudaSuccess)
                err = cudaDeviceSynchronize();

            if (err == cudaSuccess)
                err = cudaMemcpy(gray.data, d_output, gray_size,cudaMemcpyDeviceToHost);

            if (err != cudaSuccess)
                cout << "Grayscale processing failed: "<< cudaGetErrorString(err) << endl;
            else
                cv::imshow("Grayscale", gray);
        }
        else if (key == 'i')
        {
            invert.create(height, width, CV_8UC3);
            launchInvertKernel(d_input, width, height, grid, block);

            err = cudaGetLastError();
            if (err == cudaSuccess)
                err = cudaDeviceSynchronize();

            if (err == cudaSuccess)
                err = cudaMemcpy(invert.data, d_input, size,cudaMemcpyDeviceToHost);

            if (err != cudaSuccess)
                cout << "Invert processing failed: "<< cudaGetErrorString(err) << endl;
            else
                cv::imshow("Color Inversion", invert);
        }
        else if (key == 'b')
        {
            err = cudaMalloc((void **)&d_blur_output, size);
            if (err != cudaSuccess)
            {
                cout << "Blur output allocation failure: "<< cudaGetErrorString(err) << endl;
                cudaFree(d_input);
                break;
            }

            box.create(height, width, CV_8UC3);
            launchBoxBlur(reinterpret_cast<uchar3 *>(d_input),d_blur_output, width, height, grid, block);

            err = cudaGetLastError();
            if (err == cudaSuccess)
                err = cudaDeviceSynchronize();

            if (err == cudaSuccess)
                err = cudaMemcpy(box.data, d_blur_output, size,cudaMemcpyDeviceToHost);

            if (err != cudaSuccess)
                cout << "Box blur processing failed: "<< cudaGetErrorString(err) << endl;
            else
                cv::imshow("CUDA Box Blur", box);
        }

        cudaFree(d_blur_output);
        cudaFree(d_output);
        cudaFree(d_input);

        // waitKey both processes window events and reads the key.
        int pressed = cv::waitKey(1);
        if (pressed == 'q' || pressed == 27)
            break;
        else if (pressed == 'g' || pressed == 'i' ||
                 pressed == 'b' || pressed == 'o')
            key = static_cast<char>(pressed);
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
