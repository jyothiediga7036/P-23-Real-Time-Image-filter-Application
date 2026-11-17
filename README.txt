CUDA + OpenCV live camera project

Files:
  opencv.cpp       Camera capture, key handling, CUDA memory transfers
  grayscale.cu     Grayscale CUDA kernel
  grayscale.h      Grayscale launch declaration
  boxblur.cpp      Box blur CUDA kernel and launch wrapper
  boxblur.h        Box blur launch declaration
  invertion.cpp    Color inversion CUDA kernel and launch wrapper
  invertion.h      Color inversion launch declaration

Keys while the camera window is active:
  g = grayscale
  i = color inversion
  b = 7x7 box blur (dx,dy from -3 through +3)
  o = original frame
  q or ESC = quit

Important build note:
The .cpp files boxblur.cpp and invertion.cpp contain CUDA kernel syntax.
Compile those files with NVCC using -x cu so NVCC treats them as CUDA source.

Example build command (adjust OpenCV paths/libraries for your Jetson installation):
  nvcc -x cu -c grayscale.cu -o grayscale.o
  nvcc -x cu -c boxblur.cpp -o boxblur.o
  nvcc -x cu -c invertion.cpp -o invertion.o
  g++ -c opencv.cpp $(pkg-config --cflags opencv4) -o opencv.o
  nvcc opencv.o grayscale.o boxblur.o invertion.o \
      $(pkg-config --libs opencv4) -o camera_cuda

If your OpenCV installation has GStreamer enabled, the GStreamer pipeline
in opencv.cpp can be used. The frame processing assumes CV_8UC3 BGR input
with tightly packed rows, matching the original indexing logic.


Standalone box-blur filter test:
  boxblur_test.cpp  Self-contained live-camera CUDA box-blur test.

The standalone test uses a 7x7 neighbourhood (dx and dy are -3 through +3).
Each CUDA thread calculates the average B, G and R values for one output pixel.
The test allocates GPU buffers once, captures a fresh camera frame each loop,
copies the frame to the GPU, runs the kernel, and copies the result back.

Build the standalone test on the Jetson:
  nvcc boxblur_test.cpp $(pkg-config --cflags --libs opencv4) -o boxblur_test

Run:
  ./boxblur_test

Use q or ESC to exit. The standalone test is separate from opencv.cpp;
do not link both files into the same executable because each contains main().
