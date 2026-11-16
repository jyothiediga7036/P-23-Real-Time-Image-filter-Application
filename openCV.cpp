#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;


int main()
{
    // 1. Create VideoCapture object
    string pipeline="v4l2src device=/dev/video0 ! video/x-raw,width=640, height=480, framerate=30/1 ! videoconvert ! appsink";
    cv::VideoCapture cap(pipeline,cv::CAP_GSTREAMER);
    //cv::VideoCapture cap(0);

    // 2. Check whether camera opened successfully
    if (!cap.isOpened())
    {
        cout<<"Camera not opened!"<<endl;
        return -1;
    }

    cout<<"Camera opened successfully!"<<endl;

    cv::Mat frame;
    cout<<"width : "<<cap.get(cv::CAP_PROP_FRAME_WIDTH)<<endl;
    cout<<"height : "<<cap.get(cv::CAP_PROP_FRAME_HEIGHT)<<endl;
    cout<<"fps : "<<cap.get(cv::CAP_PROP_FPS)<<endl;
    cout<<"frame type : "<<cap.get(cv::CAP_PROP_FORMAT)<<endl;
    // 3. Continuously capture frames
    while (true)
    {
        // Capture one frame from camera
        cap.read(frame); 
        // 4. Check whether frame is received
        if (frame.empty())
        {
            cout<<"Failed to capture frame!"<<endl;
            break;
        }

        // 5. Display the captured frame
        cv::imshow("Camera Output", frame);

       /* cout<<"width : "<<frame.cols<<endl;
        cout<<"height : "<<frame.rows<<endl;
        cout<<"channels : "<<frame.channels()<<endl;
        cout<<"depth : "<<frame.depth()<<endl;
        cout<<"type : "<<frame.type()<<endl;
        cout<<"size : "<<frame.size()<<endl;
        cout<<"total : "<<frame.total()<<endl;
        cout<<"elemSize : "<<frame.elemSize()<<endl;
        */

        // 6. Wait for 1 ms and check keyboard input
       if (cv::waitKey(1) == 'q')
        {
            break;
        }
            
    }

    // 7. Release camera
    cap.release();

    // 8. Close all windows
    cv::destroyAllWindows();

    return 0;
}