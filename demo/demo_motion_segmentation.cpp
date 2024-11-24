/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

int demo_motion_segmentation(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    static const auto buff_size = 10;
    cvlib::Buffer buffer = cvlib::Buffer(buff_size);
    for (auto idx = 0; idx < buff_size; ++idx)
    {
        cv::Mat frame;
        cap >> frame;
        buffer.push_back(frame);
    }

    auto mseg = cvlib::motion_segmentation(buffer.get_mean());
    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    int threshold = 50;
    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);
    int rate_track = (int)(255.0 / buff_size);
    double learning_rate = rate_track / 255.0;
    cv::createTrackbar("rate", demo_wnd, &rate_track, 255);
    

    cv::Mat frame;
    cv::Mat frame_mseg;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        buffer.push_back(frame);
        cv::imshow(main_wnd, frame);

        mseg.apply(frame, frame_mseg, learning_rate);
        if (!frame_mseg.empty())
            cv::imshow(demo_wnd, frame_mseg);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
