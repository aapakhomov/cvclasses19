/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "utils.hpp"

void get_histogram(cv::Mat& src, cv::Mat& output) {
    const int width = 480;
    const int height = 288;
    const auto histogram_size = 256;
    const float range[] = {0, histogram_size};
    const float* ranges = {range};
    cv::Mat mask;

    const int margin = 3;
    const int min_y = margin;
    const int max_y = height - margin;
    const float bin_width = static_cast<float>(width) / static_cast<float>(histogram_size);
    cv::Mat dst(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat histogram;
    cv::calcHist(&src, 1, 0, mask, histogram, 1, &histogram_size, &ranges, 1, 0);
    cv::normalize(histogram, histogram, 0, dst.rows, cv::NORM_MINMAX);

    for (int i = 0; i < histogram_size - 1; ++i)
    {
        const int x1 = std::round(bin_width * i);
        const int x2 = std::round(bin_width * (i + 1));
        const int y1 = std::clamp(height - (int)(std::round(histogram.at<float>(i))), min_y, max_y);
        const int y2 = std::clamp(height - (int)(std::round(histogram.at<float>(i + 1))), min_y, max_y);
        cv::line(dst, cv::Point(x1, y1), cv::Point(x2, y2), 
            cv::Scalar(255, 255, 255),  // color
            1, // thickness
            cv::LINE_AA // line type
            ); 
    }
    dst.copyTo(output);
};

int demo_feature_descriptor(int argc, char* argv[])
{
cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;
    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";
    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);
    cv::Mat frame;
    cv::Mat frame_blur;
    auto detector_a = cvlib::corner_detector_fast::create();
    auto detector_b = cv::ORB::create();
    std::vector<cv::KeyPoint> corners;
    cv::Mat descriptors;

    utils::fps_counter fps;
    int pressed_key = 0;
    while (pressed_key != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);
        cv::blur(frame, frame_blur, cv::Size(9, 9));
        detector_a->detect(frame_blur, corners);
        cv::drawKeypoints(frame, corners, frame, cv::Scalar(0, 0, 255));
        utils::put_fps_text(frame, fps);
        cv::putText(frame, "corners numeber: " + std::to_string(corners.size()), 
            cv::Point(30, 30), // top left margin
            cv::FONT_HERSHEY_SIMPLEX, // font
            1.0, // thickness
            CV_RGB(0, 255, 0) // text color
            );
        cv::imshow(demo_wnd, frame);

        pressed_key = cv::waitKey(30);
        if (pressed_key == ' ') // space
        {
            const auto hist_wnd = "hist";
            cv::namedWindow(hist_wnd);

            cv::Mat descriptors_cvlib;
            cv::Mat descriptors_cv;
            detector_a->compute(frame, corners, descriptors_cvlib);
            detector_b->compute(frame, corners, descriptors_cv);
            descriptors_cv.resize(descriptors_cv.size().height);
            cv::Mat hamming_dist = cv::Mat(descriptors_cvlib.size(), descriptors_cvlib.type());
            cv::bitwise_xor(descriptors_cvlib, descriptors_cv, hamming_dist);
            cv::Mat hamming_hist;
            get_histogram(hamming_dist, hamming_hist);
            cv::imshow(hist_wnd, hamming_hist);
            std::cout << "Dump descriptors complete!\n";
        }

        std::cout << "Feature points: " << corners.size() << "\r";
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
