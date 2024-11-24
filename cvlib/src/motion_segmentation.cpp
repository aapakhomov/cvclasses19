/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <iostream>

namespace cvlib
{
Buffer::Buffer(size_t buff_size) : _buffer(buff_size) {}

void Buffer::push_back(cv::Mat frame) {
    _buffer.push_back(frame);
}

cv::Mat Buffer::get_mean() {
    cv::Mat mean = cv::Mat(_buffer.front().size(), CV_32FC3, cv::Scalar());
    for (auto& frame : _buffer)
        cv::accumulate(frame, mean);
    mean.convertTo(mean, CV_8UC3, 1.0 / _buffer.size());
    return mean;
}

motion_segmentation::motion_segmentation(cv::Mat bg) : bg_model_(bg) {}

void motion_segmentation::apply(cv::InputArray _image, cv::OutputArray _fgmask, double learning_rate)
{
    // \todo implement your own algorithm:
    //       * MinMax
    //       * Mean
    //       * 1G
    //       * GMM
    cv::Mat image = _image.getMat();
    cv::Mat fgmask = _fgmask.getMat();

    // \todo implement bg model updates
    bg_model_ = (1 - learning_rate) * bg_model_ + learning_rate * image;
    cv::absdiff(image, bg_model_, fgmask);
    threshold(fgmask, fgmask, 25, 128, cv::THRESH_BINARY);
    cv::cvtColor(fgmask, fgmask, cv::COLOR_BGR2GRAY);

    _fgmask.create(image.size(), CV_8UC3);
    _fgmask.assign(fgmask);
}
} // namespace cvlib
