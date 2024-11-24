/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

bool test_candidate(const cv::Mat& img, cv::Point point, const std::vector<cv::Point>& offsets, int pixel_num_threshold)
{
    int count = 0;
    const int threshold = 30;
    const auto pixel = img.at<uint8_t>(point);
    for (const auto& offset : offsets)
    {
        const auto offset_pixel = img.at<uint8_t>(point + offset);
        if (offset_pixel < pixel - threshold || offset_pixel > pixel + threshold) ++count;
        if (count >= pixel_num_threshold) return true;
    }
    return false;
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear();
    const auto img_ = image.getMat();
    cv::Mat img;
    img_.copyTo(img);
    if (img.channels() == 3)
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    // точки для проверки, представленные как смещения от тестируемой точки
    // первая группа пикселей (крест) проверяется в начале, если предикат не выполнен, процедура завершается
    // если выполнен, проверяются оставшиеся пиксели
    const auto offsets_init = std::vector<cv::Point>{
        cv::Point(0, 3), 
        cv::Point(3, 0), 
        cv::Point( 0, -3), 
        cv::Point(-3, 0)
        };
    const auto offsets_residual = std::vector<cv::Point>{
        cv::Point(-1, 3), cv::Point(1, 3),
        cv::Point(2, 2), 
        cv::Point(3, 1), cv::Point(3, -1), 
        cv::Point(2, -2),
        cv::Point(1, -3), cv::Point(-1, -3),
        cv::Point(-2, -2),
        cv::Point(-3, -1), cv::Point(-3, 1),
        cv::Point(-2, 2)
        };

    // крайние 3 ряда пикселей не обрабатываются, т.к. окно не помещается на изображении
    for (auto row = 3; row < img.rows - 3; ++row)
    {
        for (auto col = 3; col < img.cols - 3; ++col)
        {
            const auto center = cv::Point(col, row);
            if (test_candidate(img, center, offsets_init, 3) && test_candidate(img, center, offsets_residual, 12)) {
                keypoints.push_back(cv::KeyPoint(center, 3));
            }
        }
    }
}

void corner_detector_fast::compute(cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    // \todo implement any binary descriptor
    const int desc_length = 2;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
        for (int i = 0; i < desc_length; ++i)
        {
            *ptr = std::rand();
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray image, cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool /*= false*/)
{
    // \todo implement me
    this->detect(image, keypoints);
    this->compute(image, keypoints, descriptors);
}
} // namespace cvlib
