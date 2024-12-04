/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <random>
#include <ctime>

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

cv::Point make_point(int sigma)
{
    std::default_random_engine generator;
    // нормальное распределение показывает большую эффективность
    // в сравнении с другими способами генерации пар
    std::normal_distribution<float> distribution_x(0, sigma);
    std::normal_distribution<float> distribution_y(0, sigma);
    int x = std::round(distribution_x(generator));
    int y = std::round(distribution_y(generator));
    return {std::clamp(x, -sigma, sigma), std::clamp(y, -sigma, sigma)};
}

void make_point_pairs(std::vector<std::pair<cv::Point, cv::Point>> &pairs, const int descriptor_size, const int neighbours_num)
{
    pairs.clear();
    const int pairs_num = neighbours_num / 2;
    for (int i = 0; i < descriptor_size; i++)
        pairs.push_back(std::make_pair(make_point(pairs_num), make_point(pairs_num)));
}

bool test_candidate(const cv::Mat& img, cv::Point point, const std::vector<cv::Point>& offsets, int pixel_num_threshold)
{
    int count_a = 0;
    int count_b = 0;
    int sz = offsets.size();
    const int threshold = 30;
    const auto pixel = img.at<uint8_t>(point);
    for (int i = 0; i < 2 * sz; ++i)
    {
        int normed_iter = i % sz;
        const auto offset_pixel = img.at<uint8_t>(point + offsets[normed_iter]);
        if (offset_pixel < pixel - threshold) {
            ++count_a;
            count_b = 0;
        }
        if (offset_pixel > pixel + threshold) {
            ++count_b;
            count_a = 0;
        }
        if (count_a >= pixel_num_threshold || count_b >= pixel_num_threshold) return true;
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

bool predicate(cv::Mat& img, cv::Point keypoint, std::pair<cv::Point, cv::Point> point) {
    return img.at<uint8_t>(keypoint + point.first) < img.at<uint8_t>(keypoint + point.second); 
}

void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    cv::Mat img;
    image.getMat().copyTo(img);
    if (img.channels() == 3)
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

    std::vector<std::pair<cv::Point, cv::Point>> pairs;
    auto descriptor_mat = descriptors.getMat();
    descriptor_mat.setTo(0);

    const int descriptor_size = 3;
    descriptors.create(static_cast<int>(keypoints.size()), descriptor_size, CV_32S);
    const int neighbours_num = 30;
    descriptors.create(static_cast<int>(keypoints.size()), descriptor_size, CV_8U);
    
    make_point_pairs(pairs, descriptor_size, neighbours_num);

    auto ptr = reinterpret_cast<uint8_t*>(descriptor_mat.ptr());
    for (const auto& keypoint : keypoints)
    {
        for (int i = 0; i < descriptor_size; ++i)
        {
            uint8_t descriptor = 0;
            for (auto j = 0; j < pairs.size(); ++j)
                descriptor |= (predicate(img, keypoint.pt, pairs.at(j)) << (pairs.size() - 1 - j));
            *ptr = descriptor;
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
