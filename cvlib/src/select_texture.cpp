/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace
{
struct descriptor : public std::vector<double>
{
    using std::vector<double>::vector;
    descriptor operator-(const descriptor& right) const
    {
        descriptor temp = *this;
        for (size_t i = 0; i < temp.size(); ++i)
        {
            temp[i] -= right[i];
        }
        return temp;
    }

    double norm_l1() const
    {
        double res = 0.0;
        for (auto v : *this)
        {
            res += std::abs(v);
        }
        return res;
    }

    double norm_l2() const
    {
        double res = 0.0;
        for (auto v : *this)
        {
            res += v*v;
        }
        return res;
    }
};


int round_to_odd(double d) {
    // corner cases
    if (d >= std::numeric_limits<int>::max()) return std::numeric_limits<int>::max();
    if (d <= std::numeric_limits<int>::min()) return std::numeric_limits<int>::min() + 1;
    
    bool negative = d < 0;
    int truncated = negative ? -d : d; 
    if (truncated % 2 == 0) 
        ++truncated; 

    return negative ? -truncated : truncated;
}


void calculateDescriptor(const cv::Mat& image, int kernel_size, descriptor& descr)
{
    descr.clear();
    const std::vector<double> lm = {17, 29, 41, 59, 71, 89, 97};
    cv::Mat response;
    cv::Mat mean;
    cv::Mat dev;

    // \todo implement complete texture segmentation based on Gabor filters
    // (find good combinations for all Gabor's parameters)
    
    // no prior info is given, so it's better to make such a descriptor that is capable to represent
    // different types of images
    // so for each parameter a set of values will be considered

    // 8 directions -- to cover horizontal, vertical and diagonal directions of texture
    // since we use 8 directions, no need to consider gamma > 1
    // because it's the same as reciprocate gamma + rotation (if we consider gamma as ellipticity)
    
    // to make effective descriptor, we should use such lambdas that gabor filters with different lambdas
    // would intersect as little as possible
    // e.g. not like lambda_1 = 10 and lambda_2 = 20, since they are multiples of 10,
    // half of information will be reduntant

    // let's choose mutually prime numbers

    for (auto th = 0.0; th <= 6 * CV_PI / 8; th += CV_PI / 8)
    {
        for (auto sig = 5; sig <= 15; sig += 5)
        {
            for (auto l : lm) {
                for (auto gm = 0.25; gm <= 1; ++gm) {
                    cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sig, th, l, gm);
                    cv::filter2D(image, response, CV_32F, kernel);
                    cv::meanStdDev(response, mean, dev);
                    descr.emplace_back(mean.at<double>(0));
                    descr.emplace_back(dev.at<double>(0));
                }
            }
        }
    }
}
} // namespace

namespace cvlib
{
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps)
{
    cv::Mat imROI = image(roi);

    const int kernel_size = round_to_odd(std::min(roi.height, roi.width) / 2); // \todo round to nearest odd

    descriptor reference;
    calculateDescriptor(image(roi), kernel_size, reference);

    cv::Mat res = cv::Mat::zeros(image.size(), CV_8UC1);

    descriptor test(reference.size());
    cv::Rect baseROI = roi - roi.tl();

    // \todo move ROI smoothly pixel-by-pixel
    // to move roi pixel-by-pixel, we should consider corner cases, where filter doesnt fit into the image
    // e.g. point 0,0
    // we can use padding or, as here, just don't consider such points
    for (int i = roi.width; i < image.size().width - roi.width; ++i)
    {
        for (int j = roi.height; j < image.size().height - roi.height; ++j)
        {
            auto curROI = baseROI + cv::Point(i, j); // pixel by pixel roi
            calculateDescriptor(image(curROI), kernel_size, test);
            res(curROI) = 255 * ((test - reference).norm_l2() <= eps);
        }
    }

    return res;
}
} // namespace cvlib
