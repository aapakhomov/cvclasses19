/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <vector>
#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <cassert>
#include <numeric>

// should be class fields
// segments are defined as set of rectangles
std::vector<cv::Rect> candidates;
std::map<int, std::vector<cv::Rect>> segments;

namespace
{
// passing point is needed to identify rectangle position
void split_image(cv::Mat image, double stddev, cv::Point top_left)
{    
    cv::Mat mean;
    cv::Mat dev;
    cv::meanStdDev(image, mean, dev);

    const auto width = image.cols;
    const auto height = image.rows;

    if (dev.at<double>(0) <= stddev)
    {
        image.setTo(mean);
        // since number of segments during split is unknown
        // initially segments are put into vector
        // keys for map cannot be defined using static variable or function parameter
        // because of collision issues
        candidates.push_back(cv::Rect(top_left, image.size()));
        return;
    }

    split_image(image(cv::Range(0, height / 2), cv::Range(0, width / 2)), stddev, top_left);
    split_image(image(cv::Range(0, height / 2), cv::Range(width / 2, width)), stddev, top_left + cv::Point(0, width / 2));
    split_image(image(cv::Range(height / 2, height), cv::Range(width / 2, width)), stddev, top_left + cv::Point(height / 2, width / 2));
    split_image(image(cv::Range(height / 2, height), cv::Range(0, width / 2)), stddev, top_left + cv::Point(height / 2, 0));
}

bool rects_are_neighbours(cv::Rect seg_a, cv::Rect seg_b)
{
    if (
        (seg_a.y <= seg_b.y) && (seg_a.y + seg_a.height >= seg_b.y) &&
        (
            (seg_a.x >= seg_b.x) && (seg_a.x < seg_b.x + seg_b.width)
            ||
            (seg_a.x < seg_b.x) && (seg_a.x + seg_a.width > seg_b.x)
        )
        ||
        (seg_a.x <= seg_b.x + seg_b.width) && (seg_a.x + seg_a.width >= seg_b.x + seg_b.width) && 
        (
            (seg_a.y >= seg_b.y) && (seg_a.y < seg_b.y + seg_b.height)
            ||
            (seg_a.y < seg_b.y) && (seg_a.y + seg_a.height > seg_b.y)
        )
        ||
        (seg_a.y <= seg_b.y) && (seg_a.y <= seg_b.y + seg_b.height) &&
        (
            (seg_a.x >= seg_b.x) && (seg_a.x < seg_b.x + seg_b.width)
            ||
            (seg_a.x < seg_b.x) && (seg_a.x + seg_a.width > seg_b.x)
        )
        ||
        (seg_a.x <= seg_b.x) && (seg_a.x + seg_a.width >= seg_b.x) &&
        (
            (seg_a.y <= seg_b.y) && (seg_a.y + seg_a.height > seg_b.y)
            ||
            (seg_a.y > seg_b.y) && (seg_a.y < seg_b.y + seg_b.height)
        )
    ) {
        return true;
    }
    return false;
}

std::vector<uchar> Flatten(const cv::Mat& mat) {
    std::vector<uchar> array;
    if (mat.isContinuous()) {
        array.assign(mat.data, mat.data + mat.total()*mat.channels());
    } else {
        // contiguity can be violated while extracting rois
        for (int i = 0; i < mat.rows; ++i) {
            array.insert(array.end(), mat.ptr<uchar>(i), mat.ptr<uchar>(i) + mat.cols*mat.channels());
        }
    }
    return array;
}

void vec_to_map() {
    for (int i = 0; i < candidates.size(); ++i) {
        std::vector<cv::Rect> segment;
        segment.push_back(candidates[i]);
        segments.emplace(i, segment);
    }
}

// debug
void print(std::vector<cv::Rect> rois) {
    for (const auto& roi : rois) {
        std::cout << roi << std::endl;
    }
}

// debug
void print(std::string str) {
    std::cout << str << std::endl;
}

bool segments_are_neighbours(std::vector<cv::Rect> seg_a, std::vector<cv::Rect> seg_b) {
    // segments are neighbours if any of rects comprising segments are neighbours
    for (const auto& rect_a : seg_a) {
        for (const auto& rect_b : seg_b) {
            if (rects_are_neighbours(rect_a, rect_b) || rects_are_neighbours(rect_b, rect_a)) return true;
        }
    }
    return false;
}

int get_segment_area(std::vector<cv::Rect> seg) {
    int area = 0; // number of pixels
    for (const auto& rect : seg) {
        area += rect.area();
    }
    return area;
}


uchar dev_criterion(std::vector<cv::Rect> seg_a, std::vector<cv::Rect> seg_b, cv::Mat image, double stddev) {
    // the following procedure can be errrorprone for small and large float numbers
    // image must be converted to uchar
    std::vector<uchar> flat_a;
    std::vector<uchar> flat_b;
    std::vector<uchar> concated;

    concated.reserve(get_segment_area(seg_a) + get_segment_area(seg_b));
    for (int i = 0; i < seg_a.size(); ++i) {
        flat_a = Flatten(image(seg_a[i]));
        for (int j = 0; j < seg_b.size(); ++j) {
            flat_b = Flatten(image(seg_b[j]));
            concated.insert(concated.end(), flat_b.begin(), flat_b.end());
        }
        // since Var(A+B) = Var(A) + Var(B) - 2Cov(A, B)
        // rects should be concated in order to calculate stddev without calculating covariance
        // we cannot just sum individual stddevs up
        // to concatenate different-sized segments, they are flattened initially
        concated.insert(concated.end(), flat_a.begin(), flat_a.end());
    }

    double sum = std::accumulate(concated.begin(), concated.end(), 0.0);
    double mean = sum / concated.size();
    assert(mean >= 0); // ensure no overflow after casting to uchar 
    double sq_sum = std::inner_product(concated.begin(), concated.end(), concated.begin(), 0.0);
    double dev = std::sqrt(sq_sum / concated.size() - mean * mean);

    if (dev < stddev) return (uchar)mean;
    return 0; 
    // if mean will be zero, ans since brightness cannot be negative, all the regions
    // would be zero too, so we don't have to merge them
    // and can use 0 as flag for predicate
}


void propagate_segments(cv::Mat image, double stddev, bool paint=false) {
    bool merge_flag = 0;
    for (auto it_a = segments.begin(); it_a != segments.end(); ++it_a) {
        if (it_a->second.empty()) continue;
        for (auto it_b = segments.begin(); it_b != segments.end(); ++it_b) {
            if (it_b->second.empty()) continue;
            if (segments_are_neighbours(it_a->second, it_b->second)) {
                uchar mean = dev_criterion(it_a->second, it_b->second, image, stddev);
                if (mean && it_a != it_b) {
                    merge_flag = 1;
                    for (const auto& rect : it_b->second) {
                        // paint one segment
                        if (paint) {
                            image(rect).setTo(mean);
                        } else {
                            it_a->second.push_back(cv::Rect(rect));
                        }
                    }
                    // change segments only if paint=false
                    // segment merged into another segment is defined as empty vector
                    // since deleting key would cause iterator invalidation
                    if (!paint) it_b->second.clear();
                }
            }
        }
    }
    // we don't need recursive call during painting
    if (merge_flag && !paint) propagate_segments(image, stddev);
}


void merge_image(cv::Mat image, double stddev)
{
    vec_to_map();
    propagate_segments(image, stddev);
    propagate_segments(image, stddev, true);
    // clear segments before new test case
    segments.clear();
    candidates.clear();
}
} // namespace

namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    // split part
    cv::Mat res = image;
    split_image(res, stddev, cv::Point(0, 0));

    // merge part
    // \todo implement merge algorithm
    merge_image(res, stddev);
    return res;
}
} // namespace cvlib