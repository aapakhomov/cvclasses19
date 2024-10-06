/* Split and merge segmentation algorithm testing.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

using namespace cvlib;

TEST_CASE("constant image", "[split_and_merge]")
{
    const cv::Mat image(100, 100, CV_8UC1, cv::Scalar{15});

    const auto res = split_and_merge(image, 1);
    REQUIRE(image.size() == res.size());
    REQUIRE(image.type() == res.type());
    REQUIRE(cv::Scalar(15) == cv::mean(res));
}

TEST_CASE("simple regions", "[split_and_merge]")
{
    SECTION("2x2")
    {
        const cv::Mat reference = (cv::Mat_<char>(2, 2) << 2, 2, 2, 2);
        cv::Mat image = (cv::Mat_<char>(2, 2) << 0, 1, 2, 3);
        auto res = split_and_merge(image, 5);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }

    SECTION("3x3")
    {
        const cv::Mat reference = (cv::Mat_<char>(3, 3) << 2, 2, 2, 
                                                            45, 45, 45, 
                                                            45, 45, 45);
        cv::Mat image = (cv::Mat_<char>(3, 3) << 2, 1, 2, 
                                                45, 40, 50, 
                                                45, 40, 50);
        auto res = split_and_merge(image, 10);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }
}

TEST_CASE("compex regions", "[split_and_merge]")
{
    SECTION("2x2")
    {
        const cv::Mat reference = (cv::Mat_<char>(2, 2) << 2, 2, 
                                                            2, 2);
        cv::Mat image = (cv::Mat_<char>(2, 2) << 2, 3, 
                                                1, 4);
        auto res = split_and_merge(image, 10);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }

    SECTION("3x3")
    {
        const cv::Mat reference = (cv::Mat_<char>(3, 3) << 1, 28, 28,
                                                          31, 31, 28, 
                                                          31, 31, 4);
        cv::Mat image = (cv::Mat_<char>(3, 3) << 1, 28, 29, 
                                                30, 31, 28, 
                                                32,  31, 4);
        auto res = split_and_merge(image, 2);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }

    SECTION("4x4")
    {
        const cv::Mat reference = (cv::Mat_<char>(4, 4) << 2, 2, 10, 10, 
                                                           2, 2, 10, 10, 
                                                           4, 4, 10, 5, 
                                                           4, 4, 5, 1
                                                           );
        cv::Mat image = (cv::Mat_<char>(4, 4) << 1, 1, 9, 12, 
                                                 1, 5, 10, 9,
                                                 4, 4, 10, 5,
                                                 5, 4, 5, 1
                                                 );
        auto res = split_and_merge(image, 2);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }
}
