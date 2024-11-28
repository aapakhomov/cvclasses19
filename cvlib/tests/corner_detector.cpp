/* FAST corner detector algorithm testing.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

using namespace cvlib;

TEST_CASE("simple check", "[corner_detector_fast]")
{
    auto fast = corner_detector_fast::create();
    cv::Mat image(10, 10, CV_8UC1, cv::Scalar(0));
    SECTION("empty image")
    {
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.empty());
    }

    // \todo add 5 or more tests (SECTIONs)
    // 
    SECTION("constant")
    {
        const auto image = cv::Mat(3, 3, CV_8UC1, cv::Scalar(127));
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.empty());
    }

    SECTION("^")
    {
        cv::Mat image = (cv::Mat_<uint8_t>(7, 7) <<
                15, 15, 15, 15, 15, 15, 15,
                15, 15, 15, 15, 15, 15, 15,
                15, 15, 15, 15, 15, 15, 15,
                15, 15, 15, 228, 15, 15, 15,
                15, 15, 228, 228, 228, 15, 15,
                15, 15, 228, 15, 228, 15, 15,
                15, 228, 15, 15, 15, 228, 15
                );
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(1 == out.size());
        REQUIRE(out[0].pt.x == 3);
        REQUIRE(out[0].pt.y == 3);
    }

    SECTION("v")
    {
        cv::Mat image = (cv::Mat_<uint8_t>(7, 7) <<
                15, 228, 15, 15, 15, 15, 15,
                15, 15, 228, 15, 15, 15, 228,
                15, 15, 228, 15, 228, 228, 15,
                15, 15, 15, 228, 15, 15, 15,
                15, 15, 15, 15, 15, 15, 15,
                15, 15, 15, 15, 15, 15, 15,
                15, 15, 15, 15, 15, 15, 15
                );
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(1 == out.size());
        REQUIRE(out[0].pt.x == 3);
        REQUIRE(out[0].pt.y == 3);
    }

    SECTION("<")
    {
        cv::Mat image = (cv::Mat_<uint8_t>(7, 7) <<
                15, 15, 15, 15, 15, 15, 15,
                15, 15, 15, 15, 15, 15, 228,
                15, 15, 15, 15, 228, 228, 15,
                15, 15, 15, 228, 15, 15, 15,
                15, 15, 15, 15, 228, 228, 15,
                15, 15, 15, 15, 15, 15, 228,
                15, 15, 15, 15, 15, 15, 15
                );
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(1 == out.size());
        REQUIRE(out[0].pt.x == 3);
        REQUIRE(out[0].pt.y == 3);
    }

    SECTION("strip")
    {
        cv::Mat image = (cv::Mat_<uint8_t>(7, 7) <<
                15, 228, 228, 228, 228, 228, 15,
                15, 228, 228, 228, 228, 228, 15,
                15, 228, 228, 228, 228, 228, 15,
                15, 228, 228, 228, 228, 228, 15,
                15, 228, 228, 228, 228, 228, 15,
                15, 228, 228, 228, 228, 228, 15,
                15, 228, 228, 228, 228, 228, 15
                );
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.empty());
    }
}
