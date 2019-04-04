#pragma once
#include <opencv2/core/mat.hpp>

class FloodFill
{
public:
	FloodFill() = default;
	~FloodFill() = default;
	static void FloodFill8(cv::Mat& image, int x, int y, std::vector<cv::Point>& regionPixels);
	static void FillAll(cv::Mat& image, const std::vector<std::vector<cv::Point>>& contours, std::vector<std::vector<cv::Point>>& regionPixels);
};

