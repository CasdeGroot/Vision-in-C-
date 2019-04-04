#pragma once
#include <opencv2/core/mat.hpp>

class BoundaryFill
{
public:
	BoundaryFill() = default;
	~BoundaryFill() = default;

	static void FillAllBoundaries(cv::Mat & image, const std::vector<std::vector<cv::Point>>& contours, std::vector<std::vector<cv::Point>>& regionPixels);
	static void BoundaryFill8(cv::Mat& image, int x, int y, std::vector<cv::Point>& regionPixels, bool animate = false);
	static void AddBoundaryBorder(cv::Mat& image, const std::vector<cv::Point>& boundary, std::vector<cv::Point>& doubleBoundaryVec);
	static cv::Point FindPixelInBoundary(const cv::Mat& image, const std::vector<cv::Point>& contour);
private: 
};

