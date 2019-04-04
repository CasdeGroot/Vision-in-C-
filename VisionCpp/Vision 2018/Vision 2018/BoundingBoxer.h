#pragma once
#include <opencv2/core/mat.hpp>

class BoundingBoxer
{
public:
	BoundingBoxer() = default;
	~BoundingBoxer() = default;
	static int AllBoundingBoxes(const std::vector<std::vector<cv::Point>> & contours, std::vector<std::vector<cv::Point>> & bbs);
	static cv::Point GetLargestBox(std::vector<std::vector<cv::Point>> bbs);
	static void DrawBoundingBoxes(cv::Mat& image, std::vector<std::vector<cv::Point>> bbs, int offset = 0);
	static void CutAllBoxes(const cv::Mat & image, std::vector<std::vector<cv::Point>> bbs, std::string filename, int offset = 0);
};

