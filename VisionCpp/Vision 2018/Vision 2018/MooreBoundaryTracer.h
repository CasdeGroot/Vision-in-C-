#pragma once
#include <opencv2/core/mat.hpp>

class MooreBoundaryTracer
{
public:
	MooreBoundaryTracer()= default;
	~MooreBoundaryTracer() = default;
	static int AllContours(cv::Point firstPixel, const cv::Mat & image, std::vector<std::vector<cv::Point>> & contours, int borderColor = 1, int backgroundColor = 0);
	static cv::Mat drawAllContours(const std::vector<std::vector<cv::Point>> & vector, int rows, int cols);
	static cv::Mat drawAllContours(const std::vector<std::vector<cv::Point>>& vector, cv::Mat& img);
	static void TestAllContours(const cv::Mat & img16S, cv::Mat & customImage, std::vector<std::vector<cv::Point>> & contours);
	static void drawContour(cv::Mat & img, const std::vector<cv::Point> & vector);
private:
	static cv::Point FindNextBoundaryPixel(const cv::Mat & image, cv::Point & currentPoint, cv::Point & backtrackPoint, std::vector<cv::Point> & boundary, int backgroundColor = 0);
	static cv::Point GetClockWisePixel(cv::Point currentPoint, int & matrixPos);
	static const int NR_OF_NEIGHBOURS = 7;
};

