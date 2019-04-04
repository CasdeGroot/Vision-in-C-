#pragma once
#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/videostab/ring_buffer.hpp>

class Utilities
{
public:
	Utilities() = default;
	~Utilities() = default;
	static void GetChainCodes(const cv::Mat & img, std::vector<std::vector<int>> & chainCodes);
	static void PrintMat(const cv::Mat& mat);
	static float GetAngleBetweenVectors(const cv::Point & vector1, const cv::Point & vector2);
	static cv::Mat ThresholdImage(const cv::Mat & src, int threshold, int value, int inverted);
	static cv::Mat ErodeImage(const cv::Mat & src, cv::MorphShapes type, const cv::Size & elementSize, const cv::Point & middle);
};

