#pragma once
#include <opencv2/core/mat.hpp>

class TimonLib
{
public:
	struct BoundingBox;

	TimonLib() = default;
	~TimonLib() = default;

	static void LoadMatFromCSV(std::string fileName, cv::Mat & matFile);

	//Load csv file and returns mat object, size is determined by csv content
	//static void LoadMatFromCSV(std::string fileName, cv::Mat &matFile);

	//Get enclosed pixels recursively, but with a point stack
	static int PointStackEnclosedPixels(const std::vector<cv::Point> &contourVec, std::vector<cv::Point> &regionPixels);

	//Works but generates a stackoverflow for larger images
	static int RecursiveEnclosedPixels(const std::vector<cv::Point> &contourVec, std::vector<cv::Point> &regionPixels);
private:
	//Expects 8 bit single channel Mat object as image and shows it to the user
	static void Show8BitBinaryImage(const cv::Mat &binaryImage);

	//Generates a binary image from a contour using bounding boxes
	static cv::Mat GenerateBinaryImageFromContour(const std::vector<cv::Point> &contourVec);

	//Gets the smallest bounding box fitting the contour given
	static BoundingBox GetBoundingBoxFromContour(const std::vector<cv::Point> contours);

	//Recursive boundaryfill using point stacks
	static void StackBoundaryFill4(int x, int y, int boundaryColor, int fillColor, cv::Mat &contourImage, std::vector<cv::Point> &regionPixels);\
	//Recursive boundaryfill using recursive functions calls(Generates stack overflow for larger images)
	static void BoundaryFill4(int x, int y, int boundaryColor, cv::Mat &contourImage, std::vector<cv::Point> &regionPixels);
};