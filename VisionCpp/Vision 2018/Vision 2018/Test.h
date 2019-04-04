#pragma once
#include <opencv2/core/mat.hpp>

class Test
{
public:
	Test() = default;
	~Test() = default;
	static void TestMooreBoundaryTracer();
	static void TestBoundingBoxer();
	static void TestBendingEnergy();
	static void TestExtraBoundary();
	static void TestBoundaryFill();
	static void TestLoadingCSV();
	static void FloodFill();
private:
	static void ProcessMonsterImage(cv::Mat& img, cv::Mat& img16S);
	static void ProcessRummikubImage(cv::Mat& img, cv::Mat& img16S);
	static void ProcessLeafImage(cv::Mat& img, cv::Mat& img16S);
	static void ProcessTestSetImage(cv::Mat& img, cv::Mat& img16S);
};

