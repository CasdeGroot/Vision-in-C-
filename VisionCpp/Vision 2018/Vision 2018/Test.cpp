#include "Test.h"
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/shape/hist_cost.hpp>
#include "lib/avansvisionlib/avansvisionlib.h"
#include "TimonLib.h"
#include "MooreBoundaryTracer.h"
#include "BoundingBoxer.h"
#include "BendingEnergyCalculator.h"
#include "BoundaryFill.h"
#include "FloodFill.h"
#include <opencv2/video/background_segm.hpp>

void Test::TestMooreBoundaryTracer() {
	cv::Mat image, image16S;
	ProcessLeafImage(image, image16S);

	cv::Mat boundaryImage;
	std::vector<std::vector<cv::Point>> contours = std::vector<std::vector<cv::Point>>();
	MooreBoundaryTracer::TestAllContours(image16S, boundaryImage, contours);
	cv::imshow("Leafs boundary", boundaryImage);
}

void Test::TestBoundingBoxer() {

	cv::Mat image, image16S;
	ProcessTestSetImage(image, image16S);

	show16SImageStretch(image16S);

	cv::Mat boundaryImage;
	std::vector<std::vector<cv::Point>> contours = std::vector<std::vector<cv::Point>>();
	MooreBoundaryTracer::TestAllContours(image16S, boundaryImage, contours);

	std::vector<std::vector<cv::Point>> bbs = std::vector<std::vector<cv::Point>>();
	BoundingBoxer::AllBoundingBoxes(contours, bbs);
	BoundingBoxer::CutAllBoxes(boundaryImage, bbs, "testset", 1);
	BoundingBoxer::DrawBoundingBoxes(image16S, bbs);

	show16SImageStretch(image16S);
}

void Test::TestBendingEnergy() {
	cv::Mat image, image16S;
	ProcessRummikubImage(image, image16S);
	std::vector<float> bendingEnergyVec = std::vector<float>();
	BendingEnergyCalculator::TestBendingEnergy(image, bendingEnergyVec);
}

void Test::TestExtraBoundary() {
	cv::Mat image, image16S;
	ProcessLeafImage(image, image16S);

	cv::Mat boundaryImage;
	std::vector<std::vector<cv::Point>> contours = std::vector<std::vector<cv::Point>>();
	MooreBoundaryTracer::TestAllContours(image16S, boundaryImage, contours);

	cv::imshow("without border", boundaryImage);

	for(auto vector: contours) {
		std::vector<cv::Point> secondBoundaryVec;
		BoundaryFill::AddBoundaryBorder(boundaryImage, vector, secondBoundaryVec);
	}

	cv::imshow("with border", boundaryImage);
}

void Test::TestBoundaryFill() {
	cv::Mat image, image16S;
	ProcessLeafImage(image, image16S);

	cv::Mat boundaryImage;
	std::vector<std::vector<cv::Point>> contours = std::vector<std::vector<cv::Point>>();
	MooreBoundaryTracer::TestAllContours(image16S, boundaryImage, contours);

	for (auto vector : contours) {
		std::vector<cv::Point> secondBoundaryVec;
		BoundaryFill::AddBoundaryBorder(boundaryImage, vector, secondBoundaryVec);
	}

	boundaryImage.convertTo(boundaryImage, CV_16S);

	std::vector<std::vector<cv::Point>> regionPixels;
	BoundaryFill::FillAllBoundaries(boundaryImage, contours, regionPixels);
	show16SImageStretch(boundaryImage, "Filled monster");
}

void Test::TestLoadingCSV()
{
	cv::Mat loadedMatObject;
	TimonLib::LoadMatFromCSV("csv\\banknotes.csv", loadedMatObject);

	std::string hello = "brabantismooi";
}

void Test::FloodFill() {
	cv::Mat image, image16S;
	ProcessMonsterImage(image, image16S);

	cv::Mat boundaryImage;
	std::vector<std::vector<cv::Point>> contours = std::vector<std::vector<cv::Point>>();
	MooreBoundaryTracer::TestAllContours(image16S, boundaryImage, contours);

	for (auto vector : contours) {
		std::vector <cv::Point> secondBoundaryVec;
		BoundaryFill::AddBoundaryBorder(boundaryImage, vector, secondBoundaryVec);
		cv::imshow("boundary", boundaryImage);
	}

	boundaryImage.convertTo(boundaryImage, CV_16S);

	std::vector<std::vector<cv::Point>> regionPixels;
	FloodFill::FillAll(boundaryImage, contours, regionPixels);
	show16SImageStretch(boundaryImage, "Flood fill");
}

void Test::ProcessMonsterImage(cv::Mat & img, cv::Mat & img16S) {
	img = cv::imread("img\\monsters.jpg", CV_LOAD_IMAGE_COLOR);
	cv::cvtColor(img, img, CV_BGR2GRAY);
	img.convertTo(img16S, CV_16S);
	img = cv::threshold(img16S, img16S, 165, 1, CV_THRESH_BINARY_INV);
	show16SImageStretch(img16S, "Processed Monster");
}

void Test::ProcessRummikubImage(cv::Mat & img, cv::Mat & img16S) {
	img = cv::imread("img\\rummikub0.jpg", CV_LOAD_IMAGE_COLOR);
	cv::cvtColor(img, img, CV_BGR2GRAY);
	img.convertTo(img16S, CV_16S);
	cv::threshold(img16S, img16S, 65, 1, CV_THRESH_BINARY_INV);
	cv::erode(img16S, img16S, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1)));
	show16SImageStretch(img16S, "Processed Rummikub");
}

void Test::ProcessLeafImage(cv::Mat & img, cv::Mat & img16S) {
	img = cv::imread("img\\bladeren.jpg", CV_LOAD_IMAGE_COLOR);
	cv::cvtColor(img, img, CV_BGR2GRAY);
	img.convertTo(img16S, CV_16S);
	cv::threshold(img16S, img16S, 200, 1, CV_THRESH_BINARY_INV);
	cv::dilate(img16S, img16S, getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1)));
	cv::erode(img16S, img16S, getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(1, 1)));
	show16SImageStretch(img16S, "Processed Leaf");
}

void Test::ProcessTestSetImage(cv::Mat & img, cv::Mat & img16S) {
	img = cv::imread("img\\testset.png", CV_LOAD_IMAGE_COLOR);
	cv::cvtColor(img, img, CV_BGR2GRAY);
	img.convertTo(img16S, CV_16S);
	cv::threshold(img16S, img16S, 200, 1, CV_THRESH_BINARY_INV);
	show16SImageStretch(img16S, "Processed test set");
}

