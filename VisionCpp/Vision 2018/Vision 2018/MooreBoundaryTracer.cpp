#include "MooreBoundaryTracer.h"
#include "lib/avansvisionlib/avansvisionlib.h"

const std::vector<cv::Point> neightborhood_Matrix = {
	cv::Point(-1,-1),
	cv::Point(0,-1),
	cv::Point(1,-1),
	cv::Point(1, 0),
	cv::Point(1, 1),
	cv::Point(0, 1),
	cv::Point(-1, 1),
	cv::Point(-1, 0)
};

int MooreBoundaryTracer::AllContours(cv::Point firstPixel, const cv::Mat & image, std::vector<std::vector<cv::Point>> & contours, int borderColor, int backgroundColor) {
	//iterate through image rowwise
	if (image.at<ushort>(cv::Point(firstPixel.y, firstPixel.x)) == borderColor) {
		//set the first backtrackPoint
		cv::Point backtrackPoint = cv::Point(firstPixel.y - 1, firstPixel.x);
		//set the startpoint
		cv::Point startPoint = cv::Point(firstPixel.y, firstPixel.x);

		//initialise boundary vector
		std::vector<cv::Point> boundary = std::vector<cv::Point>();

		//set current point to next found pixel of the boundary
		cv::Point currentPoint = FindNextBoundaryPixel(image, startPoint, backtrackPoint, boundary, backgroundColor);

		//iterate through all boundary pixels until you are back at the beginning
		while (currentPoint != startPoint) {
			currentPoint = FindNextBoundaryPixel(image, currentPoint, backtrackPoint, boundary, backgroundColor);
		}
		contours.push_back(boundary);
		return 1;
	}
	return 1;
}

void MooreBoundaryTracer::TestAllContours(const cv::Mat & img16S, cv::Mat & customImage, std::vector<std::vector<cv::Point>> & contours) {

	cv::Mat labeledImg;
	std::vector<cv::Point2d *> firstPixelVec = std::vector<cv::Point2d *>();
	std::vector<cv::Point2d *> posVec = std::vector<cv::Point2d *>();
	std::vector<int> areaVec = std::vector<int>();
	labelBLOBsInfo(img16S, labeledImg, firstPixelVec, posVec, areaVec, 100);


	for (int i = 0; i < firstPixelVec.size(); i++) {
		AllContours(static_cast<cv::Point>(*firstPixelVec[i]), img16S, contours);
	}
	customImage = drawAllContours(contours, img16S.rows, img16S.cols);
}


cv::Mat MooreBoundaryTracer::drawAllContours(const std::vector<std::vector<cv::Point>> & vector, int rows, int cols) {
	cv::Mat img(rows, cols, CV_8UC1, cv::Scalar(0));

	for (std::vector<cv::Point> v : vector) {
		drawContour(img, v);
	}

	return img;
}

cv::Mat MooreBoundaryTracer::drawAllContours(const std::vector<std::vector<cv::Point>> & vector, cv::Mat & img) {

	for (std::vector<cv::Point> v : vector) {
		drawContour(img, v);
	}

	return img;
}

void MooreBoundaryTracer::drawContour(cv::Mat & img, const std::vector<cv::Point> & vector) {
	for (cv::Point p : vector) {
		img.at<uchar>(cv::Point(p.x, p.y)) = 255;
	}
}

cv::Point MooreBoundaryTracer::FindNextBoundaryPixel(const cv::Mat & image, cv::Point & currentPoint, cv::Point & backtrackPoint, std::vector<cv::Point> & boundary, int backgroundColor) {
	//push back the next found boundary point
	boundary.push_back(currentPoint);

	int pos = std::distance(neightborhood_Matrix.begin(), std::find(neightborhood_Matrix.begin(), neightborhood_Matrix.end(), backtrackPoint - currentPoint));

	//get the next clockwise pixel
	cv::Point clockwise = GetClockWisePixel(currentPoint, pos);

	//check every offset for a white pixel
	while (image.at<ushort>(clockwise) == backgroundColor) {
		backtrackPoint = clockwise;
		clockwise = GetClockWisePixel(currentPoint, pos);
	}

	return clockwise;
}

cv::Point MooreBoundaryTracer::GetClockWisePixel(cv::Point currentPoint, int & matrixPos) {

	if (matrixPos < NR_OF_NEIGHBOURS)
		matrixPos++;
	else
		matrixPos = 0;

	return currentPoint + neightborhood_Matrix[matrixPos];
}
