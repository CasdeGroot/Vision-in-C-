#include "BoundingBoxer.h"
#include <opencv2/videostab/ring_buffer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "lib/avansvisionlib/avansvisionlib.h"

int BoundingBoxer::AllBoundingBoxes(const std::vector<std::vector<cv::Point>> & contours, std::vector<std::vector<cv::Point>> & bbs) {
	int minX = 0;
	int maxX = 0;
	int minY = 0;
	int maxY = 0;

	//loop through all contours
	for (std::vector<cv::Point> c : contours) {
		for (int i = 0; i < c.size(); i++) {
			//set all ints for the first time
			if (i == 0) {
				minX = c[i].x;
				maxX = c[i].x;
				minY = c[i].y;
				maxY = c[i].y;
			}

			//if x > maxX set x as maxX
			if (c[i].x > maxX) maxX = c[i].x;

			//if x < minX set x as minX
			if (c[i].x < minX) minX = c[i].x;

			//if y < minY set y as minY
			if (c[i].y < minY) minY = c[i].y;

			//if y > maxY set y as maxY
			if (c[i].y > maxY) maxY = c[i].y;
		}

		//make the boundingvox array
		std::vector<cv::Point> bb = std::vector<cv::Point>();
		//place both points of the bb in the vector
		bb.emplace_back(minX, minY);
		bb.emplace_back(maxX, maxY);
		//push the vector to the bbs vector
		bbs.push_back(bb);
	}

	//return numver of bbs
	return bbs.size();
}

cv::Point BoundingBoxer::GetLargestBox(std::vector<std::vector<cv::Point>> bbs) {
	int largestWidth = 0;
	int largestHeight = 0;
	for(auto bb : bbs) {
		int width = bb[1].x - bb[0].x;
		int height = bb[1].y - bb[0].y;

		if (width > largestWidth)
			largestWidth = width;

		if (height > largestHeight)
			largestHeight = height;
	}

	return cv::Point(largestWidth, largestHeight);
}

void BoundingBoxer::DrawBoundingBoxes(cv::Mat& image, std::vector<std::vector<cv::Point>> bbs, int offset) {
	for (std::vector<cv::Point> v : bbs) {
		cv::rectangle(image, v[0] + cv::Point(offset, offset), v[1] + cv::Point(offset, offset), cv::Scalar(255, 1, 255));
	}
}

void BoundingBoxer::CutAllBoxes(const cv::Mat & image, std::vector<std::vector<cv::Point>> bbs, std::string filename, int offset) {
	for (int i = 0; i < bbs.size(); i++) {
		cv::Point diff = bbs[i][1] + cv::Point(offset*2, offset*2) - bbs[i][0];
		cv::Mat cuttedImage = image(cv::Rect(bbs[i][0].x - offset, bbs[i][0].y - offset, diff.x, diff.y));
		cv::imwrite("img/" + filename + "_" + std::to_string(i) + ".jpg", cuttedImage);
	}
}
