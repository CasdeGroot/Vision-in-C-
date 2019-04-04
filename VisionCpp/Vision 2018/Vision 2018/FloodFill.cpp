#include "FloodFill.h"
#include <stack>
#include "BoundaryFill.h"
#include "lib/avansvisionlib/avansvisionlib.h"

void FloodFill::FloodFill8(cv::Mat& image, int x, int y, std::vector<cv::Point>& regionPixels) {
	std::stack<cv::Point>stack;
	stack.push(cv::Point(x, y));

		while (!stack.empty())
		{
			cv::Point currentPoint = stack.top();
			stack.pop();

			if(currentPoint.x == 393 && currentPoint.y == 251) {
				show16SImageStretch(image, "flood fill");
				cv::waitKey(0);
			}

			if (image.at<ushort>(currentPoint) == 0)
			{
				image.at<ushort>(currentPoint) = 200;
				regionPixels.push_back(currentPoint);
				stack.push(cv::Point(currentPoint.x + 1, currentPoint.y));
				stack.push(cv::Point(currentPoint.x, currentPoint.y + 1));
				stack.push(cv::Point(currentPoint.x - 1, currentPoint.y));
				stack.push(cv::Point(currentPoint.x, currentPoint.y - 1));
				stack.push(cv::Point(currentPoint.x - 1, currentPoint.y - 1));
				stack.push(cv::Point(currentPoint.x - 1, currentPoint.y + 1));
				stack.push(cv::Point(currentPoint.x + 1, currentPoint.y - 1));
				stack.push(cv::Point(currentPoint.x + 1, currentPoint.y + 1));
			}
		}
}

void FloodFill::FillAll(cv::Mat& image, const std::vector<std::vector<cv::Point>>& contours,
	std::vector<std::vector<cv::Point>>& regionPixels) {
	for (std::vector<cv::Point> contour : contours) {
		cv::Point firstPixel = BoundaryFill::FindPixelInBoundary(image, contour);
		std::vector<cv::Point> regionPixelVec;
		FloodFill8(image, firstPixel.x, firstPixel.y, regionPixelVec);
		regionPixels.push_back(regionPixelVec);
	}
}
