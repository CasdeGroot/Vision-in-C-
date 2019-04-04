#include "BoundaryFill.h"
#include <opencv2/core/mat.hpp>
#include <map>
#include <set>
#include <opencv2/highgui/highgui.hpp>
#include "lib/avansvisionlib/avansvisionlib.h"
#include <stack>

#define FILL_COLOR 200;
#define BOUNDARY_COLOR 255;


const cv::Point UP_LEFT =     cv::Point(-1, -1);
const cv::Point UP_MIDDLE =   cv::Point(-1, 0);
const cv::Point UP_RIGHT =    cv::Point(-1, 1);
const cv::Point MIDDLE_LEFT = cv::Point(0, -1);
const cv::Point MIDDLE_RIGHT = cv::Point(0, 1);
const cv::Point DOWN_LEFT =   cv::Point(1, -1);
const cv::Point DOWN_MIDDLE = cv::Point(1, 0);
const cv::Point DOWN_RIGHT =  cv::Point(1, 1);

const std::vector<cv::Point> connection_matrix = {
	UP_LEFT,
	UP_MIDDLE,
	UP_RIGHT,
	MIDDLE_LEFT,
	MIDDLE_RIGHT,
	DOWN_LEFT,
	DOWN_MIDDLE,
	DOWN_RIGHT
};

enum Direction {
	up_left,
	up_middle,
	up_right,
	middle_left,
	middle_right,
	down_left,
	down_middle,
	down_right
};

void BoundaryFill::FillAllBoundaries(cv::Mat & image, const std::vector<std::vector<cv::Point>> & contours, 
										std::vector<std::vector<cv::Point>> & regionPixels) {
	for(std::vector<cv::Point> contour: contours) {
		cv ::Point firstPixel = FindPixelInBoundary(image, contour);
		std::vector<cv::Point> regionPixelVec;
		BoundaryFill8(image, firstPixel.x, firstPixel.y, regionPixelVec);
		regionPixels.push_back(regionPixelVec);
	}
}

cv::Point BoundaryFill::FindPixelInBoundary(const cv::Mat & image, const std::vector<cv::Point>& contour) {
	const std::vector<cv::Point> scanPoints = {
		cv::Point(0,1),
		cv::Point(1,1),
		cv::Point(-1, 1)
	};

	int index = 0;
	bool found = false;
	cv::Point startPixel;

	while(!found) {
		for(cv::Point p: scanPoints) {
			if(image.at<ushort>(contour[index] + p) == 0) {
				startPixel = contour[index] + p;
				found = true;
			}
		}

		if(index < contour.size())
			index++;
		else 
			break;
	}
	return startPixel;
}

void BoundaryFill::BoundaryFill8(cv::Mat& image, int x, int y, std::vector<cv::Point>& regionPixels, bool animate)
{
	std::stack<cv::Point> pixels;
	pixels.push(cv::Point(x, y));

	while (!pixels.empty())
	{
		cv::Point currentPoint = pixels.top();
		pixels.pop();

		if(currentPoint.x >= image.cols || currentPoint.y >= image.rows ) {
			show16SImageStretch(image);
			cv::waitKey(0);
		}

		auto value = image.at<ushort>(cv::Point(currentPoint.x, currentPoint.y));

		if( value != 255 && value != 200)
		{
			image.at<ushort>(cv::Point(currentPoint.x,currentPoint.y)) = 200;

				//show16SImageStretch(image);
				//waitKey(1) && 0xFF;

			regionPixels.push_back(currentPoint);
			pixels.push(cv::Point(currentPoint.x + 1, currentPoint.y));
			pixels.push(cv::Point(currentPoint.x, currentPoint.y + 1));
			pixels.push(cv::Point(currentPoint.x -1, currentPoint.y));
			pixels.push(cv::Point(currentPoint.x, currentPoint.y - 1));
			pixels.push(cv::Point(currentPoint.x - 1, currentPoint.y -1));
			pixels.push(cv::Point(currentPoint.x - 1, currentPoint.y + 1));
			pixels.push(cv::Point(currentPoint.x + 1, currentPoint.y - 1));
			pixels.push(cv::Point(currentPoint.x + 1, currentPoint.y + 1));
		}
	}	
}

void BoundaryFill::AddBoundaryBorder(cv::Mat & image, const std::vector<cv::Point> & boundary, std::vector<cv::Point>& doubleBoundaryVec) {
		// First point of the second boundary is the first pixel of the first boundary with y = firstboundary.y -1
	const cv::Point firstClosePixel = boundary.front() + cv::Point(0, -1);
	doubleBoundaryVec.push_back(firstClosePixel);

	// Go around the boundary and construct double boundary
	// i = 1  -> Skip first pixel
	for (int i = 1; i < boundary.size(); ++i)
	{
		cv::Point boundaryPoint = boundary[i];

		// Get direction between previous and current pixel 
		const cv::Point direction = boundary[i] - boundary[i - 1];

		// Previous doubleBoundaryVec item + direction => current closeBoundaryPoint item
		cv::Point closeBoundaryPoint = doubleBoundaryVec[i - 1] + direction;

		// check if the direction is going up or down.
		if (direction.x == 0 && direction.y == 1)
		{
			closeBoundaryPoint = boundary[i - 1] + cv::Point(1, direction.y);
			doubleBoundaryVec.push_back(closeBoundaryPoint);
		}
		else if (direction.x == 0 && direction.y == -1)
		{
			closeBoundaryPoint = boundary[i - 1] + cv::Point(-1, direction.y);
			doubleBoundaryVec.push_back(closeBoundaryPoint);
		}
		else
		{
			doubleBoundaryVec.push_back(closeBoundaryPoint);
		}

		// Only change pixel if its not a border
		if (image.at<uchar>(closeBoundaryPoint) != 255)
		{
			image.at<uchar>(closeBoundaryPoint) = 255;
		}
	}
	image.at<uchar>(boundary.front() + cv::Point(-1, 0)) = 255;
}
