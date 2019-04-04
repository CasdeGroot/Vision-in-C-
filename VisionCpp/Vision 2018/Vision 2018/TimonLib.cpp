#include "TimonLib.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "MooreBoundaryTracer.h"
#include "lib/avansvisionlib/avansvisionlib.h"
#include <iostream>
#include <stack>
#include <fstream>


struct TimonLib::BoundingBox 
{
	int minX, maxX, minY, maxY;

	BoundingBox(int miX, int maX, int miY, int maY) { setValues(miX, maX, miY, maY); }
	BoundingBox(const cv::Point &point) { setValues(point.x, point.x, point.y, point.y); }

	void setValues(int miX, int maX, int miY, int maY)
	{
		minX = miX;
		maxX = maX;
		minY = miY;
		maxY = maY;
	}

	void setValues(cv::Point point)	{ setValues(point.x, point.x, point.y, point.y); }

	int GetWidth() { return maxX - minX; }
	int GetHeight() { return maxY - minY; }
};

void TimonLib::LoadMatFromCSV(std::string fileName, cv::Mat &matFile)
{
	std::ifstream inputCSVStream(fileName);
	
	std::vector<std::vector<std::string>> CSVData;

	std::string firstLine;
	std::getline(inputCSVStream, firstLine);
	
	std::string rowOfData;
	while (std::getline(inputCSVStream, rowOfData))
	{
		std::cout << rowOfData << std::endl;
		std::stringstream stringStreamOfData(rowOfData);
		
		std::vector<std::string> CSVRow;
		std::string itemData;
		while (std::getline(stringStreamOfData, itemData, ','))
		{
			CSVRow.push_back(itemData);
		}
		CSVData.push_back(CSVRow);

		stringStreamOfData.flush();
	}

	//File is completely parsed, close filestream
	inputCSVStream.close();

	//TODO Load mat object from the supplied list of vector strings
	std::string secondValue;
}

int TimonLib::PointStackEnclosedPixels(const std::vector<cv::Point> &contourVec, std::vector<cv::Point> &regionPixels)
{
	cv::Mat contourImage = GenerateBinaryImageFromContour(contourVec);

	StackBoundaryFill4(50, 50, 1, 255, contourImage, regionPixels);

	Show8BitBinaryImage(contourImage);

	return regionPixels.size();
}

void TimonLib::StackBoundaryFill4(int x, int y, int boundaryColor, int fillColor, cv::Mat &contourImage, std::vector<cv::Point> &regionPixels)
{
	std::stack<cv::Point> pixels;
	pixels.push(cv::Point(x, y));

	while (!pixels.empty())
	{
		cv::Point currentPoint = pixels.top();
		pixels.pop();
		
		if (contourImage.at<uchar>(currentPoint.y, currentPoint.x) != boundaryColor && contourImage.at<uchar>(currentPoint.y, currentPoint.x) != fillColor)
		{
			contourImage.at<uchar>(currentPoint.y, currentPoint.x) = fillColor;
			regionPixels.push_back(currentPoint);

			pixels.push(cv::Point(currentPoint.x + 1, currentPoint.y));
			pixels.push(cv::Point(currentPoint.x - 1, currentPoint.y));
			pixels.push(cv::Point(currentPoint.x, currentPoint.y + 1));
			pixels.push(cv::Point(currentPoint.x, currentPoint.y - 1));
		}
	}
}

void TimonLib::Show8BitBinaryImage(const cv::Mat &binaryImage)
{
	//Generate new image with same size as binary image default value 0
	cv::Mat convertedBinaryImage(binaryImage.rows, binaryImage.cols, CV_8UC1, cv::Scalar(0));

	for (int y = 0; y < binaryImage.rows; y++)
	{
		for (int x = 0; x < binaryImage.cols; x++)
		{
			if (binaryImage.at<uchar>(y, x) == 1 || binaryImage.at<uchar>(y, x) == 255)
			{
				convertedBinaryImage.at<uchar>(y, x) = 255;
			}
		}
	}

	imshow("Converted Binary Image", convertedBinaryImage);
}

int TimonLib::RecursiveEnclosedPixels(const std::vector<cv::Point> &contourVec, std::vector<cv::Point> &regionPixels)
{
	cv::Mat contourImage = GenerateBinaryImageFromContour(contourVec);

	std::vector<cv::Point> enclosedPixels;
	BoundaryFill4(50, 50, 1, contourImage, enclosedPixels);

	return enclosedPixels.size();
}

cv::Mat TimonLib::GenerateBinaryImageFromContour(const std::vector<cv::Point> &contourVec)
{
	BoundingBox contourBoundingBox = GetBoundingBoxFromContour(contourVec);

	cv::Mat contourImage(contourBoundingBox.GetHeight() + 1, contourBoundingBox.GetWidth() + 1, CV_8UC1, cv::Scalar(0));

	for (cv::Point point : contourVec)
	{
		contourImage.at<uchar>(point.y - contourBoundingBox.minY, point.x - contourBoundingBox.minX) = 1;
	}

	return contourImage;
}

void TimonLib::BoundaryFill4(int x, int y, int boundaryColor, cv::Mat &contourImage, std::vector<cv::Point> &regionPixels)
{
	//If pixel is already filled or pixel is boundary, return
	if (contourImage.at<uchar>(y, x) == boundaryColor || contourImage.at<uchar>(y, x) == 255)
	{
		return;
	}

	//Set current pixel to fill color
	contourImage.at<uchar>(y, x) = 255;
	//Add pixel to filled pixel
	regionPixels.push_back(cv::Point(x, y));

	//Recursively call function for 4 adjacent functions
	BoundaryFill4(x + 1, y, boundaryColor, contourImage, regionPixels);
	BoundaryFill4(x, y + 1, boundaryColor, contourImage, regionPixels);
	BoundaryFill4(x - 1, y, boundaryColor, contourImage, regionPixels);
	BoundaryFill4(x, y - 1, boundaryColor, contourImage, regionPixels);
}

TimonLib::BoundingBox TimonLib::GetBoundingBoxFromContour(const std::vector<cv::Point> contours)
{
	if (contours.size() < 1)
	{
		return BoundingBox(0, 0, 0, 0);
	}

	BoundingBox boundingBox(contours[0]);
	for(cv::Point point : contours)
	{
		if (point.x < boundingBox.minX) { boundingBox.minX = point.x; }
		if (point.x > boundingBox.maxX) { boundingBox.maxX = point.x; }
		if (point.y < boundingBox.minY) { boundingBox.minY = point.y; }
		if (point.y > boundingBox.maxY) { boundingBox.maxY = point.y; }
	}

	return boundingBox;
}