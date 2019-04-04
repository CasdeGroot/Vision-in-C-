#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <string>
#include "avansvisionlib.h"

void ManualFindContours(cv::Mat firstLetter16S);
void ShowContours(std::vector<Point2d*> contours);
void ShowContours(std::vector < std::vector<cv::Point>> contours);

int main(int argc, char *argv[])
{
	std::string imagePath = "Images\\rummikub0.jpg";

	//Display given imagepath to user
	std::cout << "Trying to load image from file " << imagePath << std::endl;

	// Load image from file
	Mat image;
	image = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);
	if (!image.data)
	{
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	else
	{
		std::cout << "File succesfully loaded!" << std::endl;
	}

	//Show loaded image to user
	cv::imshow("Original", image);

	//Convert image to grayscale image
	cv::Mat grayImage;

	cv::cvtColor(image, grayImage, CV_BGR2GRAY);
	cv::imshow("Grayscale", grayImage);

	//Create region of interest
	cv::Mat grayImageROI = grayImage(cv::Rect(34, 28, 976, 82));
	cv::imshow("Region of interest", grayImageROI);

	//Threshold image
	cv::Mat binaryImage;
	cv::threshold(grayImageROI, binaryImage, 165, 1, CV_THRESH_BINARY_INV);

	//show16SImageStretch(binary16S, "Binary");

	//Get first letter to test boundary algorithm without interference
	cv::Mat firstLetterImage = binaryImage(cv::Rect(0, 0, 82, 82));

	cv::Mat firstLetter16S;
	firstLetterImage.convertTo(firstLetter16S, CV_16S);
	show16SImageStretch(firstLetter16S, "FirstLetter");

	ManualFindContours(firstLetter16S);

	std::vector<std::vector<cv::Point>> chainCode;
	cv::findContours(firstLetterImage, chainCode, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	CvChain* chain = 0;
	CvMemStorage* storage = 0;
	storage = cvCreateMemStorage(0);
	cvFindContours(&IplImage(firstLetterImage), storage, (CvSeq**)(&chain), sizeof(*chain), CV_RETR_EXTERNAL, CV_CHAIN_CODE);

	for (; chain != NULL; chain = (CvChain*)chain->h_next)
	{
		//chain=(CvChain*)chain ->h_next; 
		//if(chain==NULL){break;}
		CvSeqReader reader;
		int i, total = chain->total;
		cvStartReadSeq((CvSeq*)chain, &reader, 0);
		printf("--------------------chain\n");

		for (i = 0; i<total; i++)
		{
			char code;
			CV_READ_SEQ_ELEM(code, reader);
			printf("%d", code);
		}
	}

	ShowContours(chainCode);


	cv::waitKey();
	std::cin.get();
	return 0;
}

void ManualFindContours(cv::Mat firstLetter16S)
{
	//Find first nonzero pixel
	bool startingPointFound = false;
	cv::Point2d* startingPoint = new Point2d(0, 0);
	for (int y = 0; y < firstLetter16S.rows; y++)
	{
		for (int x = 0; x < firstLetter16S.cols; x++)
		{
			ushort pixelData = firstLetter16S.at<ushort>(y, x);

			if (pixelData != 0 && !startingPointFound)
			{
				startingPointFound = true;
				startingPoint = new Point2d(x, y);
				//std::cout << "SP:";
			}

			//std::cout << pixelData << ",";
		}
		//std::cout << std::endl;
	}

	std::cout << "Starting point found: " << startingPoint->x << ", " << startingPoint->y << std::endl;

	//All possible offsets around a pixel
	int offsets[8][2] = {
		{ -1, 0 },
	{ -1, -1 },
	{ 0, -1 },
	{ 1, -1 },
	{ 1, 0 },
	{ 1, 1 },
	{ 0, 1 },
	{ -1, 1 }
	};
	//Amount of offsets that are possible around a pixel
	const int POSSIBLEOFFSETS = 8;

	//Keeps track of the amount of points we have found so far
	int totalPointSoFar = 1;

	//Keeps track of the current offset we are checking
	int currentOffsetIndex = 0;
	int currentOffset[2] = { -1,0 };

	//Keeps track of the offset at which we last found a neighbouring pixel
	int lastOffsetIndex = 4;
	int lastOffset[2] = { 1,0 };

	//Current pixel we are searching around
	Point2d* currentPoint = startingPoint;
	Point2d* lastPoint = startingPoint;

	//List with boundary points that have been found so far
	std::vector<Point2d*> boundaryPoints;
	boundaryPoints.push_back(startingPoint);

	bool startingPointEqualToEndPoint = false;
	while (!startingPointEqualToEndPoint)
	{
		//Loop through all 8 possible offsets and check for pixels
		for (int i = 0; i < POSSIBLEOFFSETS; i++)
		{
			//First check if we are not back at the startingpoint
			if (currentPoint->x == startingPoint->x && currentPoint->y == startingPoint->y && totalPointSoFar > 1)
			{
				startingPointEqualToEndPoint = true;
				break;
			}

			//Set offset equal to offset opposite of where we found a pixel last time
			currentOffsetIndex = (lastOffsetIndex + i + 5) % POSSIBLEOFFSETS;

			//Get x and y coordinate of current offset
			currentOffset[0] = offsets[currentOffsetIndex][0];
			currentOffset[1] = offsets[currentOffsetIndex][1];

			//Calculate x and y positions of pixel we are going to check for neighbouring pixel
			int searchX = currentPoint->x + currentOffset[0];
			int searchY = currentPoint->y + currentOffset[1];

			//Get pixel value from neighbouring pixel
			ushort pixelValue = firstLetter16S.at<ushort>(searchY, searchX);

			//If the neighbouring pixel is a boundary
			if (pixelValue != 0)
			{
				totalPointSoFar++;

				lastPoint = currentPoint;
				currentPoint = new Point2d(searchX, searchY);
				boundaryPoints.push_back(currentPoint);

				lastOffsetIndex = currentOffsetIndex;
				lastOffset[0] = currentOffset[0];
				lastOffset[1] = currentOffset[1];

				break;
			}
		}
	}

	ShowContours(boundaryPoints);
}

void ShowContours(std::vector<Point2d*> contours)
{
	cv::Mat customImage(82, 82, CV_8UC1, cv::Scalar(0));
	for (Point2d* point : contours)
	{
		int xData = point->x;
		int yData = point->y;

		customImage.at<uchar>(yData, xData) = 255;
	}

	cv::imshow("BoundaryOperation", customImage);
}

void ShowContours(std::vector<std::vector<cv::Point>> allContours)
{
	cv::Mat customImage(82, 82, CV_8UC1, cv::Scalar(0));
	for (std::vector < cv::Point> imageContours : allContours)
	{
		for (cv::Point point : imageContours)
		{
			int xData = point.x;
			int yData = point.y;

			customImage.at<uchar>(yData, xData) = 255;
		}
	}

	cv::imshow("BoundaryOperation", customImage);
}