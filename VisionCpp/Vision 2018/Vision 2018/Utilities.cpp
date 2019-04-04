#include "Utilities.h"

float Utilities::GetAngleBetweenVectors(const cv::Point & vector1, const cv::Point & vector2) {
	float dot = vector1.x * vector2.x + vector1.y * vector2.y;
	float det = vector1.x * vector2.y - vector1.y * vector2.x;
	return atan2(det, dot);
}

cv::Mat Utilities::ThresholdImage(const cv::Mat& src, int threshold, int value, int inverted) {
	cv::Mat dst;
	cv::threshold(src, dst, threshold, value, inverted);
	return dst;
}

cv::Mat Utilities::ErodeImage(const cv::Mat& src, cv::MorphShapes type, const cv::Size& elementSize,
	const cv::Point& middle) {
	const cv::Mat element = getStructuringElement(type, elementSize, middle);

	cv::Mat dst;
	/// Apply the erosion operation
	erode(src, dst, element);

	return dst;
}

void Utilities::GetChainCodes(const cv::Mat & img, std::vector<std::vector<int>> & chainCodes) {
	CvChain* chain = 0;
	CvMemStorage* storage = 0;
	storage = cvCreateMemStorage(0);
	cvFindContours(&IplImage(img), storage, (CvSeq**)(&chain), sizeof(*chain), CV_RETR_EXTERNAL, CV_CHAIN_CODE);

	for (; chain != NULL; chain = (CvChain*)chain->h_next)
	{
		std::vector<int> chainCode = std::vector<int>();
		CvSeqReader reader;
		int i, total = chain->total;
		cvStartReadSeq((CvSeq*)chain, &reader, 0);
		//std::cout << "CHAINCODE: " << std::endl;

		for (i = 0; i<total; i++)
		{
			char code;
			CV_READ_SEQ_ELEM(code, reader);
			//printf("%d", code);
			chainCode.push_back(static_cast<int>(code));
		}

		//std::cout << std::endl;

		if (chainCode.size() > 2) {
			chainCodes.push_back(chainCode);
		}
	}
}

void Utilities::PrintMat(const cv::Mat & mat) {
	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			std::cout << mat.at<double>(i, j) << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}
