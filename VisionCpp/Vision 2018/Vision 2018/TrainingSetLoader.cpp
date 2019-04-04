#include "TrainingSetLoader.h"
#include <experimental/filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/shape/shape_transformer.hpp>
#include "lib/avansvisionlib/avansvisionlib.h"
#include "MooreBoundaryTracer.h"
#include "BoundingBoxer.h"
#include "BendingEnergyCalculator.h"
#include <numeric>
#include <fstream>
#include <iostream>

const std::vector<cv::Point> OT_SET {
	cv::Point(0,0),
	cv::Point(1,0),
	cv::Point(0,1),
	cv::Point(1,1)
};

void TrainingSetLoader::saveWeightsToTxt(std::string path, const cv::Mat & V0, const cv::Mat & W0) {
	std::ofstream myfile;
	myfile.open("trainedSets\\" + path);
	myfile << "V0" << std::endl;
	for (int i = 0; i < V0.rows; i++) {
		for(int j = 0; j < V0.cols; j++) {
			myfile << V0.at<double>(i,j) << ',';
		}
		myfile << std::endl;
	}

	myfile << "W0" << std::endl;
	for (int i = 0; i < W0.rows; i++) {
		for (int j = 0; j < W0.cols; j++) {
			myfile << W0.at<double>(i,j) << ',';
		}
		myfile << std::endl;
	}
	myfile.close();
}

std::string TrainingSetLoader::CheckOutput(const cv::Mat& output) {
	if (static_cast<int>(output.at<double>(0,0)) == 0 && static_cast<int>(output.at<double>(1, 0)) == 0)
		return "Heart";

	if (static_cast<int>(output.at<double>(0,0)) == 0 && static_cast<int>(output.at<double>(1, 0)) == 1)
		return "Lightning";

	if (static_cast<int>(output.at<double>(0, 0)) == 1 && static_cast<int>(output.at<double>(1, 0)) == 0)
		return "Spades";

	if (static_cast<int>(output.at<double>(0, 0)) == 1 && static_cast<int>(output.at<double>(1, 0)) == 1)
		return "Square";
}

std::vector<std::string> split(const std::string& str, const std::string& delim)
{
	std::vector<std::string> tokens;
	size_t prev = 0, pos = 0;
	do
	{
		pos = str.find(delim, prev);
		if (pos == std::string::npos) pos = str.length();
		std::string token = str.substr(prev, pos - prev);
		if (!token.empty()) tokens.push_back(token);
		prev = pos + delim.length();
	} while (pos < str.length() && prev < str.length());
	return tokens;
}

void TrainingSetLoader::ReadSavedWeights(std::string path, cv::Mat & V0, cv::Mat & W0) {
	std::ifstream file("trainedSets\\" + path);
	std::string   line;
	int rowNr = 0;

	bool isCheckingW0 = false;
	while (std::getline(file, line))
	{
		if(line[0] == 'W') {
			rowNr = 0;
			isCheckingW0 = true;
		}

		if (line[0] == 'V') {
			rowNr = 0;
			isCheckingW0 = false;
		}

		if(isCheckingW0 && line[0] != 'W') {
			auto values = split(line, ",");
			cv::Mat row = (cv::Mat_<double>(1, values.size()));
			W0.push_back(row);
			for (int i = 0; i < values.size(); i++) {
				double value = atof(values[i].c_str());
				W0.at<double>(rowNr, i) = value;
			}

			rowNr++;
		}

		if(!isCheckingW0 && line[0] != 'V') {
			auto values = split(line, ",");
			cv::Mat row = (cv::Mat_<double>(1, values.size()));
			V0.push_back(row);
			for (int i = 0; i < values.size(); i++) {
				double value = atof(values[i].c_str());
				V0.at<double>(rowNr, i) = value;
			}

			rowNr++;
		}
	}
}

void normalizeValue(double &dataMean, double dataMin, double dataMax) {

	dataMean = (dataMean - dataMin) / (dataMax - dataMin);

}

void TrainingSetLoader::NormalizeFeatureSet(cv::Mat & featureSet, const cv::Mat & normalizationSet) {
	for(int i = 1; i < featureSet.rows; i++) {
		double min = normalizationSet.at<double>(i, 0);
		double max = normalizationSet.at<double>(i, 1);
		double val = featureSet.at<double>(0, i);
		normalizeValue(val, min, max);
		featureSet.at<double>(0, i) = val;
	}
}

void TrainingSetLoader::LoadTrainingsSet(cv::Mat & ITSet, cv::Mat & OTSet, cv::Mat & normalizationSet) {
	std::string path = "img//NeuralTestSets";
	int i = 0;
	for (const auto & p : std::experimental::filesystem::directory_iterator(path)) 
	{
		for (const auto & imgPath : std::experimental::filesystem::directory_iterator(p)) 
		{
			cv::Mat img = cv::imread(imgPath.path().u8string());
			cv::cvtColor(img, img, CV_BGR2GRAY);
			cv::threshold(img, img, 100, 1, CV_THRESH_BINARY_INV);
			GetTrainingsSet(ITSet, OTSet, img, i);
		}
		i++;

		if(i > 3) {
			NormalizeSet(ITSet, normalizationSet);
			ConvertToCSV("traingset1.csv", ITSet, OTSet, normalizationSet);
			return;
		}
	}
}

void findMaxAndMinValueInColumn(cv::Mat & ITSet, int colNr, double & min, double & max) {
	for (int i = 0; i < ITSet.rows; i++) {
		double val = ITSet.at<double>(i, colNr);
		if (val < min) {
			min = val;
		}

		if (val > max) {
			max = val;
		}
	}
}

void TrainingSetLoader::NormalizeSet(cv::Mat & ITSet, cv::Mat & normalizationSet) {
	for(int i = 1; i < ITSet.cols; i++) {
		double min = 10000, max = 0;
		findMaxAndMinValueInColumn(ITSet, i, min, max);
		cv::Mat row = (cv::Mat_<double>(1,2) << min, max);
		normalizationSet.push_back(row);

		for(int j = 0; j < ITSet.rows; j++) {
			double value = ITSet.at<double>(j, i);
			normalizeValue(value, min, max);
			ITSet.at<double>(j, i) = value;
		}
	}
}

double getCurvature(std::vector<cv::Point> const& vecContourPoints, int step)
{
	std::vector< double > vecCurvature(vecContourPoints.size());

	if (vecContourPoints.size() < step)
		return std::accumulate(vecCurvature.begin(), vecCurvature.end(), 0.0) / vecCurvature.size();

	auto frontToBack = vecContourPoints.front() - vecContourPoints.back();
	bool isClosed = ((int)std::max(std::abs(frontToBack.x), std::abs(frontToBack.y))) <= 1;

	cv::Point2f pplus, pminus;
	cv::Point2f f1stDerivative, f2ndDerivative;
	for (int i = 0; i < vecContourPoints.size(); i++)
	{
		const cv::Point2f& pos = vecContourPoints[i];

		int maxStep = step;
		if (!isClosed)
		{
			maxStep = std::min(std::min(step, i), (int)vecContourPoints.size() - 1 - i);
			if (maxStep == 0)
			{
				vecCurvature[i] = std::numeric_limits<double>::infinity();
				continue;
			}
		}


		int iminus = i - maxStep;
		int iplus = i + maxStep;
		pminus = vecContourPoints[iminus < 0 ? iminus + vecContourPoints.size() : iminus];
		pplus = vecContourPoints[iplus >= vecContourPoints.size() ? iplus - vecContourPoints.size() : iplus];


		f1stDerivative.x = (pplus.x - pminus.x) / (iplus - iminus);
		f1stDerivative.y = (pplus.y - pminus.y) / (iplus - iminus);
		f2ndDerivative.x = (pplus.x - 2 * pos.x + pminus.x) / ((iplus - iminus) / 2 * (iplus - iminus) / 2);
		f2ndDerivative.y = (pplus.y - 2 * pos.y + pminus.y) / ((iplus - iminus) / 2 * (iplus - iminus) / 2);

		double curvature2D;
		double divisor = f1stDerivative.x*f1stDerivative.x + f1stDerivative.y*f1stDerivative.y;
		if (std::abs(divisor) > 10e-8)
		{
			curvature2D = std::abs(f2ndDerivative.y*f1stDerivative.x - f2ndDerivative.x*f1stDerivative.y) /
				pow(divisor, 3.0 / 2.0);
		}
		else
		{
			curvature2D = 0;
		}

		vecCurvature[i] = curvature2D;


	}
	return std::accumulate(vecCurvature.begin(), vecCurvature.end(), 0.0) / vecCurvature.size();
}

void TrainingSetLoader::GetFeatures(cv::Mat & ITset, const cv::Mat & img) {
	cv::Mat boundary;
	cv::Mat image16S;
	std::vector<std::vector<cv::Point>> contours;
	img.convertTo(image16S, CV_16S);
	MooreBoundaryTracer::TestAllContours(image16S, boundary, contours);

	double aspectRatio, solidity, circularity, curvature;
	CalculateFeatures(contours, aspectRatio, solidity, circularity, curvature);

	cv::Mat ITRow = (cv::Mat_<double>(1, 5) << 1.0, aspectRatio, solidity, circularity, curvature);
	ITset.push_back(ITRow);
}

void TrainingSetLoader::GetFeatures(cv::Mat & ITset, const std::vector<std::vector<cv::Point>> & contours, cv::Mat normalizationSet) {

	double aspectRatio, solidity, circularity, curvature;
	CalculateFeatures(contours, aspectRatio, solidity, circularity, curvature);

	cv::Mat ITRow = (cv::Mat_<double>(1, 5) << 1.0, aspectRatio, solidity, circularity, curvature);
	ITset.push_back(ITRow);
	NormalizeFeatureSet(ITset, normalizationSet);
}

void TrainingSetLoader::ConvertToCSV(const std::string & path, const cv::Mat & ITSet, const cv::Mat & OTSet, const cv::Mat & normalizationSet) {
	std::ofstream myCSVfile;
	myCSVfile.open("trainedSets\\" + path);
	myCSVfile << "aspectRatio,solidity,circularity,curvature" << std::endl;

	Utilities::PrintMat(OTSet);

	for(int i = 0; i < ITSet.rows; i++) {
		for (int j = 1; j < ITSet.cols; j++) {
			myCSVfile << ITSet.at<double>(i, j) << ',';
		}

		for(int l = 0; l < OTSet.cols; l++) {
			myCSVfile << OTSet.at<double>(i, l) << ',';
		}

		myCSVfile << std::endl;
	}

	myCSVfile << "NormalizationSet" << std::endl;
	for(int i = 0; i < normalizationSet.rows; i++) {
		for(int j = 0; j < normalizationSet.cols; j++) {
			myCSVfile << normalizationSet.at<double>(i, j) << ',';
		}
		myCSVfile << std::endl;
	}

}

void TrainingSetLoader::GetNormalizationSet(std::string path, cv::Mat & normalizationSet) {
	std::ifstream file("trainedSets\\" + path);
	std::string   line;
	bool startAdding = false;
	int rowNr = 0;
	while (std::getline(file, line)) {

		if(startAdding) {
			auto values = split(line, ",");
			normalizationSet.push_back((cv::Mat_<double>(1, values.size())));

			for (int i = 0; i < values.size(); i++) {
				normalizationSet.at<double>(rowNr, i) = atof(values[i].c_str());
			}

			rowNr++;
		}

		if (line == "NormalizationSet") {
			startAdding = true;
		}

	}
}

void TrainingSetLoader::ReadFromCSV(const std::string path, cv::Mat & ITSet, cv::Mat & OTSet) {
	std::ifstream file("trainedSets\\" + path);
	std::string   line;
	int rowNr = 0;
	std::getline(file, line);
	while (std::getline(file, line)) {

		if (line == "NormalizationSet")
			break;

		auto values = split(line, ",");
		ITSet.push_back((cv::Mat_<double>(1, values.size() - 1)));
		ITSet.at<double>(rowNr, 0) = 1;
		OTSet.push_back((cv::Mat_<double>(1, 2)));


		for(int i = 0; i < values.size() - 2; i++) {
			ITSet.at<double>(rowNr, i + 1) = atof(values[i].c_str());
		}

		for(int j = 0; j < 2; j++) {
			OTSet.at<double>(rowNr, j) = atof(values[j + values.size() - 2].c_str());
		}

		rowNr++;
	}
}

void TrainingSetLoader::CalculateFeatures(const std::vector<std::vector<cv::Point>> & contours, double  & aspectRatio, double & solidity, double & circularity, double & curvature) {
	double perimeter = cv::arcLength(contours[0], true);
	double area = cv::contourArea(contours[0]);
	circularity = 4 * CV_PI *(area / (perimeter*perimeter));
	std::vector<cv::Point> approx;

	//calculate if contour is convex


	std::vector<cv::Point> convexHull;
	cv::convexHull(contours[0], convexHull);
	bool isConvex = cv::isContourConvex(convexHull);

	std::vector<int> convexHullI;
	cv::convexHull(contours[0], convexHullI);
	std::vector<cv::Vec4i> defectPoints(convexHullI.size());
	cv::convexityDefects(contours[0], convexHullI, defectPoints);

	auto hull_area = cv::contourArea(convexHull);

	// calculate solidity
	solidity = area / hull_area;


	//aspect ratio 
	cv::Rect boundingBox = cv::boundingRect(contours[0]);
	aspectRatio = (double)boundingBox.width / boundingBox.height;

	curvature = getCurvature(contours[0], 1);
}

void TrainingSetLoader::GetTrainingsSet(cv::Mat & ITset, cv::Mat & OTset, const cv::Mat & img, int OTPos) 
{
	cv::Mat boundary;
	cv::Mat image16S;
	std::vector<std::vector<cv::Point>> contours;
	img.convertTo(image16S, CV_16S);
	MooreBoundaryTracer::TestAllContours(image16S, boundary, contours);

	double aspectRatio, solidity, circularity, curvature;

	CalculateFeatures(contours, aspectRatio, solidity, circularity, curvature);

	cv::Mat ITRow = (cv::Mat_<double>(1, 5) << 1.0, aspectRatio, solidity, circularity, curvature);
	ITset.push_back(ITRow);

	cv::Mat OTRow = (cv::Mat_<double>(1, 2) << OT_SET[OTPos].x, OT_SET[OTPos].y);
	OTset.push_back(OTRow);
}