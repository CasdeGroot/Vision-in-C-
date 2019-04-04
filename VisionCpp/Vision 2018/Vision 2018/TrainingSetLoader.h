#pragma once
#include <opencv2/core/mat.hpp>

class TrainingSetLoader
{
public:
	TrainingSetLoader() = default;
	~TrainingSetLoader() = default;

	static void saveWeightsToTxt(std::string path, const cv::Mat& V0, const cv::Mat& W0);
	static std::string CheckOutput(const cv::Mat& output);
	static void ReadSavedWeights(std::string path, cv::Mat& V0, cv::Mat& W0);
	static void NormalizeFeatureSet(cv::Mat& featureSet, const cv::Mat& normalizationSet);
	static void LoadTrainingsSet(cv::Mat& ITSet, cv::Mat& OTSet, cv::Mat& normalizationSet);
	static void NormalizeSet(cv::Mat& ITSet, cv::Mat& normalizationSet);
	static void GetFeatures(cv::Mat& ITset, const cv::Mat& img);
	static void GetFeatures(cv::Mat& ITset, const std::vector<std::vector<cv::Point>>& contours, cv::Mat normalizationSet);
	static void ConvertToCSV(const std::string& path, const cv::Mat& ITSet, const cv::Mat& OTSet, const cv::Mat& normalizationSet);
	static void GetNormalizationSet(std::string path, cv::Mat& normalizationSet);
	static void ReadFromCSV(const std::string path, cv::Mat& ITSet, cv::Mat& OTSet);
	static void CalculateFeatures(const std::vector<std::vector<cv::Point>>& contours, double& aspectRatio, double& solidity,
	                       double& circularity, double& curvature);
	static void GetTrainingsSet(cv::Mat & ITset, cv::Mat & OTset, const cv::Mat & img, int OTPos);
};

