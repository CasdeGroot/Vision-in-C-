#pragma once
#include <vector>
#include <opencv2/core/mat.hpp>
#include "utilities.h"

class BendingEnergyCalculator
{
public:
	BendingEnergyCalculator() = default;
	~BendingEnergyCalculator() = default;
	static void TestBendingEnergy(const cv::Mat & img, std::vector<float> & bendingEnergies);
	static void GetAllBendingEnergies(const std::vector<std::vector<int>> & chainCodes, std::vector<float> & bendingEnergies);
private: 
	static float GetBendingEnergy(const std::vector<int> & chainCode);
};

