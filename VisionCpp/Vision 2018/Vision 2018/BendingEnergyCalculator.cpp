#include "BendingEnergyCalculator.h"


const std::vector<cv::Point> chainCodeVectors = {
	cv::Point(1, 0),
	cv::Point(1,-1),
	cv::Point(0,-1),
	cv::Point(-1,-1),
	cv::Point(-1, 0),
	cv::Point(-1, 1),
	cv::Point(1, 0),
	cv::Point(1, 1)
};

float BendingEnergyCalculator::GetBendingEnergy(const std::vector<int> & chainCode) {
	float bendingEnergy = 0;

	cv::Point vector1 = chainCodeVectors[chainCode[0]];
	cv::Point vector2 = chainCodeVectors[chainCode[1]];

	bendingEnergy += Utilities::GetAngleBetweenVectors(vector1, vector2);

	for (int i = 1; i < chainCode.size() - 1; i++) {
		vector1 = chainCodeVectors[chainCode[i]];
		vector2 = chainCodeVectors[chainCode[i + 1]];
		bendingEnergy += Utilities::GetAngleBetweenVectors(vector1, vector2);
	}

	return bendingEnergy;
}

void BendingEnergyCalculator::GetAllBendingEnergies(const std::vector<std::vector<int>> & chainCodes, std::vector<float> & bendingEnergies) {
	for (std::vector<int> chainCode : chainCodes) {
		float bendingEnergy = GetBendingEnergy(chainCode);
		bendingEnergies.push_back(bendingEnergy);
	}
}

void BendingEnergyCalculator::TestBendingEnergy(const cv::Mat & img, std::vector<float> & bendingEnergies) {
	std::vector<std::vector<int>> chainCodes = std::vector<std::vector<int>>();
	Utilities::GetChainCodes(img, chainCodes);
	GetAllBendingEnergies(chainCodes, bendingEnergies);

	//for (float be : bendingEnergies) {
	//	//std::cout << "bending energy of object " << be << std::endl;
	//}
}
