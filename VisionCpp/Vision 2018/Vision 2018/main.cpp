#include<opencv2/opencv.hpp>
#include<iostream>
#include "lib/avansvisionlib/avansvisionlib.h";
#include "MooreBoundaryTracer.h"
#include "utilities.h"
#include "BendingEnergyCalculator.h"
#include "BoundingBoxer.h"
#include "TimonLib.h"
#include "BoundaryFill.h"
#include "Test.h"
#include "lib/fase_3/demoBPN.h"
#include "TrainingSetLoader.h"
#include "../../../../../../../../../../../../Program Files (x86)/Windows Kits/10/Include/10.0.17134.0/ucrt/complex.h"

// Maximale fout die toegestaan wordt in de output voor de training input
static const double MAX_OUTPUT_ERROR = 1E-10;
// maximaal aantal runs dat uitgevoerd wordt bij het trainen
static const int MAXRUNS = 10000;

cv::Mat normalizationSet;

void ProcessLetterImage(const cv::Mat & grayImage, cv::Mat & processedImage) {
	//Threshold image
	const cv::Mat threshold = Utilities::ThresholdImage(grayImage, 65, 1, CV_THRESH_BINARY_INV);

	cv::Mat erodeImg = Utilities::ErodeImage(threshold, cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1));

	erodeImg.convertTo(processedImage, CV_16S);
}

void ProcessMonsterImage(const cv::Mat & grayImage, cv::Mat & processedImage) {
	const cv::Mat threshold = Utilities::ThresholdImage(grayImage, 165, 1, CV_THRESH_BINARY_INV);
	threshold.convertTo(processedImage, CV_16S);
}

void TrainSet(cv::Mat & V0, cv::Mat & W0, bool train) {
	cv::Mat ITSet, OTSet, dV0, dW0;

	if(train) 
	{
		TrainingSetLoader::LoadTrainingsSet(ITSet, OTSet, normalizationSet);
	}
	else 
	{
		TrainingSetLoader::ReadFromCSV("traingset1.csv", ITSet, OTSet);
	}

	Utilities::PrintMat(ITSet);

	initializeBPN(ITSet.cols, ITSet.cols * OTSet.cols + 2, OTSet.cols, V0, dV0, W0, dW0);
	//testBPN(ITset, OTset, V0, dV0, W0, dW0);

	std::cout << "initial values of weight matrices V0 and W0" << std::endl;
	std::cout << "*******************************************" << std::endl;
	std::cout << V0 << std::endl << std::endl << W0 << std::endl << std::endl;
	std::cout << "Press ENTER => ";
	std::string dummy;
	std::getline(std::cin, dummy);
	std::getline(std::cin, dummy);

	// IT: current training input of the inputlayer 
	// OT: desired training output of the BPN
	// OH: output of the hiddenlayer
	// OO: output of the outputlayer
	cv::Mat IT, OT, OH, OO;

	// outputError0: error on output for the current input and weighfactors V0, W0
	// outputError1: error on output for the current input and new calculated 
	//               weighfactors, i.e. V1, W1
	double outputError0, outputError1, sumSqrDiffError = MAX_OUTPUT_ERROR + 1;
	cv::Mat V1, W1;

	int runs = 0;
	while ((sumSqrDiffError > MAX_OUTPUT_ERROR) && (runs < MAXRUNS)) {

		sumSqrDiffError = 0;

		for (int inputSetRowNr = 0; inputSetRowNr < ITSet.rows; inputSetRowNr++) {

			IT = transpose(getRow(ITSet, inputSetRowNr));

			OT = transpose(getRow(OTSet, inputSetRowNr));

			calculateOutputHiddenLayer(IT, V0, OH);

			calculateOutputBPN(OH, W0, OO);

			adaptVW(OT, OO, OH, IT, W0, dW0, V0, dV0, W1, V1);

			calculateOutputBPNError(OO, OT, outputError0);

			calculateOutputBPNError(BPN(IT, V1, W1), OT, outputError1);

			sumSqrDiffError += (outputError1 - outputError0) * (outputError1 - outputError0);

			V0 = V1;
			W0 = W1;
		}
		std::cout << "sumSqrDiffError = " << sumSqrDiffError << std::endl;
		runs++;
	}

	std::cout << "BPN Training is ready!" << std::endl << std::endl;
	std::cout << "Runs = " << runs << std::endl << std::endl;

	cv::Mat inputVectorTrainingSet, outputVectorTrainingSet, outputVectorBPN;

	// druk voor elke input vector uit de trainingset de output vector uit trainingset af 
	// tezamen met de output vector die het getrainde BPN (zie V0, W0) genereerd bij de 
	// betreffende input vector.
	std::cout << std::setw(16) << " " << "Training Input" << std::setw(2) << "|" << " Expected Output "
		<< std::setw(1) << "|" << " Output BPN " << std::setw(6) << "|" << std::endl << std::endl;
	for (int row = 0; row < ITSet.rows; row++) {

		// haal volgende inputvector op uit de training set
		inputVectorTrainingSet = transpose(getRow(ITSet, row));

		// druk de inputvector af in een regel afgesloten met | 
		for (int r = 0; r < inputVectorTrainingSet.rows; r++)
			std::cout << std::setw(8) << getEntry(inputVectorTrainingSet, r, 0);
		std::cout << std::setw(2) << "|";

		// haal bijbehorende outputvector op uit de training set
		outputVectorTrainingSet = transpose(getRow(OTSet, row));

		// druk de outputvector van de training set af in dezelfde regel afgesloten met | 
		for (int r = 0; r < outputVectorTrainingSet.rows; r++)
			std::cout << std::setw(8) << round(getEntry(outputVectorTrainingSet, r, 0));
		std::cout << std::setw(2) << "|";

		// bepaal de outputvector die het getrainde BPN oplevert 
		// bij de inputvector uit de trainingset  
		outputVectorBPN = BPN(inputVectorTrainingSet, V0, W0);

		// druk de output vector van het BPN af in dezelfde regel afgesloten met |
		for (int r = 0; r < outputVectorBPN.rows; r++)
			std::cout << std::setw(8) << round(getEntry(outputVectorBPN, r, 0));
		std::cout << std::setw(2) << "|";

		std::cout << std::endl;
	}

	std::cout << std::endl << std::endl << "Press ENTER for exit";
	std::getline(std::cin, dummy);
	std::getline(std::cin, dummy);
}

void CreateEmptyBorder(cv::Mat& image, int borderSize)
{
	cv::Mat borderedImage = cv::Mat::zeros(image.rows + borderSize * 2, image.cols + borderSize * 2, image.type());

	for (int y = 0; y < image.rows; ++y)
	{
		for (int x = 0; x < image.cols; ++x)
		{
			borderedImage.at<ushort>(cv::Point(x + borderSize, y + borderSize)) = image.at<ushort>(cv::Point(x, y));
		}
	}
	image = borderedImage;
}


void CasMain()
{
	cv::Mat V0, W0;
	//TrainSet(V0, W0, false);

	//TrainingSetLoader::saveWeightsToTxt("weight1.txt", V0, W0);
	TrainingSetLoader::ReadSavedWeights("weight1.txt", V0, W0);
	TrainingSetLoader::GetNormalizationSet("traingset1.csv", normalizationSet);

	cv::VideoCapture cap0(0);
	cv::VideoCapture cap1(1);
	cv::Mat edges;
	while(true) {
		cv::Mat frame;
		cap1 >> frame;
		frame = frame(cv::Rect(cv::Point(20,70), cv::Size(350,300)));
		cvtColor(frame, edges, CV_BGR2GRAY);
		cv::threshold(edges, edges, 100, 1, CV_THRESH_BINARY_INV);
		edges.convertTo(edges, CV_16S);
		CreateEmptyBorder(edges, 10);
		cv::Mat boundary;
		std::vector<std::vector<cv::Point>> contours;
		MooreBoundaryTracer::TestAllContours(edges, boundary, contours);
		boundary.convertTo(boundary, CV_16S);

		if(contours.size() <= 10) {
			CreateEmptyBorder(boundary, 20);
			std::vector<std::vector<cv::Point>> bbs;
			BoundingBoxer::AllBoundingBoxes(contours, bbs);
			//BoundingBoxer::DrawBoundingBoxes(boundary, bbs, 20);

			show16SImageStretch(boundary);
			cv::waitKey(100);

			int num = 0;
			for (std::vector<cv::Point> v : bbs) {
				//
				//cv::Point startPoint = v[0] + cv::Point(-10, -10);
				//cv::Point endPoint = v[1] + cv::Point(30, 30);
				//cv::Mat img = boundary(cv::Rect(startPoint, endPoint));
				cv::Mat ITSet;
				TrainingSetLoader::GetFeatures(ITSet, contours, normalizationSet);
				cv::Mat output = BPN(ITSet, V0, W0);
				cv::Point textPoint = v[0] + cv::Point(20, 20) - cv::Point(0, -10);
				cv::putText(boundary,
					TrainingSetLoader::CheckOutput(output),
					textPoint, // Coordinates
					cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
					1.0, // Scale. 2.0 = 2x bigger
					cv::Scalar(255, 255, 255), // BGR Color
					2);
				show16SImageStretch(boundary);
				cv::waitKey(100);
				num++;
			}
		}
	}
}

void TimonMain()
{
	//Test::TestLoadingCSV();
	cv::waitKey(0);
}

void DemoBPNMain() {
	DemoBPN::RunDemo();
}

int main() 
{
	CasMain();
	//TimonMain();
	//DemoBPNMain();
	//cv::waitKey(0);
	return 0;
}
