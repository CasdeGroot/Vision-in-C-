// Demo: Training of a Neural Network / Back-Propagation algorithm 
// Jan Oostindie, Avans Hogeschool, dd 6-12-2016
// email: jac.oostindie@avans.nl

//#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <iomanip>
#include "../avansvisionlib/avansvisionlib.h" // versie 2.0 (!)
#include "demoBPN.h";

using namespace cv;
using namespace std;

// Maximale fout die toegestaan wordt in de output voor de training input
static const double MAX_OUTPUT_ERROR = 1E-10;
// maximaal aantal runs dat uitgevoerd wordt bij het trainen
static const int MAXRUNS = 10000;

int DemoBPN::RunDemo()
{
	// IT, OT: input trainingset, output trainingset
	Mat ITset, OTset;

	// V0, W0   : weightfactor matrices
	// dV0, dW0 : weightfactor correction matrices
	Mat V0, W0, dW0, dV0;

	// default number of hiddenNeurons. The definite number is user input  
	// inputNeurons and outputNeurons are implicitly determined via
	// the trainingset, i.e.: inputNeurons = ITset.cols ; outputNeurons = OTset.cols;
	int hiddenNeurons = 2;

	cout << endl << "Load testtrainingset..." << endl << endl;
	LoadCustomTrainingsSet(ITset, OTset);
	//loadBinaryTrainingSet1(ITset, OTset);

	cout << "Training Input " << endl << endl;
	cout << ITset << endl << endl;
	cout << "Training Output " << endl << endl;
	cout << OTset << endl << endl;

	cout << " ===> BPN format: " << endl <<
		"BPN Inputlayer  = " << ITset.cols << "  neurons" << endl <<
		"BPN Outputlayer = " << OTset.cols << "  neurons" << endl << endl;
	cout << "Please choose a number of hidden neurons: ";
	cin >> hiddenNeurons;
	cout << "Thank you!" << endl << endl << endl;

	cout << "Initialize BPN ..." << endl;
	initializeBPN(ITset.cols, hiddenNeurons, OTset.cols, V0, dV0, W0, dW0);
	//testBPN(ITset, OTset, V0, dV0, W0, dW0);

	cout << "initial values of weight matrices V0 and W0" << endl;
	cout << "*******************************************" << endl;
	cout << V0 << endl << endl << W0 << endl << endl;
	cout << "Press ENTER => ";
	string dummy;
	getline(cin, dummy);
	getline(cin, dummy);

	// IT: current training input of the inputlayer 
	// OT: desired training output of the BPN
	// OH: output of the hiddenlayer
	// OO: output of the outputlayer
	Mat IT, OT, OH, OO;

	// outputError0: error on output for the current input and weighfactors V0, W0
	// outputError1: error on output for the current input and new calculated 
	//               weighfactors, i.e. V1, W1
	double outputError0, outputError1, sumSqrDiffError = MAX_OUTPUT_ERROR + 1;
	Mat V1, W1;

	int runs = 0;
	while ((sumSqrDiffError > MAX_OUTPUT_ERROR) && (runs < MAXRUNS)) {

		sumSqrDiffError = 0;

		for (int inputSetRowNr = 0; inputSetRowNr < ITset.rows; inputSetRowNr++) {

			IT = transpose(getRow(ITset, inputSetRowNr));

			OT = transpose(getRow(OTset, inputSetRowNr));

			calculateOutputHiddenLayer(IT, V0, OH);

			calculateOutputBPN(OH, W0, OO);

			adaptVW(OT, OO, OH, IT, W0, dW0, V0, dV0, W1, V1);

			calculateOutputBPNError(OO, OT, outputError0);

			calculateOutputBPNError(BPN(IT, V1, W1), OT, outputError1);

			sumSqrDiffError += (outputError1 - outputError0) * (outputError1 - outputError0);

			V0 = V1;
			W0 = W1;
		}
		cout << "sumSqrDiffError = " << sumSqrDiffError << endl;
		runs++;
	}

	cout << "BPN Training is ready!" << endl << endl;
	cout << "Runs = " << runs << endl << endl;

	Mat inputVectorTrainingSet, outputVectorTrainingSet, outputVectorBPN;

	// druk voor elke input vector uit de trainingset de output vector uit trainingset af 
	// tezamen met de output vector die het getrainde BPN (zie V0, W0) genereerd bij de 
	// betreffende input vector.
	cout << setw(16) << " " << "Training Input" << setw(2) << "|" << " Expected Output "
		<< setw(1) << "|" << " Output BPN " << setw(6) << "|" << endl << endl;
	for (int row = 0; row < ITset.rows; row++) {

		// haal volgende inputvector op uit de training set
		inputVectorTrainingSet = transpose(getRow(ITset, row));

		// druk de inputvector af in een regel afgesloten met | 
		for (int r = 0; r < inputVectorTrainingSet.rows; r++)
			cout << setw(8) << getEntry(inputVectorTrainingSet, r, 0);
		cout << setw(2) << "|";

		// haal bijbehorende outputvector op uit de training set
		outputVectorTrainingSet = transpose(getRow(OTset, row));

		// druk de outputvector van de training set af in dezelfde regel afgesloten met | 
		for (int r = 0; r < outputVectorTrainingSet.rows; r++)
			cout << setw(8) << round(getEntry(outputVectorTrainingSet, r, 0));
		cout << setw(2) << "|";

		// bepaal de outputvector die het getrainde BPN oplevert 
		// bij de inputvector uit de trainingset  
		outputVectorBPN = BPN(inputVectorTrainingSet, V0, W0);

		// druk de output vector van het BPN af in dezelfde regel afgesloten met |
		for (int r = 0; r < outputVectorBPN.rows; r++)
			cout << setw(8) << round(getEntry(outputVectorBPN, r, 0));
		cout << setw(2) << "|";

		cout << endl;
	}

	cout << endl << endl << "Press ENTER for exit";
	getline(cin, dummy);
	getline(cin, dummy);

	return 0;
}

void DemoBPN::LoadCustomTrainingsSet(cv::Mat & ITset, cv::Mat & OTset) {
	// input of trainingset
	// remark: nummber of columns == number of inputneurons of the BPN
	ITset = (cv::Mat_<double>(12, 7) <<
		1, 0.1, -0.3, 0.5, 0.16, 0.1, 0.2,
		1, 0.25, -0.1, 0.10, 0.25, 0.3, 0.4,
		1, 0.17, 0.34, -0.23, 0.5, 0.5, 0.6,
		1, 0.89, -0.34, -0.56, 0.67, 0.7, 0.8,
		1, 0.2, -0.22, 0.36, 1.0, 0.9, 0.10,
		1, 0.1, -0.34, -0.57, 0.67, 0.11, 0.12,
		1, 0.32, 0.67, 0.34, -1.0, 0.13, 0.14,
		1, 0.87, -0.45, 0.12, 0.82, 0.15, 0.16,
		1, 0.25, 0.17, -0.39, 0.54, 0.17, 0.18,
		1, -0.56, -0.1, 0.8, 0.14, 0.19, 0.2,
		1, 0.9, -0.1, 0.8, 0.4, 0.21, 0.22,
		1, -0.84, 0.32, 0.12, 0.56, 0.23, 0.24);

	// output of trainingset
	// remark: nummber of columns == number of outputneurons of the BPN
	OTset = (cv::Mat_<double>(12, 2) <<
		1, 1,
		0, 0,
		0, 0,
		0, 1,
		1, 1,
		0, 0,
		0, 0,
		1, 0,
		0, 0,
		0, 1,
		1,0,
		0,1);
}

