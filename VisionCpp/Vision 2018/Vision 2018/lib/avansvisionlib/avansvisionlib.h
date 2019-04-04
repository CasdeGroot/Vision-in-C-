#pragma once

// avansvisionlib - Growing Visionlibrary of Avans based on OpenCV 2.4.10 
// Goal: deep understanding of vision algorithms by means of developing own (new) algorithms.
//       deep understanding of neural networks
// 
// Copyright Jan Oostindie, version 2.0 dd 5-12-2016 (= Neural Network (BPN) added to version 1.0 dd 5-11-2016.) 
//      Contains basic functions to perform calculations on matrices/images of class Mat. Including BLOB labeling functions 
//      Contains a BPN neural network. 
// Note: Students of Avans are free to use this library in projects and for own vision competence development. Others may ask permission to use it by means 
// of sending an email to Jan Oostindie, i.e. jac.oostindie@avans.nl

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <iostream>
#include <string>

// remark: a function call with a Mat-object parameter is a call by reference

/*********************** PROTOTYPES of the function library ************************/

// func: setup a specified entry (i,j) of a matrix m with a specific value 
// pre: (i < m.rows) & (j < m.cols)
void setEntry(cv::Mat m, int i, int j, double value);

// func: get the value of a specified entry (i,j) of a matrix m 
// pre: (i < m.rows) & (j < m.cols)
// return: <return_value> == m(i,j)
double getEntry(cv::Mat m, int i, int j);

// func: calculate product of a row and column of equal length  
// pre: (row.cols == col.rows) & (row.rows == 1) & (col.cols == 1) 
double inproduct(cv::Mat row, cv::Mat col);

// func: prints matrix m in the console
// pre: true
void printMatrix(cv::Mat m);

// func: select and get a row of a matrix m. rowNr contains the row number
// pre: 0 < rowNr < m.rows
// return: <result matrix> contains the selected row
cv::Mat getRow(cv::Mat m, int rowNr);

// func: get a column of a matrix m. colNr contains the column number
// pre: 0 < colNr < m.cols
// return: <result matrix> contains the selected column
cv::Mat getCol(cv::Mat m, int colNr);

// func: multiply two matrices a and b
// pre: (a.cols == b.rows)
// return: <result matrix>.rows == b.rows & <result matrix>.cols == b.cols
cv::Mat multiply(cv::Mat a, cv::Mat b);

// pre: matrices have equal dimensions i.e. (a.cols == b.cols) & (a.rows == b.rows)
// return: <result matrix>(i,j) == a(i,j) + b(i,j) for all (0,0) <= (i,j) < (a.rows,a.cols) 
cv::Mat add(cv::Mat a, cv::Mat b);

// func: transposes a matrix
// return: <return_matrix>(i,j) = m(j,i) & <return_matrix>.rows = m.cols & <return_matrix>.cols = m.rows       
cv::Mat transpose(cv::Mat m);

// func: sets all entries of a matrix to a certain value
// pre: true
void setValue(cv::Mat m, double value);

// func: generates a randomvalue between min and max
// pre: true
double generateRandomValue(double min, double max);


// func: sets all entries of a matrix to a random value
// pre: true
void setRandomValue(cv::Mat m, double min, double max);



/*********************************** Image operaties ****************************************/
// NB images are supposed to have 1 channel (B/W image) and depth 16 bits signed (CV_16S) 
/********************************************************************************************/

// func: setup a specified entry (i,j) of a matrix m with a specific value 
// pre: (i < m.rows) & (j < m.cols)
void setEntryImage(cv::Mat m, int i, int j, _int16 value);

// func: get the value of a specified entry (i,j) of a matrix m 
// pre: (i < m.rows) & (j < m.cols)
// return: <return_value> == m(i,j)
_int16 getEntryImage(cv::Mat m, int i, int j);

// func: calculate product of a row and column of equal length  
// pre: (row.cols == col.rows) & (row.rows == 1) & (col.cols == 1) 
_int16 inproductImage(cv::Mat row, cv::Mat col);

// func: select and get a row of a matrix m. rowNr contains the row number
// pre: 0 < rowNr < m.rows
// return: <result matrix> contains the selected row
cv::Mat getRowImage(cv::Mat m, int rowNr);

// func: get a column of a matrix m. colNr contains the column number
// pre: 0 < colNr < m.cols
// return: <result matrix> contains the selected column
cv::Mat getColImage(cv::Mat m, int colNr);

// func: multiply two matrices a and b
// pre: (a.cols == b.rows)
// return: <result matrix>.rows == b.rows & <result matrix>.cols == b.cols
cv::Mat multiplyImage(cv::Mat a, cv::Mat b);

// pre: matrices have equal dimensions i.e. (a.cols == b.cols) & (a.rows == b.rows)
// return: <result matrix>(i,j) == a(i,j) + b(i,j) for all (0,0) <= (i,j) < (a.rows,a.cols) 
cv::Mat addImage(cv::Mat a, cv::Mat b);


// func: searches the maximum pixel value in the image
// return: maximum pixel value
_int16 maxPixelImage(cv::Mat m);

// func: searches the minimum pixel value in the image
// return: minimum pixel value
_int16 minPixelImage(cv::Mat m);

// func: determines the range of the image, i.e. the minimum 
// and maximum pixel value in the image
// post: range = minPixelValue, maxPixelValue
void getPixelRangeImage(cv::Mat m, _int16 &minPixelValue, _int16 &maxPixelValue);

// func: transform scale the image
// return: maximum pixel value
void stretchImage(cv::Mat m, _int16 minPixelValue, _int16 maxPixelValue);

// func: shows a 16S image on the screen. All values mapped on the interval 0-255 
/// pre: m is a 16S image (depth 16 bits, signed)
void show16SImageStretch(cv::Mat m, std::string windowTitle = "show16SImageStretch");


// func: shows a 16S image on the screen. All values clipped to the interval 0-255
// i.e. value < 0 => 0; 0 <= value <= 255 => value ; value > 255 => 255 
/// pre: m is a 16S image (depth 16 bits, signed)
void show16SImageClip(cv::Mat m, std::string windowTitle = "show16SImageClip");



// func: histogram gamma correction
// pre: image has depth 8 bits unsigned and 1 or 3 channels
// post: entry(i,j) = 255*power(entry@pre(i,j)/255)^gamma
void gammaCorrection(cv::Mat image, float gamma);


// func: makes a administration used for labeling blobs.
//       the function adds a edge of 1 pixel wide tot a binary image, all with value 0. 
//       All 1's are made -1. The result is returned.
//       This function is used by function labelBLOBs
// pre : binaryImage has depth 16 bits signed int. Contains only values 0 and 1.
// return_matrix: All "1" are made "-1" meaning value 1 and unvisited.
cv::Mat makeAdmin(cv::Mat binaryImage);


// func: Searches the next blob after position (row,col)
// post: if return_value == 1 then (row,col) contains the position
//       where the next blob starts.	  
// return_value: true =>  blob found    ; starting position is (row,col)
//           	 false => no blob found ; (row, col) == (-1, -1)                   
bool findNextBlob(cv::Mat admin, int & row, int & col);


// func: searches the first 1 when rotating around the pixel (currX,currY), 
//       starting at position 0. Definition of relative positions: 
//          7  0  1
//          6  X  2
//          5  4  3
void findNext1(cv::Mat admin, int & currX, int & currY, int & next1);

// func: gets the entry of a neighbour pixel with relative position nr. 
//       Definition of relative positions nr:
//          7  0  1
//          6  X  2 
//          5  4  3
_int16 getEntryNeighbour(const cv::Mat & admin, int x, int y, int nr);


// func: determines if there are more than 1 adjacent 1's 
bool moreNext1(const cv::Mat & admin, int x, int y);



//  func: labels all pixels of one blob which starts at position (row,col) with blobNr. 
//        This function is used by function labelBLOB's which labels all blobs. 
//  return_value: area of the blob 
//  Evaluation: This function uses a iterative algorithm in which a special labeling technique is
//        is used which gives the opportunity to trace all individiual pixels. This makes it
//        possible for example to save only these pixels on disk or to translate the object in 
//        in the image.
//        The disadvantagae however is that the algorithm is more complicated an maybe a little bit
//        slower than the recursive variant. 
int labelIter(cv::Mat & admin, int row, int col, int blobNr);


//  func: labels all pixels of one blob which starts at position (row,col) with blobNr. 
//  return_value: area of the blob 
//  Evaluation: This function uses a recursive algorithm which has the advantage that it is easy and trasparent.
//        The disadvantagae however is that it claims a lot of spacee on the stack. I.e. every found 
//        pixel results in a function call which in case of large blobs causes a stack overflow. 
int labelRecursive(cv::Mat & admin, int row, int col, int blobNr);

// func: retrieves a labeledImage from the labeling administration
// pre : admin is contains labeled pixels with neighbour number information. 
// post: labeledImage: binary 8-connected pixels with value 1 in binaryImage are 
//       labeled with the number of the object they belong to.
void retrieveLabeledImage(const cv::Mat & admin, cv::Mat & labeledImage);

// func: labeling of all blobs in a binary image
// pre : binaryImage has depth 16 bits signed int. Contains only values 0 and 1.
// post: labeledImage: binary 8-connected pixels with value 1 in binaryImage are 
//       labeled with the number of the object they belong to.
// return_value: the total number of objects.  
int labelBLOBs(cv::Mat binaryImage, cv::Mat & labeledImage);


// func: labeling of all blobs in a binary image with a area in [threshAreaMin,threshAreaMax]. Default
//       threshold is [1,INT_MAX]. Alle gathered data during the labeling proces is returned, 
//       i.e. the positions of the firstpixel of each blob, the position of the blobs (i.e. the
//       centres of gravity) and the area's of all blobs.
// pre : binaryImage has depth 16 bits signed int. Contains only values 0 and 1.
// post: labeledImage: binary 8-connected pixels with value 1 in binaryImage are 
//       labeled with the number of the object they belong to.
//       areaVec: contains all area's of the blobs. The index corresponds to the number
//       of the blobs. Index 0 has no meaning.
// return_value: the total number of objects.  
int labelBLOBsInfo(cv::Mat binaryImage, cv::Mat & labeledImage,
	std::vector<cv::Point2d *> & firstpixelVec, std::vector<cv::Point2d *> & posVec,
	std::vector<int> & areaVec,
	int threshAreaMin = 1, int threshAreaMax = INT_MAX);


/*****************************************************************************************************************************************************/
/*BEGIN********************************************** BACK PROPAGATION NEURAL NETWORK ****************************************************************/
/*****************************************************************************************************************************************************/

// func: loads an example of a training set
// pre: true
// post: ITset input training set. Each row contains a number of features.
//      OTset output training set. Each row contains the expected output belonging to the corresponding row of features in the input training set.
//
// TRAININGSET:  I0 because of bias V0 
//
// setnr     I0     I1     I2    I3    I4    O1   O2
//   1	     1.0    0.4   -0.7   0.1   0.71  0.0  0.0
//   2       1.0    0.3   -0.5   0.05  0.34  0.0  0.0
//   3       1.0    0.6    0.1   0.3   0.12  0.0  1.0
//   4       1.0    0.2    0.4   0.25  0.34  0.0  1.0
//   5		 1.0   -0.2    0.12  0.56  1.0   1.0  0.0
//   6		 1.0	0.1   -0.34  0.12  0.56  1.0  0.0
//   7		 1.0   -0.6    0.12  0.56  1.0   1.0  1.0
//   8		 1.0	0.56  -0.2   0.12  0.56  1.0  1.0
void loadTrainingSet1(cv::Mat & ITset, cv::Mat & OTset);


// func: loads an example of a training set in which only binary numbers are used.
// pre: true
// post: ITset input training set. Each row contains a number of binary numbers.
//       OTset output training set. Each row contains the expected output belonging to the corresponding row of binary numbers in the input training set.
//
// TRAININGSET binary function O1 = (I1 OR I2) AND I3 
// without bias
// setnr    I1   I2    I3   O1   
//   1	     0    0    0    0 	
//   2       0    0    1    0 
//   3       0    1    0    0              
//   4       0    1    1    1
//   5	     1    0    0    0 	
//   6       1    0    1    1 
//   7       1    1    0    0
//   8       1    1    1    1
void loadBinaryTrainingSet1(cv::Mat & ITset, cv::Mat & OTset);


// func: Initialization of the (1) weigthmatrices V0 and W0 and (2) of the delta matrices dV0 and dW0. 
// pre: inputNeurons, hiddenNeurons and outputNeurons define the Neural Network. 
//      (from these numbers the dimensions of the weightmatrices can be determined)
// post: V0 and W0 have random values between 0.1 and 0.9
void initializeBPN(int inputNeurons, int hiddenNeurons, int outputNeurons,
	cv::Mat & V0, cv::Mat & dV0, cv::Mat & W0, cv::Mat & dW0);


// Test of a BPN with all values defined explicitly. 
// pre: true
// post: IT is the input training set ; OT is the corresponding output training set. ; V0, W0 are the weight matrices of a BPN with 1 hidden layer;
//       dV0, dW0 are the initial delta matrices of the weight factor matrices.
void testBPN(cv::Mat & IT, cv::Mat & OT, cv::Mat & V0, cv::Mat & dV0, cv::Mat & W0, cv::Mat & dW0);

// func: Given an inputvector of the inputlayer and a weightmatrix V calculates the outputvector of the hiddenlayer
// pre: II is input of the inputlayer. V = matrix with weightfactors between inputlayer and the hiddenlayer.
// post: OH is the outputvector of the hidden layer
void calculateOutputHiddenLayer(cv::Mat II, cv::Mat V, cv::Mat & OH);


// func: Given the outputvector of the hiddenlayer and a weigthmatrix W calculates the outputvector of the outputlayer
// pre: OH is the outputvector of the hiddenlayer. W = matrix with weightfactors between hiddenlayer and the outputlayer.
// post: OO is the outputvector of the output layer
void calculateOutputBPN(cv::Mat OH, cv::Mat W, cv::Mat & OO);


// func: Calculates the total error Error = 1/2*Sigma(OTi-OOi)^2. 
//       OTi is the expected output according to the trainingvector i
//       OOi is the calculated output from the current neural network of the traininngvector i
// pre: OO is the outputvector of the outputlayer. OT is the expected outputvector from the trainingset 
// post: OO is the outputvector of the output layer
void calculateOutputBPNError(cv::Mat OO, cv::Mat OT, double & outputError);


// func: calculates the updates of the weight factor matrices V0 and W0 on basics of the calculated output matrix and the expected output matrix.
//       A back propagation algorithm is used.
// pre: OT is the expected outputvector from the trainingset ; OO is the calculated outputvector of the outputlayer ; 
//      OH is the calculated output of the hiddenlayer ; OI is the output of the inputlayer (normaly equal to the input of the inputlayer)
//      V0 is the weight matrix between the input layer and the hidden layer ; W0 is the weight matrix between the hiddenlayer and the output layer.
//      dV0, dW0 are the correction matrices.
// post: V is the adapted weight matrix between the inputlayer and the hidden layer ; W is the weight matrix between the hiddenlayer and the output layer.
void adaptVW(cv::Mat OT, cv::Mat OO, cv::Mat OH, cv::Mat OI, cv::Mat W0, cv::Mat dW0, cv::Mat V0, cv::Mat dV0, cv::Mat & W, cv::Mat & V,
	double ALPHA = 1.0, double ETHA = 0.6);


// func: given an inputvector calculates the output of a BPN with weigth matrices V and W. 
// pre:  II is the input vector of the BPN ; 
//       V is the weight factor matrix between the input layer and the hidden layer 
//       W is the weight factor matrix between the hidden layer and the output layer 
// return: output vector
cv::Mat BPN(cv::Mat II, cv::Mat V, cv::Mat W);


/*****************************************************************************************************************************************************/
/*END********************************************** BACK PROPAGATION NEURAL NETWORK ******************************************************************/
/*****************************************************************************************************************************************************/


