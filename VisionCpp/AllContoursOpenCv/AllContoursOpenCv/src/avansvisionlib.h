// avansvisionlib - Growing Visionlibrary of Avans based on OpenCV 2.4.10 
// Goal: deep understanding of vision algorithms by means of developing own (new) algorithms.
// 
// Copyright Jan Oostindie, basic version 0.2 dd 15-9-2016. Contains basic functions to perform calculations on matrices/images of class Mat.
// Including BLOB labeling functions
//
// Note: Students of Avans are free to use this library in projects and for own vision competence development. Others may ask permission to use it by means 
// of sending an email to Jan Oostindie, i.e. jac.oostindie@avans.nl

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// remark: a functioncall with a Mat-object parameter is a call by reference

/*********************** PROTOTYPES of the function library ************************/

// func: setup a specified entry (i,j) of a matrix m with a specific value 
// pre: (i < m.rows) & (j < m.cols)
void setEntry(Mat m, int i, int j, double value);

// func: get the value of a specified entry (i,j) of a matrix m 
// pre: (i < m.rows) & (j < m.cols)
// return: <return_value> == m(i,j)
double getEntry(Mat m, int i, int j);

// func: calculate product of a row and column of equal length  
// pre: (row.cols == col.rows) & (row.rows == 1) & (col.cols == 1) 
double inproduct(Mat row, Mat col);

// func: prints matrix m in the console
// pre: true
void printMatrix(Mat m);

// func: select and get a row of a matrix m. rowNr contains the row number
// pre: 0 < rowNr < m.rows
// return: <result matrix> contains the selected row
Mat getRow(Mat m, int rowNr);

// func: get a column of a matrix m. colNr contains the column number
// pre: 0 < colNr < m.cols
// return: <result matrix> contains the selected column
Mat getCol(Mat m, int colNr);

// func: multiply two matrices a and b
// pre: (a.cols == b.rows)
// return: <result matrix>.rows == b.rows & <result matrix>.cols == b.cols
Mat multiply(Mat a, Mat b);

// pre: matrices have equal dimensions i.e. (a.cols == b.cols) & (a.rows == b.rows)
// return: <result matrix>(i,j) == a(i,j) + b(i,j) for all (0,0) <= (i,j) < (a.rows,a.cols) 
Mat add(Mat a, Mat b);

// func: transposes a matrix
// return: <return_matrix>(i,j) = m(j,i) & <return_matrix>.rows = m.cols & <return_matrix>.cols = m.rows       
Mat transpose(Mat m);

// func: sets all entries of a matrix to a certain value
// pre: true
void setValue(Mat m, double value);

// func: generates a randomvalue between min and max
// pre: true
double generateRandomValue(double min, double max);


// func: sets all entries of a matrix to a random value
// pre: true
void setRandomValue(Mat m, double min, double max);



/*********************************** Image operaties ****************************************/
// NB images are supposed to have 1 channel (B/W image) and depth 16 bits signed (CV_16S) 
/********************************************************************************************/

// func: setup a specified entry (i,j) of a matrix m with a specific value 
// pre: (i < m.rows) & (j < m.cols)
void setEntryImage(Mat m, int i, int j, _int16 value);

// func: get the value of a specified entry (i,j) of a matrix m 
// pre: (i < m.rows) & (j < m.cols)
// return: <return_value> == m(i,j)
_int16 getEntryImage(Mat m, int i, int j);

// func: calculate product of a row and column of equal length  
// pre: (row.cols == col.rows) & (row.rows == 1) & (col.cols == 1) 
_int16 inproductImage(Mat row, Mat col);

// func: select and get a row of a matrix m. rowNr contains the row number
// pre: 0 < rowNr < m.rows
// return: <result matrix> contains the selected row
Mat getRowImage(Mat m, int rowNr);

// func: get a column of a matrix m. colNr contains the column number
// pre: 0 < colNr < m.cols
// return: <result matrix> contains the selected column
Mat getColImage(Mat m, int colNr);

// func: multiply two matrices a and b
// pre: (a.cols == b.rows)
// return: <result matrix>.rows == b.rows & <result matrix>.cols == b.cols
Mat multiplyImage(Mat a, Mat b);

// pre: matrices have equal dimensions i.e. (a.cols == b.cols) & (a.rows == b.rows)
// return: <result matrix>(i,j) == a(i,j) + b(i,j) for all (0,0) <= (i,j) < (a.rows,a.cols) 
Mat addImage(Mat a, Mat b);


// func: searches the maximum pixel value in the image
// return: maximum pixel value
_int16 maxPixelImage(Mat m);

// func: searches the minimum pixel value in the image
// return: minimum pixel value
_int16 minPixelImage(Mat m);

// func: determines the range of the image, i.e. the minimum 
// and maximum pixel value in the image
// post: range = minPixelValue, maxPixelValue
void getPixelRangeImage(Mat m, _int16 &minPixelValue, _int16 &maxPixelValue);

// func: transform scale the image
// return: maximum pixel value
void stretchImage(Mat m, _int16 minPixelValue, _int16 maxPixelValue);

// func: shows a 16S image on the screen. All values mapped on the interval 0-255 
/// pre: m is a 16S image (depth 16 bits, signed)
void show16SImageStretch(Mat m, string windowTitle = "show16SImageStretch");


// func: shows a 16S image on the screen. All values clipped to the interval 0-255
// i.e. value < 0 => 0; 0 <= value <= 255 => value ; value > 255 => 255 
/// pre: m is a 16S image (depth 16 bits, signed)
void show16SImageClip(Mat m, string windowTitle = "show16SImageClip");



// func: histogram gamma correction
// pre: image has depth 8 bits unsigned and 1 or 3 channels
// post: entry(i,j) = 255*power(entry@pre(i,j)/255)^gamma
void gammaCorrection(Mat image, float gamma);


// func: makes a administration used for labeling blobs.
//       the function adds a edge of 1 pixel wide tot a binary image, all with value 0. 
//       All 1's are made -1. The result is returned.
//       This function is used by function labelBLOBs
// pre : binaryImage has depth 16 bits signed int. Contains only values 0 and 1.
// return_matrix: All "1" are made "-1" meaning value 1 and unvisited.
Mat makeAdmin(Mat binaryImage);


// func: Searches the next blob after position (row,col)
// post: if return_value == 1 then (row,col) contains the position
//       where the next blob starts.	  
// return_value: true =>  blob found    ; starting position is (row,col)
//           	 false => no blob found ; (row, col) == (-1, -1)                   
bool findNextBlob(Mat admin, int & row, int & col);


// func: searches the first 1 when rotating around the pixel (currX,currY), 
//       starting at position 0. Definition of relative positions: 
//          7  0  1
//          6  X  2
//          5  4  3
void findNext1(Mat admin, int & currX, int & currY, int & next1);

// func: gets the entry of a neighbour pixel with relative position nr. 
//       Definition of relative positions nr:
//          7  0  1
//          6  X  2 
//          5  4  3
_int16 getEntryNeighbour(const Mat & admin, int x, int y, int nr);


// func: determines if there are more than 1 adjacent 1's 
bool moreNext1(const Mat & admin, int x, int y);



//  func: labels all pixels of one blob which starts at position (row,col) with blobNr. 
//        This function is used by function labelBLOB's which labels all blobs. 
//  return_value: area of the blob 
//  Evaluation: This function uses a iterative algorithm in which a special labeling technique is
//        is used which gives the opportunity to trace all individiual pixels. This makes it
//        possible for example to save only these pixels on disk or to translate the object in 
//        in the image.
//        The disadvantagae however is that the algorithm is more complicated an maybe a little bit
//        slower than the recursive variant. 
int labelIter(Mat & admin, int row, int col, int blobNr);


//  func: labels all pixels of one blob which starts at position (row,col) with blobNr. 
//  return_value: area of the blob 
//  Evaluation: This function uses a recursive algorithm which has the advantage that it is easy and trasparent.
//        The disadvantagae however is that it claims a lot of spacee on the stack. I.e. every found 
//        pixel results in a function call which in case of large blobs causes a stack overflow. 
int labelRecursive(Mat & admin, int row, int col, int blobNr);

// func: retrieves a labeledImage from the labeling administration
// pre : admin is contains labeled pixels with neighbour number information. 
// post: labeledImage: binary 8-connected pixels with value 1 in binaryImage are 
//       labeled with the number of the object they belong to.
void retrieveLabeledImage(const Mat & admin, Mat & labeledImage);

// func: labeling of all blobs in a binary image
// pre : binaryImage has depth 16 bits signed int. Contains only values 0 and 1.
// post: labeledImage: binary 8-connected pixels with value 1 in binaryImage are 
//       labeled with the number of the object they belong to.
// return_value: the total number of objects.  
int labelBLOBs(Mat binaryImage, Mat & labeledImage);


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
int labelBLOBsInfo(Mat binaryImage, Mat & labeledImage,
	vector<Point2d *> & firstpixelVec, vector<Point2d *> & posVec,
	vector<int> & areaVec,
	int threshAreaMin = 1, int threshAreaMax = INT_MAX);
