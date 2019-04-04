// avansvisionlib - Growing Visionlibrary of Avans based on OpenCV 2.4.10 
// Goal: deep understanding of vision algorithms by means of developing own (new) algorithms.
// 
// Copyright Jan Oostindie, basic version 1.0 dd 8-11-2016. Contains basic functions to perform calculations on matrices/images of class Mat.
// Including BLOB labeling functions
//
// Note: Students of Avans are free to use this library in projects and for own vision competence development. Others may ask permission to use it by means 
// of sending an email to Jan Oostindie, i.e. jac.oostindie@avans.nl

#include "avansvisionlib.h"
#include <math.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

// pre: (i < m.rows) & (j < m.cols)
// Mat is call by reference
void setEntry(Mat m, int i, int j, double value) {
	int index = i * m.cols + j;
	double * p = m.ptr<double>(0);
	p[index] = value;
} //  setEntry

// pre: (i < m.rows) & (j < m.cols)
double getEntry(Mat m, int i, int j) {
	int index = i * m.cols + j;
	double * p = m.ptr<double>(0);
	return *(p + index);
} //  getEntry


// func: calculate product of a row and column of equal length  
// pre: (row.cols == col.rows) 
double inproduct(Mat row, Mat col) {
	double * p1 = row.ptr<double>(0);
	double * p2 = col.ptr<double>(0);
	double sumproducts = 0;
	for (int i = 0; i < row.cols; i++)
		sumproducts += p1[i] * p2[i];
	return sumproducts;
}  // inproduct

void printMatrix(Mat m) {
	for (int i = 0; i < m.rows; i++) {
		for (int j = 0; j < m.cols; j++)
			cout << getEntry(m, i, j) << "  ";
		cout << endl;
	}
	cout << endl;
} // printMatrix


Mat getRow(Mat m, int rowNr) {
	Mat result = Mat_<double>(1, m.cols);
	double entry;
	for (int colNr = 0; colNr < m.cols; colNr++) {
		entry = getEntry(m, rowNr, colNr);
		//	cout << " ** getRow ** " << endl;
		//	cout << " entry = " << entry << endl;
		setEntry(result, 0, colNr, entry);
	}
	return result;
} // getRow

Mat getCol(Mat m, int colNr) {
	Mat result = Mat_<double>(m.rows, 1);
	double entry;
	for (int rowNr = 0; rowNr < m.rows; rowNr++) {
		entry = getEntry(m, rowNr, colNr);
		//	cout << " ** getColumn ** " << endl;
		//	cout << " entry = " << entry << endl;
		setEntry(result, rowNr, 0, entry);
	}
	return result;
} // getCol



// pre: (a.cols == b.rows)
Mat multiply(Mat a, Mat b)
{
	Mat result = Mat_<double>(a.rows, b.cols);
	double inprod;
	for (int arow = 0; arow < a.rows; arow++) {
		for (int bcol = 0; bcol < b.cols; bcol++) {
			inprod = inproduct(getRow(a, arow), getCol(b, bcol));
			setEntry(result, arow, bcol, inprod);
		}
	}
	return result;
} // multiply

// pre: matrices have equal dimensions i.e. (a.cols == b.cols) & (a.rows == b.rows) 
Mat add(Mat a, Mat b)
{
	Mat result = Mat_<double>(a.rows, a.cols);
	double entrysum;
	for (int row = 0; row < a.rows; row++) {
		for (int col = 0; col < a.cols; col++) {
			entrysum = getEntry(a, row, col) + getEntry(b, row, col);
			setEntry(result, row, col, entrysum);
		}
	}
	return result;
} // add


Mat transpose(Mat m) {
	Mat result = Mat_<double>(m.cols, m.rows);
	for (int row = 0; row < m.rows; row++)
	for (int col = 0; col < m.cols; col++)
		setEntry(result, col, row, getEntry(m, row, col));
	return result;
} // transpose


// func: sets all entries of a matrix to a value
// pre: true
void setValue(Mat m, double value) {
	for (int row = 0; row < m.rows; row++)
	for (int col = 0; col < m.cols; col++)
		setEntry(m, row, col, value);
} // setValue


double generateRandomValue(double min, double max) {
	int steps = rand() % 100 + 1;
	double dx = (max - min) / 100;
	return min + dx * steps;
} // generateRandomValue


// func: sets all entries of a matrix to a random value in interval [min,max]
// pre: true
void setRandomValue(Mat m, double min, double max) {
	srand(time(NULL));
	for (int row = 0; row < m.rows; row++)
	for (int col = 0; col < m.cols; col++)
		setEntry(m, row, col, generateRandomValue(min, max));
} // randomValue



/*********************************** Image operaties ****************************************/
// NB images are supposed to have 1 channel (B/W image) and depth 16 bits signed (CV_16S) 
/********************************************************************************************/

void setEntryImage(Mat m, int i, int j, _int16 value) {
	int index = i * m.cols + j;
	_int16 * p = m.ptr<_int16>(0);
	p[index] = value;
} //  setEntry

// pre: (i < m.rows) & (j < m.cols)
_int16 getEntryImage(Mat m, int i, int j) {
	int index = i * m.cols + j;
	_int16 * p = m.ptr<_int16>(0);
	return *(p + index);
} //  getEntryImage

// func: calculate product of a row and column of equal length  
// pre: (row.cols == col.rows) 
_int16 inproductImage(Mat row, Mat col) {
	_int16 * p1 = row.ptr<_int16>(0);
	_int16 * p2 = col.ptr<_int16>(0);
	_int16 sumproducts = 0;
	for (int i = 0; i < row.cols; i++)
		sumproducts += p1[i] * p2[i];
	return sumproducts;
}  // inproductImage


Mat getRowImage(Mat m, int rowNr) {
	Mat result = Mat_<_int16>(1, m.cols);
	_int16 entry;
	for (int colNr = 0; colNr < m.cols; colNr++) {
		entry = getEntryImage(m, rowNr, colNr);
		setEntryImage(result, 0, colNr, entry);
	}
	return result;
} // getRow

Mat getColImage(Mat m, int colNr) {
	Mat result = Mat_<_int16>(m.rows, 1);
	_int16 entry;
	for (int rowNr = 0; rowNr < m.rows; rowNr++) {
		entry = getEntryImage(m, rowNr, colNr);
		setEntryImage(result, rowNr, 0, entry);
	}
	return result;
} // getColImage

Mat multiplyImage(Mat a, Mat b)
{
	Mat result = Mat_<_int16>(a.rows, b.cols);
	_int16 inprod;
	for (int arow = 0; arow < a.rows; arow++) {
		for (int bcol = 0; bcol < b.cols; bcol++) {
			inprod = inproductImage(getRow(a, arow), getColImage(b, bcol));
			setEntry(result, arow, bcol, inprod);
		}
	}
	return result;
} // multiplyImage

// pre: matrices have equal dimensions i.e. (a.cols == b.cols) & (a.rows == b.rows) 
Mat addImage(Mat a, Mat b)
{
	Mat result = Mat_<_int16>(a.rows, a.cols);
	_int16 entrysum;
	for (int row = 0; row < a.rows; row++) {
		for (int col = 0; col < a.cols; col++) {
			entrysum = getEntryImage(a, row, col) + getEntryImage(b, row, col);
			setEntryImage(result, row, col, entrysum);
		}
	}
	return result;
} // addImage


// func: searches the maximum pixel value in the image
// return: maximum pixel
_int16 maxPixelImage(Mat m) {
	_int16 max = getEntryImage(m, 0, 0);
	_int16 next;
	for (int row = 0; row < m.rows; row++) {
		for (int col = 0; col < m.cols; col++) {
			next = getEntryImage(m, row, col);
			if (next > max) max = next;
		}
	}
	return max;
} // maxPixelImage

// func: searches the minimum pixel value in the image
// return: minimum pixel value
_int16 minPixelImage(Mat m) {
	_int16 min = getEntryImage(m, 0, 0);
	_int16 next;
	for (int row = 0; row < m.rows; row++) {
		for (int col = 0; col < m.cols; col++) {
			next = getEntryImage(m, row, col);
			if (next < min) min = next;
		}
	}
	return min;
} // minPixelImage

// func: determines the range of the image, i.e. the minimum 
// and maximum pixel value in the image
// post: range = minPixelValue, maxPixelValue
void getPixelRangeImage(Mat m, _int16 &minPixelValue, _int16 &maxPixelValue) {
	_int16 max = getEntryImage(m, 0, 0);
	_int16 min = getEntryImage(m, 0, 0);
	_int16 next;
	for (int row = 0; row < m.rows; row++) {
		for (int col = 0; col < m.cols; col++) {
			next = getEntryImage(m, row, col);
			if (next > max) max = next;
			else
			if (next < min) min = next;
		}
	}
	minPixelValue = min;
	maxPixelValue = max;
} // getPixelRangeImage


// func: stretches the image to a specified range
void stretchImage(Mat m, _int16 minPixelValue, _int16 maxPixelValue) {
	_int16 min, max, oldValue, newValue;
	getPixelRangeImage(m, min, max);
	double scale = maxPixelValue - minPixelValue;
	scale /= (max - min);
	for (int row = 0; row < m.rows; row++) {
		for (int col = 0; col < m.cols; col++) {
			oldValue = getEntryImage(m, row, col);
			newValue = scale * (oldValue - min) + minPixelValue;
			setEntryImage(m, row, col, newValue);
		}
	}
} // stretchImage


// func: shows a 16S image on the screen 
// pre: m is a 16S image (depth 16 bits, signed)
void show16SImageStretch(Mat m, string windowTitle) {
	Mat mCopy;
	m.copyTo(mCopy);
	stretchImage(mCopy, 0, 255);
	mCopy.convertTo(mCopy, CV_8U);
//	namedWindow(windowTitle, CV_WINDOW_AUTOSIZE);
	imshow(windowTitle, mCopy);
	waitKey(0);
} // show16SImage

// func: shows a 16S image on the screen. All values clipped to the interval 0-255
// i.e. value < 0 => 0; 0 <= value <= 255 => value ; value > 255 => 255 
/// pre: m is a 16S image (depth 16 bits, signed)
void show16SImageClip(Mat m, string windowTitle) {
	Mat mCopy;
	m.copyTo(mCopy);
	mCopy.convertTo(mCopy, CV_8U);
//	namedWindow("show16SImageClip", CV_WINDOW_AUTOSIZE);
	imshow(windowTitle, mCopy);
	waitKey(0);
} // show16SImage


// func: histogram gamma correction
// pre: image has depth 8 bits unsigned and 1 or 3 channels
// post: entry(i,j) = 255*power(entry@pre(i,j)/255)^gamma
void gammaCorrection(Mat image, float gamma) {
	unsigned char lut[256];
	for (int i = 0; i < 256; i++) {
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
	}
	//	dst = src.clone();
	const int channels = image.channels();
	switch (channels) {
		case 1: {
					MatIterator_<uchar> it, end;
					for (it = image.begin<uchar>(), end = image.end<uchar>(); it != end; it++)
						*it = lut[(*it)];
					break;
		}
		case 3: {
					MatIterator_<Vec3b> it, end;
					for (it = image.begin<Vec3b>(), end = image.end<Vec3b>(); it != end; it++) {
						(*it)[0] = lut[((*it)[0])];
						(*it)[1] = lut[((*it)[1])];
						(*it)[2] = lut[((*it)[2])];
					}
					break;
		}
	} // switch
} // gammaCorrection


// func: makes a administration used for labeling blobs.
//       the function adds a edge of 1 pixel wide tot a binary image, all with value 0. 
//       All 1's are made -1. The result is returned.
//       This function is used by function labelBLOBs
// pre : binaryImage has depth 16 bits signed int. Contains only values 0 and 1.
// return_matrix: All "1" are made "-1" meaning value 1 and unvisited.
Mat makeAdmin(Mat binaryImage) {
	Mat result = Mat_<_int16>(binaryImage.rows+2, binaryImage.cols+2);

	// eerste rij 0 maken
	for (int col = 0; col < result.cols; col++) 
	   setEntryImage(result,0,col,0);

	// binaryImage copieren naar admin waarbij een 1 steeds wordt omgezet naar -1.
	for (int row = 1; row < (result.rows-1); row++) {

		// 0 vooraan de rij zetten
		setEntryImage(result,row,0,0);

		// rij binaryImage copieren 
		_int16 value;
		for (int col = 1; col < result.cols - 1; col++) {
			value = getEntryImage(binaryImage, row - 1, col - 1);
			if (value == 1) value = -1;
			setEntryImage(result, row, col, value);
		}

		// 0 achteraan de rij zetten
		setEntryImage(result, row, result.cols - 1, 0);
	} // for

	// laatste rij 0 maken
	for (int col = 0; col < result.cols; col++)
		setEntryImage(result, result.rows-1, col, 0);

	return result;
} // makeAdmin


// func: Searches the next blob after position (row,col) i.e. searches
//       the next -1 in admin 
// post: if return_value == 1 then (row,col) contains the position
//       where the next blob starts.	  
// return_value: 1 next blob found ; starting position is (row,col)
//           	 0 no blob found   ; (row, col) == (-1, -1)                   
bool findNextBlob(Mat admin, int & row, int & col) {
	
	bool found = false;

	// zoeken in de huidige rij
	for (int currCol = col; (currCol < (admin.cols - 1)) & !found; currCol++)
		if (getEntryImage(admin, row, currCol) == -1) {
			found = true;
			// row unchanged
			col = currCol;
		}
	
	// zoeken vanaf de volgende rij
	for (int currRow = row+1; (currRow < (admin.rows-1)) &!found; currRow++) 
		for (int currCol = 1; (currCol < (admin.cols-1)) & !found; currCol++)
			if (getEntryImage(admin, currRow, currCol) == -1) {
				found = true;
				row = currRow;
				col = currCol;
			}
	
	if (!found) {
		row = -1;
		col = -1;
	}
	return found;
}; // findNextBlob

// func: gets the entry of a neighbour pixel with relative position nr. 
//       Definition of relative positions nr:
//          7  0  1
//          6  X  2 
//          5  4  3
_int16 getEntryNeighbour(const Mat & admin , int x, int y, int nr) {
	switch (nr) {
	case 0: return getEntryImage(admin, x - 1, y    ); break;
	case 1: return getEntryImage(admin, x - 1, y + 1); break;
	case 2: return getEntryImage(admin, x    , y + 1); break;
	case 3: return getEntryImage(admin, x + 1, y + 1); break;
	case 4: return getEntryImage(admin, x + 1,     y); break;
	case 5: return getEntryImage(admin, x + 1, y - 1); break;
	case 6: return getEntryImage(admin, x,     y - 1); break;
	case 7: return getEntryImage(admin, x - 1, y - 1); break;
	default: cout << "ERROR getEntryNeighbour " << endl;
	}
} // getEntryNeighbour

// func: determines if there are more than 1 adjacent 1's 
bool moreNext1(const Mat & admin, int x, int y) {
	int cnt1 = 0;
	bool more = false;
	for (int nr = 0; (nr <= 7) & !more; nr++) 
		if (getEntryNeighbour(admin, x, y, nr) == -1) {
			cnt1++;
			if (cnt1 > 1) more = true;
		}
	return more;
} // moreNext1

// func: searches the first 1 when rotating around the pixel (currX,currY), 
//       starting at position 0. Definition of relative positions: 
//          7  0  1
//          6  X  2
//          5  4  3
void findNext1(Mat admin, int & currX, int & currY, int & next1) {
	int rotX, rotY;

	rotX = currX - 1 ; rotY = currY    ; //0 
	if (getEntryImage(admin, rotX, rotY) == -1) next1 = 0;
	else {
		rotX = currX - 1; rotY = currY + 1; //1
		if (getEntryImage(admin, rotX, rotY) == -1) next1 = 1;
		else {
			rotX = currX; rotY = currY + 1; //2 
			if (getEntryImage(admin, rotX, rotY) == -1) next1 = 2;
			else {
				rotX = currX + 1; rotY = currY + 1; //3
				if (getEntryImage(admin, rotX, rotY) == -1) next1 = 3;
				else {
					rotX = currX + 1; rotY = currY; //4 
					if (getEntryImage(admin, rotX, rotY) == -1) next1 = 4;
					else {
						rotX = currX + 1; rotY = currY - 1; //5
						if (getEntryImage(admin, rotX, rotY) == -1) next1 = 5;
						else {
							rotX = currX; rotY = currY - 1; //6
							if (getEntryImage(admin, rotX, rotY) == -1) next1 = 6;
							else {
								rotX = currX - 1; rotY = currY - 1; //7
								if (getEntryImage(admin, rotX, rotY) == -1) next1 = 7;
								else next1 = -99;
							} // 6
						} // 5
					} // 4
				} // 3
			} // 2
		} // 1
	} //0
	if (next1 >= 0) {
		currX = rotX; 
		currY = rotY;
	}
} // findNext1


//  func: labels all pixels of one blob which starts at position (row,col) with blobNr. 
//        This function is used by function labelBLOB's which labels all blobs. 
//  return_value: area of the blob 
//  Evaluation: This function uses a iterative algorithm in which a special labeling technique is
//        is used which gives the opportunity to trace all individiual pixels. This makes it
//        possible for example to save only these pixels on disk or to translate the object in 
//        in the image.
//        The disadvantagae however is that the algorithm is more complicated an maybe a little bit
//        slower than the recursive variant. 
int labelIter(Mat & admin, int row, int col, int blobNr) {
	//  Every visited pixel is labeled with:
	//  blobNr*10 + <relative position to the parent >
	//
	//  definition of relative positions 
	//  7  0  1
	//  6  X  2
	//  5  4  3
	//
	//  The first visited pixel, i.e. (row,col), is labeled with:
	//  blobNr * 10 + 8

	int x = row, y = col;
	setEntryImage(admin, x, y, blobNr*10 + 8);

	int next1 = -999;
	int area = 1;

	// flag more is set when any entry in the path has two or more 
	// unvisited neighbours because we visit only one at a time.
	bool allLabeledFlag = true;
	while (allLabeledFlag) {
		allLabeledFlag = false;
		bool pathLabeled = false;
		while (!pathLabeled) {

			if (!allLabeledFlag) allLabeledFlag = moreNext1(admin, x, y);
			findNext1(admin, x, y, next1);

			if (next1 >= 0) {
				setEntryImage(admin, x, y, blobNr*10 + next1);
				area++;
			}
			else {
				//findprevious
				switch (getEntryImage(admin, x, y) % 10) {
				case 0: x += 1; break;
				case 1: x += 1; y -= 1; break;
				case 2: y -= 1; break;
				case 3: x -= 1; y -= 1; break;
				case 4: x -= 1; break;
				case 5: x -= 1; y += 1; break;
				case 6: y += 1; break;
				case 7: x += 1; y += 1; break;
				case 8: pathLabeled = true; break; // currIndex should be 0 now
				default: cout << "Error func labelIter!";
				} // case
			} // else
		}  // while
	} // while (more)
	return area;
} // labelIter


//  func: labels all pixels of one blob which starts at position (topX,topY) with blobNr. 
//        During the labeling proces the centre of gravity is calculated. 
//        This function is used by function labelBLOBInfo 
//  return_value: area of the blob 
//  Evaluation: This function uses a iterative algorithm in which a special labeling technique is
//        is used which gives the opportunity to trace all individiual pixels. This makes it
//        possible for example to save only these pixels on disk or to translate the object in 
//        in the image.
//        The disadvantagae however is that the algorithm is more complicated an maybe a little bit
//        slower than the recursive variant. 
int labelIterInfo(Mat & admin, int topX, int topY, int blobNr,
				  int & xGravity, int & yGravity) {

	//  Every visited pixel is labeled with:
	//  blobNr*10 + <relative position to the parent >
	//
	//  definition of relative positions 
	//  7  0  1
	//  6  X  2
	//  5  4  3
	//
	//  The first visited pixel, i.e. (row,col), is labeled with:
	//  blobNr * 10 + 8

	xGravity = topX;
	yGravity = topY;
	int x = topX, y = topY;
	setEntryImage(admin, topX, topY, blobNr * 10 + 8);

	int next1 = -999;
	int area = 1;

	// allLabeledFlag is set when any entry in the path has two or more 
	// unvisited neighbours because we visit only one at a time.
	// This algorithm good be speeded up by using a stack (future improvement)
	bool allLabeledFlag = true;
	while (allLabeledFlag) {
		allLabeledFlag = false;
		bool pathLabeled = false;
		while (!pathLabeled) {

			if (!allLabeledFlag) allLabeledFlag = moreNext1(admin, x, y);
			findNext1(admin, x, y, next1);

			if (next1 >= 0) {
				setEntryImage(admin, x, y, blobNr * 10 + next1);
				area++;
				xGravity += x;
				yGravity += y;
			}
			else {
				//findprevious
				switch (getEntryImage(admin, x, y) % 10) {
				case 0: x += 1; break;
				case 1: x += 1; y -= 1; break;
				case 2: y -= 1; break;
				case 3: x -= 1; y -= 1; break;
				case 4: x -= 1; break;
				case 5: x -= 1; y += 1; break;
				case 6: y += 1; break;
				case 7: x += 1; y += 1; break;
				case 8: pathLabeled = true; break; // currIndex should be 0 now
				default: cout << "Error func labelIter!";
				} // case
			} // else
		}  // while
	} // while (more)
	xGravity /= area;
	yGravity /= area;
   return area;
} // labelIterInfo



//  func: labels all pixels of one blob which starts at position (row,col) with blobNr. 
//  return_value: area of the blob 
//  Evaluation: This function uses a recursive algorithm which has the advantage that it is easy and trasparent.
//        The disadvantagae however is that it claims a lot of spacee on the stack. I.e. every found 
//        pixel results in a function call which in case of large blobs causes a stack overflow. 
int labelRecursive(Mat & admin, int row, int col, int blobNr) {
	int area = 0;
	// bij waarde -1 is het pixel nog niet bezocht
	if (getEntryImage(admin, row, col) == -1) {
		//cout << "(row,col) = " << "(" << row << "," << col << ")" << endl;
		// pixel labelen met het volgnummer van de blob
		setEntryImage(admin, row, col, blobNr);
		area = 1;

		// alle pixels rondom huidige pixel bezoeken
		// (row-1,col-1) (row-1,col ) (row-1,col+1) 
		// (row  ,col-1) (row,  col ) (row  ,col+1)
		// (row+1,col-1) (row,  col ) (row+1,col+1)
		area += labelRecursive(admin, row - 1, col, blobNr);
		area += labelRecursive(admin, row - 1, col + 1, blobNr);
		area += labelRecursive(admin, row, col + 1, blobNr);
		area += labelRecursive(admin, row + 1, col + 1, blobNr);
		area += labelRecursive(admin, row, col, blobNr);
		area += labelRecursive(admin, row + 1, col - 1, blobNr);
		area += labelRecursive(admin, row, col - 1, blobNr);
		area += labelRecursive(admin, row - 1, col - 1, blobNr);
	}
	return area;
} // label

// func: retrieves a labeledImage from the labeling administration
// pre : admin is contains labeled pixels with neighbour number information. 
// post: labeledImage: binary 8-connected pixels with value 1 in binaryImage are 
//       labeled with the number of the object they belong to.
void retrieveLabeledImage(const Mat & admin, Mat & labeledImage) {
	labeledImage = Mat_<_int16>(admin.rows-2,admin.cols-2);

	for (int row = 1; row < admin.rows-1; row++) {
		for (int col = 1; col < admin.cols-1; col++) {
			setEntryImage(labeledImage, row-1,col-1, 
						  getEntryImage(admin, row, col) / 10);
		}
	}
} // retrieveLabeledImage


// func: labeling of all blobs in a binary image
// pre : binaryImage has depth 16 bits signed int. Contains only values 0 and 1.
// post: labeledImage: binary 8-connected pixels with value 1 in binaryImage are 
//       labeled with the number of the object they belong to.
// return_value: the total number of objects.  
int labelBLOBs(Mat binaryImage, Mat & labeledImage) {

	// admin contains the administration of the recursive process.
	// meaning of the entry values:
	// -1: a "1" which is not visited yet. Changes to 1 when visited.
	//  0: always a "0"
	//  1, 2, 3,... : a "1" which is visited and is labeled with the object number.
	Mat admin = makeAdmin(binaryImage);
	int row = 1;
	int col = 1;

	// init volgnummer
	int blobNr = 0;

	// label alle BLOBs met een volgnummer
	while ((row > 0) & (row < (admin.rows - 1)) &	
		   (col > 0) & (col < (admin.cols - 1)))
		if (findNextBlob(admin, row, col)) labelIter(admin, row, col, ++blobNr);

	retrieveLabeledImage(admin, labeledImage);

	// laatste volgnummer is gelijk aan het aantal gevonden blobs
	return blobNr;
} // labelBLOBs

// func: removes a BLOB from the labeling administration
// pre: (posx,posy) is the position of the BLOB, blobNr the number
//      of the blob to be removed.
void removeBLOB(Mat & admin, int blobNr) {
	_int16 value;
	for (int row = 1; row < admin.rows-2; row++) 
		for (int col = 1; col < admin.cols-2; col++) {
			value = getEntryImage(admin, row, col);
			while (value > 10) value /= 10;
			if (value == blobNr) setEntryImage(admin, row, col, 0);
		}
} // removeBLOB

// func: labeling of all blobs in a binary image with a area in [threhAreaMin,threhAreaMax]. Default
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
	int threshAreaMin, int threshAreaMax) {

	// admin contains the administration of the recursive process.
	// meaning of the entry values:
	// -1: a "1" which is not visited yet. 
	//  0: always a "0"
	//  1, 2, 3,... : a "1" which is visited and is labeled with the object number.
	Mat admin = makeAdmin(binaryImage);
	int row = 1;
	int col = 1;

	// init volgnummer
	int blobNr = 0;
	int area;
	int xGravity, yGravity;
	// label alle BLOBs met een volgnummer
	while ((row > 0) & (row < (admin.rows - 1)) &
		(col > 0) & (col < (admin.cols - 1)))
	
		if (findNextBlob(admin, row, col)) {

			area = labelIterInfo(admin, row, col, ++blobNr, xGravity, yGravity);
			
			if ((area >= threshAreaMin) & (area <= threshAreaMax)) {
				firstpixelVec.push_back(new Point2d(row - 1, col - 1));
				posVec.push_back(new Point2d(xGravity-1, yGravity-1));
				areaVec.push_back(area);
			}
			else removeBLOB(admin, blobNr--);
	}

	retrieveLabeledImage(admin, labeledImage);

	// laatste volgnummer is gelijk aan het aantal gevonden blobs
	return blobNr;
} // labelBLOBsInfo
