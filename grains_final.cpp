#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <sstream>
#include <math.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

int k;
int i = 0, ctr = 0;
void Callbackfunc(int Event, int x, int y, int flags, void* userdata);
Point pt;
Mat src1, src_gray;
Mat im_floodfill_inv;
Mat drawing_clone;


Mat src = imread("grain.jpg"), cloneimg, new_image;
bool mousedown;
Mat masked(src.size(), CV_8UC3, Scalar(255, 255, 255));
Mat masked1, masked3, dst, binary, binary11, src_HSV, frame_threshold;

vector<Point> pts;


int thresh = 100;
int B_thresh = 60;
int max_thresh = 255;
RNG rng(12345);
int areathresh = 0;
int maxarea_thresh;
int c = 0, b, g, r;
double aspect_ratio;
void arearesult(int, void*);
Mat drawing;
Mat final_connected;
Point pt1, pt2;




//------------------------------ BINARY TRACKBAR CALL BACK FUNCTION -------------------------------------------------------------





void ThinSubiteration1(Mat& pSrc, Mat& pDst) {
	int rows = pSrc.rows;
	int cols = pSrc.cols;
	pSrc.copyTo(pDst);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (pSrc.at<float>(i, j) == 1.0f) {
				/// get 8 neighbors
				/// calculate C(p)
				int neighbor0 = (int)pSrc.at<float>(i - 1, j - 1);
				int neighbor1 = (int)pSrc.at<float>(i - 1, j);
				int neighbor2 = (int)pSrc.at<float>(i - 1, j + 1);
				int neighbor3 = (int)pSrc.at<float>(i, j + 1);
				int neighbor4 = (int)pSrc.at<float>(i + 1, j + 1);
				int neighbor5 = (int)pSrc.at<float>(i + 1, j);
				int neighbor6 = (int)pSrc.at<float>(i + 1, j - 1);
				int neighbor7 = (int)pSrc.at<float>(i, j - 1);
				int C = int(~neighbor1 & (neighbor2 | neighbor3)) +
					int(~neighbor3 & (neighbor4 | neighbor5)) +
					int(~neighbor5 & (neighbor6 | neighbor7)) +
					int(~neighbor7 & (neighbor0 | neighbor1));
				if (C == 1) {
					/// calculate N
					int N1 = int(neighbor0 | neighbor1) +
						int(neighbor2 | neighbor3) +
						int(neighbor4 | neighbor5) +
						int(neighbor6 | neighbor7);
					int N2 = int(neighbor1 | neighbor2) +
						int(neighbor3 | neighbor4) +
						int(neighbor5 | neighbor6) +
						int(neighbor7 | neighbor0);
					int N = min(N1, N2);
					if ((N == 2) || (N == 3)) {
						/// calculate criteria 3
						int c3 = (neighbor1 | neighbor2 | ~neighbor4) & neighbor3;
						if (c3 == 0) {
							pDst.at<float>(i, j) = 0.0f;
						}
					}
				}
			}
		}
	}
}

//15
void ThinSubiteration2(Mat& pSrc, Mat& pDst) {
	int rows = pSrc.rows;
	int cols = pSrc.cols;
	pSrc.copyTo(pDst);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (pSrc.at<float>(i, j) == 1.0f) {
				/// get 8 neighbors
				/// calculate C(p)
				int neighbor0 = (int)pSrc.at<float>(i - 1, j - 1);
				int neighbor1 = (int)pSrc.at<float>(i - 1, j);
				int neighbor2 = (int)pSrc.at<float>(i - 1, j + 1);
				int neighbor3 = (int)pSrc.at<float>(i, j + 1);
				int neighbor4 = (int)pSrc.at<float>(i + 1, j + 1);
				int neighbor5 = (int)pSrc.at<float>(i + 1, j);
				int neighbor6 = (int)pSrc.at<float>(i + 1, j - 1);
				int neighbor7 = (int)pSrc.at<float>(i, j - 1);
				int C = int(~neighbor1 & (neighbor2 | neighbor3)) +
					int(~neighbor3 & (neighbor4 | neighbor5)) +
					int(~neighbor5 & (neighbor6 | neighbor7)) +
					int(~neighbor7 & (neighbor0 | neighbor1));
				if (C == 1) {
					/// calculate N
					int N1 = int(neighbor0 | neighbor1) +
						int(neighbor2 | neighbor3) +
						int(neighbor4 | neighbor5) +
						int(neighbor6 | neighbor7);
					int N2 = int(neighbor1 | neighbor2) +
						int(neighbor3 | neighbor4) +
						int(neighbor5 | neighbor6) +
						int(neighbor7 | neighbor0);
					int N = min(N1, N2);
					if ((N == 2) || (N == 3)) {
						int E = (neighbor5 | neighbor6 | ~neighbor0) & neighbor7;
						if (E == 0) {
							pDst.at<float>(i, j) = 0.0f;
						}
					}
				}
			}
		}
	}
}

//349
void normalizeLetter(Mat& inputarray, Mat& outputarray) {
	bool bDone = false;
	int rows = inputarray.rows;
	int cols = inputarray.cols;

	inputarray.convertTo(inputarray, CV_32FC1);

	inputarray.copyTo(outputarray);

	outputarray.convertTo(outputarray, CV_32FC1);

	/// pad source
	Mat p_enlarged_src = Mat(rows + 2, cols + 2, CV_32FC1);
	for (int i = 0; i < (rows + 2); i++) {
		p_enlarged_src.at<float>(i, 0) = 0.0f;
		p_enlarged_src.at<float>(i, cols + 1) = 0.0f;
	}
	for (int j = 0; j < (cols + 2); j++) {
		p_enlarged_src.at<float>(0, j) = 0.0f;
		p_enlarged_src.at<float>(rows + 1, j) = 0.0f;
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (inputarray.at<float>(i, j) >= 20.0f) {
				p_enlarged_src.at<float>(i + 1, j + 1) = 1.0f;
			}
			else
				p_enlarged_src.at<float>(i + 1, j + 1) = 0.0f;
		}
	}

	/// start to thin
	Mat p_thinMat1 = Mat::zeros(rows + 2, cols + 2, CV_32FC1);
	Mat p_thinMat2 = Mat::zeros(rows + 2, cols + 2, CV_32FC1);
	Mat p_cmp = Mat::zeros(rows + 2, cols + 2, CV_8UC1);

	while (bDone != true) {
		/// sub-iteration 1
		ThinSubiteration1(p_enlarged_src, p_thinMat1);
		/// sub-iteration 2
		ThinSubiteration2(p_thinMat1, p_thinMat2);
		/// compare
		compare(p_enlarged_src, p_thinMat2, p_cmp, CMP_EQ);
		/// check
		int num_non_zero = countNonZero(p_cmp);
		if (num_non_zero == (rows + 2) * (cols + 2)) {
			bDone = true;
		}
		/// copy
		p_thinMat2.copyTo(p_enlarged_src);
	}
	// copy result
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			outputarray.at<float>(i, j) = p_enlarged_src.at<float>(i + 1, j + 1);
		}
	}
}








Mat makebinary1(Mat imageBinary, int thresh)

{

	Mat result(imageBinary.size(), CV_8UC1);
	for (int i = 0; i < imageBinary.rows; i++)
	{
		for (int j = 0; j < imageBinary.cols; j++)
		{
			if (imageBinary.at<uchar>(i, j) > thresh)

				result.at<uchar>(i, j) = 0;    //Make pixel black

			else

				result.at<uchar>(i, j) = 255;  //Make pixel white

		}

	}
	return result;
}

Mat makebinary2(Mat imageBinary, int thresh)

{

	Mat result(imageBinary.size(), CV_8UC1);
	for (int i = 0; i < imageBinary.rows; i++)
	{
		for (int j = 0; j < imageBinary.cols; j++)
		{
			if (imageBinary.at<uchar>(i, j) > thresh)

				result.at<uchar>(i, j) = 255;    //Make pixel black

			else

				result.at<uchar>(i, j) = 0;  //Make pixel white

		}

	}
	return result;
}


//---------------------------------------------------MAIN FUNCTION --------------------------------------------------------------


int main(int argc, const char** argv)
{

	/*cout << "Please enter the maximum number of Corners to be Detected";
	cin >> k;*/
	if (src.empty())
	{
		return -1;
	}




	namedWindow("Binary", WINDOW_NORMAL);//namedWindow("contour",WINDOW_NORMAL);namedWindow("contour11",WINDOW_NORMAL);
	resizeWindow("Binary", 600, 600);///resizeWindow("contour",600,600);resizeWindow("contour11",600,600);


	/*Mat src_cleaned;
	fastNlMeansDenoising(src, src_cleaned, 25, 7, 21);*/

	Mat cloneBinary = src.clone();


	createTrackbar("Threshold", "Binary", &thresh, max_thresh);                               //TRACKBAR TO SET THRESHOLD FOR BINARY


	Mat gray;

	while (1)                                                                                //TO GENERATE BINARY IMAGE FROM TRACKBAR
	{
		cvtColor(cloneBinary, gray, COLOR_BGR2GRAY);

		binary = makebinary1(gray, thresh);

		imshow("Binary", binary);

		char a = waitKey(33);

		if (a == 'c' || a == 'C')
		{
			destroyWindow("Binary");
			break;

		}

	}

	//--------------------------------------------------------------------------------------------
	Mat res = Mat::zeros(binary.size(), CV_8UC3);
	normalizeLetter(binary, res);
	//Mat res = Mat::zeros(binary.size(), CV_8UC3);
	normalizeLetter(binary, res);
	Mat final;
	res.convertTo(final, CV_8UC1);

	Mat drawing_new = Mat::zeros(final.size(), CV_8UC3);
	Mat drawing_dilated = Mat::zeros(final.size(), CV_8UC3);
	Mat drawing_eroded = Mat::zeros(final.size(), CV_8UC3);
	vector<vector<Point> > contours_new;
	findContours(final, contours_new, RETR_LIST, CHAIN_APPROX_SIMPLE);

	/*Mat drawingForTest = Mat::zeros(final.size(), CV_8UC1);
	drawContours(drawingForTest, contours_new, -1 , Scalar(255) , 1);
	imshow("NitinPatel", drawingForTest);
	waitKey(0);*/

	//namedWindow("DDDDDDDDDDDD",WINDOW_NORMAL);
	//resizeWindow("DDDDDDDDDDDD",800,800);int c=0;

	

	for (int i = 0; i < contours_new.size(); i++)
	{

		if (contourArea(contours_new[i]) > 20)
		{
			c++;
			drawContours(drawing_new, contours_new, i, Scalar(0, 255, 255), 1);

			//imshow("DDDDDDDDDDDD",drawing_new);//waitKey(0);
		}
	}


	cout << endl << c << endl;
	//namedWindow("DDDD_new",WINDOW_NORMAL);
	//resizeWindow("DDDD_new",600,600);

	//namedWindow("DDDD_eroded",WINDOW_NORMAL);
	//resizeWindow("DDDD_eroded",800,800);


	Mat element1 = getStructuringElement(MORPH_ELLIPSE, Size(19, 19), Point(9, 9));
	Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(19, 19), Point(9, 9));

	dilate(drawing_new, drawing_dilated, element1);
	erode(drawing_dilated, drawing_eroded, element2);
	//imshow("DDDD_new",drawing_dilated);
	//imshow("DDDD_eroded",drawing_eroded);







//-------------------------------------------------------------------------------------------------
	Mat gray_new;

	while (1)                                                                                //TO GENERATE BINARY IMAGE FROM TRACKBAR
	{
		cvtColor(drawing_eroded, gray_new, COLOR_BGR2GRAY);

		binary = makebinary1(gray_new, thresh);

		imshow("Binary", binary);

		char a = waitKey(33);

		if (a == 'c' || a == 'C')
		{
			destroyWindow("Binary");
			break;

		}

	}




	Mat im_th = binary.clone();

	Mat im_floodfill = im_th.clone();
	floodFill(im_floodfill, cv::Point(0, 0), Scalar(255));

	// Invert floodfilled image

	bitwise_not(im_floodfill, im_floodfill_inv);

	// Combine the two images to get the foreground.
	Mat im_out = (im_th | im_floodfill_inv);

	int top = (int)10; int bottom = top;
	int left = (int)10; int right = left;


	Mat kernel = getStructuringElement(MORPH_RECT, Size((2 * 1) + 1, (2 * 1) + 1));


	// Applying dilate on the Image
	erode(im_floodfill, im_floodfill, kernel);

	//Mat canny_output;
	vector<vector<Point> > contours;
	vector<vector<Point> > contours_11;
	vector<Vec4i> hierarchy;



	findContours(im_floodfill, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(-1, -1));
	findContours(im_floodfill, contours_11, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(-1, -1));

	//-------------******************************Nodule Code inserted here****************************-----------

	//vector<vector<Point>>hull_new(contours.size());
	//vector<Point2f>centers_in(contours.size());
	//vector<float>radius_in(contours.size());
	//vector<Point> corners;
	//for (int i = 0; i < contours.size(); i++)
	//{
	//	convexHull(contours_new[i], hull_new[i]);

	//	minEnclosingCircle(contours[i], centers_in[i], radius_in[i]);
	//	double r = (contourArea(contours[i]) / contourArea(hull_new[i]));

	//	if ((arcLength(contours[i], 1) > 100))
	//	{

	//		//drawContours( src, hull, (int)i, Scalar(0,255,255) , 4 );

	//		vector<Point> pts;
	//		vector<Point> pts1;
	//		vector<double> distances;
	//		for (int j = 0;j < contours[i].size();j++)
	//		{
	//			double dis, dis_min = 100000000;
	//			Point P1, P2;
	//			//Point P1 = contours[i][j];

	//			for (int k = 0;k < hull_new[i].size();k++)
	//			{
	//				//P2=P1;
	//				dis = cv::norm(contours[i][j] - hull_new[i][k]);
	//				if (dis <= dis_min)
	//				{
	//					//dis2 = dis1;
	//					dis_min = dis;
	//					P1 = contours[i][j];
	//					P2 = hull_new[i][k];
	//				}
	//			}
	//			pts.push_back(P1);
	//			pts1.push_back(P2);
	//			distances.push_back(dis_min);


	//		}

	//		double dis1 = 0, dis2 = 0;
	//		Point P_new1 = Point(0, 0), P_new2, P_1_hull, P_2_hull;
	//		int T = 0;
	//		for (int j = 0;j < distances.size();j++)
	//		{
	//			//cout<<distances[j]<<endl;
	//			if (distances[j] > dis1)
	//			{
	//				dis1 = distances[j];
	//				P_new1 = pts[j];
	//				P_1_hull = pts1[j];
	//				T = j;
	//				//cout<<P_new1<<"YO"<<endl;
	//			}

	//		}

	//		/*for(int j=0;j<distances.size();j++)
	//		{
	//			if(((j>=T+distances.size()/2)||(j>=T-distances.size()/2))&&(distances[j]>dis2)&&(j!=T))
	//			{
	//				dis2 = distances[j];
	//				P_new2 = pts[j];
	//			}
	//		}*/

	//		int R = 0;
	//		for (int j = distances.size() - 1;j >= 0;j--)
	//		{
	//			//avoiding the points near the previous point
	//			if ((j != T) && (j != T - 1) && (j != T + 1) && (j != T - 2) && (j != T + 2) && (j != T - 3) && (j != T + 3) && (j != T - 4) && (j != T + 4) && (j != T - 5) && (j != T + 5) && (j != T - 6) && (j != T + 6) && (j != T - 7) && (j != T + 7) && (j != T - 8) && (j != T + 8) && (j != T - 9) && (j != T + 9) && (j != T - 10) && (j != T + 10) && (j != T - 11) && (j != T + 11) && (j != T - 12) && (j != T + 12) && (j != T - 13) && (j != T + 13) && (j != T - 14) && (j != T + 14) && (j != T - 15) && (j != T + 15) && (j != T - 16) && (j != T + 16) && (distances[j] > dis2))
	//			{
	//				dis2 = distances[j];
	//				P_new2 = pts[j];
	//				P_2_hull = pts1[j];
	//				R = j;

	//			}
	//		}
	//		double l1, l2;
	//		l1 = cv::norm(P_1_hull - P_2_hull);
	//		l2 = cv::norm(P_new1 - P_new2);

	//		if (((l2 / l1) >= 0.0))
	//		{


	//			//cout<<P_new1<<"  "<<P_new2<<endl;
	//			corners.push_back(P_new1);
	//			corners.push_back(P_new2);
	//			//cout<<distances.size()<<"  "<<contours[i].size()<<endl;
	//			//if ()

	//			//circle(src, P_new1, 4, Scalar(0, 0, 255), -1, 8, 0);
	//			//circle(src, P_new2, 4, Scalar(0, 0, 255), -1, 8, 0);

	//			//line(src, P_new1, P_new2, Scalar(0, 0, 255), 2);
	//			//drawContours(src , contours, i , Scalar(0,255,255),4);
	//		}
	//	}

	//}



	

	//------------

	int n = contours.size();

	vector<vector<Point> > contours_poly1(n);
	vector<Rect> boundRect(n);
	vector<Point2f>centers1(n);
	vector<float>radius1(n);

	//double radius[n];
	ostringstream str1;

	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(contours[i], contours_poly1[i], 3, true);
		boundRect[i] = boundingRect(contours_poly1[i]);
		minEnclosingCircle(contours_poly1[i], centers1[i], radius1[i]);
	}


	/// Draw contours
	Mat drawing = Mat::zeros(im_floodfill.size(), CV_8UC3);
	Mat drawing_N = Mat::zeros(im_floodfill.size(), CV_8UC3);
	c = 0;
	double grain = 0;int boundary = 0, in = 0;
	vector<Vec4i> lines;
	vector<Vec4i> lines1;
	Mat drawing1 = Mat::zeros(im_floodfill.size(), CV_8UC3);
	Mat bound = Mat::zeros(im_floodfill.size(), CV_8UC1);
	Mat inside = Mat::zeros(im_floodfill.size(), CV_8UC1);


	vector<vector<Point> >hull(contours.size());
	//vector<vector<Point> >remainder(contours.size() );
	//namedWindow("lined", WINDOW_NORMAL);
	//resizeWindow("lined" , 500 , 500);
	int X, Y, X_opp, Y_opp;
	int x_min, y_min, x_min_last, y_min_last;

	for (int i = 0; i < contours_11.size(); i++)
	{
		if (arcLength(contours_11[i], 1) > 300)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			approxPolyDP(contours_11[i], contours_11[i], 10, true);
			drawContours(drawing_N, contours_11, i, color, FILLED);
		}
	}


	for (int i = 0; i < contours.size(); i++)
	{


		approxPolyDP(contours[i], contours[i], 1, true);
		vector<vector<Point> > contours_inside;
		int flag = 1;
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, FILLED);

		//rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2 );


		drawContours(drawing1, contours, i, Scalar(255, 255, 255), 2);

		for (int j = 0;j < contours[i].size();j++)
		{
			int px = contours[i][j].x;
			int py = contours[i][j].y;
			//cout<<x<<" , "<<y<<"            ";
			int width = im_floodfill.size().width - 3;
			int height = im_floodfill.size().height - 3;
			if (px <= 1 || px >= width || py <= 1 || py >= height)
			{
				flag = 0;break;
			}
		}

		if (flag == 1)
		{
			grain = grain + 1;drawContours(inside, contours, i, Scalar(255), 1);in++;
		}
		else
		{
			grain = grain + 0.5;
			drawContours(bound, contours, i, Scalar(255), 1);boundary++;
		}

		drawContours(drawing1, contours, i, Scalar(255, 255, 255), 1);

	}


	namedWindow("Contours", WINDOW_NORMAL);
	resizeWindow("Contours", 600, 600);

	namedWindow("Boundary", WINDOW_NORMAL);
	resizeWindow("Boundary", 500, 500);

	imshow("Contours", drawing);
	imshow("Boundary", bound);

	namedWindow("Inside", WINDOW_NORMAL);
	resizeWindow("Inside", 500, 500);
	imshow("Inside", inside);

	final_connected = drawing.clone();

	//-------------------------------Convexity  Defect Code added---------------------
	vector<Point> corners(2 * contours.size());

	for (int i = 0; i < 2 * contours.size(); i++)
	{
		corners[i] = Point(0, 0);
	}


	if (contours.size() > 0)
	{
		vector<std::vector<int> >hullYYY(contours.size());
		vector<vector<Vec4i>> convDef(contours.size());
		vector<vector<Point>> hull_points(contours.size());
		vector<vector<Point>> defect_points(contours.size());
		Mat drawingNP = Mat::zeros(src.size(), CV_8UC3);
		
		int t = 0;
		for (int i = 0; i < contours.size(); i++)
		{
			if (contourArea(contours[i]) > 50)
			{
				convexHull(contours[i], hullYYY[i], false);
				convexityDefects(contours[i], hullYYY[i], convDef[i]);

				for (int k = 0;k < hullYYY[i].size();k++)
				{
					int ind = hullYYY[i][k];
					hull_points[i].push_back(contours[i][ind]);
				}

				for (int k = 0;k < convDef[i].size();k++)
				{
					if (convDef[i][k][3] > 10 * 256) // filter defects by depth
					{
						int ind_0 = convDef[i][k][0];
						int ind_1 = convDef[i][k][1];
						int ind_2 = convDef[i][k][2];
						defect_points[i].push_back(contours[i][ind_2]);
						/*cv::circle(src, contours[i][ind_0], 5, Scalar(0, 255, 0), -1);
						cv::circle(src, contours[i][ind_1], 5, Scalar(0, 255, 0), -1);*/
						cv::circle(drawingNP, contours[i][ind_2], 3 , Scalar(0, 0, 255), -1);
						corners[t] = contours[i][ind_2];
						t++;
						/*cv::line(src, contours[i][ind_2], contours[i][ind_0], Scalar(0, 0, 255), 1);
						cv::line(src, contours[i][ind_2], contours[i][ind_1], Scalar(0, 0, 255), 1);*/
					}
				}

				drawContours(drawingNP, contours, i, Scalar(255 , 255 , 255), 1, 8, vector<Vec4i>(), 0, Point());
				/*drawContours(drawing, hull_points, i, Scalar(255, 0, 0), 1, 8, vector<Vec4i>(), 0, Point());*/
			}
		}
		imshow("DDDNP", drawingNP);
		//waitKey(0);
	}


	

	//-------------------------------------
	//namedWindow("drawing1", WINDOW_NORMAL);
	//resizeWindow("drawing1" , 500 , 500);
	//imshow( "drawing1" , drawing1);

	cout << "inside :  " << in << endl;
	cout << "boundary: " << boundary << endl;

	cout << "final  : " << grain << std::endl << endl;

	cout << contours.size() << endl;

	drawing_clone = drawing.clone();
	Mat src1 = drawing_N;
	if (src1.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		cout << "Usage: " << argv[0] << " <Input image>" << endl;
		return -1;
	}


	//std::vector<cv::Point2f> corners;

	Mat drawing_needed;
	Mat gray_needed;
	cvtColor(drawing_N, gray_needed, COLOR_BGR2GRAY);
	gray_needed.convertTo(drawing_needed, CV_32FC1);
	//imshow("YYYYYYYYY",drawing_needed);
	waitKey(0);
	//cv::goodFeaturesToTrack(drawing_needed, corners, k, 0.01, 30);

	//for (size_t idx = 0; idx < corners.size(); idx++)
	//{
	//	ctr++;
	//	str1 << ctr;
	//	string str2 = str1.str();
	//	cv::circle(drawing_clone, corners.at(idx), 1, Scalar(0, 255, 255), 4);
	//	//cout<<corners.at(idx).x<<endl;
	//	putText(drawing_clone, str2, corners.at(idx), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 255), 1, 8, 0);
	//	str1.str("");
	//	str1.clear();

	//}


	int ctr = 0;
	//ostringstream str1;
	for (size_t idx = 0; idx < corners.size(); idx++)
	{
		if (corners[i] != Point(0, 0)) 
		{
			ctr++;
			str1 << ctr;
			string str2 = str1.str();
			cv::circle(drawing_clone, corners.at(idx), 4, Scalar(0, 0, 255), -1);
			//cout<<corners.at(idx).x<<endl;
			putText(drawing_clone, str2, corners.at(idx), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 1, 8, 0);
			str1.str("");
			str1.clear();
		}

	}

	waitKey(0);



	namedWindow("detected", WINDOW_NORMAL);
	resizeWindow("detected", 600, 600);
	imshow("detected", drawing_clone);

	waitKey(0);


	return 0;


}