#include <iostream>
#include <opencv2/opencv.hpp>
#include "HOG.h"

using namespace std;
using namespace cv;
using MyHog::HOG;
#define CV_CAP_ANY "C:\\Users\\Administrator\\Desktop\\Dataset\\Outdoor\\sidesync.avi"

int main (int argc, const char * argv[])
{
	/*VideoCapture cap(CV_CAP_ANY);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

	if (!cap.isOpened())
	return -1;
	*/
	Mat img;
	HOG hog;
	hog.setSVMDetector(HOG::getPeopleDetector());

	namedWindow("video capture", CV_WINDOW_AUTOSIZE);

	double t = (double)getTickCount();
	//cap >> img;
	img = imread("pedestriancounting\\800_600.png");

	if (!img.data)
	{
		cout << "Could not open the pic" << endl;
		return -1;
	}
	vector<Rect> found, found_filtered;
	vector<double> foundWeights;
	hog.detectMultiScale(img, found, foundWeights, 0, Size(8,8), Size(32,32), 1.05, 2);
	
	size_t i, j;
	for (i=0; i<found.size(); i++)
	{
		Rect r = found[i];
		for (j=0; j<found.size(); j++)
			if (j!=i && (r & found[j])==r)
				break;
		if (j==found.size())
			found_filtered.push_back(r);
	}
	for (i=0; i<found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];
		/*r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.06);
		r.height = cvRound(r.height*0.9);*/
		rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 2);
	}
	imshow("video capture", img);

	t = (double)getTickCount() - t;
	cout << "cost time: " << t*1000/getTickFrequency() << " ms" << endl;

	cvWaitKey(0);


	return 0;
}