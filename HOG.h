#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>
using namespace cv;
class HOG
{
public:
	enum { L2Hys=0 };
	enum { DEFAULT_NLEVELS=64 };
	Size winSize;
	Size blockSize;
	Size blockStride;
	Size cellSize;
	int nbins;
	int derivAperture;
	double winSigma;
	int histogramNormType;
	double L2HysThreshold;
	bool gammaCorrection;
	vector<float> svmDetector;
	int nlevels;

	//constructor without parameter
	HOG() : winSize(64,128), blockSize(16,16), blockStride(8,8),
		cellSize(8,8), nbins(9), derivAperture(1), winSigma(-1),
		histogramNormType(HOG::L2Hys), L2HysThreshold(0.2), 
		gammaCorrection(true), nlevels(HOG::DEFAULT_NLEVELS)
	{}

	//constructor with parameter
	HOG(Size _winSize, Size _blockSize, Size _blockStride, Size _cellSize,
		int _nbins, int _derivAperture=1, double _winSigma=-1,
		int _histogramNormType=HOG::L2Hys, double _L2HysThreshold=0.2,
		bool _gammaCorrection=false, int _nlevels=HOG::DEFAULT_NLEVELS)
		: winSize(_winSize), blockSize(_blockSize), blockStride(_blockStride),
		cellSize(_cellSize), nbins(_nbins), derivAperture(_derivAperture),
		winSigma(_winSigma), histogramNormType(_histogramNormType), 
		L2HysThreshold(_L2HysThreshold), gammaCorrection(_gammaCorrection),
		nlevels(_nlevels)
	{}
	~HOG() {}

	static vector<float> getPeopleDetector();
	void setSVMDetector(InputArray _svmdetector);
	bool checkDetectorSize() const;
	//get the number of values of a descriptor window 
	size_t getDescriptorSize() const;
	void detectMultiScale(const Mat& img, vector<Rect>& foundLocations, vector<double>& foundWeights,
		double hitThreshold, Size winStride, Size padding,
		double scale0, double finalThreshold, bool useMeanshiftGrouping) const;
};