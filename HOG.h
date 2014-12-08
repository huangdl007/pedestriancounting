#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>
using namespace cv;

namespace MyHog
{
	class HOG
	{
	public:
		struct HOGCache
		{
			struct BlockData
			{
				BlockData() : histOfs(0), imgOffset() {}
				int histOfs;
				Point imgOffset;
			};

			struct PixData
			{
				size_t gradOfs, qangleOfs;
				int histOfs[4];
				float histWeights[4];
				float gradWeight;
			};

			HOGCache();
			HOGCache(const HOG* descriptor,
				const Mat& img, Size paddingTL, Size paddingBR,
				bool useCache, Size cacheStride);
			~HOGCache() {};
			void init(const HOG* descriptor,
				const Mat& img, Size paddingTL, Size paddingBR,
				bool useCache, Size cacheStride);

			Size windowsInImage(Size imageSize, Size winStride) const;
			Rect getWindow(Size imageSize, Size winStride, int idx) const;

			const float* getBlock(Point pt, float* buf);
			virtual void normalizeBlockHistogram(float* histogram) const;

			vector<PixData> pixData;
			vector<BlockData> blockData;

			bool useCache;
			vector<int> ymaxCached;
			Size winSize, cacheStride;
			Size nblocks, ncells;
			int blockHistogramSize;
			int count1, count2, count4;
			Point imgoffset;
			Mat_<float> blockCache;
			Mat_<uchar> blockCacheFlags;

			Mat grad, qangle;
			const HOG* descriptor;
		};

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
			histogramNormType(HOG::L2Hys), L2HysThreshold(0.2), gammaCorrection(true),
			nlevels(HOG::DEFAULT_NLEVELS)
		{}

		//constructor with parameter
		HOG(Size _winSize, Size _blockSize, Size _blockStride,
			Size _cellSize, int _nbins, int _derivAperture=1, double _winSigma=-1,
			int _histogramNormType=HOG::L2Hys,
			double _L2HysThreshold=0.2, bool _gammaCorrection=false,
			int _nlevels=HOG::DEFAULT_NLEVELS)
			: winSize(_winSize), blockSize(_blockSize), blockStride(_blockStride), cellSize(_cellSize),
			nbins(_nbins), derivAperture(_derivAperture), winSigma(_winSigma),
			histogramNormType(_histogramNormType), L2HysThreshold(_L2HysThreshold),
			gammaCorrection(_gammaCorrection), nlevels(_nlevels)
		{}
		~HOG() {}

		static vector<float> getPeopleDetector();
		void setSVMDetector(InputArray _svmdetector);
		bool checkDetectorSize() const;
		//get the number of values of a descriptor window 
		size_t getDescriptorSize() const;
		void detectMultiScale(const Mat& img, vector<Rect>& foundLocations, vector<double>& foundWeights,
			double hitThreshold, Size winStride, Size padding,
			double scale0, double finalThreshold, bool useMeanshiftGrouping=false) const;

		void detect(const Mat& img, vector<Point>& foundLocations, vector<double>& weights,
			double hitThreshold=0, Size winStride=Size(), Size padding=Size(),
			const vector<Point>& searchLocations=vector<Point>()) const;

		void computeGradient(const Mat& img, Mat& grad, Mat& qangle,
			Size paddingTL, Size paddingBR) const;

		double getWinSigma() const;
		void groupRectangles(vector<cv::Rect>& rectList, vector<double>& weights, int groupThreshold, double eps) const;
	};

}