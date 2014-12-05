#include <opencv2/opencv.hpp>

class HOGThread : public ParallelLoopBody
{
public:
	const HOG* hog;
	Mat img;
	double hitThreshold;
	Size winStride;
	Size padding;
	const double* levelScale;
	std::vector<Rect>* rec;
	std::vector<double>* weights;
	std::vector<double>* scales;
	Mutex* mtx;

	HOGThread(const HOG* _hog, const Mat& _img, double _hitThreshold,
		Size _winStride, Size _padding, const double* _levelScale,
		std::vector<Rect>* _rec, Mutex* _mtx,
		std::vector<double>* _weights=0, std::vector<double>* _scales=0)
		:hog(_hog), img(_img), hitThreshold(_hitThreshold), winStride(_winStride),
		padding(_padding), levelScale(_levelScale), rec(_rec), mtx(_mtx),
		weights(_weights), scales(_scales)
	{}

	void operator()(const Range& range) const
	{
		
	}
};