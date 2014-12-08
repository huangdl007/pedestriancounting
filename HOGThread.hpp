#include <opencv2/opencv.hpp>

namespace MyHog
{

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
			int i, i1 = range.start, i2 = range.end;
			double minScale = i1 > 0 ? levelScale[i1] : i2 > 1 ? levelScale[i1+1] : std::max(img.cols, img.rows);
			Size maxSz(cvCeil(img.cols/minScale), cvCeil(img.rows/minScale));
			Mat smallerImgBuf(maxSz, img.type());
			vector<Point> locations;
			vector<double> hitsWeights;

			for( i = i1; i < i2; i++ )
			{
				double scale = levelScale[i];
				Size sz(cvRound(img.cols/scale), cvRound(img.rows/scale));
				Mat smallerImg(sz, img.type(), smallerImgBuf.data);
				if( sz == img.size() )
					smallerImg = Mat(sz, img.type(), img.data, img.step);
				else
					resize(img, smallerImg, sz);
				hog->detect(smallerImg, locations, hitsWeights, hitThreshold, winStride, padding);
				Size scaledWinSize = Size(cvRound(hog->winSize.width*scale), cvRound(hog->winSize.height*scale));

				mtx->lock();
				for( size_t j = 0; j < locations.size(); j++ )
				{
					rec->push_back(Rect(cvRound(locations[j].x*scale),
						cvRound(locations[j].y*scale),
						scaledWinSize.width, scaledWinSize.height));
					if (scales)
					{
						scales->push_back(scale);
					}
				}
				mtx->unlock();

				if (weights && (!hitsWeights.empty()))
				{
					mtx->lock();
					for (size_t j = 0; j < locations.size(); j++)
					{
						weights->push_back(hitsWeights[j]);
					}
					mtx->unlock();
				}
			}
		}
	};

}