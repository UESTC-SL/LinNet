#pragma once
#ifndef LINNET_FUNCTION_INL
#define LINNET_FUNCTION_INL

#include <vector>
#include <string>
#include <math.h>
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <type_traits>

#include "tensor.hpp"

using std::vector;
using std::string;
using std::pair;
using std::make_pair;
using std::map;
using std::shared_ptr;

namespace lin {

	namespace func {

		Mat ReLU(Mat& src) {

			cv::Size size = src.size();
			Mat result(size, CV_64FC1);

			/*
			if (src.isContinuous() && result.isContinuous()) {
				size.width *= size.height;
				size.height = 1;
			}
			for (int j = 0; j < size.height; j++) {
				const double* pin = src.ptr<double>(j);
				double* pout = result.ptr<double>(j);
				for (int i = 0; i < size.width; i++) {
					pout[i] = (pin[i] > 0) ? pin[i] : 0;
				}
			}
			*/
			result = cv::max(src, 0.0);
			return result;
		}

		Mat ReLU_bp(Mat& src) {

			return ReLU(src);
		}

		//----------------------------------------------------------

		Mat softmax(Mat& src) {

			cv::Size size = src.size();
			Mat result(size, CV_64FC1);

			for (int j = 0; j < size.height; j++) {
				const double* pin = src.ptr<double>(j);
				double* pout = result.ptr<double>(j);
				double sum = 0.0;
				for (int i = 0; i < size.width; i++) {
					sum += exp(pin[i]);
				}
				for (int i = 0; i < size.width; i++) {
					pout[i] = exp(pin[i]) / sum;
				}
			}
			return result;
		}

		Mat softmax_bp(Mat& src) {
			
			cv::Size size = src.size();
			Mat result(src.size(), CV_64FC1);

			return result;
		}

		//----------------------------------------------------------

		Mat maxpooling(Mat& src, Size scale) {

			cv::Size size = src.size();
			assert(size.width % scale.w == 0 && size.height % scale.h == 0);
			Mat result(cv::Size(size.width / scale.w, size.height / scale.h), CV_64FC1);

			for (int j = 0; j < result.rows; j++) {
				double* pout = result.ptr<double>(j);
				for (int i = 0; i < result.cols; i++) {
					double min, max;
					cv::Rect roi(i * scale.w, j * scale.h, scale.w, scale.h);
					cv::minMaxIdx(src(roi), &min, &max);
					pout[i] = max;
				}
			}
			return result;

		}

		Mat maxpooling_bp(Mat& src, Size scale) {

			cv::Size size = src.size();
			Mat result(cv::Size(size.width * scale.w, size.height * scale.h), CV_64FC1);

			return result;

		}

	}

}

#endif
