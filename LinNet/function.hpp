#pragma once
#ifndef LINNET_FUNCTION_HPP
#define LINNET_FUNCTION_HPP

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

	namespace func{

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		Mat ReLU(Mat& src);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		Mat ReLU_bp(Mat& src);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		Mat softmax(Mat& src);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		Mat softmax_bp(Mat& src);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		Mat maxpooling(Mat& src, Size scale);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		Mat maxpooling_bp(Mat& src, Size scale);

	}

}

#include "function.inl"

#endif
