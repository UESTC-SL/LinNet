#pragma once
#ifndef LINNET_UTILITY_HPP
#define LINNET_UTILITY_HPP

#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "global.hpp"

#ifdef LINNET_USE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include "variables.hpp"

#include "rapidxml\rapidxml.hpp"
#include "rapidxml\rapidxml_utils.hpp"


using std::string;
using std::to_string;
using std::map;

namespace lin {

	namespace util {

		//----------------------------------------------------------
		//Clear any space character in string.
		//Only `output` parameter will be cleaned.
		//----------------------------------------------------------
		string clean_string(const string& input_string);

		//----------------------------------------------------------
		//Parse int Mat from string.
		//----------------------------------------------------------
		Mat parse_mat_from_xml_d(string mat_rows, string mat_cols, string mat_data);

		//----------------------------------------------------------
		//Compare string with LayerType.
		//----------------------------------------------------------
		bool compare_str_layer_type(const string& input_string, const LayerType layer_type);

		//----------------------------------------------------------
		//Compare string with FuncType.
		//----------------------------------------------------------
		bool compare_str_func_type(const string& input_string, const FuncType func_type);

		//----------------------------------------------------------
		//Compare string with ConvType.
		//----------------------------------------------------------
		bool compare_str_conv_type(const string& input_string, const ConvType conv_type);

		//----------------------------------------------------------
		//Compare string with BorderType.
		//----------------------------------------------------------
		bool compare_str_border_type(const string& input_string, const BorderType border_type);

		//----------------------------------------------------------
		//Compare string with LossFunction.
		//----------------------------------------------------------
		bool compare_str_loss_function(const string& input_string, const LossFunction border_type);

		//----------------------------------------------------------
		//Compare string with GradientMethod.
		//----------------------------------------------------------
		bool compare_str_gradient_method(const string& input_string, const GradientMethod gradient_method);

		//----------------------------------------------------------
		//Parse LayerType from string.
		//----------------------------------------------------------
		LayerType get_layer_type_from_string(const string& input_string);

		//----------------------------------------------------------
		//Parse FuncType from string.
		//----------------------------------------------------------
		FuncType get_func_type_from_string(const string& input_string);

		//----------------------------------------------------------
		//Parse ConvType from string.
		//----------------------------------------------------------
		ConvType get_conv_type_from_string(const string& input_string);

		//----------------------------------------------------------
		//Parse BorderType from string.
		//----------------------------------------------------------
		BorderType get_border_type_from_string(const string& input_string);

		//----------------------------------------------------------
		//Parse LossFunction from string.
		//----------------------------------------------------------
		LossFunction get_loss_function_from_string(const string& input_string);

		//----------------------------------------------------------
		//Parse GradientMethod from string.
		//----------------------------------------------------------
		GradientMethod get_gradient_method_from_string(const string& input_string);

#ifdef LINNET_USE_OPENCV

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		cv::Mat coloredMap(cv::Mat& img_src);

#endif

	}

}

#include "utils.inl"

#endif
