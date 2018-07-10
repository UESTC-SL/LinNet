#pragma once
#ifndef LINNET_UTILITY_INL
#define LINNET_UTILITY_INL

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

		inline string clean_string(const string& input_string) {

			string new_string(input_string);
			new_string.erase(std::remove(new_string.begin(), new_string.end(), ' '), new_string.end());
			new_string.erase(std::remove(new_string.begin(), new_string.end(), '\n'), new_string.end());
			new_string.erase(std::remove(new_string.begin(), new_string.end(), '\r'), new_string.end());
			new_string.erase(std::remove(new_string.begin(), new_string.end(), '\t'), new_string.end());
			return new_string;

		}

		//----------------------------------------------------------

		inline Mat parse_mat_from_xml_d(string mat_rows, string mat_cols, string mat_data) {

			int rows = stoi(mat_rows);
			int cols = stoi(mat_cols);
			Mat _result(cols, rows, CV_64FC1);
			double tmp;

			std::stringstream _data(mat_data);
			for (int i = 0; i < rows; i++) {
				double *p_out = _result.ptr<double>(i);
				for (int j = 0; j < cols; j++) {
					_data >> tmp;
					p_out[j] = tmp;
				}
			}

			return _result;
		}

		//----------------------------------------------------------

		inline bool compare_str_layer_type(const string& input_string, const LayerType layer_type) {

			switch (layer_type) {
			case LayerType::LINNET_LAYER_INPUT:
				if (!input_string.compare("i") || !input_string.compare("in") || !input_string.compare("input") || !input_string.compare(to_string((int)LayerType::LINNET_LAYER_INPUT)))
					return true;
				else
					return false;
				break;
			case LayerType::LINNET_LAYER_OUTPUT:
				if (!input_string.compare("o") || !input_string.compare("out") || !input_string.compare("output") || !input_string.compare(to_string((int)LayerType::LINNET_LAYER_OUTPUT)))
					return true;
				else
					return false;
				break;
			case LayerType::LINNET_LAYER_CONV:
				if (!input_string.compare("c") || !input_string.compare("conv") || !input_string.compare("convolution") || !input_string.compare("convolutional") || !input_string.compare(to_string((int)LayerType::LINNET_LAYER_CONV)))
					return true;
				else
					return false;
				break;
			case LayerType::LINNET_LAYER_LINEAR:
				if (!input_string.compare("l") || !input_string.compare("linear") || !input_string.compare("connected") || !input_string.compare("all-connected") || !input_string.compare(to_string((int)LayerType::LINNET_LAYER_LINEAR)))
					return true;
				else
					return false;
				break;
			case LayerType::LINNET_LAYER_MAXPOOL:
				if (!input_string.compare("max") || !input_string.compare("maxpool") || !input_string.compare("max-pool") || !input_string.compare("maxpooling") || !input_string.compare("max-pooling") || !input_string.compare("pool") || !input_string.compare("pooling") || !input_string.compare(to_string((int)LayerType::LINNET_LAYER_MAXPOOL)))
					return true;
				else
					return false;
				break;
			default:
				msgerror("Comparing string with LayerType failed: Invalid LayerType name: " + input_string);
				break;
			}
		}

		//----------------------------------------------------------

		inline bool compare_str_func_type(const string& input_string, const FuncType func_type) {

			switch (func_type) {
			case FuncType::LINNET_FUNC_RELU:
				if (!input_string.compare("relu") || !input_string.compare(to_string((int)FuncType::LINNET_FUNC_RELU)))
					return true;
				else
					return false;
				break;
			case FuncType::LINNET_FUNC_SIGMOID:
				if (!input_string.compare("sigmoid") || !input_string.compare("logistic") || !input_string.compare(to_string((int)FuncType::LINNET_FUNC_SIGMOID)))
					return true;
				else
					return false;
				break;
			case FuncType::LINNET_FUNC_TANH:
				if (!input_string.compare("tanh") || !input_string.compare(to_string((int)FuncType::LINNET_FUNC_TANH)))
					return true;
				else
					return false;
				break;
			case FuncType::LINNET_FUNC_SOFTMAX:
				if (!input_string.compare("softmax") || !input_string.compare(to_string((int)FuncType::LINNET_FUNC_SOFTMAX)))
					return true;
				else
					return false;
				break;
			default:
				msgerror("Comparing string with FuncType failed: Invalid FuncType name: " + input_string);
				break;
			}
		}

		//----------------------------------------------------------

		inline bool compare_str_conv_type(const string& input_string, const ConvType conv_type) {

			switch (conv_type) {
			case ConvType::LINNET_CONV_SAME:
				if (!input_string.compare("s") || !input_string.compare("same") || !input_string.compare(to_string((int)ConvType::LINNET_CONV_SAME)))
					return true;
				else
					return false;
				break;
			case ConvType::LINNET_CONV_VALID:
				if (!input_string.compare("v") || !input_string.compare("valid") || !input_string.compare(to_string((int)ConvType::LINNET_CONV_VALID)))
					return true;
				else
					return false;
				break;
			default:
				msgerror("Comparing string with ConvType failed: Invalid ConvType name: " + input_string);
				break;
			}
		}

		//----------------------------------------------------------

		inline bool compare_str_border_type(const string& input_string, const BorderType border_type) {

			switch (border_type) {
			case BorderType::LINNET_BORDER_COPY:
				if (!input_string.compare("d") || !input_string.compare("duplicate") || !input_string.compare("copy") || !input_string.compare(to_string((int)BorderType::LINNET_BORDER_COPY)))
					return true;
				else
					return false;
				break;
			case BorderType::LINNET_BORDER_ZEROS:
				if (!input_string.compare("z") || !input_string.compare("zero") || !input_string.compare("zeros") || !input_string.compare(to_string((int)BorderType::LINNET_BORDER_ZEROS)))
					return true;
				else
					return false;
				break;
			default:
				msgerror("Comparing string with BorderType failed: Invalid BorderType name: " + input_string);
				break;
			}
		}

		//----------------------------------------------------------

		inline bool compare_str_loss_function(const string& input_string, const LossFunction loss_function) {

			switch (loss_function) {
			case LossFunction::LINNET_LOSS_ABS:
				if (!input_string.compare("abs") || !input_string.compare("absolute") || !input_string.compare(to_string((int)LossFunction::LINNET_LOSS_ABS)))
					return true;
				else
					return false;
				break;
			case LossFunction::LINNET_LOSS_ENTROPY:
				if (!input_string.compare("entro") || !input_string.compare("entropy") || !input_string.compare("cross-entropy") || !input_string.compare(to_string((int)LossFunction::LINNET_LOSS_ENTROPY)))
					return true;
				else
					return false;
				break;
			case LossFunction::LINNET_LOSS_SQUARE:
				if (!input_string.compare("squ") || !input_string.compare("square") || !input_string.compare("squared") || !input_string.compare(to_string((int)LossFunction::LINNET_LOSS_SQUARE)))
					return true;
				else
					return false;
				break;
			default:
				msgerror("Comparing string with LossFunction failed: Invalid LossFunction name: " + input_string);
				break;
			}
		}

		//----------------------------------------------------------

		inline bool compare_str_gradient_method(const string& input_string, const GradientMethod gradient_method) {

			switch (gradient_method) {
			case GradientMethod::LINNET_GD_SGD:
				if (!input_string.compare("sgd") || !input_string.compare(to_string((int)GradientMethod::LINNET_GD_SGD)))
					return true;
				else
					return false;
				break;
			case GradientMethod::LINNET_GD_MOMENTUM:
				if (!input_string.compare("adam") || !input_string.compare(to_string((int)GradientMethod::LINNET_GD_MOMENTUM)))
					return true;
				else
					return false;
				break;
			default:
				msgerror("Comparing string with GradientMethod failed: Invalid GradientMethod name: " + input_string);
				break;
			}
		}

		//----------------------------------------------------------

		inline LayerType get_layer_type_from_string(const string& input_string) {

			if (compare_str_layer_type(input_string, LayerType::LINNET_LAYER_INPUT))
				return LayerType::LINNET_LAYER_INPUT;
			else if (compare_str_layer_type(input_string, LayerType::LINNET_LAYER_OUTPUT))
				return LayerType::LINNET_LAYER_OUTPUT;
			else if (compare_str_layer_type(input_string, LayerType::LINNET_LAYER_CONV))
				return LayerType::LINNET_LAYER_CONV;
			else if (compare_str_layer_type(input_string, LayerType::LINNET_LAYER_LINEAR))
				return LayerType::LINNET_LAYER_LINEAR;
			else if (compare_str_layer_type(input_string, LayerType::LINNET_LAYER_MAXPOOL))
				return LayerType::LINNET_LAYER_MAXPOOL;
			else {
				msgerror("Invalid LayerType name: " + input_string);
			}
		}

		//----------------------------------------------------------

		inline FuncType get_func_type_from_string(const string& input_string) {

			if (compare_str_func_type(input_string, FuncType::LINNET_FUNC_RELU))
				return FuncType::LINNET_FUNC_RELU;
			else if (compare_str_func_type(input_string, FuncType::LINNET_FUNC_SIGMOID))
				return FuncType::LINNET_FUNC_SIGMOID;
			else if (compare_str_func_type(input_string, FuncType::LINNET_FUNC_TANH))
				return FuncType::LINNET_FUNC_TANH;
			else if (compare_str_func_type(input_string, FuncType::LINNET_FUNC_SOFTMAX))
				return FuncType::LINNET_FUNC_SOFTMAX;
			else {
				msgerror("Invalid FuncType name: " + input_string);
			}
		}

		//----------------------------------------------------------

		inline ConvType get_conv_type_from_string(const string& input_string) {

			if (compare_str_conv_type(input_string, ConvType::LINNET_CONV_SAME))
				return ConvType::LINNET_CONV_SAME;
			else if (compare_str_conv_type(input_string, ConvType::LINNET_CONV_VALID))
				return ConvType::LINNET_CONV_VALID;
			else {
				msgerror("Invalid ConvType name: " + input_string);
			}
		}

		//----------------------------------------------------------

		inline BorderType get_border_type_from_string(const string& input_string) {

			if (compare_str_border_type(input_string, BorderType::LINNET_BORDER_COPY))
				return BorderType::LINNET_BORDER_COPY;
			else if (compare_str_border_type(input_string, BorderType::LINNET_BORDER_ZEROS))
				return BorderType::LINNET_BORDER_ZEROS;
			else {
				msgerror("Invalid BorderType name: " + input_string);
			}
		}

		//----------------------------------------------------------

		inline LossFunction get_loss_function_from_string(const string& input_string) {

			if (compare_str_loss_function(input_string, LossFunction::LINNET_LOSS_ABS))
				return LossFunction::LINNET_LOSS_ABS;
			else if (compare_str_loss_function(input_string, LossFunction::LINNET_LOSS_SQUARE))
				return LossFunction::LINNET_LOSS_SQUARE;
			else if (compare_str_loss_function(input_string, LossFunction::LINNET_LOSS_ENTROPY))
				return LossFunction::LINNET_LOSS_ENTROPY;
			else {
				msgerror("Invalid LossFunction name: " + input_string);
			}
		}

		//----------------------------------------------------------

		inline GradientMethod get_gradient_method_from_string(const string& input_string) {

			if (compare_str_gradient_method(input_string, GradientMethod::LINNET_GD_SGD))
				return GradientMethod::LINNET_GD_SGD;
			else if (compare_str_gradient_method(input_string, GradientMethod::LINNET_GD_MOMENTUM))
				return GradientMethod::LINNET_GD_MOMENTUM;
			else {
				msgerror("Invalid GradientMethod name: " + input_string);
			}
		}

#ifdef LINNET_USE_OPENCV

		//----------------------------------------------------------

		inline cv::Mat coloredMap(cv::Mat& img_src) {

			if (img_src.channels() != 1) {
				msgerror("Input image must be single-channel for function `coloredMat()`.");
			}

			cv::Mat img_rst(img_src.size(), CV_64FC3);

			double _min, _max;
			cv::minMaxIdx(img_src, &_min, &_max);

			for (int i = 0; i < img_src.rows; i++) {
				double* psrc = img_src.ptr<double>(i);
				double* prst = img_rst.ptr<double>(i);
				for (int j = 0; j < img_src.cols; j++) {

					if (psrc[j] < 0) {
						prst[j * 3] = 0.0;
						prst[j * 3 + 1] = prst[j * 3 + 2] = psrc[j] / _min;
					} else {
						prst[j * 3] = psrc[j] / _max;
						prst[j * 3 + 1] = prst[j * 3 + 2] = 0.0;
					}

				}

			}

			return img_rst;

		}

#endif

	}

}

#endif

