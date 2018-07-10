#pragma once
#ifndef LINNET_TOOLS_INL
#define LINNET_TOOLS_INL

#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "global.hpp"

#ifdef LINNET_USE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include "layer.hpp"
#include "utils.hpp"

#include "rapidxml\rapidxml.hpp"
#include "rapidxml\rapidxml_utils.hpp"


using std::string;
using std::to_string;
using std::map;

namespace lin {

	namespace tools {
		inline bool def_value_by_string(shared_ptr<Layer> layer_ptr, map<string, string>& layer_paras, const string& value_key) {

			map<string, string>::iterator paraIdx = layer_paras.find(value_key);
			if (paraIdx != layer_paras.end()) {
				if (!value_key.compare("out_channels")) {
					layer_ptr->outChannels = std::stoi(paraIdx->second);
				} else if (!value_key.compare("func_type")) {
					layer_ptr->funcType = util::get_func_type_from_string(paraIdx->second);
				} else if (!value_key.compare("kernel_size")) {
					std::istringstream _isstream(paraIdx->second);
					string _field;
					std::getline(_isstream, _field, ',');
					layer_ptr->kernelSize.w = std::stoi(_field);
					assert(layer_ptr->kernelSize.w % 2 == 1);
					std::getline(_isstream, _field, ',');
					layer_ptr->kernelSize.h = std::stoi(_field);
					assert(layer_ptr->kernelSize.h % 2 == 1);
				} else if (!value_key.compare("pool_size")) {
					std::istringstream _isstream(paraIdx->second);
					string _field;
					std::getline(_isstream, _field, ',');
					layer_ptr->kernelSize.w = std::stoi(_field);
					std::getline(_isstream, _field, ',');
					layer_ptr->kernelSize.h = std::stoi(_field);
				} else if (!value_key.compare("conv_type")) {
					layer_ptr->convType = util::get_conv_type_from_string(paraIdx->second);
				} else if (!value_key.compare("border_type")) {
					layer_ptr->borderType = util::get_border_type_from_string(paraIdx->second);
				}
				return true;
			} else {
				msgwarn("Value " + value_key + " not found for Layer: " + layer_ptr->layerName);
				return false;
			}

		}

		//----------------------------------------------------------

		void get_data_size(shared_ptr<Layer> prev_layer_ptr, shared_ptr<Layer> curr_layer_ptr) {
			Size input_size = prev_layer_ptr->dataSize;
			switch (curr_layer_ptr->layerType) {
			case LayerType::LINNET_LAYER_INPUT:
				break;
			case LayerType::LINNET_LAYER_CONV:
				if (curr_layer_ptr->convType == ConvType::LINNET_CONV_SAME)
					curr_layer_ptr->dataSize = input_size;
				else if (curr_layer_ptr->convType == ConvType::LINNET_CONV_VALID)
					curr_layer_ptr->dataSize = Size(
						input_size.w - curr_layer_ptr->kernelSize.w + 1,
						input_size.h - curr_layer_ptr->kernelSize.h + 1
					);
				else
					msgerror("Invalid ConvType: " + curr_layer_ptr->convType);
				break;
			case LayerType::LINNET_LAYER_MAXPOOL:
				curr_layer_ptr->dataSize = Size(
					input_size.w / curr_layer_ptr->kernelSize.w,
					input_size.h / curr_layer_ptr->kernelSize.h
				);
				break;
			case LayerType::LINNET_LAYER_LINEAR:
			case LayerType::LINNET_LAYER_OUTPUT:
				curr_layer_ptr->dataSize = Size(1, 1);
				break;
			default:
				msgerror("Invalid LayerType: " + curr_layer_ptr->layerType);
				break;
			}
		}

		//----------------------------------------------------------

		inline int get_weight_num(shared_ptr<Layer> layer_ptr) {
			int rst;
			switch (layer_ptr->layerType) {
			case LayerType::LINNET_LAYER_INPUT:
			case LayerType::LINNET_LAYER_MAXPOOL:
			case LayerType::LINNET_LAYER_FLATTEN:
				rst = 0;
				break;
			case LayerType::LINNET_LAYER_LINEAR:
			case LayerType::LINNET_LAYER_OUTPUT:
				rst = 1;
				break;
			case LayerType::LINNET_LAYER_CONV:
				rst = layer_ptr->inChannels * layer_ptr->outChannels;
				break;
			default:
				msgerror("Invalid LayerType: " + layer_ptr->layerType);
				break;
			}
			return rst;
		}

		//----------------------------------------------------------

		inline Mat init_weight(Size size, InitMethod init_method, int in_channels, int out_channels) {
			assert(size.checkpositive());
			Mat new_mat(size.h, size.w, CV_64FC1);
			double limit;
			switch (init_method) {
			case InitMethod::LINNET_INIT_XAVIER_NORM:
				limit = sqrt(2. / (size.w * size.h * (in_channels + out_channels)));
				cv::randn(new_mat, cv::Scalar(0.0), cv::Scalar(limit));
				break;
			case InitMethod::LINNET_INIT_XAVIER_UNIFORM:
				limit = sqrt(6. / (size.w * size.h * (in_channels + out_channels)));
				cv::randu(new_mat, cv::Scalar(-limit), cv::Scalar(limit));
				break;
			case InitMethod::LINNET_INIT_HE_NORM:
				limit = sqrt(2. / (size.w * size.h * in_channels));
				cv::randn(new_mat, cv::Scalar(0.0), cv::Scalar(limit));
				break;
			case InitMethod::LINNET_INIT_HE_UNIFORM:
				limit = sqrt(6. / (size.w * size.h * in_channels));
				cv::randu(new_mat, cv::Scalar(-limit), cv::Scalar(limit));
				break;
			default:
				msgerror("Invalid InitMethod: " + init_method);
				break;
			}
			return new_mat;
		}

	}

}

#endif
