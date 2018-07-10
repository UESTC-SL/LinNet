#pragma once
#ifndef LINNET_TOOLS_HPP
#define LINNET_TOOLS_HPP

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
		//----------------------------------------------------------
		//Parse value from string to member of LayerDef.
		//----------------------------------------------------------
		bool def_value_by_string(shared_ptr<Layer> layer_ptr, map<string, string>& layer_paras, const string& value_key);

		//----------------------------------------------------------
		//Calculate dataSize based on previous and current layers.
		//----------------------------------------------------------
		void get_data_size(shared_ptr<Layer> prev_layer_ptr, shared_ptr<Layer> curr_layer_ptr);

		//----------------------------------------------------------
		//Calculate weight number based on layer type.
		//----------------------------------------------------------
		int get_weight_num(shared_ptr<Layer> layer_ptr);

		//----------------------------------------------------------
		//Initialize weight based on Size and InitMethod.
		//----------------------------------------------------------
		Mat init_weight(Size size, InitMethod init_method, int in_channels, int out_channels);
	}

}

#include "tools.inl"

#endif
