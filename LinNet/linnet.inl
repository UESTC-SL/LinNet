#pragma once
#ifndef LINNET_LINNET_INL
#define LINNET_LINNET_INL

#include <stdio.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <math.h>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "rapidxml\rapidxml.hpp"
#include "rapidxml\rapidxml_utils.hpp"

#include "dataloader.hpp"
#include "tools.hpp"

using std::vector;
using std::string;
using std::pair;
using std::make_pair;
using std::map;
using std::make_shared;
using std::set;

namespace lin {

	Net::Net() : initialized(false), hyperValidated(false), trained(false),
		iterIdx(0), epochIdx(0), rng(0),
		batchSize(0), epochNum(0), imageChannels(1),
		optimizer(), inputSize(), loss(), layers() {}

	Net::~Net() {
		clear();
	}

	inline void Net::clear() {
		inputSize.clear();
		// Clear memory of vectors.
		// Or call `lossAll.clear()` with `lossAll.shrink_to_fit()`.
		std::vector<double>().swap(loss);
		std::vector<shared_ptr<Layer>>().swap(layers);
	}

	inline void Net::init(const string& file_name) {

		//----------------------------------------------------------
		//Check file format.
		//----------------------------------------------------------
		msgnote("Should check XML file format.");

		initialized = false;
		hyperValidated = false;
		trained = false;

		//----------------------------------------------------------
		//Parse file root node.
		//----------------------------------------------------------
		rapidxml::file<> _fdoc(file_name.c_str());
		rapidxml::xml_document<> _doc;
		_doc.parse<0>(_fdoc.data());

		//----------------------------------------------------------
		//Start parsing values in "network_definition".
		//----------------------------------------------------------
		rapidxml::xml_node<>* _root = _doc.first_node();
		if (string(_root->name()) == "network_definition") {

			batchSize = std::stoi(util::clean_string(_root->first_node("batch_size")->value()));

			epochNum = std::stoi(util::clean_string(_root->first_node("epoch_num")->value()));

			GradientMethod gradient_method = util::get_gradient_method_from_string(util::clean_string(_root->first_node("gradient_method")->value()));
			optimizer = create_optimizer(gradient_method);

			double lr = std::stod(util::clean_string(_root->first_node("learning_rate")->value()));
			optimizer->init(lr);

			lossFunction = util::get_loss_function_from_string(util::clean_string(_root->first_node("loss_function")->value()));

			//----------------------------------------------------------
			//Parse `inputSize`.
			//----------------------------------------------------------
			string _input_size_s(util::clean_string(_root->first_node("input_size")->value()));
			std::istringstream _isstream(_input_size_s);
			string _field;
			std::getline(_isstream, _field, ',');
			inputSize.w = std::stoi(_field);
			std::getline(_isstream, _field, ',');
			inputSize.h = std::stoi(_field);

			imageChannels = std::stoi(util::clean_string(_root->first_node("image_channels")->value()));

			classNum = std::stoi(util::clean_string(_root->first_node("class_num")->value()));

			//input layer 0
			layers.emplace_back(make_shared<InputLayer>());
			layers.back()->layerName = "input";
			layers.back()->layerType = LayerType::LINNET_LAYER_INPUT;
			layers.back()->funcType = FuncType::LINNET_FUNC_IDENTITY;
			layers.back()->dataSize = inputSize;
			layers.back()->inChannels = imageChannels;
			layers.back()->outChannels = imageChannels;

			//conv1 layer 1
			layers.emplace_back(make_shared<ConvLayer>());
			layers.back()->layerName = "conv1";
			layers.back()->layerType = LayerType::LINNET_LAYER_CONV;
			layers.back()->funcType = FuncType::LINNET_FUNC_RELU;
			layers.back()->convType = ConvType::LINNET_CONV_VALID;
			layers.back()->dataSize = Size(24, 24);
			layers.back()->kernelSize = Size(5, 5);
			layers.back()->inChannels = imageChannels;
			layers.back()->outChannels = 6;

			//maxpool1 layer 2
			layers.emplace_back(make_shared<MaxPoolLayer>());
			layers.back()->layerName = "maxpool1";
			layers.back()->layerType = LayerType::LINNET_LAYER_MAXPOOL;
			layers.back()->funcType = FuncType::LINNET_FUNC_IDENTITY;
			layers.back()->dataSize = Size(12, 12);
			layers.back()->kernelSize = Size(2, 2);
			layers.back()->inChannels = 6;
			layers.back()->outChannels = 6;

			//conv2 layer 3
			layers.emplace_back(make_shared<ConvLayer>());
			layers.back()->layerName = "conv2";
			layers.back()->layerType = LayerType::LINNET_LAYER_CONV;
			layers.back()->funcType = FuncType::LINNET_FUNC_RELU;
			layers.back()->convType = ConvType::LINNET_CONV_VALID;
			layers.back()->dataSize = Size(8, 8);
			layers.back()->kernelSize = Size(5, 5);
			layers.back()->inChannels = 6;
			layers.back()->outChannels = 12;

			//maxpool2 layer 4
			layers.emplace_back(make_shared<MaxPoolLayer>());
			layers.back()->layerName = "maxpool2";
			layers.back()->layerType = LayerType::LINNET_LAYER_MAXPOOL;
			layers.back()->funcType = FuncType::LINNET_FUNC_IDENTITY;
			layers.back()->dataSize = Size(4, 4);
			layers.back()->kernelSize = Size(2, 2);
			layers.back()->inChannels = 12;
			layers.back()->outChannels = 12;

			//convpool layer 5
			layers.emplace_back(make_shared<ConvLayer>());
			layers.back()->layerName = "convpool";
			layers.back()->layerType = LayerType::LINNET_LAYER_CONV;
			layers.back()->funcType = FuncType::LINNET_FUNC_RELU;
			layers.back()->convType = ConvType::LINNET_CONV_VALID;
			layers.back()->dataSize = Size(1, 1);
			layers.back()->kernelSize = Size(4, 4);
			layers.back()->inChannels = 12;
			layers.back()->outChannels = 10;

			//output layer 6
			layers.emplace_back(make_shared<OutputLayer>());
			layers.back()->layerName = "output";
			layers.back()->layerType = LayerType::LINNET_LAYER_OUTPUT;
			layers.back()->funcType = FuncType::LINNET_FUNC_SOFTMAX;
			layers.back()->convType = ConvType::LINNET_CONV_SAME;
			layers.back()->dataSize = Size(1, 1);
			layers.back()->kernelSize = Size(1, 1);
			layers.back()->inChannels = 10;
			layers.back()->outChannels =  classNum;
			

		}//end if: "network_definition"
		else {
			msgerror("The definition of networks need to be placed within `network_definition` block.");
		}

		initialize_variables();

		return;
	}

	//----------------------------------------------------------

	inline void Net::initialize_variables() {

		if (initialized) {
			msgwarn("The networks have already been initialized.");
			return;
		}

		hyperValidated = false;

		for (id _def_idx = 0; _def_idx < layers.size(); _def_idx++) {

			init_units(_def_idx);

			switch (layers[_def_idx]->layerType) {
			case LayerType::LINNET_LAYER_INPUT:
			case LayerType::LINNET_LAYER_MAXPOOL:
			case LayerType::LINNET_LAYER_FLATTEN:

				break;
			case LayerType::LINNET_LAYER_LINEAR:
			case LayerType::LINNET_LAYER_OUTPUT:

				init_weights(_def_idx, InitMethod::LINNET_INIT_XAVIER_UNIFORM);

				init_biases(_def_idx);

				break;
			case LayerType::LINNET_LAYER_CONV:

				init_weights(_def_idx, InitMethod::LINNET_INIT_XAVIER_NORM);

				init_biases(_def_idx);

				break;
			default:
				msgerror("Invalid `layer_type` for Layer: " + layers[_def_idx]->layerName);
			}

		}

		initialized = true;

		hyper_validate();

		return;
	}

	//----------------------------------------------------------

	inline bool Net::hyper_validate() {

		if (!initialized) {
			msgerror("The network must be initialized before hyper-validation.");
			return false;
		}

		std::cout << "Start hyper-validation..." << std::endl;

		assert(layers.size() == layers.size());

		//Might need other validations when network is more complex.

		hyperValidated = true;

		std::cout << "Validation passed." << std::endl << std::endl;

		return true;
	}

	//----------------------------------------------------------

	inline void Net::open(const string& file_name) {

		hyperValidated = false;
		trained = true;

		msgwarn("Net::open() is not implemented yet.");

		return;
	}

	//----------------------------------------------------------

	inline void Net::save(const string& file_name) {

		if (!trained) {
			msgwarn("The network has not been trained yet.");
		}
		if (!initialized) {
			msgerror("The network has not been initialized yet.");
		}
		if (!hyperValidated) {
			msgwarn("The hyperparameters of the network has not been validated yet.");
		}

		msgwarn("Net::save() is not implemented yet.");

		return;
	}

	//----------------------------------------------------------

	inline void Net::forward(shared_ptr<MatIntLoader> data_loader,
							 int epoch_idx, int batch_idx,
							 Mat& output_err, double& total_accuracy) {

		if (!hyperValidated) {
			msgerror("The network must pass validation before feeding forward.");
			return;
		}

		vector<Tensor> data = feed_sample_batch(data_loader);

		layers[0]->forward(data);						//input
		layers[1]->forward(layers[0]->units);			//conv1
		layers[2]->forward(layers[1]->units);			//maxpool1
		layers[3]->forward(layers[2]->units);			//conv2
		layers[4]->forward(layers[3]->units);			//maxpool2
		layers[5]->forward(layers[4]->units);			//convpool
		layers[6]->forward(layers[5]->units);			//output
		output_loss(data_loader, batch_idx, output_err, total_accuracy);

		return;
	}

	//----------------------------------------------------------

	inline void Net::backward(shared_ptr<MatIntLoader> data_loader,
							  int epoch_idx) {

		if (!hyperValidated) {
			msgerror("The network must pass validation before back propagation.");
			return;
		}

		msgnote("Can be rewritten as for loop.");
		layers[6]->backward(optimizer);					//output
		layers[5]->backward(optimizer);					//convpool
		layers[4]->backward(optimizer);					//maxpool2
		layers[3]->backward(optimizer);					//conv2
		layers[2]->backward(optimizer);					//maxpool1
		layers[1]->backward(optimizer);					//conv1

		return;
	}

	//----------------------------------------------------------

	inline void Net::train(shared_ptr<MatIntLoader> data_loader,
						   vector<Mat>& iteration_err) {

		if (!hyperValidated) {
			msgerror("The network must pass validation before training.");
			return;
		}

		if (!data_loader->is_initialized()) {
			msgerror("The feeding data has not been initialized.");
			return;
		}

		for (int _epoch_idx = 0; _epoch_idx < epochNum; _epoch_idx++) {

			std::cout << "Epoch " << _epoch_idx << ":" << std::endl;

			double _epoch_accuracy = 0.0;

			int _batch_num = data_loader->train_size() / batchSize;

			for (int _batch_idx = 0; _batch_idx < _batch_num; _batch_idx++) {

				Mat _loss;
				double _batch_accuracy;

				forward(data_loader, _epoch_idx, _batch_idx, _loss, _batch_accuracy);

				_epoch_accuracy += _batch_accuracy;
				if (_batch_idx % 100 == 0 && _batch_idx > 0) {
					std::cout << "Step: " << _batch_idx << ", Train Accuracy: " << _batch_accuracy << std::endl;
//					std::cout << "Loss: " << allErrors.back() << std::endl;
				}

				iteration_err.emplace_back(_loss);

				backward(data_loader, _epoch_idx);

			}

		}

		return;
	}

	//----------------------------------------------------------

	inline void Net::predict(shared_ptr<MatIntLoader> data_loader,
							 vector<Mat>& output) {

		if (!hyperValidated) {
			msgerror("The network must pass validation before applying in prediction.");
			return;
		}

		if (!data_loader->is_initialized()) {
			msgerror("The feeding data has not been initialized.");
			return;
		}

		/*------------------------------------------------------*/
		/* Add your snippets here.								*/
		/*------------------------------------------------------*/

		return;
	}

	//----------------------------------------------------------

	inline void Net::visualize() {

#ifdef LINNET_USE_OPENCV

		/*------------------------------------------------------*/
		/* Add your snippets here.								*/
		/*------------------------------------------------------*/

#else
		msgerror("Needs OpenCV for visualization.");
#endif

		return;

	}

	//----------------------------------------------------------

	inline void Net::print() {

		string output = "Network structure:\n";

		/*------------------------------------------------------*/
		/* Add your snippets here.								*/
		/*------------------------------------------------------*/

		std::cout << output << std::endl;

		return;

	}

	//----------------------------------------------------------

	inline void Net::init_units(id layer_idx) {
		Size data_size = layers[layer_idx]->dataSize;
		int unit_num = batchSize * layers[layer_idx]->outChannels;
		//layers[layer_idx]->units.reserve(unit_num);
		for (int i = 0; i < unit_num; i++) {
			Tensor new_tensor = Tensor::zeros(data_size);
			layers[layer_idx]->add_unit(new_tensor);
		}
		return;
	}

	//----------------------------------------------------------

	inline void Net::init_weights(id layer_idx, InitMethod init_method) {
		Size kernel_size = layers[layer_idx]->kernelSize;
		int weight_num = tools::get_weight_num(layers[layer_idx]);
		//layers[layer_idx]->weights.reserve(weight_num);
		for (int i = 0; i < weight_num; i++) {
			Mat new_mat = tools::init_weight(kernel_size, init_method, layers[layer_idx]->inChannels, layers[layer_idx]->outChannels);
			Tensor new_tensor(new_mat);
			layers[layer_idx]->add_weight(new_tensor);
		}
	}

	//----------------------------------------------------------

	inline void Net::init_biases(id layer_idx) {
		Size data_size = layers[layer_idx]->dataSize;
		Tensor new_tensor = Tensor::zeros(data_size);
		layers[layer_idx]->assign_bias(new_tensor);
	}

	//----------------------------------------------------------

	inline vector<Tensor> Net::feed_sample_batch(shared_ptr<MatIntLoader> data_loader) {

		vector<Tensor> output(batchSize);
		for (int i = 0; i < batchSize; i++) {
			output[i].assign_value(data_loader->next_train_sample().first);
		}
		return output;
	}

	//----------------------------------------------------------

	inline void Net::output_loss(shared_ptr<MatIntLoader> data_loader, int batch_idx, Mat& output_err, double& total_accuracy) {

		int _correct_count = 0;

		Mat _output;
		layers.back()->units;
		output_err = Mat::zeros(_output.size(), CV_64FC1);

		Mat show_correct = Mat::zeros(_output.size(), CV_64FC1);

		for (int j = 0; j < batchSize; j++) {

			int _predict = -1;
			double _max = -100000000.0;

			double *pdata = _output.ptr<double>(j);
			double *pshow = show_correct.ptr<double>(j);

			for (int i = 0; i < classNum; i++) {
				if (pdata[i] > _max) {
					_max = pdata[i];
					_predict = i;
				}
			}

			//Cast ground truth label to indices in output Mat.
			int _label = (int)data_loader->nth_train_sample(batch_idx - batchSize + j).second;

			if (_predict == _label) {
				_correct_count++;
			}

			pshow[_label] = 1.0;

		}

		Mat _output_err;
		_output.copyTo(_output_err);

		if (lossFunction = LossFunction::LINNET_LOSS_ABS) {

			for (int j = 0; j < batchSize; j++) {
				//Cast ground truth label to indices in output Mat.
				int _label = (int)data_loader->nth_train_sample(batch_idx - batchSize + j).second;
				_output_err.at<double>(j, _label) += -1.0;
			}

			output_err = cv::abs(_output_err);

		}
		else if (lossFunction = LossFunction::LINNET_LOSS_SQUARE) {

			for (int j = 0; j < batchSize; j++) {
				//Cast ground truth label to indices in output Mat.
				int _label = (int)data_loader->nth_train_sample(batch_idx - batchSize + j).second;
				_output_err.at<double>(j, _label) += -1.0;
			}

			output_err = _output_err.mul(_output_err, 0.5);

		}
		else if (lossFunction = LossFunction::LINNET_LOSS_ENTROPY) {

			for (int j = 0; j < batchSize; j++) {
				//Cast ground truth label to indices in output Mat.
				int _label = (int)data_loader->nth_train_sample(batch_idx - batchSize + j).second;
				output_err.at<double>(j, _label) = log(_output_err.at<double>(j, _label));
			}

		}
		else {
			msgerror("Invalid Loss Function");
			return;
		}

		total_accuracy = (double)_correct_count / (double)batchSize;

		return;

	}

}

#endif
