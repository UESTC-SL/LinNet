#pragma once
#ifndef LINNET_LINNET_HPP
#define LINNET_LINNET_HPP

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
using std::shared_ptr;

namespace lin {

	using dataloader::MatIntLoader;

	//----------------------------------------------------------
	//
	//----------------------------------------------------------
	class Net {
	public:

		int batchSize;
		int epochNum;
		int imageChannels;
		int classNum;

		shared_ptr<Optimizer> optimizer;

		LossFunction lossFunction;

		Size inputSize;

		//All loss.
		vector<double> loss;

		vector<shared_ptr<Layer>> layers;

		//default constructor
		Net();
		//destructor
		virtual ~Net();

		virtual void clear();

		//----------------------------------------------------------
		//Start constructing network XML file instructs.
		//The following values of LayerDef are initialized:
		//	layerName, layerType, funcType, convType, borderType,
		//  inChannels, outChannels, kernelSize
		//----------------------------------------------------------
		virtual void init(const string& file_name);

		//----------------------------------------------------------
		//Initialize all variables in the network.
		//The following values of LayerDef are initialized:
		//  dataSize
		//The following values of Layer are initialized:
		//  units, weights, bias
		//----------------------------------------------------------
		virtual void initialize_variables();

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		virtual bool hyper_validate();

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		virtual void open(const string& file_name);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		virtual void save(const string& file_name);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		void forward(shared_ptr<MatIntLoader> data_loader,
					 int epoch_idx, int batch_idx,
					 Mat& output_err, double& total_accuracy);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		void backward(shared_ptr<MatIntLoader> data_loader,
					  int epoch_idx);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		void train(shared_ptr<MatIntLoader> data_loader,
				   vector<Mat>& iteration_err);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		void predict(shared_ptr<MatIntLoader> data_loader,
					 vector<Mat>& output);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		virtual void visualize();

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		virtual void print();

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		string author() {
			return "John Hany, johnhany@163.com, http://johnhany.net";
		}

	protected:

		bool initialized;
		bool hyperValidated;
		bool trained;

		//std::default_random_engine stdRNG;

		cv::RNG rng;

		int iterIdx;
		int epochIdx;

		//----------------------------------------------------------
		//Initialize unit values with 0.
		//----------------------------------------------------------
		void init_units(id set_idx);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		void init_weights(id layer_idx, InitMethod init_method);

		//----------------------------------------------------------
		//Initialize bias values with 0.
		//----------------------------------------------------------
		void init_biases(id layer_idx);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		vector<Tensor> feed_sample_batch(shared_ptr<MatIntLoader> data_loader);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		void output_loss(shared_ptr<MatIntLoader> data_loader, int batch_idx, Mat& output_err, double& total_accuracy);

	private:

		//Disable copy and copy-assignment
		Net(const Net& rhs) {}
		Net& operator = (const Net& rhs) { return *this; }

	};



}

#include "linnet.inl"

#endif
