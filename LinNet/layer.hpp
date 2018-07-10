#pragma once
#ifndef LINNET_LAYER_HPP
#define LINNET_LAYER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <utility>

#include "function.hpp"
#include "optim.hpp"

using std::vector;
using std::pair;
using std::make_pair;
using std::string;


namespace lin {

	//----------------------------------------------------------
	//Defines Layer structure and parameters.
	//----------------------------------------------------------
	class Layer {

	public:

		string layerName;

		LayerType layerType;
		FuncType funcType;

		ConvType convType;
		BorderType borderType;		//useless for now

		Size dataSize;
		Size kernelSize;			//Size of kernel (for convolution) or pooling window (for Max-Pooling).

		int inChannels;				//Number of input channels.
		int outChannels;			//Number of output channels.

		vector<Tensor> units;				//Values in all units.
		vector<Tensor> weights;				//Values in all weights.
		Tensor bias;						//Values of bias.

		shared_ptr<vector<Tensor>> inputPtr;

		//default constructor
		Layer() : enabled(true) {}
		Layer(const Layer& rhs) {}
		Layer& operator = (const Layer& rhs) { return *this; }
		Layer(LayerType layer_type) : layerType(layer_type), enabled(true) {}
		Layer(LayerType layer_type, FuncType func_type, int in_channels, int out_channels, Size data_size,
			  Size kernel_size=Size(0,0), ConvType conv_type=ConvType::LINNET_CONV_VALID,
			  BorderType border_type=BorderType::LINNET_BORDER_COPY, string layer_name="") :
			layerType(layer_type), funcType(func_type), convType(conv_type), borderType(border_type),
			dataSize(data_size), kernelSize(kernel_size), inChannels(in_channels), outChannels(out_channels),
			layerName(layer_name), enabled(true) {}
		//destructor
		virtual ~Layer() {
			clear();
		}

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		void add_unit(Tensor new_tensor);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		void assign_unit(Tensor new_tensor, id pos);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		void add_weight(Tensor new_weight);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		void assign_weight(Tensor new_tensor, id pos);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		void add_bias(Tensor new_bias);

		//----------------------------------------------------------
		//
		// Arg `pos` serves no purpose.
		//----------------------------------------------------------
		void assign_bias(Tensor new_bias, id pos=0);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		virtual void clear();

		//----------------------------------------------------------
		//Forward method. Needs to be overrided for specific layer.
		//----------------------------------------------------------
		virtual void forward(vector<Tensor>& input) = 0;

		//----------------------------------------------------------
		//Backward method. Needs to be overrided for specific layer.
		//----------------------------------------------------------
		virtual void backward(shared_ptr<Optimizer> optimizer) = 0;

		//----------------------------------------------------------
		//Check if the Layer is enabled.
		//----------------------------------------------------------
		bool is_enabled();

		//----------------------------------------------------------
		//Enable the Layer.
		//----------------------------------------------------------
		void enable();

		//----------------------------------------------------------
		//Disable the Layer.
		//----------------------------------------------------------
		void disable();

		//----------------------------------------------------------
		//Inverse the enablity of the Layer.
		//----------------------------------------------------------
		void inv_enabled();

	protected:

		bool enabled;

	};

	class InputLayer : public Layer {
		virtual void forward(vector<Tensor>& input);
		virtual void backward(shared_ptr<Optimizer> optimizer);
	};

	class ConvLayer : public Layer {
		virtual void forward(vector<Tensor>& input);
		virtual void backward(shared_ptr<Optimizer> optimizer);
	};

	class MaxPoolLayer : public Layer {
		virtual void forward(vector<Tensor>& input);
		virtual void backward(shared_ptr<Optimizer> optimizer);
	};

	class LinearLayer : public Layer {
		virtual void forward(vector<Tensor>& input);
		virtual void backward(shared_ptr<Optimizer> optimizer);
	};

	class OutputLayer : public Layer {
		virtual void forward(vector<Tensor>& input);
		virtual void backward(shared_ptr<Optimizer> optimizer);
	};

	class FlattenLayer : public Layer {
		virtual void forward(vector<Tensor>& input);
		virtual void backward(shared_ptr<Optimizer> optimizer);
	};

}

#include "layer.inl"

#endif
