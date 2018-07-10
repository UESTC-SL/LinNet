#pragma once
#ifndef LINNET_VARIABLES_HPP
#define LINNET_VARIABLES_HPP

#include "global.hpp"

#ifdef LINNET_USE_OPENCV
#include <opencv2\opencv.hpp>
#endif

namespace lin {

	//----------------------------------------------------------
	//Index of a Layer.
	//----------------------------------------------------------
	typedef size_t id;

	using cv::Vec;
	using cv::Mat;

	//----------------------------------------------------------
	//Custom data type for representing dimensions of Matrices
	//----------------------------------------------------------
	class Size {
	public:
		int w;
		int h;

		//default constructor
		Size();
		//copy constructor
		Size(const Size& rhs);
		//move constructor
		Size(Size&& rhs);
		//copy/move assignment
		Size& operator =(Size rhs);
		//operator ==
		bool operator ==(Size rhs);
		//operator !=, defined by operator ==
		bool operator !=(Size rhs);
		//destructor
		virtual ~Size();

		//Initialize Size with (int, int).
		explicit Size(int width, int height);
		//Initialize Size with (id, id).
		explicit Size(id width, id height);

		//custom swap() function for Size
		friend void swap(Size& lhs, Size& rhs) noexcept {
			using std::swap;
			swap(lhs.w, rhs.w);
			swap(lhs.h, rhs.h);
		}

		//Clear all elements in Size.
		virtual void clear();

		//Change size of Size.
		virtual void resize(int x, int y);

		//Check if both w and h are no smaller than 0.
		virtual bool checknonzero();

		//Check if both w and h are larger than 0.
		virtual bool checkpositive();

	};

	//----------------------------------------------------------
	//Defines layer type.
	//----------------------------------------------------------
	enum LayerType {
		LINNET_LAYER_INPUT,
		LINNET_LAYER_OUTPUT,
		LINNET_LAYER_CONV,
		LINNET_LAYER_LINEAR,
		LINNET_LAYER_MAXPOOL,
		LINNET_LAYER_FLATTEN
	};

	//----------------------------------------------------------
	//Defines activation function.
	//----------------------------------------------------------
	enum FuncType {
		LINNET_FUNC_IDENTITY,
		LINNET_FUNC_RELU,
		LINNET_FUNC_SIGMOID,
		LINNET_FUNC_TANH,
		LINNET_FUNC_SOFTMAX
	};

	//----------------------------------------------------------
	//Defines size changes after convolution. 
	//  `LINNET_CONV_SAME`: same size. 
	//  `LINNET_CONV_VALID`: out_size = in_size - 2 * (ker_size / 2).
	//----------------------------------------------------------
	enum ConvType {
		LINNET_CONV_SAME,
		LINNET_CONV_VALID
	};

	//----------------------------------------------------------
	//Defines border behavior. Only works when applying ConvType::LINNET_CONV_SAME
	//  `LINNET_BORDER_COPY`: fill in with duplicated pixels near border. 
	//  `LINNET_BORDER_ZEROS`: fill in with zeros. 
	//----------------------------------------------------------
	enum BorderType {
		LINNET_BORDER_COPY,
		LINNET_BORDER_ZEROS
	};

	//----------------------------------------------------------
	//Defines loss function.
	//	`LINNET_LOSS_ABS`: Absolute loss.
	//	`LINNET_LOSS_SQUARE`: Squared loss.
	//	`LINNET_LOSS_ETP`: Cross-entropy loss.
	//----------------------------------------------------------
	enum LossFunction {
		LINNET_LOSS_ABS,
		LINNET_LOSS_SQUARE,
		LINNET_LOSS_ENTROPY
	};

	//----------------------------------------------------------
	//Defines gradient method.
	//----------------------------------------------------------
	enum GradientMethod {
		LINNET_GD_SGD,
		LINNET_GD_MOMENTUM
	};

	//----------------------------------------------------------
	//Defines weight initialization method.
	//----------------------------------------------------------
	enum InitMethod {
		LINNET_INIT_XAVIER_NORM,
		LINNET_INIT_XAVIER_UNIFORM,
		LINNET_INIT_HE_NORM,
		LINNET_INIT_HE_UNIFORM
	};

}

#include "variables.inl"

#endif
