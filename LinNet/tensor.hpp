#pragma once
#ifndef LINNET_TENSOR_HPP
#define LINNET_TENSOR_HPP

#include "global.hpp"
#include "variables.hpp"

namespace lin {

	class Tensor {

	public:

		// Tensor value.
		Mat value;

		// Derivate for back-propagation.
		Mat grad;

		//default constructor
		Tensor();
		//copy constructor
		Tensor(const Tensor& rhs);
		//move constructor
		Tensor(Tensor&& rhs);
		//copy/move assignment
		Tensor& operator =(Tensor rhs);
		//operator ==
		bool operator ==(Tensor rhs);
		//operator !=, defined by operator ==
		bool operator !=(Tensor rhs);
		//operate +
		Tensor operator +(const Tensor& rhs);
		//operate +=
		Tensor& operator +=(const Tensor& rhs);
		//destructor
		virtual ~Tensor();
			
		//Initialize Tensor with value Mat.
		explicit Tensor(Mat var);

		//Initialize Tensor with value Mat and grad mat.
		explicit Tensor(Mat var, Mat new_grad);

		//Initialize Tensor based on Size.
		//All values will be initialized as 0.
		explicit Tensor(Size size);

		//custom swap() function for Tensor
		friend void swap(Tensor& lhs, Tensor& rhs) noexcept {
			using std::swap;
			assert(lhs.with_grad() == rhs.with_grad());
			swap(lhs.value, rhs.value);
			swap(lhs.grad, rhs.grad);
		}

		//----------------------------------------------------------
		//Assign value, and leave grad untouched.
		//----------------------------------------------------------
		void assign_value(Mat var);

		//----------------------------------------------------------
		//Assign grad, and leave value untouched.
		//----------------------------------------------------------
		void assign_grad(Mat new_grad);

		//----------------------------------------------------------
		//Apply gradient to value based on grad and scale.
		//----------------------------------------------------------
		void apply_grad(double scale);

		//----------------------------------------------------------
		//Assign all values as zero, in-place.
		//----------------------------------------------------------
		void zeros_(Size size);

		//----------------------------------------------------------
		//Returns a Tensor with all values as zero.
		//----------------------------------------------------------
		static Tensor zeros(Size size);

		//----------------------------------------------------------
		//Returns the Size of Tensor.
		//----------------------------------------------------------
		Size size();

		//----------------------------------------------------------
		//Makes the Tensor derivable.
		//----------------------------------------------------------
		void require_grad();

		//----------------------------------------------------------
		//Makes the Tensor underivable.
		//----------------------------------------------------------
		void require_no_grad();

		//----------------------------------------------------------
		//Check if the Tensor is derivable.
		//----------------------------------------------------------
		bool with_grad();

		//Clear all elements in Tensor.
		virtual void clear();

	protected:

		//Defines if the Tensor is derivable.
		bool derivable;

	};
}

#include "tensor.inl"

#endif
