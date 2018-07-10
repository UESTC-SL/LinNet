#pragma once
#ifndef LINNET_TENSOR_INL
#define LINNET_TENSOR_INL

#include "global.hpp"
#include "variables.hpp"


namespace lin {

	Tensor::Tensor() : value(), grad(), derivable(true) {}

	Tensor::Tensor(const Tensor& rhs) : value(rhs.value), grad(rhs.grad), derivable(rhs.derivable) {}

	Tensor::Tensor(Tensor&& rhs) : value(rhs.value), grad(rhs.grad), derivable(rhs.derivable) {
		rhs.clear();
	}

	Tensor& Tensor::operator =(Tensor rhs) {
		swap(*this, rhs);
		return *this;
	}

	bool Tensor::operator ==(Tensor rhs) {
		msgerror("Not implemented.");
		return true;
	}

	bool Tensor::operator !=(Tensor rhs) {
		msgerror("Not implemented.");
		return false;
	}

	Tensor Tensor::operator +(const Tensor& rhs) {
		Tensor rst;
		rst.value = value + rhs.value;
		return rst;
	}

	Tensor& Tensor::operator +=(const Tensor& rhs) {
		value += rhs.value;
		return *this;
	}

	Tensor::~Tensor() {
		clear();
	}

	Tensor::Tensor(Mat var) : value(var) {
		assert(!value.empty());
		grad.release();
		grad = Mat::zeros(var.size(), CV_64FC1);
	}

	Tensor::Tensor(Mat var, Mat new_grad) : value(var), grad(new_grad) {
		assert(!value.empty() && !grad.empty());
	}

	Tensor::Tensor(Size size) {
		assert(size.checkpositive());
		zeros_(size);
	}

	//----------------------------------------------------------

	inline void Tensor::assign_value(Mat var) {
		assert(!var.empty());
		value.release();
		value = var;
	}

	inline void Tensor::assign_grad(Mat new_grad) {
		assert(!new_grad.empty());
		grad.release();
		grad = new_grad;
	}

	inline void Tensor::apply_grad(double scale) {
		assert(!value.empty() && !grad.empty());
		value += scale * grad;
	}

	//----------------------------------------------------------

	inline void Tensor::zeros_(Size size) {
		assert(size.checkpositive());
		value.release();
		grad.release();
		value = Mat::zeros(cv::Size(size.h, size.w), CV_64FC1);
		grad = Mat::zeros(cv::Size(size.h, size.w), CV_64FC1);
	}

	inline Tensor Tensor::zeros(Size size) {
		assert(size.checkpositive());
		Tensor rst;
		rst.value = Mat::zeros(cv::Size(size.h, size.w), CV_64FC1);
		rst.grad = Mat::zeros(cv::Size(size.h, size.w), CV_64FC1);
		return rst;
	}

	//----------------------------------------------------------

	inline Size Tensor::size() {
		return Size(value.cols, value.rows);
	}

	//----------------------------------------------------------

	inline void Tensor::require_grad() {
		derivable = true;
	}

	inline void Tensor::require_no_grad() {
		derivable = false;
	}

	inline bool Tensor::with_grad() {
		return derivable;
	}

	inline void Tensor::clear() {
		value.release();
		grad.release();
	}
}

#endif
