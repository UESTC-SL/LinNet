#pragma once
#ifndef LINNET_OPTIM_INL
#define LINNET_OPTIM_INL

#include <vector>
#include <string>
#include <math.h>
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <type_traits>

#include "tensor.hpp"

using std::vector;
using std::string;
using std::pair;
using std::make_pair;
using std::map;
using std::shared_ptr;
using std::make_shared;

namespace lin {

	Optimizer::Optimizer() : optim_method(GradientMethod::LINNET_GD_SGD), learning_rate(1.0) {}

	Optimizer::~Optimizer() {
		clear();
	}

	Optimizer::Optimizer(GradientMethod gradient_method) : optim_method(gradient_method), learning_rate(1.0) {}

	Optimizer::Optimizer(GradientMethod gradient_method, double lr) : optim_method(gradient_method), learning_rate(lr) {}

	inline void Optimizer::clear() {}

	inline void Optimizer::init(double lr) {
		learning_rate = lr;
	}

	inline void Optimizer::step() {}

	//----------------------------------------------------------

	inline void OptimizerSGD::init(double lr) {
		optim_method = GradientMethod::LINNET_GD_SGD;
		learning_rate = lr;
	}

	inline void OptimizerSGD::step() {}

	//----------------------------------------------------------

	shared_ptr<Optimizer> create_optimizer(GradientMethod gradient_method) {
		if (gradient_method == GradientMethod::LINNET_GD_SGD) {
			return make_shared<OptimizerSGD>();
		} else {
			msgerror("GradientMethod not implemented: " + gradient_method);
		}
	}

}

#endif
