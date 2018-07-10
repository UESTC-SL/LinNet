#pragma once
#ifndef LINNET_OPTIM_HPP
#define LINNET_OPTIM_HPP

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

	class Optimizer {

	public:

		//default constructor
		Optimizer();
		//destructor
		virtual ~Optimizer();
		//constructor by GradientMethod
		explicit Optimizer(GradientMethod gradient_method);
		//constructor by GradientMethod and learning_rate
		explicit Optimizer(GradientMethod gradient_method, double lr);

		virtual void init(double lr);

		virtual void step();

		virtual void clear();

	protected:

		GradientMethod optim_method;

		double learning_rate;

	private:

		//Disable copy and copy-assignment.
		Optimizer(const Optimizer& rhs) {}
		Optimizer& operator = (const Optimizer& rhs) { return *this; }

	};

	//----------------------------------------------------------

	class OptimizerSGD : public Optimizer {

	public:

		OptimizerSGD() : Optimizer(GradientMethod::LINNET_GD_SGD) {}

		void init(double lr);

		void step();

	};

	//----------------------------------------------------------
	//Creates pointer to Optimizer based on GradientMethod.
	//----------------------------------------------------------
	shared_ptr<Optimizer> create_optimizer(GradientMethod gradient_method);

}

#include "optim.inl"

#endif
