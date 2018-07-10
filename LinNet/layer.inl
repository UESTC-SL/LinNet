#pragma once
#ifndef LINNET_LAYER_INL
#define LINNET_LAYER_INL

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

	void Layer::add_unit(Tensor new_tensor) {
		units.push_back(new_tensor);
	}

	void Layer::assign_unit(Tensor new_tensor, id pos) {
		try {
			units.at(pos) = new_tensor;
		} catch (const std::out_of_range& e) {
			msgerror("Vector `units` out of Range.");
		}
	}

	void Layer::add_weight(Tensor new_tensor) {
		weights.push_back(new_tensor);
	}

	void Layer::assign_weight(Tensor new_tensor, id pos) {
		try {
			weights.at(pos) = new_tensor;
		} catch (const std::out_of_range& e) {
			msgerror("Vector `weights` out of Range.");
		}
	}

	void Layer::add_bias(Tensor new_tensor) {
		bias += new_tensor;
	}

	void Layer::assign_bias(Tensor new_tensor, id pos) {
		bias = new_tensor;
	}

	inline void Layer::clear() {
		units.clear();
		weights.clear();
		bias.clear();
		disable();
	}

	inline bool Layer::is_enabled() {
		return enabled;
	}

	inline void Layer::enable() {
		enabled = true;
	}

	inline void Layer::disable() {
		enabled = false;
	}

	inline void Layer::inv_enabled() {
		enabled = !enabled;
	}

	//----------------------------------------------------------

	inline void InputLayer::forward(vector<Tensor>& input) {

		assert(units.size() == input.size());
		for (int i = 0; i < units.size(); i++) {
			units[i] = input[i];
		}

		inputPtr = make_shared<vector<Tensor>>(input);
	}

	inline void InputLayer::backward(shared_ptr<Optimizer> optimizer) {

		/*------------------------------------------------------*/
		/* Add your snippets here.								*/
		/*------------------------------------------------------*/

	}

	//----------------------------------------------------------

	inline void OutputLayer::forward(vector<Tensor>& input) {

		int batch_size = (int)units.size() / outChannels;
		assert(outChannels * batch_size == units.size());
		assert(input[0].size() == Size(1, 1));

		/*------------------------------------------------------*/
		/* Add your snippets here.								*/
		/*------------------------------------------------------*/

		inputPtr = make_shared<vector<Tensor>>(input);
	}

	inline void OutputLayer::backward(shared_ptr<Optimizer> optimizer) {

		/*------------------------------------------------------*/
		/* Add your snippets here.								*/
		/*------------------------------------------------------*/

	}

	//----------------------------------------------------------

	inline void ConvLayer::forward(vector<Tensor>& input) {

		int batch_size = (int)units.size() / outChannels;
		assert(outChannels * batch_size == units.size());
		assert(units[0].size().checkpositive());
		
		/*------------------------------------------------------*/
		/* Add your snippets here.								*/
		/*------------------------------------------------------*/

		inputPtr = make_shared<vector<Tensor>>(input);
	}

	inline void ConvLayer::backward(shared_ptr<Optimizer> optimizer) {

		/*------------------------------------------------------*/
		/* Add your snippets here.								*/
		/*------------------------------------------------------*/

	}

	//----------------------------------------------------------

	inline void MaxPoolLayer::forward(vector<Tensor>& input) {

		assert(units.size() == input.size());
		
		/*------------------------------------------------------*/
		/* Add your snippets here.								*/
		/*------------------------------------------------------*/

		inputPtr = make_shared<vector<Tensor>>(input);
	}

	inline void MaxPoolLayer::backward(shared_ptr<Optimizer> optimizer) {

		/*------------------------------------------------------*/
		/* Add your snippets here.								*/
		/*------------------------------------------------------*/

	}

	//----------------------------------------------------------

	inline void LinearLayer::forward(vector<Tensor>& input) {

		/*------------------------------------------------------*/
		/* Add your snippets here.								*/
		/*------------------------------------------------------*/

		inputPtr = make_shared<vector<Tensor>>(input);
	}

	inline void LinearLayer::backward(shared_ptr<Optimizer> optimizer) {

		/*------------------------------------------------------*/
		/* Add your snippets here.								*/
		/*------------------------------------------------------*/

	}

	//----------------------------------------------------------

	inline void FlattenLayer::forward(vector<Tensor>& input) {

		/*------------------------------------------------------*/
		/* Add your snippets here.								*/
		/*------------------------------------------------------*/

		inputPtr = make_shared<vector<Tensor>>(input);
	}

	inline void FlattenLayer::backward(shared_ptr<Optimizer> optimizer) {

		/*------------------------------------------------------*/
		/* Add your snippets here.								*/
		/*------------------------------------------------------*/

	}

}

#endif
