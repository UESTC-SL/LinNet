#pragma once
#ifndef LINNET_DATALOADER_HPP
#define LINNET_DATALOADER_HPP

#include <fstream>
#include <list>
#include <string>
#include <utility>
#include <vector>

#include "variables.hpp"

using std::vector;
using std::pair;
using std::make_pair;
using std::string;
using std::list;

namespace lin {

	namespace dataloader {

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		template<typename DType, typename LType>
		class DataLoader {
		public:

			DataLoader() : initialized(false) {}

			virtual ~DataLoader() {
				trainSamples.clear();
				trainSamples.shrink_to_fit();
				testSamples.clear();
				testSamples.shrink_to_fit();
				trainIndices.clear();
				testIndices.clear();
			}

			//----------------------------------------------------------
			//
			//----------------------------------------------------------
			virtual void init(const string& data_path) = 0;

			//----------------------------------------------------------
			//
			//----------------------------------------------------------
			virtual pair<DType, LType> next_train_sample();

			//----------------------------------------------------------
			//
			//----------------------------------------------------------
			virtual pair<DType, LType> next_test_sample();

			//----------------------------------------------------------
			//
			//----------------------------------------------------------
			virtual pair<DType, LType> nth_train_sample(id pos);

			//----------------------------------------------------------
			//
			//----------------------------------------------------------
			virtual pair<DType, LType> nth_test_sample(id pos);

			//----------------------------------------------------------
			//Rearrange train and test samples by random orders.
			//`trainIdx` and `testIdx` will NOT be set to 0.
			//----------------------------------------------------------
			virtual void shuffle();

			//----------------------------------------------------------
			//
			//----------------------------------------------------------
			bool is_initialized() {
				return initialized;
			}

			//----------------------------------------------------------
			//
			//----------------------------------------------------------
			int train_size() {
				return (int)trainSamples.size();
			}

			//----------------------------------------------------------
			//
			//----------------------------------------------------------
			int test_size() {
				return (int)testSamples.size();
			}

		protected:

			bool initialized;

			id trainIdx;
			id testIdx;

			vector<pair<DType, LType>> trainSamples;
			vector<pair<DType, LType>> testSamples;

			vector<id> trainIndices;
			vector<id> testIndices;

		};

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		class MatIntLoader : public DataLoader<Mat, int> {
		public:
			virtual void init(const string& mnist_path) {
				msgerror("Not implemented.");
			}
		};
		
		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		class MNISTLoader : public MatIntLoader {
		public:

			MNISTLoader() : MatIntLoader() {}

			virtual ~MNISTLoader() {}

			virtual void init(const string& mnist_path);

		};

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		int reverseInt(int i);

		//----------------------------------------------------------
		//
		//----------------------------------------------------------
		//template<typename DType, typename LType>
		int loadMNIST(const string pic_filename, const string label_filename, vector<pair<Mat, int>> &sample_set);

	}

	using dataloader::MNISTLoader;

}

#include "dataloader.inl"

#endif
