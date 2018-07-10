#pragma once
#ifndef LINNET_DATALOADER_INL
#define LINNET_DATALOADER_INL

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

		template<typename DType, typename LType>
		inline pair<DType, LType> DataLoader<DType, LType>::next_train_sample() {
			if (!initialized)
				msgerror("DataLoader is not initialized!");

			if (trainIdx >= trainSamples.size()) {
				shuffle();
				trainIdx = 0;
			}
			pair<DType, LType> output = nth_train_sample(trainIdx);
			trainIdx++;

			return output;
		}

		//----------------------------------------------------------

		template<typename DType, typename LType>
		inline pair<DType, LType> DataLoader<DType, LType>::next_test_sample() {
			if (!initialized)
				msgerror("DataLoader is not initialized!");

			if (testIdx >= testSamples.size()) {
				shuffle();
				testIdx = 0;
			}
			pair<DType, LType> output = nth_test_sample(testIdx);
			testIdx++;

			return output;
		}

		//----------------------------------------------------------

		template<typename DType, typename LType>
		inline pair<DType, LType> DataLoader<DType, LType>::nth_train_sample(id pos) {
			if (!initialized)
				msgerror("DataLoader is not initialized!");
			assert(pos < trainSamples.size());

			std::vector<id>::iterator itr = trainIndices.begin();
			std::advance(itr, (int)pos);
			return trainSamples[*itr];
		}

		//----------------------------------------------------------

		template<typename DType, typename LType>
		inline pair<DType, LType> DataLoader<DType, LType>::nth_test_sample(id pos) {
			if (!initialized)
				msgerror("DataLoader is not initialized!");
			assert(pos < testSamples.size());

			std::vector<id>::iterator itr = trainIndices.begin();
			std::advance(itr, (int)pos);
			return testSamples[*itr];
		}

		//----------------------------------------------------------

		template<typename DType, typename LType>
		inline void DataLoader<DType, LType>::shuffle() {

			std::random_shuffle(trainIndices.begin(), trainIndices.end());
			//std::random_shuffle(testIndices.begin(), testIndices.end());

		}

		//----------------------------------------------------------

		inline void MNISTLoader::init(const string& mnist_path) {

			if (loadMNIST(mnist_path + "train-images.idx3-ubyte", mnist_path + "train-labels.idx1-ubyte", trainSamples)
				|| loadMNIST(mnist_path + "t10k-images.idx3-ubyte", mnist_path + "t10k-labels.idx1-ubyte", testSamples)) {
				msgerror("Cannot load MNIST dataset.");
			}

			for (id i = 0; i < trainSamples.size(); i++) {
				trainIndices.emplace_back(i);
			}
			for (id i = 0; i < testSamples.size(); i++) {
				testIndices.emplace_back(i);
			}

			trainIdx = 0;
			testIdx = 0;

			initialized = true;

		}

		//----------------------------------------------------------

		inline int reverseInt(int i) {
			unsigned char ch1, ch2, ch3, ch4;
			ch1 = i & 255;
			ch2 = (i >> 8) & 255;
			ch3 = (i >> 16) & 255;
			ch4 = (i >> 24) & 255;

			return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
		}

		//----------------------------------------------------------

		inline int loadMNIST(const string pic_filename, const string label_filename, vector<pair<Mat, int>> &sample_set) {
			std::ifstream pic_file(pic_filename, std::ios::binary);
			std::ifstream label_file(label_filename, std::ios::binary);

			if (pic_file.is_open() && label_file.is_open()) {
				int magic_number = 0;
				int number_of_images = 0;
				int n_rows = 0;
				int n_cols = 0;

				label_file.read((char*)&magic_number, sizeof(magic_number));
				pic_file.read((char*)&magic_number, sizeof(magic_number));
				magic_number = reverseInt(magic_number);

				label_file.read((char*)&number_of_images, sizeof(number_of_images));
				pic_file.read((char*)&number_of_images, sizeof(number_of_images));
				number_of_images = reverseInt(number_of_images);

				pic_file.read((char*)&n_rows, sizeof(n_rows));
				n_rows = reverseInt(n_rows);
				pic_file.read((char*)&n_cols, sizeof(n_cols));
				n_cols = reverseInt(n_cols);

//				for (int i = 0; i < number_of_images; ++i) {
				for (int i = 0; i < 1000; ++i) {
					char label = 0;
					label_file.read((char*)&label, sizeof(label));

					Mat sample = Mat(n_rows, n_cols, CV_64FC1);
					for (int r = 0; r < n_rows; ++r) {
						double *p_out = sample.ptr<double>(r);
						for (int c = 0; c < n_cols; ++c) {
							unsigned char tmp = 0;
							pic_file.read((char*)&tmp, sizeof(tmp));
							p_out[c] = (double)tmp;
						}
					}
					sample_set.emplace_back(make_pair(sample, (int)(label - '0')));
				}
			}
			else {
				return 1;
			}
			return 0;
		}

	}

}

#endif
