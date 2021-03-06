// LinNet.cpp : Defines the entry point for the console application.
//
#include <stdio.h>
#include <time.h>

#include <iostream>

#include "linnet.hpp"

using namespace std;

int main()
{
	lin::Net network;
	network.init("hyperparas.xml");

	lin::MNISTLoader dataLoader;
	dataLoader.init("G:\\Datasets\\Classifications\\mnist\\");

	vector<lin::Mat> errs;
	network.train(make_shared<lin::MatIntLoader>(dataLoader), errs);

	vector<lin::Mat> predicts;
	network.predict(make_shared<lin::MatIntLoader>(dataLoader), predicts);

	network.visualize();

	return 0;
}

