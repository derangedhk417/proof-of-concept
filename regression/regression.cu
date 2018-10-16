#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "defs.h"

#define MP_COUNT                       = 20
#define CORES_PER_MP                   = 128
#define MAX_THREADS_PER_BLOCK          = 1024
#define MAX_THREADS_PER_MULTIPROCESSOR = 2048

using namespace std;

double lowerBound;
double upperBound;
int    points;

double * devParameters;
double * 

void configure(double lb, double ub, int p) {
	lowerBound = lb;
	upperBound = ub;
	points     = p;

	double * devParameters;
	cudaMallocManaged(&devParameters, sizeof(double) * PARAMETER_COUNT);
}

void finish() {
	cudaFree(devParameters);
}

double getRMSE(double * parameters) {
	for (int i = 0; i < PARAMETER_COUNT; ++i) {
		devParameters[i] = parameters[i];
	}

	cudaDeviceSynchronize();


}