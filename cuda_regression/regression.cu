#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "defs.h"

#define MP_COUNT                       = 20
#define CORES_PER_MP                   = 128
#define MAX_THREADS_PER_BLOCK          = 1024
#define MAX_THREADS_PER_MULTIPROCESSOR = 2048

// #define THREADS_PER_BLOCK_TARGET 128
// #define BLOCK_TARGET             1024

using namespace std;

float lowerBound;
float upperBound;
float pointIncrement;
int   points;
int   pointsPerThread;
int   blocks;
int   threads;

int threadTarget;
int blockTarget;

float * d_Parameters; // Array where the parameters are copied before a call.
float * d_FitData;    // The fit data to compute RMSE from.
float * d_Sums;       // The array where partial sums are stored for error
                      // calculation. The CPU will take the square root if the
                      // sum of this array before returing,

float * h_Sums; // Host copy of the sums.

__device__ float fitFunction(float * p, float x) {
	return p[0]  * -powf(x - 1.0/1.0,  2.0) + 
		   p[1]  *  powf(x - 1.0/2.0,  2.0) + 
		   p[2]  * -powf(x - 1.0/3.0,  2.0) + 
		   p[3]  *  powf(x - 1.0/4.0,  3.0) + 
		   p[4]  * -powf(x - 1.0/5.0,  2.0) + 
		   p[5]  *  powf(x - 1.0/6.0,  2.0) + 
		   p[6]  * -powf(x - 1.0/7.0,  2.0) + 
		   p[7]  *  powf(x - 1.0/8.0,  3.0) + 
		   p[8]  * -powf(x - 1.0/9.0,  2.0) + 
		   p[9]  *  powf(x - 1.0/10.0, 2.0) + 
		   p[10] * -powf(x - 1.0/11.0, 2.0) + 
		   p[11] *  powf(x - 1.0/12.0, 3.0) + 
		   p[12] * -powf(x - 1.0/13.0, 2.0) + 
		   p[13] *  powf(x - 1.0/14.0, 2.0) + 
		   p[14] * -powf(x - 1.0/15.0, 2.0); 
}

__global__ void computeRMSE(
		float * params,         // Pointer to fit parameters.
		float * fit,            // Pointer to data to fit.
		float * sums,           // Pointer to array to put sums into.
		float   lowerBound,     // Lower bound of the domain.
		float   pointIncrement, // The distance between points in the domain.
		int     pointsPerThread // Number of data points each thread should process.
	) {

	float threadLowerBound =  (blockDim.x * blockIdx.x + threadIdx.x) * 
	                          pointIncrement * pointsPerThread;
	      threadLowerBound += lowerBound;

	int fitIndex = (blockDim.x * blockIdx.x + threadIdx.x) * pointsPerThread;

	for (int i = 0; i < pointsPerThread; ++i) {
		float index_X = threadLowerBound + i * pointIncrement;
		float val     = fitFunction(params, index_X) - fit[fitIndex + i];
		float result  = val * val;

		float * sumAddr = &sums[threadIdx.x];

		atomicAdd(sumAddr, result);
	}
}


extern "C" void configure(float lb, float ub, int p, float * data, int thr, int bl) {
	lowerBound = lb;
	upperBound = ub;
	points     = p;

	pointIncrement = (float)(((double)ub - (double)lb) / (double)p);

	cudaMalloc(&d_Parameters, sizeof(float) * PARAMETER_COUNT);
	cudaMalloc(&d_FitData,    sizeof(float) * points);
	cudaMalloc(&d_Sums,       sizeof(float) * thr);

	h_Sums = (float *)malloc(sizeof(float) * thr);

	cudaMemcpy(d_FitData, data, sizeof(float) * points, cudaMemcpyHostToDevice);

	// Here we determine the number of blocks and the
	// number of data points to process per thread. These
	// values are determined based on the size of the data
	// and the desired number of threads per block (threads).

	blocks  = bl;
	threads = thr;

	double ppt     = (double)points / (blocks * threads);
	while (ppt != floor(ppt)) {
		ppt = (double)points / (++blocks * threads);
	}

	pointsPerThread = floor(ppt);

	printf("CUDA Parameters:\n");
	printf("Blocks: %d\nThreads/Block: %d\nPoints/Thread: %d\nData Points: %d\n\n", 
		blocks, threads, pointsPerThread, points);
}

extern "C" void finish() {
	cudaFree(d_Parameters);
	cudaFree(d_FitData);
	cudaFree(d_Sums);
	free(h_Sums);
}

extern "C" float getRMSE(float * parameters) {
	cudaMemcpy(
		d_Parameters, parameters, sizeof(float) * PARAMETER_COUNT, cudaMemcpyHostToDevice);
	cudaMemset(d_Sums, 0, sizeof(float) * threads);

	computeRMSE<<<blocks, threads>>>(
		d_Parameters,
		d_FitData,
		d_Sums,
		lowerBound,
		pointIncrement,
		pointsPerThread
	);

	cudaMemcpy(
		h_Sums, d_Sums, sizeof(float) * threads, cudaMemcpyDeviceToHost);


	float sum = 0.0;
	for (int i = 0; i < threads; ++i) {
		sum += h_Sums[i];
	}

	return sqrtf(sum / (float)points);
}