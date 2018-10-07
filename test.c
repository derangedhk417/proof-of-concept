#include <math.h>
#include <stdlib.h>
#include "mpi_controller.h"


struct MPIController * inst;

// Iniitializes the MPI comm world and returns the number
// of processes running.
int init() {
	printf("Init called\n");
	inst = createControllerInstance(
		"test_controller", "-n 4 ./primary_slave.o");

	int code;
	int length;
	int type;
	int * size = recvMessage(inst, &code, &length, &type);

	return *size;
}

void sendMatrix(float * matrix) {
	sendMessage(inst, matrix, 0, sizeof(float)*1024*6000, MSG_TYPE_FLOAT);
}

float * processArray(float * array) {
	sendMessage(inst, array, 0, sizeof(float)*1024, MSG_TYPE_FLOAT);

	int code;
	int len;
	int type;
	float * result = recvMessage(inst, &code, &len, &type);

	return result;
}

// double LennardJonesPotential(double r, double epsilon, double sigma) {
// 	return 4.0 * epsilon * (pow(sigma / r, 12) - pow(sigma / r, 6));
// }

// double * LennardJonesPotentialArray(double * r, double epsilon, double sigma, int n) {
// 	double * result = malloc(n * sizeof(double));
// 	for (int i = 0; i < n; ++i){
// 		result[i] = LennardJonesPotential(r[i], epsilon, sigma);
// 	}

// 	return result;
// }

// double * LennardJonesPotentialIntegralArray(double * r, double epsilon, double sigma, int n) {
// 	double * result = malloc(n * sizeof(double));
// 	for (int i = 0; i < n; ++i){
// 		double currentR      = r[i];
// 		double currentResult = 0.0;
// 		double R             = 1.0;
// 		while (R < currentR) {
// 			currentResult += 0.001 * LennardJonesPotential(R, epsilon, sigma);
// 			R += 0.001;
// 		}
// 		result[i] = currentResult;
// 	}

// 	return result;
// }