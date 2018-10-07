#include <math.h>
#include <stdlib.h>
#include "mpi_controller.h"


struct MPIController * inst;

// Iniitializes the MPI comm world and returns the number
// of processes running.
int init() {
	inst = createControllerInstance(
		"test_controller", "-n 4 ./primary_slave.o");

	int code;
	int length;
	int type;
	int * size = recvMessage(inst, &code, &length, &type);

	return *size;
}

void sendMatrix(double * matrix) {
	sendMessage(inst, matrix, 0, sizeof(double)*1024*6000, MSG_TYPE_FLOAT);
}

double * processArray(double * array) {
	sendMessage(inst, array, 0, sizeof(double)*1024, MSG_TYPE_FLOAT);

	int code;
	int len;
	int type;
	double * result = recvMessage(inst, &code, &len, &type);

	return result;
}