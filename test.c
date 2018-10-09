#include <math.h>
#include <stdlib.h>
#include "mpi_controller.h"

#define WIDTH 1024
#define ROWS  6000

struct MPIController * inst;

// Iniitializes the MPI comm world and returns the number
// of processes running.
int init(char * args) {
	char * hostname = malloc(sizeof(char) * 128);
	memset(hostname, 0, sizeof(char) * 128);
	gethostname(hostname, sizeof(char) * 128);
	printf("[CONTROLLER] Controller starting on %s\n", hostname);
	// inst = createControllerInstance(
	// 	"test_controller", "-n 8 -hosts localhost,ubuntu-vm ./primary_slave.o");
	inst = createControllerInstance(
		"test_controller", args);

	int code;
	int length;
	int type;
	int * size = recvMessage(inst, &code, &length, &type);

	return *size;
}

void sendMatrix(double * matrix) {
	printf("[CONTROLLER] Sending matrix\n");
	sendMessage(inst, matrix, 0, sizeof(double)*WIDTH*ROWS, MSG_TYPE_DOUBLE);
	printf("[CONTROLLER] Matrix sent\n");
}

double * processArray(double * array) {
	printf("[CONTROLLER] Sending array\n");
	sendMessage(inst, array, 0, sizeof(double)*WIDTH, MSG_TYPE_DOUBLE);
	printf("[CONTROLLER] Array sent\n");
	int code;
	int len;
	int type;
	printf("[CONTROLLER] Waiting for result\n");
	double * result = recvMessage(inst, &code, &len, &type);
	printf("[CONTROLLER] Received result\n");

	return result;
}