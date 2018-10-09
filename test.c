#include <math.h>
#include <stdlib.h>
#include "mpi_controller.h"

// This program acts as an intermediary between the Python script
// that is controlling everything and the MPI processes that are
// doing the intensive calculations.

#define WIDTH 1024
#define ROWS  6000

// This data structure, defined in mpi_controller.h, holds all
// of the information necessary to communicate with the rank 0
// process within the mpi world.
struct MPIController * inst;

// Iniitializes the MPI comm world and returns the number
// of processes running. The args argument of this function
// is passed to mpirun as command line arguments.
int init(char * args) {

    // Print the hostname that we are running in. Not necessary,
    // but helpful for debugging.
	char * hostname = malloc(sizeof(char) * 128);
	memset(hostname, 0, sizeof(char) * 128);
	gethostname(hostname, sizeof(char) * 128);
	printf("[CONTROLLER] Controller starting on %s\n", hostname);

    // Create and initialize a data structure for controlling
    // the mpi processes. This function will internally call
    // mpirun on a separate thread and wait until the rank0
    // process initializes its own instance using the same 
    // name.
	inst = createControllerInstance("test_controller", args);

    // Now we wait for the child instance to send back the number
    // of processes that were launched. Python will use this info
    // to create the necessary number of matrices.
	int code;
	int length;
	int type;
	int * size = recvMessage(inst, &code, &length, &type);

	return *size;
}

// Sends the specified array of double precision floating points
// to the rank0 MPI process. Blocks until the rank0 process 
// confirms that it has received the matrix.
void sendMatrix(double * matrix) {
	printf("[CONTROLLER] Sending matrix\n");
	sendMessage(inst, matrix, 0, sizeof(double)*WIDTH*ROWS, MSG_TYPE_DOUBLE);
	printf("[CONTROLLER] Matrix sent\n");
}

// Sends an array fo double precision floating point values to the 
// rank0 MPI process where it will in turn send the array to every 
// other MPI process. This function will then wait for results to
// be returned and will return those results to the caller, in an
// array.
double * processArray(double * array) {
    // Send the array.
	printf("[CONTROLLER] Sending array\n");
	sendMessage(inst, array, 0, sizeof(double)*WIDTH, MSG_TYPE_DOUBLE);
	printf("[CONTROLLER] Array sent\n");

    // Wait for a response.
	int code;
	int len;
	int type;
	printf("[CONTROLLER] Waiting for result\n");
	double * result = recvMessage(inst, &code, &len, &type);
	printf("[CONTROLLER] Received result\n");

	return result;
}