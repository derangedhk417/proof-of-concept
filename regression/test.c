#include <math.h>
#include <stdlib.h>
#include "mpi_controller.h"
#include "defs.h"



// This data structure, defined in mpi_controller.h, holds all
// of the information necessary to communicate with the rank 0
// process within the mpi world.
struct MPIController * inst;

// The number of processes running.
int processCount;  

// Number of data points being processed.
int pointCount;

// Iniitializes the MPI comm world and returns the number
// of processes running. The args argument of this function
// is passed to mpirun as command line arguments.
int init(char * args) {
    // Create and initialize a data structure for controlling
    // the mpi processes. This function will internally call
    // mpirun on a separate thread and wait until the rank0
    // process initializes its own instance using the same 
    // name.
	inst = createControllerInstance("regression_controller", args);

    // Now we wait for the child instance to send back the number
    // of processes that were launched. Python will use this info
    // to create the necessary number of matrices.
	int code;
	int length;
	int type;
	int * size = recvMessage(inst, &code, &length, &type);

	processCount = *size;

	return *size;
}

// Sends the size of the data to the child nodes.
// At this point, it is assumed that there are 
// nDataPoints * sizeof(double) x - values and
// nDataPoints * sizeof(double) y - values.
void configureDataSize(double lowerBound, double upperBound, int points) {
	sendMessage(
		inst, 
		&lowerBound, 
		SEND_DATA_DIMENSIONS, 
		sizeof(int), 
		MSG_TYPE_DOUBLE
	);

	sendMessage(
		inst, 
		&upperBound, 
		SEND_DATA_DIMENSIONS, 
		sizeof(int), 
		MSG_TYPE_DOUBLE
	);

	sendMessage(
		inst, 
		&points, 
		SEND_DATA_DIMENSIONS, 
		sizeof(int), 
		MSG_TYPE_INT
	);

	pointCount = points;
}

// Sends the x and y values of the data we are fitting to the
// child nodes.
void sendData(double * data) {
	sendMessage(
		inst, 
		data, 
		SEND_DATA, 
		sizeof(double) * pointCount,
		MSG_TYPE_DOUBLE                                  
	);
}

// Sends the fit parameters to the rank0 process so that
// they can be distributed to the other ranks and used to
// compute RMSE.
double computeRMSE(double * data) {
	sendMessage(
		inst, 
		data, 
		SEND_MODEL_FOR_RMSE_COMP, 
		sizeof(double) * PARAMETER_COUNT, 
		MSG_TYPE_DOUBLE
	);

    // Wait for a response.
	int code;
	int len;
	int type;
	double * result = recvMessage(inst, &code, &len, &type);

	return *result;
}

void finish() {
	double unused = 0.0;
	sendMessage(
		inst, 
		&unused, 
		PROCESSING_DONE, 
		sizeof(double), 
		MSG_TYPE_DOUBLE
	);
	destroyInstance(inst);
}