#include "mpi_controller.h"
#include <mpi.h>
#include <stdio.h>

#define MSG_COUNT 50

int main(int argc, char ** argv) {
	MPI_Init(NULL, NULL);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	if (world_rank == 0) {
		struct MPIController * inst = createChildInstance("test_controller");

		// The first thing we do is tell the controller how many
		// MPI processes there are, so it can generate a set
		// of matrices.

		sendMessage(inst, &world_size, 1, sizeof(int), MSG_TYPE_INT);

		// Now we need to wait for n matrices and send them
		// each to a separate process.

		for (int proc = 1; proc < world_size; ++proc) {
			int code;
			int length;
			int type;
			double * matrix;
			
			matrix = recvMessage(inst, &code, &length, &type);

			MPI_Send(matrix, 1024*6000, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);

			free(matrix);
		}

		// Now we wait for arrays to be sent in to process.

		for (int i = 0; i < MSG_COUNT; ++i) {
			int code;
			int length;
			int type;
			double * array;
			
			array = recvMessage(inst, &code, &length, &type);

			for (int proc = 1; proc < world_size; ++proc) {
				MPI_Send(array, 1024, MPI_DOUBLE, proc, 2, MPI_COMM_WORLD);
			}
			free(array);

			double * results = malloc(sizeof(double) * (world_size - 1));

			for (int proc = 1; proc < world_size; ++proc) {
				double sum;
				// tag 2 = processed result
				MPI_Recv(&sum, 1, MPI_DOUBLE, proc, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				results[proc - 1] = sum;
			}

			// Send the results back.
			sendMessage(inst, results, 2, sizeof(double) * (world_size - 1), MSG_TYPE_DOUBLE);

			free(results);
		}

		destroyInstance(inst);
		
	} else {
		// First we wait for a message containing the matrix to 
		// work with. 
		double * matrix = malloc(sizeof(double)*1024*6000);

		// tag 1 = matrix send
		MPI_Recv(matrix, 1024*6000, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// tag 2 = process something
		for (int i = 0; i < MSG_COUNT; ++i) {
			double * array = malloc(sizeof(double)*1024);
			MPI_Recv(array, 1024, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			double result = 0.0;
			for (int j = 0; j < 6000; ++j) {
				for (int k = 0; k < 1024; ++k) {
					result += matrix[j*1024 + k]*array[k];
				}
			}

			free(array);
			MPI_Send(&result, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
		}

		free(matrix);
	}

	MPI_Finalize();
}