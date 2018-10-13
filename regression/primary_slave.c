#include "mpi_controller.h"
#include "defs.h"
#include <mpi.h>
#include <stdio.h>
#include <math.h>


int main(int argc, char ** argv) {
	// Basic MPI initialization stuff.
	MPI_Init(NULL, NULL);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	// Print out what rank this is and what machine
	// it started on.
	char * hostname = malloc(sizeof(char) * 128);
	memset(hostname, 0, sizeof(char) * 128);
	gethostname(hostname, sizeof(char) * 128);
	printf("[MPI] Rank %d started on %s\n", world_rank, hostname);

	if (world_rank == 0) {
		// This is rank0, so we need to initialize the child instance
		// for communicating with the controller.
		struct MPIController * inst = createChildInstance("test_controller");

		// The first thing we do is tell the controller how many
		// MPI processes there are.

		sendMessage(inst, &world_size, 1, sizeof(int), MSG_TYPE_INT);

		int code;
		int length;
		int type;

		// Now we wait for the length of the data to be sent in.
		int * nDataPoints = recvMessage(inst, &code, &length, &type);

		// Now we wait for the model data so that we can split it
		// up and send it to the child nodes.

		double * data = recvMessage(inst, &code, &length, &type);

		// Now we split up the data and tell each child node
		// how much data it is getting. Then we send the chunks
		// to the child nodes.

		int chunk_size = int(floor(nDataPoints / (world_size - 1)));
		int extra      = nDataPoints -  (chunk_size * (world_size - 1));

		// Now we need (world_size - 1) arrays of length chunk_size to
		// send to the child nodes. We can keep the extra on the rank0
		// and just have it process that.

		double ** chunks = malloc(sizeof(double*) * (world_size - 1));
		for (int i = 0; i < world_size - 1; ++i) {
			chunks[i] = malloc(sizeof(double) * chunk_size * 2);

			for (int j = 0; j < chunk_size; ++j) {
				
			}
		}



		for (int proc = 1; proc < world_size; ++proc) {
			

			printf("[MPI] Rank 0 received matrix, sending to rank %d\n", proc);

			MPI_Send(matrix, WIDTH*ROWS, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);

			printf("[MPI] Rank 0 finished sending matrix to rank %d\n", proc);

			free(matrix);
		}

		printf("[MPI] Done receiving matrices\n");

		// Now we wait for arrays to be sent in to process.

		for (int i = 0; i < MSG_COUNT; ++i) {
			int code;
			int length;
			int type;
			double * array;
			
			array = recvMessage(inst, &code, &length, &type);

			// Send the array to every other process.
			for (int proc = 1; proc < world_size; ++proc) {
				printf("[MPI] Received array %d, sending to rank %d\n", i, proc);
				MPI_Send(array, WIDTH, MPI_DOUBLE, proc, 2, MPI_COMM_WORLD);
				printf("[MPI] Finished sending array %d to rank %d\n", i, proc);
			}
			free(array);

			// Allocate an array to store results in.
			double * results = malloc(sizeof(double) * (world_size - 1));

			// Retrieve results from every other process.
			for (int proc = 1; proc < world_size; ++proc) {
				double sum;
				// tag 2 = processed result
				MPI_Recv(&sum, 1, MPI_DOUBLE, proc, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				printf("[MPI] Received result %d from rank %d\n", i, proc);
				results[proc - 1] = sum;
			}

			// Send the results back.
			printf("[MPI] Sending results back to controller\n");
			sendMessage(inst, results, 2, sizeof(double) * (world_size - 1), MSG_TYPE_DOUBLE);
			printf("[MPI] Finished sending results back to controller\n");
			free(results);
		}

		destroyInstance(inst);
		
	} else {
		// First we wait for a message containing the matrix to 
		// work with. 
		double * matrix = malloc(sizeof(double)*WIDTH*ROWS);

		printf("[MPI] Rank %d waiting for matrix\n", world_rank);
		// tag 1 = matrix send
		MPI_Recv(matrix, WIDTH*ROWS, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("[MPI] Rank %d received matrix\n", world_rank);

		// tag 2 = process something
		for (int i = 0; i < MSG_COUNT; ++i) {
			// Wait for an array to process.

			double * array = malloc(sizeof(double)*WIDTH);
			printf("[MPI] Rank %d waiting for array\n", world_rank);
			MPI_Recv(array, WIDTH, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("[MPI] Rank %d received array\n", world_rank);

			// Perform the calculation.
			double result = 0.0;
			for (int j = 0; j < ROWS; ++j) {
				for (int k = 0; k < WIDTH; ++k) {
					result += matrix[j*WIDTH + k]*array[k];
				}
			}

			free(array);

			// Send the results back.
			printf("[MPI] Rank %d sending result\n", world_rank);
			MPI_Send(&result, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
			printf("[MPI] Rank %d finished sending result\n", world_rank);
		}

		free(matrix);
	}

	MPI_Finalize();
}