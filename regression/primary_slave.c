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
	printf("[MPI %d] Started on %s.\n", world_rank, hostname);

	if (world_rank == 0) {
		// This is rank0, so we need to initialize the child instance
		// for communicating with the controller.
		struct MPIController * inst = createChildInstance("regression_controller");

		// The first thing we do is tell the controller how many
		// MPI processes there are.

		sendMessage(inst, &world_size, 1, sizeof(int), MSG_TYPE_INT);

		int code;
		int length;
		int type;

		// Now we wait for the length of the data to be sent in.
		int * nDataPoints = recvMessage(inst, &code, &length, &type);

		printf("[MPI 0] Received message indicating %d data points are being used.\n", 
			*nDataPoints);

		// Now we wait for the model data so that we can split it
		// up and send it to the child nodes.

		double * data = recvMessage(inst, &code, &length, &type);

		printf("[MPI 0] Received array of size %ld.\n", length / sizeof(double));

		// Now we split up the data and tell each child node
		// how much data it is getting. Then we send the chunks
		// to the child nodes.

		int chunk_size = (int)(floor(*nDataPoints / (world_size - 1)));
		int extra      = *nDataPoints - (chunk_size * (world_size - 1));

		printf("[MPI 0] Splitting data into %d chunks of size %d with extra chunk %d.\n",
			world_size - 1, chunk_size, extra);

		// Now we need (world_size - 1) arrays of length chunk_size to
		// send to the child nodes. We can keep the extra on the rank0
		// and just have it process that.

		for (int i = 0; i < world_size - 1; ++i) {
			double * chunk = malloc(sizeof(double) * chunk_size);

			for (int j = 0; j < chunk_size; ++j) {
				chunk[j] = data[i*chunk_size + j];
			}

			// First we tell the child node how large the
			// data will be.
			MPI_Send(
				&chunk_size, 
				1, 
				MPI_INT, 
				i + 1,                // This is the node to send to.
				SEND_DATA_DIMENSIONS, // We aren't sending to the rank0, 
				MPI_COMM_WORLD        // which is why this is i + 1 
			);  

			// Now we send this chunk to the child node.
			MPI_Send(
				chunk, 
				chunk_size, 
				MPI_DOUBLE, 
				i + 1,          // This is the node to send to.
				SEND_DATA,      // We aren't sending to the rank0, 
				MPI_COMM_WORLD  // which is why this is i + 1 
			);                  // instead of just i.

			// Now that the data has been sent off,
			// we can free it.
			free(chunk);
		}

		// Now we construct the array of extra values that don't
		// fit nicely. This array will be processed on the rank0
		// node.
		double * extraData = malloc(sizeof(double) * extra);
		for (int i = 0; i < extra; ++i) {
			extraData[i] = data[(chunk_size * (world_size - 1)) + i];
		}

		free(data);

		printf("[MPI 0] All data chunks have been initialized and sent.\n");

		// Now we have all of the data necessary for processing in
		// the memory of the respective processes. We can start waiting
		// for chunks of data to compute RMSE values for.

		while (true) {
			double * toProcess = recvMessage(inst, &code, &length, &type);

			//printf("[MPI 0] Received message with length %d and tag %d.\n", length, code);

			if (code == PROCESSING_DONE) {
				printf("[MPI 0] Received PROCESSING_DONE, sending to children.\n");
				// We are done processing, tell the other
				// processes to quit.
				double unused = 0.0;
				for (int proc = 1; proc < world_size; ++proc) {
					MPI_Send(
						&unused,
						1,
						MPI_DOUBLE,
						proc,
						PROCESSING_DONE,
						MPI_COMM_WORLD
					);
				}
				printf("[MPI 0] Exiting.\n");
				break;
			} else if (code == SEND_MODEL_FOR_RMSE_COMP) {
				// We have some data to process. Send it
				// off to the processes. We need to split
				// it up as we send it.
				for (int proc = 1; proc < world_size; ++proc) {
					// Now we send this chunk to the child node.
					//printf("[MPI 0] Sending data chunk to process %d.\n", proc);
					MPI_Send(
						&toProcess[chunk_size * (proc - 1)], 
						chunk_size, 
						MPI_DOUBLE, 
						proc,          
						SEND_MODEL_FOR_RMSE_COMP,
						MPI_COMM_WORLD
					);
				}

				double * local = malloc(sizeof(double) * extra);
				for (int i = 0; i < extra; ++i) {
					local[i] = toProcess[(chunk_size * (world_size - 1)) + i];
				}

				// Now we do the calculations on the extra chunk
				// of data.

				double sum  = 0.0;
				double diff = 0.0;
				for (int i = 0; i < extra; ++i) {
					diff = local[i] - extraData[i];
					sum += diff * diff;
				}

				//printf("[MPI 0] Reading results back from processes.\n");
				// Here we read the results from each process.
				double * tempResult = malloc(sizeof(double));
				for (int proc = 1; proc < world_size; ++proc) {
					MPI_Recv(
						tempResult,
						1,
						MPI_DOUBLE,
						proc,
						RETURN_RMSE,
						MPI_COMM_WORLD,
						MPI_STATUS_IGNORE
					);
					sum += *tempResult;	
				}
				free(tempResult);

				double rmse = sqrt(sum / *nDataPoints);

				// Send the RMSE value back to the controller.
				sendMessage(inst, &rmse, RETURN_RMSE, sizeof(double), MSG_TYPE_DOUBLE);
			}

			free(toProcess);
		}

		destroyInstance(inst);
		
	} else {
		// Here is where all of the child nodes do their processing.

		// The first message coming in should be size of the data we 
		// will be processing.
		int chunkSize = 0;
		MPI_Recv(
			&chunkSize,
			1,
			MPI_INT,
			0,
			SEND_DATA_DIMENSIONS,
			MPI_COMM_WORLD,
			MPI_STATUS_IGNORE
		);

		// Now we allocate an array for this data and receive it.
		double * chunk = malloc(sizeof(double) * chunkSize);

		MPI_Recv(
			chunk,
			chunkSize,
			MPI_DOUBLE,
			0,
			SEND_DATA,
			MPI_COMM_WORLD,
			MPI_STATUS_IGNORE
		);

		// Here we receive chunks of data to process until
		// we receive a message to stop.
		double * toProcess = malloc(sizeof(double) * chunkSize);
		while (true) {
			MPI_Status status;
			MPI_Recv(
				toProcess,
				chunkSize,
				MPI_DOUBLE,
				0,
				MPI_ANY_TAG,
				MPI_COMM_WORLD,
				&status
			);

			//printf("[MPI %d] Received message with tag %d.\n", world_rank, status.MPI_TAG);

			if (status.MPI_TAG == PROCESSING_DONE) {
				break;
			} else if (status.MPI_TAG == SEND_MODEL_FOR_RMSE_COMP) {
				// Process the data and return a value.
				double sum  = 0.0;
				double diff = 0.0;
				for (int i = 0; i < chunkSize; ++i) {
					diff = chunk[i] - toProcess[i];
					sum += diff*diff;
				}
				//printf("[MPI %d] Done processing RMSE chunk.\n", 
					//world_rank);

				// Send the result back.
				MPI_Send(
					&sum,
					1,
					MPI_DOUBLE,
					0,          
					RETURN_RMSE,
					MPI_COMM_WORLD
				);
			} else {
				printf("Unrecognized MPI Tag (%d) sent to rank %d\n", 
					status.MPI_TAG, world_rank);
			}
		}

		free(toProcess);
	}

	MPI_Finalize();
}