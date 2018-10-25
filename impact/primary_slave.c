#include "mpi_controller.h"
#include "defs.h"
#include <mpi.h>
#include <stdio.h>
#include <math.h>


double fit_function(double x, double * p) {
	return p[0]  * -pow((x - 1.0/1.0),  2.0) + 
		   p[1]  *  pow((x - 1.0/2.0),  2.0) + 
		   p[2]  * -pow((x - 1.0/3.0),  2.0) + 
		   p[3]  *  pow((x - 1.0/4.0),  3.0) +
		   p[4]  * -pow((x - 1.0/5.0),  2.0) +
		   p[5]  *  pow((x - 1.0/6.0),  2.0) + 
		   p[6]  * -pow((x - 1.0/7.0),  2.0) + 
		   p[7]  *  pow((x - 1.0/8.0),  3.0) +
		   p[8]  * -pow((x - 1.0/9.0),  2.0) + 
		   p[9]  *  pow((x - 1.0/10.0), 2.0) +
		   p[10] * -pow((x - 1.0/11.0), 2.0) + 
		   p[11] *  pow((x - 1.0/12.0), 3.0) + 
		   p[12] * -pow((x - 1.0/13.0), 2.0) + 
		   p[13] *  pow((x - 1.0/14.0), 2.0) + 
		   p[14] * -pow((x - 1.0/15.0), 2.0); 
}

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

		
		double * lowerBound = recvMessage(inst, &code, &length, &type);
		double * upperBound = recvMessage(inst, &code, &length, &type);
		int    * points     = recvMessage(inst, &code, &length, &type);

		printf("[MPI 0] Function range [%f, %f] with %d points.\n", 
			*lowerBound, *upperBound, *points);

		// Now we wait for the model data so that we can split it
		// up and send it to the child nodes.

		double * data = recvMessage(inst, &code, &length, &type);

		printf("[MPI 0] Received array of size %ld.\n", length / sizeof(double));

		// Now we split up the data and tell each child node
		// how much data it is getting. Then we send the chunks
		// to the child nodes.

		int chunk_size = (int)(floor(*points / (world_size - 1)));
		int extra      = *points - (chunk_size * (world_size - 1));

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

			// Now we need to compute and send the lower bound for
			// this rank. The upper bound will follow naturally from
			// the number of data points.
			double pointIncrement = (*upperBound - *lowerBound) / (*points);
			double rankLowerBound = *lowerBound + (pointIncrement * chunk_size * i);

			double params[2];
			params[0] = rankLowerBound;
			params[1] = pointIncrement;

            MPI_Send(
				params, 
				2, 
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

		double pointIncrement  = (*upperBound - *lowerBound) / (*points);
		double localLowerBound = *lowerBound + (pointIncrement * chunk_size * (world_size - 1));

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
						toProcess, 
						PARAMETER_COUNT, 
						MPI_DOUBLE, 
						proc,          
						SEND_MODEL_FOR_RMSE_COMP,
						MPI_COMM_WORLD
					);
				}

				// double * local = malloc(sizeof(double) * extra);
				// for (int i = 0; i < extra; ++i) {
				// 	local[i] = toProcess[(chunk_size * (world_size - 1)) + i];
				// }

				// Now we calculate the data to do the RMSE computation for.

				double x = localLowerBound;
				double * local = malloc(sizeof(double) * extra);
				for (int i = 0; i < extra; ++i, x += pointIncrement) {
					local[i] = fit_function(x, toProcess);
				}

				// Now we do the calculations on the extra chunk
				// of data.

				double sum  = 0.0;
				double diff = 0.0;
				for (int i = 0; i < extra; ++i) {
					diff = local[i] - extraData[i];
					sum += diff * diff;
				}

				free(local);

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

				double rmse = sqrt(sum / *points);

				// Send the RMSE value back to the controller.
				sendMessage(inst, &rmse, RETURN_RMSE, sizeof(double), MSG_TYPE_DOUBLE);
			}

			free(toProcess);
		}
	} else {
		// Here is where all of the child nodes do their processing.

		// The first message coming in should be size of the data we 
		// will be processing.
		int    chunkSize      = 0;
		double lowerBound     = 0.0;
		double pointIncrement = 0.0;

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

		double params[2];

		// Receive the lower bound for this rank.
		MPI_Recv(
			params,
			2,
			MPI_DOUBLE,
			0,
			SEND_DATA,
			MPI_COMM_WORLD,
			MPI_STATUS_IGNORE
		);

		lowerBound     = params[0];
		pointIncrement = params[1];

		// Here we receive chunks of data to process until
		// we receive a message to stop.
		double * toProcess = malloc(sizeof(double) * PARAMETER_COUNT);
		while (true) {
			MPI_Status status;
			MPI_Recv(
				toProcess,
				PARAMETER_COUNT,
				MPI_DOUBLE,
				0,
				MPI_ANY_TAG,
				MPI_COMM_WORLD,
				&status
			);

			//printf("[MPI %d] Received message with tag %d.\n", world_rank, status.MPI_TAG);

			if (status.MPI_TAG == PROCESSING_DONE) {
				printf("[MPI %d] Exiting.\n", world_rank);
				break;
			} else if (status.MPI_TAG == SEND_MODEL_FOR_RMSE_COMP) {
				// Generate the values to compute RMSE against.
				
				double x = lowerBound;
				double * data = malloc(sizeof(double) * chunkSize);
				for (int i = 0; i < chunkSize; ++i, x += pointIncrement) {
					data[i] = fit_function(x, toProcess);
				}
				// Process the data and return a value.
				double sum  = 0.0;
				double diff = 0.0;
				for (int i = 0; i < chunkSize; ++i) {
					diff = chunk[i] - data[i];
					sum += diff*diff;
				}

				free(data);
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