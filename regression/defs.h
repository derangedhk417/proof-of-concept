// Message id for when the size of the data is
// being sent to the MPI nodes.
#define SEND_DATA_DIMENSIONS     1

// Message id for when the original data
// chunks are being sent to the child nodes.
// More specifically, this is the data that
// the model is being fit to.
#define SEND_DATA                2

// Message id for when a set of data points is
// being sent for the child nodes to compute
// the RMSE against the real data.
#define SEND_MODEL_FOR_RMSE_COMP 3

// Message id used when the child nodes are
// returning the RMSE data to the controller.
#define RETURN_RMSE              4

// A message with this id is sent to the child
// when it is time to stop processing and exit.
#define PROCESSING_DONE          5