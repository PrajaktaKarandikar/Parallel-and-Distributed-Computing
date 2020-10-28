#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv){
    int size, rank;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    int Msg;

	if (rank == 0) {
		Msg = 451;
		// Process 0 sends Msg value to Process 1
		printf("Process %d: Initially Msg = %d\n", rank, Msg);
		MPI_Send(&Msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		MPI_Recv(&Msg, 1, MPI_INT, 3, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		// Process 0 receives updated Msg value from Process 3 and prints it
		printf("Process %d: Received Msg = %d. Done!", rank, Msg);
	} 
	else if (rank == 1) {
		// Process 1 receives Msg value from Process 0
		MPI_Recv(&Msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		Msg = Msg + 1;
		printf("Process %d: Msg = %d\n", rank, Msg);
		// Process 1 sends updated Msg value to Process 2
		MPI_Send(&Msg, 1, MPI_INT, 2, 1, MPI_COMM_WORLD);
	} 
	else if (rank == 2) {
		// Process 2 receives Msg value from Process 1
		MPI_Recv(&Msg, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
		Msg = Msg + 1;
		printf("Process %d: Msg = %d\n", rank, Msg);
		// Process 2 sends updated Msg value to Process 3
		MPI_Send(&Msg, 1, MPI_INT, 3, 2, MPI_COMM_WORLD);
	} 
	else if (rank == 3) {
		// Process 3 receives Msg value from Process 2
		MPI_Recv(&Msg, 1, MPI_INT, 2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		Msg = Msg + 1;
		printf("Process %d: Msg = %d\n", rank, Msg);
		//Process 3 sends updated Msg value to Process 0
		MPI_Send(&Msg, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}

	
	
    
    

