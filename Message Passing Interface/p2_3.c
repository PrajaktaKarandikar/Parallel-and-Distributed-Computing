#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define arr_size  64
#define chunk  16

int main(int argc, char** argv){
    int size, rank;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    
    int psum, tsum, array[arr_size], local[(chunk/4)], process_sum[chunk];

	if (rank == 0){
		//Process 0 reads the array
	    FILE *fp;
	    fp = fopen("number.txt", "r");
	    if (fp == NULL){
	    	printf("Cannot open file");
            return 1;
        }
	    for (int m = 0; m < arr_size; m++){
	    	fscanf(fp, "%d", &array[m]);
	    }
	    fclose(fp);
	}
	MPI_Scatter(array, chunk , MPI_INT, process_sum, chunk, MPI_INT, 0, MPI_COMM_WORLD); //Scatter data to other processes
	
	psum = 0;
    for(int i = 0; i < chunk; i++) {
   		psum = psum + process_sum[i];    // Compute partial sums for respective chunks of data
	}
	
	MPI_Gather(&psum, 1, MPI_INT, &local, 1, MPI_INT, 0, MPI_COMM_WORLD); //Receive and collect all partial sums from other processes

	if (rank == 0) {
		tsum = 0;
		for(int j = 0; j < (chunk/4); j++){
			tsum = tsum + local[j];
		}
		printf("Total Sum Approach 3 = %d\n", tsum); //Process 0 prints the final sum
	}

	MPI_Finalize();
	return 0;
}