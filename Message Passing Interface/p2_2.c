#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define arr_size  64

int main(int argc, char** argv){
    int size, rank;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    
    int psum, tsum, array[arr_size];

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
	MPI_Bcast(array, arr_size, MPI_INT, 0, MPI_COMM_WORLD); // Broadcast the array data to all Processes

	psum = 0;
	if (rank == 0) {
		for(int i = 0; i < 16; i++){
			psum = psum + array[i];
		}
	}
	else if (rank == 1) {
		for (int i = 16; i < 32; i++){
			psum = psum + array[i];
		}
	}
	else if (rank == 2) {
		for (int k = 32; k < 48; k++){
			psum = psum + array[k];
		}
	}
	else if (rank == 3) {
		for (int l = 48; l < 64; l++){
			psum = psum + array[l];
		}
	}

	MPI_Reduce(&psum, &tsum, 1, MPI_INT,MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		printf("Total Sum Approach 2 = %d\n", tsum); //Process 0 prints the final sum
	}

	MPI_Finalize();
	return 0;
}
    
