#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define arr_size  64

int main(int argc, char** argv){
    int size, rank;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    
    int array[arr_size], psum_0, psum_1, psum_2, psum_3;
	//All processes read the array
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

	if (rank == 0){
		psum_0 = 0;
		int tsum = 0;
		for(int i = 0; i < 16; i++){
			psum_0 = psum_0 + array[i];
		}
		//Process 0 recevies all the partial sums and computes final sum
		MPI_Recv(&psum_1, 1, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&psum_2, 1, MPI_INT, 2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&psum_3, 1, MPI_INT, 3, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		tsum = psum_0 + psum_1 + psum_2 + psum_3;
		printf("Total Sum Approach 1 = %d\n", tsum); //Process 0 prints the final sum
	}
	else if (rank == 1) {
		psum_1 = 0;
		for (int j = 16; j < 32; j++){
			psum_1 = psum_1 + array[j];
		}
		MPI_Send(&psum_1, 1, MPI_INT, 0, 1, MPI_COMM_WORLD); //Process 1 sends its partial sum to Process 0
	}
	else if (rank == 2) {
		psum_2 = 0;
		for (int k = 32; k < 48; k++){
			psum_2 = psum_2 + array[k];
		}
		MPI_Send(&psum_2, 1, MPI_INT, 0, 2, MPI_COMM_WORLD); //Process 2 sends its partial sum to Process 0
	}
	else if (rank == 3) {
		psum_3 = 0;
		for (int l = 48; l < 64; l++){
			psum_3 = psum_3 + array[l];
		}
		MPI_Send(&psum_3, 1, MPI_INT, 0, 3, MPI_COMM_WORLD); //Process 3 sends its partial sum to Process 0
	}
	MPI_Finalize();
	return 0;
}
    
