#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>

struct  thread_data{
   int	thread_id;
   double **A;
   double **B;
   double **C;
   int n;
   int jump;
   int ind;
};



void *Matrix_Multiply(void *threadarg){
	struct  thread_data * my_data;
	my_data = (struct thread_data *) threadarg;

	int  thread_id = my_data->thread_id; 
	double **A = my_data->A;
	double **B = my_data->B;
	double **C = my_data->C;

	int jump = my_data->jump;
	int n = my_data->n;
	int ind = my_data->ind;

	printf("Entered Matrix_Multiply, thread=%d", thread_id);

    int t_row, t_col, t_k, br_start, bc_start;

    for(int i = 0; i <n; i+=jump){
    	for(int j = 0; j<n; j+=jump){
    		br_start = i;
    		bc_start = j;
    	}
    }

    for(t_row = br_start; t_row < br_start+jump; t_row++){
    	//printf("In For 1\n");
    	for(t_col = bc_start; t_col < bc_start+jump; t_col++){
    	//	    	printf("In For 2\n");
    		for(t_k = 0; t_k < n; t_k++){
    			C[t_row][t_col] += (A[t_row][t_k] * B[t_k][t_col]);
    		}
    		
    	//	printf("C[%d][%d]: %f \n", t_row, t_col, C[t_row][t_col]);
    	}
    }


  printf("Thread %d executed.", thread_id);

	pthread_exit(NULL);
}

int main(int argc, char* argv[]){
	if(argc<2){
		printf("not enough arguemnets\n");
		return 1; 
	}
	    
	const int p = atoi(argv[1]);
	    
	struct thread_data thread_data_array[p*p]; 

    pthread_t threads[p*p]; 
	int i, j, k;
	struct timespec start, stop; 
	double time;
	int n = 4096; // matrix size is n*n
		
	double **A = (double**) malloc (sizeof(double*)*n);
	double **B = (double**) malloc (sizeof(double*)*n);
	double **C = (double**) malloc (sizeof(double*)*n);


    for (i=0; i<n; i++) {
    	A[i] = (double*) malloc(sizeof(double)*n);
		B[i] = (double*) malloc(sizeof(double)*n);
		C[i] = (double*) malloc(sizeof(double)*n);
	}
		
    for (i=0; i<n; i++){
		for(j=0; j< n; j++){
			A[i][j]=i;
			B[i][j]=i+j;
			C[i][j]=0;			
		}
	}
				
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
		
	// Your code goes here //
	// Matrix C = Matrix A * Matrix B //	
	//*******************************//
	int rc;

	for (int ti = 0; ti < (p*p); ti++){
		thread_data_array[ti].thread_id = ti;
		thread_data_array[ti].jump = n/p;
				
		thread_data_array[ti].A = A;
		thread_data_array[ti].B = B;
		thread_data_array[ti].C = C;

		thread_data_array[ti].n = n;

		rc = pthread_create(&threads[ti], NULL, Matrix_Multiply, (void *) &thread_data_array[ti]);
		if (rc) {
			printf("ERROR; return code from pthread_create() is %d\n", rc); 
			exit(-1);
		}
	}

	for(i = 0; i< (p*p); i++){
    	rc = pthread_join(threads[i],NULL); 
		if (rc) {
			printf("ERROR; joining error return code from pthread_join() is %d\n", rc);
			exit(-1); 
		}			 
	}		
		
	//*******************************//
		
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
		
	printf("\nExecution time = %f sec,\n %d threads \n", time, p*p);		
	printf("\nC[100][100]=%f\n", C[100][100]);
		
	// release memory
	for (i=0; i<n; i++) {
		free(A[i]);
		free(B[i]);
		free(C[i]);
	}
	free(A);
	free(B);
	free(C);
	return 0;

}
