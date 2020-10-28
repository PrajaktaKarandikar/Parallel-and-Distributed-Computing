#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>

struct  thread_data{
	int	x,y,b,n;
	double **A, **B, **C;
};

static pthread_mutex_t mutex;


void *block_multiply(void *threadarg){
	struct thread_data *my_data;
	my_data = (struct thread_data *) threadarg;
	int b = my_data -> b;
	int n = my_data -> n;
	int x = my_data -> x;
	int y = my_data -> y;
	double **A = my_data -> A;
	double **B = my_data -> B;
	double **C = my_data -> C;
	int i,j,k;
	double sum;
	// printf("%d %d %d",b,x,y);
	for(i=x; i<x+b; i++){
		for(j=y; j<y+b; j++){
			sum = 0;
			for(k=0; k<n; k++){
				sum += A[i][k] * B[k][j];
			}
			pthread_mutex_lock(&mutex);
			C[i][j] = sum;
			pthread_mutex_unlock(&mutex);
		}
	}
	// printf("%d %d \n",x,y);
	pthread_exit(NULL);
}

int main(int argc, char *argv[]){
	if( argc == 2 ) { 
  		printf("Number of threads: %d\n", atoi(argv[1])*atoi(argv[1]));
  	}
	else if(argc > 2) {
		printf("Too many arguments.\n");
		exit(0);
  	}
	else {
		printf("One argument expected.\n");
		exit(0);
	}

	// number of threads used
	const int num_threads = atoi(argv[1])*atoi(argv[1]);

	// thread initialisations
	struct thread_data block_ids[num_threads];

	pthread_t  threads[num_threads];
	int rc;

	int i, j;
	int n = 4096;
	int b = n/(atoi(argv[1])); 

	// mutex initialisation
	pthread_mutex_init(&mutex, NULL);

	// time initialisations
	struct timespec start, stop; 
	double time;

	// matrix initialisations
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
			
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) {perror("clock gettime");}
	
	// Matrix C = Matrix A * Matrix B //	
	//*******************************//

	int index = 0;
	for(i=0; i<n; i+=b){
		for(j=0; j<n; j+=b){
			block_ids[index].b = b;
			block_ids[index].n = n;
			block_ids[index].x = i;
			block_ids[index].y = j;
			block_ids[index].A = A;
			block_ids[index].B = B;
			block_ids[index].C = C;
			rc = pthread_create(&threads[index], NULL, block_multiply, (void *) &block_ids[index]);
			if (rc) { 
				printf("ERROR; return code from pthread_create() is %d\n", rc); 
				exit(-1);
			}
			// if(num_threads != 1)
			// pthread_detach(threads[index]);
			index += 1;
		}
	}

	
	for(i=0; i<num_threads; i++){
		rc = pthread_join(threads[i], NULL);
		if (rc) { 
			printf(" joining error %d ", rc);   
			exit(-1);
		}
	}
	pthread_mutex_destroy(&mutex);

	//*******************************//
	
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	 
	printf("Number of FLOPs = %lu, Execution time = %f sec,\n%lf MFLOPs per sec\n", 2*n*n*n, time, 1/time/1e6*2*n*n*n);		
	printf("C[100][100]=%f\n", C[100][100]);
	
	// release memory
	for (i=0; i<n; i++) {
		free(A[i]);
		free(B[i]);
		free(C[i]);
	}
	free(A);
	free(B);
	free(C);
	// pthread_exit(NULL);
	return 0;
}
