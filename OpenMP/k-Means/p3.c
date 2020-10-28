#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <math.h>
#include <omp.h>

#define h  800 
#define w  800
#define K  6

#define input_file  "input.raw"
#define output_file "output.raw"

pthread_mutex_t mutex;
pthread_cond_t waitCond;

static int num_iter = 50;
double mu[K] = {0, 65, 100, 125, 190, 255};
double sum[K];
int num_elem[K]; 

struct  thread_data{
	int thread_id;
   	volatile double sum[K];
   	volatile int num_elem[K]; 
   	int i_start;
   	int i_stop;
   	unsigned char * a;
   	int n_threads; 
   	struct thread_data* tda_Ptr;
};

void *KMeans(void *threadarg){
	struct  thread_data * my_data;
	my_data = (struct thread_data *) threadarg;

	int  thread_id = my_data->thread_id;
	int i_start = my_data->i_start;
	int i_stop = my_data->i_stop;
	unsigned char *a = my_data->a;
	int n_threads = my_data->n_threads;
	double dist[] = {0,0,0,0,0,0};
	float int_sum[] = {0,0,0,0,0,0};
	int int_numelem[] = {0,0,0,0,0,0};  

	    
	for(int i =0; i < K; i ++){
		my_data->sum[i] = 0; 
		my_data->num_elem[i] = 0;
	}
	//compute minimum distance
	for(int y = i_start; y < i_stop; y++){
		double min = fabs(a[y] - mu[0]); 
	    int group_no = 0;  

	    for(int z = 1; z < K; z++){
	    	dist[z] = fabs(a[y] - mu[z]);
		    if(dist[z] < min){
		    	min = dist[z];
			    group_no = z; 
		    }
	    }
	    sum[group_no] = sum[group_no] + a[y];
	    num_elem[group_no] = num_elem[group_no] + 1;
    }
    pthread_mutex_lock(&mutex); //set lock

    //check value of 'r' as said in question and do as required
    int r;
    if(r < my_data->n_threads-1){
    	r = r + 1;
        pthread_cond_wait(&waitCond,&mutex); 
    }
    else{
    	for(int v = 0; v < K; v++){ //recompute means for each cluster
	    	double elem_sum= 0;
		    int elem_count = 0;
		    for(int m = 0; m < n_threads ; m++){
		    	elem_sum = elem_sum + my_data->tda_Ptr[m].sum[v];
		        elem_count = elem_count + my_data->tda_Ptr[m].num_elem[v]; 
		    }
		    if(elem_sum == 0) {
		    	elem_sum = mu[v];
			}
		    mu[v] = elem_sum/ elem_count;
	    }

      	r = 0;
        pthread_cond_broadcast(&waitCond);
    }
    pthread_mutex_unlock(&mutex); //unlock
   	pthread_exit(NULL);
}


int main(int argc, char* argv[]){
	if(argc< 2 || argc >2){
		printf("Invalid commandline arguements" );
		return 0;
	}

	int p = atoi(argv[1]);

	struct  thread_data  thread_data_array[p];
	int rc; 

    pthread_cond_init(&waitCond,NULL); 
    pthread_mutex_init(&mutex,NULL);

	FILE *fp;
	unsigned char *a = (unsigned char*) malloc (sizeof(unsigned char)*h*w);
	struct timespec start, stop; 
	double time;
	
    pthread_t  threads[p];

	// the matrix is stored in a linear array in row major fashion
	if (!(fp=fopen(input_file, "rb"))) {
		printf("can not open file\n");
		return 1;
	}
	fread(a, sizeof(unsigned char), w*h, fp);
	fclose(fp);
    
	// measure the start time here
	if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}

	//  Your code goes here

	for(int i = 0; i < num_iter  ; i++){
		for(int j = 0; j < p; j++){
			thread_data_array[j].i_start = (j*h*w)/p ;
		    thread_data_array[j].i_stop =  ((j+1)*h*w)/p;
		    thread_data_array[j].a = a; 
		    thread_data_array[j].n_threads = p; 
		    thread_data_array[j].tda_Ptr = thread_data_array; 

		    rc = pthread_create(&threads[j], NULL, KMeans, (void *) &thread_data_array[j] );
		    if (rc) { 
			    printf("ERROR; return code from pthread_create() is %d\n", rc); exit(-1);
		    }
	    }
	    
	    for(int u = 0; u < p; u++) {
	    	rc = pthread_join(threads[u], NULL);
		    if (rc) { 
		    	printf("ERROR; return code from pthread_join() is %d\n", rc); exit(-1);
		    }
	    }
	}

    // map the values computed to get the output image
    double d[K];
	for(int x = 0; x < K; x++){
		d[x] = 0;
	} 
	for(int y = 0 ; y < h*w; y++){
		double min = fabs(a[y] - mu[0]);
		int group_no = 0;  
		for(int z = 1; z < K; z++){
			d[z] = fabs(((int)a[y]) - mu[z]);
			if(d[z] < min ){
				min = d[z];
				group_no = z; 
			}
		}
		a[y] = (char)mu[group_no];
	}
	
	// measure the end time here
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}		
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;

    // print out the execution time here
    printf("\nExecution time = %f sec\n", time);
	

	if (!(fp=fopen(output_file,"wb"))) {
	   	printf("can not opern file\n");
	    return 1;
	}	

    fwrite(a, sizeof(unsigned char),w*h, fp);
    fclose(fp);

    pthread_cond_destroy(&waitCond);
    pthread_mutex_destroy(&mutex); 
    
    return 0;
}


