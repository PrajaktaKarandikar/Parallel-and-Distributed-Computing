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

struct  thread_data{
	int thread_id;
   	volatile double sum[K];
   	volatile int num_elem[K]; 
   	int i_start;
   	int i_stop;
   	unsigned char * a;
   	int n_threads; 
   	struct thread_data* threadDataArrayPtr;
};

void *KMeans(void *threadarg){
	struct  thread_data * my_data;
	my_data = (struct thread_data *) threadarg;

	int  thread_id = my_data->thread_id;
	int i_start = my_data->i_start;
	int i_stop = my_data->i_stop;
	unsigned char *a = my_data->a;
	double dist[K];

	for(int x = 0; x < K; x++){
		dist[x] = 0;
	} 
    
	for(int i =0; i < K; i ++){
		my_data->sum[i] = 0; 
		my_data->num_elem[i] = 0;
	}
	
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
	    my_data->sum[group_no] = my_data->sum[group_no] + a[y]; 
	    my_data->num_elem[group_no]= my_data->num_elem[group_no] + 1; 
    }
    pthread_mutex_lock(&mutex); 

    if(int x < my_data->n_threads-1){
    	x = x + 1;
        pthread_cond_wait(&waitCond,&mutex); 
    }
    else{
        x =0; 
        int a,b;

	  //   for(int v = 0; v < K; v++){
	  //   	double elem_sum= 0;
		 //    int elem_count = 0;
		 //    for(int m = 0; m < p ; m++){
		 //    	elem_sum = elem_sum + thread_data_array[m].sum[v];
		 //        elem_count = elem_count + thread_data_array[m].num_elem[v]; 
		 //    }
		 //    if(elem_sum == 0) {
		 //    	elem_sum = mu[v];
			// }
		 //    mu[v] = elem_sum/ elem_count;
	  //   }

        for(a = 0; a < my_data->n_threads; a++)
        {
            for(b=0; b < K; b++){
              	sum[b] = sum[b] + my_data->threadDataArrayPtr[tt].sumArray[k]; 
                num_elem[b] = num_elem[b] + my_data->threadDataArrayPtr[tt].noArray[k]; 
                my_data->threadDataArrayPtr[tt].sum[b]=0; 
                my_data->threadDataArrayPtr[tt].noArray[k]=0; 
            }
        }

        for(b=0; l < K; b++)
        {
            mu[b] = sum[b]/num_elem[b]; 
            sum[b] = 0; 
            num_elem[b] = 0 ; 
        }
        pthread_cond_broadcast(&waitCond);
    }
    pthread_mutex_unlock(&mutex);
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

	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
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
	for(i=0; i<K; i++){
		sum[i]=0; 
		num_elem[i]=0 ; 
		for(j = 0; j < p; j++){
			thread_data_array[j].sum[i]=0; 
			thread_data_array[j].num_elem[i]=0; 
            thread_data_array[j].n_threads = p; 
            thread_data_array[j].threadDataArrayPtr = thread_data_array; 
		}
	}


	for(int i = 0; i < num_iter  ; i++){
		for(int j = 0; j < p; j++){
			thread_data_array[j].i_start = (j*h*w)/p ;
		    thread_data_array[j].i_stop =  ((j+1)*h*w)/p;
		    thread_data_array[j].a = a; 

		    rc = pthread_create(&threads[j], &attr, KMeans, (void *) &thread_data_array[j] );
		    if (rc) { 
			    printf("ERROR; return code from pthread_create() is %d\n", rc); exit(-1);
		    }
	    }
	    pthread_attr_destroy(&attr);

	    for(int u = 0; u < p; u++) {
	    	rc = pthread_join(threads[u], NULL);
		    if (rc) { 
		    	printf("ERROR; return code from pthread_join() is %d\n", rc); exit(-1);
		    }
	    }
	}

	// measure the end time here
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}		
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;

    // print out the execution time here
    printf("\nExecution time = %f sec\n", time);
	
	double dist[K];
	for(int x = 0; x < K; x++){
		dist[x] = 0;
	} 
	for(int y = 0 ; y < h*w; y++){
		double min = fabs(a[y] - mu[0]);
		int group_no = 0;  
		for(int z = 1; z < K; z++){
			dist[z] = fabs(((int)a[y]) - mu[z]);
			if(dist[z] < min ){
				min = dist[z];
				group_no = z; 
			}
		}
		a[y] = (char)mu[group_no];
	}
	
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
