#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define n  1024
#define bk  32

__global__ void matmul(int *a, int *b, int *c){
	int trow = threadIdx.y;
	int tcol = threadIdx.x;

	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

    int my_x = blockIdx.y * blockDim.y + threadIdx.y;
	int my_y = blockIdx.x * blockDim.x + threadIdx.x;

    // a and b are stored in shared memory
	__shared__ int a_share[bk][bk];
	__shared__ int b_share[bk][bk]; 

	int local_c = 0;
	for (int i = 0; i < (n/bk); i++) {
		//reads one element from a and b matrices into the shared sub-matrices
		a_share[trow][tcol] = a[my_y*n + (i*blockDim.y + tcol)];
		b_share[trow][tcol] = b[(i*blockDim.x + tcol)*n + my_x];

		__syncthreads(); // Ensure synchronization

		for (int j = 0; j < bk; j++) {
			local_c += a_share[trow][j] * b_share[j][tcol];
		}

		__syncthreads(); //Ensure synchronization of all threads after computing step
	}
	c[my_y*n + my_x] = local_c;
}

int main(){
	//Allocate memory in CPU for a and b matrices
	int *a = (int*)malloc(sizeof(int) * n * n);
	int *b = (int*)malloc(sizeof(int) * n * n);
    int *c = (int*)malloc(sizeof(int) * n * n);

    //Fill in the elements of a and b as specified
	for(int i = 0; i < (n*n); i++){
		a[i]=1;
		b[i]=2;
		c[i]=0;
  	}
  	int *gpu_a, *gpu_b, *gpu_c; //alloacte gpu space to a, b and c
  	cudaMalloc((void**)&gpu_a, sizeof(int) * n * n);
  	cudaMalloc((void**)&gpu_b, sizeof(int) * n * n);
  	cudaMalloc((void**)&gpu_c, sizeof(int) * n * n);

  	struct timespec start, stop;
	double time;

	cudaMemcpy(gpu_a, a, sizeof(int)*n*n, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, b, sizeof(int)*n*n, cudaMemcpyHostToDevice);
	
	dim3 dimGrid(32, 32); //Grid configuration is 32x32
	dim3 dimBlock(32, 32); //Thread block configuration is 32x32

	if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
	matmul<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c);
	cudaMemcpy(c, gpu_c, sizeof(int)*n*n, cudaMemcpyDeviceToHost);

	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	printf("Execution time is %f ns\n", time*1e9);

	printf("c[451][451]= %d\n", c[451*n + 451]);

	free(a);
	free(b);
	free(c);
	cudaFree(gpu_a);  
	cudaFree(gpu_b);  
	cudaFree(gpu_c);  
	return 0;
}