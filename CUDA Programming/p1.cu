#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define n  1024

__global__ void matmul(int *a, int *b, int *c){
	
	int my_x = blockIdx.y * blockDim.y + threadIdx.y;
	int my_y = blockIdx.x * blockDim.x + threadIdx.x;

	int local_c = 0;
	for (int i = 0; i < n; i++) {
		local_c += a[my_x*n + i] * b[my_y*n + i];
	}
	c[my_x*n + my_y] = local_c;
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
  	}
  	int *gpu_a, *gpu_b, *gpu_c; //alloacte gpu space to a, b and c
  	cudaMalloc((void**)&gpu_a, sizeof(int) * n * n);
  	cudaMalloc((void**)&gpu_b, sizeof(int) * n * n);
  	cudaMalloc((void**)&gpu_c, sizeof(int) * n * n);

  	struct timespec start, stop;
	double time;

	cudaMemcpy(gpu_a, a, sizeof(int)*n*n, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_b, b, sizeof(int)*n*n, cudaMemcpyHostToDevice);

	dim3 dimGrid(64, 64); //Grid configuration is 64x64
	dim3 dimBlock(16, 16); //Thread block configuration is 16x16

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