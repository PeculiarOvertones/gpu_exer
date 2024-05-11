#include <cmath>
#include <iomanip>
#include <math.h>
#include <iostream>
#include <assert.h>
/** Compile with one of three options for matrix multiplication:
  * NAIVE, CONSTMEM, TILED_CONSTMEM_TYPE_1, TILED_CONSTMEM_TYPE_2, TILED_CONSTMEM_CACHEHALO
  * For Printing use flag: PRINT
  **/

#define EXCLUSIVE 0
#define SECTION_SIZE 1024

#ifdef KOGG_STONE
__global__ void reduce_sum_naive(float *Y, const float *X, unsigned int N) 
{
    /*assume that the kernal is launched with 
      dim3 blockDim(SECTION_SIZE, 1, 1);
      dim3 gridDim(ceil(N/SECTION_SIZE),1,1);
      */

    __shared__ float A_s[SECTION_SIZE]; 	

    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;

    /*define shared memory*/
    #ifdef EXCLUSIVE
    if(i<N && threadIdx.x != 0) {
        A_s[threadIdx.x] = Y[i-1];
    }
    else { 
        A_s[threadIdx.x] = 0.f;
    }
    #else 
    if(i<N) {
        A_s[threadIdx.x] = Y[i];
    }
    else { 
        A_s[threadIdx.x] = 0.f;
    }
    #endif

    /*Kogg-Stone loop*/
    for(unsigned int stride = 1; stride <= blockDim.x ; stride *= 2) 
    {
       __syncthreads();
       float temp;	    
       if(threadIdx.x >= stride) 
       {
           temp = A_s[threadIdx.x] + A_s[threadIdx.x - stride];
       }
       /*Write After Read synchronization*/
       __syncthreads();

       if(threadIdx.x >= stride) 
       {
           A_s[threadIdx.x] = temp;
       }
    }

    /*copy to global memory*/
    if(i < N) 
    {
        Y[i] = A_s[threadIdx.x];
    }
}
#endif

#ifdef KOGG_STONE_DOUBLE_BUFFER
__global__ void reduce_sum_naive(float *Y, const float *X, unsigned int N) 
{
    /*assume that the kernal is launched with 
      dim3 blockDim(SECTION_SIZE, 1, 1);
      dim3 gridDim(ceil(N/SECTION_SIZE),1,1);
      */

    __shared__ float A_s[SECTION_SIZE]; 	
    __shared__ float B_s[SECTION_SIZE]; 	

    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;

    /*define shared memory*/
    #ifdef EXCLUSIVE
    if(i<N && threadIdx.x != 0) {
        A_s[threadIdx.x] = Y[i-1];
        B_s[threadIdx.x] = A_s[threadIdx.x];
    }
    else { 
        A_s[threadIdx.x] = 0.f;
        B_s[threadIdx.x] = 0.f;
    }
    #else 
    if(i<N) {
        A_s[threadIdx.x] = Y[i];
        B_s[threadIdx.x] = A_s[threadIdx.x];
    }
    else { 
        A_s[threadIdx.x] = 0.f;
        B_s[threadIdx.x] = 0.f;
    }
    #endif

    float* read_ptr = &A_s; 	    
    float* write_ptr = &B_s; 	    
    /*Kogg-Stone loop*/
    for(unsigned int stride = 1; stride <= blockDim.x ; stride *= 2) 
    {
       __syncthreads();
       if(threadIdx.x >= stride) 
       {
          write_ptr[threadIdx.x] = read_ptr[threadIdx.x] + read_ptr[threadIdx.x - stride];
       }

       float* temp_ptr = write_ptr;
       write_ptr = read_ptr;
       read_ptr = temp_ptr;
    }

    /*copy to global memory*/
    if(i < N) 
    {
        Y[i] = write_ptr[threadIdx.x];
    }
}
#endif


#ifdef BRENT_KUNG
__global__ void prefix_sum_brent_kung(float *Y, const float *X, unsigned int N) 
{
    /*assume that the kernal is launched with 
      dim3 blockDim(SECTION_SIZE/2, 1, 1);
      dim3 gridDim(ceil(N/SECTION_SIZE),1,1);
      */
    __shared__ float A_s[SECTION_SIZE]; 	
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;

    /*define shared memory*/
    #ifdef EXCLUSIVE
    if(threadIdx.x == 0)
    {
        A_s[threadIdx.x] = 0.f;
    }
    else if(i<N) 
    {
        A_s[threadIdx.x] = Y[i-1];
    }

    if (i+blockDim.x < N)
    {
        A_s[threadIdx.x] = Y[i + blockDim.x - 1];
    }
    #else 
    if(i<N) 
    {
        A_s[threadIdx.x] = Y[i];
    }
    else if(i+blockDim.x < N) 
    {
        A_s[threadIdx.x+ blockDim.x] = Y[i + blockDim.x];
    }
    #endif

    /*Forward loop*/
    for(unsigned int stride = 1; stride <= blockDim.x ; stride *= 2) 
    {
       __syncthreads();
       int index = (threadIdx.x+1)*2*stride - 1;
       if(index < SECTION_SIZE) 
       {
           A_s[index] += A_s[index - stride];
       }
    }

    /*Backward loop*/
    for(unsigned int stride = SECTION_SIZE/4; stride > 0 ; stride /= 2) 
    {
       __syncthreads();
       int index = (threadIdx.x+1)*2*stride - 1;
       if(index + stride < SECTION_SIZE) 
       {
           A_s[index+stride] += A_s[index];
       }
    }
 
    __syncthreads(); //here syncthreads is necessary because threads will be responsible for tranferring elements that they did not work on.

    /*Copy to global memory*/
    if(i<N) 
    {
        Y[i] = A_s[threadIdx.x];
    }
    else if(i+blockDim.x < N) 
    {
        Y[i + blockDim.x] = A_s[threadIdx.x+ blockDim.x];
    }

}
#endif


void set_zero(float *M) 
{
    if(M != NULL) 
    {	
        int size = sizeof(M)/sizeof(M[0]);

	std::cout << "setting array of size: " << size << " to zero\n";
        for (int i = 0; i < size; ++i) 
        {
            M[i] = 0;    
        }
    }
}


void print_matrix(const float *M, int COL, int ROW) 
{
    for (int row = 0; row < ROW; ++row) 
    {
        for (int col = 0; col < COL; ++col) 
	{
            std::cout << std::setw(5) << M[row*COL + col];
	}
        std::cout << "\n";
    }
    std::cout << "\n";
}


void check_error(const float* h_output, const float* answer_check, const int size) {

    bool test_passed = true;
    for(int n=0; n<size; ++n) {	
        if(h_output[n] != answer_check[n]) {
           std::cout << "error: n, output, correct_ans:" << std::setw(10) << n << std::setw(10) << h_output[n] << std::setw(10) << answer_check[n] << "\n";
	   test_passed = false;
           break; 	    
        }
    }
    if(test_passed) std::cout << "Matrix Convolution Test Passed! \n";
}

int main (int argc, char* argv[])
{ 
    /*define dimensions*/ 
    /*A (Height x InnerSize)  x B (InnerSize x Width)  = M (Height x Width) **/
    const int N = 512;
    const int FilterSize = 2*FILTER_RADIUS+1;

    const int matA_memsize = Height*Width*sizeof(float);
    const int matM_memsize = Height*Width*sizeof(float);
    const int matF_memsize = FilterSize*FilterSize*sizeof(float);

#ifdef NAIVE 
    dim3 dimGrid(ceil(N/static_cast<float>(BLOCK_SIZE)), 
		 ceil(N/static_cast<float>(BLOCK_SIZE), 
		 ceil(N/static_cast<float>(BLOCK_SIZE)));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    std::cout << "cubic BLOCK_SIZE: " << std::setw(10) << BLOCK_SIZE  << "\n";

#elif REGISTERTILING_THREADCOARSENING
    dim3 dimGrid(ceil(N/static_cast<float>(OUT_TILE_SIZE)), 
	         ceil(N/static_cast<float>(OUT_TILE_SIZE)), 
		 ceil(N/static_cast<float>(OUT_TILE_SIZE));

    dim3 dimBlock(IN_TILE_SIZE, IN_TILE_SIZE, IN_TILE_SIZE);
    std::cout << "IN_TILE_SIZE, OUT_TILE_SIZE (square): " << std::setw(10) << IN_TILE_SIZE  << std::setw(10) << OUT_TILE_SIZE << "\n";
#endif

    int devID=0;
    if(argc > 1) devID = atoi(argv[1]);

    /*print cuda device properties*/
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devID);
    std::cout << "\nDevice: " << prop.name << "\n";
    std::cout << "Matrix sizes (height, width, filter size): "    << std::setw(10) << Height << std::setw(10) << Width << std::setw(10) << FilterSize << "\n";
    std::cout << "dimGrid (x,y,z):  "<< std::setw(10) << dimGrid.x  << std::setw(10) << dimGrid.y << std::setw(10) << dimGrid.z << "\n";
    std::cout << "dimBlock (x,y,z): "<< std::setw(10) << dimBlock.x << std::setw(10) << dimBlock.y << std::setw(10) << dimBlock.z << "\n";

    std::cout << "\nconstant memory (KB): " << prop.totalConstMem/1024 << "\n";
    std::cout << "total global memory (GB): " << prop.totalGlobalMem/(pow(1024,3)) << "\n";
    std::cout << "shared memory per block (KB): " << prop.sharedMemPerBlock/1024 << "\n";
    std::cout << "shared memory per multiprocessor (KB): " << prop.sharedMemPerMultiprocessor/1024 << "\n";
    std::cout << "register per block: " << prop.regsPerBlock << "\n";
    std::cout << "register per multiprocessor: " << prop.regsPerMultiprocessor << "\n";
    std::cout << "multiProcessorCount: " << prop.multiProcessorCount << "\n";
    std::cout << "warpSize: " << prop.warpSize<< "\n";
    /*cudaSetDevice(devID)*/

    /*define arrays on host and device*/
    /*A*B = M*/
    float* h_A = (float *) malloc(matA_memsize);
    float* h_F = (float *) malloc(matF_memsize);
    float* h_M = (float *) malloc(matM_memsize);

    float* M_check = (float *) malloc(matM_memsize);

    float* d_A = NULL;
    cudaMalloc(&d_A, matA_memsize);
    float* d_F = NULL;
    cudaMalloc(&d_F, matF_memsize);
    float* d_M = NULL;
    cudaMalloc(&d_M, matM_memsize);

    /*initializing input array*/
    for (int j=0; j < Height; ++j) {
	for (int i=0; i < Width; ++i) {
	    h_A [j*Width + i] = static_cast<float>(j);
	}
    }
    for (int j=0; j < FilterSize; ++j) {
	for (int i=0; i < FilterSize; ++i) {
	    h_F [j*FilterSize + i] = static_cast<float>((j));
	}
    }
    /*correct answer for error checking*/
    for (int row=0; row < Height; ++row) 
    {
	for (int col=0; col < Width; ++col) 
	{
	    float sum = 0.f;
            for(int j =  0; j < FilterSize; ++j) 
	    {
                for(int i = 0; i < FilterSize; ++i) 
	        {
	            int inCol = col + i - FILTER_RADIUS; 		    
	            int inRow = row + j - FILTER_RADIUS;
	            if(inRow >= 0 && inRow < Height && inCol >=0 && inCol < Width) 
	            {
                        sum += h_A[inRow*Width + inCol] * h_F[j*FilterSize + i];     
	            }
	        }
	    }
	    M_check [row*Width + col] = sum;
	}
    }

#ifdef PRINT
    std::cout << "\nWriting A matrix:\n";
    print_matrix(h_A, Width, Height);

    std::cout << "Writing Filter F:\n";
    print_matrix(h_F, FilterSize, FilterSize);

    std::cout << "Writing correct answer for M matrix:\n";
    print_matrix(M_check, Width, Height);
#endif

    cudaMemcpy(d_A, h_A, matA_memsize, cudaMemcpyHostToDevice);

#ifdef NAIVE
    cudaMemcpy(d_F, h_F, matF_memsize, cudaMemcpyHostToDevice);
#else  //use constant memory
    cudaMemcpyToSymbol(F, h_F, matF_memsize);
#endif

    cudaMemset(d_M, 0, matM_memsize);
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float ms;
    cudaEventRecord(startEvent, 0);

#ifdef NAIVE
    stencil_naive<<<dimGrid, dimBlock>>>(d_out, d_in, N);
#elif REGISTERTILING_THREADCOARSENING
    stencil_registertiling_threadcoarsening<<<dimGrid, dimBlock>>>(d_out, d_in, N);
#endif

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    std::cout << "\nElapsed time to run kernel (ms): " << ms << "\n";

    cudaMemcpy(h_M, d_M, matM_memsize, cudaMemcpyDeviceToHost); 
  
#ifdef PRINT
    std::cout << "Writing M matrix:\n";
    print_matrix(h_M, Width, Height);
#endif

   check_error(h_M, M_check, Width*Height);

//error_exit:
    /*free memory*/
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    free(h_A);
    free(h_F);
    free(h_M);
    free(M_check);

    cudaFree(d_A);
    cudaFree(d_F);
    cudaFree(d_M);

    cudaDeviceReset();
    return 0;
}

