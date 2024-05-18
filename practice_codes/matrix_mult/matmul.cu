#include <cmath>
#include <iomanip>
#include <math.h>
#include <iostream>
#include <assert.h>
#include <nvtx3/nvToolsExt.h>
/** Compile with one of three options for matrix multiplication:
  * NAIVE: naive matmul with blocks.
  * TILED: matmul with 2D tiles
  * TILED_COARSENED: more work per thread 
    (loading fewer threads in a block than 
     the elements they will be responsible to process)
  * PRINT: printing matrices for debugging
  * e.g. nvcc -arch=sm_70 -DTILED -DPRINT matmul.cu -o solver.x
  *
  *                                  [.|.|.] B (InnerSize x Width)
  *                                  [.|.|.]     
  *                                  [.|.|.]
  *                                  [.|.|.]
  *                                                     
  *          [.|.|.|.]               [.|.|.]   
  *          [.|.|.|.]               [.|.|.]
  *          [.|.|.|.]               [.|.|.]
  * 	     A (Height x InnerSize)  M (Height x Width)         	     
  **/

#ifdef NAIVE
const int BLOCK_ROWS = 4;
#elif defined(TILED) || defined(TILED_COARSENED)
const int TILE_SIZE = 16;
#endif

#ifdef TILED_COARSENED
const int COARSE_FACTOR = 2;
#endif

#ifdef NAIVE
__global__ void matmul_naive(float *M, const float *A, const float *B, const int Height, const int Width, const int InnerSize) 
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
 
    if((row < Height) && (col < Width)) 
    {
    	float sum = 0.f;    
        #pragma unroll
        for(int i = 0; i < InnerSize; ++i) 
	    {
	        sum += A[row*InnerSize + i] * B[i*Width + col];     	
	    }
        M[row*Width+col] = sum;
    }
}
#endif


#ifdef TILED
__global__ void matmul_tiled(float *M, const float *A, const float *B, const int Height, const int Width, const int InnerSize) 
{
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x*TILE_SIZE + threadIdx.x;
    int row = blockIdx.y*TILE_SIZE + threadIdx.y;
  
    float sum = 0.f;

    for(int istart=0; istart < InnerSize; istart += TILE_SIZE) 
    {
        /*load tile A*/
        int colA = (istart + threadIdx.x);

        if(row < Height && colA < InnerSize) 
            tile_A[threadIdx.y][threadIdx.x] = A[row*InnerSize + colA];
        else 
            tile_A[threadIdx.y][threadIdx.x] = 0.f;

        /*load tile B*/
        int rowB = (istart + threadIdx.y);

        if(rowB < InnerSize && col < Width) 
            tile_B[threadIdx.y][threadIdx.x] = B[rowB*Width + col];
        else 
            tile_B[threadIdx.y][threadIdx.x] = 0.f;

        __syncthreads();

        /*do computations*/
        for (int t=0; t<TILE_SIZE; ++t) 
        {
    	    sum += tile_A[threadIdx.y][t] * tile_B[t][threadIdx.x];    
        }

        __syncthreads();
    }

    if(row < Height && col < Width) 
    {
        M[row*Width+col] = sum;
    }

    /*Note: load individual tiles and only check for boundary conditions of respective rows and cols*/
    /*Note: Another way*/
    //  for(int phase=0; phase < (InnerSize/TILE_SIZE); ++phase) 
    //  {
    //    int colA = (phase*TILE_SIZE + threadIdx.x);
    //    int rowB = (phase*TILE_SIZE + threadIdx.y);
}
#endif


#ifdef TILED_COARSENED
__global__ void matmul_tiled_coarsened(float *M, const float *A, const float *B, const int Height, const int Width, const int InnerSize) 
{
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int col = blockIdx.x*TILE_SIZE*COARSE_FACTOR + threadIdx.x;
    int row = blockIdx.y*TILE_SIZE + threadIdx.y;
  
    float sum[COARSE_FACTOR];
    for(int c=0; c<COARSE_FACTOR; ++c) sum[c] = 0.f;

    for(int istart=0; istart < InnerSize; istart += TILE_SIZE) 
    {
        /*load tile A*/
        int colA = (istart + threadIdx.x);

        if(row < Height && colA < InnerSize) 
            tile_A[threadIdx.y][threadIdx.x] = A[row*InnerSize + colA];
        else 
            tile_A[threadIdx.y][threadIdx.x] = 0.f;
        

        int rowB = (istart + threadIdx.y);
        for(int c=0; c<COARSE_FACTOR; ++c) 
	    {
            /*load tile B*/
            int colB = c*TILE_SIZE + col;

            if(rowB < InnerSize && colB < Width) 
                tile_B[threadIdx.y][threadIdx.x] = B[rowB*Width + colB];
            else 
                tile_B[threadIdx.y][threadIdx.x] = 0.f;

            __syncthreads();

            /*do computations*/
            for (int t=0; t<TILE_SIZE; ++t) 
            {
    	        sum[c] += tile_A[threadIdx.y][t] * tile_B[t][threadIdx.x];    
            }

            __syncthreads();
	    }
    }

    if(row < Height) {
        for(int c=0; c<COARSE_FACTOR; ++c) {
            int col_glo = c*TILE_SIZE + col;

            if(col_glo < Width) M[row*Width + col_glo] = sum[c];
        }
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
    if(test_passed) std::cout << "Matrix Multiplication Test Passed! \n";
}


int main (int argc, char* argv[])
{ 
    /*define dimensions*/ 
    /*A (Height x InnerSize)  x B (InnerSize x Width)  = M (Height x Width) **/
    const int Height = 256;
    const int Width  = 256;
    const int InnerSize = 2048;

    const int matA_memsize = Height*InnerSize*sizeof(float);
    const int matB_memsize = InnerSize*Width*sizeof(float);
    const int matM_memsize = Height*Width*sizeof(float);

#ifdef NAIVE
    dim3 dimGrid(1, ceil(Height/static_cast<float>(BLOCK_ROWS)), 1);
    dim3 dimBlock(Width, BLOCK_ROWS, 1);
#elif TILED
    dim3 dimGrid(ceil(Width/static_cast<float>(TILE_SIZE)), 
                 ceil(Height/static_cast<float>(TILE_SIZE)), 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
#elif TILED_COARSENED
    dim3 dimGrid(ceil(Width/static_cast<float>(TILE_SIZE*COARSE_FACTOR)), 
                 ceil(Height/static_cast<float>(TILE_SIZE)), 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1); 
#endif

    int devID=0;
    if(argc > 1) devID = atoi(argv[1]);

    /*print cuda device properties*/
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devID);
    std::cout << "\nDevice: " << prop.name << "\n";
    std::cout << "Matrix sizes (height, innersize, width): " << std::setw(10) 
              << Height << std::setw(10) 
              << InnerSize << std::setw(10) 
              << Width << "\n";

#if defined(TILED) || defined(TILED_COARSENED)
    std::cout << "TILE_SIZE (height, width): " << std::setw(10) 
              << TILE_SIZE  << std::setw(10) << TILE_SIZE << "\n";
#endif
    std::cout << "dimGrid (x,y,z):  "<< std::setw(10) << dimGrid.x  
                                     << std::setw(10) << dimGrid.y 
                                     << std::setw(10) << dimGrid.z << "\n";
    std::cout << "dimBlock (x,y,z): "<< std::setw(10) << dimBlock.x 
                                     << std::setw(10) << dimBlock.y 
                                     << std::setw(10) << dimBlock.z << "\n";

    /*cudaSetDevice(devID)*/

    /*define arrays on host and device*/
    /*A*B = M*/
    float* h_A = (float *) malloc(matA_memsize);
    float* h_B = (float *) malloc(matB_memsize);
    float* h_M = (float *) malloc(matM_memsize);

    float* M_check = (float *) malloc(matM_memsize);

    float* d_A = NULL;
    cudaMalloc(&d_A, matA_memsize);
    float* d_B = NULL;
    cudaMalloc(&d_B, matB_memsize);
    float* d_M = NULL;
    cudaMalloc(&d_M, matM_memsize);

    /*initializing input array*/
    for (int j=0; j < Height; ++j) {
	    for (int i=0; i < InnerSize; ++i) {
	        h_A [j*InnerSize + i] = static_cast<float>(j);
	    }
    }
    for (int j=0; j < InnerSize; ++j) {
	    for (int i=0; i < Width; ++i) {
	        h_B [j*Width + i] = static_cast<float>(j);
	    }
    }
    /*correct answer for error checking*/
    for (int row=0; row < Height; ++row) {
	    for (int col=0; col < Width; ++col) {
	        float sum = 0.f;
	        for (int i=0; i < InnerSize; ++i) {
	            sum += h_A[row*InnerSize + i] * h_B[i*Width + col];
	        }
	        M_check [row*Width + col] = sum;
	    }
    }

    #ifdef PRINT
    std::cout << "Writing A matrix:\n";
    print_matrix(h_A, InnerSize, Height);

    std::cout << "Writing B matrix:\n";
    print_matrix(h_B, Width, InnerSize);

    std::cout << "Writing correct answer for M matrix:\n";
    print_matrix(M_check, Width, Height);
    #endif

    cudaMemcpy(d_A, h_A, matA_memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matB_memsize, cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float ms;
    cudaEventRecord(startEvent, 0);

    nvtxRangePush("Start Profiling");
#ifdef NAIVE
    matmul_naive<<<dimGrid, dimBlock>>>(d_M, d_A, d_B, Height, Width, InnerSize);
#elif TILED
    matmul_tiled<<<dimGrid, dimBlock>>>(d_M, d_A, d_B, Height, Width, InnerSize);
#elif TILED_COARSENED
    matmul_tiled_coarsened<<<dimGrid, dimBlock>>>(d_M, d_A, d_B, Height, Width, InnerSize);
#endif
    nvtxRangePop();

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    std::cout << "Elapsed time to run kernel (ms): " << ms << "\n";

    cudaMemcpy(h_M, d_M, matM_memsize, cudaMemcpyDeviceToHost); 
  
    cudaDeviceSynchronize();
#ifdef PRINT
    std::cout << "Writing M matrix:\n";
    print_matrix(h_M, Width, Height);
#endif

    check_error(h_M, M_check, Width*Height);

    /*free memory*/
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    free(h_A);
    free(h_B);
    free(h_M);
    free(M_check);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_M);

    cudaDeviceReset();
    return 0;
}

