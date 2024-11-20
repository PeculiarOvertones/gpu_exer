#include <cmath>
#include <iomanip>
#include <math.h>
#include <iostream>
#include <assert.h>
/** Compile with one of three options for matrix multiplication:
  * NAIVE, CONSTMEM, TILED_CONSTMEM_TYPE_1, TILED_CONSTMEM_TYPE_2, TILED_CONSTMEM_CACHEHALO
  * For Printing use flag: PRINT
  **/

#define FILTER_RADIUS 1

#if defined(NAIVE) || defined(CONSTMEM)
const int BLOCK_SIZE = 32;
#endif

#ifndef NAIVE
__constant__ float F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
#endif

#ifdef TILED_CONSTMEM_TYPE_1
const int IN_TILE_SIZE = 32;
const int OUT_TILE_SIZE = IN_TILE_SIZE - 2*FILTER_RADIUS;
#elif TILED_CONSTMEM_TYPE_2
const int IN_TILE_SIZE = 32;
const int OUT_TILE_SIZE = IN_TILE_SIZE - 2*FILTER_RADIUS;
#elif TILED_CONSTMEM_CACHEHALO
const int TILE_SIZE = 32;
#endif

#ifdef NAIVE
__global__ void convolution_naive(float *M, const float *A, const float *Filter, const int Height, const int Width) 
{
    /*A = pointer to input array 
     *M = pointer to output array
     *F = pointer to filter array
     *Width and Height of input and output arrays
     *R = radius of square filter
     */ 
    
    int FilterSize = 2*FILTER_RADIUS + 1;	
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;
 
    float sum = 0.f;    
    for(int j =  0; j < FilterSize; j++) 
    {
        for(int i = 0; i < FilterSize; i++) 
        {
    	    int inCol = outCol + i - FILTER_RADIUS;    
    	    int inRow = outRow + j - FILTER_RADIUS;    
                        	
    	    if(inRow >= 0 && inRow < Height && inCol >=0 && inCol < Width) 
    	    {
                    sum += A[inRow*Width + inCol] * Filter[j*FilterSize + i];     
    	    }
        }
    }

    if(outRow < Height && outCol < Width) 
    {
        M[outRow*Width + outCol] = sum;
    }
}
#endif

#ifdef CONSTMEM
__global__ void convolution_constmem(float *M, const float *A, const int Height, const int Width) 
{
	
    int FilterSize = 2*FILTER_RADIUS + 1;	
    int outCol = blockIdx.x*blockDim.x + threadIdx.x;
    int outRow = blockIdx.y*blockDim.y + threadIdx.y;

    float sum = 0.f;    
    for(int j =  0; j < FilterSize; j++) 
    {
        for(int i = 0; i < FilterSize; i++) 
        {
    	    int inCol = outCol + i - FILTER_RADIUS;    
    	    int inRow = outRow + j - FILTER_RADIUS;    
                        	
    	    if(inRow >= 0 && inRow < Height && inCol >=0 && inCol < Width) 
    	    {
                sum += A[inRow*Width + inCol] * F[j][i];     
    	    }
        }
    }

    if(outRow < Height && outCol < Width) 
    {
        M[outRow*Width + outCol] = sum;
    }
}
#endif


#ifdef TILED_CONSTMEM_TYPE_1

__global__ void convolution_tiled_constmem_type1(float *M, const float *A, const int Height, const int Width) 
{
    /* Tile size equal to input tile.
     *
     * Overview:
     * - Define inRow and inCol corresponding to input tile.
     *
     * - Define a tile of size IN_TILE_SIZE * IN_TILE_SIZE
     *
     * - Load tile
     *   - Use inRow and inCol that are within domain bounds
     *     otherwise set tile values to zero
     * - Synchronize
     *
     * - Compute convolution (this is tricky because we have to switch off some threads)
     *
     *   - Use inRow and inCol that are within domain bounds
     *
     *     - Switch off threads that are outside of output tile size
     *       (e.g. threadIdx.y - FilterRadius >=0 && < OUT_TILE_SIZE)
     *
     *       - Compute convolution where tile[threadIdx.y-FilterRadius+i][...]
     *         where i and j are from 0 to FilterSize
     *     
     *       - store sum in M[inRow*Width+inCol]
     */
       

    int FilterSize = 2*FILTER_RADIUS + 1;

    int inCol = blockIdx.x*OUT_TILE_SIZE - FILTER_RADIUS + threadIdx.x;
    int inRow = blockIdx.y*OUT_TILE_SIZE - FILTER_RADIUS + threadIdx.y;

    __shared__ float tile_A[IN_TILE_SIZE][IN_TILE_SIZE];

    /*load tile*/
    if(inRow >=0 && inRow < Height && inCol >=0 && inCol < Width) 
        tile_A[threadIdx.y][threadIdx.x] = A[inRow*Width + inCol];
    else
        tile_A[threadIdx.y][threadIdx.x] = 0.;
     
    __syncthreads();

    if(inRow >=0 && inRow < Height && inCol >=0 && inCol < Width) 
    {
        int local_out_rowId = threadIdx.y - FILTER_RADIUS;
        int local_out_colId = threadIdx.x - FILTER_RADIUS;

        if(local_out_colId >=0 && local_out_colId  < OUT_TILE_SIZE &&
           local_out_rowId >=0 && local_out_rowId < OUT_TILE_SIZE )
        {
            float sum = 0.f;	    
            for(int j=0; j<FilterSize; ++j) 
            {
                for(int i=0; i<FilterSize; ++i) 
                {
                    sum += tile_A[threadIdx.y + j - FILTER_RADIUS][threadIdx.x + i - FILTER_RADIUS] * F[j][i];	    
                }
            }
            M[inRow*Width + inCol] = sum;
        }
    }
}
#endif


#ifdef TILED_CONSTMEM_TYPE_2

__global__ void convolution_tiled_constmem_type2(float *M, const float *A, const int Height, const int Width) 
{
    /* Tile size equal to output tile.
     *
     * Overview:
     * - Define outRow and outCol corresponding to output tile.
     *
     * - Define a tile of size IN_TILE_SIZE x IN_TILE_SIZE
     *
     * - Load tile (this is tricky, each thread may load more than 1 elements)
     *   - for loop over j and i, (e.g. j=threadIdx.y; j<IN_TILE_SIZE; j+=OUT_TILE_SIZE)
     * 
     *     - compute inRow, inCol (e.g. inRow = blockIdx.y*OUT_TILE_SIZE - FilterRadius + j)
     *     
     *     - load: tile[j][i] = A[inRow*Width+inCol]
     *
     *     - Make sure inRow and inCol within bounds otherwise set to zero
     *
     * - Synchronize
     *
     * - Compute convolution
     *
     *   - Use outRow and outCol that are within domain bounds
     *
     *     - Compute convolution by looping over i,j from 0 to FilterSize
     *       - sum += tile[tile_row][...] *F[j][i];
     *         where tile_row = threadIdx.y + j  //Note this
     *     
     *     - store sum in M[inRow*Width+inCol]
     */
	
    int FilterSize = 2*FILTER_RADIUS + 1;
    int outRow = blockIdx.y*OUT_TILE_SIZE + threadIdx.y;
    int outCol = blockIdx.x*OUT_TILE_SIZE + threadIdx.x;

    __shared__ float tile_A[IN_TILE_SIZE][IN_TILE_SIZE];

    /*load tile*/
    for(int j=threadIdx.y; j < IN_TILE_SIZE; j += OUT_TILE_SIZE) 
    {
        for(int i=threadIdx.x; i < IN_TILE_SIZE; i += OUT_TILE_SIZE)	
        {
            int inRow = j + (blockIdx.y*OUT_TILE_SIZE - FILTER_RADIUS);
            int inCol = i + (blockIdx.x*OUT_TILE_SIZE - FILTER_RADIUS);
           
            if(inRow >=0 && inRow < Height && inCol >=0 && inCol < Width) 
            {  
                tile_A[j][i] = A[inRow*Width + inCol];
    	    }
            else 
            {
                tile_A[j][i] = 0.f;
            }
        }
    }
     
    __syncthreads();

    if(outRow < Height && outCol < Width) 
    {
        float sum = 0.f;	    
        for(int j=0; j<FilterSize; ++j) 
        {
            for(int i=0; i<FilterSize; ++i) 
            {
        		int tile_row = threadIdx.y + j;
	            int tile_col = threadIdx.x + i;
                   
                sum += tile_A[tile_row][tile_col] * F[j][i];	    
            }
        }
        M[outRow*Width + outCol] = sum;
    }
}
#endif


#ifdef TILED_CONSTMEM_CACHEHALO
__global__ void convolution_tiled_constmem_cachehalo(float *M, const float *A, const int Height, const int Width) 
{
    /* Tile size equal to output tile. We load elements in the halo from memory, 
     * expecting them to be located most likely in the L2 cache.
     *
     * Overview:
     * - Define outRow and outCol corresponding to output tile.
     *
     * - Define a tile of size OUT_TILE_SIZE x OUT_TILE_SIZE
     *
     * - Load tile, if outside domain bounds then set value to zero.
     *
     * - Synchronize
     *
     * - Compute convolution
     *
     *   - Make sure outRow and outCol are within the domain bounds
     *
     *     - Loop over i,j from 0 to FilterSize
     *       - Compute local_rowID and local_colID=threadIdx.x - FilterRadius + i
     *         - if these are within 0 to OUT_TILE_SIZE
     *           - then load elements from tile for convolution
     *         - else 
     *           - compute the global_rowID and global_colID=col-FilterRadius+i
     *           - if these are within the domain bounds
     *             - then load elements from memory for convolution
     *     - store sum in M[inRow*Width+inCol]
     */
	
    int FilterSize = 2*FILTER_RADIUS + 1;

    int row = blockIdx.y*TILE_SIZE + threadIdx.y;
    int col = blockIdx.x*TILE_SIZE + threadIdx.x;

    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];

    /*load tile*/
    if(row < Height && col < Width) 
        tile_A[threadIdx.y][threadIdx.x] = A[row*Width + col];
    else
        tile_A[threadIdx.y][threadIdx.x] = 0.;
     
    __syncthreads();

    if(row < Height && col < Width) 
    {
        float sum = 0.f;	    
        for(int j=0; j<FilterSize; ++j) 
        {
            for(int i=0; i<FilterSize; ++i) 
            {
	            int local_A_rowId = threadIdx.y + j - FILTER_RADIUS;
	         	int local_A_colId = threadIdx.x + i - FILTER_RADIUS;

		        if(local_A_colId >=0 && local_A_colId < TILE_SIZE && 
		           local_A_rowId >=0 && local_A_rowId < TILE_SIZE)     
		        {
                    sum += tile_A[local_A_rowId][local_A_colId] * F[j][i];	    
	            }
                else 
                {
                    int global_A_rowId = row + j - FILTER_RADIUS;
                    int global_A_colId = col + i - FILTER_RADIUS;
        
                    if (global_A_colId >= 0 && global_A_colId < Width &&   
                        global_A_rowId >= 0 && global_A_rowId < Height)  
                    {
                        sum += A[ global_A_rowId*Width + global_A_colId ] * F[j][i];
                    }
                }
            }
        }
        M[row*Width + col] = sum;
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
    const int Height = 4096;
    const int Width = 4096;
    const int FilterSize = 2*FILTER_RADIUS+1;

    const int matA_memsize = Height*Width*sizeof(float);
    const int matM_memsize = Height*Width*sizeof(float);
    const int matF_memsize = FilterSize*FilterSize*sizeof(float);

#if defined(NAIVE) || defined(CONSTMEM)
    dim3 dimGrid(ceil(Width/static_cast<float>(BLOCK_SIZE)), ceil(Height/static_cast<float>(BLOCK_SIZE)), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    std::cout << "Square BLOCK_SIZE (height, width): " << std::setw(10) << BLOCK_SIZE  << "\n";

#elif TILED_CONSTMEM_TYPE_1
    /*In this case, we load blocks with threads equal to IN_TILE_SIZE*/
    dim3 dimGrid(ceil(Width/static_cast<float>(OUT_TILE_SIZE)), ceil(Height/static_cast<float>(OUT_TILE_SIZE)), 1);
    dim3 dimBlock(IN_TILE_SIZE, IN_TILE_SIZE, 1);
    std::cout << "IN_TILE_SIZE, OUT_TILE_SIZE (square): " << std::setw(10) << IN_TILE_SIZE  << std::setw(10) << OUT_TILE_SIZE << "\n";

#elif TILED_CONSTMEM_TYPE_2
    /*In this case, we load blocks with threads equal to OUT_TILE_SIZE*/
    dim3 dimGrid(ceil(Width/static_cast<float>(OUT_TILE_SIZE)), ceil(Height/static_cast<float>(OUT_TILE_SIZE)), 1);
    dim3 dimBlock(OUT_TILE_SIZE, OUT_TILE_SIZE, 1); 
    std::cout << "IN_TILE_SIZE, OUT_TILE_SIZE (square): " << std::setw(10) << IN_TILE_SIZE  << std::setw(10) << OUT_TILE_SIZE << "\n";

#elif TILED_CONSTMEM_CACHEHALO
    dim3 dimGrid(ceil(Width/static_cast<float>(TILE_SIZE)), ceil(Height/static_cast<float>(TILE_SIZE)), 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1); 
    std::cout << "TILE_SIZE (square): " << std::setw(10) << TILE_SIZE << "\n";
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
    convolution_naive<<<dimGrid, dimBlock>>>(d_M, d_A, d_F, Height, Width);
#elif CONSTMEM
    convolution_constmem<<<dimGrid, dimBlock>>>(d_M, d_A, Height, Width);
#elif TILED_CONSTMEM_TYPE_1
    convolution_tiled_constmem_type1<<<dimGrid, dimBlock>>>(d_M, d_A, Height, Width);
#elif TILED_CONSTMEM_TYPE_2
    convolution_tiled_constmem_type2<<<dimGrid, dimBlock>>>(d_M, d_A, Height, Width);
#elif TILED_CONSTMEM_CACHEHALO
    convolution_tiled_constmem_cachehalo<<<dimGrid, dimBlock>>>(d_M, d_A, Height, Width);
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

