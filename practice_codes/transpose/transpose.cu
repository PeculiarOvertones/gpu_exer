#include <cmath>
#include <iomanip>
#include <math.h>
#include <iostream>
#include <assert.h>

const int TILE_SIZE = 32;
const int BLOCK_ROWS = 8; /* We'll launch fewer threads than the tile size in the y direction, 
                             i.e. each thread will read in more rows */

#ifdef NAIVE
__global__ void transpose_simple(float *output, const float *input, const int Width, const int Height) 
{
    /* Threadcoarsening in the row direction. 
       Using a thread block with fewer threads than elements in a tile 
       is advantageous for the matrix transpose kernel, 
       because each thread transposes four matrix elements, 
       as a result much of the index calculation cost is amortized over these elements.
     */

    int inCol = blockIdx.x * TILE_SIZE + threadIdx.x;
    int inRow = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    /* The loop iterates over the row-wise dimension (y) and not the column-wise, 
       so that contiguous threads load and store contiguous data. 
     * In this kernel, reads are coalesced, but writes are not.
     */
    for (int j=0; j < TILE_SIZE; j += BLOCK_ROWS) 
    {
        int row_glo = inRow + j;
        output[inCol*Height + row_glo] = input[row_glo*Width + inCol];
    }
}
#endif

#ifdef CORNER_TURNING
__global__ void transpose_sharedtile(float *output, 
                                     const float *input, const int Width, const int Height) 
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
	
    int inCol = blockIdx.x * TILE_SIZE + threadIdx.x;
    int inRow = blockIdx.y * TILE_SIZE + threadIdx.y;

    /* load tile */
    for (int j=0; j < TILE_SIZE; j += BLOCK_ROWS) 
    {
        tile[threadIdx.y+j][threadIdx.x] = input[(inRow+j)*Width + inCol];
    }
    __syncthreads(); /* we need this because threads write different data 
                        to output than they read from input */

    /* here only the block is offset; this insures that the write will be contiguous */

    inCol = blockIdx.y * TILE_SIZE + threadIdx.x; 
    inRow = blockIdx.x * TILE_SIZE + threadIdx.y;

    for (int j=0; j < TILE_SIZE; j += BLOCK_ROWS) 
    {
        output[(inRow+j)*Height + inCol] = tile[threadIdx.x][threadIdx.y+j];
    }
}
#endif


#ifdef NO_BANK_CONFLICT
__global__ void transpose_sharedtile_bankconflictavoid(float *output, 
                                                       const float *input, 
                                                       const int Width, const int Height) 
{
    /* By padding the shared memory tile width in the x direction by 1, 
       we avoid bank conflict.
       This is because at the time of writing to output, 
       all threads are accessing elements from the same column of the tile.
       In this case, if the tile width size is the same as the number of channels/2 
       (accounting for double data rate access), 
       then all threads would try to access from the same bank.

     */	
    __shared__ float tile[TILE_SIZE][TILE_SIZE+1];
	
    /* The rest of the code remains the same. */

    int inCol = blockIdx.x * TILE_SIZE + threadIdx.x;
    int inRow = blockIdx.y * TILE_SIZE + threadIdx.y;

    for (int j=0; j < TILE_SIZE; j += BLOCK_ROWS) 
    {
        tile[threadIdx.y+j][threadIdx.x] = input[(inRow+j)*Width + inCol];
    }
    __syncthreads(); 

    inCol = blockIdx.y * TILE_SIZE + threadIdx.x; 
    inRow = blockIdx.x * TILE_SIZE + threadIdx.y;

    for (int j=0; j < TILE_SIZE; j += BLOCK_ROWS) 
    {
        output[(inRow+j)*Height + inCol] = tile[threadIdx.x][threadIdx.y+j];
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


void check_transpose_error(const float* h_output, const float* answer_check, const int size) {

    bool test_passed = true;
    for(int n=0; n<size; ++n) {	
        if(h_output[n] != answer_check[n]) {
           std::cout << "error: n, output, correct_ans:" << std::setw(10) << n << std::setw(10) << h_output[n] << std::setw(10) << answer_check[n] << "\n";
	   test_passed = false;
           break; 	    
        }
    }
    if(test_passed) std::cout << "Transpose Test Passed! \n";
}


int main (int argc, char* argv[])
{ 
    /*define dimensions*/
    const int Width = 1024;
    const int Height = 768;

    const int mat_memsize = Width*Height*sizeof(float);

    dim3 dimGrid(ceil(Width/TILE_SIZE), ceil(Height/TILE_SIZE), 1);    
    dim3 dimBlock(TILE_SIZE, BLOCK_ROWS, 1);    

    int devID=0;
    if(argc > 1) devID = atoi(argv[1]);

    /*print cuda device properties*/
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devID);

    std::cout << "\nDevice: " << prop.name << "\n";

    std::cout << "Matrix (width/cols, height/rows): " 
        << std::setw(10) << Width << std::setw(10) << Height << "\n";

    std::cout << "TILE_SIZE (width/cols, height/rows): " 
        << std::setw(10) << TILE_SIZE  << std::setw(10) << BLOCK_ROWS << "\n";

    std::cout << "dimGrid (x,y,z):  " << std::setw(10) 
              << dimGrid.x  << std::setw(10) 
              << dimGrid.y << std::setw(10) 
              << dimGrid.z << "\n";

    std::cout << "dimBlock (x,y,z): " << std::setw(10) 
              << dimBlock.x << std::setw(10) 
              << dimBlock.y << std::setw(10) 
              << dimBlock.z << "\n";

    /*define arrays on host and device*/
    float* h_input = (float *) malloc(mat_memsize);
    float* h_output = (float *) malloc(mat_memsize);
    float* answer_check = (float *) malloc(mat_memsize);

    float* d_input = NULL;
    cudaMalloc(&d_input, mat_memsize);

    float* d_output = NULL;
    cudaMalloc(&d_output, mat_memsize);

    /*check parameters*/
    if(Width % TILE_SIZE || Height % TILE_SIZE) {
        std::cout << "Width and Heigh must be a multipler of TILE_SIZE\n";
	goto error_exit;
    }
    if(TILE_SIZE % BLOCK_ROWS) {
        std::cout << "TILE_SIZE must be a multipler of BLOCK_ROWS\n";
	goto error_exit;
    }

    /*initializing input array*/
    for (int j=0; j < Height; ++j) {
	    for (int i=0; i < Width; ++i) {
	        h_input [j*Width + i] = j*Width + i;
	    }
    }
    /*correct answer for error checking*/
    for (int j=0; j < Height; ++j) {
	    for (int i=0; i < Width; ++i) {
	        answer_check [i*Height + j] = h_input[j*Width + i];
	    }
    }
    #ifdef PRINT
    std::cout << "Writing input matrix:\n";
    print_matrix(h_input, Width, Height);

    std::cout << "Writing correct answer matrix:\n";
    print_matrix(answer_check, Height, Width);
    #endif

    cudaMemcpy(d_input, h_input, mat_memsize, cudaMemcpyHostToDevice); 

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float ms;
    cudaEventRecord(startEvent, 0);

    /*invoke a kernel*/
    #ifdef NAIVE
    transpose_simple<<< dimGrid, dimBlock >>>(d_output, d_input, Width, Height);
    #elif CORNER_TURNING
    transpose_sharedtile<<< dimGrid, dimBlock >>>(d_output, d_input, Width, Height);
    #elif NO_BANK_CONFLICT
    transpose_sharedtile_bankconflictavoid<<< dimGrid, dimBlock >>>
        (d_output, d_input, Width, Height);
    #endif

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    std::cout << "Time elapsed: " << ms << "\n";

    cudaMemcpy(h_output, d_output, mat_memsize, cudaMemcpyDeviceToHost); 
  
    #ifdef PRINT
    std::cout << "Writing output matrix:\n";
    print_matrix(h_output, Height, Width);
    #endif

    check_transpose_error(h_output, answer_check, Width*Height);

error_exit:
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    /*free memory*/
    free(h_input);
    free(h_output);
    free(answer_check);

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

