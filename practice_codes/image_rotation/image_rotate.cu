#include <cmath>
#include <iomanip>
#include <math.h>
#include <iostream>
#include <assert.h>

const int TILE_SIZE = 32;
const int BLOCK_ROWS =8; 


#ifdef NAIVE
__global__ void rotate_matrix_simple(float *output, const float *input, const int Width, const int Height) 
{
    int inCol = blockIdx.x * TILE_SIZE + threadIdx.x;
    int inRow = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    /* Index of Transposed matrix: [tranRow][tranCol], 

       where,
       tranWidth = Height;
       tranRow   = inCol; 
       tranCol   = inRow + j;

       (j because each thread reads in more than 1 rows in our setup. see transpose.cu.)

     * For clockwise rotation, swap column indices.
       Index of output matrix rotated after switching column indices: [outRow][outCol]

       where,
       outWidth = tranWidth;
       outRow   = tranRow;
       outCol   = tranWidth - tranCol - 1;

       By substituting for tranWidth, tranRow, and tranCol,

       outWidth = Height;
       outRow   = inCol;
       outCol   = Height - (inRow + j) - 1;

       linearized index: outRow*outWidth + outCol,
       i.e. inCol*Height + Height - (inRow + j) - 1

     */

    for (int j=0; j < TILE_SIZE; j += BLOCK_ROWS) {
        int row_glo = inRow + j;
        int outCol  = Height - row_glo - 1;
        output[inCol*Height + outCol] = input[row_glo*Width + inCol];
    }
}
#endif


#ifdef CORNER_TURNING
__global__ void rotate_matrix_sharedtile(float *output, 
                                         const float *input, 
                                         const int Width, const int Height) 
{
    /* first part is the same as transpose with NO_BANK_CONFLICT */

    __shared__ float tile[TILE_SIZE][TILE_SIZE+1];

    int inCol = blockIdx.x * TILE_SIZE + threadIdx.x;
    int inRow = blockIdx.y * TILE_SIZE + threadIdx.y;

    for (int j=0; j < TILE_SIZE; j += BLOCK_ROWS)
    {
        tile[threadIdx.y+j][threadIdx.x] = input[(inRow+j)*Width + inCol];
    }
    __syncthreads(); 

    inRow = blockIdx.x * TILE_SIZE + threadIdx.y; //same as transpose

    /* column index of the block is transposed and swapped for clockwise rotation. 
     * see serial algorithm in main for better understanding.
     */
    int colId_td0 = gridDim.y - blockIdx.y - 1;
    inCol = colId_td0*TILE_SIZE + threadIdx.x; 

    /* another way, 
     * int colId_td0 = blockIdx.y - (gridDim.y-1);
     * inCol = std::abs(colId_td0)*TILE_SIZE + threadIdx.x; 
     */

    for (int j=0; j < TILE_SIZE; j += BLOCK_ROWS) 
    {
        /* columns of tile are transposed and swapped.
           we swap indices in x, because we didn't actually transpose tile after reading.
           new_ refers to after rotation.
         */
	    int new_tx = TILE_SIZE - threadIdx.x - 1;
	    int new_ty = threadIdx.y+j;
	 
	    //so we first rotation along x then transpose.
        output[(inRow+j)*Height + inCol] = tile[new_tx][new_ty];
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
    if(test_passed) std::cout << "Image Rotation Test Passed! \n";
}


int main (int argc, char* argv[])
{ 
    /*define dimensions*/
    const int Width = 1024;
    const int Height = 512;

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

    /*initializing input array*/
    for (int j=0; j < Height; ++j) {
	    for (int i=0; i < Width; ++i) {
	        h_input [j*Width + i] = j*Width + i;
	    }
    }
    /* correct answer for error checking */
      /* First do transpose and store in answer_check */
    for (int j=0; j < Height; ++j) {
	    for (int i=0; i < Width; ++i) {
	        answer_check [i*Height + j] = h_input[j*Width + i];
	    }
    }

    /* Then swap columns for rotation in clockwise direction (done here). 
     * For anti-clockwise rotation swap rows 
     */
    int newHeight = Width;
    int newWidth = Height;

    for(int j=0; j < newHeight; ++j) 
    {
	    for (int i=0; i < int(newWidth/2); ++i) 
        {     
            float temp = answer_check[j*newWidth + i];	
            answer_check[j*newWidth+i] = answer_check[j*newWidth + (newWidth-i-1)];
            answer_check[j*newWidth + (newWidth-i-1)] = temp;
	    }
    }

    #ifdef PRINT
    std::cout << "Writing input matrix:\n";
    print_matrix(h_input, Width, Height);

    std::cout << "Writing correct answer matrix:\n";
    print_matrix(answer_check, Height, Width);
    #endif

    /*check parameters*/
    if(Width % TILE_SIZE || Height % TILE_SIZE) {
        std::cout << "Width and Heigh must be a multipler of TILE_SIZE\n";
	    goto error_exit;
    }

    if(TILE_SIZE % BLOCK_ROWS) {
        std::cout << "TILE_SIZE must be a multipler of BLOCK_ROWS\n";
	    goto error_exit;
    }

    cudaMemcpy(d_input, h_input, mat_memsize, cudaMemcpyHostToDevice); 

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float ms;
    cudaEventRecord(startEvent, 0);

    #ifdef NAIVE
    rotate_matrix_simple<<< dimGrid, dimBlock >>>(d_output, d_input, Width, Height);	
    #elif CORNER_TURNING
    rotate_matrix_sharedtile<<< dimGrid, dimBlock >>>(d_output, d_input, Width, Height);	
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

    check_error(h_output, answer_check, Width*Height);

error_exit:

    /*free memory*/
    free(h_input);
    free(h_output);
    free(answer_check);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

