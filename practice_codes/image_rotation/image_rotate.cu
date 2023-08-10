#include <cmath>
#include <iomanip>
#include <math.h>
#include <iostream>
#include <assert.h>

const int TILE_SIZE = 32;
const int BLOCK_ROWS =8; /*launching fewer threads than the tile size in the y direction, i.e. each thread will read in more rows*/

__global__ void rotate_matrix_simple(float *output, const float *input, const int Width, const int Height) 
{
    /*Threadcoarsening in the row direction. Using a thread block with fewer threads than elements in a tile 
      is advantageous for the matrix transpose because each thread transposes four matrix elements, as a result 
      much of the index calculation cost is amortized over these elements.*/
    /*The loop iterates over the second dimension and not the first so that contiguous threads load and store contiguous data*/

    /*In this kernel, reads are coalesced, but writes are not*/


    /*first indices of the block are (blockIdx.y*TILE_SIZE, blockIdx.x*TILE_SIZE) */

    int inCol = blockIdx.x * TILE_SIZE + threadIdx.x;
    int inRow = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    for (int j=0; j < TILE_SIZE; j += BLOCK_ROWS) {
        /*row = inRow + j, col = inCol */
        output[inCol*Height + (inRow+j)] = input[(inRow+j)*Width + inCol];
    }

    //int inRow = blockIdx.y*blockDim*y + threadIdx.y;
    //int inCol = blockIdx.x*blockDim*x + threadIdx.x;

    ///*Index of Transposed matrix: [inCol][inRow] linearized as inCol*Height + inRow*/
    ////int tranWidth = Height;
    ////int tranRow = inCol; 
    ////int tranCol = inRow;

    ///*Index of output matrix rotated after switching column indices: [inCol][abs(inRow-Height)]*/
    //int outWidth = Height;
    //int outRow = inCol;
    //int outCol = abs(inRow - Height);

    //if(inRow < Height && inCol < Width) 
    //{
    //    output[outRow*outWidth + outCol] = input[inRow*Width + inCol];      
    //}
}


__global__ void rotate_matrix_sharedtile(float *output, const float *input, const int Width, const int Height) 
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE+1];

    int inCol = blockIdx.x * TILE_SIZE + threadIdx.x;
    int inRow = blockIdx.y * TILE_SIZE + threadIdx.y;

    /*copy data to shared time*/
    for (int j=0; j < TILE_SIZE; j += BLOCK_ROWS)
    {
        tile[threadIdx.y+j][threadIdx.x] = input[(inRow+j)*Width + inCol];
    }
    __syncthreads(); /*we need this because threads write different data to output than they read from input*/

    /*here only the block is offset; this insures that the write will be contiguous*/
    //swapping columns after transpose
    int colId_td0 = blockIdx.y - (gridDim.y-1);
    inCol = std::abs(colId_td0)*TILE_SIZE + threadIdx.x; 
    inRow = blockIdx.x * TILE_SIZE + threadIdx.y;

    for (int j=0; j < TILE_SIZE; j += BLOCK_ROWS) 
    {
        //swapping columns after transpose
	//we rotate along x because we didn't actually transpose tile after reading.
	int new_tx = threadIdx.x - (TILE_SIZE-1);
	int new_ty = threadIdx.y+j;

        output[(inRow+j)*Height + inCol] = tile[std::abs(new_tx)][new_ty];
    }
}


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
    std::cout << "Matrix (width/cols, height/rows): "    << std::setw(10) << Width << std::setw(10) << Height << "\n";
    std::cout << "TILE_SIZE (width/cols, height/rows): " << std::setw(10) << TILE_SIZE  << std::setw(10) << BLOCK_ROWS << "\n";

    std::cout << "dimGrid (x,y,z):  "<< std::setw(10) << dimGrid.x  << std::setw(10) << dimGrid.y << std::setw(10) << dimGrid.z << "\n";
    std::cout << "dimBlock (x,y,z): "<< std::setw(10) << dimBlock.x << std::setw(10) << dimBlock.y << std::setw(10) << dimBlock.z << "\n";

    /*cudaSetDevice(devID)*/

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
    /*correct answer for error checking*/
    /*first do transpose and store in answer_check*/
    for (int j=0; j < Height; ++j) {
	for (int i=0; i < Width; ++i) {
	    answer_check [i*Height + j] = h_input[j*Width + i];
	}
    }
    /*swap columns*/
    int newHeight = Width;
    int newWidth = Height;

    for(int j=0; j < newHeight; ++j) {
	for (int i=0; i < int(newWidth/2); ++i) {     
            float temp = answer_check[j*newWidth + i];	
            answer_check[j*newWidth+i] = answer_check[j*newWidth + (newWidth-i-1)];
            answer_check[j*newWidth + (newWidth-i-1)] = temp;
	}
    }
    //std::cout << "Writing input matrix:\n";
    //print_matrix(h_input, Width, Height);

    //std::cout << "Writing correct answer matrix:\n";
    //print_matrix(answer_check, Height, Width);


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

    /*invoke a kernel*/
    //rotate_matrix_simple<<< dimGrid, dimBlock >>>(d_output, d_input, Width, Height);	
    rotate_matrix_sharedtile<<< dimGrid, dimBlock >>>(d_output, d_input, Width, Height);	

    cudaMemcpy(h_output, d_output, mat_memsize, cudaMemcpyDeviceToHost); 
  
    //std::cout << "Writing output matrix:\n";
    //print_matrix(h_output, Height, Width);

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

