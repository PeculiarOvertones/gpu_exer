#include <cmath>
#include <iomanip>
#include <math.h>
using namespace amrex;

__global__ void rotate_matrix(int *input, int *output, int Width, int Height) 
{

    int inRow = blockIdx.y*blockDim*y + threadIdx.y;
    int inCol = blockIdx.x*blockDim*x + threadIdx.x;

    /*Index of Transposed matrix: [inCol][inRow] linearized as inCol*Height + inRow*/
    //int tranWidth = Height;
    //int tranRow = inCol; 
    //int tranCol = inRow;

    /*Index of output matrix rotated after switching column indices: [inCol][abs(inRow-Height)]*/
    int outWidth = Height;
    int outRow = inCol;
    int outCol = abs(inRow - Height);

    if(inRow < Height && inCol < Width) 
    {
        output[outRow*outWidth + outCol] = input[inRow*Width + inCol];      
    }
}


void set_zero(int *M) 
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


void print_matrix(int *M, int width, int height) 
{
    for (int row = 0; row < height; ++row) 
    {
        for (int col = 0; col < width; ++col) 
	{
            std::cout << std::setw(5) << M[row*width + col];
	}
        std::cout << "\n";
    }
    std::cout << "\n";
}


int main (int argc, char* argv[])
{ 
    int Width = 5;
    int Height = 5;

    int matsize = Width*Height*sizeof(int);

    int block_size = 2;

    dim3 dimGrid(ceil(Width/block_size), ceil(Height/block_size), 1);    

    dim3 dimBlock(block_size, block_size, 1);    

    int* h_input = 0, *h_output = 0;

    h_input = (int *) malloc(matsize);
    h_output = (int *) malloc(matsize);

    std::cout << "Writing input matrix:\n";
    print_matrix(h_input);

    int* d_input = NULL;
    cudaMalloc((void **) &d_input, matsize);
 
    cudaMemcpy(d_input, h_input, matsize, cudaMemcpyHostToDevice); 

    int* d_output = NULL;
    cudaMalloc((void **) &d_output, matsize);

    rotate_matrix<<< dimGrid, dimBlock >>>(d_input, d_output, Width, Height);	

    cudaMemcpy(h_output, d_output, matsize, cudaMemcpyDeviceToHost); 

    std::cout << "Writing output matrix:\n";
    print_matrix(h_output);

    free(h_input);
    free(h_output);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
