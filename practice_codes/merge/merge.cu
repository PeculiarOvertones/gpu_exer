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

void merge_sequetial(int *C, const  int *A, unsigned int m, const int* B, unsigned int n) 
{
    int a=0;	
    int b=0;
    int c=0;
    while(a < m && b < n) 
    {
        if(A[a] <= B[b] ) 
	{
           C[c++] = A[a++];
	}
	else 
	{
           C[c++] = B[b++];
	}
    }
    while (a < m) 
    {
        C[c++] = A[a++];
    }
    while (b < n) 
    {
        C[c++] = B[b++];
    }
}

int co_rank(int k, int* A, int m, int* B, int n)
{
    int i= k < m ? k : m; //min(k, m)
    int j=k-1;
    int i_low= 0 > k-n ? 0 : k-n; //max(0, k-n)
    int j_low= 0 > k-m ? 0 : k-m; //max(0, k-m)

    int delta;
    bool active = true;
    while(active) 
    {
        if(A[i-1] > B[j] && i > 0 &&  j < n) 
	{
	    delta = (i - i_low + 1) >> 1; //ceil((i-i_low)/2)
	    i = i - delta;
	    j_low = j;
	    j = j + delta;
	}
	else if(B[j-1] >= A[i] && j > 0 && i < m)
	{
	    delta = (j - j_low + 1) >> 1; 
	    j = j - delta;
	    i_loc = i;
	    i = i + delta;
	}
	else 
	{
            active = false;
	}
    }

    return i;
}

__global__ void merge_basic(int *C, const int *A, unsigned int m, const int* B, unsigned int n) 
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    int elems_per_thread = ceil((m+n)/(blockDim.x*gridDim.x));

    int k_curr = tid*elements_per_thread;
    int k_next = min((tid+1)*elements_per_thread, (m+n)); 

    int i_curr = co_rank(k, A, m, B, n);
    int j_curr = k_curr - i_curr;

    int i_next = co_rank(k, A, m, B, n);
    int j_next = k_next - i_next;

    merge_sequential(&C[k_curr], &A[i_curr], i_next - i_curr, &B[i_curr], j_next - j_curr);
}


__global__ void merge_tiled(int *C, const int *A, unsigned int m, const int* B, unsigned int n, int tile_size) 
{
    /*loading phase*/
    extern __shared__ int shareAB[];

    int * A_s = &shareAB[0];    
    int * B_s = &shareAB[tile_size];

    int elems_per_block = ceil((m+n)/gridDim.x);
    int C_curr = blockIdx.x*elems_per_block;
    int C_next = min((blockIdx.x+1)*elems_per_block, m+n);

    /*find co rank of C_curr and C_next and store it in shared memory*/
    if(threadIdx.x==0) 
    {
        A_s[0] = co_rank(C_curr, A, m, B, n);
	A_s[1] = co_rank(C_next, A, m, B, n);
    }
    __syncthreads();
    int A_curr = A_s[0];
    int A_next = A_s[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;

    __syncthreads();

    int C_length = C_next - C_curr;
    int B_length = B_next - B_curr;
    int A_length = A_next - A_curr;
    
    int counter = 0;

    int total_iterations = ceil((C_length/tile_size));

    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    while(cunter < total_iterations) 
    {
        /*load tiles*/
	
	__syncthreads();

	/*define c_curr, c_next, a_curr, a_next, b_curr, b_next*/


	/*merge seq*/
        merge_sequential(C + C_curr + C_completed + c_curr, A_s + a_curr, a_next - a_curr, B_s + b_curr, b_next - b_curr);

	/*end stuff-- updates*/
	counter++;
	C_completed += tile_size;
	A_consumed += co_rank(tile_size, A_s, tile_size, B_s, tile_size);;
	B_consumed = C_completed - A_consumed;
	__syncthreads();
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

