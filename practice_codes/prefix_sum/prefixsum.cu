#include <cmath>
#include <iomanip>
#include <math.h>
#include <iostream>
#include <assert.h>
/* Compile with one of the following options for histogram:
 * KOGG_STONE
 * KOGG_STONE_DOUBLE_BUFFER
 * BRENT_KUNG
 *
 * Use PRINT_INPUT for printing input data.
 * Use PRINT_OUTPUT for printing input data.
 **/

#define EXCLUSIVE 0 //0: inclusive scan, 1: exclusive scan
#define SECTION_SIZE 512

#ifdef KOGG_STONE
__global__ void prefix_sum_kogg_stone(float *Y, const float *X, unsigned int N) 
{
    __shared__ float A_s[SECTION_SIZE]; 	

    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;

    /*define shared memory*/
    #if EXCLUSIVE == 1
    if (threadIdx.x == 0) {
        printf("exclusive scan: \n"); // Debug output
    }
    if(i<N && threadIdx.x != 0) {
        A_s[threadIdx.x] = X[i-1];
    }
    else { 
        A_s[threadIdx.x] = 0.f;
    }
    #else 
    if (threadIdx.x == 0) {
        printf("inclusive scan: \n"); // Debug output
    }
    if(i<N) {
        A_s[threadIdx.x] = X[i];
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
__global__ void prefix_sum_kogg_stone_double_buffer(float *Y, const float *X, unsigned int N) 
{
    __shared__ float A_s[SECTION_SIZE]; 	
    __shared__ float B_s[SECTION_SIZE]; 	

    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;

    /*define shared memory*/
    #ifdef EXCLUSIVE
    if(i<N && threadIdx.x != 0) {
        A_s[threadIdx.x] = X[i-1];
        B_s[threadIdx.x] = A_s[threadIdx.x];
    }
    else { 
        A_s[threadIdx.x] = 0.f;
        B_s[threadIdx.x] = 0.f;
    }
    #else 
    if(i<N) {
        A_s[threadIdx.x] = X[i];
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
        Y[i] = read_ptr[threadIdx.x];
    }
}
#endif


#ifdef BRENT_KUNG
__global__ void prefix_sum_brent_kung(float *Y, const float *X, unsigned int N) 
{
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


void printData(const float* data, int length) {
    for (int i = 0; i < length; ++i) {
        std::cout << data[i];
        if ((i + 1) % 50 == 0)
            std::cout << "\n";
        else if (i + 1 != length)
            std::cout << ", ";
    }
    std::cout << "\n";
}


/* initialize data and compute prefix sum */
void initializeDataAndComputePrefixSum(float* h_input, float* output, unsigned int dataLength) {
    float sum = 0.0f;

    // Seed for reproducibility (optional)
    srand(time(NULL));

    for (unsigned int i = 0; i < dataLength; ++i) {
        h_input[i] = static_cast<float>(rand() % 10);
        sum += h_input[i];
        output[i] = sum;
    }
}


void check_error(const float* h_output, const float* output_check, unsigned int dataLength) {
    bool errorFound = false;
    for (unsigned int i = 0; i < dataLength; ++i) {
        if (fabs(h_output[i] - output_check[i]) > 1e-5) { 
            std::cerr << "Mismatch found at index: " << i << ": GPU value: " << h_output[i]
                      << ", correct value: " << output_check[i] << "\n";
            errorFound = true;
        }
    }

    if (!errorFound) {
        std::cout << "Prefix Sum Passed!\n";
    }
}


int main (int argc, char* argv[])
{
    /* works for dataLength <= SECTION_SIZE, i.e. gridDim.x = 1 */
    const int dataLength = SECTION_SIZE;

    int data_memsize = sizeof(float)*dataLength;

    std::cout << "dataLength: "   << dataLength << "\n";
    std::cout << "data size in (GB): " << data_memsize / std::pow(1024,3) << "\n";

#if defined(KOGG_STONE) || defined(KOGG_STONE_DOUBLE_BUFFER)
    int threadsPerBlock = SECTION_SIZE;
#elif BRENT_KUNG
    int threadsPerBlock = SECTION_SIZE/2;
#endif
    int blocksPerGrid = (dataLength - 1)/SECTION_SIZE + 1;

    std::cout << "\nblocks per grid: "   << blocksPerGrid << "\n";
    std::cout << "threads per block: " << threadsPerBlock << "\n";

    int devID=0;
    if(argc > 1) devID = atoi(argv[1]);

    cudaError_t err = cudaSetDevice(devID);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set device: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    /*define arrays on host and device*/
    float* h_input  = (float*) malloc(data_memsize);
    float* h_output = (float*) malloc(data_memsize);
    float* output_check = (float*) malloc(data_memsize);

    float* d_input = NULL;
    cudaMalloc(&d_input, data_memsize);

    float* d_output = NULL;
    cudaMalloc(&d_output, data_memsize);

    initializeDataAndComputePrefixSum(h_input, output_check, dataLength);

#ifdef PRINT_INPUT
    std::cout << "Input: \n";
    printData(h_input, dataLength);
#endif

#ifdef PRINT_OUTPUT
    std::cout << "Correct output: \n";
    printData(output_check, dataLength);
#endif

    cudaMemcpy(d_input, h_input, data_memsize, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, data_memsize);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float ms;
    cudaEventRecord(startEvent, 0);

#ifdef KOGG_STONE
    prefix_sum_kogg_stone<<<blocksPerGrid, threadsPerBlock>>>
                         (d_output, d_input, dataLength);
#elif KOGG_STONE_DOUBLE_BUFFER
    prefix_sum_kogg_stone_double_bugger<<<blocksPerGrid, threadsPerBlock>>>
                                       (d_output, d_input,dataLength);
#elif BRENT_KUNG
    prefix_sum_brent_kung<<<blocksPerGrid, threadsPerBlock>>>
                         (d_output, d_input, dataLength);
#endif

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    std::cout << "\nElapsed time to run kernel (ms): " << ms << "\n";

    //no need for device synchronize here since event synchronize is used before.
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaMemcpy(h_output, d_output, data_memsize, cudaMemcpyDeviceToHost);

#ifdef PRINT_OUTPUT
    std::cout << "GPU computed output: \n";
    printData(h_output, dataLength);
#endif

    check_error(h_output, output_check, dataLength);

    /*free memory*/
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    free(h_input); h_input = NULL;
    free(h_output); h_output = NULL;
    free(output_check); output_check = NULL;

    cudaFree(d_input); d_input = NULL;
    cudaFree(d_output); d_output = NULL;

    cudaDeviceReset();
    return 0;
}
