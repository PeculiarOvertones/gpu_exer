#include <cmath>
#include <iomanip>
#include <math.h>
#include <iostream>
#include <assert.h>
/* Compile with one of the following options for histogram:
 * NAIVE
 * PRIVATIZATION
 * THREADCOARSENING_CONTIGUOUS
 * THREADCOARSENING_INTERLEAVED
 * AGGREGATION
 *
 * Use PRINT_DATA for printing data 
 * Use PRINT_HIST for printing histogram.
 * Use BIASED_DATA for initializing non-uniform probability distribution for the data.
 **/

const int BIN_SIZE = 4;

#if defined(THREADCOARSENING_CONTIGUOUS) || defined(THREADCOARSENING_INTERLEAVED) || defined(AGGREGATION)
const int COARSE_FACTOR = 8;
#endif


#ifdef NAIVE
__global__ void hist_naive(unsigned int *hist, const char *data, const int dataLength) 
{
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
 
    if(i < dataLength) 
    {
        int value = data[i] - 'a'; //type promotion to int when two chars are subtracted.

        if(value >=0 && value < 26) 
        {
            atomicAdd(&(hist[value/BIN_SIZE]),1);
        }
    }
}
#endif


#ifdef PRIVATIZATION
__global__ void hist_privatization(unsigned int *hist, const char *data, 
                                   const int dataLength, const int histLength) 
{
    extern __shared__ unsigned int sharedmem[];
    unsigned int* hist_s =  sharedmem;
    
    /* Initialize shared memory to zero.
     * Account for the possibility that the number of bins can be greater than 
     * the number of threads in a block.
     */
    for(int bin=threadIdx.x; bin < histLength; bin += blockDim.x) {
        hist_s[bin] = 0u;
    }
    __syncthreads();

    /* Map to shared memory histogram */
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;

    if(i < dataLength) 
    {
        int value = data[i] - 'a'; //type promotion to int when two chars are subtracted.

        if(value >=0 && value < 26) 
        {
            atomicAdd(&(hist_s[value/BIN_SIZE]),1);
        }
    }

    __syncthreads();

    /* Commit to from shared to global memory. */
    for(int bin=threadIdx.x; bin < histLength; bin += blockDim.x) 
    {
        unsigned int binValue = hist_s[bin];
        if(binValue > 0) {
            atomicAdd(&(hist[bin]), binValue);
        }
    }
}
#endif


#ifdef THREADCOARSENING_CONTIGUOUS
__global__ void hist_threadcoarsening_contiguous(unsigned int *hist, const char *data, 
                                   const int dataLength, const int histLength) 
{
    extern __shared__ unsigned int sharedmem[];
    unsigned int* hist_s =  sharedmem;

    /* Initialize hist_s */
    for (unsigned int bin=threadIdx.x; bin < histLength; bin+= blockDim.x) 
    {
        hist_s[bin] = 0u;    
    }

    __syncthreads();

    /* Map data to hist_s. Note for loop. */
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;
    for(int i=tid*COARSE_FACTOR; i< min((tid+1)*COARSE_FACTOR,dataLength); ++i) 
    {
        int value = data[i] - 'a';
        if(value >=0 && value < 26) 
        {
            atomicAdd(&(hist_s[value/BIN_SIZE]),1);
        }
    }
    __syncthreads();

    /* Commit to from shared to global memory. */
    for(int bin=threadIdx.x; bin < histLength; bin+= blockDim.x ) 
    {
        unsigned int binValue = hist_s[bin];
        if(binValue > 0) 
        {
            atomicAdd(&(hist[bin]), binValue);
        }
    }
}
#endif


#ifdef THREADCOARSENING_INTERLEAVED
__global__ void hist_threadcoarsening_interleaved(unsigned int *hist, const char *data, 
                                   const int dataLength, const int histLength) 
{
    extern __shared__ unsigned int sharedmem[];
    unsigned int* hist_s =  sharedmem;

    /* Initialize hist_s */
    for (unsigned int bin=threadIdx.x; bin < histLength; bin+= blockDim.x) 
    {
        hist_s[bin] = 0u;    
    }

    __syncthreads();

    /* Map data to hist_s. Note for loop. */
    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;

    /* Note forloop bound and increment. 
       gridDim.x*blockDim.x < dataLength, because we launched fewer threads per block.
       i.e., blockDim.x < dataLength/gridDim.x, as
       blockDim.x / (dataLength/gridDim.x) = COARSE_FACTOR  */

    for(int i=tid; i< dataLength; i+= gridDim.x*blockDim.x) 
    {
        int value = data[i] - 'a';
        if(value >=0 && value < 26) 
        {
            atomicAdd(&(hist_s[value/BIN_SIZE]),1);
        }
    }
    __syncthreads();

    /* Commit to from shared to global memory. */
    for(int bin=threadIdx.x; bin < histLength; bin+= blockDim.x ) 
    {
        unsigned int binValue = hist_s[bin];
        if(binValue > 0) 
        {
            atomicAdd(&(hist[bin]), binValue);
        }
    }
}
#endif


#ifdef AGGREGATION
__global__ void hist_aggregation(unsigned int *hist, const char *data, 
                                 const int dataLength, const int histLength) 
{
    extern __shared__ unsigned int sharedmem[];
    unsigned int* hist_s =  sharedmem;

    /* Initialize hist_s */
    for (unsigned int bin=threadIdx.x; bin < histLength; bin+= blockDim.x) 
    {
        hist_s[bin] = 0u;    
    }
    __syncthreads();

    /* Map data to shared memory. 
     * For loop is the same as threadcoarsening with interleaved partitioning.
     * This routine may do well for biased data.
     */
    unsigned int accumulator = 0;
    int prevBinIdx = -1; 

    unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;

    for(int i=tid; i < dataLength; i+= gridDim.x*blockDim.x) 
    {
        int value = data[i] - 'a';
        if(value >=0 && value < 26) 
        {
            int bin = value/BIN_SIZE;		
	        if(bin == prevBinIdx) 
	        {
        		++accumulator;
	        } else {
                /* first commit existing accumulator */
		        if(accumulator > 0) {    
                    atomicAdd(&(hist_s[prevBinIdx]), accumulator);
		        }
		        accumulator = 1;
		        prevBinIdx = bin;
	        }
        }
    }
    if(accumulator > 0) 
    {
        atomicAdd(&(hist_s[prevBinIdx]),accumulator);
    }
    __syncthreads();

    /* Commit from shared to global memory. */
    for(int bin=threadIdx.x; bin < histLength; bin+= blockDim.x ) 
    {
        unsigned int binValue = hist_s[bin];
        if(binValue > 0) 
        {
            atomicAdd(&(hist[bin]), binValue);
        }
    }
}
#endif


void set_zero(unsigned int* array, int length) {
    for (int i = 0; i < length; ++i) {
        array[i] = 0;
    }
}


void printHistogram(const unsigned int* hist, int numBins) {
    std::cout << "Histogram:\n";
    for (int i = 0; i < numBins; ++i) {
        std::cout << "Bin " << i << ": " << hist[i] << "\n";
    }
}


void printData(const char* data, int length) {
    std::cout << "Data array: \n";
    for (int i = 0; i < length; ++i) {
        std::cout << data[i]; 
        if ((i + 1) % 50 == 0)
            std::cout << "\n";
        else if (i + 1 != length)
            std::cout << ", ";
    }
    std::cout << "\n";
}

void check_error(const unsigned int* h_hist, const unsigned int* hist_check, const int histLength)
{
    bool errorFound = false;
    for (int i = 0; i < histLength; ++i) {
        if (h_hist[i] != hist_check[i]) {
            std::cout << "Mismatch found at bin " << i
                      << ": GPU value = " << h_hist[i]
                      << ", CPU value = " << hist_check[i] << std::endl;
            errorFound = true;
        }
    }
    if (!errorFound) {
        std::cout << "Histogram Test Passed!\n";
    }
}

void initializeDataAndCreateHistogram(char* data, int dataLength, 
                                      unsigned int* hist, int histLength) 
{
    /* initializing input array */
#ifdef BIASED_DATA
     /* create biased data such that 'a' and 'o' appear more frequently */
     char biasedChars[] = {
        'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',  // 'a' appears more frequently
        'o', 'o', 'o', 'o', 'o',  // 'o' appears more frequently
        'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    };
    int numBiasedChars = sizeof(biasedChars) / sizeof(biasedChars[0]);
    for (int i = 0; i < dataLength; ++i) {
        data[i] = biasedChars[rand() % numBiasedChars];
    }
#else
    for (int i = 0; i < dataLength; ++i) {
        data[i] = static_cast<char>('a' + (rand() % 26));
    }
#endif

    /* correct histogram for error checking */
    set_zero(hist, histLength);    

    for (int i = 0; i < dataLength; ++i) {
        int value = data[i] - 'a'; //type promotion to int when two chars are subtracted.
        if (value >= 0 && value < 26) {
            hist[value / BIN_SIZE]++;
        }
    }
}


int main (int argc, char* argv[])
{ 
    const int dataLength = 1024*1024*1024;
    const int histLength = ceil(26.0/BIN_SIZE);

    int hist_memsize = sizeof(unsigned int)*histLength;
    int data_memsize = sizeof(char)*dataLength;

    std::cout << "dataLength: "   << dataLength << "\n";
    std::cout << "data size in (GB): " << data_memsize / std::pow(1024,3) << "\n";
    std::cout << "histLength: "   << histLength << "\n";
    std::cout << "BIN_SIZE: "     << BIN_SIZE << "\n";

    int blockSize = 4096;
    int threadsPerBlock = blockSize;
#if defined(THREADCOARSENING_CONTIGUOUS) || defined(THREADCOARSENING_INTERLEAVED) || defined(AGGREGATION)
    threadsPerBlock /= COARSE_FACTOR;
    std::cout << "COARSE_FACTOR: " << COARSE_FACTOR << "\n";
#endif
    int blocksPerGrid = (dataLength - 1)/blockSize + 1;

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
    char* h_data = (char*) malloc(data_memsize);
    unsigned int* h_hist = (unsigned int*) malloc(hist_memsize);
    unsigned int* hist_check = (unsigned int*) malloc(hist_memsize);

    char* d_data = NULL;
    cudaMalloc(&d_data, data_memsize);

    unsigned int* d_hist = NULL;
    cudaMalloc(&d_hist, hist_memsize);

    initializeDataAndCreateHistogram(h_data, dataLength, hist_check, histLength);

#ifdef PRINT_DATA
    printData(h_data, dataLength);
#endif

#ifdef PRINT_HIST
    std::cout << "Correct answer for ";
    printHistogram(hist_check, histLength);
#endif

    cudaMemcpy(d_data, h_data, data_memsize, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, hist_memsize);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float ms;
    cudaEventRecord(startEvent, 0);

#ifdef NAIVE
    hist_naive<<<blocksPerGrid, threadsPerBlock>>>(d_hist, d_data, dataLength);
#elif PRIVATIZATION
    hist_privatization<<<blocksPerGrid, threadsPerBlock, histLength>>>
                      (d_hist, d_data, dataLength, histLength);
#elif THREADCOARSENING_CONTIGUOUS
    hist_threadcoarsening_contiguous<<<blocksPerGrid, threadsPerBlock, histLength>>>
                                    (d_hist, d_data, dataLength, histLength);
#elif THREADCOARSENING_INTERLEAVED
    hist_threadcoarsening_interleaved<<<blocksPerGrid, threadsPerBlock, histLength>>>
                                     (d_hist, d_data, dataLength, histLength);
#elif AGGREGATION
    hist_aggregation<<<blocksPerGrid, threadsPerBlock, histLength>>>
                       (d_hist, d_data, dataLength, histLength);
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

    cudaMemcpy(h_hist, d_hist, hist_memsize, cudaMemcpyDeviceToHost); 
  
#ifdef PRINT_HIST
    std::cout << "\nGPU computed ";
    printHistogram(h_hist, histLength);
#endif

    check_error(h_hist, hist_check, histLength);

    /*free memory*/
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    free(h_data); h_data = NULL;
    free(h_hist); h_hist = NULL;
    free(hist_check); hist_check = NULL;

    cudaFree(d_data); d_data = NULL;
    cudaFree(d_hist); d_hist = NULL;

    cudaDeviceReset();
    return 0;
}
