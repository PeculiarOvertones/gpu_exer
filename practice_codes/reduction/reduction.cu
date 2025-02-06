#include <cmath>
#include <iomanip>
#include <math.h>
#include <iostream>
#include <assert.h>

/* Compile with one of the following options for histogram:
 * NAIVE
 * CONVERGENT
 * SHAREDMEM
 * HIERARCHICAL
 * THREADCOARSENING
 *
 * Use PRINT_INPUT for printing input data.
 **/
#ifdef THREADCOARSENING
const int COARSE_FACTOR = 2;
#endif

#ifdef NAIVE
__global__ void reduce_sum_naive(float *output, float *in) 
{
    /* - this kernel doesn't preserve input array. 
     * - Assume kernel is launched with:
     *   threadsPerBlock = dataLength/2
     *   gridDim.x = 1
     *
     *   visualization
         | represents blockDim.x
         * represent active threads
         0 1 2 3 4 5 6 7
         *   *   *   *  (stride 1)
         0   2   4   6
         *       *      (stride 2)
         0       4
         *              (stride 4)

    * Problem with this:
    * No memory coalescing (adjacent threads in warp accessing increasingly distant locations 
    * in memory, causing increased control divergence and more global memory requests.
    * Low execution resource utilization (ratio of scheduled resources 
    * including inactive threads in a warp / resources actually used, 
    * i.e. number of active threads doing the work)
    */

    unsigned int i = 2*threadIdx.x;

    for(unsigned int stride = 1; stride <= blockDim.x ; stride *= 2) {
       if(threadIdx.x % stride  == 0) {
            in[i] += in[i + stride];
       }
       __syncthreads();
    }

    if(threadIdx.x == 0) {
        *output = in[0];
    }
}
#endif


#ifdef CONVERGENT
__global__ void reduce_sum_convergent(float *output, float *in) 
{
    /* - less control divergence and better memory coalescing.
     * - doesn't preserve input array.
     * - Assume kernel is launched with:
     *   threadsPerBlock = dataLength/2
     *   gridDim.x = 1
     *
     *   visualization
         | represents blockDim.x
         * represent active threads
         0 1 2 3 | 4 5 6 7
         * * * *          (stride 4)
         0 1 2 3          
         * *              (stride 2)
         0 1
         *                (stride 1)
    *
    * Advantages:    
    * Memory is coalesced. Less control divergence. Fewer global memory requests. More efficient bandwidth utilization.
    * Better execution resource utilization.
    * 
    */

    unsigned int i = threadIdx.x;

    for(unsigned int stride = blockDim.x; stride >=1 ; stride /= 2)
    {
       if(threadIdx.x < stride) 
       {
           in[i] += in[i + stride]; /*no need for atomic*/
       }
       __syncthreads();
    }
    if(threadIdx.x == 0) {
        *output = in[0];
    }
}
#endif


#ifdef SHAREDMEM
__global__ void reduce_sum_sharedmem(float *output, const float *in) 
{
    /* - fewer accesses to global memory.
     * - preserves input array.
     * - Assume kernel is launched with:
     *   threadsPerBlock = dataLength/2
     *   gridDim.x = 1
     *
     * Advantages:
     * Fewer global memory accesses. Only N+1 (only while initially reading data into SMEM 
     * and writing data to output.)
     * Input array is not modified.
     **/
    extern __shared__ float sharedmem[];
    float* in_s = sharedmem;

    unsigned int t = threadIdx.x;

    //load tile and add elements blockDim.x distance away.
    in_s[t] = in[t] + in[t+blockDim.x];

    for(unsigned int stride = blockDim.x/2; stride >=1 ; stride /= 2) {
       __syncthreads();

       if(t < stride) {
           in_s[t] += in_s[t + stride]; /*no need for atomic*/
       }
    }
    if(t == 0) {
        *output = in_s[0];
    }
}
#endif


#ifdef HIERARCHICAL
__global__ void reduce_sum_hierarchical(float *output, const float *in) 
{
    /* - works for large data.
     * - preserves input array.
     * - Assume kernel is launched with:
     *   gridDim.x = dataLength/block_size i.e. (dataLength-1)/block_size + 1
     *   where block_size = 2*threadsPerBlock
     *   fix threadsPerBlock (blockDim.x) to say 64
     *   dim3 dimGrid(blocksPerGrid,1,1);
     *   dim3 dimBlock(threadsPerBlock,1,1);
     *
     *   e.g. dataLength = 256
     *   threadPerBlock=64 (blockDim.x)
     *   block_size=128 
     *   blocksPerGrid = 2 (gridDim.x)
     *   sharedmem size = threadsPerBlock
     *   dataLength_local = 2*blockDim.x
     *   segment = dataLength_local*blockIdx.x;
     *   i = segment + threadIdx.x;
     *
     * Advantages:
     * We can launch multiple threadblocks.
     */    

    extern __shared__ float sharedmem[];
    float* in_s = sharedmem;

    unsigned int segment =  2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    in_s[threadIdx.x] = in[i] + in[i + blockDim.x];

    for(unsigned int stride = blockDim.x/2; stride >=1 ; stride /= 2) 
    {
       __syncthreads();

       if(threadIdx.x < stride) 
       {
           in_s[threadIdx.x] += in_s[threadIdx.x + stride]; /*no need for atomic*/
       }
    }

    if(threadIdx.x == 0) 
    {
        atomicAdd(output, in_s[0]);
    }
}
#endif


#ifdef THREADCOARSENING
__global__ void reduce_sum_threadcoarsening(float *output, const float *in) 
{
    /* - works for large data.
     * - preserves input array.
     * - Assume kernel is launched with:
     *   gridDim.x = dataLength/(2*COARSE_FACTOR*threadsPerBlock));
     *
     *   fix threadsPerBlock, e.g. 64
     *   then block_size = 2*threadsPerBlock*COARSE_FACTOR
     *   blocksPerGrid = (dataLength-1)/block_size + 1; 
     *   dim3 dimGrid(blocksPerGrid,1,1);
     *   dim3 dimBlock(threadsPerBlock,1,1);
     *
     *   e.g. dataLength = 512
     *   threadPerBlock=64 (blockDim.x)
     *   block_size = 256  (assuming COARSE_FACTOR = 2)
     *   blocksPerGrid= 2  (gridDim.x)
     *   sharedmem size = threadsPerBlock
     *   dataLength_local = 2*blockDim.x*COARSE_FACTOR
     *   segment = dataLength_local*blockIdx.x;
     *   i = segment + threadIdx.x;
     *
     * Advantages:
     * In processors with limited execution resources, the hardware may only have limited 
     * resources to execute threadblocks in parallel. In this case, the hardware will serialize
     * the surplus threadblocks, executing a new thread block whenever an old one has completed.    * In this case, we pay price to distribute the work across multiple threadblocks.
     * Threadcoarsening serializes some of the work into fewer threads to reduce parallelization  overhead.
     **/
    extern __shared__ float sharedmem[];
    float* in_s = sharedmem;

    unsigned int segment =  2*COARSE_FACTOR*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    in_s[threadIdx.x] = 0.;
    float sum = 0.;
    for (int c=0; c < 2*COARSE_FACTOR; ++c) 
    {
        sum += in[i + c*blockDim.x];
    }
    in_s[threadIdx.x] = sum;

    for(unsigned int stride = blockDim.x/2; stride >=1 ; stride /= 2) 
    {
       __syncthreads();

       if(threadIdx.x < stride) 
       {
           in_s[threadIdx.x] += in_s[threadIdx.x + stride]; /*no need for atomic*/
       }
    }
    if(threadIdx.x == 0) 
    {
        atomicAdd(output, in_s[0]);
    }
}
#endif


void printData(const float* data, int length) {
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


/* initialize data and compute reduction sum */
void initializeDataAndComputeReduction(float* h_input, float& output, unsigned int dataLength) {
    float sum = 0.0f;

    // Seed for reproducibility (optional)
    srand(time(NULL));

    for (unsigned int i = 0; i < dataLength; ++i) {
        //h_input[i] = static_cast<float>(rand()) / RAND_MAX;  
        h_input[i] = static_cast<float>(rand() % 10);  
        sum += h_input[i];  
    }
    output = sum;
}


int main (int argc, char* argv[])
{
    /* make sure dataLength is a power of 2, 
       otherwise pad data with zeros */
    const int dataLength = 4096;

    int data_memsize = sizeof(float)*dataLength;

    std::cout << "dataLength: "   << dataLength << "\n";
    std::cout << "data size in (GB): " << data_memsize / std::pow(1024,3) << "\n";


#if defined(HIERARCHICAL) || defined(THREADCOARSENING)
    int threadsPerBlock = 64;
#else 
    int threadsPerBlock = dataLength/2;
#endif

    int blockSize = threadsPerBlock*2;
#ifdef THREADCOARSENING
    blockSize *= COARSE_FACTOR;
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
    float* h_input  = (float*) malloc(data_memsize);
    float h_output = 0.;
    float output_check = 0.;

    float* d_input = NULL;
    cudaMalloc(&d_input, data_memsize);

    float* d_output = NULL;
    cudaMalloc(&d_output, sizeof(float));

    initializeDataAndComputeReduction(h_input, output_check, dataLength);

#ifdef PRINT_INPUT
    std::cout << "Input: \n";
    printData(h_input, dataLength);
#endif

    std::cout << "Correct Output: " << output_check << "\n";

    cudaMemcpy(d_input, h_input, data_memsize, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float ms;
    cudaEventRecord(startEvent, 0);

#ifdef NAIVE
    reduce_sum_naive<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_input);
#elif CONVERGENT
    reduce_sum_convergent<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_input);
#elif SHAREDMEM
    reduce_sum_sharedmem<<<blocksPerGrid, threadsPerBlock, threadsPerBlock>>>(d_output, d_input);
#elif HIERARCHICAL
    reduce_sum_hierarchical<<<blocksPerGrid, threadsPerBlock, threadsPerBlock>>>
                           (d_output, d_input);
#elif THREADCOARSENING
    reduce_sum_threadcoarsening<<<blocksPerGrid, threadsPerBlock, threadsPerBlock>>>
                               (d_output, d_input);
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

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "GPU computed Output: " << h_output << "\n";

    if(std::fabs(h_output - output_check) < 1e-8) {
        std::cout << "Reduction Test Passed!\n";
    }
    else {
    std::cout << "Reduction Test Failed! correct value: " << output_check 
              << " GPU value: " << h_output << "\n";
    }

    /*free memory*/
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    free(h_input); h_input = NULL;

    cudaFree(d_input); d_input = NULL;
    cudaFree(d_output); d_output = NULL;

    cudaDeviceReset();
    return 0;

}
