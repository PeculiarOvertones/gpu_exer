/* A simple CUDA code to demonstrate use of CUDA streams to overlap
 * memory transfers and kernel executions.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>


// modifiable
const int num_streams = 8;
const size_t num_chunks = 16;
const size_t N = 1024*num_chunks;
const size_t num_threads = 256;


//Example kernel operation to double the input d_x and store it in d_y
__global__ void kernel(float *d_x, float *d_y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_y[idx] = d_x[idx] * 2; 
    }
}

//Test function to verify correctness of the output
bool test(float *h_x, float *h_y) {
    for (int i = 0; i < N; ++i) {
        if (h_y[i] != h_x[i] * 2) {
            std::cerr << "Test failed at index " << i << std::endl;
            return false;
        }
    }
    return true;
}

int main() 
{
    size_t size_float = N * sizeof(float);

    float *h_x, *h_y, *d_x, *d_y;

    h_x = (float*)malloc(size_float);
    h_y = (float*)malloc(size_float);

    //Initialize input data
    for (int i = 0; i < N; ++i) {
        h_x[i] = i + 1;
    }
    
    cudaMalloc((void**)&d_x, size_float);
    cudaMalloc((void**)&d_y, size_float);

    cudaStream_t stream[num_streams];

    //Create CUDA streams
    for(int s=0; s < num_streams; ++s) {
        cudaStreamCreate(&stream[s]);
    }

    //Start performance measurement
    auto start = std::chrono::high_resolution_clock::now();

    //Copy data to device, launch kernel, and copy data back using streams
    size_t chunk_size = (N/num_chunks);

    for (int c=0; c < num_chunks; ++c) {

        cudaMemcpyAsync(d_x+c*chunk_size, h_x+c*chunk_size, chunk_size*sizeof(float), cudaMemcpyHostToDevice, stream[c%num_streams]);

        kernel<<<ceil(chunk_size/num_threads), num_threads, 0, stream[c%num_streams]>>>(d_x + c*chunk_size, d_y + c*chunk_size);

        cudaMemcpyAsync(h_y+c*chunk_size, d_y+c*chunk_size, chunk_size*sizeof(float), cudaMemcpyDeviceToHost, stream[c%num_streams]);
    }

    //Query and synchronize streams
    for(int s=0; s < num_streams; ++s) {
        cudaStreamQuery(stream[s]);
    }

    cudaStreamSynchronize(stream[num_streams-1]);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double elapsed_time = duration.count() * 1000; // Convert to milliseconds

    //Verify output
    bool success = test(h_x, h_y);
    std::cout << "Test " << (success ? "passed" : "failed") << std::endl;

    std::cout << "Elapsed time: " << elapsed_time << " ms" << std::endl;

    for(int s=0; s < num_streams; ++s) {
        cudaStreamDestroy(stream[s]);
    }

    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
