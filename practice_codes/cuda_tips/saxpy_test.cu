#include <iostream>
#include <cuda_runtime.h>

// Define the kernel type: 1 for monolithic, 2 for grid-stride
#define KERNEL_TYPE 1

// Monolithic kernel
__global__
void saxpy_monolithic(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

// Grid-stride loop kernel
__global__
void saxpy_grid_stride(int n, float a, float *x, float *y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) {
        y[i] = a * x[i] + y[i];
    }
}

// Helper function to check CUDA errors
void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

// Host function to initialize data and verify results
void test_saxpy(int n, float a) {
    size_t size = n * sizeof(float);

    // Allocate host memory
    float *h_x = new float[n];
    float *h_y = new float[n];
    float *h_y_reference = new float[n];

    // Initialize data
    for (int i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>(i);
        h_y[i] = static_cast<float>(2 * i);
        h_y_reference[i] = h_y[i] + a * h_x[i];
    }

    // Allocate device memory
    float *d_x, *d_y;
    checkCuda(cudaMalloc(&d_x, size));
    checkCuda(cudaMalloc(&d_y, size));

    // Copy data to device
    checkCuda(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Select kernel based on preprocessor directive
#if KERNEL_TYPE == 1
    std::cout << "Running monolithic kernel..." << std::endl;
    saxpy_monolithic<<<blocksPerGrid, threadsPerBlock>>>(n, a, d_x, d_y);
#elif KERNEL_TYPE == 2
    std::cout << "Running grid-stride loop kernel..." << std::endl;
    saxpy_grid_stride<<<32, threadsPerBlock>>>(n, a, d_x, d_y);
#else
    #error "Invalid KERNEL_TYPE defined. Use 1 for monolithic or 2 for grid-stride."
#endif

    checkCuda(cudaDeviceSynchronize());

    // Copy result back to host
    float *h_result = new float[n];
    checkCuda(cudaMemcpy(h_result, d_y, size, cudaMemcpyDeviceToHost));

    // Verify results
    bool passed = true;
    for (int i = 0; i < n; ++i) {
        if (fabs(h_result[i] - h_y_reference[i]) > 1e-5) {
            passed = false;
            break;
        }
    }
    std::cout << (passed ? "Test PASSED" : "Test FAILED") << std::endl;

    // Cleanup
    delete[] h_x;
    delete[] h_y;
    delete[] h_y_reference;
    delete[] h_result;
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    int n = 1 << 20; // 1M elements
    float a = 2.0;

    test_saxpy(n, a);

    return 0;
}

