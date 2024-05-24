#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <iostream>
#include <vector>
#include <fstream>

// Error checking macro for CUDA Runtime API
#define CUDA_SAFE_CALL(call) { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(error) << ", in file " << __FILE__ << ", at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}


// Error checking macro for CUDA Driver API
#define CU_SAFE_CALL(call) { \
    CUresult error = call; \
    if (error != CUDA_SUCCESS) { \
        const char *errorString; \
        cuGetErrorName(error, &errorString); \
        std::cerr << "CUDA Driver Error: " << errorString << ", in file " << __FILE__ << ", at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// Error checking for NVRTC
#define NVRTC_SAFE_CALL(call) { \
    nvrtcResult result = call; \
    if (result != NVRTC_SUCCESS) { \
        std::cerr << "NVRTC Error: " << nvrtcGetErrorString(result) << ", in file " << __FILE__ << ", at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

const char* cudaKernelCode = R"(
extern "C"
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
)";

int main(int argc, char** argv) {
    // Parse command line arguments for compiler options
    // for example, --fmod=false: disable fused multiply-add, 
    //              --use_fast_math: replaces certain standard floating-point math functions 
    //                               with faster versions that may be less accurate.

    std::vector<std::string> compileOptions;
    for (int i = 1; i < argc; ++i) {
        compileOptions.push_back(argv[i]);
    }

    // Convert std::string options to const char* array for NVRTC
    std::vector<const char*> options;
    for (const auto& opt : compileOptions) {
        options.push_back(opt.c_str());
    }    


    // Initialize CUDA Driver API
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        const char *errorString;
        cuGetErrorName(res, &errorString);
        std::cerr << "Failed to initialize CUDA Driver API: " << errorString << std::endl;
        return 1;
    }    

    // Kernel parameters
    int N = 1024;
    size_t bytes = N * sizeof(float);

    // Allocate memory on host and device
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];

    CUDA_SAFE_CALL(cudaMalloc(&d_A, bytes));
    CUDA_SAFE_CALL(cudaMalloc(&d_B, bytes));
    CUDA_SAFE_CALL(cudaMalloc(&d_C, bytes));

    // Initialize arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    CUDA_SAFE_CALL(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));    


    // Use NVRTC to compile the kernel
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, cudaKernelCode, "vectorAdd.cu", 0, NULL, NULL));

    // Compile the kernel
    nvrtcResult compileResult = nvrtcCompileProgram(prog, options.size(), options.data());

    // Check compile errors
    if (compileResult != NVRTC_SUCCESS) {
        size_t logSize;
        NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
        std::vector<char> log(logSize);
        NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.data()));
        std::cerr << "Compilation log: " << log.data() << std::endl;
        NVRTC_SAFE_CALL(compileResult); // This will exit due to the error
    }

    // Obtain the PTX from the program
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    std::vector<char> ptx(ptxSize);
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));

    // Optionally save ptx to a file
    std::ofstream ptxFile("ptxVectorAdd.ptx", std::ios::binary);  // Open in binary mode if the data is not text
    if (ptxFile.is_open()) {
        ptxFile.write(ptx.data(), ptx.size());
        ptxFile.close();
        std::cout << "PTX code has been written to kernel.ptx" << std::endl;
    } else {
        std::cerr << "Failed to open file for writing PTX." << std::endl;
    }    

    // Load the PTX and get the kernel handle
    CUmodule module;
    CUfunction kernel;
    CU_SAFE_CALL(cuModuleLoadData(&module, ptx.data()));
    CU_SAFE_CALL(cuModuleGetFunction(&kernel, module, "vectorAdd"));

    // Launch the kernel
    void* args[] = { &d_A, &d_B, &d_C, &N };
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    CU_SAFE_CALL(cuLaunchKernel(kernel, blocksPerGrid, 1, 1, threadsPerBlock, 1, 1, 0, 0, args, 0));

    // Copy result back to host
    CUDA_SAFE_CALL(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Check results
    for (int i = 0; i < N; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            std::cerr << "Result verification failed at element " << i << std::endl;
            exit(1);
        }
    }

    std::cout << "Test PASSED" << std::endl;

    // Cleanup
    CUDA_SAFE_CALL(cudaFree(d_A));
    CUDA_SAFE_CALL(cudaFree(d_B));
    CUDA_SAFE_CALL(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
    CU_SAFE_CALL(cuModuleUnload(module));

    return 0;
}
