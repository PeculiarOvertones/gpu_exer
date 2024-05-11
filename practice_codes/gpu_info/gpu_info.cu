#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>

int main (int argc, char* argv[])
{ 
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cout << "Device Number: " << i << "\n";
    std::cout << "  Device name: " << prop.name << "\n";
    std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << "\n";
    std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << "\n";
    std::cout << "  Total global memory (Gbytes): " <<
                 (float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0 << "\n";
    std::cout << "  Shared memory per block (Kbytes): " <<
                 (float)(prop.sharedMemPerBlock)/1024.0 << "\n";
    std::cout << "  Compute capability: " << prop.major << "-" << prop.minor << "\n";
    std::cout << "  Warp-size: " << prop.warpSize << "\n";
    std::cout << "  Concurrent kernels: " << (prop.concurrentKernels ? "yes" : "no") << "\n";
    std::cout << "  Concurrent computation/communication: " << (prop.deviceOverlap ? "yes" : "no") << "\n";

    float peak_memory_bandwidth = 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6; //GB/s
    std::cout << "  Peak Memory Bandwidth (GB/s): " << peak_memory_bandwidth << "\n";

  }

    return 0;
}

