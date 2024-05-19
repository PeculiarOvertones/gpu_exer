Practice codes for GPU optimization.
- Example codes I have tried while studying GPU programming with CUDA.
- From resources usch as, "Programming Massively Parallel Processors" (4th edition), cuda_training_series from NVIDIA,
  and some other interesting problems for practice such as image_rotation.
- For each code, there are multiple kernels with different optimizations enabled based on preprocessor directives. 
  Check them at the beginning of the code before compiling.
- e.g.: nvcc -arch=sm_70 -DTILED -o solver.x matmul.cu 
- These codes need cleaning: merge, prefix_sum, histogram, atomic. I will do it soon.
