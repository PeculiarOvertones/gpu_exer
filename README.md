Practice codes for GPU optimization.
- I wrote these codes while studying "Programming Massively Parallel Processors" (4th edition).
- For these codes, check preprocessor directives at the beginning of the code to enable/test particular optimization.
- compile as nvcc -arch=sm_<architecture> -D<compiler_directive> solver.x <program>.cu
- Note: convolution, image_rotation, matrix_mult, streams are cleaned up.
- Need to cleanup merge, prefix_sum, histogram, atomic.
