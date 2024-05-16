Practice codes for GPU optimization.
- I wrote these codes while studying "Programming Massively Parallel Processors" (4th edition).
- For these codes, check preprocessor directives at the beginning of the code to enable/test particular optimization.
- e.g.: nvcc -arch=sm_70 -DTILED -o solver.x matmul.cu 
- Note: convolution, image_rotation, matrix_mult, streams are cleaned up.
- Need to cleanup merge, prefix_sum, histogram, atomic.
