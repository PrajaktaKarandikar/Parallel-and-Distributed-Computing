# CUDA Programming
Programs implementing Matrix Multiplication using 2 different approaches:-

1. Unoptimized Approach:
   Thread block congifuration = (16, 16),
   Grid configuration = (64, 64)
   
2. Optimized (Shared Memory) Approach:
   Thread block configuration = (32, 32),
   Grid configuration = (32, 32)
   
Approach 2 executes faster compared to approach 1 and therefore highlights the advantage of using shared memories especially in cases where the size of input is very high.
