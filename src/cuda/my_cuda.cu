#include <algorithm>
#include <cstdio>
#include <my_cuda.h>

// __global__ means this is called from the CPU, and runs on the GPU
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int N) {
  // Calculate global thread ID
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Boundary check
  if (tid < N) c[tid] = a[tid] + b[tid];
}

MyCuda::~MyCuda() {
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

void MyCuda::init(int N_) {
  N = N_;
  bytes = sizeof(int) * N;
  printf("before MyCuda::init %p,%p,%p\n", d_a, d_b, d_c);
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  printf("after MyCuda::init %p,%p,%p\n", d_a, d_b, d_c);
}

void MyCuda::cuda_entry(const int * a, const int * b, int * c, int N) {
  cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

  int NUM_THREADS = 1 << 10;
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;
  vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
  cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);
}