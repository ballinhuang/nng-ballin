#ifndef LINESOLVER_H
#define LINESOLVER_H

#include "fixcache.hpp"
#include <cuda_runtime.h>

class LineSolver
{
public:
  __host__ __device__ LineSolver();
  __host__ __device__ ~LineSolver();
  __host__ __device__ void setlinesolver(int *, int *);
  __host__ __device__ bool solve();
  int *row, *rowclue;

private:
  __host__ __device__ bool Paint(int, int, int *);
  __host__ __device__ bool Paintp(int, int, int *);
  __host__ __device__ void Paint0(int, int, int *);
  __host__ __device__ void Paint1(int, int, int *);

  __host__ __device__ bool Fix(int, int, int *);
  __host__ __device__ bool Fix0(int, int, int *);
  __host__ __device__ bool Fix1(int, int, int *);

  __host__ __device__ void copyrow(int *, int *);
  __host__ __device__ void mergerow(int *, int *, int *);
  FixCache m_fix1Cache;
  FixCache m_fix0Cache;
};

#endif