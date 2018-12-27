#ifndef BOARD_H
#define BOARD_H

#include <queue>
#include <cuda_runtime.h>

//Pixel
#define Unknown -1
#define UnPainted 0
#define Painted 1

//Board status
#define CONFLICT -1
#define PAINTED 0
#define SOLVED 1
#define INCOMPLETE 2

//Row status
#define RowSloved 1
#define RowInQueue 0
#define RowUnSloved -1

class Board
{
public:
  Board();
  ~Board();
  __host__ __device__ void copy(const Board *);
  __host__ __device__ void paintrow(int, int *);
  __host__ __device__ void copytorow(int, int *);
  __host__ __device__ int getUnslovedIndex();
  __host__ __device__ bool hasUnslovedIndex();
  __host__ __device__ void setP(int, int);
  __host__ __device__ int getP(int);
  __host__ __device__ int getUnPaintedP();
  __host__ __device__ int getUnPaintedP(int);
  __host__ __device__ void setRowhash(int, int);
  __host__ __device__ int getRowhash(int);
  __host__ __device__ void checkRowSloved(int);
  __host__ __device__ int getStatus();
  __host__ __device__ void updateStatus();
  __host__ __device__ void mergeBoard(Board *, Board *);
  int status;

  void printBoard(int);

  __host__ __device__ void clearlist();

  int board[625];
  bool dirty = true;
  int rowhash[51];
};

#endif