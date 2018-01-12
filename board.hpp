#ifndef BOARD_H
#define BOARD_H

#include <map>
#include <queue>

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
  void operator=(const Board);
  void paintrow(int, int *);
  void copytorow(int, int *);
  int getUnslovedIndex();
  bool hasUnslovedIndex();
  void setP(int, int);
  int getP(int);
  int getUnPaintedP();
  int getUnPaintedP(int);
  void setRowhash(int, int);
  int getRowhash(int);
  void checkRowSloved(int);
  int getStatus();
  void updateStatus();
  void mergeBoard(Board *, Board *);
  int status;

  void printBoard(int);

private:
  int *board;
  std::queue<int> list;
  std::map<int, int> rowhash;
};

#endif