#ifndef SOLVER_H
#define SOLVER_H

#include "board.hpp"

class BoardSolver
{
public:
  BoardSolver(int *);
  ~BoardSolver();
  void do_solve();
  void BACKTRACKING(Board *);
  void FP1(Board *);
  Board *getAnser();
  bool checkAns();

private:
  int *clue;
  Board *board;

  void PROPAGATE(Board *);
  void PROBE(Board *, int);

  void copyClue(int, int *);
  void copyBoard(Board *, Board *);
};

#endif