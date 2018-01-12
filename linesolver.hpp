#ifndef LINESOLVER_H
#define LINESOLVER_H

#include "fixcache.hpp"

class LineSolver
{
public:
  LineSolver();
  ~LineSolver();
  void setlinesolver(int *, int *);
  bool solve();
  int *row, *rowclue;

private:
  bool Paint(int, int, int *);
  bool Paintp(int, int, int *);
  void Paint0(int, int, int *);
  void Paint1(int, int, int *);

  bool Fix(int, int, int *);
  bool Fix0(int, int, int *);
  bool Fix1(int, int, int *);

  void copyrow(int *, int *);
  void mergerow(int *, int *, int *);
  FixCache *m_fix1Cache;
  FixCache *m_fix0Cache;
};

#endif