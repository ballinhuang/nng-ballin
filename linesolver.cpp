#include <cstring>
#include <iostream>
#include <cstdlib>

#include "linesolver.hpp"
#include "options.hpp"
#include "board.hpp"
#include "fixcache.hpp"

using namespace std;

LineSolver::LineSolver()
{
    this->m_fix1Cache = new FixCache();
    this->m_fix0Cache = new FixCache();
    this->m_fix1Cache->init();
    this->m_fix0Cache->init();
}

LineSolver::~LineSolver()
{
    delete m_fix1Cache;
    delete m_fix0Cache;
}

void LineSolver::setlinesolver(int *row, int *rowclue)
{
    this->row = row;
    this->rowclue = rowclue;
    this->m_fix1Cache->init();
    this->m_fix0Cache->init();
}

bool LineSolver::solve()
{
    int i = BOARDSIZE;
    int j = rowclue[CLUESIZE - 1];
    bool result = Paint(i, j, row);
    return result;
}

bool LineSolver::Paint(int i, int j, int *row)
{
    if (i == 0)
        return true;
    return Paintp(i, j, row);
}

bool LineSolver::Paintp(int i, int j, int *row)
{
    bool f0 = Fix0(i, j, row);
    bool f1 = Fix1(i, j, row);

    if (f0 && !f1)
    {
        Paint0(i, j, row);
    }
    else if (!f0 && f1)
    {
        Paint1(i, j, row);
    }
    else if (f0 && f1)
    {
        int p0row[BOARDSIZE], p1row[BOARDSIZE];
        this->copyrow(p0row, row);
        this->copyrow(p1row, row);
        Paint0(i, j, p0row);
        Paint1(i, j, p1row);
        this->mergerow(row, p0row, p1row);
    }
    else
    {
        return false;
    }
    return true;
}

void LineSolver::Paint0(int i, int j, int *row)
{
    row[i - 1] = UnPainted;
    Paint(i - 1, j, row);
}

void LineSolver::Paint1(int i, int j, int *row)
{
    int d = rowclue[j - 1];
    for (int k = i - 1; k >= i - d; k--)
        row[k] = Painted;
    i -= d;
    if (i > 0)
    {
        i -= 1;
        row[i] = UnPainted;
    }
    j -= 1;
    Paint(i, j, row);
}

bool LineSolver::Fix0(int i, int j, int *row)
{
    if (row[i - 1] != Painted)
    {
        bool result = Fix(i - 1, j, row);
        return result;
    }
    else
    {
        return false;
    }
}

bool LineSolver::Fix1(int i, int j, int *row)
{
    if (j == 0)
    {
        return false;
    }
    int d = rowclue[j - 1];
    if (i - d < 0)
        return false;
    for (int k = i - 1; k >= i - d; k--)
    {
        if (row[k] == UnPainted)
        {
            return false;
        }
    }
    i -= d;
    if (i > 0)
    {
        i -= 1;
        if (row[i] == Painted)
        {
            return false;
        }
    }
    j -= 1;
    bool result = Fix(i, j, row);
    return result;
}

bool LineSolver::Fix(int i, int j, int *row)
{
    if (i == 0 && j == 0)
        return true;
    if (i == 0 && j >= 1)
        return false;
    else
    {
        bool fix1Result;
        bool fix0Result;

        if (this->m_fix1Cache->hasResult(i, j))
        {
            fix1Result = this->m_fix1Cache->fixResult(i, j);
        }
        else
        {
            fix1Result = Fix1(i, j, row);
            this->m_fix1Cache->setFixResult(i, j, fix1Result);
        }

        if (this->m_fix0Cache->hasResult(i, j))
        {
            fix0Result = this->m_fix0Cache->fixResult(i, j);
        }
        else
        {
            fix0Result = Fix0(i, j, row);
            this->m_fix0Cache->setFixResult(i, j, fix0Result);
        }
        return fix1Result || fix0Result;
    }
}

void LineSolver::copyrow(int *copy, int *row)
{
    for (int i = 0; i < BOARDSIZE; i++)
        copy[i] = row[i];
}

void LineSolver::mergerow(int *row, int *p0row, int *p1row)
{
    for (int i = 0; i < BOARDSIZE; i++)
    {
        if (row[i] != p0row[i] && p0row[i] == p1row[i])
        {
            row[i] = p0row[i];
        }
    }
}