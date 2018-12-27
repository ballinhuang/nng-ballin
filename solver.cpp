#include <cstring>
#include <omp.h>

#include "solver.hpp"
#include "options.hpp"
#include "linesolver.hpp"
using namespace std;

BoardSolver::BoardSolver(int *data)
{
    clue = data;
}

BoardSolver::~BoardSolver()
{
}

void BoardSolver::do_solve()
{
    BACKTRACKING(&board);
}

void BoardSolver::BACKTRACKING(Board *G)
{
    FP1(G);

    if (G->getStatus() == SOLVED || G->getStatus() == CONFLICT)
        return;
    int p = G->getUnPaintedP();

    Board GZERO;
    copyBoard(G, &GZERO);
    GZERO.setP(p, 0);
    BACKTRACKING(&GZERO);

    if (GZERO.getStatus() == SOLVED)
    {
        copyBoard(&GZERO, G);
        return;
    }

    G->setP(p, 1);
    BACKTRACKING(G);
    return;
}

void BoardSolver::FP1(Board *G)
{
    while (G->hasUnslovedIndex())
    {
        PROPAGATE(G);
        if (G->getStatus() == SOLVED || G->getStatus() == CONFLICT)
            return;

        int p = -1;
        int firstp = -1;
        while (true)
        {
            p = G->getUnPaintedP(p);
            if (p == firstp)
            {
                return; //need backtrack
            }
            if (firstp == -1 || G->getStatus() == PAINTED)
            {
                firstp = p;
                G->status = INCOMPLETE;
            }
            PROBE(G, p);

            if (G->getStatus() == SOLVED || G->getStatus() == CONFLICT)
                return;
        }
    }
}

Board *BoardSolver::getAnser()
{
    return &board;
}

void BoardSolver::PROPAGATE(Board *G)
{
    int unslovedindex;
    int row[BOARDSIZE];
    int rowclue[CLUESIZE];
    LineSolver linesolver;
    while (G->hasUnslovedIndex())
    {
        G->clearlist();

        #pragma omp parallel for private(unslovedindex, linesolver, row, rowclue)
        for (unslovedindex = 1; unslovedindex <= 25; unslovedindex++)
        {
            if (G->getRowhash(unslovedindex) == RowInQueue)
            {
                G->setRowhash(unslovedindex, RowUnSloved);
                G->copytorow(unslovedindex, row);
                copyClue(unslovedindex, rowclue);

                linesolver.setlinesolver(row, rowclue);
                bool result = linesolver.solve();

                if (result)
                {
                    G->paintrow(unslovedindex, row);
                }
                else
                {
                    G->status = CONFLICT;
                    // return;
                }
            }
        }
        if (G->status == CONFLICT) {
            return;
        }

        #pragma omp parallel for private(unslovedindex, linesolver, row, rowclue)
        for (unslovedindex = 26; unslovedindex <= 50; unslovedindex++)
        {
            if (G->getRowhash(unslovedindex) == RowInQueue)
            {
                G->setRowhash(unslovedindex, RowUnSloved);
                G->copytorow(unslovedindex, row);
                copyClue(unslovedindex, rowclue);

                linesolver.setlinesolver(row, rowclue);
                bool result = linesolver.solve();

                if (result)
                {
                    G->paintrow(unslovedindex, row);
                }
                else
                {
                    G->status = CONFLICT;
                    // return;
                }
            }
        }

        if (G->status == CONFLICT) {
            return;
        }
    }
    G->updateStatus();
}

void BoardSolver::PROBE(Board *G, int p)
{
    Board GZERO;
    Board GONE;

    copyBoard(G, &GZERO);
    copyBoard(G, &GONE);

    GZERO.setP(p, 0);
    GONE.setP(p, 1);

    #pragma omp parallel sections
    {
        #pragma omp section
            PROPAGATE(&GZERO);
        #pragma omp section
            PROPAGATE(&GONE);
    }

    if (GZERO.getStatus() == CONFLICT && GONE.getStatus() == CONFLICT)
    {
        G->status = CONFLICT;
    }
    else if (GZERO.getStatus() == CONFLICT)
    {
        G->mergeBoard(&GONE, NULL);
    }
    else if (GONE.getStatus() == CONFLICT)
    {
        G->mergeBoard(&GZERO, NULL);
    }
    else
    {
        G->mergeBoard(&GZERO, &GONE);
    }
}

void BoardSolver::copyClue(int index, int *rowclue)
{
    index -= 1;
    memcpy(rowclue, &clue[index * CLUESIZE], sizeof(int) * CLUESIZE);
}

void BoardSolver::copyBoard(Board *G, Board *copy)
{
    copy->copy(G);
}

bool BoardSolver::checkAns()
{
    int row[BOARDSIZE];
    int rowclue[CLUESIZE];
    for (int index = 1; index <= BOARDSIZE * 2; index++)
    {

        this->board.copytorow(index, row);
        copyClue(index, rowclue);
        int i = BOARDSIZE - 1;
        int j = rowclue[CLUESIZE - 1];
        for (; i >= 0; i--)
        {
            if (row[i] == Unknown)
            {
                return false;
            }

            if (row[i] == UnPainted)
                continue;
            else
            {
                int d = rowclue[j - 1];
                for (; d > 0; d--)
                {
                    if (row[i] == Painted)
                    {
                        i -= 1;
                        continue;
                    }
                    else
                    {
                        if (DEBUGMODE)
                        {
                            cout << "Wrong at index " << index << endl;
                        }
                        return false;
                    }
                }
                if (i > 0)
                {
                    if (row[i] != UnPainted)
                    {
                        if (DEBUGMODE)
                        {
                            cout << "Wrong at index " << index << endl;
                        }
                        return false;
                    }
                }
                i += 1;
                j -= 1;
            }
        }
        if (j != 0)
        {
            return false;
        }
    }
    return true;
}