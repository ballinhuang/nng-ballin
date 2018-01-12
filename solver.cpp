#include <cstring>

#include "solver.hpp"
#include "options.hpp"
#include "linesolver.hpp"
using namespace std;

BoardSolver::BoardSolver(int *data)
{
    clue = data;
    board = new Board();
}

BoardSolver::~BoardSolver()
{
    delete board;
}

void BoardSolver::do_solve()
{
    BACKTRACKING(board);
}

void BoardSolver::BACKTRACKING(Board *G)
{
    FP1(G);
    if (G->getStatus() == SOLVED || G->getStatus() == CONFLICT)
        return;
    int p = G->getUnPaintedP();

    Board *GZERO = new Board();
    copyBoard(G, GZERO);
    GZERO->setP(p, 0);
    BACKTRACKING(GZERO);

    if (GZERO->getStatus() == SOLVED)
    {
        copyBoard(GZERO, G);
        delete GZERO;
        return;
    }
    delete GZERO;

    Board *GONE = new Board();
    copyBoard(G, GONE);
    GONE->setP(p, 1);
    BACKTRACKING(GONE);
    copyBoard(GONE, G);
    delete GONE;

    return;
}

void BoardSolver::FP1(Board *G)
{
    while (G->hasUnslovedIndex())
    {
        G->status = INCOMPLETE;
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
            if (firstp == -1)
            {
                firstp = p;
            }
            PROBE(G, p);
            if (G->getStatus() == SOLVED || G->getStatus() == CONFLICT)
                return;
            if (G->getStatus() == PAINTED)
                break;
        }
    }
}

Board *BoardSolver::getAnser()
{
    return board;
}

void BoardSolver::PROPAGATE(Board *G)
{
    int unslovedindex;
    int *row = new int[BOARDSIZE];
    int *rowclue = new int[CLUESIZE];
    LineSolver *linesolver = new LineSolver();
    while (G->hasUnslovedIndex())
    {
        unslovedindex = G->getUnslovedIndex();
        G->copytorow(unslovedindex, row);
        copyClue(unslovedindex, rowclue);

        linesolver->setlinesolver(row, rowclue);
        bool result = linesolver->solve();

        if (result)
        {
            G->paintrow(unslovedindex, row);
        }
        else
        {
            G->status = CONFLICT;
        }
    }
    G->updateStatus();
    delete[] row;
    delete[] rowclue;
    delete linesolver;
}

void BoardSolver::PROBE(Board *G, int p)
{
    Board *GZERO = new Board();
    Board *GONE = new Board();

    copyBoard(G, GZERO);
    copyBoard(G, GONE);

    GZERO->setP(p, 0);
    GONE->setP(p, 1);

    PROPAGATE(GZERO);
    PROPAGATE(GONE);

    if (GZERO->getStatus() == CONFLICT && GONE->getStatus() == CONFLICT)
    {
        G->status = CONFLICT;
    }
    else if (GZERO->getStatus() == CONFLICT)
    {
        G->mergeBoard(GONE, NULL);
    }
    else if (GONE->getStatus() == CONFLICT)
    {
        G->mergeBoard(GZERO, NULL);
    }
    else if (GZERO->getStatus() == SOLVED && GONE->getStatus() == SOLVED)
    {
        G->mergeBoard(GZERO, NULL);
    }
    else
    {
        G->mergeBoard(GZERO, GONE);
    }

    G->updateStatus();

    if (G->getStatus() != SOLVED && G->getStatus() != CONFLICT)
    {
        if (G->hasUnslovedIndex())
            G->status = PAINTED;
        else
            G->status = INCOMPLETE;
    }

    delete GZERO;
    delete GONE;
}

void BoardSolver::copyClue(int index, int *rowclue)
{
    index -= 1;
    memcpy(rowclue, &clue[index * CLUESIZE], sizeof(int) * CLUESIZE);
}

void BoardSolver::copyBoard(Board *G, Board *copy)
{
    *copy = *G;
}

bool BoardSolver::checkAns()
{
    int *row = new int[BOARDSIZE];
    int *rowclue = new int[CLUESIZE];
    for (int index = 1; index <= 50; index++)
    {

        this->board->copytorow(index, row);
        copyClue(index, rowclue);
        int i = BOARDSIZE - 1;
        int j = rowclue[CLUESIZE - 1];
        for (; i >= 0; i--)
        {
            if (row[i] == Unknown)
            {
                delete[] row;
                delete[] rowclue;
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
                        delete[] row;
                        delete[] rowclue;
                        return false;
                    }
                }
                i += 1;
                j -= 1;
            }
        }
        if (j != 0)
        {
            delete[] row;
            delete[] rowclue;
            return false;
        }
    }
    delete[] row;
    delete[] rowclue;
    return true;
}