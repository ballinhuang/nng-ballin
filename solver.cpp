#include <cstring>

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

    Board GONE;
    copyBoard(G, &GONE);
    GONE.setP(p, 1);
    BACKTRACKING(&GONE);
    copyBoard(&GONE, G);

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
        unslovedindex = G->getUnslovedIndex();
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

    PROPAGATE(&GZERO);
    PROPAGATE(&GONE);

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
    else if (GZERO.getStatus() == SOLVED && GONE.getStatus() == SOLVED)
    {
        G->mergeBoard(&GZERO, NULL);
    }
    else
    {
        G->mergeBoard(&GZERO, &GONE);
    }

    G->updateStatus();

    if (G->getStatus() != SOLVED && G->getStatus() != CONFLICT)
    {
        if (G->hasUnslovedIndex())
            G->status = PAINTED;
        else
            G->status = INCOMPLETE;
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