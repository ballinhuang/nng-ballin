#include <cstring>

#include "solver.hpp"
#include "options.hpp"
#include "linesolver.hpp"
#include <cuda_runtime.h>
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

__global__ void kernel(Board *G, int *clues)
{
    int index_row = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int index_col = blockIdx.x * blockDim.x + threadIdx.x + 26;

    int row[25];
    int rowclue[14];
    LineSolver linesolver;
    bool result;

    while (G->hasUnslovedIndex() && G->status != CONFLICT)
    {
        __syncthreads();
        G->clearlist();
        __syncthreads();

        G->setRowhash(index_row, RowUnSloved);
        G->copytorow(index_row, row);
        memcpy(rowclue, &clues[(index_row - 1) * 14], sizeof(int) * 14);

        linesolver.setlinesolver(row, rowclue);
        result = linesolver.solve();

        if (result)
        {
            G->paintrow(index_row, row);
        }
        else
        {
            G->status = CONFLICT;
        }

        __syncthreads();

        G->setRowhash(index_col, RowUnSloved);
        G->copytorow(index_col, row);
        memcpy(rowclue, &clues[(index_col - 1) * 14], sizeof(int) * 14);

        linesolver.setlinesolver(row, rowclue);
        result = linesolver.solve();

        if (result)
        {
            G->paintrow(index_col, row);
        }
        else
        {
            G->status = CONFLICT;
        }

        __syncthreads();
    }
}

void BoardSolver::PROPAGATE(Board *G)
{
    printf("%d\n", clue[0]);
    Board *G_gpu;
    int *clues;
    cudaMalloc(&G_gpu, sizeof(Board));
    cudaMemcpy(G_gpu, G, sizeof(Board), cudaMemcpyHostToDevice);
    cudaMalloc(&clues, sizeof(int) * 50 * 14);
    cudaMemcpy(clues, clue, sizeof(int) * 50 * 14, cudaMemcpyHostToDevice);

    kernel<<<1, 25>>>(G_gpu, clues);
    cudaMemcpy(G, G_gpu, sizeof(Board), cudaMemcpyDeviceToHost);

    cudaFree(G_gpu);
    cudaFree(clues);

    if (G->status == CONFLICT)
    {
        return;
    }
    G->updateStatus();
    G->printBoard(0);
    exit(0);
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
