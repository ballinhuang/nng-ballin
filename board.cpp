
#include "board.hpp"
#include "options.hpp"

using namespace std;

Board::Board()
{
    for (int i = 0; i < 25 * 25; i++)
    {
        board[i] = Unknown;
    }
    for (int i = 1; i <= 25 * 2; i++)
    {
        rowhash[i] = RowInQueue;
    }
    status = INCOMPLETE;
}

Board::~Board()
{
}

__host__ __device__ void Board::copy(const Board *b)
{
    for (int i = 0; i < 625; i++)
    {
        board[i] = b->board[i];
    }
    for (int i = 0; i < 51; i++)
    {
        this->rowhash[i] = b->rowhash[i];
    }
    this->status = b->status;
    this->dirty = b->dirty;
}

__host__ __device__ void Board::paintrow(int index, int *row)
{
    if (index <= 25)
    {
        for (int i = 0; i < 25; i++)
        {
            if (this->board[(index - 1) * 25 + i] != row[i])
            {
                this->board[(index - 1) * 25 + i] = row[i];
                int colindex = i + 1 + 25; // ex:1 -> 26
                setRowhash(colindex, RowInQueue);
            }
        }
    }
    else
    {
        index = index - 25; // ex: 26->1
        for (int i = 0; i < 25; i++)
        {
            if (this->board[(index - 1) + 25 * i] != row[i])
            {
                this->board[(index - 1) + 25 * i] = row[i];
                int rowindex = i + 1; // ex: i=0 -> row 1
                setRowhash(rowindex, RowInQueue);
            }
        }
    }
}

__host__ __device__ void Board::copytorow(int index, int *row)
{
    if (index <= 25)
    {
        for (int i = 0; i < 25; i++)
        {
            row[i] = this->board[(index - 1) * 25 + i];
        }
    }
    else
    {
        index = index - 25;
        for (int i = 0; i < 25; i++)
        {
            row[i] = this->board[(index - 1) + 25 * i];
        }
    }
}

__host__ __device__ bool Board::hasUnslovedIndex()
{
    if (this->dirty)
        return true;
    return false;
}

__host__ __device__ void Board::clearlist()
{
    this->dirty = false;
}

//useless
__host__ __device__ void Board::checkRowSloved(int index)
{
    bool sloved = true;
    if (index <= 25)
    {
        for (int i = 0; i < 25; i++)
        {
            if (this->board[(index - 1) * 25 + i] == Unknown)
            {
                sloved = false;
                break;
            }
        }
    }
    else
    {
        int colindex = index - 25;
        for (int i = 0; i < 25; i++)
        {
            if (this->board[(colindex - 1) + 25 * i] == Unknown)
            {
                sloved = false;
                break;
            }
        }
    }
    if (sloved)
        setRowhash(index, RowSloved);
}

__host__ __device__ void Board::setRowhash(int index, int status)
{
    if (status == RowInQueue)
    {
        if (getRowhash(index) == RowUnSloved)
        {
            this->dirty = true;
            rowhash[index] = status;
            return;
        }
    }
    else
    {
        rowhash[index] = status;
    }
}

__host__ __device__ int Board::getRowhash(int index)
{
    return this->rowhash[index];
}

__host__ __device__ void Board::setP(int p, int status)
{
    this->board[p] = status;
    int rowindex = p / 25 + 1;
    int colindex = p % 25 + 25 + 1;
    setRowhash(rowindex, RowInQueue);
    setRowhash(colindex, RowInQueue);
}

__host__ __device__ int Board::getP(int p)
{
    return board[p];
}

__host__ __device__ int Board::getUnPaintedP()
{
    for (int i = 0; i < 25 * 25; i++)
    {
        if (this->board[i] == Unknown)
            return i;
    }
    return -1;
}

__host__ __device__ int Board::getUnPaintedP(int p)
{
    for (int i = p + 1; i < 25 * 25; i++)
    {
        if (this->board[i] == Unknown)
            return i;
    }
    for (int i = 0; i <= p; i++)
    {
        if (this->board[i] == Unknown)
            return i;
    }
    return -1;
}

__host__ __device__ void Board::updateStatus()
{
    if (this->getUnPaintedP() == -1)
    {
        this->status = SOLVED;
    }
}

__host__ __device__ int Board::getStatus()
{
    return this->status;
}

__host__ __device__ void Board::mergeBoard(Board *a, Board *b)
{
    if (b == NULL)
    {
        for (int i = 0; i < 25 * 25; i++)
        {
            this->board[i] = a->board[i];
        }
        if (a->status == SOLVED)
            this->status = SOLVED;
        else
            this->status = PAINTED;
    }
    else
    {
        bool ispainted = false;
        for (int i = 0; i < 25 * 25; i++)
        {
            if (a->getP(i) == b->getP(i))
            {
                if (this->board[i] != a->getP(i))
                {
                    ispainted = true;
                    this->board[i] = a->board[i];
                }
            }
        }
        if (ispainted)
            this->status = PAINTED;
    }
}

void Board::printBoard(int index)
{
    if (FILEGMODE)
    {
        Options::GetInstance()->outputFile << "$" << index << endl;
        for (int i = 0; i < 25; i++)
        {
            for (int j = 0; j < 25; j++)
            {
                if (board[i * 25 + j] == Unknown)
                    Options::GetInstance()->outputFile << "U"
                                                       << "\t";
                else
                    Options::GetInstance()->outputFile << board[i * 25 + j] << "\t";
            }
            Options::GetInstance()->outputFile << endl;
        }
    }
    else
    {
        cout << "$" << index << endl;
        for (int i = 0; i < 25; i++)
        {
            for (int j = 0; j < 25; j++)
            {
                if (board[i * 25 + j] == Unknown)
                    cout << "U"
                         << "\t";
                else
                    cout << board[i * 25 + j] << "\t";
            }
            cout << endl;
        }
    }
}