#include <cstring>

#include "board.hpp"
#include "options.hpp"

using namespace std;

Board::Board()
{
    for (int i = 0; i < BOARDSIZE * BOARDSIZE; i++)
    {
        board[i] = Unknown;
    }
    for (int i = 1; i <= BOARDSIZE * 2; i++)
    {
        list.push(i);
        rowhash[i] = RowInQueue;
    }
    status = INCOMPLETE;
}

Board::~Board()
{
}

void Board::copy(const Board *b)
{
    for (int i = 0; i < BOARDSIZE * BOARDSIZE; i++)
    {
        board[i] = b->board[i];
    }
    this->list = b->list;
    this->rowhash = b->rowhash;
    this->status = b->status;
}

void Board::paintrow(int index, int *row)
{
    if (index <= BOARDSIZE)
    {
        for (int i = 0; i < BOARDSIZE; i++)
        {
            if (this->board[(index - 1) * BOARDSIZE + i] != row[i])
            {
                this->board[(index - 1) * BOARDSIZE + i] = row[i];
                int colindex = i + 1 + BOARDSIZE; // ex:1 -> 26
                setRowhash(colindex, RowInQueue);
            }
        }
    }
    else
    {
        index = index - BOARDSIZE; // ex: 26->1
        for (int i = 0; i < BOARDSIZE; i++)
        {
            if (this->board[(index - 1) + BOARDSIZE * i] != row[i])
            {
                this->board[(index - 1) + BOARDSIZE * i] = row[i];
                int rowindex = i + 1; // ex: i=0 -> row 1
                setRowhash(rowindex, RowInQueue);
            }
        }
    }
}

void Board::copytorow(int index, int *row)
{
    if (index <= BOARDSIZE)
    {
        for (int i = 0; i < BOARDSIZE; i++)
        {
            row[i] = this->board[(index - 1) * BOARDSIZE + i];
        }
    }
    else
    {
        index = index - BOARDSIZE;
        for (int i = 0; i < BOARDSIZE; i++)
        {
            row[i] = this->board[(index - 1) + BOARDSIZE * i];
        }
    }
}

bool Board::hasUnslovedIndex()
{
    if (!this->list.empty())
        return true;
    return false;
}

int Board::getUnslovedIndex()
{
    int result = -1;
    if (hasUnslovedIndex())
    {
        result = this->list.front();
        this->list.pop();
        rowhash[result] = RowUnSloved;
    }
    return result;
}

void Board::clearlist()
{
    std::queue<int> empty;
    std::swap(this->list, empty);
}

//useless
void Board::checkRowSloved(int index)
{
    bool sloved = true;
    if (index <= BOARDSIZE)
    {
        for (int i = 0; i < BOARDSIZE; i++)
        {
            if (this->board[(index - 1) * BOARDSIZE + i] == Unknown)
            {
                sloved = false;
                break;
            }
        }
    }
    else
    {
        int colindex = index - BOARDSIZE;
        for (int i = 0; i < BOARDSIZE; i++)
        {
            if (this->board[(colindex - 1) + BOARDSIZE * i] == Unknown)
            {
                sloved = false;
                break;
            }
        }
    }
    if (sloved)
        setRowhash(index, RowSloved);
}

void Board::setRowhash(int index, int status)
{
    if (status == RowInQueue)
    {
        if (getRowhash(index) == RowUnSloved)
        {
            list.push(index);
            rowhash[index] = status;
            return;
        }
    }
    else
    {
        rowhash[index] = status;
    }
}

int Board::getRowhash(int index)
{
    return this->rowhash[index];
}

void Board::setP(int p, int status)
{
    this->board[p] = status;
    int rowindex = p / BOARDSIZE + 1;
    int colindex = p % BOARDSIZE + BOARDSIZE + 1;
    setRowhash(rowindex, RowInQueue);
    setRowhash(colindex, RowInQueue);
}

int Board::getP(int p)
{
    return board[p];
}

int Board::getUnPaintedP()
{
    for (int i = 0; i < BOARDSIZE * BOARDSIZE; i++)
    {
        if (this->board[i] == Unknown)
            return i;
    }
    return -1;
}

int Board::getUnPaintedP(int p)
{
    for (int i = p + 1; i < BOARDSIZE * BOARDSIZE; i++)
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

void Board::updateStatus()
{
    if (this->getUnPaintedP() == -1)
    {
        this->status = SOLVED;
    }
}

int Board::getStatus()
{
    return this->status;
}

void Board::mergeBoard(Board *a, Board *b)
{
    if (b == NULL)
    {
        for (int i = 0; i < BOARDSIZE * BOARDSIZE; i++)
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
        for (int i = 0; i < BOARDSIZE * BOARDSIZE; i++)
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
        for (int i = 0; i < BOARDSIZE; i++)
        {
            for (int j = 0; j < BOARDSIZE; j++)
            {
                if (board[i * BOARDSIZE + j] == Unknown)
                    Options::GetInstance()->outputFile << "U"
                                                       << "\t";
                else
                    Options::GetInstance()->outputFile << board[i * BOARDSIZE + j] << "\t";
            }
            Options::GetInstance()->outputFile << endl;
        }
    }
    else
    {
        cout << "$" << index << endl;
        for (int i = 0; i < BOARDSIZE; i++)
        {
            for (int j = 0; j < BOARDSIZE; j++)
            {
                if (board[i * BOARDSIZE + j] == Unknown)
                    cout << "U"
                         << "\t";
                else
                    cout << board[i * BOARDSIZE + j] << "\t";
            }
            cout << endl;
        }
    }
}