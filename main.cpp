#include <iostream>
#include <ctime>

#include "options.hpp"
Options *Options::options = 0;

#include "scanner.hpp"
#include "solver.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    if (Options::GetInstance()->readOptions(argc, argv) == 0)
    {
        return 0;
    }

    int **datas;
    datas = new int *[1001];
    for (int i = 0; i < 1001; i++)
        datas[i] = new int[BOARDSIZE * 2 * CLUESIZE];
    for (int i = 0; i < 1001; i++)
    {
        for (int j = 0; j < BOARDSIZE * 2 * CLUESIZE; j++)
            datas[i][j] = 0;
    }

    int Case = readFile(Options::GetInstance()->inputFileName, datas);

    BoardSolver *solver;

    time_t start_time = time(NULL);
    time_t last_time = start_time;
    int anser = 0;
    for (int index = 1; index < Case; index++)
    {
        int data[BOARDSIZE * 2 * CLUESIZE];
        copyData(datas, index, data);
        solver = new BoardSolver(data);
        solver->do_solve();
        if (solver->checkAns())
        {
            cout << "$" << index << "\t"
                 << "time:" << time(NULL) - last_time << "\t\t"
                 << "Correct"
                 << "\t\t"
                 << "Total time:" << time(NULL) - start_time << endl;
        }
        else
        {
            cout << "$" << index << "\t"
                 << "time:" << time(NULL) - last_time << "\t\t"
                 << "Wrong"
                 << "\t\t"
                 << "Total time:" << time(NULL) - start_time << endl;
        }
        solver->getAnser()->printBoard(index);
        if (solver->getAnser()->status == 1)
            anser++;

        last_time = time(NULL);

        delete solver;
    }

    time_t end_time = time(NULL);
    Options::GetInstance()->outputFile.close();
    cout << "anser: " << anser << endl;
    cout << "Total use time: " << end_time - start_time << endl;
}