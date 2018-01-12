#include <fstream>
#include <sstream>
#include <cstring>

#include "scanner.hpp"
#include "options.hpp"
using namespace std;

int readFile(string filename, int **datas)
{
    fstream fs;
    fs.open(filename.c_str());
    string line;
    stringstream ss;
    int Case = 1;
    while (getline(fs, line))
    {
        // $%d
        for (int i = 0; i < BOARDSIZE * 2; i++)
        {
            int index = 0;
            getline(fs, line);
            ss << line;
            while (ss >> datas[Case][i * CLUESIZE + index])
            {
                index += 1;
            }
            datas[Case][i * CLUESIZE + CLUESIZE - 1] = index;
            ss.clear();
        }
        Case += 1;
    }
    return Case;
}

void copyData(int **datas, int index, int *data)
{
    memcpy(data, datas[index], sizeof(int) * BOARDSIZE * 2 * CLUESIZE);
}