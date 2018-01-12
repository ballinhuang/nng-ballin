#include <cstring>

#include "options.hpp"

using namespace std;

Options *Options::GetInstance()
{
    if (options == NULL)
    {
        options = new Options();
    }
    return options;
}

int Options::readOptions(int argc, char **argv)
{
    string arg;
    for (int i = 1; i < argc; ++i)
    {
        arg = std::string(argv[i]);

        if (arg == "--debug")
        {
            this->debugmode = true;
            continue;
        }

        if (i + 1 < argc && arg == "--input")
        {
            this->inputFileName = std::string(argv[i + 1]);
            i++;
            continue;
        }

        if (i + 1 < argc && arg == "--output")
        {
            this->outputFileName = std::string(argv[i + 1]);
            this->outputFile.open(this->outputFileName.c_str());
            i++;
            continue;
        }

        if (i + 1 < argc && arg == "--size")
        {
            this->boardsize = atoi(argv[i + 1]);
            i++;
            continue;
        }

        if (arg == "--show")
        {
            this->showoptions = true;
            continue;
        }

        if (arg == "--help")
        {
            this->help();
            return 0;
        }

        cout << "Error option:" << arg << endl;
        return 0;
    }

    int padding = this->boardsize % 2;
    this->cluesize = this->boardsize / 2 + padding + 1;

    if (this->showoptions == true)
    {
        this->show();
    }

    return 1;
}

void Options::help()
{
    cout << "  --input [FileName]\n";
    cout << "    Set the input file's name. Default value: input.txt\n";

    cout << "  --output [FileName]\n";
    cout << "    Set the output file's name. Default value: STD Output\n";

    cout << "  --size [Board size]\n";
    cout << "    Set the Board's size. Default value: 25\n";

    cout << "  --debug\n";
    cout << "    Open the debug mode.\n";

    cout << "  --help\n";
    cout << "    Show the options.\n";

    cout << "  --show\n";
    cout << "    Show the settings.\n";
}

void Options::show()
{
    cout << "[ Options ]" << endl
         << "inputFileName: " << this->inputFileName << endl
         << "outputFileName: " << (this->outputFileName == "" ? "STD Output" : "this->outputFileName") << endl
         << "boardsize: " << this->boardsize << endl
         << "cluesize: " << this->cluesize << endl
         << "debug: " << (this->debugmode == true ? "On" : "Off") << endl
         << "---------------------------------------------" << endl;
}