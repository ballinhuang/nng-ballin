#ifndef OPTIONS_H
#define OPTIONS_H

#include <iostream>
#include <fstream>

#define CLUESIZE Options::GetInstance()->cluesize
#define BOARDSIZE Options::GetInstance()->boardsize
#define DEBUGMODE Options::GetInstance()->debugmode
#define FILEGMODE (Options::GetInstance()->outputFileName == "" ? false : true)

class Options
{
public:
  std::string inputFileName;
  std::string outputFileName;
  bool debugmode;
  bool showoptions;
  int boardsize;
  int cluesize;
  std::ofstream outputFile;

  static Options *GetInstance();
  int readOptions(int argc, char **argv);
  void help();

private:
  static Options *options;
  Options()
  {
    inputFileName = "input.txt";
    outputFileName = "";
    debugmode = false;
    showoptions = false;
    boardsize = 25;
  }
  void show();
};

#endif