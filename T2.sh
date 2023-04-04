clang-format -i -style=Google T2.cpp T2.h
cpplint --recursive T2.cpp T2.h
g++ -std=c++14 T2.cpp && ./a.out
