// Copyright 2023 Ryan Moore

#include <iostream>

#include "T.h"

int main() {
  T t{3, 4, 2};

  t.at({0, 0, 0}) = 1.0f;
  t.at({1, 1, 1}) = 2.0f;

  std::cout << "t[0, 0, 0] = " << t.at({0, 0, 0}) << std::endl;
  std::cout << "t[1, 1, 1] = " << t.at({1, 1, 1}) << std::endl;

  return 0;
}
