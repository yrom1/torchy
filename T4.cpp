#include <iostream>
#include "T4.h"

int main() {
  Tensor<float> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::cout << tensor({0, 1}) << std::endl;
}
