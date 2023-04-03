#include <iostream>
#include "T4.h"

int main() {
  Tensor<float> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::cout << tensor({0, 1}) << std::endl;
  Tensor<float> slice1 = tensor.slice(0, 0, 2);  // First two rows
  Tensor<float> slice2 = tensor.slice(1, 1, 3);  // Last two columns
  std::cout << slice1 << std::endl;
  std::cout << slice2 << std::endl;
}
