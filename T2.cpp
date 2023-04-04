// Copyright 2023 Ryan Moore

#include <iostream>
#include "T2.h"

int main() {
  Tensor<float> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

  std::cout << "tensor @ (0, 1)" << std::endl;
  std::cout << tensor({0, 1}) << std::endl;

  try {
    std::cout << tensor({0, 15}) << std::endl;
  } catch (const std::runtime_error &e) {
    std::cout << "Error: " << e.what() << std::endl;
  }

  Tensor<float> slice1 = tensor.slice(0, 0, 2);  // First two rows
  Tensor<float> slice2 = tensor.slice(1, 1, 3);  // Last two columns

  std::cout << "tensor" << std::endl;
  std::cout << tensor << std::endl;
  std::cout << "slice1" << std::endl;
  std::cout << slice1 << std::endl;
  std::cout << "slice2" << std::endl;
  std::cout << slice2 << std::endl;
}
