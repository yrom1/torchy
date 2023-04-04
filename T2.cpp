// Copyright 2023 Ryan Moore

#include "T2.h"

#include <iostream>

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

  std::vector<size_t> dimensions = {3, 3};
  std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Tensor<int> t(dimensions, values);
  std::cout << t << std::endl;

  std::vector<int> values1 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> values2 = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  Tensor<int> t1(dimensions, values1);
  Tensor<int> t2(dimensions, values2);
  Tensor<int> t3 = t1 + t2;
  std::cout << t3 << std::endl;

  Tensor<int> t4(dimensions, values);
  Tensor<int> t5 = t4 + 3;
  std::cout << t5 << std::endl;
}
