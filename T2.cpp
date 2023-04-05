// Copyright 2023 Ryan Moore

#include "T2.h"

#include <iostream>
#include <stdexcept>
#include <vector>

void create_and_print_tensors() {
  Tensor<float> t1({3}, {1, 2, 3});
  Tensor<float> t2({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor<float> t3({3, 3, 3},
                   {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26});
  std::cout << t1 << std::endl;
  std::cout << t2 << std::endl;
  std::cout << t3 << std::endl;
}

void access_tensor_element() {
  Tensor<float> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::cout << tensor({0, 1}) << std::endl;
}

void catch_runtime_error() {
  Tensor<float> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  try {
    std::cout << tensor({0, 15}) << std::endl;
  } catch (const std::runtime_error &e) {
    std::cout << "Error: " << e.what() << std::endl;
  }
}

void create_and_print_tensor_slices() {
  Tensor<float> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor<float> t2({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor<float> slice1 = tensor.slice(0, 0, 2);  // First two rows
  Tensor<float> slice2 = tensor.slice(1, 1, 3);  // Last two columns

  std::cout << "tensor" << std::endl;
  std::cout << tensor << std::endl;
  std::cout << "slice1" << std::endl;
  std::cout << slice1 << std::endl;
  std::cout << "slice2" << std::endl;
  std::cout << slice2 << std::endl;
}

void create_and_print_tensor_with_values() {
  std::vector<size_t> dimensions = {3, 3};
  std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Tensor<int> t(dimensions, values);
  std::cout << t << std::endl;
}

void create_and_print_tensor_addition() {
  std::vector<int> values1 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> values2 = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<size_t> dimensions = {3, 3};
  Tensor<int> t1(dimensions, values1);
  Tensor<int> t2(dimensions, values2);
  Tensor<int> t3 = t1 + t2;
  std::cout << t3 << std::endl;
}

void create_and_print_tensor_scalar_addition() {
  std::vector<size_t> dimensions = {3, 3};
  std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Tensor<int> t(dimensions, values);
  Tensor<int> t2 = t + 3;
  std::cout << t2 << std::endl;
}

void create_and_print_matrix_multiplication() {
  std::vector<size_t> dimensions1 = {2, 3};
  std::vector<size_t> dimensions2 = {3, 2};
  std::vector<int> values1 = {1, 2, 3, 4, 5, 6};
  std::vector<int> values2 = {7, 8, 9, 10, 11, 12};
  Tensor<int> t1(dimensions1, values1);
  Tensor<int> t2(dimensions2, values2);
  Tensor<int> t3 = t1.matmul(t2);
  std::cout << t1 << std::endl;
  std::cout << t2 << std::endl;
  std::cout << t3 << std::endl;
}

int main() {
  // In this case, std::function<void()> is a type that represents a function
  // with no parameters and no return value. It's a "callable object" that can
  // hold any function that matches this signature.

  // If a function took one argument of type int and returned a float, for
  // example, its signature would be float(int).
  std::vector<std::function<void()>> tests = {
      create_and_print_tensors,
      access_tensor_element,
      catch_runtime_error,
      create_and_print_tensor_slices,
      create_and_print_tensor_with_values,
      create_and_print_tensor_addition,
      create_and_print_tensor_scalar_addition,
      create_and_print_matrix_multiplication};

  for (const auto &test : tests) {
    test();
    std::cout << "-----------------" << std::endl;
  }
}
