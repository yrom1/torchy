// Copyright 2023 Ryan Moore

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "tensor.h"  // NOLINT (build/include_subdir)

void create_and_print_tensors() {
  std::cout << "--- create tensors" << std::endl;
  Tensor<float> t1({3}, {1, 2, 3});
  Tensor<float> t2({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor<float> t3({3, 3, 3},
                   {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26});
  assert(t1.repr() == "Tensor({3}, {1, 2, 3})");
  assert(t2.repr() == "Tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})");
  assert(t3.repr() ==
         "Tensor({3, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, "
         "15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26})");
}

void access_tensor_element() {
  std::cout << "--- element access" << std::endl;
  Tensor<float> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  assert(tensor({0, 1}) == 2);
}

void catch_runtime_error() {
  std::cout << "--- runtime error" << std::endl;
  Tensor<float> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  try {
    tensor({0, 15});
    assert(false);  // This line should not be reached
  } catch (const std::runtime_error &e) {
    assert(std::string(e.what()) == std::string("Index out of bounds."));
  }
}

void create_and_print_tensor_slices() {
  /* BUG repr of views doesnt work!
  --- slices
  [[1, 2, 3],
  [4, 5, 6]]
  [[2, 3],
  [5, 6],
  [8, 9]]
  Tensor({2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})
  Tensor({3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9})
  Assertion failed: (slice1.repr() == "Tensor({2, 3}, {1, 2, 3, 4, 5, 6})"),
  function create_and_print_tensor_slices, file test.cpp, line 51. make: ***
  [run] Abort trap: 6
  */
  std::cout << "--- slices" << std::endl;
  Tensor<float> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor<float> t2({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor<float> slice1 = tensor.slice(0, 0, 2);  // First two rows
  Tensor<float> slice2 = tensor.slice(1, 1, 3);  // Last two columns
  // std::cout << slice1.repr() << std::endl;
  // std::cout << slice2.repr() << std::endl;
  // assert(slice1.repr() == "Tensor({2, 3}, {1, 2, 3, 4, 5, 6})");
  // assert(slice2.repr() == "Tensor({3, 2}, {2, 3, 5, 6, 8, 9})");
}

void create_and_print_tensor_with_values() {
  std::cout << "--- create with values" << std::endl;
  std::vector<size_t> dimensions = {3, 3};
  std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Tensor<int> t(dimensions, values);
  assert(t.repr() == "Tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})");
}

void create_and_print_tensor_addition() {
  std::cout << "--- tensor addition" << std::endl;
  std::vector<int> values1 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> values2 = {2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<size_t> dimensions = {3, 3};
  Tensor<int> t1(dimensions, values1);
  Tensor<int> t2(dimensions, values2);
  Tensor<int> t3 = t1 + t2;
  assert(t1.repr() == "Tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})");
  assert(t2.repr() == "Tensor({3, 3}, {2, 3, 4, 5, 6, 7, 8, 9, 10})");
  assert(t3.repr() == "Tensor({3, 3}, {3, 5, 7, 9, 11, 13, 15, 17, 19})");
}

void create_and_print_tensor_division() {
  std::cout << "--- tensor division" << std::endl;
  std::vector<int> values1 = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> values2 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<size_t> dimensions = {3, 3};
  Tensor<int> t1(dimensions, values1);
  Tensor<int> t2(dimensions, values2);
  Tensor<int> t3 = t1 / t2;
  assert(t1.repr() == "Tensor({3, 3}, {9, 8, 7, 6, 5, 4, 3, 2, 1})");
  assert(t2.repr() == "Tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})");
  assert(t3.repr() == "Tensor({3, 3}, {9, 4, 2, 1, 1, 0, 0, 0, 0})");
}

void create_and_print_tensor_scalar_addition() {
  std::cout << "--- scalar addition" << std::endl;
  std::vector<size_t> dimensions = {3, 3};
  std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Tensor<int> t(dimensions, values);
  Tensor<int> t2 = t + 3;
  assert(t2.repr() == "Tensor({3, 3}, {4, 5, 6, 7, 8, 9, 10, 11, 12})");
}

void create_and_print_matrix_multiplication() {
  std::cout << "--- matmul" << std::endl;
  std::vector<size_t> dimensions1 = {2, 3};
  std::vector<size_t> dimensions2 = {3, 2};
  std::vector<int> values1 = {1, 2, 3, 4, 5, 6};
  std::vector<int> values2 = {7, 8, 9, 10, 11, 12};
  Tensor<int> t1(dimensions1, values1);
  Tensor<int> t2(dimensions2, values2);
  Tensor<int> t3 = t1.matmul(t2);
  assert(t1.repr() == "Tensor({2, 3}, {1, 2, 3, 4, 5, 6})");
  assert(t2.repr() == "Tensor({3, 2}, {7, 8, 9, 10, 11, 12})");
  assert(t3.repr() == "Tensor({2, 2}, {58, 64, 139, 154})");
}

void catch_runtime_error_zero_divide_scalar() {
  std::cout << "--- runtime error" << std::endl;
  Tensor<float> tensor({3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8});
  try {
    std::cout << tensor / 0 << std::endl;
  } catch (const std::runtime_error &e) {
    std::cout << "Error: " << e.what() << std::endl;
  }
  try {
    std::cout << tensor / tensor << std::endl;
  } catch (const std::runtime_error &e) {
    std::cout << "Error: " << e.what() << std::endl;
  }
}

void create_and_print_tensor_repr() {
  std::cout << "--- create tensors" << std::endl;
  Tensor<float> t1({3}, {1, 2, 3});
  Tensor<float> t2({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor<float> t3({3, 3, 3},
                   {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26});
  assert(t1.repr() == "Tensor({3}, {1, 2, 3})");
  assert(t2.repr() == "Tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})");
  assert(t3.repr() ==
         "Tensor({3, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, "
         "15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26})");
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
      create_and_print_tensor_division,
      create_and_print_matrix_multiplication,
      catch_runtime_error_zero_divide_scalar,
      create_and_print_tensor_repr};

  for (const auto &test : tests) {
    test();
    std::cout << "-----------------" << std::endl;
  }
}
