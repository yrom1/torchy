// Copyright 2023 Ryan Moore
// test.cpp

#include <cmath>
#include <iostream>

#include "Tensor.h"

bool test_tensor_dimensions(const Tensor<float> &tensor,
                            const std::vector<size_t> &expected_dims) {
  const std::vector<size_t> &actual_dims = tensor.size();
  if (actual_dims != expected_dims) {
    std::cerr << "Failed: Tensor dimensions don't match the expected dimensions"
              << std::endl;
    return false;
  }
  return true;
}

bool test_tensor_assignment(const Tensor<float> &tensor) {
  const float epsilon = 1e-6f;
  if (std::abs(tensor[{0, 0, 0}] - 1.0f) > epsilon ||
      std::abs(tensor[{1, 1, 1}] - 2.0f) > epsilon) {
    std::cerr << "Failed: Tensor element assignment is incorrect" << std::endl;
    return false;
  }
  return true;
}

bool test_tensor_default_constructor(const Tensor<float> &tensor) {
  if (tensor.numel() != 0) {
    std::cerr << "Failed: Default constructor does not create an empty tensor"
              << std::endl;
    return false;
  }
  return true;
}

int main() {
  Tensor<float> t{3, 4, 2};
  Tensor<float> empty_tensor;

  t[{0, 0, 0}] = 1.0f;
  t[{1, 1, 1}] = 2.0f;

  bool all_tests_passed = true;

  // Test the dimensions of the tensor
  if (test_tensor_dimensions(t, {3, 4, 2})) {
    std::cout << "Passed: Tensor dimensions are correct." << std::endl;
  } else {
    all_tests_passed = false;
  }

  // Test tensor assignment
  if (test_tensor_assignment(t)) {
    std::cout << "Passed: Tensor element assignment is correct." << std::endl;
  } else {
    all_tests_passed = false;
  }

  // Test the default constructor
  if (test_tensor_default_constructor(empty_tensor)) {
    std::cout << "Passed: Default constructor creates an empty tensor."
              << std::endl;
  } else {
    all_tests_passed = false;
  }

  if (all_tests_passed) {
    std::cout << "All tests passed." << std::endl;
    return 0;
  } else {
    std::cerr << "Some tests failed." << std::endl;
    return 1;
  }
}