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

bool test_tensor_equality() {
  Tensor<float> t1{3, 4, 2};
  t1[{0, 0, 0}] = 1.0f;
  t1[{1, 1, 1}] = 2.0f;

  Tensor<float> t2{3, 4, 2};
  t2[{0, 0, 0}] = 1.0f;
  t2[{1, 1, 1}] = 2.0f;

  if (!(t1 == t2)) {
    std::cerr << "Failed: Tensor equality check is incorrect" << std::endl;
    return false;
  }

  return true;
}

bool test_tensor_inequality() {
  Tensor<float> t1{3, 4, 2};
  t1[{0, 0, 0}] = 1.0f;
  t1[{1, 1, 1}] = 2.0f;

  Tensor<float> t2{3, 4, 2};
  t2[{0, 0, 0}] = 1.0f;
  t2[{1, 1, 1}] = 3.0f;

  if (!(t1 != t2)) {
    std::cerr << "Failed: Tensor inequality check is incorrect" << std::endl;
    return false;
  }

  return true;
}

bool test_tensor_constructor_with_values() {
  std::vector<size_t> dimensions = {2, 2, 2};
  std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  Tensor<float> tensor(dimensions, values);

  const float epsilon = 1e-6f;
  size_t i = 0;
  for (auto &value : tensor) {
    if (std::abs(value - values[i]) > epsilon) {
      std::cerr << "Failed: Tensor values don't match the expected values in"
                   " constructor with values."
                << std::endl;
      return false;
    }
    i++;
  }

  return true;
}

bool test_copy_constructor_and_assignment() {
  Tensor<float> original({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  // Test copy constructor
  Tensor<float> copy_constructed(original);
  if (copy_constructed == original) {
    std::cout << "Passed: Copy constructor test." << std::endl;
  } else {
    std::cerr << "Failed: Copy constructor test." << std::endl;
    return false;
  }

  // Test copy assignment operator
  Tensor<float> copy_assigned;
  copy_assigned = original;
  if (copy_assigned == original) {
    std::cout << "Passed: Copy assignment test." << std::endl;
  } else {
    std::cerr << "Failed: Copy assignment test." << std::endl;
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

  if (test_tensor_dimensions(t, {3, 4, 2})) {
    std::cout << "Passed: Tensor dimensions are correct." << std::endl;
  } else {
    all_tests_passed = false;
  }

  if (test_tensor_assignment(t)) {
    std::cout << "Passed: Tensor element assignment is correct." << std::endl;
  } else {
    all_tests_passed = false;
  }

  if (test_tensor_default_constructor(empty_tensor)) {
    std::cout << "Passed: Default constructor creates an empty tensor."
              << std::endl;
  } else {
    all_tests_passed = false;
  }

  if (test_tensor_constructor_with_values()) {
    std::cout
        << "Passed: Tensor constructor with dimensions and values is correct."
        << std::endl;
  } else {
    all_tests_passed = false;
  }

  if (test_tensor_equality()) {
    std::cout << "Passed: Tensor equality check is correct." << std::endl;
  } else {
    all_tests_passed = false;
  }

  if (test_tensor_inequality()) {
    std::cout << "Passed: Tensor inequality check is correct." << std::endl;
  } else {
    all_tests_passed = false;
  }

  if (test_copy_constructor_and_assignment()) {
    std::cout << "Passed: Copy constructor and copy assignment tests."
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
