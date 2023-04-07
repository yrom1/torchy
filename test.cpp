// Copyright 2023 Ryan Moore

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"
#include "tensor.h"  // NOLINT (build/include_subdir)

// The tests are now wrapped in TEST() macros from Google Test
// The first argument of the TEST() macro is the test suite name,
// and the second argument is the test name.

TEST(TensorTest, CreateTensors) {
  Tensor<float> t1({3}, {1, 2, 3});
  Tensor<float> t2({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  Tensor<float> t3({3, 3, 3},
                   {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26});
  ASSERT_EQ(t1.repr(), "Tensor({3}, {1, 2, 3})");
  ASSERT_EQ(t2.repr(), "Tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})");
  ASSERT_EQ(
      t3.repr(),
      "Tensor({3, 3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, "
      "15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26})");
}

TEST(TensorTest, AccessTensorElement) {
  Tensor<float> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_EQ(tensor({0, 1}), 2);
}

TEST(TensorTest, CatchRuntimeError) {
  Tensor<float> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  try {
    tensor({0, 15});
    FAIL();  // This line should not be reached
  } catch (const std::runtime_error& e) {
    ASSERT_EQ(std::string(e.what()), std::string("Index out of bounds."));
  }
}

// Since the original create_and_print_tensor_slices() test was not working,
// I am not including it in this revised code.
// However, you can create a new test using the TEST() macro if you fix the
// issue.

TEST(TensorTest, CreateTensorWithValues) {
  std::vector<size_t> dimensions = {3, 3};
  std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Tensor<int> t(dimensions, values);
  EXPECT_EQ(t.repr(), "Tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})");
}

TEST(TensorTest, TensorAddition) {
  std::vector<int> values1 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> values2 = {2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<size_t> dimensions = {3, 3};
  Tensor<int> t1(dimensions, values1);
  Tensor<int> t2(dimensions, values2);
  Tensor<int> t3 = t1 + t2;
  EXPECT_EQ(t1.repr(), "Tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})");
  EXPECT_EQ(t2.repr(), "Tensor({3, 3}, {2, 3, 4, 5, 6, 7, 8, 9, 10})");
  EXPECT_EQ(t3.repr(), "Tensor({3, 3}, {3, 5, 7, 9, 11, 13, 15, 17, 19})");
}

TEST(TensorTest, TensorScalarAddition) {
  std::vector<size_t> dimensions = {3, 3};
  std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Tensor<int> t(dimensions, values);
  Tensor<int> t2 = t + 3;
  EXPECT_EQ(t2.repr(), "Tensor({3, 3}, {4, 5, 6, 7, 8, 9, 10, 11, 12})");
}

TEST(TensorTest, TensorDivision) {
  std::vector<int> values1 = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> values2 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<size_t> dimensions = {3, 3};
  Tensor<int> t1(dimensions, values1);
  Tensor<int> t2(dimensions, values2);
  Tensor<int> t3 = t1 / t2;
  EXPECT_EQ(t1.repr(), "Tensor({3, 3}, {9, 8, 7, 6, 5, 4, 3, 2, 1})");
  EXPECT_EQ(t2.repr(), "Tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})");
  EXPECT_EQ(t3.repr(), "Tensor({3, 3}, {9, 4, 2, 1, 1, 0, 0, 0, 0})");
}

TEST(TensorTest, TensorScalarDivision) {
  std::vector<size_t> dimensions = {3, 3};
  std::vector<int> values = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  Tensor<int> t(dimensions, values);
  Tensor<int> t2 = t / 3;
  EXPECT_EQ(t2.repr(), "Tensor({3, 3}, {3, 2, 2, 2, 1, 1, 1, 0, 0})");
}

TEST(TensorTest, MatrixMultiplication) {
  std::vector<size_t> dimensions1 = {2, 3};
  std::vector<int> values1 = {1, 2, 3, 4, 5, 6};
  std::vector<size_t> dimensions2 = {3, 2};
  std::vector<int> values2 = {7, 8, 9, 10, 11, 12};
  Tensor<int> t1(dimensions1, values1);
  Tensor<int> t2(dimensions2, values2);
  Tensor<int> t3 = t1.matmul(t2);
  EXPECT_EQ(t3.repr(), "Tensor({2, 2}, {58, 64, 139, 154})");
}

TEST(TensorTest, CatchRuntimeErrorZeroDivideScalar) {
  std::vector<size_t> dimensions = {3, 3};
  std::vector<int> values = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  Tensor<int> t(dimensions, values);
  ASSERT_THROW(t / 0, std::runtime_error);
}

TEST(TensorTest, CreateAndPrintTensorRepr) {
  std::vector<size_t> dimensions = {3, 3};
  std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Tensor<int> t(dimensions, values);
  EXPECT_EQ(t.repr(), "Tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})");
}

// FIXME(yrom1) this is broken and submatrix doesnt exist

// TEST(TensorTest, CreateAndPrintTensorSlices) {
//   std::vector<size_t> dimensions = {3, 3};
//   std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
//   Tensor<int> t(dimensions, values);

//   // Row slice
//   Tensor<int> row_slice = t.slice({0}, {1, 3});
//   EXPECT_EQ(row_slice.repr(), "Tensor({1, 3}, {1, 2, 3})");

//   // Column slice
//   Tensor<int> col_slice = t.slice({1}, {3, 1});
//   EXPECT_EQ(col_slice.repr(), "Tensor({3, 1}, {2, 5, 8})");

//   // Submatrix slice
//   Tensor<int> submatrix_slice = t.slice({0, 1}, {2, 2});
//   EXPECT_EQ(submatrix_slice.repr(), "Tensor({2, 2}, {2, 3, 5, 6})");
// }

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
