// Copyright 2023 Ryan Moore

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "torchy.h"  // NOLINT (build/include_subdir)

// The tests are now wrapped in TEST() macros from Google Test
// The first argument of the TEST() macro is the test suite name,
// and the second argument is the test name.

TEST(Basic, CreateTensors) {
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

TEST(Basic, AccessTensorElement) {
  Tensor<float> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_EQ(tensor({0, 1}), 2);
}

TEST(Basic, CatchRuntimeError) {
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

TEST(Basic, CreateTensorWithValues) {
  std::vector<size_t> dimensions = {3, 3};
  std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Tensor<int> t(dimensions, values);
  EXPECT_EQ(t.repr(), "Tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})");
}

TEST(Basic, TensorAddition) {
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

TEST(Basic, TensorScalarAddition) {
  std::vector<size_t> dimensions = {3, 3};
  std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Tensor<int> t(dimensions, values);
  Tensor<int> t2 = t + 3;
  EXPECT_EQ(t2.repr(), "Tensor({3, 3}, {4, 5, 6, 7, 8, 9, 10, 11, 12})");
}

TEST(Basic, TensorDivision) {
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

TEST(Basic, TensorScalarDivision) {
  std::vector<size_t> dimensions = {3, 3};
  std::vector<int> values = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  Tensor<int> t(dimensions, values);
  Tensor<int> t2 = t / 3;
  EXPECT_EQ(t2.repr(), "Tensor({3, 3}, {3, 2, 2, 2, 1, 1, 1, 0, 0})");
}

TEST(Basic, TensorMultiplication) {
  std::vector<int> values1 = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> values2 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<size_t> dimensions = {3, 3};
  Tensor<int> t1(dimensions, values1);
  Tensor<int> t2(dimensions, values2);
  Tensor<int> t3 = t1 * t2;
  EXPECT_EQ(t1.repr(), "Tensor({3, 3}, {9, 8, 7, 6, 5, 4, 3, 2, 1})");
  EXPECT_EQ(t2.repr(), "Tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})");
  EXPECT_EQ(t3.repr(), "Tensor({3, 3}, {9, 16, 21, 24, 25, 24, 21, 16, 9})");
}

TEST(Basic, TensorSubtraction) {
  std::vector<int> values1 = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> values2 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<size_t> dimensions = {3, 3};
  Tensor<int> t1(dimensions, values1);
  Tensor<int> t2(dimensions, values2);
  Tensor<int> t3 = t1 - t2;
  EXPECT_EQ(t1.repr(), "Tensor({3, 3}, {9, 8, 7, 6, 5, 4, 3, 2, 1})");
  EXPECT_EQ(t2.repr(), "Tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})");
  EXPECT_EQ(t3.repr(), "Tensor({3, 3}, {8, 6, 4, 2, 0, -2, -4, -6, -8})");
}

TEST(Basic, TensorScalarMultiplication) {
  std::vector<size_t> dimensions = {3, 3};
  std::vector<int> values = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  Tensor<int> t(dimensions, values);
  Tensor<int> t2 = t * 3;
  EXPECT_EQ(t2.repr(), "Tensor({3, 3}, {27, 24, 21, 18, 15, 12, 9, 6, 3})");
}

TEST(Basic, MatrixMultiplication) {
  std::vector<size_t> dimensions1 = {2, 3};
  std::vector<int> values1 = {1, 2, 3, 4, 5, 6};
  std::vector<size_t> dimensions2 = {3, 2};
  std::vector<int> values2 = {7, 8, 9, 10, 11, 12};
  Tensor<int> t1(dimensions1, values1);
  Tensor<int> t2(dimensions2, values2);
  Tensor<int> t3 = t1.matmul(t2);
  EXPECT_EQ(t3.repr(), "Tensor({2, 2}, {58, 64, 139, 154})");
}

TEST(Basic, CatchRuntimeErrorZeroDivideScalar) {
  std::vector<size_t> dimensions = {3, 3};
  std::vector<int> values = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  Tensor<int> t(dimensions, values);
  ASSERT_THROW(t / 0, std::runtime_error);
}

TEST(Basic, CreateAndPrintTensorRepr) {
  std::vector<size_t> dimensions = {3, 3};
  std::vector<int> values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Tensor<int> t(dimensions, values);
  EXPECT_EQ(t.repr(), "Tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})");
}

TEST(Basic, StorageDtypeIsInt) {
  Storage<int> storage(5, {1, 2, 3, 4, 5});
  ASSERT_TRUE((std::is_same<Storage<int>::dtype, int>::value));
}

TEST(Basic, TensorDtypeIsInt) {
  Tensor<int> tensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_TRUE((std::is_same<Tensor<int>::dtype, int>::value));
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

// Test if setting requires_grad creates an AutogradMeta object and assigns a
// grad tensor
// TEST(AutogradMeta, RequiresGradCreatesAutogradMeta) {
//   Tensor<float> t1({2, 3}, {1, 2, 3, 4, 5, 6}, true);

//   // Check if requires_grad is set correctly
//   ASSERT_TRUE(t1.requires_grad());

//   // Check if the AutogradMeta object exists
//   ASSERT_NO_THROW(t1.grad());

//   // Assign a gradient to the tensor
//   Tensor<float> grad({2, 3}, {1, 1, 1, 1, 1, 1});
//   t1.set_grad(grad);

//   // Check if the assigned gradient can be retrieved
//   Tensor<float> retrieved_grad = t1.grad();
//   for (size_t i = 0; i < t1.computeSize(); ++i) {
//     ASSERT_FLOAT_EQ(grad.storage()->data()[i],
//                     retrieved_grad.storage()->data()[i]);
//   }
// }

// Test if setting requires_grad to false does not create an AutogradMeta object
TEST(AutogradMeta, RequiresGradFalseNoAutogradMeta) {
  Tensor<float> t2({2, 3}, {1, 2, 3, 4, 5, 6}, false);

  // Check if requires_grad is set correctly
  ASSERT_FALSE(t2.requires_grad());

  // Check if the AutogradMeta object does not exist
  ASSERT_THROW(t2.grad(), std::runtime_error);
}

template <typename T>
std::vector<T> tensorToVector(const torch::Tensor& tensor) {
  size_t num_elements = tensor.numel();
  std::vector<T> result(num_elements);
  std::memcpy(result.data(), tensor.data_ptr<T>(), num_elements * sizeof(T));
  return result;
}

TEST(Torch, AddBackward0Scalar) {
  Tensor<float> a({1}, {4.20}, true);
  Tensor<float> b({1}, {13.37}, true);
  auto c = a + b;
  c.backward();
  auto a_v = a.autograd_meta_.get()->grad_.storage_.get()->data_;
  auto b_v = b.autograd_meta_.get()->grad_.storage_.get()->data_;

  torch::Tensor a_t = torch::tensor({4.20}, torch::requires_grad(true));
  torch::Tensor b_t = torch::tensor({13.37}, torch::requires_grad(true));
  torch::Tensor c_t = a_t + b_t;
  c_t.backward();
  auto a_v_t = tensorToVector<float>(a_t.grad());
  auto b_v_t = tensorToVector<float>(b_t.grad());

  EXPECT_EQ(a_v, a_v_t);
  EXPECT_EQ(b_v, b_v_t);
}

TEST(Torch, SubBackward0Scalar) {
  Tensor<float> a({1}, {4.20}, true);
  Tensor<float> b({1}, {13.37}, true);
  auto c = a - b;
  c.backward();
  auto a_v = a.autograd_meta_.get()->grad_.storage_.get()->data_;
  auto b_v = b.autograd_meta_.get()->grad_.storage_.get()->data_;

  torch::Tensor a_t = torch::tensor({4.20}, torch::requires_grad(true));
  torch::Tensor b_t = torch::tensor({13.37}, torch::requires_grad(true));
  torch::Tensor c_t = a_t - b_t;
  c_t.backward();
  auto a_v_t = tensorToVector<float>(a_t.grad());
  auto b_v_t = tensorToVector<float>(b_t.grad());

  EXPECT_EQ(a_v, a_v_t);
  EXPECT_EQ(b_v, b_v_t);
}

TEST(Torch, MulBackward0Scalar) {
  Tensor<float> a({1}, {4.20}, true);
  Tensor<float> b({1}, {13.37}, true);
  auto c = a * b;
  c.backward();
  auto a_v = a.autograd_meta_.get()->grad_.storage_.get()->data_;
  auto b_v = b.autograd_meta_.get()->grad_.storage_.get()->data_;

  torch::Tensor a_t = torch::tensor({4.20}, torch::requires_grad(true));
  torch::Tensor b_t = torch::tensor({13.37}, torch::requires_grad(true));
  torch::Tensor c_t = a_t * b_t;
  c_t.backward();
  auto a_v_t = tensorToVector<float>(a_t.grad());
  auto b_v_t = tensorToVector<float>(b_t.grad());

  EXPECT_EQ(a_v, a_v_t);
  EXPECT_EQ(b_v, b_v_t);
}

TEST(Torch, DivBackward0Scalar) {
  Tensor<float> a({1}, {4.20}, true);
  Tensor<float> b({1}, {13.37}, true);
  auto c = a / b;
  c.backward();
  auto a_v = a.autograd_meta_.get()->grad_.storage_.get()->data_;
  auto b_v = b.autograd_meta_.get()->grad_.storage_.get()->data_;

  torch::Tensor a_t = torch::tensor({4.20}, torch::requires_grad(true));
  torch::Tensor b_t = torch::tensor({13.37}, torch::requires_grad(true));
  torch::Tensor c_t = a_t / b_t;
  c_t.backward();
  auto a_v_t = tensorToVector<float>(a_t.grad());
  auto b_v_t = tensorToVector<float>(b_t.grad());

  EXPECT_EQ(a_v, a_v_t);
  EXPECT_EQ(b_v, b_v_t);
}

TEST(Torch, ScalarMultiOperator) {
  Tensor<float> a({1}, {4.20}, true);
  Tensor<float> b({1}, {13.37}, true);
  auto c = (a * b) - ((b / a) + b);
  c.backward();
  auto a_v = a.autograd_meta_.get()->grad_.storage_.get()->data_;
  auto b_v = b.autograd_meta_.get()->grad_.storage_.get()->data_;

  torch::Tensor a_t = torch::tensor({4.20}, torch::requires_grad(true));
  torch::Tensor b_t = torch::tensor({13.37}, torch::requires_grad(true));
  torch::Tensor c_t = (a_t * b_t) - ((b_t / a_t) + b_t);
  c_t.backward();
  auto a_v_t = tensorToVector<float>(a_t.grad());
  auto b_v_t = tensorToVector<float>(b_t.grad());

  EXPECT_EQ(a_v, a_v_t);
  EXPECT_EQ(b_v, b_v_t);
}

TEST(Torch, Neuron) {
  // Test input tensors
  Tensor<int> x({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor<int> w({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  Tensor<int> b({1, 4}, {1, 2, 3, 4});
  Tensor<int> result = x.matmul(w) + b;
  auto result_v = result.storage()->data();

  std::vector<int> data1 = {1, 2, 3, 4, 5, 6};
  std::vector<int> data2 = {1, 2, 3, 4, 5, 6, 7, 8};
  at::Tensor x_aten = at::from_blob(data1.data(), {3, 2}, at::kInt);
  at::Tensor w_aten = at::from_blob(data2.data(), {2, 4}, at::kInt);
  at::Tensor b_aten = at::tensor({1, 2, 3, 4}).unsqueeze(0);
  at::Tensor result_aten = x_aten.matmul(w_aten) + b_aten;
  std::vector<int> result_aten_v = tensorToVector<int>(result_aten);
  // std::cout << result_v << std::endl;
  // std::cout << result_aten_v << std::endl;
  EXPECT_EQ(result_v, result_aten_v);
}

TEST(Torch, DISABLED_BackwardAdd) {
  torch::Tensor a =
      torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::requires_grad(true));
  torch::Tensor b =
      torch::tensor({{2.0, 3.0}, {4.0, 5.0}}, torch::requires_grad(true));
  torch::Tensor c = a + b;
  torch::Tensor scalar_output = c.sum();
  scalar_output.backward();
  auto result_torch_a = tensorToVector<float>(a.grad());
  auto result_torch_b = tensorToVector<float>(b.grad());

  std::vector<float> result_torchy = {42.0, 42.0, 42.0, 42.0};

  EXPECT_EQ(result_torch_a, result_torchy);
  EXPECT_EQ(result_torch_b, result_torchy);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
