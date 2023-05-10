// Copyright 2023 Ryan Moore

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "torchy.hpp"  // NOLINT (build/include_subdir)

TEST(Basic, Shared) {
  ag::t foo = ag::tensor({1}, {42.0});
  ag::t bar = foo->get_shared();

  EXPECT_EQ(foo, bar);
}

TEST(Basic, Add) {
  ag::t a = ag::tensor({1}, {42.0});
  ag::t b = ag::tensor({1}, {42.0});
  auto c = a + b;
  c.get()->backward();

  EXPECT_EQ(c.get()->data_[0], 84.0);
  EXPECT_EQ(a.get()->grad_[0], 1.0);
  EXPECT_EQ(b.get()->grad_[0], 1.0);
}

TEST(Basic, Sum) {
  /*
  >>> import torch
  >>> a = torch.tensor((42.0, 24.0, 12.0), requires_grad=True)
  >>> l = a.sum()
  >>> l.backward()
  >>> l.data
  tensor(78.)
  >>> a.grad
  tensor([1., 1., 1.])
  */
  ag::t a = ag::tensor({3}, {42.0, 24.0, 12.0});
  auto l = a.get()->sum();
  l.get()->backward();

  EXPECT_EQ(l.get()->data_[0], 78.0);
  EXPECT_EQ(l.get()->grad_.size(), 1);
  EXPECT_EQ(a.get()->grad_.size(), 3);
  EXPECT_EQ(a.get()->grad_[0], 1.0);
  EXPECT_EQ(a.get()->grad_[1], 1.0);
  EXPECT_EQ(a.get()->grad_[2], 1.0);
}

TEST(Basic, Minus) {
  /*
  >>> a = torch.tensor((5.0, 4.0, 3.0, 2.0), requires_grad=True)
  >>> b = torch.tensor((2.0, 3.0, 4.0, 5.0), requires_grad=True)
  >>> a
  tensor([5., 4., 3., 2.], requires_grad=True)
  >>> b
  tensor([2., 3., 4., 5.], requires_grad=True)
  >>> c = a - b
  >>> l = c.sum()
  >>> l.backward()
  >>> l
  tensor(0., grad_fn=<SumBackward0>)
  >>> a.grad
  tensor([1., 1., 1., 1.])
  >>> b.grad
  tensor([-1., -1., -1., -1.])
  */
  ag::t a = ag::tensor({4}, {5.0, 4.0, 3.0, 2.0});
  ag::t b = ag::tensor({4}, {2.0, 3.0, 4.0, 5.0});
  ag::t c = a - b;
  auto l = c.get()->sum();
  l.get()->backward();

  EXPECT_EQ(l.get()->data_[0], 0.0);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_EQ(b.get()->grad_.size(), 4);

  EXPECT_EQ(a.get()->grad_[0], 1.0);
  EXPECT_EQ(a.get()->grad_[1], 1.0);
  EXPECT_EQ(a.get()->grad_[2], 1.0);
  EXPECT_EQ(a.get()->grad_[3], 1.0);

  EXPECT_EQ(b.get()->grad_[0], -1.0);
  EXPECT_EQ(b.get()->grad_[1], -1.0);
  EXPECT_EQ(b.get()->grad_[2], -1.0);
  EXPECT_EQ(b.get()->grad_[3], -1.0);
}

TEST(Basic, Multiply) {
  /*
  >>> a = torch.tensor((5.0, 4.0, 3.0, 2.0), requires_grad=True)
  >>> b = torch.tensor((2.0, 3.0, 4.0, 5.0), requires_grad=True)
  >>> c = a * b
  >>> l = c.sum()
  >>> l
  tensor(44., grad_fn=<SumBackward0>)
  >>> l.backward()
  >>> l
  tensor(44., grad_fn=<SumBackward0>)
  >>> a.grad
  tensor([2., 3., 4., 5.])
  >>> b.grad
  tensor([5., 4., 3., 2.])
  */
  ag::t a = ag::tensor({4}, {5.0, 4.0, 3.0, 2.0});
  ag::t b = ag::tensor({4}, {2.0, 3.0, 4.0, 5.0});
  ag::t c = a * b;
  auto l = c.get()->sum();
  l.get()->backward();

  EXPECT_EQ(l.get()->data_[0], 44.0);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_EQ(b.get()->grad_.size(), 4);

  EXPECT_EQ(a.get()->grad_[0], 2.0);
  EXPECT_EQ(a.get()->grad_[1], 3.0);
  EXPECT_EQ(a.get()->grad_[2], 4.0);
  EXPECT_EQ(a.get()->grad_[3], 5.0);

  EXPECT_EQ(b.get()->grad_[0], 5.0);
  EXPECT_EQ(b.get()->grad_[1], 4.0);
  EXPECT_EQ(b.get()->grad_[2], 3.0);
  EXPECT_EQ(b.get()->grad_[3], 2.0);
}

TEST(Basic, Divide) {
  /*
  >>> a = torch.tensor((5.0, 4.0, 3.0, 2.0), requires_grad=True)
  >>> b = torch.tensor((2.0, 3.0, 4.0, 5.0), requires_grad=True)
  >>> a
  tensor([5., 4., 3., 2.], requires_grad=True)
  >>> b
  tensor([2., 3., 4., 5.], requires_grad=True)
  >>> c = a / b
  >>> l = c.sum()
  >>> l.backward()
  >>> l
  tensor(4.9833, grad_fn=<SumBackward0>)
  >>> a.grad
  tensor([0.5000, 0.3333, 0.2500, 0.2000])
  >>> b.grad
  tensor([-1.2500, -0.4444, -0.1875, -0.0800])
  */
  ag::t a = ag::tensor({4}, {5.0, 4.0, 3.0, 2.0});
  ag::t b = ag::tensor({4}, {2.0, 3.0, 4.0, 5.0});
  ag::t c = a / b;
  auto l = c.get()->sum();
  l.get()->backward();

  EXPECT_NEAR(l.get()->data_[0], 4.9833, 0.1);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_EQ(b.get()->grad_.size(), 4);

  EXPECT_NEAR(a.get()->grad_[0], 0.5, 0.1);
  EXPECT_NEAR(a.get()->grad_[1], 0.3333, 0.1);
  EXPECT_NEAR(a.get()->grad_[2], 0.25, 0.1);
  EXPECT_NEAR(a.get()->grad_[3], 0.2, 0.1);

  EXPECT_NEAR(b.get()->grad_[0], -1.25, 0.1);
  EXPECT_NEAR(b.get()->grad_[1], -0.444, 0.1);
  EXPECT_NEAR(b.get()->grad_[2], -0.1875, 0.1);
  EXPECT_NEAR(b.get()->grad_[3], -0.08, 0.1);
}

TEST(Basic, MatMul) {
  /*
  >>> a = torch.tensor(((5.0, 4.0), (3.0, 2.0)), requires_grad=True)
  >>> b = torch.tensor(((2.0, 3.0), (4.0, 5.0)), requires_grad=True)
  >>> c = a.matmul(b)
  >>> l = c.sum()
  >>> l.backward()
  >>> l
  tensor(94., grad_fn=<SumBackward0>)
  >>> a.grad
  tensor([[5., 9.],
          [5., 9.]])
  >>> b.grad
  tensor([[8., 8.],
          [6., 6.]])
  */
  ag::t a = ag::tensor({2, 2}, {5.0, 4.0, 3.0, 2.0});
  ag::t b = ag::tensor({2, 2}, {2.0, 3.0, 4.0, 5.0});
  ag::t c = ag::matmul(a, b);
  auto l = c.get()->sum();
  l.get()->backward();

  EXPECT_NEAR(l.get()->data_[0], 94.0, 0.1);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_EQ(b.get()->grad_.size(), 4);

  EXPECT_NEAR(a.get()->grad_[0], 5.0, 0.1);
  EXPECT_NEAR(a.get()->grad_[1], 9.0, 0.1);
  EXPECT_NEAR(a.get()->grad_[2], 5.0, 0.1);
  EXPECT_NEAR(a.get()->grad_[3], 9.0, 0.1);

  EXPECT_NEAR(b.get()->grad_[0], 8.0, 0.1);
  EXPECT_NEAR(b.get()->grad_[1], 8.0, 0.1);
  EXPECT_NEAR(b.get()->grad_[2], 6.0, 0.1);
  EXPECT_NEAR(b.get()->grad_[3], 6.0, 0.1);
}

TEST(Basic, ChainedMM) {
  /*
  >>> import torch
  >>> a = torch.tensor(((5.0, 4.0), (3.0, 2.0)), requires_grad=True)
  >>> b = torch.tensor(((2.0, 3.0), (4.0, 5.0)), requires_grad=True)
  >>> c = a.matmul(b)
  >>> d = c.matmul(b)
  >>> l = d.sum()
  >>> l
  tensor(686., grad_fn=<SumBackward0>)
  >>> l.backward()
  >>> l
  tensor(686., grad_fn=<SumBackward0>)
  >>> d
  tensor([[192., 253.],
          [104., 137.]], grad_fn=<MmBackward0>)
  >>> c
  tensor([[26., 35.],
          [14., 19.]], grad_fn=<MmBackward0>)
  >>> a.grad
  tensor([[37., 65.],
          [37., 65.]])
  >>> b.grad
  tensor([[ 80., 112.],
          [ 84., 108.]])
  */
  ag::t a = ag::tensor({2, 2}, {5.0, 4.0, 3.0, 2.0});
  ag::t b = ag::tensor({2, 2}, {2.0, 3.0, 4.0, 5.0});
  ag::t c = ag::matmul(a, b);
  ag::t d = ag::matmul(c, b);
  ag::t l = d.get()->sum();
  l.get()->backward();

  EXPECT_NEAR(l.get()->data_[0], 686.0, 0.1);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_EQ(b.get()->grad_.size(), 4);

  EXPECT_NEAR(a.get()->grad_[0], 37.0, 0.1);
  EXPECT_NEAR(a.get()->grad_[1], 65.0, 0.1);
  EXPECT_NEAR(a.get()->grad_[2], 37.0, 0.1);
  EXPECT_NEAR(a.get()->grad_[3], 65.0, 0.1);

  EXPECT_NEAR(b.get()->grad_[0], 80.0, 0.1);
  EXPECT_NEAR(b.get()->grad_[1], 112.0, 0.1);
  EXPECT_NEAR(b.get()->grad_[2], 84.0, 0.1);
  EXPECT_NEAR(b.get()->grad_[3], 108.0, 0.1);
}

TEST(Basic, ScalarComplexAB) {
  /*
  >>> import torch
  >>> a = torch.tensor((2.0), requires_grad=True)
  >>> b = torch.tensor((43.0), requires_grad=True)
  >>> c = (a * b) - ((b / a) + b)
  >>> c.backward()
  >>> c
  tensor(21.5000, grad_fn=<SubBackward0>)
  >>> a.grad
  tensor(53.7500)
  >>> b.grad
  tensor(0.5000)
  */
  ag::t a = ag::tensor({1}, {2.0});
  ag::t b = ag::tensor({1}, {43.0});
  auto l = (a * b) - ((b / a) + b);  // NOTE im calling this l not c lazy
  l.get()->backward();

  EXPECT_NEAR(l.get()->data_[0], 21.5, 0.1);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(a.get()->grad_.size(), 1);
  EXPECT_EQ(b.get()->grad_.size(), 1);

  EXPECT_NEAR(a.get()->grad_[0], 53.7500, 0.1);

  EXPECT_NEAR(b.get()->grad_[0], 0.5000, 0.1);
}

TEST(Basic, ChainedComplexOperations) {
  /*
  >>> import torch
  >>> a = torch.tensor(((2.0, 3.0), (4.0, 5.0)), requires_grad=True)
  >>> b = torch.tensor(((6.0, 7.0), (8.0, 9.0)), requires_grad=True)
  >>> c = torch.tensor(((10.0, 10.0), (10.0, 10.0)), requires_grad=True)
  >>> d = torch.tensor(((11.0, 11.0), (11.0, 11.0)), requires_grad=True)
  >>> e = (a.matmul(b) + c) * d
  >>> f = e.sum()
  >>> f.backward()
  >>> a.grad
  tensor([[143., 187.],
          [143., 187.]])
  >>> b.grad
  tensor([[66., 66.],
          [88., 88.]])
  >>> c.grad
  tensor([[11., 11.],
          [11., 11.]])
  >>> d.grad
  tensor([[46., 51.],
          [74., 83.]])
  >>> f
  tensor(2794., grad_fn=<SumBackward0>)
  >>> f
  tensor(2794., grad_fn=<SumBackward0>)
  */
  ag::t a = ag::tensor({2, 2}, {2.0, 3.0, 4.0, 5.0});
  ag::t b = ag::tensor({2, 2}, {6.0, 7.0, 8.0, 9.0});
  ag::t c = ag::tensor({2, 2}, {10.0, 10.0, 10.0, 10.0});
  ag::t d = ag::tensor({2, 2}, {11.0, 11.0, 11.0, 11.0});
  ag::t e = (ag::matmul(a, b) + c) * d;
  ag::t f = e.get()->sum();
  f.get()->backward();

  EXPECT_NEAR(f.get()->data_[0], 2794.0, 0.1);
  EXPECT_EQ(f.get()->grad_.size(), 1);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_EQ(b.get()->grad_.size(), 4);
  EXPECT_EQ(c.get()->grad_.size(), 4);
  EXPECT_EQ(d.get()->grad_.size(), 4);

  EXPECT_NEAR(a.get()->grad_[0], 143.0, 0.1);
  EXPECT_NEAR(a.get()->grad_[1], 187.0, 0.1);
  EXPECT_NEAR(a.get()->grad_[2], 143.0, 0.1);
  EXPECT_NEAR(a.get()->grad_[3], 187.0, 0.1);

  EXPECT_NEAR(b.get()->grad_[0], 66.0, 0.1);
  EXPECT_NEAR(b.get()->grad_[1], 66.0, 0.1);
  EXPECT_NEAR(b.get()->grad_[2], 88.0, 0.1);
  EXPECT_NEAR(b.get()->grad_[3], 88.0, 0.1);

  EXPECT_NEAR(c.get()->grad_[0], 11.0, 0.1);
  EXPECT_NEAR(c.get()->grad_[1], 11.0, 0.1);
  EXPECT_NEAR(c.get()->grad_[2], 11.0, 0.1);
  EXPECT_NEAR(c.get()->grad_[3], 11.0, 0.1);

  EXPECT_NEAR(d.get()->grad_[0], 46.0, 0.1);
  EXPECT_NEAR(d.get()->grad_[1], 51.0, 0.1);
  EXPECT_NEAR(d.get()->grad_[2], 74.0, 0.1);
  EXPECT_NEAR(d.get()->grad_[3], 83.0, 0.1);
}

TEST(Basic, ReLU) {
  /*
  >>> import torch
  >>> a = torch.tensor(((-1.0, -2.0), (1.0, 2.0)), requires_grad=True)
  >>> b = a.relu()
  >>> b
  tensor([[0., 0.],
          [1., 2.]], grad_fn=<ReluBackward0>)
  >>> l = b.sum()
  >>> l.backward()
  >>> l
  tensor(3., grad_fn=<SumBackward0>)
  >>> a.grad
  tensor([[0., 0.],
          [1., 1.]])
  */
  ag::t a = ag::tensor({2, 2}, {-1.0, -2.0, 1.0, 2.0});
  ag::t b = ag::relu(a);
  auto l = b.get()->sum();
  l.get()->backward();

  EXPECT_EQ(l.get()->data_[0], 3.0);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(b.get()->data_.size(), 4);
  EXPECT_EQ(b.get()->data_[0], 0.0);
  EXPECT_EQ(b.get()->data_[1], 0.0);
  EXPECT_EQ(b.get()->data_[2], 1.0);
  EXPECT_EQ(b.get()->data_[3], 2.0);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_EQ(a.get()->grad_[0], 0.0);
  EXPECT_EQ(a.get()->grad_[1], 0.0);
  EXPECT_EQ(a.get()->grad_[2], 1.0);
  EXPECT_EQ(a.get()->grad_[3], 1.0);
}

template <typename T>
std::vector<T> tensorToVector(const torch::Tensor& tensor) {
  size_t num_elements = tensor.numel();
  std::vector<T> result(num_elements);
  std::memcpy(result.data(), tensor.data_ptr<T>(), num_elements * sizeof(T));
  return result;
}

template <typename T>
std::vector<T> tensorGradToVector(const torch::Tensor& tensor) {
  torch::Tensor grad = tensor.grad();
  if (!grad.defined()) {
    throw std::runtime_error("No gradient available for the input tensor");
  }
  size_t num_elements = grad.numel();
  std::vector<T> result(num_elements);
  std::memcpy(result.data(), grad.data_ptr<T>(), num_elements * sizeof(T));
  return result;
}

TEST(Torch, Neuron) {
  ag::t x = ag::tensor({3, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  ag::t w = ag::tensor({2, 4}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
  ag::t b = ag::tensor({1, 4}, {1.0, 2.0, 3.0, 4.0});
  ag::t result = ag::matmul(x, w) + b;
  auto result_v = result.get()->data_;
  result.get()->backward();
  auto w_g = w.get()->grad_;
  auto b_g = b.get()->grad_;

  std::vector<float> data1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<float> data2 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  at::Tensor x_aten = at::from_blob(data1.data(), {3, 2}, at::kFloat);
  at::Tensor w_aten = at::from_blob(data2.data(), {2, 4}, at::kFloat);
  at::Tensor b_aten = at::tensor({1.0, 2.0, 3.0, 4.0}).unsqueeze(0);
  at::Tensor result_aten = x_aten.matmul(w_aten) + b_aten;
  result_aten.backward();
  std::vector<float> result_aten_v = tensorToVector<float>(result_aten);
  std::vector<float> w_aten_g = tensorGradToVector<float>(w_aten);
  std::vector<float> b_aten_g = tensorGradToVector<float>(b_aten);

  for (auto r_torchy : result_v) {
    for (auto r_torch : result_aten_v) {
      EXPECT_NEAR(r_torchy, r_torch, 0.1);
    }
  }

  for (auto w_torchy : w_g) {
    for (auto w_torch : w_aten_g) {
      EXPECT_NEAR(w_torchy, w_torch, 0.1);
    }
  }

  for (auto b_torchy : b_g) {
    for (auto b_torch : b_aten_g) {
      EXPECT_NEAR(b_torchy, b_torch, 0.1);
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
