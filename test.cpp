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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
