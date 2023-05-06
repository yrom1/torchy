// Copyright 2023 Ryan Moore

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "torchy.hpp"  // NOLINT (build/include_subdir)

TEST(Basic, Shared) {
  ag::T foo = ag::tensor({1}, {42.0});
  ag::T bar = foo->get_shared();

  EXPECT_EQ(foo, bar);
}

TEST(Basic, Add) {
  ag::T a = ag::tensor({1}, {42.0});
  ag::T b = ag::tensor({1}, {42.0});
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
  ag::T a = ag::tensor({1}, {42.0, 24.0, 12.0});
  auto l = a.get()->sum();
  l.get()->backward();

  EXPECT_EQ(l.get()->data_[0], 78.0);
  EXPECT_EQ(l.get()->grad_.size(), 1);
  EXPECT_EQ(a.get()->grad_.size(), 3);
  EXPECT_EQ(a.get()->grad_[0], 1.0);
  EXPECT_EQ(a.get()->grad_[1], 1.0);
  EXPECT_EQ(a.get()->grad_[2], 1.0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
