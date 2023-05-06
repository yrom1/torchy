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

  EXPECT_EQ(c.get()->data_[0], 84.0);
  EXPECT_EQ(a.get()->grad_[0], 1.0);
  EXPECT_EQ(b.get()->grad_[0], 1.0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
