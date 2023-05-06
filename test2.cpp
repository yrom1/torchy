// Copyright 2023 Ryan Moore

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "torchy.h"  // NOLINT (build/include_subdir)

TEST(Basic, Shared) {
    ag::T foo = ag::tensor(42);
    ag::T bar = foo->get_shared();

    EXPECT_EQ(foo, bar);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
