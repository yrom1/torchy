#include <iostream>

#include "torchy.h"

int main() {
  auto a = std::make_shared<Tensor<int>>();
  std::shared_ptr<Tensor<int>> shared_a = a->shared_from_this();
  std::cout << shared_a << std::endl;
}
