#include <iostream>

#include "torchy.h"

int main() {
  Tensor<float> t({1}, {42}, true);
  std::cout << t << std::endl;
}
