#include <iostream>
#include <ATen/ATen.h>

int main() {
  // create two tensors
  at::Tensor a = at::ones({2, 2}, at::kInt);
  at::Tensor b = at::ones({2, 2}, at::kInt);

  // element-wise comparison
  at::Tensor elementwise_comparison = a.eq(b);
  std::cout << "Element-wise comparison:\n" << elementwise_comparison << std::endl;

  // whole tensor comparison
  bool whole_tensor_comparison = at::equal(a, b);
  std::cout << "Whole tensor comparison: " << std::boolalpha << whole_tensor_comparison << std::endl;

  std::cout << "HUH " << (a == b) << std::endl;

  return 0;
}
