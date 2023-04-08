#include <iostream>
#include <vector>
#include <ATen/ATen.h>

int main() {
  // Initialize a 2x3 float tensor with a vector
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  at::Tensor t = at::from_blob(data.data(), {2, 3}, at::kFloat);

  // Print the tensor's shape and values
  std::cout << "Tensor shape: " << t.sizes() << std::endl;
  std::cout << "Tensor values: " << t << std::endl;

  return 0;
}
