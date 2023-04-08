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

  // Perform some slices on the tensor t
  at::Tensor row0 = t.slice(0, 0, 1); // First row
  at::Tensor row1 = t.slice(0, 1, 2); // Second row
  at::Tensor col0 = t.slice(1, 0, 1); // First column
  at::Tensor col1 = t.slice(1, 1, 2); // Second column

  // Print the sliced tensors
  std::cout << "First row: " << row0 << std::endl;
  std::cout << "Second row: " << row1 << std::endl;
  std::cout << "First column: " << col0 << std::endl;
  std::cout << "Second column: " << col1 << std::endl;

  return 0;
}
