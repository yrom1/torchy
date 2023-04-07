#include <iostream>
#include <ATen/ATen.h>

int main() {
  std::vector<int> data1 = {1, 2, 3, 4, 5, 6};
  std::vector<int> data2 = {1, 2, 3, 4, 5, 6, 7, 8};

  // create two tensors from the vectors
  at::Tensor x_aten = at::from_blob(data1.data(), {3, 2}, at::kInt);
  at::Tensor w_aten = at::from_blob(data2.data(), {2, 4}, at::kInt);
  at::Tensor b_aten = at::tensor({1, 2, 3, 4}).unsqueeze(0);

  at::Tensor result_aten = x_aten.matmul(w_aten) + b_aten;

  std::cout << "NEURON" << std::endl;
  std::cout << result_aten << std::endl;

  return 0;
}
