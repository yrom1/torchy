#include <iostream>
#include <torch/torch.h>

int main() {
  // Create 2x2 tensors with requires_grad=true
  torch::Tensor x = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::requires_grad(true));
  torch::Tensor y = torch::tensor({{2.0, 3.0}, {4.0, 5.0}}, torch::requires_grad(true));

  // Compute z = x * y + y^2
  torch::Tensor z = x * y + torch::pow(y, 2);

  // Compute the loss as the sum of all elements in z
  torch::Tensor loss = z.sum();

  // Compute gradients
  loss.backward();

  // Print gradients
  std::cout << "x.grad: " << x.grad() << std::endl;
  std::cout << "y.grad: " << y.grad() << std::endl;

  return 0;
}
