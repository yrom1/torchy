#include <iostream>
#include <torch/torch.h>

int main() {
  // 2x2 tensors
  torch::Tensor a = torch::tensor({{1.0, 2.0}, {3.0, 4.0}}, torch::requires_grad(true));
  torch::Tensor b = torch::tensor({{2.0, 3.0}, {4.0, 5.0}}, torch::requires_grad(true));
  torch::Tensor c = a + b;

  // c.backward();
  /* BAD NO GOOD!
  ERORR The backward function in PyTorch requires a scalar output to compute gradients with respect to its input tensors.
  // : terminating with uncaught exception of type c10::Error: grad can be implicitly created only for scalar outputs
  */

  torch::Tensor scalar_output = c.sum();

  scalar_output.backward();

  std::cout << "a.grad: " << a.grad() << std::endl;
  std::cout << "b.grad: " << b.grad() << std::endl;

  /** JUST LIKE MICROGRAD !
  a.grad:  1  1
  1  1
  [ CPUFloatType{2,2} ]
  b.grad:  1  1
  1  1
  [ CPUFloatType{2,2} ]
  */

  return 0;
}
