#include "torchy.hpp"

int main() {
  auto n = ag::nn::Neuron(2, 0.05);
  std::vector<std::shared_ptr<ag::Tensor>> inputs = {ag::Tensor::rand({1}), ag::Tensor::rand({1})};
  auto response = n(inputs);
  std::cout << response << std::endl;
  response.get()->graph();
  response.get()->backward();
  n.train();
}
