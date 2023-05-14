#include "torchy.hpp"

int main() {
  auto n = ag::nn::Neuron(2);
  std::vector<std::shared_ptr<ag::Tensor>> inputs = {ag::Tensor::rand({1}), ag::Tensor::rand({1})};
  auto response = n(inputs);
  std::cout << response << std::endl;
  response.get()->graph();
  response.get()->backward();
  for (int i = 0; i < n.weights_.size(); ++i) {
    std::cout << "weight before: " << n.weights_[i].get()->data_[0] << std::endl;
    n.weights_[i].get()->data_[0] = n.weights_[i].get()->data_[0] + (-0.05 * n.weights_[i].get()->grad_[0]);
    std::cout << "weight after: " << n.weights_[i].get()->data_[0] << std::endl;
  }
  std::cout << "bias before: " << n.bias_.get()->data_[0] << std::endl;
  n.bias_.get()->data_[0] = n.bias_.get()->data_[0] + (-0.05 * n.bias_.get()->grad_[0]);
  std::cout << "bias after: " << n.bias_.get()->data_[0] << std::endl;
}
