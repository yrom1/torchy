// c++ -std=c++20 or.cpp && ./a.out

#include "torchy.hpp"
#include <cmath>

int main() {
  std::vector<float> a = {0.0f, 0.0f, 1.0f, 1.0f};
  std::vector<float> b = {0.0f, 1.0f, 0.0f, 1.0f};
  std::vector<float> ans = {0.0f, 1.0f, 1.0f, 1.0f}; // OR function

  auto n = ag::nn::Neuron(2, 0.025); // Lower the learning rate
  for (int epoch = 0; epoch < 301; ++epoch) {
    auto sum = ag::tensor({1}, {0.0f});
    for (int i = 0; i < a.size(); ++i) {
      std::vector<std::shared_ptr<ag::Tensor>> inputs = {
        ag::tensor({1}, {a[i]}), ag::tensor({1}, {b[i]})
      };

      auto response = n(inputs);
      sum = sum + ((ag::tensor({1}, {ans[i]}) - response) * (ag::tensor({1}, {ans[i]}) - response));
    }
    auto loss = sum / ag::tensor({1}, {static_cast<float>(a.size())});

    loss.get()->zero_grad();
    loss.get()->backward();
    n.train();

    // after each epoch, evaluate and print the outputs
    if (epoch % 50 == 0) {
      std::cout << "EPOCH: " << epoch << " LOSS: " << loss.get()->data_[0] << std::endl;

      for (int i = 0; i < a.size(); ++i) {
        std::vector<std::shared_ptr<ag::Tensor>> inputs = {
          ag::tensor({1}, {a[i]}), ag::tensor({1}, {b[i]})
        };
        auto response = n(inputs);
        if (round(response.get()->data_[0]) == ans[i]) {
          std::cout << a[i] << " OR " << b[i] << " = " << "🔥" << response.get()->data_[0] << std::endl;
        } else {
          std::cout << a[i] << " OR " << b[i] << " = " << "  " << response.get()->data_[0] << std::endl;
        }
      }
    }
  }
}
