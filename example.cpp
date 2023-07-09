#include "cudagrad.hpp"
int main() {
  auto a = cg::tensor({2, 2}, {2.0, 3.0, 4.0, 5.0});
  auto b = cg::tensor({2, 2}, {6.0, 7.0, 8.0, 9.0});
  auto c = cg::tensor({2, 2}, {10.0, 10.0, 10.0, 10.0});
  auto d = cg::tensor({2, 2}, {11.0, 11.0, 11.0, 11.0});
  auto e = (cg::matmul(a, b) + c) * d;
  auto f = e.get()->sum();
  f.get()->backward();

  // (std::vector<float> &) { 2794.00f }
  f.get()->data_;
  // (std::vector<float> &) { 143.000f, 187.000f, 143.000f, 187.000f }
  a.get()->grad_;
  // (std::vector<float> &) { 66.0000f, 66.0000f, 88.0000f, 88.0000f }
  b.get()->grad_;
}
