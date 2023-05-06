// Copyright 2023 Ryan Moore
//
// "If you remember just one thing out of this course, it should be this – two
// pieces of advice. The first piece of advice is whenever you can, use vector.
// The second piece of advice is if you cannot, find a way how you can. This is
// to avoid any data structures except arrays... Vector is an array, right. Sort
// of, you say, 'Well... Aren't there exceptions?' – Yes, not for you!"
//
// — Alexander Stepanov

#ifndef TORCHY_HPP_
#define TORCHY_HPP_

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace ag {

class Tensor;

class AutoGradFunction {
  virtual ~AutoGradFunction() {}

  virtual void apply(Tensor grad_output,
                     std::vector<std::shared_ptr<Tensor>> grad_inputs) = 0;
};

class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
  std::vector<int> size_;
  std::vector<float> data_;
  std::vector<float> grad_;
  std::vector<std::shared_ptr<Tensor>> children_;
  char op_;

  Tensor(std::initializer_list<int> size, std::initializer_list<float> data)
      : size_(size),
        data_(data),
        grad_(_product(size), 0),
        children_(),
        op_('?') {}
  Tensor(std::vector<int> size, std::vector<float> data,
         std::vector<std::shared_ptr<Tensor>> children, char op)
      : size_(size),
        data_(data),
        grad_(_product(size), 0),
        children_(children),
        op_(op) {}

  std::shared_ptr<Tensor> get_shared() { return this->shared_from_this(); }

  void size() {
    for (auto x : size_) std::cout << x << std::endl;
  }
  void data() {
    for (auto x : data_) std::cout << x << std::endl;
  }
  void grad() {
    for (auto x : grad_) std::cout << x << std::endl;
  }
  void children() {
    for (auto x : children_) std::cout << x << std::endl;
  }

  void graph(int depth = 0) {
    std::string tab(depth, ' ');
    std::cout << tab << get_shared() << " " << op_ << std::endl;
    for (auto c : children_) {
      c.get()->graph(depth + 2);
    }
  }

 private:
  int _product(std::vector<int> size) {
    auto product = 1;
    for (auto s : size) {
      product *= s;
    }
    return product;
  }
};

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> lhs,
                                  std::shared_ptr<Tensor> rhs) {
  std::vector<std::shared_ptr<Tensor>> children;
  children.push_back(lhs);
  children.push_back(rhs);

  assert(lhs.get()->data_.size() == rhs.get()->data_.size());
  std::vector<float> result_data(lhs.get()->data_.size());
  for (size_t i = 0; i < lhs.get()->data_.size(); ++i) {
    result_data[i] = lhs.get()->data_[i] + rhs.get()->data_[i];
  }
  auto result =
      std::make_shared<Tensor>(lhs.get()->size_, result_data, children, '+');
  return result;
}

std::shared_ptr<Tensor> tensor(std::initializer_list<int> size,
                               std::initializer_list<float> data) {
  Tensor t(size, data);
  return std::make_shared<Tensor>(t);
}

std::shared_ptr<Tensor> tensor(
    std::vector<int> size, std::vector<float> data,
    std::vector<std::shared_ptr<Tensor>> children = {}, char op = '?') {
  Tensor t(size, data, children, op);
  return std::make_shared<Tensor>(t);
}

using T = std::shared_ptr<Tensor>;

}  // namespace ag

#endif  // TORCHY_HPP_
