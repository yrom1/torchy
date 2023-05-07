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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ag {

struct AutoGradFunction;
struct AddBackward;
using SumBackward = AddBackward;

class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
  std::vector<int> size_;
  std::vector<float> data_;
  std::vector<float> grad_;
  std::vector<std::shared_ptr<Tensor>> children_;
  std::shared_ptr<AutoGradFunction> grad_fn_ = nullptr;
  char op_;

  Tensor(std::initializer_list<int> size, std::initializer_list<float> data)
      : size_(size),
        data_(data),
        grad_(data.size(), 0),
        children_(),
        grad_fn_(),
        op_('?') {}

  Tensor(std::vector<int> size, std::vector<float> data,
         std::vector<std::shared_ptr<Tensor>> children,
         std::shared_ptr<AutoGradFunction> grad_fn, char op = '?')
      : size_(size),
        data_(data),
        grad_(data.size(), 0),
        children_(children),
        grad_fn_(std::move(grad_fn)),
        op_(op) {}

  Tensor(Tensor&& other)
      : size_(std::move(other.size_)),
        data_(std::move(other.data_)),
        grad_(std::move(other.grad_)),
        children_(std::move(other.children_)),
        grad_fn_(std::move(other.grad_fn_)),
        op_(other.op_) {}

  std::shared_ptr<Tensor> get_shared() { return this->shared_from_this(); }

  void backward();
  std::shared_ptr<Tensor> sum();

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
    auto print_if_not_leaf = [this](char c) {
        if (c != '?') return c;
        return ' ';
    };
    std::string tab(depth, ' ');
    char displayed_op = print_if_not_leaf(op_);
    std::cout << tab << get_shared() << " " << displayed_op << std::endl;
    for (auto c : children_) {
      c.get()->graph(depth + 2);
    }
  }

 private:
  int _product(std::vector<int> size) {
    int product = 1;
    for (int s : size) {
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
      std::make_shared<Tensor>(lhs.get()->size_, result_data, children,
                               std::make_shared<AddBackward>(), '+');
  return result;
}

std::shared_ptr<Tensor> Tensor::sum() {
  std::vector<float> total(1, 0);
  for (float x : data_) {
    total[0] += x;
  }
  return std::make_shared<Tensor>(
      std::vector<int>{1}, std::vector<float>{total},
      std::vector<std::shared_ptr<Tensor>>{get_shared()},
      std::make_shared<SumBackward>(), 's');
}

std::shared_ptr<Tensor> tensor(std::initializer_list<int> size,
                               std::initializer_list<float> data) {
  return std::make_shared<Tensor>(size, data);
}

std::shared_ptr<Tensor> tensor(
    std::vector<int> size, std::vector<float> data,
    std::vector<std::shared_ptr<Tensor>> children = {},
    std::shared_ptr<AutoGradFunction> grad_fn =
        std::make_shared<AutoGradFunction>(),
    char op = '?') {
  return std::make_shared<Tensor>(size, data, children, std::move(grad_fn), op);
}

struct AutoGradFunction {
  AutoGradFunction() = default;
  virtual ~AutoGradFunction() = default;

  virtual void apply(std::shared_ptr<Tensor> grad_output,
                     std::vector<std::shared_ptr<Tensor>> grad_inputs) = 0;
};

struct AddBackward : public AutoGradFunction {
  AddBackward() = default;

  void apply(std::shared_ptr<Tensor> grad_output,
             std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
    for (std::shared_ptr<Tensor> grad_input : grad_inputs) {
      for (size_t i = 0; i < grad_input.get()->grad_.size(); ++i) {
        grad_input.get()->grad_[i] += grad_output.get()->grad_[0];
      }
    }
  }
};

void Tensor::backward() {
  grad_ = std::vector<float>(_product(size_), 1.0f);
  grad_fn_.get()->apply(get_shared(), children_);
  for (auto child : children_) {
    if (child.get()->grad_fn_ != nullptr) {
      child.get()->backward();
    }
  }
}

using t = std::shared_ptr<Tensor>;

}  // namespace ag

#endif  // TORCHY_HPP_
