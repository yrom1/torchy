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
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ag {

struct AutoGradFunction;
struct AddBackward;
struct SubBackward;
struct MulBackward;
struct DivBackward;
using SumBackward = AddBackward;

class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
  std::vector<int> size_;
  std::vector<float> data_;
  std::vector<float> grad_;
  std::vector<std::shared_ptr<Tensor>> children_;
  std::shared_ptr<AutoGradFunction> grad_fn_ = nullptr;
  char op_;
  int offset_;
  std::vector<int> strides_;

  Tensor(std::initializer_list<int> size, std::initializer_list<float> data)
      : size_(size),
        data_(data),
        grad_(data.size(), 0),
        children_(),
        grad_fn_(),
        op_('?'),
        offset_(0),
        strides_(size.size(), 0) {
    assert(_product(size) == data_.size());
    _computeStrides();
  }

  Tensor(std::vector<int> size, std::vector<float> data,
         std::vector<std::shared_ptr<Tensor>> children,
         std::shared_ptr<AutoGradFunction> grad_fn, char op = '?')
      : size_(size),
        data_(data),
        grad_(data.size(), 0),
        children_(children),
        grad_fn_(std::move(grad_fn)),
        op_(op),
        offset_(0),  // TODO(yrom1) pass as variable
        strides_(size.size(), 0) {
    _computeStrides();
  }

  Tensor(Tensor &&other)
      : size_(std::move(other.size_)),
        data_(std::move(other.data_)),
        grad_(std::move(other.grad_)),
        children_(std::move(other.children_)),
        grad_fn_(std::move(other.grad_fn_)),
        op_(other.op_),
        offset_(other.offset_),
        strides_(std::move(other.strides_)) {}

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
      auto print_if_not_leaf = [this](char c) -> const char {
          if (c != '?') return c;
          return ' ';
      };
      std::string tab(depth, ' ');
      char displayed_op = print_if_not_leaf(op_);
      std::cout << tab << this << " " << displayed_op << std::endl;
      for (auto c : children_) {
          c.get()->graph(depth + 2);
      }
  }

  float &data(const std::vector<size_t> &indices) {
    // checkIndicesBounds(indices);

    size_t idx = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
      idx += indices[i] * strides_[i];
    }
    return get_shared().get()->data_[idx];
  }

  float &grad(const std::vector<size_t> &indices) {
    // checkIndicesBounds(indices);

    size_t idx = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
      idx += indices[i] * strides_[i];
    }
    return get_shared().get()->grad_[idx];
  }

  static void print(std::ostream &os, const std::shared_ptr<Tensor> &tensor,
                    std::vector<size_t> indices, size_t depth,
                    bool print_grad = false) {
    if (depth == tensor.get()->size_.size() - 1) {
      os << "[";
      for (size_t i = 0; i < tensor.get()->size_[depth]; ++i) {
        indices.push_back(i);
        if (print_grad) {
          os << tensor.get()->grad(indices);
        } else {
          os << tensor.get()->data(indices);
        }
        indices.pop_back();
        if (i != tensor.get()->size_[depth] - 1) {
          os << ", ";
        }
      }
      os << "]";
    } else {
      os << "[";
      for (size_t i = 0; i < tensor.get()->size_[depth]; ++i) {
        indices.push_back(i);
        print(os, tensor, indices, depth + 1, print_grad);
        indices.pop_back();
        if (i != tensor.get()->size_[depth] - 1) {
          os << ",\n";
          os << std::string(depth + 1, ' ');
        }
      }
      os << "]";
    }
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const std::shared_ptr<Tensor> &tensor) {
    print(os, tensor, {}, 0);
    os << std::endl;
    print(os, tensor, {}, 0, true);
    return os;
  }

 private:
  int _product(std::vector<int> size) {
    int product = 1;
    for (int s : size) {
      product *= s;
    }
    return product;
  }

  void _computeStrides() {
    strides_.resize(size_.size());
    size_t stride = 1;
    for (int i = size_.size() - 1; i >= 0; i--) {
      strides_[i] = stride;
      stride *= size_[i];
    }
  }
};

template <typename T>
std::shared_ptr<Tensor> binaryOperator(std::shared_ptr<Tensor> lhs,
                                       std::shared_ptr<Tensor> rhs,
                                       std::function<float(float, float)> func,
                                       char op, std::shared_ptr<T> backward) {
  std::vector<std::shared_ptr<Tensor>> children;
  children.push_back(lhs);
  children.push_back(rhs);

  assert(lhs.get()->data_.size() == rhs.get()->data_.size());
  std::vector<float> result_data(lhs.get()->data_.size());
  for (int i = 0; i < lhs.get()->data_.size(); ++i) {
    result_data[i] = func(lhs.get()->data_[i], rhs.get()->data_[i]);
  }
  return std::make_shared<Tensor>(lhs.get()->size_, result_data, children,
                                  std::move(backward), op);
}

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> lhs,
                                  std::shared_ptr<Tensor> rhs) {
  return binaryOperator<AddBackward>(lhs, rhs, std::plus<float>(), '+',
                                     std::make_shared<AddBackward>());
}

std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> lhs,
                                  std::shared_ptr<Tensor> rhs) {
  return binaryOperator<SubBackward>(lhs, rhs, std::minus<float>(), '-',
                                     std::make_shared<SubBackward>());
}

std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> lhs,
                                  std::shared_ptr<Tensor> rhs) {
  return binaryOperator<MulBackward>(lhs, rhs, std::multiplies<float>(), '*',
                                     std::make_shared<MulBackward>());
}

std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> lhs,
                                  std::shared_ptr<Tensor> rhs) {
  return binaryOperator<DivBackward>(lhs, rhs, std::divides<float>(), '/',
                                     std::make_shared<DivBackward>());
}

std::shared_ptr<Tensor> Tensor::sum() {
  std::vector<float> total(1, 0.0f);
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
      for (int i = 0; i < grad_input.get()->grad_.size(); ++i) {
        grad_input.get()->grad_[i] += grad_output.get()->grad_[0];
      }
    }
  }
};

struct SubBackward : public AutoGradFunction {
  SubBackward() = default;

  void apply(std::shared_ptr<Tensor> grad_output,
             std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
    for (int i = 0; i < grad_inputs[0].get()->grad_.size(); ++i) {
      grad_inputs[0].get()->grad_[i] += grad_output.get()->grad_[0];
    }
    for (int i = 0; i < grad_inputs[1].get()->grad_.size(); ++i) {
      grad_inputs[1].get()->grad_[i] -= grad_output.get()->grad_[0];
    }
  }
};

struct MulBackward : public AutoGradFunction {
  MulBackward() = default;

  void apply(std::shared_ptr<Tensor> grad_output,
             std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
    for (int i = 0; i < grad_inputs[0].get()->grad_.size(); ++i) {
      grad_inputs[0].get()->grad_[i] +=
          grad_output.get()->grad_[0] * grad_inputs[1].get()->data_[i];
    }
    for (int i = 0; i < grad_inputs[1].get()->grad_.size(); ++i) {
      grad_inputs[1].get()->grad_[i] +=
          grad_output.get()->grad_[0] * grad_inputs[0].get()->data_[i];
    }
  }
};

struct DivBackward : public AutoGradFunction {
  DivBackward() = default;

  void apply(std::shared_ptr<Tensor> grad_output,
             std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
    for (int i = 0; i < grad_inputs[0].get()->grad_.size(); ++i) {
      // pro tip: don't put 1 instead of 1.0f!
      float update =
          grad_output.get()->grad_[i] * (1.0f / grad_inputs[1].get()->data_[i]);
      grad_inputs[0].get()->grad_[i] += update;
    }
    for (int i = 0; i < grad_inputs[1].get()->grad_.size(); ++i) {
      float update =
          grad_output.get()->grad_[i] *
          -(grad_inputs[0].get()->data_[i] /
            (grad_inputs[1].get()->data_[i] * grad_inputs[1].get()->data_[i]));
      grad_inputs[1].get()->grad_[i] += update;
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
