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

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ag {

struct AutoGradBackward;
struct AddBackward;
struct SubBackward;
struct MulBackward;
struct DivBackward;
using SumBackward = AddBackward;
struct ReluBackward;
struct MatMulBackward;

struct AutoGradForward;
struct MatMulForward;

class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
  std::vector<int> size_;
  std::vector<float> data_;
  std::vector<float> grad_;
  std::vector<std::shared_ptr<Tensor>> children_;
  std::shared_ptr<AutoGradBackward> grad_fn_ = nullptr;
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
         std::shared_ptr<AutoGradBackward> grad_fn, char op = '?')
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
  std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> lhs,
                                 std::shared_ptr<Tensor> rhs);

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
  void _backward();

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
std::shared_ptr<Tensor> binaryElementwiseOperator(
    std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs,
    std::function<float(float, float)> func, char op,
    std::shared_ptr<T> backward) {
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
  return binaryElementwiseOperator<AddBackward>(
      lhs, rhs, std::plus<float>(), '+', std::make_shared<AddBackward>());
}

std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> lhs,
                                  std::shared_ptr<Tensor> rhs) {
  return binaryElementwiseOperator<SubBackward>(
      lhs, rhs, std::minus<float>(), '-', std::make_shared<SubBackward>());
}

std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> lhs,
                                  std::shared_ptr<Tensor> rhs) {
  return binaryElementwiseOperator<MulBackward>(
      lhs, rhs, std::multiplies<float>(), '*', std::make_shared<MulBackward>());
}

std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> lhs,
                                  std::shared_ptr<Tensor> rhs) {
  return binaryElementwiseOperator<DivBackward>(
      lhs, rhs, std::divides<float>(), '/', std::make_shared<DivBackward>());
}

template <typename T>
std::shared_ptr<Tensor> binaryForwardOperator(std::shared_ptr<Tensor> lhs,
                                              std::shared_ptr<Tensor> rhs,
                                              T forward) {
  return (*forward)();  // forward.get()();
}

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> lhs,
                               std::shared_ptr<Tensor> rhs) {
  std::shared_ptr<MatMulForward> f = std::make_shared<MatMulForward>(lhs, rhs);
  return binaryForwardOperator<std::shared_ptr<MatMulForward>>(lhs, rhs, f);
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

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> input) {
  std::vector<float> result_data(input.get()->data_.size());
  for (int i = 0; i < input.get()->data_.size(); ++i) {
    result_data[i] = std::max(0.0f, input.get()->data_[i]);
  }
  return std::make_shared<Tensor>(
      input.get()->size_, result_data,
      std::vector<std::shared_ptr<Tensor>>{input.get()->get_shared()},
      std::make_shared<ReluBackward>(), 'r');
}

std::shared_ptr<Tensor> tensor(std::initializer_list<int> size,
                               std::initializer_list<float> data) {
  return std::make_shared<Tensor>(size, data);
}

std::shared_ptr<Tensor> tensor(
    std::vector<int> size, std::vector<float> data,
    std::vector<std::shared_ptr<Tensor>> children = {},
    std::shared_ptr<AutoGradBackward> grad_fn =
        std::make_shared<AutoGradBackward>(),
    char op = '?') {
  return std::make_shared<Tensor>(size, data, children, std::move(grad_fn), op);
}

struct AutoGradBackward {
  AutoGradBackward() = default;
  virtual ~AutoGradBackward() = default;

  virtual void apply(std::shared_ptr<Tensor> grad_output,
                     std::vector<std::shared_ptr<Tensor>> grad_inputs) = 0;
};

struct AddBackward : public AutoGradBackward {
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

struct SubBackward : public AutoGradBackward {
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

struct MulBackward : public AutoGradBackward {
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

struct DivBackward : public AutoGradBackward {
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

struct ReluBackward : public AutoGradBackward {
  ReluBackward() = default;

  void apply(std::shared_ptr<Tensor> grad_output,
             std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
    assert(grad_inputs.size() == 1);
    std::shared_ptr<Tensor> input = grad_inputs[0];
    for (int i = 0; i < input.get()->grad_.size(); ++i) {
      input.get()->grad_[i] +=
          grad_output.get()->grad_[i] * (input.get()->data_[i] > 0 ? 1 : 0);
    }
  }
};

struct AutoGradForward {
  AutoGradForward() = default;
  virtual ~AutoGradForward() = default;

  virtual std::shared_ptr<Tensor> operator()()
      // (const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>&
      // rhs)
      = 0;
};

std::vector<float> _matmul(const std::vector<int> &lhs_size,
                           const std::vector<float> &lhs_data,
                           const std::vector<int> &rhs_size,
                           const std::vector<float> &rhs_data) {
  int m = lhs_size[0];
  int n = rhs_size[1];
  int k = lhs_size[1];

  std::vector<float> result_data(m * n, 0.0f);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int p = 0; p < k; ++p) {
        result_data[i * n + j] += lhs_data[i * k + p] * rhs_data[p * n + j];
      }
    }
  }

  return result_data;
}

struct MatMulForward : public AutoGradForward {
  std::shared_ptr<Tensor> lhs_;
  std::shared_ptr<Tensor> rhs_;

  MatMulForward(const std::shared_ptr<Tensor> &lhs,
                const std::shared_ptr<Tensor> &rhs)
      : lhs_(lhs), rhs_(rhs) {
    if (lhs.get()->size_.size() != 2 || rhs.get()->size_.size() != 2 ||
        lhs.get()->size_[1] != rhs.get()->size_[0]) {
      throw std::runtime_error("Invalid dimensions for matrix multiplication.");
    }
  }

  std::shared_ptr<Tensor> operator()() {
    std::vector<float> result_data =
        _matmul(lhs_.get()->size_, lhs_.get()->data_, rhs_.get()->size_,
                rhs_.get()->data_);

    std::vector<int> result_size = {lhs_.get()->size_[0], rhs_.get()->size_[1]};
    std::vector<std::shared_ptr<Tensor>> result_children = {lhs_, rhs_};
    std::shared_ptr<MatMulBackward> result_grad_fn =
        std::make_shared<MatMulBackward>();

    return std::make_shared<Tensor>(result_size, result_data,
                                    std::move(result_children),
                                    std::move(result_grad_fn), '@');
  }
};

std::vector<float> transpose(const std::vector<int> &size,
                             const std::vector<float> &data) {
  int rows = size[0];
  int cols = size[1];
  std::vector<float> transposed_data(cols * rows, 0.0f);

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      transposed_data[j * rows + i] = data[i * cols + j];
    }
  }

  return transposed_data;
}

struct MatMulBackward : public AutoGradBackward {
  MatMulBackward() = default;

  void apply(std::shared_ptr<Tensor> grad_output,
             std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
    auto a = grad_inputs[0];
    auto b = grad_inputs[1];

    auto b_transposed_size =
        std::vector<int>{b.get()->size_[1], b.get()->size_[0]};
    auto b_transposed_data = transpose(b.get()->size_, b.get()->data_);

    auto a_transposed_size =
        std::vector<int>{a.get()->size_[1], a.get()->size_[0]};
    auto a_transposed_data = transpose(a.get()->size_, a.get()->data_);

    auto grad_a_data =
        _matmul(grad_output.get()->size_, grad_output.get()->grad_,
                b_transposed_size, b_transposed_data);
    auto grad_b_data =
        _matmul(a_transposed_size, a_transposed_data, grad_output.get()->size_,
                grad_output.get()->grad_);

    for (int i = 0; i < a.get()->grad_.size(); ++i) {
      a.get()->grad_[i] += grad_a_data[i];
    }

    for (int i = 0; i < b.get()->grad_.size(); ++i) {
      b.get()->grad_[i] += grad_b_data[i];
    }
  }
};

void Tensor::_backward() {
  grad_fn_.get()->apply(get_shared(), children_);
  for (auto child : children_) {
    if (child.get()->grad_fn_ != nullptr) {
      child.get()->_backward();
    }
  }
}

void Tensor::backward() {
  assert(data_.size() == 1);
  grad_ = std::vector<float>(_product(size_), 1.0f);
  grad_fn_.get()->apply(get_shared(), children_);
  for (auto child : children_) {
    if (child.get()->grad_fn_ != nullptr) {
      assert(child.get()->op_ != '?');
      child.get()->_backward();
    }
  }
}

using t = std::shared_ptr<Tensor>;

}  // namespace ag

#endif  // TORCHY_HPP_
