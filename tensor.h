// Copyright 2023 Ryan Moore

#ifndef TENSOR_H_
#define TENSOR_H_

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

template <typename T>
class Storage {
 public:
  typedef T dtype;

  explicit Storage(size_t size, std::vector<T> values = {})
      : data_(std::move(values)) {
    if (data_.empty()) {
      data_.resize(size);
    } else if (data_.size() != size) {
      throw std::runtime_error(
          "Size of the provided values vector does not match the size of the "
          "storage.");
    }
  }

  T &operator[](size_t idx) { return data_[idx]; }
  const T &operator[](size_t idx) const { return data_[idx]; }
  const std::vector<T> &data() const { return data_; }

 private:
  std::vector<T> data_;
};

template <typename T>
class Tensor {
 public:
  typedef T dtype;

  explicit Tensor(const std::vector<size_t> &dimensions,
                  std::vector<T> values = {})
      : sizes_(dimensions),
        storage_(
            std::make_shared<Storage<T>>(computeSize(), std::move(values))),
        offset_(0),
        autograd_meta_(nullptr) {
    computeStrides();
  }

  Tensor(std::shared_ptr<Storage<T>> storage, const std::vector<size_t> &sizes,
         const std::vector<size_t> &strides, size_t offset = 0,
         std::shared_ptr<AutogradMeta> autograd_meta = nullptr)
      : sizes_(sizes),
        storage_(std::move(storage)),
        offset_(offset),
        strides_(strides),
        autograd_meta_(std::move(autograd_meta)) {}

  // Create a view on the tensor by slicing along a dimension
  Tensor slice(size_t dimension, size_t start, size_t end) const {
    if (dimension >= sizes_.size()) {
      throw std::runtime_error("Dimension out of bounds.");
    }
    if (start >= end || end > sizes_[dimension]) {
      throw std::runtime_error("Invalid slice range.");
    }

    std::vector<size_t> new_sizes = sizes_;
    new_sizes[dimension] = end - start;
    size_t new_offset = offset_ + start * strides_[dimension];

    return Tensor(storage_, new_sizes, strides_, new_offset);
  }

  // Reshape the tensor while preserving its underlying storage
  Tensor reshape(const std::vector<size_t> &new_sizes) const {
    if (computeSize(new_sizes) != computeSize()) {
      throw std::runtime_error(
          "New sizes do not match the original number of elements.");
    }

    std::vector<size_t> new_strides(new_sizes.size());
    size_t stride = 1;
    for (int i = new_sizes.size() - 1; i >= 0; i--) {
      new_strides[i] = stride;
      stride *= new_sizes[i];
    }

    return Tensor(storage_, new_sizes, new_strides, offset_);
  }

  size_t computeSize() const { return computeSize(sizes_); }

  size_t computeSize(const std::vector<size_t> &sizes) const {
    size_t total_size = 1;
    for (const auto &dim : sizes) {
      total_size *= dim;
    }
    return total_size;
  }

  T &operator()(const std::vector<size_t> &indices) {
    checkIndicesBounds(indices);

    size_t idx = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
      idx += indices[i] * strides_[i];
    }
    return (*storage_)[idx];
  }

  const T &operator()(const std::vector<size_t> &indices) const {
    checkIndicesBounds(indices);

    size_t idx = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
      idx += indices[i] * strides_[i];
    }
    return (*storage_)[idx];
  }

  const std::vector<size_t> &sizes() const { return sizes_; }
  const std::vector<size_t> &strides() const { return strides_; }
  const std::shared_ptr<Storage<T>> &storage() const { return storage_; }

  friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
    printTensor(os, tensor, {}, 0);
    return os;
  }

  // TODO(yrom1) if same shape just add the underlying vectors?
  Tensor<T> operator+(const Tensor<T> &other) const {
    return applyElementwiseWithBroadcast(other, std::plus<T>());
  }

  Tensor<T> operator+(const T &scalar) const {
    return applyElementwise(scalar, std::plus<T>());
  }

  Tensor<T> operator-(const Tensor<T> &other) const {
    return applyElementwiseWithBroadcast(other, std::plus<T>());
  }

  Tensor<T> operator-(const T &scalar) const {
    return applyElementwise(scalar, std::plus<T>());
  }

  Tensor<T> operator*(const Tensor<T> &other) const {
    return applyElementwiseWithBroadcast(other, std::divides<T>());
  }

  Tensor<T> operator*(const T &scalar) const {
    return applyElementwise(scalar, std::divides<T>());
  }

  Tensor<T> operator/(const Tensor<T> &other) const {
    return applyElementwiseWithBroadcast(other, std::divides<T>());
  }

  Tensor<T> operator/(const T &scalar) const {
    return applyElementwise(scalar, std::divides<T>());
  }

  Tensor<T> matmul(const Tensor<T> &other) const {
    // https://pytorch.org/docs/stable/generated/torch.matmul.html
    // This is supported:
    // If both arguments are 2-dimensional, the matrix-matrix product is
    // returned. Everything else is not supported yet.
    if (sizes_.size() != 2 || other.sizes_.size() != 2) {
      throw std::runtime_error("Both tensors must be 2-dimensional.");
    }

    if (sizes_[1] != other.sizes_[0]) {
      throw std::runtime_error(
          "Incompatible dimensions for matrix multiplication.");
    }

    std::vector<size_t> result_sizes = {sizes_[0], other.sizes_[1]};
    std::vector<T> result_values(result_sizes[0] * result_sizes[1], 0);

    for (size_t i = 0; i < result_sizes[0]; ++i) {
      for (size_t j = 0; j < result_sizes[1]; ++j) {
        for (size_t k = 0; k < sizes_[1]; ++k) {
          // Access the element at (i, j) in the result_values vector by
          // converting the 2-dimensional index (i, j) to a 1-dimensional index
          // The formula for this conversion is:
          //   linear_index = i * number_of_columns + j
          // where i represents the row index and j represents the column index.
          result_values[i * result_sizes[1] + j] +=
              (*this)(std::vector<size_t>{i, k}) *
              other(std::vector<size_t>{k, j});
        }
      }
    }

    return Tensor<T>(result_sizes, result_values);
  }

  std::string repr() const {
    std::ostringstream ss;
    ss << "Tensor({";
    for (size_t i = 0; i < sizes_.size(); ++i) {
      ss << sizes_[i];
      if (i != sizes_.size() - 1) {
        ss << ", ";
      }
    }
    ss << "}, {";
    std::vector<T> values = storage()->data();
    for (size_t i = 0; i < values.size(); ++i) {
      ss << values[i];
      if (i != values.size() - 1) {
        ss << ", ";
      }
    }
    ss << "})";
    return ss.str();
  }

 private:
  std::vector<size_t> sizes_;
  std::shared_ptr<Storage<T>> storage_;
  size_t offset_;
  std::vector<size_t> strides_;
  std::shared_ptr<AutogradMeta> autograd_meta_;

  Tensor<T> applyElementwiseWithBroadcast(
      const Tensor<T> &other,
      const std::function<T(const T &, const T &)> &func) const {
    std::vector<size_t> result_sizes = broadcastableShape(sizes_, other.sizes_);

    Tensor<T> result(result_sizes);
    for (size_t i = 0; i < result.computeSize(); ++i) {
      std::vector<size_t> result_indices = result.unravelIndex(i);
      std::vector<size_t> this_indices =
          broadcastIndices(result_indices, sizes_);
      std::vector<size_t> other_indices =
          broadcastIndices(result_indices, other.sizes_);

      T this_value = (*this)(this_indices);
      T other_value = other(other_indices);
      if (func.target_type().name() == typeid(std::divides<T>).name() &&
          other_value == static_cast<T>(0)) {
        throw std::runtime_error("Division by zero.");
      }
      result(result_indices) = func(this_value, other_value);
    }

    return result;
  }

  std::vector<size_t> broadcastIndices(const std::vector<size_t> &indices,
                                       const std::vector<size_t> &shape) const {
    std::vector<size_t> broadcasted_indices(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      broadcasted_indices[i] = (shape[i] == 1) ? 0 : indices[i];
    }
    return broadcasted_indices;
  }

  std::vector<size_t> broadcastableShape(
      const std::vector<size_t> &shape1,
      const std::vector<size_t> &shape2) const {
    size_t rank1 = shape1.size();
    size_t rank2 = shape2.size();
    size_t max_rank = std::max(rank1, rank2);

    std::vector<size_t> padded_shape1(max_rank, 1);
    std::vector<size_t> padded_shape2(max_rank, 1);

    for (size_t i = 0; i < rank1; ++i) {
      padded_shape1[max_rank - rank1 + i] = shape1[i];
    }
    for (size_t i = 0; i < rank2; ++i) {
      padded_shape2[max_rank - rank2 + i] = shape2[i];
    }

    std::vector<size_t> result_shape(max_rank);
    for (size_t i = 0; i < max_rank; ++i) {
      if (padded_shape1[i] == padded_shape2[i]) {
        result_shape[i] = padded_shape1[i];
      } else if (padded_shape1[i] == 1) {
        result_shape[i] = padded_shape2[i];
      } else if (padded_shape2[i] == 1) {
        result_shape[i] = padded_shape1[i];
      } else {
        throw std::runtime_error(
            "Shapes are not broadcastable: mismatch in dimension " +
            std::to_string(i) + ".");
      }
    }
    return result_shape;
  }

  Tensor<T> applyElementwise(
      const Tensor<T> &other,
      const std::function<T(const T &, const T &)> &func) const {
    if (sizes_ != other.sizes_) {
      throw std::runtime_error(
          "Tensors must have the same shape for element-wise operations.");
    }

    std::vector<T> result_values(computeSize());
    for (size_t i = 0; i < result_values.size(); ++i) {
      std::vector<size_t> indices = unravelIndex(i);
      T other_value = other(indices);
      if (func.target_type().name() == typeid(std::divides<T>).name() &&
          other_value == static_cast<T>(0)) {
        throw std::runtime_error("Division by zero.");
      }
      result_values[i] = func((*this)(indices), other_value);
    }

    return Tensor<T>(sizes_, result_values);
  }

  Tensor<T> applyElementwise(
      const T &scalar,
      const std::function<T(const T &, const T &)> &func) const {
    if (func.target_type().name() == typeid(std::divides<T>).name() &&
        scalar == static_cast<T>(0)) {
      throw std::runtime_error("Division by zero.");
    }

    std::vector<T> result_values(computeSize());
    for (size_t i = 0; i < result_values.size(); ++i) {
      std::vector<size_t> indices = unravelIndex(i);
      result_values[i] = func((*this)(indices), scalar);
    }

    return Tensor<T>(sizes_, result_values);
  }

  std::vector<size_t> unravelIndex(size_t index) const {
    std::vector<size_t> indices(sizes_.size());
    for (int i = sizes_.size() - 1; i >= 0; --i) {
      indices[i] = (index % sizes_[i]);
      index /= sizes_[i];
    }
    return indices;
  }

  void computeStrides() {
    strides_.resize(sizes_.size());
    size_t stride = 1;
    for (int i = sizes_.size() - 1; i >= 0; i--) {
      strides_[i] = stride;
      stride *= sizes_[i];
    }
  }

  void checkIndicesBounds(const std::vector<size_t> &indices) const {
    if (indices.size() != sizes_.size()) {
      throw std::runtime_error(
          "Number of indices does not match the number of dimensions.");
    }

    for (size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] >= sizes_[i]) {
        throw std::runtime_error("Index out of bounds.");
      }
    }
  }

  static void printTensor(std::ostream &os, const Tensor<T> &tensor,
                          std::vector<size_t> indices, size_t depth) {
    if (depth == tensor.sizes_.size() - 1) {
      os << "[";
      for (size_t i = 0; i < tensor.sizes_[depth]; ++i) {
        indices.push_back(i);
        os << tensor(indices);
        indices.pop_back();
        if (i != tensor.sizes_[depth] - 1) {
          os << ", ";
        }
      }
      os << "]";
    } else {
      os << "[";
      for (size_t i = 0; i < tensor.sizes_[depth]; ++i) {
        indices.push_back(i);
        printTensor(os, tensor, indices, depth + 1);
        indices.pop_back();
        if (i != tensor.sizes_[depth] - 1) {
          os << ",\n";
          for (size_t j = 0; j <= depth; ++j) {
            os << " ";
          }
        }
      }
      os << "]";
    }
  }
};

class AutogradFunction {
 public:
  virtual ~AutogradFunction() = default;

  virtual Tensor<float> apply(const Tensor<float> &input) const = 0;
};

class AutogradMeta {
 public:
  AutogradMeta() : gradient_(nullptr), grad_fn_(nullptr) {}

  void set_gradient(const Tensor<float> &gradient) { gradient_ = gradient; }
  Tensor<float> &gradient() { return gradient_; }

  void set_grad_fn(std::shared_ptr<AutogradFunction> grad_fn) {
    grad_fn_ = std::move(grad_fn);
  }
  std::shared_ptr<AutogradFunction> grad_fn() const { return grad_fn_; }

 private:
  Tensor<float> gradient_;
  std::shared_ptr<AutogradFunction> grad_fn_;
};

std::shared_ptr<AutogradMeta> autograd_meta() const { return autograd_meta_; }

void set_autograd_meta(std::shared_ptr<AutogradMeta> autograd_meta) {
  autograd_meta_ = std::move(autograd_meta);
}

Tensor<float> &gradient() {
  if (!autograd_meta_) {
    throw std::runtime_error("No AutogradMeta is set.");
  }
  return autograd_meta_->gradient();
}

void set_gradient(const Tensor<float> &gradient) {
  if (!autograd_meta_) {
    autograd_meta_ = std::make_shared<AutogradMeta>();
  }
  autograd_meta_->set_gradient(gradient);
}

std::shared_ptr<AutogradFunction> grad_fn() const {
  if (!autograd_meta_) {
    throw std::runtime_error("No AutogradMeta is set.");
  }
  return autograd_meta_->grad_fn();
}

void set_grad_fn(Tensor &result, Tensor &input, Tensor &weight) {
  result.grad_fn =
      [input_ptr = input.shared_from_this(),
       weight_ptr = weight.shared_from_this()](
          const Tensor &grad_output) mutable -> std::vector<Tensor> {
    auto input_grad = mm(weight, grad_output.t());
    auto weight_grad = mm(grad_output, input.t());
    return {input_grad, weight_grad};
  };
}

Tensor backward(const Tensor &grad_output, const Tensor &tensor) {
  if (tensor.grad_fn) {
    auto grads = tensor.grad_fn(grad_output);
    for (size_t i = 0; i < grads.size(); ++i) {
      grads[i] = backward(grads[i], tensor.requires_grad_parents[i]);
    }
    return grads[0];
  }
  return grad_output;
}

#endif  // TENSOR_H_
