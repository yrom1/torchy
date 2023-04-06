// Copyright 2023 Ryan Moore

#ifndef TENSOR_H_
#define TENSOR_H_

#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

template <typename T>
class Storage {
 public:
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

 private:
  std::vector<T> data_;
};

template <typename T>
class Tensor {
 public:
  Tensor(std::initializer_list<size_t> dimensions, std::vector<T> values = {})
      : Tensor(std::vector<size_t>(dimensions), std::move(values)) {}

  explicit Tensor(const std::vector<size_t> &dimensions,
                  std::vector<T> values = {})
      : sizes_(dimensions),
        storage_(
            std::make_shared<Storage<T>>(computeSize(), std::move(values))),
        offset_(0) {
    computeStrides();
  }

  Tensor(std::shared_ptr<Storage<T>> storage, const std::vector<size_t> &sizes,
         const std::vector<size_t> &strides, size_t offset = 0)
      : sizes_(sizes),
        storage_(std::move(storage)),
        offset_(offset),
        strides_(strides) {}

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
    return applyElementwise(other, std::plus<T>());
  }

  Tensor<T> operator+(const T &scalar) const {
    return applyElementwise(scalar, std::plus<T>());
  }

  Tensor<T> operator-(const Tensor<T> &other) const {
    return applyElementwise(other, std::plus<T>());
  }

  Tensor<T> operator-(const T &scalar) const {
    return applyElementwise(scalar, std::plus<T>());
  }

  Tensor<T> operator*(const Tensor<T> &other) const {
    return applyElementwise(other, std::divides<T>());
  }

  Tensor<T> operator*(const T &scalar) const {
    return applyElementwise(scalar, std::divides<T>());
  }

  Tensor<T> operator/(const Tensor<T> &other) const {
    return applyElementwise(other, std::divides<T>());
  }

  Tensor<T> operator/(const T &scalar) const {
    return applyElementwise(scalar, std::divides<T>());
  }

  Tensor<T> matmul(const Tensor<T> &other) const {
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

 private:
  std::vector<size_t> sizes_;
  std::shared_ptr<Storage<T>> storage_;
  size_t offset_;
  std::vector<size_t> strides_;

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

#endif  // TENSOR_H_
