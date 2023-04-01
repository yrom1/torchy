// Copyright 2023 Ryan Moore
// Tensor.h

#ifndef TENSOR_H_
#define TENSOR_H_

#include <stdexcept>
#include <vector>

template <typename T>
class Tensor {
 public:
  Tensor() {}

  Tensor(std::initializer_list<size_t> dimensions)
      : dims(dimensions), data(calculate_size(dimensions)) {}

  // Tensor(const std::vector<size_t> &dimensions, const std::vector<T> &values)
  //     : dims(dimensions), data(values) {}

  friend bool operator==(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    return lhs.dims == rhs.dims && lhs.data == rhs.data;
  }

  friend bool operator!=(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    return !(lhs == rhs);
  }

  T &operator[](const std::initializer_list<size_t> &indices) {
    size_t index = calculate_index(indices);
    return data[index];
  }

  const T &operator[](const std::initializer_list<size_t> &indices) const {
    size_t index = calculate_index(indices);
    return data[index];
  }

  // Tensor<T> operator+(const Tensor<T> &other) const {
  //   if (dims != other.size()) {
  //     throw std::invalid_argument(
  //         "Tensors must have the same dimensions for element-wise
  //         addition.");
  //   }

  //   Tensor<T> result(dims);
  //   for (size_t i = 0; i < data.size(); ++i) {
  //     result.data[i] = data[i] + other.data[i];
  //   }

  //   return result;
  // }

  // Tensor<T> operator-(const Tensor<T> &other) const {
  //   if (dims != other.size()) {
  //     throw std::invalid_argument(
  //         "Tensors must have the same dimensions for element-wise "
  //         "subtraction.");
  //   }

  //   Tensor<T> result(dims);
  //   for (size_t i = 0; i < data.size(); ++i) {
  //     result.data[i] = data[i] - other.data[i];
  //   }

  //   return result;
  // }

  // Tensor<T> operator*(const Tensor<T> &other) const {
  //   if (dims != other.size()) {
  //     throw std::invalid_argument(
  //         "Tensors must have the same dimensions for element-wise "
  //         "multiplication.");
  //   }

  //   Tensor<T> result(dims);
  //   for (size_t i = 0; i < data.size(); ++i) {
  //     result.data[i] = data[i] * other.data[i];
  //   }

  //   return result;
  // }

  // Tensor<T> operator/(const Tensor<T> &other) const {
  //   if (dims != other.size()) {
  //     throw std::invalid_argument(
  //         "Tensors must have the same dimensions for element-wise
  //         division.");
  //   }

  //   Tensor<T> result(dims);
  //   for (size_t i = 0; i < data.size(); ++i) {
  //     if (other.data[i] == 0) {
  //       throw std::runtime_error("Division by zero.");
  //     }
  //     result.data[i] = data[i] / other.data[i];
  //   }

  //   return result;
  // }

  const std::vector<size_t> &size() const { return dims; }

  size_t numel() const { return data.size(); }

 private:
  std::vector<size_t> dims;
  std::vector<T> data;

  size_t calculate_size(const std::initializer_list<size_t> &dimensions) const {
    size_t size = 1;
    for (size_t dim : dimensions) {
      size *= dim;
    }
    return size;
  }

  size_t calculate_index(const std::initializer_list<size_t> &indices) const {
    if (indices.size() != dims.size()) {
      throw std::out_of_range("Incorrect number of indices provided.");
    }

    size_t index = 0;
    size_t stride = 1;

    auto dims_it = dims.begin();
    for (auto idx_it = indices.begin(); idx_it != indices.end();
         ++idx_it, ++dims_it) {
      if (*idx_it >= *dims_it) {
        throw std::out_of_range("Index out of range.");
      }

      index += stride * (*idx_it);
      stride *= (*dims_it);
    }

    return index;
  }
};

#endif  // TENSOR_H_
