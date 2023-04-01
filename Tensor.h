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

  T &operator[](const std::initializer_list<size_t> &indices) {
    size_t index = calculate_index(indices);
    return data[index];
  }

  const T &operator[](const std::initializer_list<size_t> &indices) const {
    size_t index = calculate_index(indices);
    return data[index];
  }

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
