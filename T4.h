// Copyright 2023 Ryan Moore
// Tensor.h

#ifndef TENSOR_H_
#define TENSOR_H_

#include <memory>
#include <stdexcept>
#include <vector>

template <typename T>
class Tensor {
 public:
  Tensor(std::vector<size_t> dimensions, std::vector<T> values = {})
      : sizes_(std::move(dimensions)), data_(std::move(values)) {
    size_t total_size = 1;
    for (const auto &dim : sizes_) {
        total_size *= dim;
    }
    if (data_.empty()) {
      data_.resize(total_size);
    } else {
      if (data_.size() != expected_size) {
        throw std::runtime_error(
            "Size of the provided values vector does not match the product of "
            "dimensions.");
      }
    }
    computeStrides();
  }

  const std::vector<size_t> &sizes() const { return sizes_; }
  const std::vector<T> &data() const { return data_; }
  const std::vector<size_t> &strides() const { return strides_; }

 private:
  void computeStrides() {
    strides_.resize(sizes_.size());
    size_t stride = 1;
    for (int i = sizes_.size() - 1; i >= 0; i--) {
      strides_[i] = stride;
      stride *= sizes_[i];
    }
  }

  // typedef dtype T
  std::vector<size_t> sizes_;
  std::vector<T> data_;
  std::vector<size_t> strides_;
};

#endif  // TENSOR_H_
