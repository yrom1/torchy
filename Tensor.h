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
  Tensor() : storage_(std::make_shared<std::vector<T>>()) {}
  ~Tensor() = default;

  Tensor(std::initializer_list<size_t> dimensions)
      : dims(dimensions),
        storage_(std::make_shared<std::vector<T>>(calculate_size(dimensions))) {
  }

  Tensor(const std::vector<size_t> &dimensions, const std::vector<T> &values)
      : dims(dimensions), storage_(std::make_shared<std::vector<T>>(values)) {}

  Tensor(const Tensor<T> &other)
      : dims(other.dims), storage_(other.storage_), offset_(other.offset_) {}

  Tensor<T> &operator=(const Tensor<T> &other) {
    dims = other.dims;
    storage_ = other.storage_;
    offset_ = other.offset_;
    return *this;
  }

  friend bool operator==(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    return lhs.dims == rhs.dims && *(lhs.storage_) == *(rhs.storage_) &&
           lhs.offset_ == rhs.offset_;
  }

  friend bool operator!=(const Tensor<T> &lhs, const Tensor<T> &rhs) {
    return !(lhs == rhs);
  }

  Tensor<T> operator[](size_t index) {
    if (dims.empty()) {
      throw std::out_of_range("Cannot index a scalar tensor.");
    }
    if (index >= dims[0]) {
      throw std::out_of_range("Index out of range.");
    }
    std::vector<size_t> new_dims(dims.begin() + 1, dims.end());
    size_t new_offset = offset_ + index * strides()[0];
    return Tensor<T>(new_dims, storage_, new_offset);
  }

  const Tensor<T> operator[](size_t index) const {
    if (dims.empty()) {
      throw std::out_of_range("Cannot index a scalar tensor.");
    }
    if (index >= dims[0]) {
      throw std::out_of_range("Index out of range.");
    }
    std::vector<size_t> new_dims(dims.begin() + 1, dims.end());
    size_t new_offset = offset_ + index * strides()[0];
    return Tensor<T>(new_dims, storage_, new_offset);
  }

  T &operator()() {
    if (!dims.empty()) {
      throw std::out_of_range("Cannot access a non-scalar tensor.");
    }
    return (*storage_)[offset_];
  }

  const T &operator()() const {
    if (!dims.empty()) {
      throw std::out_of_range("Cannot access a non-scalar tensor.");
    }
    return (*storage_)[offset_];
  }

  const std::vector<size_t> &size() const { return dims; }

  size_t numel() const { return storage_->size(); }

 private:
  Tensor(const std::vector<size_t> &dimensions,
         std::shared_ptr<std::vector<T>> storage, size_t offset)
      : dims(dimensions), storage_(storage), offset_(offset) {}

  std::vector<size_t> dims;
  std::shared_ptr<std::vector<T>> storage_;
  size_t offset_ = 0;

  std::vector<size_t> strides() const {
    std::vector<size_t> strides;
    strides.reserve(dims.size());
    size_t stride = 1;
    for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
      strides.push_back(stride);
      stride *= *it;
    }
    std::reverse(strides.begin(), strides.end());
    return strides;
  }

  size_t calculate_size(const std::initializer_list<size_t> &dimensions) const {
    size_t size = 1;
    for (size_t dim : dimensions) {
      size *= dim;
    }
    return size;
  }
};

#endif  // TENSOR_H_
