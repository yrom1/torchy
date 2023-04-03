// Copyright 2023 Ryan Moore
// Tensor.h

#ifndef TENSOR_H_
#define TENSOR_H_

#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

template <typename T>
class Storage {
 public:
  explicit Storage(size_t size, std::vector<T> values = {})
      : data_(std::move(values)) {
    if (data_.empty()) {
      data_.resize(size);
    } else if (data_.size() != size) {
      throw std::runtime_error("Size of the provided values vector does not match the size of the storage.");
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
      : sizes_(dimensions),
        storage_(std::make_shared<Storage<T>>(computeSize(), std::move(values))),
        offset_(0) {
    computeStrides();
  }

  size_t computeSize() const {
    size_t total_size = 1;
    for (const auto &dim : sizes_) {
      total_size *= dim;
    }
    return total_size;
  }

  T &operator()(const std::vector<size_t> &indices) {
    size_t idx = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
      idx += indices[i] * strides_[i];
    }
    return (*storage_)[idx];
  }

  const std::vector<size_t> &sizes() const { return sizes_; }
  const std::vector<size_t> &strides() const { return strides_; }
  const std::shared_ptr<Storage<T>> &storage() const { return storage_; }

 private:
  void computeStrides() {
    strides_.resize(sizes_.size());
    size_t stride = 1;
    for (int i = sizes_.size() - 1; i >= 0; i--) {
      strides_[i] = stride;
      stride *= sizes_[i];
    }
  }

  std::vector<size_t> sizes_;
  std::shared_ptr<Storage<T>> storage_;
  size_t offset_;
  std::vector<size_t> strides_;
};

#endif  // TENSOR_H_
