// Copyright 2023 Ryan Moore

#include <iostream>
#include <stdexcept>
#include <vector>

class T {
 public:
  T(std::initializer_list<size_t> dimensions)
      : dims(dimensions), data(calculate_size(dimensions)) {}

  float &at(std::initializer_list<size_t> indices) {
    size_t index = calculate_index(indices);
    return data[index];
  }

  const float &at(std::initializer_list<size_t> indices) const {
    size_t index = calculate_index(indices);
    return data[index];
  }

 private:
  std::vector<size_t> dims;
  std::vector<float> data;

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

int main() {
  T t{3, 4, 2};

  t.at({0, 0, 0}) = 1.0f;
  t.at({1, 1, 1}) = 2.0f;

  std::cout << "t[0, 0, 0] = " << t.at({0, 0, 0}) << std::endl;
  std::cout << "t[1, 1, 1] = " << t.at({1, 1, 1}) << std::endl;

  return 0;
}
