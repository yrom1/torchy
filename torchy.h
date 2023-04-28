// Copyright 2023 Ryan Moore

#ifndef TORCHY_H_
#define TORCHY_H_

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

template <typename T>
class Storage;
template <typename T>
class AutogradMeta;
template <typename T>
class Tensor;

template <typename T>
class AutogradFunction {
 public:
  virtual ~AutogradFunction() {}
  virtual void apply(const Tensor<T> &grad_output,
                     std::vector<std::shared_ptr<Tensor<T>>>
                         &grad_inputs) = 0;  // NOLINT (runtime/reference)
  virtual char op() const { return '?'; }
};

template <typename T>
class AddBackward0 : public AutogradFunction<T> {
 public:
  AddBackward0() = default;
  void apply(const Tensor<T> &grad_output,
             std::vector<std::shared_ptr<Tensor<T>>>
                 &grad_inputs)  // NOLINT (runtime/references)
      override {
    std::cout << "Inside AddBackward0::apply before" << std::endl;
    std::cout << "input vec size: " << grad_inputs.size() << std::endl;
    // std::cout << "grad output data " <<
    // grad_output.autograd_meta_.get()->grad_.storage_.get()->data_ <<
    // std::endl; for (auto& grad_input : grad_inputs) {
    //   grad_input = grad_input +
    //   grad_output.autograd_meta_.get()->grad_.storage_.get()->data_;
    // }

    // for (auto &grad_input : grad_inputs) {
    //   std::cout << "iter " << std::endl;
    //   grad_input->autograd_meta_.get()->grad_.storage_.get()->data_ =
    //   grad_input->autograd_meta_.get()->grad_.storage_.get()->data_ +
    //   grad_output.autograd_meta_.get()->grad_.storage_.get()->data_;
    // }

    // size_t grad_input_size =
    // grad_input->autograd_meta_.get()->grad_.storage_.get()->data_.size();
    std::cout << "-1" << std::endl;
    std::cout << grad_output << std::endl;
    std::cout << "size: " << grad_output.storage_.get()->data_.size()
              << std::endl;

    size_t grad_output_size = grad_output.storage_.get()->data_.size();
    std::cout << grad_output_size << std::endl;
    // TODO(yrom1) check inputs outputs same length vector
    for (auto &grad_input : grad_inputs) {
      std::cout << "a0" << std::endl;
      for (size_t i = 0; i < grad_output_size; ++i) {
        std::cout << "a1" << std::endl;
        grad_input->autograd_meta_.get()->grad_.storage_.get()->data_[i] +=
            grad_output.storage_.get()->data_[i];
        std::cout << "a2" << std::endl;
      }
    }

    // grad_inputs[0] =
    // grad_output.autograd_meta_.get()->grad_.storage_.get()->data_
    // grad_inputs[1] = std::make_shared<Tensor<T>>(grad_output);
    std::cout << "Inside AddBackward0::apply after" << std::endl;
  }
  char op() const override { return '+'; };
};

// wrong
// Tensor::Tensor(at::Tensor data, std::shared_ptr<AutogradFunction> grad_fn) :
// data(data), grad_fn(grad_fn) {}

// void Tensor::backward(const at::Tensor& grad_output) {
//     if (!grad_fn) {
//         return;
//     }
//     std::vector<Tensor> grad_inputs(2);
//     grad_fn->apply(grad_output, grad_inputs);
//     for (auto& grad_input : grad_inputs) {
//         grad_input.backward(grad_input.data);
//     }
// }

// Tensor add(const Tensor& tensor1, const Tensor& tensor2) {
//     at::Tensor result_data = tensor1.data + tensor2.data;
//     auto result_grad_fn = std::make_shared<AddBackward0>(); // wrong
//     return Tensor(result_data, result_grad_fn);
// }

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

  //  private:
  std::vector<T> data_;
};

template <typename T>
class AutogradMeta {
 public:
  AutogradMeta() : grad_() {}

  explicit AutogradMeta(std::vector<size_t> dimensions)
      : grad_(Tensor<T>::zeros(dimensions)) {}

  // TODO(yrom1) remove pointless getter setters...
  const Tensor<T> &grad() const { return grad_; }
  Tensor<T> &grad() { return grad_; }

  const Tensor<T> &gradfn() const { return grad_fn_; }
  Tensor<T> &gradfn() { return grad_fn_; }

  //  private:
  Tensor<T> grad_;
  // SPEED oh no! you have to use pointers for abstract class members in c++!
  std::shared_ptr<AutogradFunction<T>> grad_fn_;
  std::vector<std::shared_ptr<Tensor<T>>> children_;
};

template <typename T>
class Tensor {
 public:
  typedef T dtype;

  Tensor()
      : sizes_(),
        storage_(),
        offset_(0),
        requires_grad_(false),
        autograd_meta_(nullptr) {}

  Tensor(std::initializer_list<size_t> dimensions, std::vector<T> values = {},
         bool requires_grad = false)
      : Tensor(std::vector<size_t>(dimensions), std::move(values),
               requires_grad) {}

  Tensor(const std::vector<size_t> &dimensions, std::vector<T> values = {},
         bool requires_grad = false)
      : sizes_(dimensions),
        storage_(
            std::make_shared<Storage<T>>(computeSize(), std::move(values))),
        offset_(0),
        requires_grad_(requires_grad) {
    if (!is_allowed_grad_type() && requires_grad) {
      throw std::runtime_error(
          "requires_grad can only be set to true for float, double, or long "
          "double types");
    }
    if (requires_grad_) {
      autograd_meta_ = std::make_shared<AutogradMeta<T>>(dimensions);
    }
    computeStrides();
  }

  Tensor(std::shared_ptr<Storage<T>> storage, const std::vector<size_t> &sizes,
         const std::vector<size_t> &strides, size_t offset = 0)
      : sizes_(sizes),
        storage_(std::move(storage)),
        offset_(offset),
        strides_(strides) {}

  // Tensor(const Tensor& other)
  //     : sizes_(other.sizes_),
  //       storage_(other.storage_),
  //       offset_(other.offset_),
  //       strides_(other.strides_),
  //       requires_grad_(other.requires_grad_) {
  //   if (other.autograd_meta_) {
  //     autograd_meta_ =
  //     std::make_shared<AutogradMeta<T>>(*other.autograd_meta_);
  //   }
  // }

  static Tensor expand(const T &x, const std::vector<size_t> &dimensions, const bool requires_grad = false) {
    size_t size = computeSizeFromDimensions(dimensions);
    std::vector<T> values(size, static_cast<T>(x));
    return Tensor(dimensions, std::move(values), requires_grad);
  }

  static Tensor ones(const std::vector<size_t> &dimensions, const bool requires_grad = false) {
    size_t size = computeSizeFromDimensions(dimensions);
    std::vector<T> values(size, static_cast<T>(1));
    return Tensor(dimensions, std::move(values), requires_grad);
  }

  static Tensor zeros(const std::vector<size_t> &dimensions, const bool requires_grad = false) {
    size_t size = computeSizeFromDimensions(dimensions);
    std::vector<T> values(size, static_cast<T>(0));
    return Tensor(dimensions, std::move(values), requires_grad);
  }

  // Create a view on the tensor by slicing along a dimension
  // TODO(yrom1) this is broken to my knowledge in terms of not changing
  //      the strides, offsets...
  //      it's also not essential at the moment
  // Tensor slice(size_t dimension, size_t start, size_t end) const {
  //   if (dimension >= sizes_.size()) {
  //     throw std::runtime_error("Dimension out of bounds.");
  //   }
  //   if (start >= end || end > sizes_[dimension]) {
  //     throw std::runtime_error("Invalid slice range.");
  //   }

  //   std::vector<size_t> new_sizes = sizes_;
  //   new_sizes[dimension] = end - start;
  //   size_t new_offset = offset_ + start * strides_[dimension];

  //   return Tensor(storage_, new_sizes, strides_, new_offset);
  // }

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


  // Overloaded binaryOperator for tensors
  Tensor<T> binaryOperator(const Tensor<T> &other, const std::function<T(T, T)> &operation) const {
    // WARNING addition is associative, subtraction is not
    //         you must carefully not flip the order in binaryOperator
    // TODO(yrom1) check this is right for subtraction!
    auto t = applyElementwiseWithBroadcast(operation, other);
    if (t.requires_grad_) {
      t.autograd_meta_.get()->grad_fn_ = std::make_shared<AddBackward0<T>>();
      t.autograd_meta_.get()->children_.push_back(std::make_shared<Tensor<T>>(*this));
      t.autograd_meta_.get()->children_.push_back(std::make_shared<Tensor<T>>(other));
    }
    return t;
  }

  // Overloaded binaryOperator for scalars
  Tensor<T> binaryOperator(const T &scalar, const std::function<T(T, T)> &operation) const {
    auto expandedTensor = Tensor<T>::expand(scalar, (*this).sizes_, true);
    auto t = binaryOperator(expandedTensor, operation);
    return t;
  }

  // Tensor operator+ overload for tensors
  Tensor<T> operator+(const Tensor<T> &other) const {
    return binaryOperator(other, std::plus<T>());
  }

  // Tensor operator+ overload for scalars
  Tensor<T> operator+(const T &scalar) const {
    return binaryOperator(scalar, std::plus<T>());
  }

  Tensor<T> operator-(const Tensor<T> &other) const {
    return applyElementwiseWithBroadcast(other, std::minus<T>());
  }

  Tensor<T> operator-(const T &scalar) const {
    return applyElementwise(scalar, std::minus<T>());
  }

  Tensor<T> operator*(const Tensor<T> &other) const {
    return applyElementwiseWithBroadcast(other, std::multiplies<T>());
  }

  Tensor<T> operator*(const T &scalar) const {
    return applyElementwise(scalar, std::multiplies<T>());
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

  bool requires_grad() const { return requires_grad_; }

  // std::shared_ptr<AutogradMeta<T>> autograd_meta() const { return
  // autograd_meta_; }

  Tensor<T> grad() const {
    if (!requires_grad_) {
      throw std::runtime_error("This tensor does not require gradients.");
    }
    return autograd_meta_->grad();
  }

  void set_grad(const Tensor<T> &grad) {
    if (!requires_grad_) {
      throw std::runtime_error("This tensor does not require gradients.");
    }
    autograd_meta_->grad() = grad;
  }

  //  private:
  std::vector<size_t> sizes_;
  std::shared_ptr<Storage<T>> storage_;
  size_t offset_;
  std::vector<size_t> strides_;
  bool requires_grad_;
  std::shared_ptr<AutogradMeta<T>> autograd_meta_;

  // TODO(yrom1)  we also have to change this so that when you perform an
  // operation on
  // a
  //       tensor with autograd_meta_ set, and do an operation like operator+
  //       we set it's t.autograd_meta_.get()->children_, which we can use a
  //       vector for now
  void backward() {
    /* called on the scalar output tensor calculated in the forward pass
    0) we check this tensor is a scalar
    1) we set the grad_ of this scalar output tensor to [1]
    2) we look for this scalar output tensor's @ grad_fn_ (could be nullptr)
    3) we look for the children of this output tensor @ children_
    4) we call the apply method of the grad_fn_:
        apply(const this tensor (output tensor), children_ tensors (input
    tensors) 5) then, propagate backwards
      ```cpp
      grad_fn->apply(grad_output, grad_inputs);
      for (auto& grad_input : grad_inputs) {
          grad_input.backward(grad_input.data);
      }
      ```
    */
    std::cout << "0" << std::endl;
    if (!is_allowed_grad_type()) {
      throw std::runtime_error(
          "backward can only be called on floating point tensors");
    }
    std::cout << "1" << std::endl;
    if (computeSize() != 1) {
      throw std::runtime_error(
          "backward can only be called on single element tensors");
    }
    std::cout << "2" << std::endl;
    autograd_meta_.get()->grad_ = Tensor<T>::ones(sizes_);
    std::cout << "3" << std::endl;

    std::vector<std::shared_ptr<Tensor<T>>> grad_inputs =
        autograd_meta_.get()->children_;  // Update this line
    std::cout << "4" << std::endl;
    std::cout << autograd_meta_.get()->children_.size() << std::endl;
    for (auto &x : autograd_meta_.get()->children_) {
      std::cout << "children: " << x << std::endl;
    }

    // Pass the grad_ tensor as grad_output to the apply() method

    std::cout << "Before apply()" << std::endl;

    std::cout << "autograd_meta_: " << autograd_meta_.get() << std::endl;
    std::cout << "grad_fn_: " << autograd_meta_.get()->grad_fn_.get()
              << std::endl;

    autograd_meta_.get()->grad_fn_.get()->apply(autograd_meta_.get()->grad_,
                                                grad_inputs);
    std::cout << "After apply()" << std::endl;

    std::cout << "5" << std::endl;

    // Call the backward() method on each grad_input tensor
    // TODO(yrom1) need to think is this correct for more complex scenarios
    for (auto &grad_input : grad_inputs) {
      std::cout << "6" << std::endl;
      if (grad_input->autograd_meta_.get()->children_.size() != 0) {
        grad_input->backward();  // Update this line
      }
    }
  }

  static size_t computeSizeFromDimensions(
      const std::vector<size_t> &dimensions) {
    size_t size = 1;
    for (const auto &dim : dimensions) {
      size *= dim;
    }
    return size;
  }

  constexpr bool is_allowed_grad_type() const {
    return std::is_same<T, float>::value || std::is_same<T, double>::value ||
           std::is_same<T, long double>::value;
  }

  Tensor<T> applyElementwiseWithBroadcast(
      const Tensor<T> &other,
      const std::function<T(const T &, const T &)> &func) const {
    std::vector<size_t> result_sizes = broadcastableShape(sizes_, other.sizes_);

    Tensor<T> result(result_sizes, {},
                     requires_grad() || other.requires_grad());
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

    return Tensor<T>(sizes_, result_values,
                     requires_grad() || other.requires_grad());
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

    return Tensor<T>(sizes_, result_values, requires_grad());
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

  void _print_tensor_graph_helper(
      Tensor<T> &tensor,  // NOLINT (runtime/reference)
      int level,
      std::unordered_set<const Tensor<T> *>
          &visited)  // NOLINT (runtime/reference)
  {                  // NOLINT (whitespace/braces)
    if (visited.find(&tensor) != visited.end()) return;
    visited.insert(&tensor);

    for (int i = 0; i < level; ++i) {
      std::cout << "  ";
    }
    std::cout << "Tensor@" << &tensor;
    if (tensor.autograd_meta_.get()->grad_fn_.get() != nullptr) {
      std::cout << " (" << tensor.autograd_meta_.get()->grad_fn_.get()->op()
                << ")" << std::endl;
    } else {
      // std::string leaf = "🍁";
      // std::cout << leaf << std::endl;
      std::cout << std::endl;
    }

    if (!tensor.autograd_meta_) return;

    for (const auto &child : tensor.autograd_meta_->children_) {
      _print_tensor_graph_helper(*child, level + 1, visited);
    }
  }

  void graph() {
    std::unordered_set<const Tensor<T> *> visited;
    _print_tensor_graph_helper(*this, 0, visited);
  }
};

#endif  // TORCHY_H_
