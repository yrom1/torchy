#include <iostream>
#include <ATen/ATen.h>

int main() {
    // Create a Tensor using aten
    at::Tensor a = at::ones({2, 3}, at::kFloat);

    // Access TensorImpl
    at::TensorImpl* a_impl = a.unsafeGetTensorImpl();

    // Perform operations on the Tensor
    at::Tensor b = a * 2;

    // Access TensorImpl of the result
    at::TensorImpl* b_impl = b.unsafeGetTensorImpl();

    // Print the TensorImpl pointers
    std::cout << "Tensor a impl pointer: " << a_impl << std::endl;
    std::cout << "Tensor b impl pointer: " << b_impl << std::endl;

    // Print the contents of the tensors
    std::cout << "Tensor a:\n" << a << std::endl;
    std::cout << "Tensor b:\n" << b << std::endl;

    return 0;
}
