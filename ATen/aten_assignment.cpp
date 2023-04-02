#include <iostream>
#include <ATen/ATen.h>

int main() {
    // Create a tensor of size 2x2 with zeros
    at::Tensor tensor = at::zeros({2, 2});

    // Set some values in the tensor
    tensor[0][0] = 1;
    tensor[0][1] = 2;
    tensor[1][0] = 3;
    tensor[1][1] = 4;

    // Print the tensor
    std::cout << "Tensor:\n" << tensor << std::endl;

    return 0;
}
