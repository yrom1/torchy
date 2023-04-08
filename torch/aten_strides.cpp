#include <ATen/ATen.h>
#include <iostream>

int main() {
    at::Tensor x = at::randn({3, 4, 5});
    auto strides = x.strides();

    std::cout << "Strides: ";
    for (auto stride : strides) {
        std::cout << stride << ", ";
    }
    std::cout << std::endl;

    return 0;
}
