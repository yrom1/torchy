#include <iostream>
#include <memory>

namespace ag {

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    int value_;

    using dtype = std::shared_ptr<Tensor>;

    Tensor(int value = 0) : value_(value) {}

    std::shared_ptr<Tensor> getSharedFromThis() {
        return shared_from_this();
    }
};

using T = std::shared_ptr<Tensor>;

}

int main() {
    ag::T foo = std::make_shared<ag::Tensor>(42);
    ag::T bar = foo->getSharedFromThis();
    std::cout << foo->value_ << " " << foo << std::endl;
    std::cout << bar->value_ << " " << bar << std::endl;
}

/*
(main) Ryans-Air:torchy ryan$ c++ -std=c++20 shared.cpp && ./a.out
42 0x60000312c258
42 0x60000312c258
*/
