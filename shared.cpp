#include <iostream>
#include <memory>

class Foo : public std::enable_shared_from_this<Foo> {
public:
    int value_;

    Foo(int value = 0) : value_(value) {}

    std::shared_ptr<Foo> getSharedFromThis() {
        return shared_from_this();
    }
};
namespace t {
    using Foo = std::shared_ptr<Foo>;
}

int main() {
    t::Foo foo = std::make_shared<Foo>(42);
    t::Foo afoo = foo->getSharedFromThis();
    std::cout << foo->value_ << " " << foo << std::endl;
    std::cout << afoo->value_ << " " << afoo << std::endl;
}

/*
(main) Ryans-Air:torchy ryan$ c++ -std=c++20 shared.cpp && ./a.out
42 0x60000312c258
42 0x60000312c258
*/
