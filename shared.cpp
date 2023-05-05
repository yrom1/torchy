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

int main() {
    std::shared_ptr<Foo> fooPtr = std::make_shared<Foo>(42);
    std::shared_ptr<Foo> anotherPtr = fooPtr->getSharedFromThis();
    std::cout << fooPtr->value_ << " " << fooPtr << std::endl;
    std::cout << anotherPtr->value_ << " " << anotherPtr << std::endl;
}

/*
(main) Ryans-Air:torchy ryan$ c++ -std=c++20 shared.cpp && ./a.out
42 0x60000312c258
42 0x60000312c258
*/
