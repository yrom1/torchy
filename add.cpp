/**
 * The class Foo represents a simple integer value with a vector of child Foo objects.
 * This class provides three different operator+ methods for adding two Foo objects or an integer value
 * to a Foo object:
 *
 * - operator+(const std::shared_ptr<Foo>& other):
 *   Adds two Foo objects together and returns a shared_ptr<Foo> that represents the sum.
 *   Used when adding two Foo objects together.
 *
 * - operator+(int value):
 *   Adds an integer value to a Foo object and returns a shared_ptr<Foo> that represents the sum.
 *   Used when adding an integer value to a Foo object.
 *
 * - operator+(const Foo& other):
 *   Adds two Foo objects together using the + operator and returns a shared_ptr<Foo> that represents the sum.
 *   Used when adding two Foo objects together using the + operator.
 *
 * In each of these methods, a new Foo object is created to represent the sum of the original Foo object(s)
 * and the input parameter(s), and both the original Foo object(s) and the new Foo object are added to the
 * child vector of the first Foo object.
 */

#include <iostream>
#include <memory>
#include <vector>

class Foo : public std::enable_shared_from_this<Foo> {
public:
    int value_;
    std::vector<std::shared_ptr<Foo>> children_;

    Foo(int value = 0) : value_(value) {}

    std::shared_ptr<Foo> binaryAdd(const std::shared_ptr<Foo>& other) {
        children_.push_back(other);
        return std::make_shared<Foo>(value_ + other->value_);
    }

    std::shared_ptr<Foo> operator+(const std::shared_ptr<Foo>& other) {
        return binaryAdd(other);
    }

    std::shared_ptr<Foo> operator+(int value) {
        Foo tmp(value);
        return binaryAdd(std::make_shared<Foo>(tmp));
    }

    std::shared_ptr<Foo> operator+(const Foo& other) {
        return *this + std::make_shared<Foo>(other);
    }

    void children() {
        for (const auto& x : children_) {
            std::cout << x << std::endl;
            std::cout << x->value_ << std::endl;
            for (auto x : x->children_) {
              std::cout << "" << std::endl;
            }
        }
    }
};

std::shared_ptr<Foo> operator+(int value, Foo& other) {
    Foo tmp(value);
    return other.binaryAdd(std::make_shared<Foo>(tmp));
}

int main() {
    Foo foo1(5);
    Foo foo2(7);
    Foo foo3(4);

    auto result = foo1 + 3;
    auto resultc = foo1 + foo3;
    auto resulti = foo1 + Foo(3);
    auto resultl = 2 + foo1;

    std::cout << "Foo(5) + 3" << std::endl;
    std::cout << result->value_ << std::endl;
    std::cout << "Foo(5) + Foo(4)" << std::endl;
    std::cout << resultc->value_ << std::endl;
    std::cout << "Foo(5) + Foo(3)" << std::endl;
    std::cout << resulti->value_ << std::endl;
    std::cout << "children" << std::endl;
    foo1.children();
}
