#include <iostream>
#include <memory>
#include <vector>

class Foo;

class FooWrapper {
public:
    static std::shared_ptr<Foo> create(int value = 0);
};

class Foo : public std::enable_shared_from_this<Foo> {
public:
    int value_;
    std::vector<std::shared_ptr<Foo>> children_;

    Foo(int value = 0) : value_(value) {}

    void addChild(const std::shared_ptr<Foo>& other) {
        children_.push_back(other);
    }

    void printChildren() const {
        std::cout << "Node value: " << value_ << ", children: ";
        for (const auto& child : children_) {
            std::cout << child->value_ << " ";
            std::cout << child.get() << " ";
        }
        std::cout << std::endl;
    }

    // Friend functions to enable operator overloading
    friend std::shared_ptr<Foo> operator+(const std::shared_ptr<Foo>& lhs, const std::shared_ptr<Foo>& rhs);
    friend std::shared_ptr<Foo> operator+(const std::shared_ptr<Foo>& lhs, int value);
    friend std::shared_ptr<Foo> operator+(int value, const std::shared_ptr<Foo>& rhs);
};

std::shared_ptr<Foo> FooWrapper::create(int value) {
    return std::make_shared<Foo>(value);
}

// Overloaded operator+ functions
std::shared_ptr<Foo> operator+(const std::shared_ptr<Foo>& lhs, const std::shared_ptr<Foo>& rhs) {
    auto result = FooWrapper::create(lhs->value_ + rhs->value_);
    result->addChild(lhs);
    result->addChild(rhs);
    return result;
}

std::shared_ptr<Foo> operator+(const std::shared_ptr<Foo>& lhs, int value) {
    auto result = FooWrapper::create(lhs->value_ + value);
    result->addChild(lhs);
    return result;
}

std::shared_ptr<Foo> operator+(int value, const std::shared_ptr<Foo>& rhs) {
    auto result = FooWrapper::create(value + rhs->value_);
    result->addChild(rhs);
    return result;
}

int main() {
    auto foo1 = FooWrapper::create(1);
    auto foo2 = FooWrapper::create(2);

    auto sum1 = foo1 + foo2;
    auto sum2 = foo1 + FooWrapper::create(42);
    auto sum3 = FooWrapper::create(42) + foo1;
    auto sum4 = foo1 + 1;
    auto sum5 = 1 + foo1;

    std::cout << "foo1 + foo2: " << sum1->value_ << std::endl;
    sum1->printChildren();
    std::cout << "foo1 + Foo(42): " << sum2->value_ << std::endl;
    sum2->printChildren();
    std::cout << "Foo(42) + foo1: " << sum3->value_ << std::endl;
    sum3->printChildren();
    std::cout << "foo1 + 1: " << sum4->value_ << std::endl;
    sum4->printChildren();
    std::cout << "1 + foo1: " << sum5->value_ << std::endl;
    sum5->printChildren();

    return 0;
}
