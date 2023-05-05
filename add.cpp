#include <iostream>
#include <memory>
#include <vector>

class Foo : public std::enable_shared_from_this<Foo> {
public:
  int value_;
  std::vector<std::shared_ptr<Foo>> children_;

  Foo(int value = 0) : value_(value) {}

  std::shared_ptr<Foo> binaryAdd(const std::shared_ptr<Foo>& other) {
    children_.push_back(this->shared_from_this());
    children_.push_back(other);
    return std::make_shared<Foo>(this->value_ + other->value_);
  }

  std::shared_ptr<Foo> operator+(const std::shared_ptr<Foo>& other) {
    return binaryAdd(other);
  }

  std::shared_ptr<Foo> operator+(int value) {
    return binaryAdd(std::make_shared<Foo>(value));
  }

  void children() {
    for (const auto & x : children_) {
      std::cout << x->value_ << std::endl; // Print the value of each child
    }
  }
};

int main() {
  auto foo1 = std::make_shared<Foo>(5);
  auto foo2 = std::make_shared<Foo>(7);
  auto foo3 = std::make_shared<Foo>(4);
  auto result = *foo1 + 3;
  auto resultc = *foo1 + foo3;
  std::cout << result->value_ << std::endl;
  std::cout << resultc->value_ << std::endl;
  foo1->children();
}
