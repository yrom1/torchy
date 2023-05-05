#include <iostream>
#include <memory>
#include <vector>

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
};

class FooWrapper {
public:
    static std::shared_ptr<Foo> create(int value = 0) {
        return std::make_shared<Foo>(value);
    }
};

int main() {
    auto node1 = FooWrapper::create(1);
    auto node2 = FooWrapper::create(2);
    auto node3 = FooWrapper::create(3);
    auto node4 = FooWrapper::create(4);
    auto node5 = FooWrapper::create(5);

    node1->addChild(node2);
    node1->addChild(node3);
    node1->addChild(node2);
    node2->addChild(node4);
    node2->addChild(node5);

    node1->printChildren();
    node2->printChildren();
    node3->printChildren();
    node4->printChildren();
    node5->printChildren();

    return 0;
}
