#include <iostream>
#include <vector>

class Node {
public:
    int value_;
    std::vector<Node*> children_;

    Node(int value = 0) : value_(value) {}

    void addChild(Node* child) {
        children_.push_back(child);
    }

    void printChildren() const {
        for (const auto& child : children_) {
            std::cout << "Value: " << child->value_ << " Address: " << child << std::endl;
        }
    }
};

int main() {
    Node node1(1);
    Node node2(2);
    Node node3(3);
    Node node4(4);

    node1.addChild(&node2);
    node1.addChild(&node3);
    node1.addChild(&node4);
    node1.addChild(&node3);
    node2.addChild(&node3);

    node1.printChildren();
    std::cout << "Address of Node1 instance: " << &node1 << std::endl;
    std::cout << "Address of Node2 instance: " << &node2 << std::endl;
    std::cout << "Address of Node3 instance: " << &node3 << std::endl;
    std::cout << "Address of Node4 instance: " << &node4 << std::endl;

    return 0;
}
