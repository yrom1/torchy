// Copyright 2023 Ryan Moore
//
// "If you remember just one thing out of this course, it should be this – two
// pieces of advice. The first piece of advice is whenever you can, use vector.
// The second piece of advice is if you cannot, find a way how you can. This is
// to avoid any data structures except arrays... Vector is an array, right. Sort
// of, you say, 'Well... Aren't there exceptions?' – Yes, not for you!"
//
// — Alexander Stepanov

#ifndef TORCHY_HPP_
#define TORCHY_HPP_

#include <iostream>
#include <memory>
#include <vector>

namespace ag {

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<int> size_;
    std::vector<float> data_;
    std::vector<float> grad_;
    std::vector<std::shared_ptr<Tensor>> children_;
    char op_;

    Tensor(std::initializer_list<int> size, std::initializer_list<float> data) : size_(size), data_(data), grad_(_product(size), 0), children_(), op_('?') {};
    Tensor(std::vector<int> size, std::vector<float> data) : size_(size), data_(data), grad_(_product(size), 0), children_(), op_('?') {};

    std::shared_ptr<Tensor> get_shared() {
        return this->shared_from_this();
    }

    void data() { for (auto x : data_) std::cout << x << std::endl; }
    void grad() { for (auto x : grad_) std::cout << x << std::endl; }

private:
    int _product(std::vector<int> size) {
        auto product = 1;
        for (auto s : size) {
            product *= s;
        }
        return product;
    }
};

std::shared_ptr<Tensor> tensor(std::initializer_list<int> size, std::initializer_list<float> data) {
    Tensor t(size, data);
    return std::make_shared<Tensor>(t);
}

std::shared_ptr<Tensor> tensor(std::vector<int> size, std::vector<float> data) {
    Tensor t(size, data);
    return std::make_shared<Tensor>(t);
}

using T = std::shared_ptr<Tensor>;

}

#endif  // TORCHY_HPP_
