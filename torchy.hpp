#include <iostream>
#include <memory>

namespace ag {

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    float value_;

    Tensor(float value = 0) : value_(value) {}

    std::shared_ptr<Tensor> get_shared() {
        return this->shared_from_this();
    }
};

using T = std::shared_ptr<Tensor>;

T tensor(float value) {
    return std::make_shared<Tensor>(value);
}

}
