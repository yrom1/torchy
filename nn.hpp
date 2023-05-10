#include "torchy.hpp"

class Neuron {
  public:
    ag::t weights_;
    ag::t bias_;

    Neuron(ag::t weights, ag::t bias) {
        weights_ = weights;
        bias_ = bias;
    }

    ag::t forward(ag::t input) {
        // z = Wx + b
        ag::t z = ag::matmul(input, weights_) + bias_; // order right?
        return ag::relu(z);
    }
};
