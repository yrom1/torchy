# torchy

A small tensor-valued autograd engine, inspired by PyTorch and micrograd.

## Hello torchy

```c++
#include <iostream>

#include "torchy.h"

int main() {
  using namespace std;
  Tensor<int> t({1}, {42}); // Tensor of 1 element holding 42
  cout << t; // [42]
}
```

The next examples use a C++ interpreter called [Cling](https://github.com/root-project/cling) ([`brew install cling`](https://formulae.brew.sh/formula/cling)).

```c++
[cling]$ #include "torchy.h"
[cling]$ Tensor<int> t({2,2}, {0,1,2,3});
[cling]$ using namespace std;
[clnig]$ cout << t << endl;
[[0, 1],
 [2, 3]]
```

## Design

The plan is to be similar to PyTorch's internals, particularily the [Variable/Tensor Merge Proposal](https://github.com/pytorch/pytorch/issues/13638) design.

```
Tensor<T> has-a [
  vector<size_t> sizes_;
  shared_ptr<Storage<T>> storage_;
  size_t offset_;
  vector<size_t> strides_;
]

Storage<T> has-a [
  vector<T> data_;
]
```
## Mathematical Operations

```c++
[cling]$ cout << t << endl;
[[0, 1],
 [2, 3]]
[cling]$ cout << t + t << endl;
[[0, 2],
 [4, 6]]
[cling]$ cout << t - t << endl;
[[0, 0],
 [0, 0]]
[cling]$ cout << t / t << endl;
[[0, 1],
 [1, 1]]
[cling]$ cout << t * t << endl; // Hadamard product
[[0, 1],
 [4, 9]]
[cling]$ cout << t.matmul(t) << endl;
[[2, 3],
 [6, 11]]
```

## Goals

The goal of this project is to learn more about PyTorch's internals, neural networks, and C++.

To do this, I'll gradually add support to torchy for the mathematical operations required to create the expression graph of various neural networks. The long term goals are to implement a Multilayer perceptron by the summer of 2023, and a Transformer by end of the year.

## Running tests

Taking inspiration from [`micrograd`'s tests](https://github.com/karpathy/micrograd/blob/master/test/test_engine.py), we will use [PyTorch's C++ frontend](https://pytorch.org/cppdocs/frontend.html) for high level sanity checks using GoogleTest.

To run the tests use:

```sh
sh test.sh
```

Running the tests requires: `cmake`, `torch` installed (on the version of Python accessed by the `python` command), `git`, and a C++ compiler. Note that these requirements are only for when you need to run the tests, otherwise except the C++ compiler they are not needed.
