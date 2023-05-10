# torchy🔥

A small tensor-valued autograd engine, inspired by PyTorch and micrograd

```cpp
[cling]$ ag::t a = ag::tensor({2, 2}, {2.0, 3.0, 4.0, 5.0});
[cling]$ ag::t b = ag::tensor({2, 2}, {6.0, 7.0, 8.0, 9.0});
[cling]$ ag::t c = ag::tensor({2, 2}, {10.0, 10.0, 10.0, 10.0});
[cling]$ ag::t d = ag::tensor({2, 2}, {11.0, 11.0, 11.0, 11.0});
[cling]$ ag::t e = (ag::matmul(a, b) + c) * d;
[cling]$ ag::t f = e.get()->sum();
[cling]$ f.get()->backward();
[cling]$ f.get()->data_
(std::vector<float> &) { 2794.00f }
[cling]$ a.get()->grad_
(std::vector<float> &) { 143.000f, 187.000f, 143.000f, 187.000f }
[cling]$ b.get()->grad_
(std::vector<float> &) { 66.0000f, 66.0000f, 88.0000f, 88.0000f }
```

```py
>>> import torch
>>> a = torch.tensor(((2.0, 3.0), (4.0, 5.0)), requires_grad=True)
>>> b = torch.tensor(((6.0, 7.0), (8.0, 9.0)), requires_grad=True)
>>> c = torch.tensor(((10.0, 10.0), (10.0, 10.0)), requires_grad=True)
>>> d = torch.tensor(((11.0, 11.0), (11.0, 11.0)), requires_grad=True)
>>> e = (a.matmul(b) + c) * d
>>> f = e.sum()
>>> f.backward()
>>> f
tensor(2794., grad_fn=<SumBackward0>)
>>> a.grad
tensor([[143., 187.],
        [143., 187.]])
>>> b.grad
tensor([[66., 66.],
        [88., 88.]])
```

## Design

~~The plan is to be similar to PyTorch's internals, particularily the [Variable/Tensor Merge Proposal](https://github.com/pytorch/pytorch/issues/13638) design.~~ The design is a mix of PyTorch and micrograd, with micrograd like members and PyTorch like backward classes with an `apply()` interface.

For simplicity, many features PyTorch has torchy does not, like broadcasting and views. All operations are defined only on `std::shared_ptr<Tensor>`, for now at least.

## Goals

The goal of this project is to learn more about PyTorch's internals, neural networks, and C++.

To do this, I'll gradually add support to torchy for the mathematical operations required to create the expression graph of various neural networks. The long term goals are to implement a Multilayer perceptron by the summer of 2023, and a Transformer by end of the year.

> "Maybe it's a bad idea to have really big ambitions initially. Because the bigger your ambitions, the longer they're going to take to realize, and the longer you're projecting into the future, the more likely you're going to be wrong."
>
> [paulg @ PyCon US 2012](https://youtu.be/R9ITLdmfdLI?t=1927)

## Running tests

Taking inspiration from [micrograd's tests](https://github.com/karpathy/micrograd/blob/master/test/test_engine.py), we will use [PyTorch's C++ frontend](https://pytorch.org/cppdocs/frontend.html) for high level sanity checks using GoogleTest.

To run the tests use:

```sh
sh test.sh
```

Running the tests requires: `cmake`, `make`, `torch` installed (on the version of Python accessed by the `python` command), `git`, and a C++ compiler. Note that these requirements are only for when you need to run the tests, otherwise except the C++ compiler they are not needed.
