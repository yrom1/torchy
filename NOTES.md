# static vs dynamic autodiff

![](https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/dynamic_graph.gif)

> Dynamic Neural Networks: Tape-Based Autograd
> PyTorch has a unique way of building neural networks: using and replaying a tape recorder.
>
> Most frameworks such as TensorFlow, Theano, Caffe, and CNTK have a static view of the world. One has to build a neural network and reuse the same structure again and again. Changing the way the network behaves means that one has to start from scratch.
>
> With PyTorch, we use a technique called reverse-mode auto-differentiation, which allows you to change the way your network behaves arbitrarily with zero lag or overhead. Our inspiration comes from several research papers on this topic, as well as current and past work such as torch-autograd, autograd, Chainer, etc.
>
> While this technique is not unique to PyTorch, it's one of the fastest implementations of it to date. You get the best of speed and flexibility for your crazy research.
>
[PyTorch REAMDE](https://github.com/yrom1/pytorch#dynamic-neural-networks-tape-based-autograd)

I understand how pytorch is tape based, making the graph on the fly. but does that mean tensorflow or other neural network librarys make the graphs statically? how exactly does that work, how would you implement autodiff in such a way you *couldnt* create the expression graph on demand?

---

ChatGPT
In some machine learning frameworks like TensorFlow (prior to version 2.0), Theano, and CNTK, computational graphs are defined statically before execution. That means, the structure of the graph, which includes the operations and the relationships between them, must be fully specified and compiled before any actual computation or data flow happens.

Here's a basic example to illustrate the static approach. Let's say you're building a simple neural network with one hidden layer. In a static graph framework, you'd specify the entire structure upfront: the input layer, the weights, the hidden layer, its activation function, the output layer, the loss function, etc. Only after you've defined this entire structure, you can feed in the input data.

If you want to change the structure of the network - say, add another hidden layer - you'd need to define a new graph from scratch. This can be more efficient for certain types of large-scale computations, because the system can optimize the entire computation ahead of time, but it's less flexible and less intuitive, particularly for research and development where you're often experimenting with different architectures.

In contrast, frameworks like PyTorch use dynamic computation graphs, also known as define-by-run approach. The graph structure can be modified as the computations are being performed. Each operation is added to the graph as you go along. This makes it more flexible and intuitive to work with, particularly for complex architectures that have control flow structures like loops and conditional statements.

Automatic differentiation (autodiff) works with both static and dynamic graphs, but the implementation details differ. In a static graph, the graph's structure is known ahead of time, so it's possible to compute the derivatives for the entire graph in one go. This is done by applying the chain rule of calculus to each operation in the graph, going backwards from the output to the input (hence "reverse-mode autodiff").

In a dynamic graph, the derivatives are computed on-the-fly as operations are added to the graph. Each operation has a forward pass (computing the output) and a backward pass (computing the derivative). This allows PyTorch's autograd system to compute gradients dynamically as you're building the graph.

It's worth noting that TensorFlow 2.0 and later introduced eager execution, which allows dynamic computation graphs similar to PyTorch. This was a move to make TensorFlow more user-friendly and flexible, while still allowing the option to use static graphs for performance-critical applications.

# forward declare of private class member and outside definition works

```c++
[cling]$ l.get()->_backward()
input_line_19:2:11: error: '_backward' is a private member of 'cg::Tensor'
 l.get()->_backward()
          ^
./torchy.hpp:179:8: note: declared private here
  void _backward();
       ^
```

# dont move from r values what is copy elipson?

```cpp
/Users/ryan/torchy/torchy.hpp:99:38: warning: moving a temporary object prevents copy elision [-Wpessimizing-move]
      lhs.get()->size_, result_data, std::move(std::make_unique<AddBackward>()),
                                     ^
/Users/ryan/torchy/torchy.hpp:99:38: note: remove std::move call here
      lhs.get()->size_, result_data, std::move(std::make_unique<AddBackward>()),
                                     ^~~~~~~~~~
```

# shared_ptr minimal example

```cpp
#include <iostream>
#include <memory>

class Foo : public std::enable_shared_from_this<Foo> {
public:
    int value_;

    Foo(int value = 0) : value_(value) {}

    std::shared_ptr<Foo> getSharedFromThis() {
        return shared_from_this();
    }
};

int main() {
    std::shared_ptr<Foo> fooPtr = std::make_shared<Foo>(42);
    std::shared_ptr<Foo> anotherPtr = fooPtr->getSharedFromThis();
    std::cout << fooPtr->value_ << " " << fooPtr << std::endl;
    std::cout << anotherPtr->value_ << " " << anotherPtr << std::endl;
}

/*
(main) Ryans-Air:torchy ryan$ c++ -std=c++20 shared.cpp && ./a.out
42 0x60000312c258
42 0x60000312c258
*/
```

# building cling from source

```
(master) Ryans-Air:packaging ryan$ chmod +x cpt.py
(master) Ryans-Air:packaging ryan$
(master) Ryans-Air:packaging ryan$ ./cpt.py --check-requirements && ./cpt.py --create-dev-env Debug --with-workdir=./cling-build/

Cling Packaging Tool (CPT)
Arguments vector: ['./cpt.py', '--check-requirements']

╔══════════════════════════════════════════════════════════════════════════════╗
║ cling (arm64)                                Wed, 03 May 2023 01:24:14 -0400 ║
╚══════════════════════════════════════════════════════════════════════════════╝
Thread Model: POSIX
Operating System: Darwin
Distribution: MacOSX
Release: 22.3.0
Revision: 13.2.1
Architecture: arm64

+-----------------------------------------------------------------------------+
| Check availability of required softwares                                    |
+-----------------------------------------------------------------------------+
git                 [OK]
cmake               [OK]
clang               [OK]
clang++             [OK]
subprocess.CalledProcessError: Command 'type -p zlib*' returned non-zero exit status 1
```

```
+-----------------------------------------------------------------------------+
| Building cling (using 8 cores)                                              |
+-----------------------------------------------------------------------------+
```

# enabled shared from this

```cpp
#include <iostream>
#include <memory>

class MyClass : public std::enable_shared_from_this<MyClass> {
public:
    MyClass() {
        std::cout << "MyClass constructor called" << std::endl;
    }
    ~MyClass() {
        std::cout << "MyClass destructor called" << std::endl;
    }

    std::shared_ptr<MyClass> getSharedPtr() {
        return shared_from_this();
    }
};

int main() {
    std::shared_ptr<MyClass> ptr1 = std::make_shared<MyClass>();
    std::shared_ptr<MyClass> ptr2 = ptr1->getSharedPtr();

    std::cout << "ptr1 address: " << ptr1.get() << std::endl;
    std::cout << "ptr2 address: " << ptr2.get() << std::endl;

    return 0;
}
```

```
MyClass constructor called
ptr1 address: 0x6000009dc258
ptr2 address: 0x6000009dc258
MyClass destructor called
```

# c = (a * b) - ((b / a) + b) bug

```c++
[cling]$ Tensor<float> a({1}, {2.0}, true)
(Tensor<float> &) @0x109b60460
[cling]$ Tensor<float> b({1}, {43.0}, true)
(Tensor<float> &) @0x1098ec8e0
[cling]$ auto c = (a * b) - ((b / a) + b)
(Tensor<float> &) @0x109ba0ae0
[cling]$ c.graph()
Tensor@0x109ba0ae0 (-)
  Tensor@0x6000013c1998 (*)
    Tensor@0x6000013c1598
    Tensor@0x6000013c1718
  Tensor@0x6000013c1918 (+)
    Tensor@0x6000013c1798 (/)
      Tensor@0x6000013c1618
      Tensor@0x6000013c1698
    Tensor@0x6000013c1a18
[cling]$ c.backward()
[cling]$ a.autograd_meta_.get()->grad_.storage_.get()->data_
(std::vector<float> &) { 32.2500f }
[cling]$ b.autograd_meta_.get()->grad_.storage_.get()->data_
(std::vector<float> &) { 3.50000f }
```
```py
Python 3.11.1 (main, Apr 19 2023, 18:41:42) [Clang 14.0.3 (clang-1403.0.22.14.1)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> a = torch.tensor((2.0), requires_grad=True)
>>> b = torch.tensor((43.0), requires_grad=True)
>>> c = (a * b) - ((b / a) + b)
>>> c.backward()
>>> a.grad
tensor(53.7500)
>>> b.grad
tensor(0.5000)
```

# div example working scalar

```py
>>> import torch
>>> a = torch.tensor((2.0), requires_grad=True)
>>> b = torch.tensor((43.0), requires_grad=True)
>>> c = a / b
>>> c.backward()
>>> a.grad
tensor(0.0233)
>>> b.grad
tensor(-0.0011)
>>>
(main) Ryans-Air:torchy ryan$ sh cling.sh
[cling]$ Tensor<float> a({1}, {2.0}, true)
(Tensor<float> &) @0x109adc460
[cling]$ Tensor<float> b({1}, {43.0}, true)
(Tensor<float> &) @0x1098688e8
[cling]$ auto c = a / b
tensor binop 0
tensor binop 1
tensor binop 2
tensor binop 3
(Tensor<float> &) @0x109b0c6a0
[cling]$ c.backward()
0
1
2
3
4
2
children: 0x600001aa8618
children: 0x600001aa8998
Before apply()
autograd_meta_: 0x6000005790f8
grad_fn_: 0x600003435658
After apply()
5
6
6
[cling]$ a.autograd_meta_.get()->grad_.storage_.get()->data_
(std::vector<float> &) { 0.0232558f }
[cling]$ b.autograd_meta_.get()->grad_.storage_.get()->data_
(std::vector<float> &) { -0.00108167f }
```

# mul example working scalar

```py
>>> b = torch.tensor((2.0), requires_grad=True)
>>> a = torch.tensor((43.0), requires_grad=True)
>>> c = a * b
>>> c.backward()
>>> a.grad
tensor(2.)
>>> b.grad
tensor(43.)
>>>
```
```cpp
(main) Ryans-Air:torchy ryan$ sh cling.sh
[cling]$ Tensor<float> a({1}, {43.0}, true)
(Tensor<float> &) @0x105970460
[cling]$ Tensor<float> b({1}, {2.0}, true)
(Tensor<float> &) @0x1056fc8e8
[cling]$ auto c = a * b
tensor binop 0
tensor binop 1
tensor binop 2
tensor binop 3
(Tensor<float> &) @0x1059a0690
[cling]$ c.backward()
0
1
2
3
4
2
children: 0x600003b1c098
children: 0x600003b1c118
Before apply()
autograd_meta_: 0x6000024a17d8
grad_fn_: 0x6000015a4038
After apply()
5
6
6
[cling]$ c.graph()
Tensor@0x1059a0690 (*)
  Tensor@0x600003b1c098
  Tensor@0x600003b1c118
[cling]$ a.autograd_meta_.get()->grad_.storage_.get()->data_
(std::vector<float> &) { 2.00000f }
[cling]$ b.autograd_meta_.get()->grad_.storage_.get()->data_
(std::vector<float> &) { 43.0000f }
```

# c = a + b vs b = a + a

```py
>>> a = torch.tensor((1.0), requires_grad=True)
>>> a
tensor(1., requires_grad=True)
>>> b = a + a
>>> b
tensor(2., grad_fn=<AddBackward0>)
>>> b.backward()
>>> a.grad
tensor(2.)
>>> a = torch.tensor((1.0), requires_grad=True)
>>> b = torch.tensor((1.0), requires_grad=True)
>>> c = a + b
>>> c
tensor(2., grad_fn=<AddBackward0>)
>>> c.backward()
>>> c
tensor(2., grad_fn=<AddBackward0>)
>>> a.grad
tensor(1.)
>>> b.grad
tensor(1.)
```

Wow we're actually handling it correctly right now must of been by accident:

```
[cling]$ Tensor<float> a({1},{1},true)
(Tensor<float> &) @0x108f44460
[cling]$ a
(Tensor<float> &) @0x108f44460
[cling]$ auto b = a + a
tensor binop 0
tensor binop 1
tensor binop 2
tensor binop 3
(Tensor<float> &) @0x108f70690
[cling]$ b.backward()
0
1
2
3
4
2
children: 0x600001ee3f98
children: 0x600001ee0598
Before apply()
autograd_meta_: 0x6000001f80b8
grad_fn_: 0x6000030b6a18
After apply()
5
6
6
[cling]$ a
(Tensor<float> &) @0x108f44460
[cling]$ a.autograd_meta_
(std::shared_ptr<AutogradMeta<float> > &) std::shared_ptr -> 0x6000001a4ab8
[cling]$ a.autograd_meta_.get()
(std::shared_ptr<AutogradMeta<float> >::element_type *) 0x6000001a4ab8
[cling]$ a.autograd_meta_.get()->grad_
(Tensor<float> &) @0x6000001a4ab8
[cling]$ a.autograd_meta_.get()->grad_.storage_
(std::shared_ptr<Storage<float> > &) std::shared_ptr -> 0x600003ee4078
[cling]$ a.autograd_meta_.get()->grad_.storage_.get()
(std::shared_ptr<Storage<float> >::element_type *) 0x600003ee4078
[cling]$ a.autograd_meta_.get()->grad_.storage_.get()->data_
(std::vector<float> &) { 2.00000f }
```

Maybe make a grad method? This is kinda annoying to type out.

# Derivatives of vector element-wise binary operators

I think my `AddBackward` should be identity not torch.ones? Not sure yet.

![](https://explained.ai/matrix-calculus/images/blkeqn-71B7180689F0C8BB7D610CFD5C41CCD8.svg)

![](https://explained.ai/matrix-calculus/images/blkeqn-B50DDF4EB7C7148BD5EED63AE5B83271.svg)

# Jacobian of identity

> The Jacobian of the identity function , with , has n functions and each function has n parameters held in a single vector x. The Jacobian is, therefore, a square matrix since :

![](https://explained.ai/matrix-calculus/images/latex-FEFCAA3AA3D0B051E4960CE52D92A7C0.svg)

![](https://explained.ai/matrix-calculus/images/latex-A936169BDEA3E18A4B377A21C743CB21.svg)

I get it!

# graph operations

```
[cling]$ Tensor<float> a({1}, {42.0}, true);
[cling]$ Tensor<float> b({1}, {42.0}, true);
[cling]$ auto c = a + b;
operator+ grad_fn_ before: 0x0
operator+ grad_fn_ after: 0x600002f502f8
[cling]$ c
(Tensor<float> &) @0x1072f04d8
[cling]$ c.backward()
0
1
2
3
4
2
children: 0x6000001a6398
children: 0x6000001a7d98
Before apply()
autograd_meta_: 0x600001e40d38
grad_fn_: 0x600002f502f8
Inside AddBackward0::apply before
input vec size: 2
-1
[1]
size: 1
1
a0
a1
a2
a0
a1
a2
Inside AddBackward0::apply after
After apply()
5
6
6
[cling]$ c
(Tensor<float> &) @0x1072f04d8
[cling]$ print_tensor_graph(c)
Tensor@0x1072f04d8 (+)
  Tensor@0x6000001a6398
  Tensor@0x6000001a7d98
```

# graph
```cpp
(main) Ryans-MacBook-Air:torchy ryan$ sh cling.sh
[cling]$ Tensor<float> a({1}, {42.0}, true);  Tensor<float> b({1}, {42.0}, true); Tensor<float> c({1}, {42.0}, true);  Tensor<float> d({1}, {42.0}, true);
[cling]$ auto x = a + b
operator+ grad_fn_ before: 0x0
operator+ grad_fn_ after: 0x600002acc998
(Tensor<float> &) @0x10700c5c8
[cling]$ auto y = c + d
operator+ grad_fn_ before: 0x0
operator+ grad_fn_ after: 0x600002ac7a38
(Tensor<float> &) @0x1070284d8
[cling]$ auto l = x + y
operator+ grad_fn_ before: 0x0
operator+ grad_fn_ after: 0x600002ac7938
(Tensor<float> &) @0x1070444d8
[cling]$ print_tensor_graph(l)
Tensor@0x1070444d8
  Tensor@0x600000463818
    Tensor@0x6000004a1698
    Tensor@0x6000004a1818
  Tensor@0x600000461d98
    Tensor@0x600000456118
    Tensor@0x600000455d98
[cling]$ .q
```

# auto l = a + b + c + d
`:)`
```
(main) Ryans-MacBook-Air:torchy ryan$ sh cling.sh
[cling]$ Tensor<float> a({1}, {42.0}, true);  Tensor<float> b({1}, {42.0}, true); Tensor<float> c({1}, {42.0}, true);  Tensor<float> d({1}, {42.0}, true);
[cling]$ auto l = a + b + c + d
operator+ grad_fn_ before: 0x0
operator+ grad_fn_ after: 0x6000036480f8
operator+ grad_fn_ before: 0x0
operator+ grad_fn_ after: 0x600003648218
operator+ grad_fn_ before: 0x0
operator+ grad_fn_ after: 0x6000036481b8
(Tensor<float> &) @0x1050b05e0
[cling]$ l.autograd_meta_.get()->storage_.get()->data_
input_line_7:2:26: error: no member named 'storage_' in 'AutogradMeta<float>'
 l.autograd_meta_.get()->storage_.get()->data_
 ~~~~~~~~~~~~~~~~~~~~~~  ^
[cling]$ l.autograd_meta_.get()->grad_.get()->storage_.get()->data_
input_line_8:2:32: error: no member named 'get' in 'Tensor<float>'
 l.autograd_meta_.get()->grad_.get()->storage_.get()->data_
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ^
[cling]$ l.autograd_meta_.get()->grad_.storage_.get()->data_
(std::vector<float> &) { 0.00000f }
[cling]$ l.backward()
0
1
2
3
4
2
children: 0x600001801218
children: 0x600001801018
Before apply()
autograd_meta_: 0x6000007712d8
grad_fn_: 0x6000036481b8
Inside AddBackward0::apply before
input vec size: 2
-1
[1]
size: 1
1
a0
a1
a2
a0
a1
a2
Inside AddBackward0::apply after
After apply()
5
6
0
1
2
3
4
2
children: 0x600001801618
children: 0x600001801e18
Before apply()
autograd_meta_: 0x600000770ab8
grad_fn_: 0x600003648218
Inside AddBackward0::apply before
input vec size: 2
-1
[1]
size: 1
1
a0
a1
a2
a0
a1
a2
Inside AddBackward0::apply after
After apply()
5
6
0
1
2
3
4
2
children: 0x600001801718
children: 0x600001801798
Before apply()
autograd_meta_: 0x600000770dd8
grad_fn_: 0x6000036480f8
Inside AddBackward0::apply before
input vec size: 2
-1
[1]
size: 1
1
a0
a1
a2
a0
a1
a2
Inside AddBackward0::apply after
After apply()
5
6
6
6
6
[cling]$ l.autograd_meta_.get()->grad_.storage_.get()->data_
(std::vector<float> &) { 1.00000f }
[cling]$ a.autograd_meta_.get()->grad_.storage_.get()->data_
(std::vector<float> &) { 1.00000f }
[cling]$ b.autograd_meta_.get()->grad_.storage_.get()->data_
(std::vector<float> &) { 1.00000f }
[cling]$ c.autograd_meta_.get()->grad_.storage_.get()->data_
(std::vector<float> &) { 1.00000f }
[cling]$ d.autograd_meta_.get()->grad_.storage_.get()->data_
(std::vector<float> &) { 1.00000f }
```

```c
[cling]$ Tensor<float> a({1}, {42.0}, true);  Tensor<float> b({1}, {42.0}, true);
[cling]$ a
(Tensor<float> &) @0x10761c478
[cling]$ b
(Tensor<float> &) @0x10761c4d8
[cling]$ auto c = a + b
(Tensor<float> &) @0x107644478
[cling]$ c
(Tensor<float> &) @0x107644478
[cling]$ c.storage_.get()
(std::shared_ptr<Storage<float> >::element_type *) 0x6000021028c8
[cling]$ c.storage_.get().data_
input_line_11:2:18: error: member reference type 'std::__1::shared_ptr<Storage<float> >::element_type *' (aka 'Storage<float> *') is
      a pointer; did you mean to use '->'?
 c.storage_.get().data_
 ~~~~~~~~~~~~~~~~^
                 ->
[cling]$ c.storage_.get()->data_
(std::vector<float> &) { 84.0000f }
[cling]$ c.autograd_meta_.get()->storage_.get()->data_
input_line_13:2:26: error: no member named 'storage_' in 'AutogradMeta<float>'
 c.autograd_meta_.get()->storage_.get()->data_
 ~~~~~~~~~~~~~~~~~~~~~~  ^
[cling]$ c.autograd_meta_.get()->grad_.get()->storage_.get()->data_
input_line_14:2:32: error: no member named 'get' in 'Tensor<float>'
 c.autograd_meta_.get()->grad_.get()->storage_.get()->data_
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ^
[cling]$ c.autograd_meta_.get()->grad_.storage_.get()->data_
(std::vector<float> &) { 0.00000f }
```

# shared_ptr

```
[cling]$ Tensor<float> t({1}, {42}, true)
(Tensor<float> &) @0x106c9c858
[cling]$ t
(Tensor<float> &) @0x106c9c858
[cling]$ t.autograd_meta_
(std::shared_ptr<AutogradMeta<float> > &) std::shared_ptr -> 0x6000022559f8
[cling]$ t.autograd_meta_.get()
(std::shared_ptr<AutogradMeta<float> >::element_type *) 0x6000022559f8
[cling]$ t.autograd_meta_.get()->grad_
(Tensor<float> &) @0x6000022559f8
[cling]$ t.autograd_meta_.get()->grad_fn_
(std::shared_ptr<Function<float> > &) std::shared_ptr -> nullptr
[cling]$ t.autograd_meta_.get()->grad_fn_ = std::make_shared<AddBackward0<float>>()
(std::shared_ptr &) std::shared_ptr -> 0x6000010b0eb8
```

unique pointer broken in cling? idk using shared

---

tldr use `.get()` for raw pointer

```
(communism) Ryans-MacBook-Air:torchy ryan$ sh cling.sh
^[[A[cling]$ Tensor<float> t({1},{42},true);
[cling]$ t
(Tensor<float> &) @0x1079fc420
[cling]$ t.autograd_meta_
(std::shared_ptr<AutogradMeta<float> > &) std::shared_ptr -> 0x6000000b6618
[cling]$ t.autograd_meta_->data()
input_line_8:2:20: error: no member named 'data' in 'AutogradMeta<float>'
 t.autograd_meta_->data()
 ~~~~~~~~~~~~~~~~  ^
[cling]$ t.autograd_meta_.grad_
input_line_9:2:19: error: no member named 'grad_' in 'std::__1::shared_ptr<AutogradMeta<float> >'
 t.autograd_meta_.grad_
 ~~~~~~~~~~~~~~~~ ^
[cling]$ t.autograd_meta_->grad_
input_line_10:2:18: error: member reference type 'std::shared_ptr<AutogradMeta<float> >' is not a pointer; did you mean to use '.'?
 t.autograd_meta_->grad_
 ~~~~~~~~~~~~~~~~^~
                 .
input_line_10:2:20: error: no member named 'grad_' in 'std::__1::shared_ptr<AutogradMeta<float> >'
 t.autograd_meta_->grad_
 ~~~~~~~~~~~~~~~~  ^
[cling]$ t.autograd_meta_.grad_.data()
input_line_11:2:19: error: no member named 'grad_' in 'std::__1::shared_ptr<AutogradMeta<float> >'
 t.autograd_meta_.grad_.data()
 ~~~~~~~~~~~~~~~~ ^
[cling]$ t
(Tensor<float> &) @0x1079fc420
[cling]$ t.autograd_meta_
(std::shared_ptr<AutogradMeta<float> > &) std::shared_ptr -> 0x6000000b6618
[cling]$ t.autograd_meta_.grad()
input_line_14:2:19: error: no member named 'grad' in 'std::__1::shared_ptr<AutogradMeta<float> >'
 t.autograd_meta_.grad()
 ~~~~~~~~~~~~~~~~ ^
[cling]$ t.autograd_meta_->grad()
input_line_15:2:18: error: member reference type 'std::shared_ptr<AutogradMeta<float> >' is not a pointer; did you mean to use '.'?
 t.autograd_meta_->grad()
 ~~~~~~~~~~~~~~~~^~
                 .
input_line_15:2:20: error: no member named 'grad' in 'std::__1::shared_ptr<AutogradMeta<float> >'
 t.autograd_meta_->grad()
 ~~~~~~~~~~~~~~~~  ^
[cling]$ t.autograd_meta_->grad()
input_line_16:2:18: error: member reference type 'std::shared_ptr<AutogradMeta<float> >' is not a pointer; did you mean to use '.'?
 t.autograd_meta_->grad()
 ~~~~~~~~~~~~~~~~^~
                 .
input_line_16:2:20: error: no member named 'grad' in 'std::__1::shared_ptr<AutogradMeta<float> >'
 t.autograd_meta_->grad()
 ~~~~~~~~~~~~~~~~  ^
[cling]$ auto grad_t = t.autograd_meta_->grad();
input_line_17:2:32: error: member reference type 'std::shared_ptr<AutogradMeta<float> >' is not a pointer; did you mean to use '.'?
 auto grad_t = t.autograd_meta_->grad();
               ~~~~~~~~~~~~~~~~^~
                               .
input_line_17:2:34: error: no member named 'grad' in 'std::__1::shared_ptr<AutogradMeta<float> >'
 auto grad_t = t.autograd_meta_->grad();
               ~~~~~~~~~~~~~~~~  ^
[cling]$ auto grad_t = t.autograd_meta_.grad();
input_line_18:2:33: error: no member named 'grad' in 'std::__1::shared_ptr<AutogradMeta<float> >'
 auto grad_t = t.autograd_meta_.grad();
               ~~~~~~~~~~~~~~~~ ^
[cling]$ auto grad_t = t.autograd_meta_.get()->grad();
[cling]$ grad_t
(Tensor<float> &) @0x1077e8540
[cling]$ grad_t.storage_
(std::shared_ptr<Storage<float> > &) std::shared_ptr -> 0x6000020a2da8
[cling]$ grad_t.storage_.get()->data_
(std::vector<float> &) { 0.00000f }
[cling]$ grad_t.storage_.get()->data_[0]
(float) 0.00000f
```

# autograd

```cpp
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                               Node
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// A `Node` is an abstract class that represents an operation taking zero
// or more input `Variable`s and producing zero or more output `Variable`s. All
// functions in PyTorch's autograd machinery derive from this class and
// override its `apply` method. Instances of such subclasses will then be
// invokeable via the call operator.
//
//                    Nodes in the Autograd Graph
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// When viewing the autograd system as a graph, `Node`s are the vertices or
// nodes, connected to each other via (directed) `Edge`s, which themselves are
// represented via (`Node`, input_nr) pairs. `Variable`s are the outputs to
// and inputs of `Node`s, and travel between these edges during execution
// of the graph. When two or more `Edge`s (from different sources) point at the
// same input to a `Node`, the values produced along all of these edges are
// implicitly summed prior to being forwarded to the target `Node`.
//
//                              Hierarchy
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Subclasses usually represent differentiable functions as well as their
// gradient operators. Note, however, that due to the very general definition
// of a `Node` taking *zero* or more inputs and producing *zero* or more
// outputs, uses of `Node`s are flexible and extend beyond purely
// mathematical operations. For example, the `AccumulateGrad` function is a
// *sink*: it takes one input, but produces no outputs, instead accumulating
// the input as a side effect. At the other extreme, the `GraphRoot` function
// receives no inputs from other functions, but produces multiple outputs.
//
//                              Interface
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The most important method on `Node` is the call operator, which takes in
// a list of variables and produces a list of variables. The precise size of
// these lists can be determined with `num_inputs()` and `num_outputs()`.
// `Node`s are stitched together via their `next_edge` interface, which let
// you manipulate the set of outgoing edges of a `Node`. You can add an
// edge with `add_next_edge()`, retrieve an edge with `next_edge(index)` and
// iterate over them via the `next_edges()` method. Other methods exist for
// integration with the JIT and other parts of PyTorch. Every `Node` has a
// *sequence number* that increases monotonically in the order of `Node`
// construction. It can be retrieved via the `sequence_nr()` method. Note that
// this sequence number is *thread local*. This means that when `Node`s
// `A`, `B` and `C` are created consecutively in the same thread, their
// sequence numbers will be ordered `A` < `B` < `C`. If, however, `A` and `B`
// are created in one thread and `C` is created in a new thread, there are *no
// guarantees* w.r.t. the ordering of `C` relative to `A` or `B`.
// See NOTE [ Sequence Number] for more details on the usages of sequence number.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

## https://pytorch.org/blog/overview-of-pytorch-autograd-engine/

> Formally, what we are doing here, and PyTorch autograd engine also does, is computing a Jacobian-vector product (Jvp) to calculate the gradients of the model parameters, since the model parameters and inputs are vectors.

- automatic differentiation is a technique that, given a computational graph, calculates the gradients of the inputs.
- Automatic differentiation can be performed in two different ways; forward and reverse mode:
  - forward mode means that we calculate the gradients along with the result of the function
  - while reverse mode requires us to evaluate the function first, and then we calculate the gradients starting from the output.


https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf

> Autodiff is not finite differences.
> Finite differences are expensive, since you need to do a forward pass for
each derivative. It also induces huge numerical error. Normally, we only use it for testing.
> Autodiff is both efficient (linear in the cost of computing the value) and numerically stable.


- Automatic differentiation relies on a classic calculus formula known as the chain-rule. The chain rule allows us to calculate very complex derivatives by splitting them and recombining them later.
- `f(g(x))`'s derivative can be calculated as `d/dx f(g(x)) = f'(g(x))g'(x)` (Lagrange's notation)
- dz/dx = dz/dy + dy/dx is leibniz's notation
> As put by George F. Simmons: "if a car travels twice as fast as a bicycle and the bicycle is four times as fast as a walking man, then the car travels 2 × 4 = 8 times as fast as the man."[1]
> Let z, y and x be the (variable) positions of the car, the bicycle, and the walking man, respectively
> `dz/dx = (dz/dy)/(dy/dx)`

- This result is what makes automatic differentiation work.
- By combining the derivatives of the simpler functions that compose a larger one, such as a neural network, **it is possible to compute the exact value of the gradient at a given point rather than relying on the numerical approximation**, which would require multiple perturbations in the input to obtain a value.


Let's do a concrete example, `f(x,y) = log(x*y)`. Okay so `x` and `y` are inputs on the left, this is the compute graph to make output `z`:
```
x --|
    --> * --> v --> log --> w
y --|
```

- The automatic differentiation engine will normally execute this graph. It will also extend it to calculate the derivatives of w with respect to the inputs x, y, and the intermediate result v.
- `f(x,y) = log(g(x,y))` and `g(x,y)=xy`
- **Every time the engine executes an operation in the graph, the derivative of that operation is added to the graph to be executed later in the backward pass.**
- `d/dx g(x,y) = y` `d/dy g(x,y) = x`

![](https://pytorch.org/assets/images/multi_derivative_graph.png)

- Note that the backward graph (green nodes) will not be executed until all the forward steps are completed.
- graph now executes `log(v)`, `dw/dv log(v) = 1/v`

![](https://pytorch.org/assets/images/extended_computational_graph.png)

- with `dw/dv` propagated backward and multiplied by the multiplication derivative as in the chain rule, generates the derivatives `dw/dx` and `dw/dx`

IMPLEMENTATION DETAIL!

- **The original computation graph is extended with a new dummy variable z that is the same w. The derivative of z with respect to w is 1 as they are the same variable,** this trick allows us to apply the chain rule to calculate the derivatives of the inputs.
- we execute `LogDerivative` and multiple it's result by `dz/dw` (which is 1) as per chain rule. Then exectute `Mult Derivative` and chain again and get `dz/dx`, `dz/dy`

- Formally, what we are doing here, and PyTorch autograd engine also does, is computing a Jacobian-vector product (Jvp) to calculate the gradients of the model parameters, **since the model parameters and inputs are vectors.**

[To represent `x` hat, `x` with `->` hat vecotr on top I'm using `x|`]

- gradient of vector valued function f(x| = y|) is esentially calculating a jacobian matrix
- with chain rule, multiplying the jacobian matrix of a function `f(x|) = y|` by a vector `v`, with the previously calculated gradients of a scalar function `z = g(y|)` results in the gradients `dz/dx1 ... dz/dxn` of scalar output with respect to the vector valued function inputs. [(this sounds complicated but it isnt just think about it)]

```py
def f(x1, x2):
      a = x1 * x2
      y1 = log(a)
      y2 = sin(x2)
      return (y1, y2)

def g(y1, y2):
      return y1 * y2
```

- Now, if we derive this by hand using the chain rule and the definition of the derivatives, we obtain the following set of identities that we can directly plug into the Jacobian matrix of `f(x1, x2)`: `dy1/dx = 1/x1`, `dy1/dx2 = 1/x2`, `dy2/dx1 = 0`, `dy2/dx2 = cos(x2)`... And `z = g(y1, y2)`: `dz/y1 = y2`, `dz/y2 = y1`

```
|∂y₁/∂x₁   ∂y₁/∂x₂|ᵀ |y₂|   |1/x₁  0      |ᵀ |y₂|   |1/x₁*y₂|
|∂y₂/∂x₁   ∂y₂/∂x₂|  |y₁| = |1/x₂  cos(x₂)|  |y₁| = |1/x₂*y₂ + cos(x₂)*y₁|

Jvp for (x₁, x₂) = (0.5, 0.75): (dy/dx₁, dy/dx₂) = (1.3633, 0.1912)
```

```py
>>> import torch
>>> x = torch.tensor([0.5, 0.75], requires_grad=True)
>>> y = torch.log(x[0] * x[1]) * torch.sin(x[1])
>>> y.backward(torch.tensor(1)) # modified from example for torch 2.0
>>> x.grad
tensor([1.3633, 0.1912])
```

- The result is the same as our hand-calculated Jacobian-vector product! However, PyTorch never constructed the matrix as it could grow prohibitively large but instead, created a graph of operations that traversed backward while applying the Jacobian-vector products defined in tools/autograd/derivatives.yaml.

https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml

- In PyTorch, the initial gradient is explicitly set by the user when he calls the backward method.
- Then, the Jvp calculation starts **but it never constructs the matrix**. Instead, when PyTorch records the computational graph, the derivatives of the executed forward operations are added (Backward Nodes). Figure 5 shows a backward graph generated by the execution of the functions `f(x_1, x_2)` and `g(y_1, y_2)` seen before.

![](https://pytorch.org/assets/images/computational_graph_backward_pass.png)

- The basic derivatives are stored in the tools/autograd/derivatives.yaml file and they are not regular derivatives but the Jvp versions of them [3]. (references https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf)
- Chain baby chain:

![](https://pytorch.org/assets/images/chain_rule_backward_differentiation.png)

Let's see an example of how the derivatives are stored in PyTorch:

1. Suppose we are processing the backward propagation of the log function, in the LogBackward node in Figure 2.
2. The derivative of log in derivatives.yaml is specified as `grad.div(self.conj())`. `grad` is the already calculated gradient `dz/dy1`, and `self.conj()` is the complex conjugate of the input vector.
3. For complex numbers, PyTorch calculates a special derivative called the conjugate Wirtinger derivative. This derivative takes the complex number and its conjugate, and they are the direction of steepest descent when plugged into optimizers.
4. The code translates to `(dz/dy1 * 1/v)`, corresponding to the green and red squares in Figure 3.
5. The autograd engine continues executing the next operation, the backward of the multiplication. The inputs are the original function's inputs and the gradient calculated from the log backward step.
6. This step repeats until we reach the gradient with respect to the inputs, and the computation finishes. The gradient `dz/dx2` is only completed once the multiplication and sin gradients are added together.

As you can see, we computed the equivalent of the Jvp without constructing the matrix. In the next post, we will dive inside PyTorch code to see how this graph is constructed and where the relevant pieces are, should you want to experiment with it.

## https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/

- **tools**/autograd:
  - Here we can find the definition of the derivatives as we saw in the previous post derivatives.yaml
  - several python scripts and a folder called templates.
    - These scripts and the templates are used at building time to generate the C++ code for the derivatives as specified in the yaml file.
    - Also, the scripts here generate wrappers for the regular ATen functions so that the computational graph can be constructed.

https://github.com/pytorch/pytorch/tree/release/1.9/tools/autograd


- **torch**/autograd:
  - This folder is where the autograd components that can be used directly from python are located.
  - In function.py we find the actual definition of torch.autograd.Function, a class used by users to write their own differentiable functions in python as per the documentation.
  - functional.py holds components for functionally computing the jacobian vector product, hessian, and other gradient related computations of a given function.
  - The rest of the files have additional components such as **gradient checkers**, anomaly detection, and the autograd profiler.

https://github.com/pytorch/pytorch/tree/851e89c8e817f28270e0fc21d74ced9446bea747/torch/autograd

- torch/csrc/autograd:
  - This is where the graph creation and execution-related code lives. All this code is written in C++, since it is a critical part that is required to be extremely performant.
  - Here we have:
    - several files that implement the engine, metadata storage, and all the needed components.
    - Alongside this, we have several files whose names start with python_, and their main responsibility is to allow python objects to be used in the autograd engine.


![](https://pytorch.org/assets/images/augmented_computational_graph.png)

how PyTorch creates these graphs with references to the actual codebase.

```py
>>> # request a tensor to require the gradient.
>>> x = torch.tensor([0.5, 0.75], requires_grad=True)
```

c10 will allocate an AutogradMeta object that is used to hold the graph information.

```cpp
void TensorImpl::set_requires_grad(bool requires_grad) {
  // ...
  if (!autograd_meta_)
    autograd_meta_ = impl::GetAutogradMetaFactory()->make();
    autograd_meta_->set_requires_grad(requires_grad, this);
}
```

```cpp

struct TORCH_API AutogradMeta : public c10::AutogradMetaInterface {
  std::string name_;

  Variable grad_;
  std::shared_ptr<Node> grad_fn_;
  std::weak_ptr<Node> grad_accumulator_;
  // other fields and methods
  // ...
};
```

`AutogradMeta` in `variable.h`: https://github.com/pytorch/pytorch/blob/release/1.9/torch/csrc/autograd/variable.h#L190-L286

The most important fields in this structure are:
  - the computed gradient in grad_ and a pointer to the function grad_fn that will be called by the engine to produce the actual gradient.
  - Also, there is a gradient accumulator object that is used to add together all the different gradients where this tensor is involved as we will see in the graph execution.


## Graphs edges and Nodes


Now, when we call a differentiable function that takes this tensor as an argument, the associated metadata will be populated.


Let’s suppose that we call a regular torch function that is implemented in ATen. Let it be the multiplication as in our previous blog post example. The resulting tensor has a field called grad_fn that is essentially a pointer to the function that will be used to compute the gradient of that operation.

```py
>>> x = torch.tensor([0.5, 0.75], requires_grad=True)
>>> v = x[0] * x[1]
>>> v
tensor(0.3750, grad_fn=<MulBackward0>)
```

Here we see that the tensors’ grad_fn has a MulBackward0 value. This function is the same that was written in the derivatives.yaml file, and its C++ code was generated automatically by all the scripts in tools/autograd:

```yaml
- name: mul.Tensor(Tensor self, Tensor other) -> Tensor
  self: mul_tensor_backward(grad, other, self.scalar_type())
  other: mul_tensor_backward(grad, self, other.scalar_type())
  result: other_t * self_p + self_t * other_p
```

https://github.com/pytorch/pytorch/blob/e7cd59c7a061c78d8d0265e4308b5933e44f9176/tools/autograd/derivatives.yaml#L840-L843

> It’s auto-generated source code can be seen in torch/csrc/autograd/generated/Functions.cpp.
>
> https://github.com/pytorch/pytorch/blob/e7cd59c7a061c78d8d0265e4308b5933e44f9176/torch/csrc/autograd/function.h#L50-L533


```cpp
variable_list MulBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (mul_tensor_backward(grad, self, other_scalar_type)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mul_tensor_backward(grad, other, self_scalar_type)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
```

🤯🤯🤯🤯🤯🤯!!

The `grad_fn` objects inherit from the `TraceableFunction` class, a descendant of `Node`:
  - adds just a property set to enable tracing for debugging and optimization purposes.

A graph by definition has nodes and edges:
  - so these functions are indeed the nodes of the computational graph that are linked together by using `Edge` objects to enable the graph traversal later on.


#### Nodes

```cpp
struct TORCH_API Node : std::enable_shared_from_this<Node> {
 // ...
 /// Evaluates the function on the given inputs and returns the result of the
  /// function call.
  variable_list operator()(variable_list&& inputs) {
  // ...
  }

protected:
  /// Performs the `Node`'s actual operation.
  virtual variable_list apply(variable_list&& inputs) = 0;
  …
  edge_list next_edges_;
```

- Essentially we see that it has an override of the operator () that performs the call to the actual function
- and a pure virtual function called apply
The automatically generated functions override this apply method as we saw in the MulBackward0 example above.
- Finally, the node also has a **list** of edges to enable graph connectivity


#### Edges

The `Edge` object is used to link Nodes together and its implementation is straightforward.
```cpp
struct Edge {
  ...
  /// The function this `Edge` points to.
  std::shared_ptr<Node> function;
  /// The identifier of a particular input to the function.
  uint32_t input_nr;
};
```

- It only requires a function pointer (the actual grad_fn objects that the edges link together), and an input number that acts as an id for the edge.

#### linking nodes

When we invoke the product operation of two tensors, we enter into the realm of autogenerated code. All the scripts that we saw in tools/autograd fill a series of templates that wrap the differentiable functions in ATen. These functions have code to construct the backward graph during the forward pass.

https://github.com/pytorch/pytorch/blob/release/1.9/tools/autograd/gen_variable_type.py


Pytorch does a lot:
```py
GRADIENT_IMPLEMENTED_FOR_COMPLEX = {
    't', 'view', 'reshape', 'reshape_as', 'view_as', 'roll', 'clone',
    'repeat', 'expand', 'flip', 'fliplr', 'flipud', 'rot90', 'transpose',
    'permute', 'squeeze', 'unsqueeze', 'resize', 'resize_as', 'tril',
    'triu', 'chunk', 'zero_', 'eq_', 'ne_', 'add', '__radd__', 'sum',
    '_conj', 'sin', 'cos', 'mul', 'sinc', 'sinh', 'cosh', '__rmul__',
    'sgn', 'asin', 'acos', 'sub', 'div', 'cat', 'view_as_complex',
    'neg', 'complex', 'select', '_s_where', 'as_strided', 'slice', 'constant_pad_nd',
    'unbind', 'split', 'split_with_sizes', 'unsafe_split', 'split_with_sizes_backward',
    'dot', 'vdot', 'cholesky', 'triangular_solve', 'mm', '_unsafe_view', 'mv', 'ger',
    'bmm', 'diagonal', 'alias', 'atan', 'log', 'log10', 'log1p', 'log2', 'reciprocal',
    'tan', 'pow', 'rsqrt', 'tanh', 'tanh_backward', 'asinh', 'acosh', 'atanh', 'take', 'fill_',
    'exp', 'nonzero', 'mean', 'inverse', 'solve', 'linalg_cholesky', 'addcmul', 'addcdiv',
    'matrix_exp', 'linalg_eigh', 'cholesky_solve', 'linalg_qr', '_svd_helper', '_fft_c2c', '_fft_r2c',
    'linalg_solve', 'sqrt', 'stack', 'gather', 'index_select', 'index_add_', 'linalg_inv', 'linalg_inv_ex',
    'l1_loss_backward', 'baddbmm', 'addbmm', 'addmm', 'addmv', 'addr', 'linalg_householder_product',
    'constant_pad_nd', 'reflection_pad1d', 'reflection_pad2d', 'linalg_cholesky_ex', 'linalg_eig',
    'reflection_pad1d_backward', 'reflection_pad2d_backward', 'symeig',
    'replication_pad1d', 'replication_pad2d', 'replication_pad3d', 'take', 'put_',
    'replication_pad1d_backward', 'replication_pad2d_backward', 'replication_pad3d_backward',
    'diag', 'masked_scatter', 'masked_select', 'index_fill', 'trace', 'polar', 'cumsum', 'rsub',
    'eig', 'lerp', 'linalg_vector_norm', 'cumprod', 'prod', 'index_copy', 'lu', 'unfold', 'unfold_backward',
    'index', 'masked_fill', 'cross', 'lu_unpack'
}
```
https://github.com/pytorch/pytorch/blob/release/1.9/tools/autograd/gen_autograd.py


The gen_variable_type.py script is in charge of writing all this wrapping code.


```cpp
at::Tensor mul_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  ...
  auto _any_requires_grad = compute_requires_grad( self, other );
  std::shared_ptr<MulBackward0> grad_fn;
  if (_any_requires_grad) {
    // Creates the link to the actual grad_fn and links the graph for backward traversal
    grad_fn = std::shared_ptr<MulBackward0>(new MulBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    ...
  }
  …
  // Does the actual function call to ATen
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::mul(ks & c10::after_autograd_keyset, self_, other_);
  })();

  auto result = std::move(_tmp);
    if (grad_fn) {
       // Connects the result to the graph
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  ...
  return result;
}
```


> After the `grad_fn` object is created, the edges used to link the nodes together are created by using the `grad_fn->set_next_edges(collect_next_edges( self, other ))`; calls.

```cpp
struct MakeNextFunctionList : IterArgs<MakeNextFunctionList> {
  edge_list next_edges;
  using IterArgs<MakeNextFunctionList>::operator();
  void operator()(const Variable& variable) {
    if (variable.defined()) {
      next_edges.push_back(impl::gradient_edge(variable));
    } else {
      next_edges.emplace_back();
    }
  }
  void operator()(const c10::optional<Variable>& variable) {
    if (variable.has_value() && variable->defined()) {
      next_edges.push_back(impl::gradient_edge(*variable));
    } else {
      next_edges.emplace_back();
    }
  }
};

template <typename... Variables>
edge_list collect_next_edges(Variables&&... variables) {
  detail::MakeNextFunctionList make;
  make.apply(std::forward<Variables>(variables)...);
  return std::move(make.next_edges);
}
```

NOTE ON LEAF NODES

> Given an input variable (it’s just a regular tensor), collect_next_edges will create an Edge object by calling impl::gradient_edge

```cpp
Edge gradient_edge(const Variable& self) {
    // If grad_fn is null (as is the case for a leaf node), we instead
    // interpret the gradient function to be a gradient accumulator, which will
    // accumulate its inputs into the grad property of the variable. These
    // nodes get suppressed in some situations, see "suppress gradient
    // accumulation" below. Note that only variables which have `requires_grad =
    // True` can have gradient accumulators.
    if (const auto& gradient = self.grad_fn()) {
      return Edge(gradient, self.output_nr());
    } else {
      return Edge(grad_accumulator(self), 0);
    }
  }
```

![](https://pytorch.org/assets/images/computational_graph_creation.gif)
## https://pytorch.org/blog/how-computational-graphs-are-executed-in-pytorch/

## ez autograd visuals

https://github.com/pytorch/pytorch/blob/e7cd59c7a061c78d8d0265e4308b5933e44f9176/torch/csrc/autograd/function.h#L50

![](http://blog.ezyang.com/img/pytorch-internals/slide-21.png)
![](http://blog.ezyang.com/img/pytorch-internals/slide-22.png)
![](http://blog.ezyang.com/img/pytorch-internals/slide-23.png)

I dont get this one:
![](http://blog.ezyang.com/img/pytorch-internals/slide-24.png)
![](http://blog.ezyang.com/img/pytorch-internals/slide-25.png)


# tensors


# References

## http://blog.ezyang.com/2019/05/pytorch-internals/

### Strided Representation

![](http://blog.ezyang.com/img/pytorch-internals/slide-08.png)


### https://ezyang.github.io/stride-visualizer/index.html
> Strides specify a factor by which an index is multiplied when computing its index into an array.
> Contiguous: each stride is the product of the corresponding tail of sizes

#### Chatgpt
> For example, let's consider a 3-dimensional tensor with dimensions (H, W, C), where H is the height, W is the width, and C is the number of channels. The strides for each dimension would be:

```
Stride for height (H): W * C
Stride for width (W): C
Stride for channels (C): 1
```

### Views

![](http://blog.ezyang.com/img/pytorch-internals/slide-10.png)

### Tensor and Storage
![](http://blog.ezyang.com/img/pytorch-internals/slide-11.png)

## https://github.com/pytorch/pytorch/blob/da28af3286b2b0c54ec182fe2d4264f3e87d50f4/aten/src/ATen/templates/TensorBody.h

### Tensor
```cpp
// Tensor is a "generic" object holding a pointer to the underlying TensorImpl object, which
// has an embedded reference count. In this way, Tensor is similar to boost::intrusive_ptr.
//
// For example:
//
// void func(Tensor a) {
//   Tensor b = a;
//   ...
// }
//
// In this example, when we say Tensor b = a, we are creating a new object that points to the
// same underlying TensorImpl, and bumps its reference count. When b goes out of scope, the
// destructor decrements the reference count by calling release() on the TensorImpl it points to.
// The existing constructors, operator overloads, etc. take care to implement the correct semantics.
//
// Note that Tensor can also be NULL, i.e. it is not associated with any underlying TensorImpl, and
// special care must be taken to handle this.
```

### Assignment and Not Copying
```cpp
// The following overloads are very intruiging.  Consider the following
// program:
//
//    x[1] = 3;
//
// We would expect that the first entry of x is written to 3.  But how can we
// actually achieve this?  x[1] evaluates to a tensor...
//
// The answer is, using a ref-qualifier.  x[1] is an rvalue, which cannot be
// (profitably) assigned to in the traditional sense, so we overload
// assignment to mean, "Actually, copy 3 into the tensor data."  This is done
// with an rvalue-reference ref-qualified overload (the methods with && at the
// end of their type.)
//
// There's one more fly in the ointment: We also want
//
//    Tensor x = y;
//
// to work, and we want it NOT to copy.  So we need a traditional operator=
// overload.  But we MUST specify a mutable lvalue ref-qualifier, to
// disambiguate the traditional overload from the rvalue-reference
// ref-qualified overload.  Otherwise, it will be ambiguous, because
// a non ref-qualified method is eligible for all situations.
```

## https://www.tensors.net/tutorial-1

## Automatic Differentiation

### https://github.com/pytorch/pytorch/issues/13638

![](http://blog.ezyang.com/img/pytorch-internals/slide-19.png)

> Overly Complex OOP design: Currently the distinction between Variable and Tensor is hard to grasp: Variable::Impl is a subclass of TensorImpl, but it also has an at::Tensor data member which internally wraps another TensorImpl. This co-existence of "is-a" and "has-a" relationship makes the code complicated and adds cognitive overhead. In particular, it's difficult to track which functions we have overridden in Variable::Impl, and which functions are applicable to Tensor vs. Variable (e.g. is_wrapped_number() is only valid on Tensor, not Variable) (for more context, also see note: We regret making Variable hold a Tensor). Ideally, we want to use the same tensor type everywhere in PyTorch code.

> Virtual functions are slow: We care about how much time it takes to execute common Tensor functions such as numel() / sizes() / dim(). Currently, these functions are virtual in TensorImpl, so that Variable::Impl (a subclass of TensorImpl) can override them and dispatch those calls to the Variable::Impl's underlying at::Tensor. Virtual function calls are slow because they involve an extra vtable lookup. Specifically, we did the following comparison on the most common Tensor functions (all timings are in ns):

# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

> Generally speaking, torch.autograd is an engine for computing vector-Jacobian product.

> The backward pass kicks off when .backward() is called on the DAG root. autograd then:
> - computes the gradients from each .grad_fn,
> - accumulates them in the respective tensor’s .grad attribute, and
> - using the chain rule, propagates all the way to the leaf tensors.

# a9

> "It was much easier when I first did STL. I had one gigantical file called 'STL.h' – but at that time people said: 'It's utterly unacceptable, you have 20,000 lines of code. People cannot have includes that large.' If you think about it nowadays, it appears to be very funny."
>
> — Alexander Stepanov [@A9 Day 1](https://youtu.be/aIHAEYyoTUc?t=2002)

---

> "Some of you will drop after the first lecture, right? This always happens because you will be bored. You say, 'This guy doesn't know anything' – all true. However, before you drop, let me tell you a very simple piece of advice. If you remember just one thing out of this course, it should be this – two pieces of advice. The first piece of advice is whenever you can, use vector. The second piece of advice is if you cannot, find a way how you can. This is to avoid any data structures except arrays... Vector is an array, right. Sort of, you say, 'Well... Aren't there exceptions?' – yes, not for you!"
>
> — Alexander Stepanov [@A9 Day 1](https://youtu.be/aIHAEYyoTUc?t=1543)

---

```cpp
struct B {
  virtual void bar() { std::cout << "B" << std::endl; }
};
struct C : public B {
  void bar() override { std::cout << "C" << std::endl; }
};
//...
B* b = new C();
b->bar();
```

> "C++ is a multi-paradigm language. What it means is, once upon a time, it was an object-oriented language. And then some people chased away object-oriented and said you could program differently. And I am showing you how you could program differently. If you program in an object-oriented way, then many good things might happen. I don't know what they are, but you are not going to be efficient, alright? As Bjarne Stroustrup, you might have heard of him, used to joke, he would say, 'What are object-oriented systems?' He would say, 'These are systems with slow graphics.' And you know, if you go through all this virtual stuff, it is going to be slow. Replacing `++` on an `int` with virtual call is a wonderful idea. You could do it! But you see, `++` is really fast, and virtual function call is really slow. And as time progressed, `++` is getting faster and faster, and virtual function is getting slower and slower. Their spread is growing. Right? And what we're not going to address any parts of C++ in this course, which slow the computations down."
>
> — Alexander Stepanov [@A9 Day 2](https://youtu.be/B5yiLvaxPS4?t=333)

---

> "One thing that sort of that every programmer needs is to being relaxed. You're going from now on– It's your third day?... Let me tell you, from now on for the next 40 years, you're going to screw up, you're going to write bad code. I do all the time. I just did remember? I forgot to put angle brackets with `int` while standing in front of the room being recorded. Someone *kindly* corrected me. So you should not have– I'm actually dead serious. You know, we should be very relaxed about, ya, you know, I do something wrong, everyday. Try!"
>
> — Alexander Stepanov [@A9 Day 5](https://youtu.be/8bWx2WTYvB8?t=462)

---

```cpp
template <typename T>
// requires T is TotallyOrdered
struct less {
  bool operator()(const T&, const T& b) const {
    return a < b;
  }
};
// ...
std::sort(a.begin(), a.end(), less<int>())
```

> "Which function? There is no function. It's not a function. Let us observe what happens. This is not a function call, this is just `a < b`. It will literally be inlined at compile time. Is it good to be inlined at compile time? It is very good guys. Because instead of a function call, you're going to have how many instructions? Yes... Say one. One isntruction. Which will most likley be done in parallel with some other instruction. It is free! Function calls are not."
>
> — Alexander Stepanov [@A9 Day 5](https://youtu.be/8bWx2WTYvB8?t=1355)

---

> “A virtual method table (VMT),…, is a mechanism used in a programming language to support dynamic dispatch (or run-time method binding)..”

> In computer science, dynamic dispatch is the process of selecting which implementation of a polymorphic operation (method or function) to call at run time. It is commonly employed in, and considered a prime characteristic of, object-oriented programming (OOP) languages and systems.[1]

https://pabloariasal.github.io/2017/06/10/understanding-virtual-tables/
https://en.wikipedia.org/wiki/Dynamic_dispatch

```cpp
B* b = new C();
b->bar();
```

> You might be thinking: vtables are cool and all, but how exactly do they solve the problem? When the compiler sees b->bar() in the example above, it will lookup B’s vtable for bar’s entry and follow the corresponding function pointer, right? We would still be calling B::bar() and not C::bar()…

> Very true, I still need to tell the second part of the story: vpointers. Every time the compiler creates a vtable for a class, it adds an extra argument to it: a pointer to the corresponding virtual table, called the vpointer.

![](https://pabloariasal.github.io/assets/img/posts/vtables/vpointer.png)

> Note that the vpointer is just another class member added by the compiler **and increases the size of every object that has a vtable by sizeof(vpointer)**.

> Hopefully you have grasped how dynamic function dispatch can be implemented by using vtables: when a call to a virtual function on an object is performed, the vpointer of the object is used to find the corresponding vtable of the class. Next, the function name is used as index to the vtable to find the correct (most specific) routine to be executed. Done!

...

> By now it should also be clear why it is always a good idea to make destructors of base classes virtual. Since derived classes are often handled via base class references, declaring a non-virtual destructor will be dispatched statically, obfuscating the destructor of the derived class:


---

```cpp
template <typename T>
// requires T is TotallyOrdered
struct less {
  bool operator()(const T&, const T& b) const {
    return a < b;
  }
};
// std::sort(a.begin(), a.end(), less<int>())
/*
less<int>
what is this?
it's a type, do we need a type when sorting?
no, we need an object because it's a function call
how do we create an object of a type? ()
this calls the default constructor
*/
```

https://youtu.be/8bWx2WTYvB8?t=1259

> We are not passing a function pointer. We are passing an empty object (1 byte) that just carries this information. Function is a very important technique to allow you to pass things to template algorithms.
>
> https://youtu.be/8bWx2WTYvB8?t=1303

> Which function? There is no function. It's not a function. Let us observe what happens. This is not a function call, this is just `a < b`. It will literally be inlined at compile time. Is it good to be inlined at compile time? It is very good guys. Because instead of a function call, you're going to have how many instructions? Yes... Say one. One isntruction. Which will most likley be done in parallel with some other instruction. It is free! Function calls are not.
>
> https://youtu.be/8bWx2WTYvB8?t=1355

# other

Brew saves the day yet again: https://formulae.brew.sh/formula/cling!
```cpp
Ryans-Air:~ ryan$ cd tensorgrad/
(main) Ryans-Air:tensorgrad ryan$ cling

****************** CLING ******************
* Type C++ code and press enter to run it *
*             Type .q to exit             *
*******************************************
[cling]$ #include "tensor.h"
[cling]$
```
Don't forget to double check you aren't using the old Docker alias which is much slower to start and makes including local files hard:
```cpp
(main) Ryans-Air:tensorgrad ryan$ cat ~/.bash_profile | grep cling
alias cling='docker run -it maddouri/cling-ubuntu-docker'
```

---

An autograd python defined function looks like the following:

```py
class Exp(torch.autograd.Function):
     @staticmethod
     def forward(ctx, i):
         result = i.exp()
         ctx.save_for_backward(result)
         return result

     @staticmethod
     def backward(ctx, grad_output):
         result, = ctx.saved_tensors
         return grad_output * result

# Call the function
Exp.apply(torch.tensor(0.5, requires_grad=True))
# Outputs: tensor(1.6487, grad_fn=<ExpBackward>)
```
