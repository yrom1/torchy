# autograd

![](http://blog.ezyang.com/img/pytorch-internals/slide-21.png)
![](http://blog.ezyang.com/img/pytorch-internals/slide-22.png)
![](http://blog.ezyang.com/img/pytorch-internals/slide-23.png)

I dont get this one:
![](http://blog.ezyang.com/img/pytorch-internals/slide-24.png)
![](http://blog.ezyang.com/img/pytorch-internals/slide-25.png)

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

## https://pytorch.org/blog/how-computational-graphs-are-executed-in-pytorch/

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
