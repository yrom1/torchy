# tensorgrad

A small tensor-valued autograd engine, inspired by PyTorch and micrograd

> "It was much easier when I first did STL. I had one gigantical file called 'STL.h' – but at that time people said: 'It's utterly unacceptable, you have 20,000 lines of code. People cannot have includes that large.' If you think about it nowadays, it appears to be very funny."
>
> — Alexander Stepanov [@A9 Day 1](https://youtu.be/aIHAEYyoTUc?t=2002)


> "Some of you will drop after the first lecture, right? This always happens because you will be bored. You say, 'This guy doesn't know anything' – all true. However, before you drop, let me tell you a very simple piece of advice. If you remember just one thing out of this course, it should be this – two pieces of advice. The first piece of advice is whenever you can, use vector. The second piece of advice is if you cannot, find a way how you can. This is to avoid any data structures except arrays... Vector is an array, right. Sort of, you say, 'Well... Aren't there exceptions?' – yes, not for you!"
>
> — Alexander Stepanov [@A9 Day 1](https://youtu.be/aIHAEYyoTUc?t=1543)


> "One thing that sort of that every programmer needs is to being relaxed. You're going from now on– It's your third day?... Let me tell you, from now on for the next 40 years, you're going to screw up, you're going to write bad code. I do all the time. I just did remember? I forgot to put angle brackets with `int` while standing in front of the room being recorded. Someone *kindly* corrected me. So you should not have– I'm actually dead serious. You know, we should be very relaxed about, ya, you know, I do something wrong, everyday. Try!"
>
> — Alexander Stepanov [@A9 Day 4](https://youtu.be/8bWx2WTYvB8?t=462)

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
