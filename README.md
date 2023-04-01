# Tensor.h

A tensor library, inspired by PyTorch's ATen

> "It was much easier when I first did STL. I had one gigantical file called 'STL.h' – but at that time people said: 'It's utterly unacceptable, you have 20,000 lines of code. People cannot have includes that large.' If you think about it nowadays, it appears to be very funny."
>
> — Alexander Stepanov [@A9](https://youtu.be/aIHAEYyoTUc?t=2002)


# References

## Strided Representation

![](http://blog.ezyang.com/img/pytorch-internals/slide-08.png)

- http://blog.ezyang.com/2019/05/pytorch-internals/

## TensorBody.h
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
- https://github.com/pytorch/pytorch/blob/da28af3286b2b0c54ec182fe2d4264f3e87d50f4/aten/src/ATen/templates/TensorBody.h
