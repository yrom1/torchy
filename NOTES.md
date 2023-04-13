

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

---

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
