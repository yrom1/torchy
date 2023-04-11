```cpp
template <typename T>
// requires T is TotallyOrdered
struct less {
  bool operator()(const T&, const T& b) const {
    return a < b;
  }
};

std::sort(a.begin(), a.end(), less<int>()) // usage
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
