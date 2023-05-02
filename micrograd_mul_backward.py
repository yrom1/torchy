class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            print("_backward")
            print(f"+=", f"{other.data=}", f"{out.grad=}")
            self.grad += other.data * out.grad
            print(f"+=", f"{self.data=}", f"{out.grad=}")
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __rmul__(self, other): # other * self
        return self * other

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


a = Value(2.0)
c = a * a * a
c.backward()
print(f"{a.data=}", f"{a.grad=}")
