#include "tensor.h"
#include <vector>
#include <iostream>

int main() {
  Tensor<int> x({3, 2}, {1, 2, 3, 4, 5, 6});
  Tensor<int> w({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  Tensor<int> b({1, 4}, {1, 2, 3, 4});
  // BUG broadcasting we need to broadcast O_O
  //     broken because of + b!
  // Does not work:
  // Tensor<int> result = x.matmul(w) + b;
  // Works:
  Tensor<int> result = x.matmul(w);
  std::cout << x << std::endl;
  std::cout << w << std::endl;
  std::cout << b << std::endl;
  std::cout << result << std::endl;
  /* this is without the + b
  [[1, 2],
  [3, 4],
  [5, 6]]
  [[1, 2, 3, 4],
  [5, 6, 7, 8]]
  [[1, 2, 3, 4]]
  [[11, 14, 17, 20],
  [23, 30, 37, 44],
  [35, 46, 57, 68]]
  */
  /*
  >>> x = torch.tensor(((1, 2), (3, 4), (5, 6)))
  >>> w = torch.tensor(((1, 2, 3, 4), (5, 6, 7, 8)))
  >>> x
  tensor([[1, 2],
          [3, 4],
          [5, 6]])
  >>> w
  tensor([[1, 2, 3, 4],
          [5, 6, 7, 8]])
  >>> x.matmul(w)
  tensor([[11, 14, 17, 20],
          [23, 30, 37, 44],
          [35, 46, 57, 68]])
  */
  auto result_v = result.storage()->data();
}
