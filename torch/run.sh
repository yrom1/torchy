#!/bin/bash
rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
cmake --build .

for f in aten_*; do
  ./"$f"
done
