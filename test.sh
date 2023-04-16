#!/bin/bash
git submodule update --init --recursive

rm -rf build
mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
cmake ..
make
./Tensor
# make clean
