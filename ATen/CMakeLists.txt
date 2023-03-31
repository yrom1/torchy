cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(aten_example)

find_package(Torch REQUIRED)

add_executable(aten_example aten_example.cpp)
target_link_libraries(aten_example "${TORCH_LIBRARIES}")
set_property(TARGET aten_example PROPERTY CXX_STANDARD 14)
