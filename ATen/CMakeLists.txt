cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(aten_example)

find_package(Torch REQUIRED)

add_executable(aten_example aten_example.cpp)
add_executable(aten_equal aten_equal.cpp)

target_link_libraries(aten_example "${TORCH_LIBRARIES}")
target_link_libraries(aten_equal "${TORCH_LIBRARIES}")

set_property(TARGET aten_example PROPERTY CXX_STANDARD 14)
set_property(TARGET aten_equal PROPERTY CXX_STANDARD 14)
