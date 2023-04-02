cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(aten_example)

find_package(Torch REQUIRED)

# Get a list of all C++ files in the directory that start with "aten_"
file(GLOB_RECURSE ATEN_SOURCES "aten_*.cpp")

# Create an executable for each ATen-related file
foreach(SOURCE ${ATEN_SOURCES})
  get_filename_component(EXECUTABLE_NAME ${SOURCE} NAME_WE)
  add_executable(${EXECUTABLE_NAME} ${SOURCE})
  target_link_libraries(${EXECUTABLE_NAME} "${TORCH_LIBRARIES}")
  set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CXX_STANDARD 14)
endforeach()
