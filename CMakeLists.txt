cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(Tensor)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable AddressSanitizer
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

# Find the Torch package
find_package(Torch REQUIRED)

# Build googletest from the submodule
add_subdirectory(googletest)

# Add the target executable
add_executable(${PROJECT_NAME} test.cpp)

# Link the executable against the googletest and torch libraries
target_link_libraries(${PROJECT_NAME} gtest gtest_main ${TORCH_LIBRARIES})

# Add compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -fsanitize=address)

# Copy headers to build directory
configure_file(tensor.h tensor.h COPYONLY)
