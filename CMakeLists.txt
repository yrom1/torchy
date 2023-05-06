cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(Tensor)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable AddressSanitizer
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

# Find the Torch package
find_package(Torch REQUIRED CONFIG NO_DEFAULT_PATH HINTS ${CMAKE_PREFIX_PATH} QUIET)
set(USE_CUDA OFF)

# Build googletest from the submodule
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/googletest)

# Add the target executable
add_executable(${PROJECT_NAME} test2.cpp)

# Link the executable against the googletest and torch libraries
target_link_libraries(${PROJECT_NAME} gtest gtest_main ${TORCH_LIBRARIES})

# Add compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -fsanitize=address)

# Copy headers to build directory
configure_file(torchy.h torchy.h COPYONLY)

# Clang-format and cpplint custom targets
find_program(CLANG_FORMAT NAMES clang-format)
find_program(CPPLINT NAMES cpplint)

if(CLANG_FORMAT)
  add_custom_target(
    format
    COMMAND ${CLANG_FORMAT} -i -style=Google ${CMAKE_SOURCE_DIR}/test.cpp ${CMAKE_SOURCE_DIR}/torchy.h
    COMMENT "Running clang-format on the source files"
  )
endif()

if(CPPLINT)
  add_custom_target(
    lint
    COMMAND ${CPPLINT} ${CMAKE_SOURCE_DIR}/test.cpp ${CMAKE_SOURCE_DIR}/torchy.h
    COMMENT "Running cpplint on the source files"
  )
endif()

# Add the format and lint targets to the default build target
add_dependencies(${PROJECT_NAME} format lint)
