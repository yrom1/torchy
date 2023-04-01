CXXFLAGS=-std=c++14 -Wall -Wextra -Wpedantic -fsanitize=address

all: build
	./program

program: test.o
	g++ -o program test.o $(CXXFLAGS)

test.o: test.cpp Tensor.h
	g++ -c test.cpp -o test.o $(CXXFLAGS)

clean:
	rm -f program *.o

lint:
	cpplint --recursive test.cpp Tensor.h

format:
	clang-format -i -style=Google test.cpp Tensor.h

build: program format lint

.PHONY: all clean lint format build
.DEFAULT_GOAL := all
