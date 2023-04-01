CXXFLAGS=-std=c++14 -Wall -Wextra -Wpedantic -fsanitize=address

all: build
	./program

program: test.o
	g++ -o program test.o $(CXXFLAGS)

test.o: test.cpp T.h
	g++ -c test.cpp -o test.o $(CXXFLAGS)

clean:
	rm -f program *.o

lint:
	cpplint --recursive test.cpp T.h

format:
	clang-format -i -style=Google test.cpp T.h

build: program format lint

.PHONY: all clean lint format build
.DEFAULT_GOAL := all
