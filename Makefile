CXXFLAGS=-std=c++14 -Wall -Wextra -Wpedantic -fsanitize=address

all: build
	./program

program: main.o
	g++ -o program main.o $(CXXFLAGS)

main.o: main.cpp T.h
	g++ -c main.cpp -o main.o $(CXXFLAGS)

clean:
	rm -f program *.o

lint:
	cpplint --recursive main.cpp T.h

format:
	clang-format -i -style=Google main.cpp T.h

build: program format lint

.PHONY: all clean lint format build
.DEFAULT_GOAL := all
