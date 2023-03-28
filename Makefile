CXXFLAGS=-std=c++14 -Wall -Wextra -Wpedantic -fsanitize=address
LINT=cpplint
FORMAT=clang-format

all: build
	./T

T: T.cpp
	g++ -o T T.cpp $(CXXFLAGS)

clean:
	rm -f T

lint:
	$(LINT) --recursive .

format:
	$(FORMAT) -i -style=Google T.cpp

build: T format lint

.PHONY: all clean lint format build
.DEFAULT_GOAL := all
