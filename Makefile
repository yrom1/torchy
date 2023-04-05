CC=g++
CFLAGS=-std=c++14 -Wall -Wextra -Wpedantic -fsanitize=address
TARGET=a.out
SOURCES=test.cpp
HEADERS=tensor.h

all: format lint compile run

format:
	clang-format -i -style=Google $(SOURCES) $(HEADERS)

lint:
	cpplint --recursive $(SOURCES) $(HEADERS)

compile:
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET)

run:
	./$(TARGET)

clean:
	rm -f $(TARGET)
