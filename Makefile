CC=g++
CFLAGS=-std=c++14
TARGET=a.out
SOURCES=T2.cpp
HEADERS=T2.h

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
