all: T
	./T

T: T.cpp
	g++ -o T T.cpp -std=c++14

clean:
	rm -f T
