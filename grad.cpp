#include <stdio.h>

extern "C" void hello();

int main() {
    printf("Hello, CPU!\n");
    hello();
    return 0;
}
