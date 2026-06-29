#include "test.hpp"
#include <iostream>

int main()
{
    std::cout << std::unitbuf;

    test_mlp();
    test_cnn();
    return 0;
}
