#include "globeik_kernel.h"
#include "../GRiD/grid.cuh"

#include <iostream>

template<typename T>
void test_compile() {
	printf("Hello\n");
}

template void test_compile<int>();
template void test_compile<float>();
template void test_compile<double>();