#include <cstdio>

#define TPB 256
#define BDIMX 1

__global__ void helloWorld() {
    printf("Hello World! My threadId is %d\n", threadIdx.x);
}

int main() {
    helloWorld<<<dim3(BDIMX), dim3(TPB)>>>();

    cudaDeviceSynchronize();

    return 0;
}
