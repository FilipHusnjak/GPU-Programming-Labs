#include <curand_kernel.h>
#include <curand.h>
#include <cstdio>
#include <chrono>
#include <cuda_fp16.h>

#define TPB 256
#define NUM_SAMPLES 10000000
#define N 100000

#define PI 3.14159265359

__global__ void ornl(unsigned int *res, curandState *states) {
    const unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= NUM_SAMPLES / N) return;

    __shared__  unsigned int count[TPB];

    curand_init(idx, idx, 0, &states[idx]);

    count[threadIdx.x] = 0;
    for (int i = 0; i < N; i++) {
        float x = curand_uniform(&states[idx]);
        float y = curand_uniform(&states[idx]);

        int z = (int) (x * x + y * y);
        count[threadIdx.x] += 1 - z;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned int blockSum = 0;
        for (unsigned int i = 0; i < blockDim.x; i++)
            blockSum += count[i];

        atomicAdd(res, blockSum);
    }
}

int main() {
    curandState *states;
    unsigned int thread_num = NUM_SAMPLES / N;
    cudaMalloc(&states, thread_num * sizeof(curandState));

    unsigned int *d_res;
    unsigned int a = 0;
    cudaMalloc(&d_res, sizeof(*d_res));

    auto t1 = std::chrono::system_clock::now();
    cudaMemcpy(d_res, &a, sizeof(a), cudaMemcpyHostToDevice);
    ornl<<<(thread_num + TPB - 1) / TPB, TPB>>>(d_res, states);
    cudaDeviceSynchronize();
    unsigned int res;
    cudaMemcpy(&res, d_res, sizeof(*d_res), cudaMemcpyDeviceToHost);
    auto t2 = std::chrono::system_clock::now();
    printf("Calculating PI on the GPU done in: %lf ms!\n",
           std::chrono::duration<double, std::chrono::milliseconds::period>(t2 - t1).count());

    double pi = 4 * (double) res / (NUM_SAMPLES);
    printf("Obtained PI \t\t= %f\n", pi);
    printf("Real PI \t\t= %f\n", PI);
}
