#include <cstdio>
#include <chrono>

#define TPB 256
#define ARRAY_SIZE 1000000

__global__ void saxpy(const float *x, float *y, float a) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    y[idx] += x[idx] * a;
}

int main() {
    dim3 grid((ARRAY_SIZE + TPB - 1) / TPB);
    dim3 block(TPB);

    auto *x = (float *) malloc(ARRAY_SIZE * sizeof(float));
    auto *y = (float *) malloc(ARRAY_SIZE * sizeof(float));

    for (int i = 0; i < ARRAY_SIZE; i++) {
        x[i] = y[i] = float(i);
    }

    float *d_x;
    cudaMalloc(&d_x, ARRAY_SIZE * sizeof(float));
    cudaMemcpy(d_x, x, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    float *d_y;
    cudaMalloc(&d_y, ARRAY_SIZE * sizeof(float));
    cudaMemcpy(d_y, y, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    float a = 1;
    auto t1 = std::chrono::system_clock::now();
    saxpy<<<grid, block>>>(d_x, d_y, a);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::system_clock::now();
    printf("Computing SAXPY on the GPU done in: %lf ms!\n",
           std::chrono::duration<double, std::chrono::milliseconds::period>(t2 - t1).count());

    t1 = std::chrono::system_clock::now();
    for (int i = 0; i < ARRAY_SIZE; i++) {
        y[i] += x[i] * a;
    }
    t2 = std::chrono::system_clock::now();
    printf("Computing SAXPY on the CPU done in: %lf ms!\n",
           std::chrono::duration<double, std::chrono::milliseconds::period>(t2 - t1).count());

    auto *y_cpy = (float *) malloc(ARRAY_SIZE * sizeof(float));
    cudaMemcpy(y_cpy, d_y, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Comparing the output for each implementation: ");

    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (y[i] != y_cpy[i]) {
            printf("Wrong!\n");
            cudaFree(d_x);
            cudaFree(d_y);
            free(x);
            free(y);
            free(y_cpy);
            return 0;
        }
    }

    printf("Correct!\n");

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
    free(y_cpy);

    return 0;
}
