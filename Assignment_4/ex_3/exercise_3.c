#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

#define N 100000000
#define EPSILON 0.00001

void saxpy_naive(float* y, const float* x, const float c, const int n){
    for(int i=0; i<n; i++){
        y[i] = c*x[i] + y[i];
    }
}

void saxpy_acc(float* y, const float* x, const float c, const int n){
    #pragma acc parallel loop independent
    for(int i=0; i<n; i++){
        y[i] = c*x[i] + y[i];
    }
}

void initialize_rand(float* arr, int n){
    for(int i=0; i<n; i++){
        arr[i] = rand();
    }
}

int main(){
    float* y = malloc(N * sizeof(float));
    float* x = malloc(N * sizeof(float));

    float* y_naive = malloc(N * sizeof(float));
    float* y_acc = malloc(N * sizeof(float));

    srand(0);
    initialize_rand(y, N);
    initialize_rand(x, N);

    float c = 1.5f;

    clock_t t;
    double elapsed;

    int step = N / 100;
    int start = 0;

    printf("%d\n", N);
    printf("%d\n", step);
    printf("%d\n", start);

    memcpy(y_naive, y, N * sizeof(float));
    for(int i=start; i<N; i+=step){
        t = clock();
        saxpy_naive(y_naive, x, c, i);
        elapsed = (clock() - t) * 1000 / CLOCKS_PER_SEC;

        printf("%.4f\n", elapsed);
    }

    memcpy(y_acc, y, N * sizeof(float));
    for(int i=start; i<N; i+=step) {
        t = clock();
        saxpy_acc(y_acc, x, c, i);
        elapsed = (clock() - t) * 1000 / CLOCKS_PER_SEC;

        printf("%.4f\n", elapsed);
    }

    memcpy(y_naive, y, N * sizeof(float));
    memcpy(y_acc, y, N * sizeof(float));

    saxpy_naive(y_naive, x, c, N);
    saxpy_acc(y_acc, x, c, N);
    for(int i=0; i<N; i++){
        assert(fabsf(y_naive[i] - y_acc[i]) < EPSILON);
    }

    free(y);
    free(x);
    free(y_naive);
    free(y_acc);

    return 0;
}