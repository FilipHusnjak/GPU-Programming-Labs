#include <cstdio>
#include <chrono>

#define NUM_PARTICLES 10000000
#define TPB 100

#define NUM_STREAMS 4

typedef struct {
    float3 Position;
    float3 Velocity;
} Particle;

__host__ __device__ void sum(float3 &left, const float3 &right) {
    left.x += right.x;
    left.y += right.y;
    left.z += right.z;
}

__host__ __device__ void update(Particle &particle) {
    particle.Velocity.y += 9.81f;
    sum(particle.Position, particle.Velocity);
}

__global__ void update(Particle *particles) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    update(particles[idx]);
}

bool equal(const float3 &left, const float3 &right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

bool equal(const Particle &left, const Particle &right) {
    return equal(left.Position, right.Position) && equal(left.Velocity, right.Velocity);
}

int main() {
    Particle *particles;
    cudaMallocHost(&particles, NUM_PARTICLES * sizeof(Particle));

    auto *particles_cpy = (Particle *) malloc(NUM_PARTICLES * sizeof(Particle));
    memcpy(particles_cpy, particles, NUM_PARTICLES * sizeof(Particle));

    Particle *d_particles;
    cudaMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle));

    int batch_size = NUM_PARTICLES / NUM_STREAMS;
    batch_size += NUM_PARTICLES % NUM_STREAMS == 0 ? 0 : 1;

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    auto t1 = std::chrono::system_clock::now();
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMemcpyAsync(&d_particles[i * batch_size], &particles[i * batch_size], batch_size * sizeof(Particle),
                        cudaMemcpyHostToDevice, streams[i]);
        update<<<(batch_size + TPB - 1) / TPB, TPB, 0, streams[i]>>>(&d_particles[i * batch_size]);
        cudaMemcpyAsync(&particles[i * batch_size], &d_particles[i * batch_size], batch_size * sizeof(Particle),
                        cudaMemcpyDeviceToHost, streams[i]);
    }
    cudaDeviceSynchronize();
    auto t2 = std::chrono::system_clock::now();
    printf("Updating particles on the GPU done in: %lf ms!\n",
           std::chrono::duration<double, std::chrono::milliseconds::period>(t2 - t1).count());

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }

    t1 = std::chrono::system_clock::now();
    for (int i = 0; i < NUM_PARTICLES; i++) {
        update(particles_cpy[i]);
    }
    t2 = std::chrono::system_clock::now();
    printf("Updating particles on the CPU done in: %lf ms!\n",
           std::chrono::duration<double, std::chrono::milliseconds::period>(t2 - t1).count());

    for (int i = 0; i < NUM_PARTICLES; i++) {
        if (!equal(particles[i], particles_cpy[i])) {
            printf("Wrong! %d\n", i);
            cudaFree(particles);
            free(particles_cpy);
            cudaFree(d_particles);
            return 0;
        }
    }

    printf("Correct!\n");
    cudaFree(particles);
    free(particles_cpy);
    cudaFree(d_particles);

    return 0;
}
