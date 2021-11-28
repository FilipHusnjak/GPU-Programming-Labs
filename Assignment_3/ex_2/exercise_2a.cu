#include <cstdio>
#include <chrono>

#define NUM_PARTICLES 10000000
#define NUM_ITERATIONS 1
#define TPB 64

#define MANAGED 1

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

int mainPinnedMemory() {
    Particle *particles;
    cudaMallocHost(&particles, NUM_PARTICLES * sizeof(Particle));

    auto *particles_cpy = (Particle *) malloc(NUM_PARTICLES * sizeof(Particle));

    memcpy(particles_cpy, particles, NUM_PARTICLES * sizeof(Particle));

    Particle *d_particles;
    cudaMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle));

    auto t1 = std::chrono::system_clock::now();

    for(int i=0; i<NUM_ITERATIONS; i++){
        cudaMemcpy(d_particles, particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);

        update<<<(NUM_PARTICLES + TPB - 1) / TPB, TPB>>>(d_particles);

        cudaMemcpy(particles, d_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyDeviceToHost);
    }

    auto t2 = std::chrono::system_clock::now();

    printf("Updating particles on the GPU done in: %lf ms!\n",
           std::chrono::duration<double, std::chrono::milliseconds::period>(t2 - t1).count());


    t1 = std::chrono::system_clock::now();
    for(int it=0; it<NUM_ITERATIONS; it++){
        for (int i = 0; i < NUM_PARTICLES; i++) {
            update(particles_cpy[i]);
        }
    }
    t2 = std::chrono::system_clock::now();
    printf("Updating particles on the CPU done in: %lf ms!\n",
           std::chrono::duration<double, std::chrono::milliseconds::period>(t2 - t1).count());

    for (int i = 0; i < NUM_PARTICLES; i++) {
        if (!equal(particles[i], particles_cpy[i])) {
            printf("Wrong!\n");
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

int mainManagedMemory(){
    Particle *particles;
    cudaMallocManaged(&particles, NUM_PARTICLES * sizeof(Particle));

    auto *particles_cpy = (Particle *) malloc(NUM_PARTICLES * sizeof(Particle));

    memcpy(particles_cpy, particles, NUM_PARTICLES * sizeof(Particle));

    auto t1 = std::chrono::system_clock::now();

    for(int i=0; i<NUM_ITERATIONS; i++){
        update<<<(NUM_PARTICLES + TPB - 1) / TPB, TPB>>>(particles);

        cudaDeviceSynchronize();
    }

    auto t2 = std::chrono::system_clock::now();

    printf("Updating particles on the GPU done in: %lf ms!\n",
           std::chrono::duration<double, std::chrono::milliseconds::period>(t2 - t1).count());


    t1 = std::chrono::system_clock::now();
    for(int it=0; it<NUM_ITERATIONS; it++){
        for (int i = 0; i < NUM_PARTICLES; i++) {
            update(particles_cpy[i]);
        }
    }
    t2 = std::chrono::system_clock::now();
    printf("Updating particles on the CPU done in: %lf ms!\n",
           std::chrono::duration<double, std::chrono::milliseconds::period>(t2 - t1).count());

    for (int i = 0; i < NUM_PARTICLES; i++) {
        if (!equal(particles[i], particles_cpy[i])) {
            printf("Wrong!\n");
            cudaFree(particles);
            free(particles_cpy);
            return 0;
        }
    }

    printf("Correct!\n");

    cudaFree(particles);
    free(particles_cpy);

    return 0;
}

int main(){
#if MANAGED
    return mainManagedMemory();
#else
    return mainPinnedMemory();
#endif
}
