#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define NUM_PARTICLES 256 * 1000
#define NUM_ITERATIONS 500

#define RAND_SIZE 10000

#define EPSILON 0.00001

#define CL_CALL(x) {                                                            \
    {                                                                           \
        cl_int err = x;                                                         \
        if (err != CL_SUCCESS){                                                 \
            fprintf(stderr, "OpenCL error on line %d: %d\n", __LINE__, err);    \
        }                                                                       \
    }                                                                           \
}

const char* update_kernel = "typedef struct {                                                                                                                   \n"
                        "   float position[3];                                                                                                                  \n"
                        "   float velocity[3];                                                                                                                  \n"
                        "} Particle;                                                                                                                            \n"
                        "                                                                                                                                       \n"
                        "__kernel                                                                                                                               \n"
                        "void update(__global Particle* particles, int num_particles, int num_iterations, int seed){                       \n"
                        "   int index = get_global_id(0);                                                                                                       \n"
                        "   for(int i=0; i<num_iterations; i++){                                                                                                \n"
                        "       for(int j=0; j<3; j++){                                                                                                         \n"
                        "           particles[index].velocity[j] = (seed + index * j * i * 1000) % 10000;                                              \n"
                        "           particles[index].position[j] += particles[index].velocity[j];                                                               \n"
                        "       }                                                                                                                               \n"
                        "   }                                                                                                                                   \n"
                        "}                                                                                                                                      \n";

typedef struct {
    float position[3];
    float velocity[3];
} Particle;



float naive_random(int seed, int t){

    return (seed + t * 1000) % RAND_SIZE;
}

void update_cpu(Particle* particles, int num_particles, int num_iterations, int seed){
    for(int t=0; t<num_iterations; t++) {
        for (int i = 0; i < num_particles; i++) {
            for (int j = 0; j < 3; j++) {
                particles[i].velocity[j] = naive_random(seed, t*i*j);
                particles[i].position[j] += particles[i].velocity[j];
            }
        }
    }
}

void initializeParticles(Particle* particles, int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<3; j++){
            particles[i].position[j] = rand() % RAND_SIZE;
            particles[i].velocity[j] = rand() % RAND_SIZE;
        }
    }
}

void initializeRandomArray(float* out_array, int size){
    for(int i=0; i<size; i++){
        out_array[i] = rand() % RAND_SIZE;
    }
}

void assert_particles(Particle* a, Particle* b, int n){
    for(int i=0; i<n; i++){
        for(int j=0; j<3; j++){
            assert(fabsf(a[i].position[j] - b[i].position[j]) < EPSILON);
            assert(fabsf(a[i].velocity[j] - b[i].velocity[j]) < EPSILON);
        }
    }
}

void update_gpu(
        cl_context context,
        cl_kernel kernel,
        cl_command_queue cmd_queue,
        cl_mem particles_dev,
        Particle* particles,
        int num_particles,
        int num_iterations,
        int block_size,
        int seed
        ){
    const int particles_size = sizeof(Particle) * num_particles;

    CL_CALL(clEnqueueWriteBuffer (cmd_queue, particles_dev, CL_TRUE, 0, particles_size, particles, 0, NULL, NULL));

    CL_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &particles_dev));
    CL_CALL(clSetKernelArg(kernel, 1, sizeof(int), &num_particles));
    CL_CALL(clSetKernelArg(kernel, 2, sizeof(int), &num_iterations));
    CL_CALL(clSetKernelArg(kernel, 3, sizeof(int), &seed));

    size_t n_workitem = NUM_PARTICLES;
    size_t workgroup_size = block_size;

    // Launch the kernel!
    CL_CALL(clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &n_workitem, &workgroup_size, 0, NULL, NULL));

    // Transfer the data from C back
    CL_CALL(clEnqueueReadBuffer (cmd_queue, particles_dev, CL_TRUE, 0, particles_size, particles, 0, NULL, NULL));

    CL_CALL(clFlush(cmd_queue));
    CL_CALL(clFinish(cmd_queue));
}

void initialize_gpu(
        cl_context *context,
        cl_kernel *kernel,
        cl_command_queue *cmd_queue,
        cl_mem *particles_dev,
        int particles_size
        ){
    cl_platform_id *platforms;
    cl_uint n_platform;

    // Identify Platforms in the system
    CL_CALL(clGetPlatformIDs(0, NULL, &n_platform));
    platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * n_platform);
    CL_CALL(clGetPlatformIDs(n_platform, platforms, NULL));

    // Identify Devices inside platforms[0]
    cl_device_id *device_list;
    cl_uint n_devices;
    CL_CALL(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &n_devices));
    device_list = (cl_device_id *) malloc(sizeof(cl_device_id) * n_devices);
    CL_CALL(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL));

    cl_int err;

    *context = clCreateContext(NULL, n_devices, device_list, NULL, NULL, &err);
    *cmd_queue = clCreateCommandQueue(*context, device_list[0], 0, &err);

    *particles_dev = clCreateBuffer(*context, CL_MEM_READ_WRITE, sizeof(Particle) * particles_size, NULL, &err);

    cl_program program = clCreateProgramWithSource(*context, 1, (const char **) &update_kernel, NULL, &err);
    err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr, "Build error: %s\n", buffer);
        exit(1);
    }

    *kernel = clCreateKernel(program, "update", &err);

    free(platforms);
}

int main(){
    Particle particles[NUM_PARTICLES];
    Particle particle_cpy_gpu[NUM_PARTICLES];
    Particle particle_cpy_cpu[NUM_PARTICLES];
    int array_size = sizeof(Particle) * NUM_PARTICLES;

    initializeParticles(particles, NUM_PARTICLES);

    cl_context context;
    cl_kernel kernel;
    cl_command_queue cmd_queue;
    cl_mem particles_dev;
    initialize_gpu(&context, &kernel, &cmd_queue, &particles_dev, NUM_PARTICLES);

    srand(0);
    const int seed = rand();

    const int num_iterations = NUM_ITERATIONS;

    const int initial_num_particles = 0;
    const int max_num_particles = NUM_PARTICLES;
    const int step_num_particles = max_num_particles / 10;

    printf("%d\n", max_num_particles);
    printf("%d\n", step_num_particles);
    printf("%d\n", initial_num_particles);

    clock_t t;
    double elapsed;

    const int initial_block_size = 0;
    const int max_block_size = 256;

    printf("%d\n", initial_block_size);
    printf("%d\n", max_block_size);

    for(int block_size=initial_block_size; block_size<=max_block_size; block_size*=2){
        memcpy(particle_cpy_gpu, particles, array_size);
        for(int i=initial_num_particles; i<=max_num_particles; i+=step_num_particles){
            t = clock();

            update_gpu(context, kernel, cmd_queue, particles_dev, particle_cpy_gpu, i, num_iterations, block_size, seed);

            elapsed = (clock() - t) * 1000 / CLOCKS_PER_SEC;
            printf("%.4f\n", elapsed);
        }
    }

    memcpy(particle_cpy_cpu, particles, array_size);
    for(int i=initial_num_particles; i<=max_num_particles; i+=step_num_particles){
        t = clock();

        update_cpu(particle_cpy_cpu, i, num_iterations, seed);

        elapsed = (clock() - t) * 1000 / CLOCKS_PER_SEC;
        printf("%.4f\n", elapsed);
    }

    assert_particles(particle_cpy_cpu, particle_cpy_gpu, NUM_PARTICLES);

    printf("SUCCESS\n");

    return 0;
}