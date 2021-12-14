#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#define NUM_PARTICLES 256
#define NUM_ITERATIONS 1
#define BLOCK_SIZE 64

#define RANDOM_SIZE (NUM_ITERATIONS * NUM_PARTICLES * 3)

#define EPSILON 0.00001

#define CL_CALL(x) {                                                            \
    {                                                                           \
        cl_int err = x;                                                         \
        if (err != CL_SUCCESS){                                                 \
            fprintf(stderr, "OpenCL error on line %d: %d\n", __LINE__, err);    \
        }                                                                       \
    }                                                                           \
}

const char* update_kernel = "typedef struct {                                                           \n"
                            "   float position[3];                                                      \n"
                            "   float velocity[3];                                                      \n"
                            "} Particle;                                                                \n"
                            "                                                                           \n"
                            "__kernel                                                                   \n"
                            "void update(__global Particle* particles, __global int* num_iterations, __global float* random_values, __global int* num_particles){                    \n"
                            "   int index = get_global_id(0);                                           \n"
                            "   for(int i=0; i<*num_iterations; i++){                                           \n"
                            "       for(int j=0; j<3; j++){"
                            "           particles[index].velocity[j] = random_values[i*(*num_particles)*3 + index*3 + j];                                           \n"
                            "           particles[index].position[j] += particles[index].velocity[j];   \n"
                            "       }                                                                   \n"
                            "   }                                                                       \n"
                            "}                                                                          \n";

typedef struct {
    float position[3];
    float velocity[3];
} Particle;

void update_cpu(Particle* particles, float* random_array, int num_particles, int num_iterations){
    for(int t=0; t<num_iterations; t++) {
        for (int i = 0; i < num_particles; i++) {
            for (int j = 0; j < 3; j++) {
                particles[i].velocity[j] = random_array[t*num_particles*3 + i*3 + j];
                particles[i].position[j] += particles[i].velocity[j];
            }
        }
    }
}

#define RAND_SIZE 100000

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

int main(){
    cl_platform_id * platforms;
    cl_uint n_platform;

    // Identify Platforms in the system
    CL_CALL(clGetPlatformIDs(0, NULL, &n_platform));
    platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*n_platform);
    CL_CALL(clGetPlatformIDs(n_platform, platforms, NULL));

    // Identify Devices inside platforms[0]
    cl_device_id *device_list; cl_uint n_devices;
    CL_CALL(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &n_devices));
    device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*n_devices);
    CL_CALL(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL));

    Particle particles[NUM_PARTICLES];
    Particle particle_cpy[NUM_PARTICLES];
    int array_size = sizeof(Particle) * NUM_PARTICLES;

    float random_array[RANDOM_SIZE];
    initializeRandomArray(random_array, RANDOM_SIZE);

    srand(0);
    initializeParticles(particles, NUM_PARTICLES);

    memcpy(particle_cpy, particles, array_size);

    cl_int err;

    cl_context context = clCreateContext( NULL, n_devices, device_list, NULL, NULL, &err);
    cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err);

    cl_mem particles_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, array_size, NULL, &err);
    CL_CALL(clEnqueueWriteBuffer (cmd_queue, particles_dev, CL_TRUE, 0, array_size, particles, 0, NULL, NULL));

    int num_iterations = NUM_ITERATIONS;
    cl_mem iterations_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
    CL_CALL(clEnqueueWriteBuffer (cmd_queue, iterations_dev, CL_TRUE, 0, sizeof(int), &num_iterations, 0, NULL, NULL));

    int num_particles = NUM_PARTICLES;
    cl_mem num_particles_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
    CL_CALL(clEnqueueWriteBuffer (cmd_queue, num_particles_dev, CL_TRUE, 0, sizeof(int), &num_particles, 0, NULL, NULL));

    cl_mem random_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * RANDOM_SIZE, NULL, &err);
    CL_CALL(clEnqueueWriteBuffer (cmd_queue, random_dev, CL_TRUE, 0, sizeof(float) * RANDOM_SIZE, &random_array, 0, NULL, NULL));

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&update_kernel, NULL, &err);
    err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

    if (err != CL_SUCCESS){
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr,"Build error: %s\n", buffer);
        return 0;
    }

    cl_kernel kernel = clCreateKernel(program, "update", &err);

    CL_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &particles_dev));
    CL_CALL(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &iterations_dev));
    CL_CALL(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &random_dev));
    CL_CALL(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &num_particles_dev));

    size_t n_workitem = NUM_PARTICLES;
    size_t workgroup_size = BLOCK_SIZE;

    // Launch the kernel!
    CL_CALL(clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &n_workitem, &workgroup_size, 0, NULL, NULL));

    // Transfer the data from C back
    CL_CALL(clEnqueueReadBuffer (cmd_queue, particles_dev, CL_TRUE, 0, array_size, particles, 0, NULL, NULL));

    CL_CALL(clFlush(cmd_queue));
    CL_CALL(clFinish(cmd_queue));

    update_cpu(particle_cpy, random_array, NUM_PARTICLES, NUM_ITERATIONS);

    assert_particles(particles, particle_cpy, NUM_PARTICLES);

    printf("SUCCESS\n");

    return 0;
}