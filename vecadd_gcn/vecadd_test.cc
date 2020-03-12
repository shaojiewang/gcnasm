/********
 * no license
 * 
********/

#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include <hip/hip_runtime.h>

#define VEC_LEN (1024 * 1024 * 16)

#define WARM_UPS 5

#define MEM_ALIGN_128 128

#define THREADS_PER_BLOCK 1024

// Device (Kernel) function, it must be void
__global__ void vector_add(float* out, float* in_0, float* in_1, const int vec_length) {
    int index = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    __shared__ float in_0_sm[1];
    __shared__ float in_1_sm[1];
    __shared__ float out_sm[1];

    in_0_sm[0] = in_0[index];
    in_1_sm[0] = in_1[index];
    __syncthreads();

    out_sm[0] = in_0_sm[0] + in_1_sm[0];

    out[0] = out_sm[0];

}

int main(int argc, char** argv)
{
    float* vector_0;
    float* vector_1;
    float* vector_out;
    float* cpu_vector_out;

    float* gpu_vector_0;
    float* gpu_vector_1;
    float* gpu_vector_out;

    float vecadd_ms = 0.f;
    
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    int i, errors;

    int length_alloc = (VEC_LEN / MEM_ALIGN_128 + 1) * MEM_ALIGN_128;

    vector_0 = (float* )memalign(MEM_ALIGN_128, length_alloc * sizeof(float));
    vector_1 = (float* )memalign(MEM_ALIGN_128, length_alloc * sizeof(float));
    vector_out = (float* )memalign(MEM_ALIGN_128, length_alloc * sizeof(float));

    cpu_vector_out = (float* )memalign(MEM_ALIGN_128, length_alloc * sizeof(float));

    // allocate the memory on the device side
    hipMalloc((void**)&gpu_vector_0, VEC_LEN * sizeof(float));
    hipMalloc((void**)&gpu_vector_1, VEC_LEN * sizeof(float));
    hipMalloc((void**)&gpu_vector_out, VEC_LEN * sizeof(float));

    // initialize the input data
    for (i = 0; i < VEC_LEN; i++) {
        vector_0[i] = (float)i * 10.0f;
        vector_1[i] = (float)i * 10.0f;
    }

    // copy data to device
    hipMemcpy(gpu_vector_0, vector_0, VEC_LEN * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(gpu_vector_1, vector_1, VEC_LEN * sizeof(float), hipMemcpyHostToDevice);

    // warm ups
    for (i = 0; i < WARM_UPS; i++)
    {
        hipLaunchKernelGGL(vector_add, dim3(VEC_LEN / THREADS_PER_BLOCK, 1),
                    dim3(THREADS_PER_BLOCK, 1), 0, 0, gpu_vector_out,
                    gpu_vector_0, gpu_vector_1, VEC_LEN);
    }

    // run kernel code vec add
    hipEventRecord(start, NULL);

    hipLaunchKernelGGL(vector_add, dim3(VEC_LEN / THREADS_PER_BLOCK, 1),
                    dim3(THREADS_PER_BLOCK, 1), 0, 0, gpu_vector_out,
                    gpu_vector_0, gpu_vector_1, VEC_LEN);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&vecadd_ms, start, stop);

    printf("vec add kernel_time (hipEventElapsedTime) =%6.6fms\n", vecadd_ms);

    // copy gpu res to ddr
    hipMemcpy(vector_out, gpu_vector_out, VEC_LEN * sizeof(float), hipMemcpyDeviceToHost);

    printf("res=[%f, %f, %f, %f]\r\n", vector_out[0], vector_out[1], vector_out[2], vector_out[3]);

    // free the resources on device side
    hipFree(gpu_vector_0);
    hipFree(gpu_vector_1);
    hipFree(gpu_vector_out);

    // free the resources on host side
    free(vector_0);
    free(vector_1);
    free(vector_out);
    free(cpu_vector_out);

    return 0;
}