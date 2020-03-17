/********
 * no license
 * 
********/

#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include <hip/hip_runtime.h>

#define __USE_LDS__ 1
#define __MAX_VEC_LEN__ 32

#define VEC_LEN (1024 * 1024 * 16)

#define WARM_UPS 5

#define MEM_ALIGN_128 128

#define THREADS_PER_BLOCK 64

// Device (Kernel) function, it must be void
__global__ void vector_add(float* out, float* in_0, float* in_1, const int vec_length) {
    assert(vec_length < 16);
    int index = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int i = 0;

#if __USE_LDS__
    __shared__ float in_0_sm[256 * __MAX_VEC_LEN__];
    __shared__ float in_1_sm[256 * __MAX_VEC_LEN__];
    __shared__ float out_sm[256 * __MAX_VEC_LEN__];

    for (i = 0; i < vec_length; i++)
    {
        in_0_sm[hipThreadIdx_x * vec_length + i] = in_0[index * vec_length + i];
        in_1_sm[hipThreadIdx_x * vec_length + i] = in_1[index * vec_length + i];
        out[index * vec_length + i] = in_0_sm[hipThreadIdx_x * vec_length + i] + in_1_sm[hipThreadIdx_x * vec_length + i];
    }
    //__syncthreads();

    //out_sm[hipThreadIdx_x] = in_0_sm[hipThreadIdx_x] + in_1_sm[hipThreadIdx_x];

    //out[index] = out_sm[hipThreadIdx_x];
    //for (i = 0; i < vec_length; i++)
    //{
    //    out[index * vec_length + i] = in_0_sm[hipThreadIdx_x * vec_length + i] + in_1_sm[hipThreadIdx_x * vec_length + i];
    //}
#else
    out[index] = in_0[index] + in_1[index];
#endif

}

void vector_add_cpu(float* out, float* in_0, float* in_1, const int vec_length)
{
    for(int i = 0; i < vec_length; i++)
    {
        out[i] = in_0[i] + in_1[i];
    }
}

int vector_add_verification(float* out_cpu, float* out_gpu, const int vec_length)
{
    for(int i = 0; i < vec_length; i++)
    {
        if(out_cpu[i] != out_gpu[i])
        {
            return -i;
        }
    }
    return 1;
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

    int len_per_thread = 2;
    
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

    // initialize the input data
    for (i = 0; i < VEC_LEN; i++) {
        vector_0[i] = (float)i * 10.0f;
        vector_1[i] = (float)i * 10.0f;
    }

    // computation reference
    vector_add_cpu(cpu_vector_out, vector_0, vector_1, VEC_LEN);

    // allocate the memory on the device side
    hipMalloc((void**)&gpu_vector_0, VEC_LEN * sizeof(float));
    hipMalloc((void**)&gpu_vector_1, VEC_LEN * sizeof(float));
    hipMalloc((void**)&gpu_vector_out, VEC_LEN * sizeof(float));

    // copy data to device
    hipMemcpy(gpu_vector_0, vector_0, VEC_LEN * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(gpu_vector_1, vector_1, VEC_LEN * sizeof(float), hipMemcpyHostToDevice);

    // warm ups
    for (i = 0; i < WARM_UPS; i++)
    {
        hipLaunchKernelGGL(vector_add, dim3(VEC_LEN / (THREADS_PER_BLOCK * len_per_thread), 1),
                    dim3(THREADS_PER_BLOCK, 1), 0, 0, gpu_vector_out,
                    gpu_vector_0, gpu_vector_1, len_per_thread);
    }

    // run kernel code vec add
    hipEventRecord(start, NULL);

    hipLaunchKernelGGL(vector_add, dim3(VEC_LEN / (THREADS_PER_BLOCK * len_per_thread), 1),
                    dim3(THREADS_PER_BLOCK, 1), 0, 0, gpu_vector_out,
                    gpu_vector_0, gpu_vector_1, len_per_thread);

    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&vecadd_ms, start, stop);

    printf("vec add kernel_time (hipEventElapsedTime) =%6.6fms\n", vecadd_ms);

    // copy gpu res to ddr
    hipMemcpy(vector_out, gpu_vector_out, VEC_LEN * sizeof(float), hipMemcpyDeviceToHost);

    // verification
    int wrong_num = vector_add_verification(cpu_vector_out, vector_out, VEC_LEN);
    if (wrong_num != 1)
    {
        std::cout << "wrong result, wrong num is: " << wrong_num << std::endl;
    }
    else
    {
        std::cout << "right gpu code!" << std::endl;
    }
    
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