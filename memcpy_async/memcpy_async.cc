#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>
#include <utility>

#define WARMUP 13
#define LOOP 50

#define RAND_INT 0

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

#define ABS(x) ((x) > 0 ? (x) : -(x))

using fp32_t = float;

using int32x4_t  = int32_t __attribute__((ext_vector_type(4)));

using fp32x1_t = fp32_t __attribute__((ext_vector_type(1)));
using fp32x2_t = fp32_t __attribute__((ext_vector_type(2)));
using fp32x4_t = fp32_t __attribute__((ext_vector_type(4)));
using fp32x16_t = fp32_t __attribute__((ext_vector_type(16)));
using index_t  = int;

template<int bytes>
struct bytes_to_vector;

template<> struct bytes_to_vector<16> { using type = fp32x4_t; };
template<> struct bytes_to_vector<8> { using type = fp32x2_t; };
template<> struct bytes_to_vector<4> { using type = fp32_t; };

template<int bytes>
using bytes_to_vector_t = typename bytes_to_vector<bytes>::type;



template<std::size_t N>
struct num { static const constexpr auto value = N; };

template <class F, std::size_t... Is>
__host__ __device__ void for_(F func, std::index_sequence<Is...>)
{
    (func(num<Is>{}), ...);
}

template <typename Kernel, typename... Args>
__global__ void kentry(Args... args)
{
    Kernel{}(args...);
}

#define BUFFER_LOAD_DWORD3 0x00020000
struct buffer_resource {
    const void * ptr;
    uint32_t range;
    uint32_t config;
};
__device__ int32x4_t make_buffer_resource(const void * ptr, uint32_t range = 0xffffffff)
{
    buffer_resource res {ptr, range, BUFFER_LOAD_DWORD3};
    return __builtin_bit_cast(int32x4_t, res);
}

__device__ void
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,
                                __attribute__((address_space(3))) uint32_t* lds_ptr,
                                index_t size,
                                index_t voffset,
                                index_t soffset,
                                index_t offset,
                                index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");


__device__ fp32x4_t llvm_amdgcn_raw_buffer_load_fp32x4(int32x4_t srsrc, index_t voffset, index_t soffset, index_t glc_slc)
                            __asm("llvm.amdgcn.raw.buffer.load.v4f32");

__device__ void llvm_amdgcn_raw_buffer_store_fp32x4(fp32x4_t vdata, int32x4_t rsrc, index_t voffset, index_t soffset, index_t glc_slc)
                            __asm("llvm.amdgcn.raw.buffer.store.v4f32");


template<typename T>
__device__ void buffer_load_dwordx4_raw(T & value, int32x4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
    static_assert(sizeof(T) == 16);
    using v_type = float __attribute__((ext_vector_type(4)));
    asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen offset:%4"
        : "+v"(reinterpret_cast<v_type&>(value)) : "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
}

__device__ void async_buffer_load_dword_v(void* smem,
                                              int32x4_t rsrc,
                                              index_t voffset,
                                              index_t /*soffset*/,
                                              index_t ioffset /*max 0xFFF*/,
                                              index_t /*flag*/       = 0)
{
    asm volatile("buffer_load_dword %1, %2, 0 offen offset:%3 lds"
                    : "=r"(smem) /*dummy dependency for smem*/
                    : "v"(voffset), "s"(rsrc), "n"(ioffset)
                    : "memory");
}

__device__ void m0_set_with_memory(index_t v)
{
    asm volatile("s_mov_b32 m0, %0" : : "s"(v) : "memory");
}

__device__ void m0_inc_with_memory(index_t v)
{
    asm volatile("s_add_u32 m0, %0, m0" : : "n"(v) : "memory");
}

template<typename T>
__device__ void buffer_store_dwordx4_raw(const T & value, int32x4_t res/*buffer resource*/, index_t v_offset, index_t s_offset, index_t i_offset/*max 0xFFF*/, index_t /*flag*/ = 0){
    static_assert(sizeof(T) == 16);
    using v_type = float __attribute__((ext_vector_type(4)));
    asm volatile("buffer_store_dwordx4 %0, %1, %2, %3 offen offset:%4"
        : : "v"(reinterpret_cast<const v_type&>(value)), "v"(v_offset), "s"(res), "s"(s_offset), "n"(i_offset) : "memory");
}

__device__ void buffer_fence(index_t cnt)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n" (cnt) : "memory");
}

template<int BLOCK_SIZE = 256, int UNROLL = 16>
__global__ void memcpy_persistent(void * src,
    void * dst,
    int bytes)
{
    // bytes should align to dword x4
    int grids = gridDim.x;
    int total = bytes / 16; // dwordx4

    int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    int32x4_t src_r = make_buffer_resource(src, bytes);
    int32x4_t dst_r = make_buffer_resource(dst, bytes);

    fp32x4_t d[UNROLL];

    while(index < total) {
        for(auto u = 0; u < UNROLL; u++) {
            buffer_load_dwordx4_raw(d[u], src_r, (index + u * grids) * sizeof(int32x4_t), 0, 0);
        }

        for_(
            [&] (auto uu) {
                // std::get<i.value>(t); // do stuff
                constexpr auto u = uu.value;
                buffer_fence(UNROLL - 1 - u);
                buffer_store_dwordx4_raw(d[u], dst_r, (index + u * grids) * sizeof(int32x4_t), 0, 0);
            },
            std::make_index_sequence<UNROLL>{});

        index += UNROLL * grids;
    }
}

template<typename T>
__device__ __forceinline__ T nt_load(const T& ref)
{
    return __builtin_nontemporal_load(&ref);
}

template<typename T>
__device__ __forceinline__ void nt_store(const T& value, T& ref) {
    __builtin_nontemporal_store(value, &ref);
}

template<int BLOCK_SIZE = 256, int CHUNKS = 8>
struct memcpy_stream_async{
    struct karg {
        void * src;
        void * dst;
        int64_t bytes;
    };

    static constexpr int bytes_per_issue = 16;  // dwordx4
    static constexpr int issues_per_block = BLOCK_SIZE * CHUNKS;
    using vector_type = bytes_to_vector_t<bytes_per_issue>;
    using harg = karg;
    karg a;
    dim3 grids;

    __host__ memcpy_stream_async(harg harg_) {
        a = harg_;
        auto elems = a.bytes / bytes_per_issue;
        grids = dim3((elems + issues_per_block - 1) / issues_per_block);
    }

    __host__ void operator()() const
    {
        kentry<kernel><<<grids, BLOCK_SIZE>>>(a);
    }

    struct kernel {
        __device__ void operator()(harg args){
            int64_t base_offset = blockIdx.x * BLOCK_SIZE * bytes_per_issue;
            // int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x);
            int wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
            int32_t total = static_cast<int32_t>((args.bytes - base_offset) / bytes_per_issue);

            vector_type * p_dst = reinterpret_cast<vector_type*>(reinterpret_cast<uint8_t*>(args.dst) + base_offset);
            int32x4_t src_r = make_buffer_resource(reinterpret_cast<uint8_t*>(args.src) + base_offset, args.bytes - base_offset);

            __shared__ char smem[BLOCK_SIZE * sizeof(vector_type)];

            m0_set_with_memory(64 * sizeof(float) * wave_id);

            constexpr int loops_per_vector = sizeof(vector_type)/sizeof(float);
            for(auto c = 0; c < CHUNKS; c++) {
                for_([&](auto i) {
                    async_buffer_load_dword_v(smem, src_r,
                            c * gridDim.x * BLOCK_SIZE * sizeof(vector_type) + threadIdx.x * sizeof(float),
                            0, i.value * sizeof(float) * BLOCK_SIZE);
                    // m0_inc_with_memory(BLOCK_SIZE * sizeof(float));
                    },
                    std::make_index_sequence<loops_per_vector>{}
                );

                buffer_fence(0);
                __builtin_amdgcn_s_barrier();

                auto d = reinterpret_cast<vector_type*>(smem)[threadIdx.x];

                auto current = threadIdx.x + c * gridDim.x * BLOCK_SIZE;
                if(current < total)
                    p_dst[current] = d;
                __builtin_amdgcn_s_barrier();
            }
        }
    };
};

template<int BLOCK_SIZE = 256, int CHUNKS = 8>
struct memcpy_stream{
    struct karg {
        void * src;
        void * dst;
        int64_t bytes;
    };

    static constexpr int bytes_per_issue = 16;  // dwordx4
    static constexpr int issues_per_block = BLOCK_SIZE * CHUNKS;
    using vector_type = bytes_to_vector_t<bytes_per_issue>;
    using harg = karg;
    karg a;
    dim3 grids;

    __host__ memcpy_stream(harg harg_) {
        a = harg_;
        auto elems = a.bytes / bytes_per_issue;
        grids = dim3((elems + issues_per_block - 1) / issues_per_block);
    }

    __host__ void operator()() const
    {
        kentry<kernel><<<grids, BLOCK_SIZE>>>(a);
    }

    struct kernel {
        __device__ void operator()(harg args){
            int64_t base_offset = blockIdx.x * BLOCK_SIZE * bytes_per_issue;
            int total = static_cast<int32_t>((args.bytes - base_offset) / bytes_per_issue);
            fp32x4_t * p_src = reinterpret_cast<fp32x4_t*>(reinterpret_cast<uint8_t*>(args.src) + base_offset);
            fp32x4_t * p_dst = reinterpret_cast<fp32x4_t*>(reinterpret_cast<uint8_t*>(args.dst) + base_offset);
            for(auto c = 0; c < CHUNKS; c++) {
                auto current = threadIdx.x + c * gridDim.x * BLOCK_SIZE;
                if(current < total) {
                    //auto d = nt_load(p_src[current]);
                    //nt_store(d, p_dst[current]);
                    p_dst[current] = p_src[current];
                }
            }
        }
    };
};

template<int BLOCK_SIZE = 256, int CHUNKS = 8, int INNER = 1>
struct memcpy_stream_swizzled{
    struct karg {
        void * src;
        void * dst;
        int64_t bytes;
    };

    static constexpr int bytes_per_issue = 16;  // dwordx4
    static constexpr int issues_per_block = BLOCK_SIZE * CHUNKS * INNER;
    using vector_type = bytes_to_vector_t<bytes_per_issue>;
    using harg = karg;
    karg a;
    dim3 grids;

    __host__ memcpy_stream_swizzled(harg harg_) {
        a = harg_;
        auto elems = a.bytes / bytes_per_issue;
        grids = dim3((elems + issues_per_block - 1) / issues_per_block);
    }

    __host__ void operator()() const
    {
        kentry<kernel><<<grids, BLOCK_SIZE>>>(a);
    }

    struct kernel {
        __device__ void operator()(harg args){
            // simulate mfma 32x32x8(16)
            constexpr int total_waves = BLOCK_SIZE / 64;
            int wave_id = threadIdx.x / 64;
            int lane_id = threadIdx.x % 64;
            int lo_hi = lane_id / 32;
            int sub_lane_id = lane_id % 32;

            int64_t base_offset = blockIdx.x * BLOCK_SIZE * INNER * bytes_per_issue;
            int idx = (lo_hi * 8 + wave_id * 16 + sub_lane_id * total_waves * 16) * INNER;

            // int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x);
            int total = (args.bytes - base_offset) / 2 ; /// 16; // dwordx4
            ushort * p_src = reinterpret_cast<ushort*>(reinterpret_cast<uint8_t*>(args.src) + base_offset);
            ushort * p_dst = reinterpret_cast<ushort*>(reinterpret_cast<uint8_t*>(args.dst) + base_offset);
            for(auto c = 0; c < CHUNKS; c++) {
                for(auto i = 0; i < INNER; i++) {
                    auto current = idx + c * gridDim.x * BLOCK_SIZE * 8 * INNER + i * 8;
                    if(current < total) {
                        //auto d = nt_load(p_src[current]);
                        //nt_store(d, p_dst[current]);
                        *reinterpret_cast<fp32x4_t*>(p_dst + current) = *reinterpret_cast<fp32x4_t*>(p_src + current);
                    }
                }
            }
        }
    };
};

#ifdef RAND_INT
#define PER_PIXEL_CHECK
#endif

template<typename T>
int valid_vector_integer(const T* lhs, const T * rhs, size_t len){
    int err_cnt = 0;
    for(size_t i = 0;i < len; i++){
        if(lhs[i] == rhs[i])
            ;
        else{
            printf(" diff at %d, lhs:%d, rhs:%d\n", (int)i, lhs[i], rhs[i]);
            err_cnt++;
        }
    }
    return err_cnt;
}

template<typename T>
void rand_vector(T* v, int pixels)
{
    for(auto i = 0; i < pixels; i++) {
        v[i] =  ((T)(rand() % 10)) - 5;
    }
}

static inline void b2s(size_t bytes, char * str){
	if(bytes<1024){
		sprintf(str, "%luB", bytes);
	}else if(bytes<(1024*1024)){
		double b= (double)bytes/1024.0;
		sprintf(str, "%.2fKB", b);
	}else if(bytes<(1024*1024*1024)){
		double b= (double)bytes/(1024.0*1024);
		sprintf(str, "%.2fMB", b);
	}else{
		double b= (double)bytes/(1024.0*1024*1024);
		sprintf(str, "%.2fGB", b);
	}
}

template<typename K>
float bench_kernel(K k, int warmup, int loop)
{
    hipEvent_t start_ev, stop_ev;
    HIP_CALL(hipEventCreate(&start_ev));
    HIP_CALL(hipEventCreate(&stop_ev));

    for(int i=0;i<warmup;i++)
        k();

    HIP_CALL(hipEventRecord(start_ev, 0));
    for(int i=0;i<loop;i++)
        k();
    HIP_CALL(hipEventRecord( stop_ev, 0 ));
    HIP_CALL(hipEventSynchronize(stop_ev));

    float ms;
    HIP_CALL(hipEventElapsedTime(&ms,start_ev, stop_ev));
    ms = ms / loop;
    return ms;
}


int main(int argc, char ** argv)
{
    int64_t pixels = 64 * 1024 * 1024;
    if (argc >= 2)
        pixels = atoll(argv[1]);

    int *host_a, *host_b;
    void *dev_a, *dev_b;

    //fp32 on host
    host_a = (int*)malloc(pixels*sizeof(int));
    host_b = (int*)malloc(pixels*sizeof(int));

    rand_vector(host_a, pixels);

    HIP_CALL(hipMalloc(&dev_a, pixels*sizeof(int)));
    HIP_CALL(hipMalloc(&dev_b, pixels*sizeof(int)));

    HIP_CALL(hipMemcpy(dev_a, host_a, pixels*sizeof(int), hipMemcpyHostToDevice));

    int num_cu = [&](){
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        HIP_CALL(hipGetDevice( &dev ));
        HIP_CALL(hipGetDeviceProperties( &dev_prop, dev ));
        return dev_prop.multiProcessorCount;
    }();

    char str[64];
    b2s(pixels*sizeof(float), str);
    printf("%s ", str); fflush(stdout);
    auto get_gbps = [](double bytes, float ms_){
        return  ((double)bytes)/((double)ms_/1000)/1000000000.0;
    };

    {
        auto k = [=](){
            constexpr int block_size = 1024;
            int grids = num_cu * 2;
            return [=](){
                // printf("xxxxx\n");
                memcpy_persistent<<<grids, block_size>>>(dev_a, dev_b, pixels * 4);
            };
        }();
        HIP_CALL(hipMemset(dev_b, 0, pixels*sizeof(int)));
        auto ms = bench_kernel(k, WARMUP, LOOP);
        HIP_CALL(hipMemcpy(host_b, dev_b, pixels*sizeof(int), hipMemcpyDeviceToHost));
        int res = valid_vector_integer(host_b, host_a, pixels);
        (void) res;
        printf("%.3f(GB/s) ", get_gbps(pixels * 2 * sizeof(int), ms)); fflush(stdout);
    }

    {
        auto k = [=](){
            constexpr int block_size = 256;
            constexpr int chunks = 4;
            auto k = memcpy_stream<block_size, chunks>{{dev_a, dev_b, pixels * 4}};
            return k;
        }();
        HIP_CALL(hipMemset(dev_b, 0, pixels*sizeof(int)));
        auto ms = bench_kernel(k, WARMUP, LOOP);
        HIP_CALL(hipMemcpy(host_b, dev_b, pixels*sizeof(int), hipMemcpyDeviceToHost));
        int res = valid_vector_integer(host_b, host_a, pixels);
        (void) res;
        printf("%.3f(GB/s) ", get_gbps(pixels * 2 * sizeof(int), ms)); fflush(stdout);
    }

    {
        auto k = [=](){
            constexpr int block_size = 256;
            constexpr int chunks = 4;
            auto k = memcpy_stream_async<block_size, chunks>{{dev_a, dev_b, pixels * 4}};
            return k;
        }();
        HIP_CALL(hipMemset(dev_b, 0, pixels*sizeof(int)));
        auto ms = bench_kernel(k, WARMUP, LOOP);
        HIP_CALL(hipMemcpy(host_b, dev_b, pixels*sizeof(int), hipMemcpyDeviceToHost));
        int res = valid_vector_integer(host_b, host_a, pixels);
        (void) res;
        printf("%.3f(GB/s) ", get_gbps(pixels * 2 * sizeof(int), ms)); fflush(stdout);
    }

    {
        auto k = [=](){
            constexpr int block_size = 256;
            constexpr int chunks = 4;
            auto k = memcpy_stream_swizzled<block_size, chunks>{{dev_a, dev_b, pixels * 4}};
            return k;
        }();
        HIP_CALL(hipMemset(dev_b, 0, pixels*sizeof(int)));
        auto ms = bench_kernel(k, WARMUP, LOOP);
        HIP_CALL(hipMemcpy(host_b, dev_b, pixels*sizeof(int), hipMemcpyDeviceToHost));
        int res = valid_vector_integer(host_b, host_a, pixels);
        (void) res;
        printf("%.3f(GB/s) ", get_gbps(pixels * 2 * sizeof(int), ms)); fflush(stdout);
    }

    printf("\n");
    fflush(stdout);
    free(host_a);
    free(host_b);

    HIP_CALL(hipFree(dev_a));
    HIP_CALL(hipFree(dev_b));
}
