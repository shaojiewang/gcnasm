#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#ifdef USE_MKL
#include <mkl.h>
#endif
#define HALF
#ifdef HALF
#include "half.hpp"
#endif
// #define PER_PIXEL_CHECK
#define ASSERT_ON_FAIL
#define MFMA
//#define ASM_PRINT

#ifndef ABS
#define ABS(x) ((x)>0?(x):-1*(x))
#endif

using float16=half_float::half;
static inline bool valid_vector( const float* ref, const float16* pred, int n, double nrms = 1e-3 )
{    
    double s0=0.0;
    double s1=0.0;
#ifdef PER_PIXEL_CHECK
    int pp_err = 0;
#endif
    int i_start = 0, i_end=n;
    int i_num = i_end - i_start;
    for( int i=i_start; i<i_end; ++i ){
        double ri=(double)ref[i];
        double pi=(double)pred[i];
        double d=ri-pi;
        double dd=d*d;
        double rr=2.0*ri*ri;
        s0+=dd;
        s1+=rr;
        
#ifdef PER_PIXEL_CHECK
        double delta = ABS(ri-pi)/ri;
        if(delta>1e-3){
#ifdef ASSERT_ON_FAIL
            if(pp_err<100)
            printf("diff at %4d, ref:%lf, pred:%lf(0x%04x), d:%lf\n",i,ri,pi,((uint16_t*)pred)[i],delta);
#endif
            pp_err++;
        }
#endif
    }
    printf("pp_crr:%d, pp_err:%d, crr_ratio:%.3f, nrms:%lf, s0:%lf, s1:%lf\n",i_num-pp_err, pp_err, (float)(i_num-pp_err)/(float)i_num, sqrt(s0/s1),s0,s1);

    return (sqrt(s0/s1)<nrms)
#ifdef PER_PIXEL_CHECK
        && (pp_err==0)
#endif
    ;
}

void hgemm_cr_kpack2(
    float*  ptr_c,
    const float*  __restrict__ ptr_a,
    const float*  __restrict__ ptr_b,
    float alpha,
    unsigned int m,
    unsigned int n,
    unsigned int k,
    unsigned int lda,
    unsigned int ldb,
    unsigned int ldc)
{
#ifdef USE_MKL
    cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,
        m,n,k,alpha,ptr_a,lda,ptr_b,ldb,0,ptr_c,ldc);
#else
//change the layout
    unsigned int im,in,ik;
    for(in=0;in<n;in++){
        for(im=0;im<m;im++){
            #ifndef MFMA
            float c = .0;
            for(ik=0;ik<(k>>1);ik++){
                c += ptr_a[ik*lda*2+im*2]*ptr_b[ik*ldb*2+in*2];
                c += ptr_a[ik*lda*2+im*2+1]*ptr_b[ik*ldb*2+in*2+1];
            }
            ptr_c[in*ldc+im] = alpha*c;
            #endif

            #ifdef MFMA
            float c = .0;
            for(ik=0;ik<(k>>2);ik++){
                c += ptr_a[ik*4*lda+im*4+0]*ptr_b[ik*4*ldb+in*4+0]
                   + ptr_a[ik*4*lda+im*4+1]*ptr_b[ik*4*ldb+in*4+1]
                   + ptr_a[ik*4*lda+im*4+2]*ptr_b[ik*4*ldb+in*4+2]
                   + ptr_a[ik*4*lda+im*4+3]*ptr_b[ik*4*ldb+in*4+3];
            }
            ptr_c[in*ldc+im] = alpha*c;
            #endif
        }
    }
#endif
}

#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

static inline int get_int(const char* env_name, int def_value)
{
    char * v = getenv(env_name);
    if(v)
        return atoi(v);
    return def_value;
}

void rand_vector_2d(float* v, int row, int col, int ld){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }

    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            v[r*ld+c] = ((float)(rand() % 100)) / 100.0f;
            //v[r*ld+c] = ((float)(r % 100)+1) / 100.0f + ((float)(c % 100)+1) / 1000.0f;
            //v[r*ld+c] = 1.0;
        }
    }
}

//#define HSACO "hgemm128x128.hsaco"
#define HSACO "kernel_asm.co"
//#define HSACO "hgemm_128x128_kpack2"
#define HSA_KERNEL "hgemm_128x128_kpack4"


#define HGEMM_M 960
#define HGEMM_N 1024
#define HGEMM_K 1024

int main(int argc, char ** argv){
    hipModule_t module;
    hipFunction_t kernel_func;
    hipEvent_t evt_00, evt_11;
    HIP_CALL(hipSetDevice(0));

    HIP_CALL(hipModuleLoad(&module, HSACO));
    HIP_CALL(hipModuleGetFunction(&kernel_func, module, HSA_KERNEL));

    int validate = get_int("VALIDATE", 0);
    int m = get_int("M", HGEMM_M);
    int n = get_int("N", HGEMM_N);
    int k = get_int("K", HGEMM_K);
    int lda = m*sizeof(float);
    int ldb = n*sizeof(float);
    int ldc = m*sizeof(float);
    float alpha = 1.0f;
    float *host_a, *host_b, *host_c;
    float16 *fp16_a, *fp16_b, *fp16_c, *dev_a, *dev_b, *dev_c;
    int bdx = 256;
    int gdx = ((m+127)>>7)*((n+127)>>7);

    //fp32 on host
    host_a = (float*)malloc(lda*k);
    host_b = (float*)malloc(ldb*k);
    host_c = (float*)malloc(ldc*n);
    rand_vector_2d(host_a, k, m, lda/sizeof(float));
    rand_vector_2d(host_b, k, n, ldb/sizeof(float));
    //fp16 on host
    fp16_a = (float16*)malloc(lda*(k>>1));
    fp16_b = (float16*)malloc(ldb*(k>>1));
    fp16_c = (float16*)malloc(ldc*(n>>1));
    //convert fp32 a and b into fp16 on host
    for(int i=0; i<m*k; i++)fp16_a[i]=__float2half_rn(host_a[i]);
    for(int i=0; i<n*k; i++)fp16_b[i]=__float2half_rn(host_b[i]);
    HIP_CALL(hipSetDevice(0));
    //fp16 on device
    HIP_CALL(hipMalloc(&dev_a, lda*(k>>1)));
    HIP_CALL(hipMalloc(&dev_b, ldb*(k>>1)));
    HIP_CALL(hipMalloc(&dev_c, ldc*(n>>1)));
    //fp16 cpy to device
    HIP_CALL(hipMemcpy(dev_a, fp16_a, lda*(k>>1), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_b, fp16_b, ldb*(k>>1), hipMemcpyHostToDevice));

    int total_loop=10;
    int warm_ups = 10;
    int i;

#ifdef ASM_PRINT
    //debug pointer
    float *host_print, *print;
    host_print = (float*)malloc(bdx*8);
    HIP_CALL(hipMalloc(&print, bdx*8));
#endif
    struct __attribute__((packed)) {
        void*  ptr_c;
        void*  ptr_a;
        void*  ptr_b;
        float alpha;
        unsigned int m;
        unsigned int n;
        unsigned int k;
        unsigned int lda;
        unsigned int ldb;
        unsigned int ldc;
        #ifdef ASM_PRINT
        void*  print;
        #endif
    } args;
    size_t arg_size = sizeof(args);
    args.ptr_c  = (void*)dev_c;
    args.ptr_a  = (void*)dev_a;
    args.ptr_b  = (void*)dev_b;
    args.alpha  = alpha;
    args.m      = m;
    args.n      = n;
    args.k      = k;
    args.lda    = lda;
    args.ldb    = ldb;
    args.ldc    = ldc;
    #ifdef ASM_PRINT
    args.print  = (void*)print;
    #endif
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &arg_size, HIP_LAUNCH_PARAM_END};
    
    for(i=0;i<warm_ups;i++){
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));
        //std::cout<<"safe here"<<std::endl;
    }

#ifdef ASM_PRINT
    int max_i=256;
    HIP_CALL(hipMemcpy(host_print, print, 8*max_i, hipMemcpyDeviceToHost));
    for(int i=0; i<max_i; i++){
        if(((uint32_t*)host_print)[2*i+1]!=0x5c005c00)
        printf("Thread%d, PrintVal:0x%x\n",((int*) host_print)[2*i], ((uint32_t*)host_print)[2*i+1]);
        //std::cout<<"Thread"<<((int*) host_print)[2*i]<<", PrintVal1:"<<(((float16*)host_print)[4*i+2])<<
        //", PrintVal2:"<<( ( (float16*)host_print )[4*i+3] )<<std::endl;
    }    
#endif

    hipEventCreate(&evt_00);
    hipEventCreate(&evt_11);
    hipDeviceSynchronize();
    hipEventRecord(evt_00, NULL);
    for(i=0;i<total_loop;i++)
        HIP_CALL(hipModuleLaunchKernel(kernel_func, gdx,1,1, bdx,1,1,  0, 0, NULL, (void**)&config ));

    float elapsed_ms;
    hipEventRecord(evt_11, NULL);
    hipEventSynchronize(evt_11);
    hipDeviceSynchronize();
    hipEventElapsedTime(&elapsed_ms, evt_00, evt_11);
    hipEventDestroy(evt_00);
    hipEventDestroy(evt_11);

    float time_per_loop = elapsed_ms/total_loop;
    float gflops = (float)2*m*n*k/time_per_loop/(1e6);
    printf("m:%d,n:%d,k:%d,gflops:%.3f\n",m,n,k,gflops);
    if(validate){
        hgemm_cr_kpack2(host_c, host_a, host_b, alpha, m,n,k,lda/sizeof(float),ldb/sizeof(float),ldc/sizeof(float));
        HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*(n>>1), hipMemcpyDeviceToHost));
        bool res = valid_vector( host_c, fp16_c, m*n );
        printf(",%s",res?"valid":"fail");
    }
    printf("\n");
    
    free(host_a);
    free(host_b);
    free(host_c);
    free(fp16_a);
    free(fp16_b);
    free(fp16_c);
    
    hipFree(dev_a);
    hipFree(dev_b);
    hipFree(dev_c);

#ifdef ASM_PRINT
    free(host_print);
    hipFree(print);
#endif 
    //printf("CU:%d, TIPS:%.3f(2x:%.3f, 4x:%.3f), cost:%fms per loop\n", num_cu, tips, 2*tips, 4*tips, time_per_loop);

}
