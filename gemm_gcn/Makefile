# some important env
#export KMOPTLLC="-mattr=+enable-ds128"
export KMDUMPISA=1
export KMDUMPLLVM=1

#ROCM_PATH=/opt/rocm
ROCM_PATH=/opt/rocm-3.3.0
ARCH=gfx900

# mkl path
USE_MKL=1
MKLROOT=/home/shaowang/intel/mkl/

ifeq (,$(ROCM_PATH))
	$(error "pls choose a rocm")
endif

HIPCC=$(ROCM_PATH)/hip/bin/hipcc
HCC=$(ROCM_PATH)/hcc/bin/hcc
CLANG=$(ROCM_PATH)/hcc/bin/clang
OBJDUMP=$(ROCM_PATH)/hcc/bin/llvm-objdump

CXX=$(HIPCC)

SRC=$(wildcard ./host/*.cc)
KSRC=$(wildcard ./kernel/*.cc)
KSRC_ASM=$(wildcard ./kernel/*.s)

KDUMP_ASM=$(notdir $(KSRC_ASM))

OBJS=$(patsubst %.cc,%.o,$(SRC))
KOBJS=$(patsubst %.cc,%.o,$(KSRC))
KOBJS_ASM=$(patsubst %.s,%.o,$(KSRC_ASM))

CXXFLAGS=-D__HIP_PLATFORM_HCC__=  -I$(ROCM_PATH)/hip/include -I$(ROCM_PATH)/hcc/include -I$(ROCM_PATH)/hsa/include \
	-Wall -O2  -std=c++11
LDFLAGS=-L$(ROCM_PATH)/hcc/lib -L$(ROCM_PATH)/lib -L$(ROCM_PATH)/lib64 \
	-Wl,-rpath=$(ROCM_PATH)/hcc/lib:$(ROCM_PATH)/lib -ldl -lm -lpthread -lhc_am \
	-Wl,--whole-archive -lmcwamp -lhip_hcc -lhsa-runtime64 -lhsakmt -Wl,--no-whole-archive

ifeq (1,$(USE_MKL))
	CXXFLAGS+=-DUSE_MKL -DMKL_ILP64 -m64 -I${MKLROOT}/include
	LDFLAGS+=-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
		${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a \
		-Wl,--end-group -lpthread -lm -ldl
endif

KTARGET=./gemm_gcn.co
KTARGET_ASM=./gemm_gcn_asm.co
TARGET=./gemm_gcn_test
DTARGET=./gemm_gcn_debug

.PHONY: host
.PHONY: kernel
.PHONY: clean
.PHONY: debug

all: $(TARGET) $(KTARGET) $(KTARGET_ASM)
	$(OBJDUMP) -disassemble -mcpu=$(ARCH) $(KTARGET_ASM) > dump.vega_gemm.s

$(TARGET): $(OBJS)
	$(HIPCC) $(OBJS) $(LDFLAGS) $(CXXFLAGS) -o $@

$(KTARGET): $(KSRC)
	$(HIPCC) --genco --targets $(ARCH) $(KSRC) -o $@

$(KTARGET_ASM): $(KSRC_ASM)
	$(CLANG) -x assembler -target amdgcn--amdhsa -mcpu=$(ARCH) -mno-code-object-v3 $(KSRC_ASM) -o $@
	
debug: $(DTARGET) $(KTARGET)

$(DTARGET): $(SRC)
	$(HIPCC) $(SRC) $(LDFLAGS) $(CXXFLAGS) -g -o $@

clean:
	rm -f $(TARGET) $(KTARGET) $(OBJS) $(KOBJS) $(DTARGET) $(KTARGET_ASM)
	rm -f dump*