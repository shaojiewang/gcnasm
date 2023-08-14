#!/bin/sh

KSRC=memcpy_kernel_gfx90a.s
KOUT=memcpy_kernel_gfx90a.hsaco
SRC=main.cpp
TARGET=out.exe

rm -rf $KOUT
/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa -mcpu=gfx90a $KSRC -o $KOUT

rm -rf $TARGET
/opt/rocm/hip/bin/hipcc $SRC -mcpu=gfx90a -o $TARGET
