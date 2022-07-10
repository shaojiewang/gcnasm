.text
.global memcpy_kernel
.p2align 8
.type memcpy_kernel,@function
memcpy_kernel:
; This is just an example, not the optimal one
.set s_dptr,            0
.set s_karg,            2
.set s_bx,              4

.set s_ptr_in,          6
.set s_ptr_out,         8
.set s_loops_per_block, 10
.set s_stride_block,    12
.set s_tmp,             14
.set s_gdx,             18

.set v_buf,             0
.set v_offset,          16
.set v_tmp,             32

    ; http://www.hsafoundation.com/html/Content/Runtime/Topics/02_Core/hsa_kernel_dispatch_packet_t.htm
    ;s_load_dword s[s_gdx],                  s[s_dptr:s_dptr+1], 12
    ;s_waitcnt           lgkmcnt(0)
    ;s_lshr_b32      s[s_gdx],   s[s_gdx],   8
    s_mov_b32   s[s_gdx], 60

    s_load_dwordx2 s[s_ptr_in:s_ptr_in+1],      s[s_karg:s_karg+1],     0
    s_load_dwordx2 s[s_ptr_out:s_ptr_out+1],    s[s_karg:s_karg+1],     8
    s_load_dword s[s_loops_per_block],          s[s_karg:s_karg+1],     16

    s_mul_i32 s[s_tmp+1], s[s_bx], 256*4
    v_lshlrev_b32 v[v_tmp], 2, v0
    v_add_u32 v[v_offset+0], s[s_tmp+1], v[v_tmp]

    s_mul_i32 s[s_tmp],  s[s_gdx],  256*4
    v_add_u32 v[v_offset+1],    s[s_tmp],   v[v_offset+0]
    v_add_u32 v[v_offset+2],    s[s_tmp],   v[v_offset+1]
    v_add_u32 v[v_offset+3],    s[s_tmp],   v[v_offset+2]
    v_add_u32 v[v_offset+4],    s[s_tmp],   v[v_offset+3]
    v_add_u32 v[v_offset+5],    s[s_tmp],   v[v_offset+4]
    v_add_u32 v[v_offset+6],    s[s_tmp],   v[v_offset+5]
    v_add_u32 v[v_offset+7],    s[s_tmp],   v[v_offset+6]
    v_add_u32 v[v_offset+8],    s[s_tmp],   v[v_offset+7]
    v_add_u32 v[v_offset+9],    s[s_tmp],   v[v_offset+8]
    v_add_u32 v[v_offset+10],   s[s_tmp],   v[v_offset+9]
    v_add_u32 v[v_offset+11],   s[s_tmp],   v[v_offset+10]
    v_add_u32 v[v_offset+12],   s[s_tmp],   v[v_offset+11]
    v_add_u32 v[v_offset+13],   s[s_tmp],   v[v_offset+12]
    v_add_u32 v[v_offset+14],   s[s_tmp],   v[v_offset+13]
    v_add_u32 v[v_offset+15],   s[s_tmp],   v[v_offset+14]
    s_lshl_b32 s[s_stride_block],   s[s_tmp],   4   ; unroll 16

    s_waitcnt           lgkmcnt(0)

label_memcopy_start:
    global_load_dword   v[v_buf+0],  v[v_offset+0:v_offset+1],   s[s_ptr_in:s_ptr_in+1]
    global_load_dword   v[v_buf+1],  v[v_offset+1:v_offset+2],   s[s_ptr_in:s_ptr_in+1]
    global_load_dword   v[v_buf+2],  v[v_offset+2:v_offset+3],   s[s_ptr_in:s_ptr_in+1]
    global_load_dword   v[v_buf+3],  v[v_offset+3:v_offset+4],   s[s_ptr_in:s_ptr_in+1]
    global_load_dword   v[v_buf+4],  v[v_offset+4:v_offset+5],   s[s_ptr_in:s_ptr_in+1]
    global_load_dword   v[v_buf+5],  v[v_offset+5:v_offset+6],   s[s_ptr_in:s_ptr_in+1]
    global_load_dword   v[v_buf+6],  v[v_offset+6:v_offset+7],   s[s_ptr_in:s_ptr_in+1]
    global_load_dword   v[v_buf+7],  v[v_offset+7:v_offset+8],   s[s_ptr_in:s_ptr_in+1]
    global_load_dword   v[v_buf+8],  v[v_offset+8:v_offset+9],   s[s_ptr_in:s_ptr_in+1]
    global_load_dword   v[v_buf+9],  v[v_offset+9:v_offset+10],  s[s_ptr_in:s_ptr_in+1]
    global_load_dword   v[v_buf+10], v[v_offset+10:v_offset+11], s[s_ptr_in:s_ptr_in+1]
    global_load_dword   v[v_buf+11], v[v_offset+11:v_offset+12], s[s_ptr_in:s_ptr_in+1]
    global_load_dword   v[v_buf+12], v[v_offset+12:v_offset+13], s[s_ptr_in:s_ptr_in+1]
    global_load_dword   v[v_buf+13], v[v_offset+13:v_offset+14], s[s_ptr_in:s_ptr_in+1]
    global_load_dword   v[v_buf+14], v[v_offset+14:v_offset+15], s[s_ptr_in:s_ptr_in+1]
    global_load_dword   v[v_buf+15], v[v_offset+15:v_offset+16], s[s_ptr_in:s_ptr_in+1]

    s_add_u32   s[s_ptr_in],   s[s_stride_block], s[s_ptr_in]
    s_addc_u32  s[s_ptr_in+1], s[s_ptr_in+1], 0

    s_waitcnt       vmcnt(0)

    global_store_dword  v[v_offset+0:v_offset+1],   v[v_buf+0],   s[s_ptr_out:s_ptr_out+1]
    global_store_dword  v[v_offset+1:v_offset+2],   v[v_buf+1],   s[s_ptr_out:s_ptr_out+1]
    global_store_dword  v[v_offset+2:v_offset+3],   v[v_buf+2],   s[s_ptr_out:s_ptr_out+1]
    global_store_dword  v[v_offset+3:v_offset+4],   v[v_buf+3],   s[s_ptr_out:s_ptr_out+1]
    global_store_dword  v[v_offset+4:v_offset+5],   v[v_buf+4],   s[s_ptr_out:s_ptr_out+1]
    global_store_dword  v[v_offset+5:v_offset+6],   v[v_buf+5],   s[s_ptr_out:s_ptr_out+1]
    global_store_dword  v[v_offset+6:v_offset+7],   v[v_buf+6],   s[s_ptr_out:s_ptr_out+1]
    global_store_dword  v[v_offset+7:v_offset+8],   v[v_buf+7],   s[s_ptr_out:s_ptr_out+1]
    global_store_dword  v[v_offset+8:v_offset+9],   v[v_buf+8],   s[s_ptr_out:s_ptr_out+1]
    global_store_dword  v[v_offset+9:v_offset+10],  v[v_buf+9],   s[s_ptr_out:s_ptr_out+1]
    global_store_dword  v[v_offset+10:v_offset+11], v[v_buf+10],  s[s_ptr_out:s_ptr_out+1]
    global_store_dword  v[v_offset+11:v_offset+12], v[v_buf+11],  s[s_ptr_out:s_ptr_out+1]
    global_store_dword  v[v_offset+12:v_offset+13], v[v_buf+12],  s[s_ptr_out:s_ptr_out+1]
    global_store_dword  v[v_offset+13:v_offset+14], v[v_buf+13],  s[s_ptr_out:s_ptr_out+1]
    global_store_dword  v[v_offset+14:v_offset+15], v[v_buf+14],  s[s_ptr_out:s_ptr_out+1]
    global_store_dword  v[v_offset+15:v_offset+16], v[v_buf+15],  s[s_ptr_out:s_ptr_out+1]

    s_add_u32   s[s_ptr_out],   s[s_stride_block], s[s_ptr_out]
    s_addc_u32  s[s_ptr_out+1], s[s_ptr_out+1], 0

    s_sub_u32 s[s_loops_per_block], s[s_loops_per_block], 1
    s_cmp_eq_u32 s[s_loops_per_block], 0
    s_waitcnt       vmcnt(0)
    s_cbranch_scc0  label_memcopy_start
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel memcpy_kernel
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_user_sgpr_dispatch_ptr 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 32
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: memcpy_kernel
    .symbol: memcpy_kernel.kd
    .sgpr_count: 32
    .vgpr_count: 64
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [256, 1, 1]
    .max_flat_workgroup_size: 256
    .args:
    - { .name: input,           .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: output,          .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
    - { .name: loops_per_block, .size: 4, .offset:  16, .value_kind: by_value, .value_type: i32}
    - { .name: __pack0,         .size: 4, .offset:  20, .value_kind: by_value, .value_type: i32}
...
.end_amdgpu_metadata
