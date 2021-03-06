.hsa_code_object_version 2,1
.hsa_code_object_isa
.text
.p2align 8
.amdgpu_hsa_kernel sgemm_128x64

.macro .v_u32_div_ss v_q, s_n, s_d, v_tmp4, s_tmp4
     v_cvt_f32_u32     v[\v_tmp4+0], s[\s_d]
     v_rcp_f32         v[\v_tmp4+0], v[\v_tmp4+0]
     v_mul_f32         v[\v_tmp4+0], 0x4f800000, v[\v_tmp4+0]
     v_cvt_u32_f32     v[\v_tmp4+0], v[\v_tmp4+0]
     v_mul_lo_u32      v[\v_tmp4+1], s[\s_d], v[\v_tmp4+0]
     v_mul_hi_u32      v[\v_tmp4+2], s[\s_d], v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+3], vcc, 0, v[\v_tmp4+1]
     v_cmp_ne_i32      s[\s_tmp4:\s_tmp4+1], 0, v[\v_tmp4+2]
     v_cndmask_b32     v[\v_tmp4+1], v[\v_tmp4+3], v[\v_tmp4+1], s[\s_tmp4:\s_tmp4+1]
     v_mul_hi_u32      v[\v_tmp4+1], v[\v_tmp4+1], v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+2], vcc, v[\v_tmp4+0], v[\v_tmp4+1]
     v_add_co_u32      v[\v_tmp4+0], vcc, v[\v_tmp4+0], v[\v_tmp4+1]
     v_cndmask_b32     v[\v_tmp4+0], v[\v_tmp4+0], v[\v_tmp4+2], s[\s_tmp4:\s_tmp4+1]
     v_mul_hi_u32      v[\v_tmp4+0], s[\s_n], v[\v_tmp4+0]
     v_mul_lo_u32      v[\v_tmp4+1], s[\s_d], v[\v_tmp4+0]
     v_sub_co_u32      v[\v_tmp4+2], vcc, s[\s_n], v[\v_tmp4+1]
     v_cmp_ge_u32      s[\s_tmp4:\s_tmp4+1], s[\s_n], v[\v_tmp4+1]
     v_cmp_le_u32      s[\s_tmp4+2:\s_tmp4+3],  s[\s_d], v[\v_tmp4+2]
     v_add_co_u32      v[\v_tmp4+2], vcc, 1, v[\v_tmp4+0]
     s_and_b64         s[\s_tmp4+2:\s_tmp4+3], s[\s_tmp4:\s_tmp4+1], s[\s_tmp4+2:\s_tmp4+3]
     v_add_co_u32      v[\v_tmp4+1], vcc, -1, v[\v_tmp4+0]
     v_cndmask_b32     v[\v_tmp4+2], v[\v_tmp4+0], v[\v_tmp4+2], s[\s_tmp4+2:\s_tmp4+3]
     v_cndmask_b32     v[\v_tmp4+2], v[\v_tmp4+1], v[\v_tmp4+2], s[\s_tmp4:\s_tmp4+1]
     v_cmp_ne_i32      vcc, s[\s_d], 0
     v_cndmask_b32     v[\v_q], -1, v[\v_tmp4+2], vcc
.endm

.macro .s_fma4x8 c, a, b ; real 4x8
    v_mac_f32 v[\c+0 ], v[\a+0],  v[\b+0]
    ;s_setprio 1
    v_mac_f32 v[\c+1 ], v[\a+1],  v[\b+0]
    v_mac_f32 v[\c+2 ], v[\a+2],  v[\b+0]
    v_mac_f32 v[\c+3 ], v[\a+3],  v[\b+0]
    v_mac_f32 v[\c+4 ], v[\a+0],  v[\b+1]
    v_mac_f32 v[\c+5 ], v[\a+1],  v[\b+1]
    v_mac_f32 v[\c+6 ], v[\a+2],  v[\b+1]
    v_mac_f32 v[\c+7 ], v[\a+3],  v[\b+1]
    v_mac_f32 v[\c+8 ], v[\a+0],  v[\b+2]
    v_mac_f32 v[\c+9 ], v[\a+1],  v[\b+2]
    v_mac_f32 v[\c+10], v[\a+2],  v[\b+2]
    v_mac_f32 v[\c+11], v[\a+3],  v[\b+2]
    v_mac_f32 v[\c+12], v[\a+0],  v[\b+3]
    v_mac_f32 v[\c+13], v[\a+1],  v[\b+3]
    v_mac_f32 v[\c+14], v[\a+2],  v[\b+3]
    v_mac_f32 v[\c+15], v[\a+3],  v[\b+3]

    v_mac_f32 v[\c+16], v[\a+0],  v[\b+4]
    v_mac_f32 v[\c+17], v[\a+1],  v[\b+4]
    v_mac_f32 v[\c+18], v[\a+2],  v[\b+4]
    v_mac_f32 v[\c+19], v[\a+3],  v[\b+4]
    v_mac_f32 v[\c+20], v[\a+0],  v[\b+5]
    v_mac_f32 v[\c+21], v[\a+1],  v[\b+5]
    v_mac_f32 v[\c+22], v[\a+2],  v[\b+5]
    v_mac_f32 v[\c+23], v[\a+3],  v[\b+5]
    v_mac_f32 v[\c+24], v[\a+0],  v[\b+6]
    v_mac_f32 v[\c+25], v[\a+1],  v[\b+6]
    v_mac_f32 v[\c+26], v[\a+2],  v[\b+6]
    v_mac_f32 v[\c+27], v[\a+3],  v[\b+6]
    v_mac_f32 v[\c+28], v[\a+0],  v[\b+7]
    v_mac_f32 v[\c+29], v[\a+1],  v[\b+7]
    v_mac_f32 v[\c+30], v[\a+2],  v[\b+7]
    v_mac_f32 v[\c+31], v[\a+3],  v[\b+7]

    ;s_setprio 0

.endm

.macro .s_fma4x8_fake c, a, b ; fake 4x8
    v_mac_f32 v[\c+0 ], v[\a+0],  v[\b+0]
    s_setprio 1
    v_mac_f32 v[\c+1 ], v[\a+1],  v[\b+0]
    v_mac_f32 v[\c+2 ], v[\a+2],  v[\b+0]
    v_mac_f32 v[\c+3 ], v[\a+3],  v[\b+0]
    v_mac_f32 v[\c+4 ], v[\a+0],  v[\b+1]
    v_mac_f32 v[\c+5 ], v[\a+1],  v[\b+1]
    v_mac_f32 v[\c+6 ], v[\a+2],  v[\b+1]
    v_mac_f32 v[\c+7 ], v[\a+3],  v[\b+1]
    v_mac_f32 v[\c+8 ], v[\a+0],  v[\b+2]
    v_mac_f32 v[\c+9 ], v[\a+1],  v[\b+2]
    v_mac_f32 v[\c+10], v[\a+2],  v[\b+2]
    v_mac_f32 v[\c+11], v[\a+3],  v[\b+2]
    v_mac_f32 v[\c+12], v[\a+0],  v[\b+3]
    v_mac_f32 v[\c+13], v[\a+1],  v[\b+3]
    v_mac_f32 v[\c+14], v[\a+2],  v[\b+3]
    v_mac_f32 v[\c+15], v[\a+3],  v[\b+3]

    v_mac_f32 v[\c+16], v[\a+0],  v[\b+4]
    v_mac_f32 v[\c+17], v[\a+1],  v[\b+4]
    v_mac_f32 v[\c+18], v[\a+2],  v[\b+4]
    v_mac_f32 v[\c+19], v[\a+3],  v[\b+4]
    v_mac_f32 v[\c+20], v[\a+0],  v[\b+5]
    v_mac_f32 v[\c+21], v[\a+1],  v[\b+5]
    v_mac_f32 v[\c+22], v[\a+2],  v[\b+5]
    v_mac_f32 v[\c+23], v[\a+3],  v[\b+5]
    v_mac_f32 v[\c+24], v[\a+0],  v[\b+6]
    v_mac_f32 v[\c+25], v[\a+1],  v[\b+6]
    v_mac_f32 v[\c+26], v[\a+2],  v[\b+6]
    v_mac_f32 v[\c+27], v[\a+3],  v[\b+6]
    v_mac_f32 v[\c+28], v[\a+0],  v[\b+7]
    v_mac_f32 v[\c+29], v[\a+1],  v[\b+7]
    v_mac_f32 v[\c+30], v[\a+2],  v[\b+7]
    v_mac_f32 v[\c+31], v[\a+3],  v[\b+7]


    v_mac_f32 v[\c+0 ], v[\a+0],  v[\b+0]
    v_mac_f32 v[\c+1 ], v[\a+1],  v[\b+0]
    v_mac_f32 v[\c+2 ], v[\a+2],  v[\b+0]
    v_mac_f32 v[\c+3 ], v[\a+3],  v[\b+0]
    v_mac_f32 v[\c+4 ], v[\a+0],  v[\b+1]
    v_mac_f32 v[\c+5 ], v[\a+1],  v[\b+1]
    v_mac_f32 v[\c+6 ], v[\a+2],  v[\b+1]
    v_mac_f32 v[\c+7 ], v[\a+3],  v[\b+1]
    v_mac_f32 v[\c+8 ], v[\a+0],  v[\b+2]
    v_mac_f32 v[\c+9 ], v[\a+1],  v[\b+2]
    v_mac_f32 v[\c+10], v[\a+2],  v[\b+2]
    v_mac_f32 v[\c+11], v[\a+3],  v[\b+2]
    v_mac_f32 v[\c+12], v[\a+0],  v[\b+3]
    v_mac_f32 v[\c+13], v[\a+1],  v[\b+3]
    v_mac_f32 v[\c+14], v[\a+2],  v[\b+3]
    v_mac_f32 v[\c+15], v[\a+3],  v[\b+3]

    v_mac_f32 v[\c+16], v[\a+0],  v[\b+4]
    v_mac_f32 v[\c+17], v[\a+1],  v[\b+4]
    v_mac_f32 v[\c+18], v[\a+2],  v[\b+4]
    v_mac_f32 v[\c+19], v[\a+3],  v[\b+4]
    v_mac_f32 v[\c+20], v[\a+0],  v[\b+5]
    v_mac_f32 v[\c+21], v[\a+1],  v[\b+5]
    v_mac_f32 v[\c+22], v[\a+2],  v[\b+5]
    v_mac_f32 v[\c+23], v[\a+3],  v[\b+5]
    v_mac_f32 v[\c+24], v[\a+0],  v[\b+6]
    v_mac_f32 v[\c+25], v[\a+1],  v[\b+6]
    v_mac_f32 v[\c+26], v[\a+2],  v[\b+6]
    v_mac_f32 v[\c+27], v[\a+3],  v[\b+6]
    v_mac_f32 v[\c+28], v[\a+0],  v[\b+7]
    v_mac_f32 v[\c+29], v[\a+1],  v[\b+7]
    v_mac_f32 v[\c+30], v[\a+2],  v[\b+7]
    v_mac_f32 v[\c+31], v[\a+3],  v[\b+7]

    s_setprio 0

.endm

.set k_ptr_c,           0
.set k_ptr_a,           8
.set k_ptr_b,           16
.set k_alpha,           24
.set k_m,               28
.set k_n,               32
.set k_k,               36
.set k_lda,             40
.set k_ldb,             44
.set k_ldc,             48
.set k_end,             52

.set s_ka,              0
.set s_bx,              2
.set s_by,              3
.set s_ptr_c,           4
.set s_ptr_a,           6
.set s_ptr_b,           8
.set s_alpha,           10
.set s_m,               11
.set s_n,               12
.set s_k,               13
.set s_lda,             14
.set s_ldb,             15
.set s_ldc,             16
.set s_bs_a,            17
.set s_bs_b,            18
.set s_m_blocks,        19
.set s_kitr,            20
.set s_wave_id,         21
.set s_m_idx,           22
.set s_n_idx,           23
.set s_tmp,             24
.set s_end,             s_tmp+13

.set v_c,               0   ; for 64x128 kernel, only use 32 vc
.set v_a0,              32  ; 4 vpgr for v_a0
.set v_a1,              36  ; 4 vgpr for v_a1
.set v_b0,              40  ; 8 vgpr for v_b0
.set v_b1,              48  ; 8 vgpr for v_b1
.set v_p0,              56  ; 2 vpgr for v_p0 here can insert 2 vpgrs(58 and 59)
.set v_q0,              60  ; 4 vpgr for v_q0
.set v_lane_k,          58  ; 1 vgpr for v_lane_k insert to 58
.set v_lane_l,          59  ; 1 vgpr for v_lane_l insert to 59
.set v_lane_id,         64  ; 1 vpgr for v_lane_id
.set v_wave_p,          65  ; 1 vpgr for v_wave_p
.set v_wave_q,          66  ; 1 vpgr for v_wave_q
.set v_lane_lo,         67  ; 1 vpgr for v_lane_lo
.set v_lane_hi,         68  ; 1 vpgr for v_lane_hi
.set v_lane_w,          69  ; 1 vpgr for v_lane_w
.set v_lane_u,          70  ; 1 vpgr for v_lane_u
.set v_lane_v,          71  ; 1 vpgr for v_lane_v
.set v_smem_store_a,    72  ; 1 vpgr for v_smem_store_a
.set v_smem_store_b,    73  ; 1 vpgr for v_smem_store_b
.set v_os_a,            74  ; 1 vpgr for v_os_a
.set v_os_b,            75  ; 1 vpgr for v_os_b
.set v_os_c,            76  ; 1 vpgr for v_os_c
.set v_smem_load_a,     77  ; 1 vpgr for v_smem_load_a
.set v_smem_load_b,     78  ; 1 vpgr for v_smem_load_b
.set v_smem_store_c,    79  ; 1 vpgr for v_smem_store_c
.set v_smem_load_c,     80  ; 1 vpgr for v_smem_load_c
.set v_lane_m,          81  ; 1 vpgr for v_lane_m
.set v_lane_lm,         82  ; 1 vpgr for v_lane_lm
.set v_tmp,             84  
.set v_end,             v_tmp+3
.set v_p1,              64  ; 2 vpgr for v_p1
.set v_q1,              68  ; 4 vpgr for v_q1

sgemm_128x64:
    .amd_kernel_code_t
        enable_sgpr_kernarg_segment_ptr = 1         ;
        user_sgpr_count = 2
        enable_sgpr_workgroup_id_x = 1              ;        blockIdx.x
        enable_sgpr_workgroup_id_y = 1              ;        blockIdx.y

        enable_vgpr_workitem_id = 0
        is_ptr64 = 1
        float_mode = 192
        workgroup_group_segment_byte_size = 16384
        kernarg_segment_byte_size = k_end
        wavefront_sgpr_count = s_end+1+2*3          ; VCC, FLAT_SCRATCH and XNACK must be counted
        workitem_vgpr_count = v_end+1
        granulated_workitem_vgpr_count = v_end/4    ; (workitem_vgpr_count-1)/4
        granulated_wavefront_sgpr_count = (s_end+2*3)/8     ; (wavefront_sgpr_count-1)/8
    .end_amd_kernel_code_t

    ; debug vgpr
    ;v_lshlrev_b32 v[32], 2, v0
    ;s_load_dwordx2 s[s_tmp+10:s_tmp+11], s[s_ka:s_ka+1], k_ptr_c

    ; kernel params
    s_load_dwordx4 s[s_ptr_c:s_ptr_c+3], s[s_ka:s_ka+1], k_ptr_c
    s_load_dwordx4 s[s_ptr_b:s_ptr_b+3], s[s_ka:s_ka+1], k_ptr_b
    s_load_dwordx4 s[s_n:s_n+3], s[s_ka:s_ka+1], k_n
    s_load_dword s[s_ldc], s[s_ka:s_ka+1], k_ldc

    v_and_b32 v[v_lane_id], 63, v0
    v_lshrrev_b32 v[v_tmp+3], 6, v0
    v_lshrrev_b32 v[v_wave_p], 1, v[v_tmp+3]
    v_and_b32 v[v_wave_q], 1, v[v_tmp+3]

    v_readfirstlane_b32 s[s_wave_id], v[v_tmp+3] ; in one wavefront, all workitem has the
                                                 ; wave id, so it can be stored in sgpr
    v_and_b32 v[v_lane_lo], 31, v[v_lane_id]
    v_lshrrev_b32 v[v_lane_hi], 5, v[v_lane_id]
    v_lshrrev_b32 v[v_lane_w], 4, v[v_lane_id]
    v_and_b32 v[v_lane_k], 15, v[v_lane_id]
    v_lshrrev_b32 v[v_lane_u], 1, v[v_lane_k]
    v_and_b32 v[v_lane_v], 1, v[v_lane_k]
    v_lshrrev_b32 v[v_tmp], 1, v[v_lane_w]
    v_lshlrev_b32 v[v_lane_l], 2, v[v_tmp]
    v_and_b32 v[v_lane_m], 1, v[v_lane_w]
    v_or_b32 v[v_lane_lm], v[v_lane_m], v[v_lane_l]

    v_lshlrev_b32 v[v_smem_store_a], 3, v0
    v_lshlrev_b32 v[v_smem_store_b], 4, v0

    .cnt=0
    .rept 32
        v_mov_b32 v[.cnt], 0
        .cnt=.cnt+1
    .endr

    s_waitcnt lgkmcnt(0)
    s_add_u32 s[s_tmp], 63, s[s_m]
    s_lshr_b32 s[s_m_blocks], s[s_tmp], 6
    .v_u32_div_ss v_os_a, s_bx, s_m_blocks, v_tmp, s_tmp
    v_readfirstlane_b32 s[s_n_idx], v[v_os_a]
    s_mul_i32 s[s_tmp], s[s_m_blocks], s[s_n_idx]
    s_sub_u32 s[s_m_idx], s[s_bx], s[s_tmp]
    s_lshl_b32 s[s_bs_a], s[s_lda], 3
    s_lshl_b32 s[s_bs_b], s[s_ldb], 3

    s_lshl_b32 s[s_tmp], s[s_m_idx], 8
    s_lshl_b32 s[s_tmp+1], s[s_n_idx], 9
    s_lshl_b32 s[s_tmp+3], s[s_wave_id], 1
    s_mul_i32 s[s_tmp+2], s[s_tmp+3], s[s_lda]
    s_mul_i32 s[s_tmp+3], s[s_tmp+3], s[s_ldb]
    s_add_u32 s[s_tmp], s[s_tmp], s[s_tmp+2]
    s_add_u32 s[s_tmp+1], s[s_tmp+1], s[s_tmp+3]

    s_add_u32 s[s_ptr_a], s[s_ptr_a], s[s_tmp]
    s_addc_u32 s[s_ptr_a+1], s[s_ptr_a+1], 0
    s_add_u32 s[s_ptr_b], s[s_ptr_b], s[s_tmp+1]
    s_addc_u32 s[s_ptr_b+1], s[s_ptr_b+1], 0

    v_mul_lo_u32 v[v_tmp], s[s_lda], v[v_lane_hi]
    v_mul_lo_u32 v[v_tmp+1], s[s_ldb], v[v_lane_hi]
    v_lshl_add_u32 v[v_os_a], v[v_lane_lo], 3, v[v_tmp]
    v_lshl_add_u32 v[v_os_b], v[v_lane_lo], 4, v[v_tmp+1]  

    global_load_dwordx2 v[v_p0:v_p0+1], v[v_os_a:v_os_a+1], s[s_ptr_a:s_ptr_a+1]
    global_load_dwordx4 v[v_q0:v_q0+3], v[v_os_b:v_os_b+1], s[s_ptr_b:s_ptr_b+1]
    s_add_u32 s[s_ptr_a], s[s_ptr_a], s[s_bs_a]
    s_addc_u32 s[s_ptr_a+1], s[s_ptr_a+1], 0
    s_add_u32 s[s_ptr_b], s[s_ptr_b], s[s_bs_b]
    s_addc_u32 s[s_ptr_b+1], s[s_ptr_b+1], 0

    v_lshlrev_b32 v[v_tmp], 4, v[v_lane_u]
    v_lshl_or_b32 v[v_smem_load_a], v[v_wave_p], 7, v[v_tmp]
    v_lshlrev_b32 v[v_tmp], 4, v[v_lane_v]
    v_lshl_or_b32 v[v_tmp+1], v[v_lane_w], 5, v[v_tmp]
    v_lshl_or_b32 v[v_smem_load_b], v[v_wave_q], 8, v[v_tmp+1]

    v_lshlrev_b32 v[v_tmp], 4, v[v_lane_u]
    v_lshl_or_b32 v[v_tmp+1], v[v_wave_p], 7, v[v_tmp]
    v_lshl_or_b32 v[v_tmp], v[v_lane_v], 9, v[v_tmp+1]
    v_lshl_or_b32 v[v_tmp+1], v[v_lane_w], 10, v[v_tmp]
    v_lshl_or_b32 v[v_smem_store_c], v[v_wave_q], 12, v[v_tmp+1]

    v_lshlrev_b32 v[v_tmp], 4, v[v_lane_id]
    s_lshl_b32 s[s_tmp], s[s_wave_id], 10
    v_or_b32 v[v_smem_load_c], s[s_tmp], v[v_tmp]

    s_lshl_b32 s[s_tmp], s[s_n_idx], 7
    s_lshl_b32 s[s_tmp+1], s[s_wave_id], 3
    s_add_u32 s[s_tmp], s[s_tmp], s[s_tmp+1]
    s_mul_i32 s[s_tmp+3], s[s_tmp], s[s_ldc]
    s_lshl_b32 s[s_tmp+2], s[s_m_idx], 8
    s_add_u32 s[s_tmp], s[s_tmp+2], s[s_tmp+3]
    s_add_u32 s[s_ptr_c], s[s_ptr_c], s[s_tmp]
    s_addc_u32 s[s_ptr_c+1], s[s_ptr_c+1], 0
    v_mul_lo_u32 v[v_tmp], s[s_ldc], v[v_lane_lm]
    v_lshl_add_u32 v[v_os_c], v[v_lane_k], 4, v[v_tmp]

    global_load_dwordx2 v[v_p1:v_p1+1], v[v_os_a:v_os_a+1], s[s_ptr_a:s_ptr_a+1]
    global_load_dwordx4 v[v_q1:v_q1+3], v[v_os_b:v_os_b+1], s[s_ptr_b:s_ptr_b+1]

    s_waitcnt vmcnt(3)
    ds_write_b64 v[v_smem_store_a], v[v_p0:v_p0+1]
    s_waitcnt vmcnt(2)
    ds_write_b128 v[v_smem_store_b], v[v_q0:v_q0+3], offset:0x1000

    s_mov_b32 s[s_kitr], 16
    s_cmp_lt_u32 s[s_kitr], s[s_k]
    s_cbranch_scc0 L_sgemm64x128_k_loop_end

L_sgemm64x128_k_loop_start:
    s_waitcnt lgkmcnt(0)
    s_barrier

    ds_read_b128 v[v_a0+0:v_a0+3], v[v_smem_load_a], offset:0
    s_add_u32 s[s_ptr_a], s[s_ptr_a], s[s_bs_a]
    s_addc_u32 s[s_ptr_a+1], s[s_ptr_a+1], 0

    ds_read_b128 v[v_b0+0:v_b0+3], v[v_smem_load_b], offset:0x1000
    
    s_add_u32 s[s_ptr_b], s[s_ptr_b], s[s_bs_b]
    ds_read_b128 v[v_b0+4:v_b0+7], v[v_smem_load_b], offset:0x1000+0x80
    s_addc_u32 s[s_ptr_b+1], s[s_ptr_b+1], 0
    .cnt = 0
    .rept 4
        ds_read_b128 v[v_a1+0:v_a1+3], v[v_smem_load_a], offset:(.cnt+1)*0x100+0
        .if .cnt == 0
            global_load_dwordx2 v[v_p0:v_p0+1], v[v_os_a:v_os_a+1], s[s_ptr_a:s_ptr_a+1]
        .endif
        ds_read_b128 v[v_b1+0:v_b1+3], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0
        ds_read_b128 v[v_b1+4:v_b1+7], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0x80

        .if .cnt == 6
            s_waitcnt vmcnt(3)
            ds_write_b64 v[v_smem_store_a], v[v_p1:v_p1+1], offset:0x2000
            s_waitcnt lgkmcnt(4)
        .else
            s_waitcnt lgkmcnt(3)
        .endif
        .s_fma4x8 v_c, v_a0, v_b0
        .cnt = .cnt + 1

        .if .cnt == 5
            ds_read_b128 v[v_a0+0:v_a0+3], v[v_smem_load_a], offset:(.cnt+1)*0x100+0
            global_load_dwordx4 v[v_q0:v_q0+3], v[v_os_b:v_os_b+1], s[s_ptr_b:s_ptr_b+1]
            ds_read_b128 v[v_b0+0:v_b0+3], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0
            ds_read_b128 v[v_b0+4:v_b0+7], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0x80
            s_waitcnt lgkmcnt(3)
        .else
        .if .cnt == 7
            s_waitcnt vmcnt(2)
            ds_write_b128 v[v_smem_store_b], v[v_q1:v_q1+3], offset:0x3000
            s_waitcnt lgkmcnt(1)
        .else
            ds_read_b128 v[v_a0+0:v_a0+3], v[v_smem_load_a], offset:(.cnt+1)*0x100+0
            ds_read_b128 v[v_b0+0:v_b0+3], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0
            ds_read_b128 v[v_b0+4:v_b0+7], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0x80
            s_waitcnt lgkmcnt(3)
        .endif
        .endif
        .s_fma4x8 v_c, v_a1, v_b1
        .cnt = .cnt + 1
    .endr

    s_waitcnt lgkmcnt(0)
    s_barrier

    ds_read_b128 v[v_a0+0:v_a0+3], v[v_smem_load_a], offset:0x2000
    s_add_u32 s[s_ptr_a], s[s_ptr_a], s[s_bs_a]
    s_addc_u32 s[s_ptr_a+1], s[s_ptr_a+1], 0

    ds_read_b128 v[v_b0+0:v_b0+3], v[v_smem_load_b], offset:0x3000
    s_add_u32 s[s_ptr_b], s[s_ptr_b], s[s_bs_b]
    ds_read_b128 v[v_b0+4:v_b0+7], v[v_smem_load_b], offset:0x3000+0x80
    s_addc_u32 s[s_ptr_b+1], s[s_ptr_b+1], 0
    .cnt = 0
    .rept 4
        ds_read_b128 v[v_a1+0:v_a1+3], v[v_smem_load_a], offset:0x2000+(.cnt+1)*0x100+0
        .if .cnt == 0
            global_load_dwordx2 v[v_p1:v_p1+1], v[v_os_a:v_os_a+1], s[s_ptr_a:s_ptr_a+1]
        .endif
        ds_read_b128 v[v_b1+0:v_b1+3], v[v_smem_load_b], offset:0x3000+(.cnt+1)*0x200+0
        ds_read_b128 v[v_b1+4:v_b1+7], v[v_smem_load_b], offset:0x3000+(.cnt+1)*0x200+0x80

        .if .cnt == 6
            s_waitcnt vmcnt(3)
            ds_write_b64 v[v_smem_store_a], v[v_p0:v_p0+1], offset:0
            s_waitcnt lgkmcnt(4)
        .else
            s_waitcnt lgkmcnt(3)
        .endif
        .s_fma4x8 v_c, v_a0, v_b0
        .cnt = .cnt + 1

        .if .cnt == 5
            ds_read_b128 v[v_a0+0:v_a0+3], v[v_smem_load_a], offset:0x2000+(.cnt+1)*0x100+0
            global_load_dwordx4 v[v_q1:v_q1+3], v[v_os_b:v_os_b+1], s[s_ptr_b:s_ptr_b+1]
            ds_read_b128 v[v_b0+0:v_b0+3], v[v_smem_load_b], offset:0x3000+(.cnt+1)*0x200+0
            ds_read_b128 v[v_b0+4:v_b0+7], v[v_smem_load_b], offset:0x3000+(.cnt+1)*0x200+0x80
            s_waitcnt lgkmcnt(3)
        .else
        .if .cnt == 7
            s_waitcnt vmcnt(2)
            ds_write_b128 v[v_smem_store_b], v[v_q0:v_q0+3], offset:0x1000
            s_waitcnt lgkmcnt(1)
        .else
            ds_read_b128 v[v_a0+0:v_a0+3], v[v_smem_load_a], offset:0x2000+(.cnt+1)*0x100+0
            ds_read_b128 v[v_b0+0:v_b0+3], v[v_smem_load_b], offset:0x3000+(.cnt+1)*0x200+0
            ds_read_b128 v[v_b0+4:v_b0+7], v[v_smem_load_b], offset:0x3000+(.cnt+1)*0x200+0x80
            s_waitcnt lgkmcnt(3)
        .endif
        .endif
        .s_fma4x8 v_c, v_a1, v_b1
        .cnt = .cnt + 1
    .endr

    ;s_branch  debug_code_seg

    s_add_u32 s[s_kitr], s[s_kitr], 16
    s_cmp_lt_u32 s[s_kitr], s[s_k]
    s_cbranch_scc1 L_sgemm64x128_k_loop_start

L_sgemm64x128_k_loop_end:
    s_waitcnt lgkmcnt(0)
    s_barrier

    ds_read_b128 v[v_a0+0:v_a0+3], v[v_smem_load_a], offset:0
    ds_read_b128 v[v_b0+0:v_b0+3], v[v_smem_load_b], offset:0x1000
    ds_read_b128 v[v_b0+4:v_b0+7], v[v_smem_load_b], offset:0x1000+0x80
    .cnt = 0
    .rept 4
        ds_read_b128 v[v_a1+0:v_a1+3], v[v_smem_load_a], offset:(.cnt+1)*0x100+0
        ds_read_b128 v[v_b1+0:v_b1+3], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0
        ds_read_b128 v[v_b1+4:v_b1+7], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0x80

        .if .cnt == 6
            s_waitcnt vmcnt(1)
            ds_write_b64 v[v_smem_store_a], v[v_p1:v_p1+1], offset:0x2000
            s_waitcnt lgkmcnt(4)
        .else
            s_waitcnt lgkmcnt(3)
        .endif
        .s_fma4x8 v_c, v_a0, v_b0
        .cnt = .cnt + 1

        .if .cnt == 1
            ds_read_b128 v[v_a0+0:v_a0+3], v[v_smem_load_a], offset:(.cnt+1)*0x100+0
            ds_read_b128 v[v_b0+0:v_b0+3], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0
            ds_read_b128 v[v_b0+4:v_b0+7], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0x80
            s_waitcnt lgkmcnt(3)
        .else
        .if .cnt == 7
            s_waitcnt vmcnt(0)
            ds_write_b128 v[v_smem_store_b], v[v_q1:v_q1+3], offset:0x3000
            s_waitcnt lgkmcnt(1)
        .else
            ds_read_b128 v[v_a0+0:v_a0+3], v[v_smem_load_a], offset:(.cnt+1)*0x100+0
            ds_read_b128 v[v_b0+0:v_b0+3], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0
            ds_read_b128 v[v_b0+4:v_b0+7], v[v_smem_load_b], offset:0x1000+(.cnt+1)*0x200+0x80
            s_waitcnt lgkmcnt(3)
        .endif
        .endif
        .s_fma4x8 v_c, v_a1, v_b1
        .cnt = .cnt + 1
    .endr

    s_waitcnt lgkmcnt(0)
    s_barrier

    ds_read_b128 v[v_a0+0:v_a0+3], v[v_smem_load_a], offset:0x2000
    ds_read_b128 v[v_b0+0:v_b0+3], v[v_smem_load_b], offset:0x3000
    ds_read_b128 v[v_b0+4:v_b0+7], v[v_smem_load_b], offset:0x3000+0x80

    .cnt = 0
    .rept 4
        ds_read_b128 v[v_a1+0:v_a1+3], v[v_smem_load_a], offset:0x2000+(.cnt+1)*0x100+0
        ds_read_b128 v[v_b1+0:v_b1+3], v[v_smem_load_b], offset:0x3000+(.cnt+1)*0x200+0
        ds_read_b128 v[v_b1+4:v_b1+7], v[v_smem_load_b], offset:0x3000+(.cnt+1)*0x200+0x80

        s_waitcnt lgkmcnt(3)
        .s_fma4x8 v_c, v_a0, v_b0

        .cnt = .cnt + 1

        .if .cnt == 1
            ds_read_b128 v[v_a0+0:v_a0+3], v[v_smem_load_a], offset:0x2000+(.cnt+1)*0x100+0
            ds_read_b128 v[v_b0+0:v_b0+3], v[v_smem_load_b], offset:0x3000+(.cnt+1)*0x200+0
            ds_read_b128 v[v_b0+4:v_b0+7], v[v_smem_load_b], offset:0x3000+(.cnt+1)*0x200+0x80
            s_waitcnt lgkmcnt(3)
        .else
        .if .cnt == 7
            s_waitcnt lgkmcnt(0)
        .else
            ds_read_b128 v[v_a0+0:v_a0+3], v[v_smem_load_a], offset:0x2000+(.cnt+1)*0x100+0
            ds_read_b128 v[v_b0+0:v_b0+3], v[v_smem_load_b], offset:0x3000+(.cnt+1)*0x200+0
            ds_read_b128 v[v_b0+4:v_b0+7], v[v_smem_load_b], offset:0x3000+(.cnt+1)*0x200+0x80
            s_waitcnt lgkmcnt(3)
        .endif
        .endif
        .s_fma4x8 v_c, v_a1, v_b1
        .cnt = .cnt + 1
    .endr

    ;s_branch program_end
    ; store c process
    .cnt=0
    .rept 4
        .set .cid, .cnt<<1
        .set .cof, ((.cnt>>1)<<5)|((.cnt&1)<<1)

        ds_write_b128 v[v_smem_store_c], v[v_c+4*.cid+0:v_c+4*.cid+3], offset:0
        ds_write_b128 v[v_smem_store_c], v[v_c+4*.cid+4:v_c+4*.cid+7], offset:0x100
        s_waitcnt lgkmcnt(0)
        s_barrier

        ds_read_b128 v[v_c+4*.cid+0:v_c+4*.cid+3], v[v_smem_load_c], offset:0
        ds_read_b128 v[v_c+4*.cid+4:v_c+4*.cid+7], v[v_smem_load_c], offset:0x1000

        s_mul_i32 s[s_tmp], .cof+0, s[s_ldc]
        v_add_co_u32 v[v_tmp], vcc, s[s_tmp], v[v_os_c]
        v_addc_co_u32  v[v_tmp+1], vcc, 0, v[v_os_c+1], vcc
        s_waitcnt lgkmcnt(1)
        global_store_dwordx4 v[v_tmp:v_tmp+1], v[v_c+4*.cid+0:v_c+4*.cid+3], s[s_ptr_c: s_ptr_c+1]

        s_mul_i32 s[s_tmp], .cof+64, s[s_ldc]
        v_add_co_u32 v[v_tmp], vcc, s[s_tmp], v[v_os_c]
        v_addc_co_u32  v[v_tmp+1], vcc, 0, v[v_os_c+1], vcc
        s_waitcnt lgkmcnt(0)
        global_store_dwordx4 v[v_tmp:v_tmp+1], v[v_c+4*.cid+4:v_c+4*.cid+7], s[s_ptr_c: s_ptr_c+1]
        .if .cnt!=3
            s_barrier
        .endif
        .cnt=.cnt+1
    .endr


    ; debug code to cpy vgpr to host
debug_code_seg:
    ;s_waitcnt lgkmcnt(0)
    ;s_barrier
    ;s_cmp_lg_u32 s[s_bx], 0
    ;s_cbranch_scc1  program_end
    ;v_add_co_u32 v34, vcc, 0, v[v_c+1]
    ;global_store_dword v[32:33], v34, s[s_tmp+10:s_tmp+11]

program_end:
    s_endpgm