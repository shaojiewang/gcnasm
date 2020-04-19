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

.macro .s_fma_8x4 c, a, b
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
.set s_end,             s_tmp+3

.set v_c,               0 ; for 64x128 kernel, only use 32 vc
.set v_a0,              64
.set v_a1,              72
.set v_b0,              80
.set v_b1,              88
.set v_p0,              96
.set v_q0,              100
.set v_lane_id,         104
.set v_wave_p,          105
.set v_wave_q,          106
.set v_lane_lo,         107
.set v_lane_hi,         108
.set v_lane_w,          109
.set v_lane_u,          110
.set v_lane_v,          111
.set v_smem_store,      112
.set v_os_a,            113
.set v_os_b,            114
.set v_os_c,            115
.set v_smem_load_a,     116
.set v_smem_load_b,     117
.set v_smem_store_c,    118
.set v_smem_load_c,     119
.set v_tmp,             124
.set v_end,             v_tmp+3
.set v_p1,              104
.set v_q1,              108

; 64x128 kernel addition vars
.set v_lane_k,          33
.set v_lane_l,          34
.set v_lane_m,          35
.set v_smem_store_a,    36
.set v_smem_store_b,    37

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

    v_lshlrev_b32 v[v_smem_store_a], 3, v0

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

    