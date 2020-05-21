.hsa_code_object_version 2,1
.hsa_code_object_isa
.text
.p2align 8
.amdgpu_hsa_kernel sgemm_xdlops_128x128

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

.macro .s_fma8x8 c, a, b
    v_mac_f32 v[\c+0 ], v[\a+0],  v[\b+0]
    ; priority can modify the phase of 2 waves
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

    v_mac_f32 v[\c+16], v[\a+4],  v[\b+0]
    v_mac_f32 v[\c+17], v[\a+5],  v[\b+0]
    v_mac_f32 v[\c+18], v[\a+6],  v[\b+0]
    v_mac_f32 v[\c+19], v[\a+7],  v[\b+0]
    v_mac_f32 v[\c+20], v[\a+4],  v[\b+1]
    v_mac_f32 v[\c+21], v[\a+5],  v[\b+1]
    v_mac_f32 v[\c+22], v[\a+6],  v[\b+1]
    v_mac_f32 v[\c+23], v[\a+7],  v[\b+1]
    v_mac_f32 v[\c+24], v[\a+4],  v[\b+2]
    v_mac_f32 v[\c+25], v[\a+5],  v[\b+2]
    v_mac_f32 v[\c+26], v[\a+6],  v[\b+2]
    v_mac_f32 v[\c+27], v[\a+7],  v[\b+2]
    v_mac_f32 v[\c+28], v[\a+4],  v[\b+3]
    v_mac_f32 v[\c+29], v[\a+5],  v[\b+3]
    v_mac_f32 v[\c+30], v[\a+6],  v[\b+3]
    v_mac_f32 v[\c+31], v[\a+7],  v[\b+3]

    v_mac_f32 v[\c+32], v[\a+0],  v[\b+4]
    v_mac_f32 v[\c+33], v[\a+1],  v[\b+4]
    v_mac_f32 v[\c+34], v[\a+2],  v[\b+4]
    v_mac_f32 v[\c+35], v[\a+3],  v[\b+4]
    v_mac_f32 v[\c+36], v[\a+0],  v[\b+5]
    v_mac_f32 v[\c+37], v[\a+1],  v[\b+5]
    v_mac_f32 v[\c+38], v[\a+2],  v[\b+5]
    v_mac_f32 v[\c+39], v[\a+3],  v[\b+5]
    v_mac_f32 v[\c+40], v[\a+0],  v[\b+6]
    v_mac_f32 v[\c+41], v[\a+1],  v[\b+6]
    v_mac_f32 v[\c+42], v[\a+2],  v[\b+6]
    v_mac_f32 v[\c+43], v[\a+3],  v[\b+6]
    v_mac_f32 v[\c+44], v[\a+0],  v[\b+7]
    v_mac_f32 v[\c+45], v[\a+1],  v[\b+7]
    v_mac_f32 v[\c+46], v[\a+2],  v[\b+7]
    v_mac_f32 v[\c+47], v[\a+3],  v[\b+7]

    v_mac_f32 v[\c+48], v[\a+4],  v[\b+4]
    v_mac_f32 v[\c+49], v[\a+5],  v[\b+4]
    v_mac_f32 v[\c+50], v[\a+6],  v[\b+4]
    v_mac_f32 v[\c+51], v[\a+7],  v[\b+4]
    v_mac_f32 v[\c+52], v[\a+4],  v[\b+5]
    v_mac_f32 v[\c+53], v[\a+5],  v[\b+5]
    v_mac_f32 v[\c+54], v[\a+6],  v[\b+5]
    v_mac_f32 v[\c+55], v[\a+7],  v[\b+5]
    v_mac_f32 v[\c+56], v[\a+4],  v[\b+6]
    v_mac_f32 v[\c+57], v[\a+5],  v[\b+6]
    v_mac_f32 v[\c+58], v[\a+6],  v[\b+6]
    v_mac_f32 v[\c+59], v[\a+7],  v[\b+6]
    v_mac_f32 v[\c+60], v[\a+4],  v[\b+7]
    v_mac_f32 v[\c+61], v[\a+5],  v[\b+7]
    v_mac_f32 v[\c+62], v[\a+6],  v[\b+7]
    v_mac_f32 v[\c+63], v[\a+7],  v[\b+7]
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
.set s_bs_a,            10
.set s_bs_b,            11
.set s_alpha,           12
.set s_m,               13
.set s_n,               14
.set s_k,               15
.set s_lda,             16
.set s_ldb,             17
.set s_ldc,             18
.set s_m_blocks,        19
.set s_kitr,            20
.set s_wave_id,         21
.set s_m_idx,           22
.set s_n_idx,           23
.set s_tmp,             24
.set s_end,             s_tmp+3

.set v_c,               0
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

sgemm_xdlops_128x128:
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

    s_load_dwordx4 s[s_ptr_c:s_ptr_c+3], s[s_ka:s_ka+1], 0+k_ptr_c
    s_load_dwordx2 s[s_ptr_b:s_ptr_b+1], s[s_ka:s_ka+1], 0+k_ptr_b
    s_load_dwordx4 s[s_alpha:s_alpha+3], s[s_ka:s_ka+1], 0+k_alpha
    s_load_dwordx2 s[s_lda:s_lda+1], s[s_ka:s_ka+1], 0+k_lda
    s_load_dword s[s_ldc], s[s_ka:s_ka+1], 0+k_ldc

    ; debug vgpr
    v_lshlrev_b32 v[32], 2, v0 ; every thread write one float
    s_load_dwordx2 s[s_tmp+10:s_tmp+11], s[s_ka:s_ka+1], k_ptr_c

    ; clear accvgpr
    .cnt=0
    .rept 32
        v_accvgpr_write acc[.cnt], 0
        .cnt=.cnt+1
    .endr

    ; load A B data to vpgr
    v_add_co_u32 v[7], vcc, 1, v[0]
    v_add_co_u32 v[8], vcc, 2, v[0]

    v_cvt_f32_u32 v[9], v[7]
    v_cvt_f32_u32 v[10], v[8]

    s_nop 4

    ; do mfma
    v_mfma_f32_32x32x2f32 acc[0:15], v[9], v[10], acc[0:15]

    s_nop 4
    v_accvgpr_read v[16], acc[0]

    s_nop 4

    ; debug code to cpy vgpr to host
debug_code_seg:
    s_waitcnt lgkmcnt(0)
    s_barrier
    s_cmp_lg_u32 s[s_bx], 0
    s_cbranch_scc1  program_end
    v_add_co_u32 v34, vcc, 0, v[16]

    global_store_dword v[32:33], v34, s[s_tmp+10:s_tmp+11]

    s_waitcnt vmcnt(0)

program_end:
    s_endpgm