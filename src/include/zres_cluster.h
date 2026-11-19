/**
 * ============================================================
 * @File     : zres_cluster.h
 * @Author   : CaoWei
 * @Colleges : Zhejiang University
 * @Email    : 12245028@zju.edu.cn
 * @Time     : 2026-07-29 14:26
 * ============================================================
 */


#ifndef ALPHA_2D_GPU_FP32_MOC_ZRES_CLUSTER_H
#define ALPHA_2D_GPU_FP32_MOC_ZRES_CLUSTER_H

#include "zres_common_struct.h"

/* Default numeric parameters only; runtime options control FSR clustering. */
/* FsrClusterAlgorithm=sse_dp (or sorted1d) selects exact 1D SSE DP. */
#ifndef ZRES_FSR_CLUSTER_SSE_LIMIT
    #define ZRES_FSR_CLUSTER_SSE_LIMIT 100.0
#endif

#ifndef ZRES_FSR_CLUSTER_MAX_PER_MAT
    #define ZRES_FSR_CLUSTER_MAX_PER_MAT 1000
#endif

#ifndef ZRES_FSR_CLUSTER_MAX_ITER
    #define ZRES_FSR_CLUSTER_MAX_ITER 20
#endif

#ifndef ZRES_FSR_CLUSTER_TOLERANCE
    #define ZRES_FSR_CLUSTER_TOLERANCE 1.0e-8
#endif

#ifndef ZRES_FSR_CLUSTER_SEED
    #define ZRES_FSR_CLUSTER_SEED 20240727u
#endif

#ifdef __cplusplus
extern "C" {
#endif

void zres_free_fsr_cluster_info(void);
int zres_is_fsr_cluster_ready(void);
int zres_get_fsr_cluster_rep_fsr(int);
int zres_cluster_fsr_by_macro_t_sse_seed_external(FP_RES_XS, int, int, FP_RES_XS, unsigned int, int);
void zres_apply_fsr_cluster_macro_t_nuf(int, int);
void zres_apply_fsr_cluster_chi();

#ifdef __cplusplus
}
#endif

#endif //ALPHA_2D_GPU_FP32_MOC_ZRES_CLUSTER_H
