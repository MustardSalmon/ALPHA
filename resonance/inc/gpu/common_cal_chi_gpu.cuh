
#ifndef ZRES_COMMON_CAL_CHI_GPU_CUH
#define ZRES_COMMON_CAL_CHI_GPU_CUH

#include "zres_common_struct.h"

#ifdef __cplusplus
extern "C" {
#endif

void zres_init_ss(int, FP_RES_SS*);
void zres_sum(int, int, FP_RES_SS*, FP_RES_SS*);
void zres_gemm(int, int, FP_RES_XS*, FP_RES_FX*, FP_RES_SS*, int);
void zres_update_homo_flux_old(int, FP_RES_SS*, FP_RES_SS*, FP_RES_XS*, FP_RES_FX*, FP_RES_XS);
void zres_update_homo_flux_opt(int, int, FP_RES_SS*, FP_RES_SS*, FP_RES_XS*, FP_RES_FX*, FP_RES_XS*);

#ifdef __cplusplus
}
#endif

#endif // ZRES_COMMON_CAL_CHI_GPU_CUH
