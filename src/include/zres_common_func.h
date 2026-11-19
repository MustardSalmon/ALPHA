#ifndef ZRES_COMMON_FUNC_H
#define ZRES_COMMON_FUNC_H

#include <stddef.h>
#include <math.h>
#include "zres_common_struct.h"
#include "zres_cluster.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RNAME(a) (#a)
#define ZresBit2MB(a) ((double)(a) / 1024.0 / 1024.0)
#define ZSLD_CLUSTER_BATCH_BY_TSP 0
#define ZSLD_CLUSTER_BATCH_UPSCATTER_SINGLE 1
#define ZSLD_CLUSTER_BATCH_ALL_SINGLE 2

/* Steady transport compatibility and runtime clustering interface. */
void zres_normLocalFlxAndCalRelErr(FP_RES_FX *, FP_RES_FX *);

void zres_init_gen_info(void);
void zres_set_gen_info_tsp_group(int);
void zres_set_fsr_cluster(int);
void zres_set_fsr_cluster_sse_limit(FP_RES_XS);
void zres_set_fsr_cluster_max_per_mat(int);
void zres_set_fsr_cluster_max_iter(int);
void zres_set_fsr_cluster_tolerance(FP_RES_XS);
void zres_set_fsr_cluster_seed(unsigned int);
void zres_set_fsr_cluster_algorithm(char *);
void zres_set_gen_info_num_fsr(int);
void zres_set_gen_info_vol(double *);
void zres_set_gen_info_sca_order(int);
void zres_set_fsr2map(int *);
void zres_set_macro_t(FP_RES_XS *);
void zres_set_macro_a(FP_RES_XS *);
void zres_set_macro_f(FP_RES_XS *);
void zres_set_macro_nuf(FP_RES_XS *);
void zres_set_chi(FP_RES_XS *);
ZRES_GI *zres_get_gen_info(void);
void zres_get_argv(int, char * []);
double zres_get_norm_flux_rmse(void);

void zres_init_memlog(char *);
void zres_close_memlog(void);
void *zres_malloc(size_t, size_t, char *);
void zres_free_(void *, size_t, char *);
#define zres_free(ptr, dim) (zres_free_((ptr), (dim), RNAME(ptr)))
void zres_raise_error(char *);

ZRES_MTC *zres_get_local_mat(void);

#ifdef __cplusplus
}
#endif

#endif /* ZRES_COMMON_FUNC_H */
