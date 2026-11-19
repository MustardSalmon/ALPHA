#include "zres_common_func.h"

/* Minimal state boundary used by the external FSR-clustering path.  The
 * resonance library, slow-down, and legacy input-setting routines are not
 * part of this transport snapshot. */
ZRES_GI* _zresGenInfo = NULL;

static const char *zres_get_env_alias(const char *name, const char *upperName)
{
    const char *value = getenv(name);
    if (value == NULL && upperName != NULL) {
        value = getenv(upperName);
    }
    return value;
}

void zres_init_gen_info(void)
{
    _zresGenInfo = (ZRES_GI *) zres_malloc(1, sizeof(ZRES_GI), RNAME(_zresGenInfo));
    _zresGenInfo->cutOffGrp = -1;
    _zresGenInfo->fsrClusterSseLimit = (FP_RES_XS) ZRES_FSR_CLUSTER_SSE_LIMIT;
    _zresGenInfo->fsrClusterMaxPerMat = ZRES_FSR_CLUSTER_MAX_PER_MAT;
    _zresGenInfo->fsrClusterMaxIter = ZRES_FSR_CLUSTER_MAX_ITER;
    _zresGenInfo->fsrClusterTolerance = (FP_RES_XS) ZRES_FSR_CLUSTER_TOLERANCE;
    _zresGenInfo->fsrClusterSeed = (unsigned int) ZRES_FSR_CLUSTER_SEED;
    _zresGenInfo->fsrClusterAlgorithm = 0;
    _zresGenInfo->fsrClusterSldBatchMode = ZSLD_CLUSTER_BATCH_UPSCATTER_SINGLE;
    _zresGenInfo->_rmse = 1.0;
}

void zres_set_gen_info_tsp_group(int numGroup)
{
    _zresGenInfo->tspGrpNum = numGroup;
    _zresGenInfo->resGrpNum = numGroup;
}

void zres_set_fsr_cluster(int ifFsrCluster)
{
    _zresGenInfo->ifFsrCluster = ifFsrCluster;
}

void zres_set_fsr_cluster_sse_limit(FP_RES_XS sseLimit)
{
    _zresGenInfo->fsrClusterSseLimit = sseLimit;
}

void zres_set_fsr_cluster_max_per_mat(int maxPerMat)
{
    _zresGenInfo->fsrClusterMaxPerMat = maxPerMat;
}

void zres_set_fsr_cluster_max_iter(int maxIter)
{
    _zresGenInfo->fsrClusterMaxIter = maxIter;
}

void zres_set_fsr_cluster_tolerance(FP_RES_XS tolerance)
{
    _zresGenInfo->fsrClusterTolerance = tolerance;
}

void zres_set_fsr_cluster_seed(unsigned int seed)
{
    _zresGenInfo->fsrClusterSeed = seed;
}

void zres_set_fsr_cluster_algorithm(char *algorithm)
{
    if (algorithm == NULL || strcmp(algorithm, "kmeans") == 0 ||
        strcmp(algorithm, "kmeans++") == 0) {
        _zresGenInfo->fsrClusterAlgorithm = 0;
    }
    else if (strcmp(algorithm, "sorted1d") == 0 ||
             strcmp(algorithm, "sorted-1d") == 0 ||
             strcmp(algorithm, "sse_dp") == 0 ||
             strcmp(algorithm, "sse-dp") == 0 ||
             strcmp(algorithm, "dp1d") == 0) {
        _zresGenInfo->fsrClusterAlgorithm = 1;
    }
    else {
        zres_raise_error(" [  ERROR  ]  FsrClusterAlgorithm must be kmeans or sse_dp (sorted1d alias)\n");
    }
}

void zres_set_gen_info_num_fsr(int numFsr)
{
    _zresGenInfo->numFsr = numFsr;
}

void zres_set_gen_info_vol(double *fsrVol)
{
    _zresGenInfo->vol = fsrVol;
}

void zres_set_gen_info_sca_order(int scaOrder)
{
    _zresGenInfo->scaOrder = scaOrder;
}

void zres_set_fsr2map(int *fsr2matMap)
{
    _zresGenInfo->fsr2matMap = fsr2matMap;
}

void zres_set_macro_t(FP_RES_XS *macroT)
{
    _zresGenInfo->macro_T = macroT;
}

void zres_set_macro_a(FP_RES_XS *macroA)
{
    _zresGenInfo->macro_A = macroA;
}

void zres_set_macro_f(FP_RES_XS *macroF)
{
    _zresGenInfo->macro_F = macroF;
}

void zres_set_macro_nuf(FP_RES_XS *macroNuf)
{
    _zresGenInfo->macro_Nuf = macroNuf;
}

void zres_set_chi(FP_RES_XS *chi)
{
    _zresGenInfo->chi = chi;
}

ZRES_GI *zres_get_gen_info(void)
{
    return _zresGenInfo;
}

void zres_get_argv(int argc, char *argv[])
{
    for (int idx = 0; idx + 1 < argc; ++idx) {
        if (strcmp(argv[idx], "--FsrCluster") == 0) {
            _zresGenInfo->FsrCluster = argv[idx + 1];
        }
        else if (strcmp(argv[idx], "--FsrClusterSseLimit") == 0) {
            _zresGenInfo->FsrClusterSseLimit = argv[idx + 1];
        }
        else if (strcmp(argv[idx], "--FsrClusterMaxPerMat") == 0) {
            _zresGenInfo->FsrClusterMaxPerMat = argv[idx + 1];
        }
        else if (strcmp(argv[idx], "--FsrClusterMaxIter") == 0) {
            _zresGenInfo->FsrClusterMaxIter = argv[idx + 1];
        }
        else if (strcmp(argv[idx], "--FsrClusterTolerance") == 0) {
            _zresGenInfo->FsrClusterTolerance = argv[idx + 1];
        }
        else if (strcmp(argv[idx], "--FsrClusterSeed") == 0) {
            _zresGenInfo->FsrClusterSeed = argv[idx + 1];
        }
        else if (strcmp(argv[idx], "--FsrClusterAlgorithm") == 0) {
            _zresGenInfo->FsrClusterAlgorithm = argv[idx + 1];
        }
    }

    const char *value = zres_get_env_alias("FsrCluster", "FSR_CLUSTER");
    if (value != NULL) _zresGenInfo->FsrCluster = (char *) value;
    value = zres_get_env_alias("FsrClusterSseLimit", "FSR_CLUSTER_SSE_LIMIT");
    if (value != NULL) _zresGenInfo->FsrClusterSseLimit = (char *) value;
    value = zres_get_env_alias("FsrClusterMaxPerMat", "FSR_CLUSTER_MAX_PER_MAT");
    if (value != NULL) _zresGenInfo->FsrClusterMaxPerMat = (char *) value;
    value = zres_get_env_alias("FsrClusterMaxIter", "FSR_CLUSTER_MAX_ITER");
    if (value != NULL) _zresGenInfo->FsrClusterMaxIter = (char *) value;
    value = zres_get_env_alias("FsrClusterTolerance", "FSR_CLUSTER_TOLERANCE");
    if (value != NULL) _zresGenInfo->FsrClusterTolerance = (char *) value;
    value = zres_get_env_alias("FsrClusterSeed", "FSR_CLUSTER_SEED");
    if (value != NULL) _zresGenInfo->FsrClusterSeed = (char *) value;
    value = zres_get_env_alias("FsrClusterAlgorithm", "FSR_CLUSTER_ALGORITHM");
    if (value != NULL) _zresGenInfo->FsrClusterAlgorithm = (char *) value;
}

double zres_get_norm_flux_rmse(void)
{
    return _zresGenInfo->_rmse;
}
