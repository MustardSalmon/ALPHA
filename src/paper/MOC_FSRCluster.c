/*
 * Publication snapshot: FSR clustering and CSR compression used by the
 * external-XS transport path.  Resonance and slowing-down drivers are not
 * included; this module starts after MacroXSsEvaluation_subDomain().
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "MOC_AllocAndFreeMem.h"
#include "MOC_CommonMacro.h"
#include "MOC_Geometry.h"
#include "MOC_Input.h"
#include "MOC_Macro_XSs.h"
#include "MOC_Output.h"
#include "paper_fsr_cluster.h"
#include "zres_common_func.h"

static const char *paper_get_env_alias(const char *name, const char *upperName)
{
    const char *value = getenv(name);
    if (value == NULL && upperName != NULL) {
        value = getenv(upperName);
    }
    return value;
}

static double paper_wall_time_seconds(void)
{
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0.0;
    }
    return (double) value.tv_sec + (double) value.tv_nsec * 1.0e-9;
}

/*
 * Compress an externally supplied FSR CSR scattering matrix after the FSR
 * cluster map has been built.  The FSR/group/order row arrays remain full
 * sized; only the value buffer is made representative-row based.  The GPU
 * high-order source kernels use the explicit row-end arrays installed here.
 */
static void zres_compress_external_fsr_scatter_csr(void)
{
    const int _numFsr = TotalNumFSRs_subDom;
    const int _numFsrPerLayer = TotalNumFSRs_perLayer;
    const int _numLayer = MOCNumLayersZ;
    const int _numGroup = NumGroups;
    const int _numOrder = NumScattOrder;
    const size_t _rowStride = (size_t) (_numOrder + 1) * (size_t) _numGroup;
    const size_t _rowCount = (size_t) _numFsr * _rowStride;
    const M_UINT64 _unset = (M_UINT64) (~(M_UINT64) 0);

    if (_numLayer <= 0 || _numFsrPerLayer <= 0 || _numFsr <= 0 ||
        MacroXS_s == NULL || MacroXS_s_NZRowStartID == NULL ||
        MacroXS_s_NZColStartID == NULL || MacroXS_s_NNZ_Layer == NULL ||
        MacroXS_s_NZStartID_Layer == NULL ||
        MacroXS_s_NZRowStartID_Layer == NULL) {
        zres_raise_error(" [  ERROR  ]  Invalid external CSR data for FSR clustering\n");
    }

    M_UINT64 *_oldRowStart = MacroXS_s_NZRowStartID;
    M_UINT64 *_oldColStart = MacroXS_s_NZColStartID;
    M_UINT64 *_oldLayerRowStart = MacroXS_s_NZRowStartID_Layer;
    M_UINT64 *_oldNnzLayer = MacroXS_s_NNZ_Layer;
    M_UINT64 *_oldStartLayer = MacroXS_s_NZStartID_Layer;
    FP_PRECISION *_oldXs = MacroXS_s;

    M_UINT64 *_oldRowEnd = (M_UINT64 *) malloc((_rowCount + 1) * sizeof(M_UINT64));
    M_UINT64 *_repRowOffset = (M_UINT64 *) malloc(_rowCount * sizeof(M_UINT64));
    if (_oldRowEnd == NULL || _repRowOffset == NULL) {
        zres_raise_error(" [  ERROR  ]  Failed to allocate FSR cluster CSR workspace\n");
    }

    for (size_t _row = 0; _row < _rowCount; ++_row) {
        _oldRowEnd[_row] = _oldRowStart[_row + 1];
    }
    _oldRowEnd[_rowCount] = _oldRowStart[_rowCount];
    const M_UINT64 _oldNnz = _oldRowStart[_rowCount];

    M_UINT64 _newNnz = 0;
    M_UINT64 _newMaxNnzLayer = 0;
    for (int _lay = 0; _lay < _numLayer; ++_lay) {
        for (size_t _row = 0; _row < _rowCount; ++_row) {
            _repRowOffset[_row] = _unset;
        }

        M_UINT64 _layerNnz = 0;
        for (int _localFsr = 0; _localFsr < _numFsrPerLayer; ++_localFsr) {
            const int _fsr = _lay * _numFsrPerLayer + _localFsr;
            const int _repFsr = zres_get_fsr_cluster_rep_fsr(_fsr);
            if (_repFsr < 0 || _repFsr >= _numFsr) {
                zres_raise_error(" [  ERROR  ]  Invalid representative FSR in external CSR clustering\n");
            }

            for (size_t _rowInFsr = 0; _rowInFsr < _rowStride; ++_rowInFsr) {
                const size_t _repRow = (size_t) _repFsr * _rowStride + _rowInFsr;
                if (_repRowOffset[_repRow] != _unset) {
                    continue;
                }

                const M_UINT64 _rowStart = _oldRowStart[_repRow];
                const M_UINT64 _rowEnd = _oldRowEnd[_repRow];
                if (_rowEnd < _rowStart) {
                    zres_raise_error(" [  ERROR  ]  Invalid external CSR row range in FSR clustering\n");
                }
                _repRowOffset[_repRow] = _layerNnz;
                _layerNnz += _rowEnd - _rowStart;
            }
        }

        _newNnz += _layerNnz;
        if (_newMaxNnzLayer < _layerNnz) {
            _newMaxNnzLayer = _layerNnz;
        }
    }

    M_UINT64 *_newRowStart = AllocMem_ui64_I(_rowCount + 1, (char *)("ClusterMacroXS_s_NZRowStartID"));
    M_UINT64 *_newRowEnd = AllocMem_ui64_I(_rowCount + 1, (char *)("ClusterMacroXS_s_NZRowEndID"));
    M_UINT64 *_newLayerRowStart = AllocMem_ui64_I(
        _rowCount + (size_t) _numLayer, (char *)("ClusterMacroXS_s_NZRowStartID_Layer"));
    M_UINT64 *_newLayerRowEnd = AllocMem_ui64_I(
        _rowCount + (size_t) _numLayer, (char *)("ClusterMacroXS_s_NZRowEndID_Layer"));
    M_UINT64 *_newColStart = AllocMem_ui64_I(_rowCount, (char *)("ClusterMacroXS_s_NZColStartID"));
    M_UINT64 *_newNnzLayer = AllocMem_ui64_I((size_t) _numLayer, (char *)("ClusterMacroXS_s_NNZ_Layer"));
    M_UINT64 *_newStartLayer = AllocMem_ui64_I((size_t) _numLayer, (char *)("ClusterMacroXS_s_NZStartID_Layer"));
    FP_PRECISION *_newXs = AllocMem_FP_I((size_t) _newNnz, (char *)("ClusterMacroXS_s"));

    M_UINT64 _globalLayerBase = 0;
    for (int _lay = 0; _lay < _numLayer; ++_lay) {
        for (size_t _row = 0; _row < _rowCount; ++_row) {
            _repRowOffset[_row] = _unset;
        }

        M_UINT64 _layerNnz = 0;
        const size_t _layerRowBase = (size_t) _lay * (size_t) _numFsrPerLayer * _rowStride
            + (size_t) _lay;
        _newStartLayer[_lay] = _globalLayerBase;

        for (int _localFsr = 0; _localFsr < _numFsrPerLayer; ++_localFsr) {
            const int _fsr = _lay * _numFsrPerLayer + _localFsr;
            const int _repFsr = zres_get_fsr_cluster_rep_fsr(_fsr);

            for (size_t _rowInFsr = 0; _rowInFsr < _rowStride; ++_rowInFsr) {
                const size_t _row = (size_t) _fsr * _rowStride + _rowInFsr;
                const size_t _repRow = (size_t) _repFsr * _rowStride + _rowInFsr;
                const M_UINT64 _rowStart = _oldRowStart[_repRow];
                const M_UINT64 _rowEnd = _oldRowEnd[_repRow];
                const M_UINT64 _rowLength = _rowEnd - _rowStart;

                if (_repRowOffset[_repRow] == _unset) {
                    _repRowOffset[_repRow] = _layerNnz;
                    if (_rowLength > 0) {
                        memcpy(_newXs + _globalLayerBase + _layerNnz,
                               _oldXs + _rowStart,
                               (size_t) _rowLength * sizeof(FP_PRECISION));
                    }
                    _layerNnz += _rowLength;
                }

                const M_UINT64 _localStart = _repRowOffset[_repRow];
                _newRowStart[_row] = _globalLayerBase + _localStart;
                _newRowEnd[_row] = _globalLayerBase + _localStart + _rowLength;
                _newColStart[_row] = _oldColStart[_repRow];

                const size_t _layerRow = _layerRowBase + _localFsr * _rowStride + _rowInFsr;
                _newLayerRowStart[_layerRow] = _localStart;
                _newLayerRowEnd[_layerRow] = _localStart + _rowLength;
            }
        }

        _newLayerRowStart[_layerRowBase + (size_t) _numFsrPerLayer * _rowStride] = _layerNnz;
        _newLayerRowEnd[_layerRowBase + (size_t) _numFsrPerLayer * _rowStride] = _layerNnz;
        _newNnzLayer[_lay] = _layerNnz;
        _globalLayerBase += _layerNnz;
    }

    _newRowStart[_rowCount] = _newNnz;
    _newRowEnd[_rowCount] = _newNnz;

    MacroXS_s = FreeMem_FP_I(_oldXs, (char *)("MacroXS_s"));
    MacroXS_s_NZRowStartID = FreeMem_ui64_I(_oldRowStart, (char *)("MacroXS_s_NZRowStartID"));
    MacroXS_s_NZColStartID = FreeMem_ui64_I(_oldColStart, (char *)("MacroXS_s_NZColStartID"));
    MacroXS_s_NZRowStartID_Layer = FreeMem_ui64_I(_oldLayerRowStart, (char *)("MacroXS_s_NZRowStartID_Layer"));
    MacroXS_s_NNZ_Layer = FreeMem_ui64_I(_oldNnzLayer, (char *)("MacroXS_s_NNZ_Layer"));
    MacroXS_s_NZStartID_Layer = FreeMem_ui64_I(_oldStartLayer, (char *)("MacroXS_s_NZStartID_Layer"));

    MacroXS_s = _newXs;
    MacroXS_s_NZRowStartID = _newRowStart;
    MacroXS_s_NZRowEndID = _newRowEnd;
    MacroXS_s_NZColStartID = _newColStart;
    MacroXS_s_NZRowStartID_Layer = _newLayerRowStart;
    MacroXS_s_NZRowEndID_Layer = _newLayerRowEnd;
    MacroXS_s_NNZ_Layer = _newNnzLayer;
    MacroXS_s_NZStartID_Layer = _newStartLayer;
    MacroXS_s_NNZ = _newNnz;
    MacroXS_s_MaxNNZ_Layer = _newMaxNnzLayer;
    IfUseExplicitScaRowEnd = 1;

    free(_oldRowEnd);
    free(_repRowOffset);

    printf(" [  INFO   ]  External FSR scattering CSR clustered: old_nnz=%llu, new_nnz=%llu, reduction=%.2f%%\n",
           (unsigned long long) _oldNnz,
           (unsigned long long) _newNnz,
           _oldNnz == 0 ? 0.0 :
           100.0 * (1.0 - (double) _newNnz / (double) _oldNnz));
    fflush(stdout);
}

void paper_prepare_external_fsr_cluster(int argc, char* argv[])
{
    const char *_fsrClusterEnv = paper_get_env_alias("FsrCluster", "FSR_CLUSTER");
    if (_fsrClusterEnv == NULL || atoi(_fsrClusterEnv) == 0) {
        return;
    }

    const int _ifFsrCluster = atoi(_fsrClusterEnv);

    zres_init_memlog((char *) ("zres_external_cluster_mem.log"));
    zres_init_gen_info();
    zres_set_gen_info_tsp_group(NumGroups);
    zres_set_gen_info_num_fsr(TotalNumFSRs_subDom);
    zres_set_gen_info_vol(FSRVol_SubDom);
    zres_set_gen_info_sca_order(NumScattOrder);
    zres_set_fsr2map(FSRsXSID_subDom);
    zres_set_macro_t(MacroXS_t);
    zres_set_macro_a(MacroXS_a);
    zres_set_macro_f(MacroXS_f);
    zres_set_macro_nuf(MacroXS_nf);
    zres_set_chi(Chi);
    zres_get_argv(argc, argv);
    zres_set_fsr_cluster(_ifFsrCluster);

    ZRES_GI* _gi = zres_get_gen_info();
    if (_gi == NULL || !_gi->ifFsrCluster) {
        zres_close_memlog();
        return;
    }

    if (_gi->FsrClusterSseLimit != NULL) {
        zres_set_fsr_cluster_sse_limit((FP_RES_XS) atof(_gi->FsrClusterSseLimit));
    }
    if (_gi->FsrClusterMaxPerMat != NULL) {
        zres_set_fsr_cluster_max_per_mat(atoi(_gi->FsrClusterMaxPerMat));
    }
    if (_gi->FsrClusterMaxIter != NULL) {
        zres_set_fsr_cluster_max_iter(atoi(_gi->FsrClusterMaxIter));
    }
    if (_gi->FsrClusterTolerance != NULL) {
        zres_set_fsr_cluster_tolerance((FP_RES_XS) atof(_gi->FsrClusterTolerance));
    }
    if (_gi->FsrClusterSeed != NULL) {
        zres_set_fsr_cluster_seed((unsigned int) strtoul(_gi->FsrClusterSeed, NULL, 10));
    }
    if (_gi->FsrClusterAlgorithm != NULL) {
        zres_set_fsr_cluster_algorithm(_gi->FsrClusterAlgorithm);
    }

    int _materialNum = 0;
    for (int _fsr = 0; _fsr < TotalNumFSRs_subDom; ++_fsr) {
        if (FSRsXSID_subDom[_fsr] + 1 > _materialNum) {
            _materialNum = FSRsXSID_subDom[_fsr] + 1;
        }
    }

    const double _clusterStartTime = paper_wall_time_seconds();
    zres_cluster_fsr_by_macro_t_sse_seed_external(_gi->fsrClusterSseLimit,
                                                  _gi->fsrClusterMaxPerMat,
                                                  _gi->fsrClusterMaxIter,
                                                  _gi->fsrClusterTolerance,
                                                  _gi->fsrClusterSeed,
                                                  _materialNum);
    const double _clusterEndTime = paper_wall_time_seconds();

    printf(" [  CLUST  ]  FSR clustering time: %.6f sec\n",
           _clusterEndTime - _clusterStartTime);
    fflush(stdout);

    /* Keep all FSR-dependent transport data consistent with the clustered
     * scattering matrix.  Fluxes and geometry remain indexed by the original
     * FSR; only the cross-section lookup is replaced by representative FSR
     * data. */
    zres_apply_fsr_cluster_macro_t_nuf(1, 1);
    zres_apply_fsr_cluster_chi();
    printf(" [  CLUST  ]  Applied representative FSR values to Sigma_t, nuSigma_f, and Chi\n");
    fflush(stdout);

    const double _compressStartTime = paper_wall_time_seconds();
    zres_compress_external_fsr_scatter_csr();
    const double _compressEndTime = paper_wall_time_seconds();

    printf(" [  CLUST  ]  External FSR CSR compression time: %.6f sec\n",
           _compressEndTime - _compressStartTime);
    printf(" [  CLUST  ]  Total FSR cluster preparation time: %.6f sec\n",
           _compressEndTime - _clusterStartTime);
    TimeRec_ClusterPreparation +=
        (FP_PRECISION_TIME) (_compressEndTime - _clusterStartTime);
    fflush(stdout);

    zres_close_memlog();
}
