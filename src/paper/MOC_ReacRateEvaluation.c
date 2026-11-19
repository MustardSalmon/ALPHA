/********************************************************************************/
/* Minimal reaction-rate routines used by the publication transport snapshot.  */
/* The resonance, CMFD, and legacy post-processing routines from the upstream   */
/* translation unit are intentionally not part of this transport-only file.    */
/********************************************************************************/
#include "MOC_CommonMacro.h"
#include "MOC_Geometry.h"
#include "MOC_Input.h"
#include "MOC_Macro_XSs.h"
#include "MOC_NTE_Solver.h"
#include "MOC_Output.h"
#include "MOC_ReacRateEvaluation.h"
#include "MOC_Track.h"

#include "R_xs_treat.h"
#include "R_I_O_File.h"
#include "R_DH_PIJ.h"
#include "R_DH_Mat_XS.h"

#ifdef MPI_USED
#include "mpi.h"
#endif

M_INT32 NumFissPinsSubDom[1] = {0};
M_INT32 NumFissPinsGlobDom[1] = {0};

FP_DOUBLE CalcReacRate_SubDom(FP_PRECISION_FLUX *_scalarFlux_subDom,
    int _numLayersZ, int _numCellsInDom, int *_numFSRsInCell, int _numGroup,
    int *_cellRTMType, FP_PRECISION *_macroXS_r, FP_PRECISION **_inpXSs_r,
    struct RTMFSRINFO **_RTModFSRsInfo,
    int *_sumNumFSRsBasedGlobCellID)
{
    int g, _cell, _reg, _lay, _fsrID;
    FP_PRECISION _XS;
    FP_DOUBLE _reac, _totReac = 0.0, _fsrVolume;
    int _ino, _inr, itio, itla;
    FP_PRECISION vol_fra0;

    if (Input_MATorXSs == 100 || Input_Type == 100) {
        Cal_Eff_Scalar_Flux_DH(ProcID);
        for (_lay = 0; _lay < _numLayersZ; _lay++) {
            for (_cell = 0; _cell < _numCellsInDom; _cell++) {
                for (_reg = 0;
                     _reg < _numFSRsInCell[_lay * _numCellsInDom + _cell];
                     _reg++) {
                    _fsrVolume = (FP_PRECISION)
                        (_RTModFSRsInfo[_cellRTMType[_lay * _numCellsInDom + _cell]]
                            [_reg].fsr_area);
                    _fsrID = _sumNumFSRsBasedGlobCellID
                        [_lay * _numCellsInDom + _cell] + _reg;
                    _reac = 0.0;

                    if (Triso_Num[_fsrID] == 0) {
                        for (g = 0; g < _numGroup; g++) {
                            _XS = GetFSRXSs(_lay, _cell, _reg, _numCellsInDom,
                                NumGroups, g, _macroXS_r, _inpXSs_r,
                                _sumNumFSRsBasedGlobCellID);
                            _reac += _XS * _scalarFlux_subDom
                                [_fsrID * _numGroup + g];
                        }
                        _totReac += _reac * _fsrVolume;
                    }

                    if (Triso_Num[_fsrID] > 0) {
                        vol_fra0 = 1.0;
                        for (itio = 0; itio < Triso_Num[_fsrID]; itio++) {
                            _ino = Triso_Sum_byGlob_FsrID[_fsrID] + itio;
                            vol_fra0 -= Triso_Vol_Fra[_ino];
                        }

                        for (itio = 0; itio < Triso_Num[_fsrID]; itio++) {
                            _ino = Triso_Sum_byGlob_FsrID[_fsrID] + itio;
                            for (itla = 0; itla < Triso_Lay_Num[_ino]; itla++) {
                                _inr = TLay_Sum_byGlob_TrisoID[_ino] + itla;
                                _reac = 0.0;
                                for (g = 0; g < _numGroup; g++) {
                                    _XS = MacroXS_nf_DH[_inr * _numGroup + g];
                                    _reac += _XS * ScalarFlux_DH
                                        [_inr * _numGroup + g]
                                        * Triso_Lay_Vol_Fra[_inr];
                                }
                                _totReac += _reac;
                            }
                        }

                        _reac = 0.0;
                        for (g = 0; g < _numGroup; g++) {
                            _XS = MacroXS_nf_MTX[_fsrID * _numGroup + g];
                            _reac += _XS * _scalarFlux_subDom
                                [_fsrID * _numGroup + g] * vol_fra0;
                        }
                        _totReac += _reac;
                        _totReac *= _fsrVolume;
                    }
                }
            }
        }
    } else {
        for (_fsrID = 0; _fsrID < TotalNumFSRs_subDom; _fsrID++) {
#if defined(M_YQZ_CMFD_Keff)
            _fsrVolume = 1.0;
#else
            _fsrVolume = FSRVol_SubDom[_fsrID];
#endif
            _reac = 0.0;
            for (g = 0; g < _numGroup; g++) {
                _XS = _macroXS_r[_fsrID * _numGroup + g];
                _reac += (FP_DOUBLE)(_XS * _scalarFlux_subDom
                    [_fsrID * _numGroup + g]);
            }
            _totReac += _reac * _fsrVolume;
        }
    }

    return _totReac;
}

FP_DOUBLE MOC_CalcGlobalReacRate(FP_PRECISION *_reacRate_procs,
    int _procID, int _numProcs, FP_PRECISION_FLUX *_scalarFlux_subDom,
    int _numLayersZ, int _numCellsInSubDom, int *_numFSRsinCell_subDom,
    int _numGroup, int *_cellRTMType_subDom, FP_PRECISION *_macroXS_r,
    FP_PRECISION **_inpXSs_r, struct RTMFSRINFO **_RTModFSRsInfo,
    int *_sumNumFSRsBasedGlobCellID)
{
    FP_DOUBLE _globReacRate = 0.0;
    FP_DOUBLE _reacRate_SubDom = CalcReacRate_SubDom(
        _scalarFlux_subDom, _numLayersZ, _numCellsInSubDom,
        _numFSRsinCell_subDom, _numGroup, _cellRTMType_subDom, _macroXS_r,
        _inpXSs_r, _RTModFSRsInfo, _sumNumFSRsBasedGlobCellID);

#ifdef MPI_USED
    MPI_Reduce((void *)&_reacRate_SubDom, (void *)&_globReacRate, 1,
        MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    _globReacRate = _reacRate_SubDom;
#endif
    return _globReacRate;
}

void CalcPinPower_SubDom(int _numLayersZ, FP_PRECISION *_scalarFlux_subDom,
    int _numCellsInDom, int *_numFSRsInCell, int _numGroup,
    FP_PRECISION *_norm_pin_power, int *_sumNumFSRsBasedGlobCellID,
    FP_PRECISION_GEO *_FSRArea_SubDom, FP_PRECISION *_macroXS_f,
    FP_PRECISION_GEO *_WidthLayersInCore)
{
    if (OUTPUT_normPinPower) {
        int g, _lay, _cell, _reg, _fsrID, _ind, _pinID;
        FP_PRECISION _fiss_rate, _ene_perFiss = 1.0;
        FP_PRECISION _fsrVolume, _hight;

        for (_lay = 0; _lay < _numLayersZ; _lay++) {
            _hight = _WidthLayersInCore[_lay];
            for (_cell = 0; _cell < _numCellsInDom; _cell++) {
                _fiss_rate = 0.0;
                _pinID = _lay * _numCellsInDom + _cell;
                for (_reg = 0; _reg < _numFSRsInCell[_pinID]; _reg++) {
                    _fsrID = _sumNumFSRsBasedGlobCellID[_pinID] + _reg;
                    _fsrVolume = _FSRArea_SubDom[_fsrID];
                    for (g = 0; g < _numGroup; g++) {
                        _ind = _fsrID * _numGroup + g;
                        _fiss_rate += _macroXS_f[_ind]
                            * _scalarFlux_subDom[_ind]
                            * _fsrVolume * _ene_perFiss;
                    }
                }
                _norm_pin_power[_pinID] = _fiss_rate * _hight;
            }
        }
    }
}

void SumPinPower_Lay(int _PinPowerSumNumSet, int *_PinPowerSumLayBound,
    int _numCellsInDomPerLay, FP_PRECISION *_NormPinPower,
    FP_PRECISION *_NormPinPower_SumLay)
{
    int i, j, _cell;
    FP_PRECISION _sum;

    if (OUTPUT_normPinPower) {
        for (i = 0; i < _PinPowerSumNumSet; i++) {
            for (_cell = 0; _cell < _numCellsInDomPerLay; _cell++) {
                _sum = 0.0;
                for (j = _PinPowerSumLayBound[i];
                     j < _PinPowerSumLayBound[i + 1]; j++) {
                    _sum += _NormPinPower[j * _numCellsInDomPerLay + _cell];
                }
                _NormPinPower_SumLay[i * _numCellsInDomPerLay + _cell] = _sum;
            }
        }
    }
}
