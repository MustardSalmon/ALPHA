/* Minimal exponential-table range check used by the 2-D GPU transport path. */
#include <stdio.h>

#include "MOC_CommonMacro.h"
#include "MOC_ExpEvaluation.h"
#include "MOC_Input.h"
#include "MOC_NTE_Solver.h"

extern FILE *LogFile;

void ErrorCheck_Jacobi_MOCSolver_2D_expTab(
    int _procID, int _probType, int _threadID, int _numGroups,
    int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim,
    int _groupBoundUpLim, int _polarBoundLowLim, int _polarBoundUpLim,
    int _numCellsInDom_Layer, int *_cellRTMGeoType, int *_trackIndexInLTs,
    struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT,
    int *_sumNumFSRsBasedGlobCellID, int *_numSegsInTrack,
    FP_PRECISION *_macroXS_t, int _totNumTracksInOneRTM_2PI,
    int *_indexTrackInRTM_SumAzim_2PI, int _numOMPThreads_comp,
    int *_sumNumSegsInRTMTrack_2PI, FP_PRECISION *_segInfoInTrack_2PI_length,
    int *_segInfoInTrack_2PI_AntiClockFSR, int _numLongtrack_halfSpace,
    FP_PRECISION *_SinThetaForExp, FP_PRECISION_GEO *_LayWidth,
    FP_PRECISION *_CosTheta, int *_LTsIndex_AzimAngID)
{
    int _cell, _pol, _ray, _track, _seg, _reg, g, _ltIter, _fsrID;
    int _trackIndex;
    FP_PRECISION _length, _x2 = 0.0, _xs_t, _sigmaTByLength;
    FP_PRECISION _twoOneOverWidth;
    int _globCellID, _index, _lay, _azim;
    M_UINT64 _segCount, _x2OverCount;
    FP_PRECISION _max_x2;

    _segCount = 0;
    _x2OverCount = 0;
    _max_x2 = 0.0;

    for (_lay = _layBoundLowLim; _lay < _layBoundUpLim; _lay++) {
        _twoOneOverWidth = (FP_PRECISION)(2.0) / _LayWidth[_lay];
        if (_probType == 1) {
            _twoOneOverWidth = 0.0;
        }

        _ltIter = _threadID;
        while (_ltIter < _numLongtrack_halfSpace) {
            _azim = _LTsIndex_AzimAngID[_ltIter];

            for (_track = _trackIndexInLTs[_ltIter];
                 _track < _trackIndexInLTs[_ltIter + 1]; _track++) {
                _cell = _tracksInfoInLT[_track].cell_ID;
                _ray = _tracksInfoInLT[_track].track_ID;
                _globCellID = _lay * _numCellsInDom_Layer + _cell;
                _trackIndex = _totNumTracksInOneRTM_2PI *
                                  _cellRTMGeoType[_globCellID] +
                              _indexTrackInRTM_SumAzim_2PI[_azim] + _ray;

                for (_seg = 0; _seg < _numSegsInTrack[_trackIndex]; _seg++) {
                    _segCount++;
                    _length = _segInfoInTrack_2PI_length[
                        _sumNumSegsInRTMTrack_2PI[_trackIndex] + _seg];
                    _reg = _segInfoInTrack_2PI_AntiClockFSR[
                        _sumNumSegsInRTMTrack_2PI[_trackIndex] + _seg];
                    _fsrID = _sumNumFSRsBasedGlobCellID[_globCellID] + _reg;

                    for (g = _groupBoundLowLim; g < _groupBoundUpLim; g++) {
                        _index = _fsrID * _numGroups + g;
                        _xs_t = _macroXS_t[_index];
#if defined(M_2DMOC)
                        _sigmaTByLength = _xs_t * _length;
#endif

                        for (_pol = _polarBoundLowLim;
                             _pol < _polarBoundUpLim; _pol++) {
#if defined(M_2DMOC)
                            _x2 = _sigmaTByLength * _SinThetaForExp[_pol];
#endif
                            if (_x2 > NumExpPoints) {
                                _x2OverCount++;
                                if (_max_x2 < _x2) {
                                    _max_x2 = _x2;
                                }
                            }
                        }
                    }
                }
            }
            _ltIter += _numOMPThreads_comp;
        }
    }

    if (_procID == 0) {
        printf("\n [  INFO   ]  Checking the ExpInte range.");
        fprintf(LogFile, "\n [  INFO   ]  Checking the ExpInte range.");
    }
    if (_x2OverCount > 0) {
        printf("\n [ WARNING ]  Process %d, NumSeg=%llu, LayBound[%d, %d], "
               "GroupBound[%d, %d], PolBound[%d, %d]",
               _procID, _segCount * 2, _layBoundLowLim, _layBoundUpLim,
               _groupBoundLowLim, _groupBoundUpLim, _polarBoundLowLim,
               _polarBoundUpLim);
        printf("\n              Process %d, Number of calculation point over the expTable "
               "range: %llu (%.6lf %%)",
               _procID, _x2OverCount * 2,
               (double)(_x2OverCount) /
                   (double)(_segCount * (_groupBoundUpLim - _groupBoundLowLim) *
                            (_polarBoundUpLim - _polarBoundLowLim)) *
                   100.0);
        printf("\n              Process %d, Maximum optical length is %.2lf", _procID,
               _max_x2 / (double)(NumExpPoints_One));
        printf("\n              Process %d, Maximum expTable range is %.2lf", _procID,
               (double)(ExpTab_Range));
        printf("\n              Process %d, User can reset the parameter \"ExpTab_Range\" "
               "or use the \"expFunc\" TS kernel",
               _procID);
        printf("\n              Process %d, Automatically set: ", _procID);
        if (_probType == 0) {
            CPU_TSFunctionType = 1;
            GPU_TSKernelType = 1;
            printf("CPU_TSFunctionType = %d (expFunc), GPU_TSKernelType = %d "
                   "(ddExpFunc)",
                   CPU_TSFunctionType, GPU_TSKernelType);
        }
        if (_probType == 1) {
            CPU_FixSourSolver = 1;
            GPU_FixSourSolver = 1;
            printf("CPU_FixSourSolver = %d (expFunc), GPU_FixSourSolver = %d "
                   "(ddExpFunc)",
                   CPU_FixSourSolver, GPU_FixSourSolver);
        }
    } else if (_procID == 0) {
        printf("\n              Process %d, expTable range is sufficient for this problem.",
               _procID);
    }
}
