#include "MOC_GPU_TSSolver.cuh"
#include "MOC_GPUConstants.cuh"

/*
 * Clustered high-order source kernels.
 *
 * The clustered CSR representation keeps one row-index entry for every FSR,
 * while non-representative FSRs point to the representative FSR's value
 * interval.  Consequently, the source and flux arrays remain full-FSR arrays;
 * only the CSR values are shared.  These kernels are deliberately separate
 * from the original Type 11 and Type 13-21 kernels.  The only semantic
 * difference is that the row end is read explicitly instead of inferred from
 * rowStart[row + 1].
 */

__device__ __forceinline__ FP_PRECISION GPUGetHOScattXS_Cluster(
    FP_PRECISION *_GPUMacroXS_s,
    M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
    M_UINT64 *_GPUMacroXS_s_NZRowEndID_Layer,
    M_UINT64 *_GPUMacroXS_s_NZColStartID,
    size_t _fsrID,
    size_t _lID,
    size_t _groupInID,
    size_t _groupOutID)
{
    const size_t _numGrp = *GPUNumMOCGroups;
    const size_t _numOrder = *GPUNumScattOrder;

#ifndef TRANSPOSE_SCATT_MTX
    const size_t _rowGroupID = _groupInID;
    const size_t _targetGroupID = _groupOutID;
#else
    const size_t _rowGroupID = _groupOutID;
    const size_t _targetGroupID = _groupInID;
#endif

    const size_t _rowID = _fsrID * (_numOrder + 1) * _numGrp
        + _lID * _numGrp + _rowGroupID;
    const M_UINT64 _rowStart = _GPUMacroXS_s_NZRowStartID_Layer[_rowID];
    const M_UINT64 _rowEnd = _GPUMacroXS_s_NZRowEndID_Layer[_rowID];
    const size_t _colStart = (size_t) _GPUMacroXS_s_NZColStartID[_rowID];

    if (_targetGroupID < _colStart) {
        return 0.0;
    }

    const size_t _offset = _targetGroupID - _colStart;
    if (_offset >= (size_t) (_rowEnd - _rowStart)) {
        return 0.0;
    }

    return _GPUMacroXS_s[_rowStart + _offset];
}

__device__ __forceinline__ FP_PRECISION GPUProjectHOFluxForAngle_Cluster(
    FP_PRECISION_FLUX *_GPUScalarFlux,
    FP_PRECISION *_GPUSpheHarmCoeff,
    size_t _fsrID,
    size_t _groupInID,
    int _lID,
    int _azimID,
    int _polID)
{
    const size_t _numGrp = *GPUNumMOCGroups;
    const size_t _numFsr = *GPUNumFSRsLayer;
    const size_t _numAng = (size_t) (*GPUNumAzimAng) * (size_t) (*GPUNumPolarAng);
    const size_t _angleID = (size_t) _azimID * (size_t) (*GPUNumPolarAng)
        + (size_t) _polID;
    const size_t _fluxGroupID = _fsrID * _numGrp + _groupInID;
    const int _lmStart = _lID * _lID;
    FP_PRECISION _projectedFlux = 0.0;

    for (int _mOffset = 0; _mOffset < 2 * _lID + 1; ++_mOffset) {
        const size_t _lmID = (size_t) (_lmStart + _mOffset);
        const size_t _fluxID = _lmID * _numFsr * _numGrp + _fluxGroupID;
        const size_t _ylmID = _lmID * _numAng + _angleID;
        _projectedFlux += GPUCoeffA_HOScatt[_lmID]
            * (FP_PRECISION) _GPUScalarFlux[_fluxID]
            * _GPUSpheHarmCoeff[_ylmID];
    }

    return _projectedFlux;
}

__global__ void GPUComputeFSRScatterSourceLMOnDevice_grpOut_Cluster(
    FP_PRECISION *_GPUFSRAngScatterSourceLM,
    FP_PRECISION_FLUX *_GPUScalarFlux,
    FP_PRECISION *_GPUMacroXS_s,
    M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
    M_UINT64 *_GPUMacroXS_s_NZRowEndID_Layer,
    M_UINT64 *_GPUMacroXS_s_NZColStartID)
{
    const size_t _numGrp = *GPUNumMOCGroups;
    const size_t _numFsr = *GPUNumFSRsLayer;
    const size_t _numOrder = *GPUNumScattOrder;
    const size_t _totalLM = (_numOrder + 1) * (_numOrder + 1);
    size_t _tid = threadIdx.x + blockDim.x * blockIdx.x;

    size_t _lmID = _tid / (_numFsr * _numGrp);
    size_t _fsrID = _tid % (_numFsr * _numGrp) / _numGrp;
    size_t _groupOutID = _tid % (_numFsr * _numGrp) % _numGrp;

    while (_lmID < _totalLM) {
        const size_t _lID = _lmID == 0 ? 0 : (size_t) sqrt((double) _lmID);
        FP_PRECISION _source = 0.0;

        for (size_t _groupInID = 0; _groupInID < _numGrp; ++_groupInID) {
            const FP_PRECISION _xs = GPUGetHOScattXS_Cluster(
                _GPUMacroXS_s, _GPUMacroXS_s_NZRowStartID_Layer,
                _GPUMacroXS_s_NZRowEndID_Layer, _GPUMacroXS_s_NZColStartID,
                _fsrID, _lID, _groupInID, _groupOutID);
            if (_xs == 0.0) {
                continue;
            }
            const size_t _fluxID = _lmID * _numFsr * _numGrp
                + _fsrID * _numGrp + _groupInID;
            _source += (FP_PRECISION) (2 * _lID + 1) * _xs
                * _GPUScalarFlux[_fluxID];
        }

        _GPUFSRAngScatterSourceLM[_lmID * _numFsr * _numGrp
            + _fsrID * _numGrp + _groupOutID] = _source;

        _tid += blockDim.x * gridDim.x;
        _lmID = _tid / (_numFsr * _numGrp);
        _fsrID = _tid % (_numFsr * _numGrp) / _numGrp;
        _groupOutID = _tid % (_numFsr * _numGrp) % _numGrp;
    }
}

__global__ void GPUComputeFSRScatterSourceLMOnDevice_grpIn_Cluster(
    FP_PRECISION *_GPUFSRAngScatterSourceLM,
    FP_PRECISION_FLUX *_GPUScalarFlux,
    FP_PRECISION *_GPUMacroXS_s,
    M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
    M_UINT64 *_GPUMacroXS_s_NZRowEndID_Layer,
    M_UINT64 *_GPUMacroXS_s_NZColStartID)
{
    const size_t _numGrp = *GPUNumMOCGroups;
    const size_t _numFsr = *GPUNumFSRsLayer;
    const size_t _numOrder = *GPUNumScattOrder;
    const size_t _totalLM = (_numOrder + 1) * (_numOrder + 1);
    size_t _tid = threadIdx.x + blockDim.x * blockIdx.x;

    size_t _lmID = _tid / (_numFsr * _numGrp);
    size_t _fsrID = _tid % (_numFsr * _numGrp) / _numGrp;
    size_t _groupInID = _tid % _numGrp;

    while (_lmID < _totalLM) {
        const size_t _lID = _lmID == 0 ? 0 : (size_t) sqrt((double) _lmID);
        const size_t _fluxID = _lmID * _numFsr * _numGrp
            + _fsrID * _numGrp + _groupInID;
        const FP_PRECISION _flux = _GPUScalarFlux[_fluxID];

        for (size_t _groupOutID = 0; _groupOutID < _numGrp; ++_groupOutID) {
            const FP_PRECISION _xs = GPUGetHOScattXS_Cluster(
                _GPUMacroXS_s, _GPUMacroXS_s_NZRowStartID_Layer,
                _GPUMacroXS_s_NZRowEndID_Layer, _GPUMacroXS_s_NZColStartID,
                _fsrID, _lID, _groupInID, _groupOutID);
            if (_xs == 0.0) {
                continue;
            }
            const size_t _outID = _lmID * _numFsr * _numGrp
                + _fsrID * _numGrp + _groupOutID;
            const FP_PRECISION _value = (FP_PRECISION) (2 * _lID + 1)
                * _flux * _xs;
            atomicAdd(&_GPUFSRAngScatterSourceLM[_outID], _value);
        }

        _tid += blockDim.x * gridDim.x;
        _lmID = _tid / (_numFsr * _numGrp);
        _fsrID = _tid % (_numFsr * _numGrp) / _numGrp;
        _groupInID = _tid % _numGrp;
    }
}

__global__ void GPUComputeFSRScattSourceOnDevice_FsrGrpInAng_Cluster(
    FP_PRECISION *_GPUFSRAngScattSour,
    FP_PRECISION_FLUX *_GPUScalarFlux,
    FP_PRECISION *_GPUMacroXS_s,
    M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
    M_UINT64 *_GPUMacroXS_s_NZRowEndID_Layer,
    M_UINT64 *_GPUMacroXS_s_NZColStartID,
    FP_PRECISION *_GPUSpheHarmCoeff,
    int _polarBoundLowLim)
{
    const size_t _numFsr = *GPUNumFSRsLayer;
    const size_t _numGrp = *GPUNumMOCGroups;
    const int _numAzim = *GPUNumAzimAng;
#if defined(M_2DMOC)
    const int _numPol = *GPUNumPolarAng;
#else
    const int _numPol = (*GPUNumPolarAng) / 2;
#endif
    const size_t _numAng = (size_t) _numAzim * (size_t) _numPol;
    const size_t _total = _numFsr * _numGrp * _numAng;

    for (size_t _tid = threadIdx.x + blockDim.x * blockIdx.x; _tid < _total;
         _tid += (size_t) blockDim.x * gridDim.x) {
        const size_t _fsrID = _tid / (_numGrp * _numAng);
        const size_t _rem = _tid % (_numGrp * _numAng);
        const size_t _groupInID = _rem / _numAng;
        const size_t _angleID = _rem % _numAng;
        const int _azimID = (int) (_angleID / (size_t) _numPol);
        const int _localPolID = (int) (_angleID % (size_t) _numPol);
        const int _polID = _localPolID + _polarBoundLowLim;

        for (int _lID = 0; _lID <= *GPUNumScattOrder; ++_lID) {
            const FP_PRECISION _projectedFlux = GPUProjectHOFluxForAngle_Cluster(
                _GPUScalarFlux, _GPUSpheHarmCoeff, _fsrID, _groupInID,
                _lID, _azimID, _polID);
            for (size_t _groupOutID = 0; _groupOutID < _numGrp; ++_groupOutID) {
                const FP_PRECISION _xs = GPUGetHOScattXS_Cluster(
                    _GPUMacroXS_s, _GPUMacroXS_s_NZRowStartID_Layer,
                    _GPUMacroXS_s_NZRowEndID_Layer, _GPUMacroXS_s_NZColStartID,
                    _fsrID, (size_t) _lID, _groupInID, _groupOutID);
                if (_xs == 0.0) {
                    continue;
                }
                const size_t _outID = _fsrID * _numGrp * _numAng
                    + _groupOutID * _numAng + _angleID;
                const FP_PRECISION _source = (FP_PRECISION) (2 * _lID + 1)
                    * _xs * _projectedFlux * ONE_OVER_FOUR_PI;
                atomicAdd(&_GPUFSRAngScattSour[_outID], _source);
            }
        }
    }
}

__global__ void GPUComputeFSRScattSourceOnDevice_FsrGrpOutAng_Cluster(
    FP_PRECISION *_GPUFSRAngScattSour,
    FP_PRECISION_FLUX *_GPUScalarFlux,
    FP_PRECISION *_GPUMacroXS_s,
    M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
    M_UINT64 *_GPUMacroXS_s_NZRowEndID_Layer,
    M_UINT64 *_GPUMacroXS_s_NZColStartID,
    FP_PRECISION *_GPUSpheHarmCoeff,
    int _polarBoundLowLim)
{
    const size_t _numFsr = *GPUNumFSRsLayer;
    const size_t _numGrp = *GPUNumMOCGroups;
    const int _numAzim = *GPUNumAzimAng;
#if defined(M_2DMOC)
    const int _numPol = *GPUNumPolarAng;
#else
    const int _numPol = (*GPUNumPolarAng) / 2;
#endif
    const size_t _numAng = (size_t) _numAzim * (size_t) _numPol;
    const size_t _total = _numFsr * _numGrp * _numAng;

    for (size_t _tid = threadIdx.x + blockDim.x * blockIdx.x; _tid < _total;
         _tid += (size_t) blockDim.x * gridDim.x) {
        const size_t _fsrID = _tid / (_numGrp * _numAng);
        const size_t _rem = _tid % (_numGrp * _numAng);
        const size_t _groupOutID = _rem / _numAng;
        const size_t _angleID = _rem % _numAng;
        const int _azimID = (int) (_angleID / (size_t) _numPol);
        const int _localPolID = (int) (_angleID % (size_t) _numPol);
        const int _polID = _localPolID + _polarBoundLowLim;
        FP_PRECISION _scattSource = 0.0;

        for (int _lID = 0; _lID <= *GPUNumScattOrder; ++_lID) {
            for (size_t _groupInID = 0; _groupInID < _numGrp; ++_groupInID) {
                const FP_PRECISION _xs = GPUGetHOScattXS_Cluster(
                    _GPUMacroXS_s, _GPUMacroXS_s_NZRowStartID_Layer,
                    _GPUMacroXS_s_NZRowEndID_Layer, _GPUMacroXS_s_NZColStartID,
                    _fsrID, (size_t) _lID, _groupInID, _groupOutID);
                if (_xs == 0.0) {
                    continue;
                }
                const FP_PRECISION _projectedFlux = GPUProjectHOFluxForAngle_Cluster(
                    _GPUScalarFlux, _GPUSpheHarmCoeff, _fsrID, _groupInID,
                    _lID, _azimID, _polID);
                _scattSource += (FP_PRECISION) (2 * _lID + 1)
                    * _xs * _projectedFlux;
            }
        }

        const size_t _outID = _fsrID * _numGrp * _numAng
            + _groupOutID * _numAng + _angleID;
        _GPUFSRAngScattSour[_outID] = _scattSource * ONE_OVER_FOUR_PI;
    }
}

