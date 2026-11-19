/********************************************************************************/
/* @ File: MOC_GPU_TSSolver.c
/*   introduction: This file contains functions that relate to the Transport
/*                 Sweep Solver on GPU
/* @ Created     : 2019/08/14, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*   1st modified:
/********************************************************************************/
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <math.h>
//#include <math_functions.h>
//#include <math_functions.h>
#include <curand_kernel.h>
#include <time.h>
#include <cuda_occupancy.h>
#include "MOC_CommonMacro.h"
#include "MOC_GPUCmfd.cuh"
// extern "C" {
#include "MOC_GPU_TSSolver.cuh"
#include "MOC_GPUConstants.cuh"
// }
#include "MOC_ExpEvaluation.h"

extern "C" {
FP_PRECISION *GPUMacroXS_t = NULL;
FP_PRECISION *GPUMacroXS_t_inv = NULL;
FP_PRECISION *GPUMOCSource = NULL;
FP_PRECISION_FLUX *GPUScalarFlux = NULL;
int GPUGridDim_source_para_Layer;
}

/* Accumulate one local FSR x group x batch-azimuth angular-flux block into
 * the moment buffer used to start the next inner iteration. */
__global__ void GPUAccumAngFlux_ScalarFlux_HOScatt_PerAzim(
	int _lay, int *_GPUNumFSRs_Layer, int _polarBoundLowLim,
	int _polarBoundUpLim, FP_PRECISION_FLUX *_aveAngFlux,
	FP_PRECISION_FLUX *_scalarFlux, int _polOffset,
	FP_PRECISION *_GPUSpheHarmCoeff, int _azimBegin, int _azimEnd)
{
	const size_t _numFsr = (size_t) _GPUNumFSRs_Layer[_lay];
	const size_t _numGrp = *GPUNumMOCGroups;
	const size_t _numPol = *GPUNumPolarAng;
	const size_t _numAzim = *GPUNumAzimAng;
	const size_t _numLM = ((size_t) *GPUNumScattOrder + 1) *
		((size_t) *GPUNumScattOrder + 1);
	const size_t _tid = (size_t) threadIdx.x +
		(size_t) blockIdx.x * blockDim.x;
	const size_t _total = _numFsr * _numGrp * _numLM;
	const size_t _numBatchAzim = (size_t) (_azimEnd - _azimBegin);
	const size_t _batchAng = _numBatchAzim * _numPol;

	for (size_t _work = _tid; _work < _total; _work +=
		(size_t) blockDim.x * gridDim.x) {
		const size_t _lmID = _work / (_numFsr * _numGrp);
		const size_t _rem = _work % (_numFsr * _numGrp);
		const size_t _fsrID = _rem / _numGrp;
		const size_t _groupID = _rem % _numGrp;
		FP_PRECISION_FLUX _moment = (FP_PRECISION_FLUX) 0.0;

		for (size_t _localAzimID = 0; _localAzimID < _numBatchAzim; ++_localAzimID) {
			const size_t _azimID = (size_t) _azimBegin + _localAzimID;
			for (int _pol = _polarBoundLowLim; _pol < _polarBoundUpLim; ++_pol) {
				const size_t _localPolID = (size_t) (_pol - _polOffset);
				const size_t _angID = _fsrID * _numGrp * _batchAng +
					_groupID * _batchAng + _localAzimID * _numPol + _localPolID;
				const size_t _ylmID = _lmID * _numAzim * _numPol +
					_azimID * _numPol + (size_t) _pol;
				_moment += _aveAngFlux[_angID] * GPUTotalWeights[_azimID * *GPUNumPolarAng + _pol] *
					_GPUSpheHarmCoeff[_ylmID];
			}
		}

		_scalarFlux[_lmID * _numFsr * _numGrp + _fsrID * _numGrp + _groupID] += _moment;
	}
}

#ifdef TESTR
    #include "condition.h"
#endif

// 需要测试裂变矩阵是否有效 [  WTDEBUG  ]
__global__ void GPUComputeFSRFissSourceOnDevice(FP_PRECISION_FLUX *_GPUScalarFlux, \
	FP_PRECISION *_GPUSource, FP_PRECISION *_GPUMacroXS_nf, FP_PRECISION *_GPUMacroXS_chi, FP_PRECISION _inverseKeff)
{
	int _tid = threadIdx.x + blockIdx.x * blockDim.x, g1, _fsrID, _ind;
	FP_PRECISION _chi, _xs_nf, _fissSour;
	FP_PRECISION_FLUX _flux;
	_fsrID = _tid / (*GPUNumMOCGroups);

#ifdef FIS_MTX
    int _indChi, g2;
#endif

	/* Iterate over all FSRs */
	while (_fsrID < (*GPUNumFSRsLayer)) {
		_chi = _GPUMacroXS_chi[_tid];
		_fissSour = 0.0;

    #ifdef FIS_MTX
        g2 = _tid % (*GPUNumMOCGroups);
    #endif

		for (g1 = 0; g1 < (*GPUNumMOCGroups); g1++) {
			_ind = _fsrID*(*GPUNumMOCGroups) + g1;
			_xs_nf = _GPUMacroXS_nf[_ind];
			_flux = _GPUScalarFlux[_ind];
        #ifdef FIS_MTX
            _indChi = _fsrID * (*GPUNumMOCGroups) * (*GPUNumMOCGroups) + g1 * (*GPUNumMOCGroups) + g2;
            _chi = _GPUMacroXS_chi[_indChi];
            _fissSour += _xs_nf * _flux * _chi;
        #else
			_fissSour += _xs_nf * _flux;
        #endif
		}

    #ifdef FIS_MTX
        _fissSour *= _inverseKeff;
    #else
		_fissSour *= (_chi*_inverseKeff);
    #endif
		_GPUSource[_tid] = _fissSour;

#if defined(M_Use4PI)
		_GPUSource[_tid] = _GPUSource[_tid] * ONE_OVER_FOUR_PI;
#endif // defined(M_Use4PI)

        _tid += blockDim.x * gridDim.x;
		_fsrID = _tid / (*GPUNumMOCGroups);
	}
}

// 以固定源覆盖裂变源
__global__ void GPUOverWriteFissSourceWithGivenSource(FP_PRECISION *_GPUSource, FP_PRECISION *_GPUMacroXS_nf)
{
	int _tid = threadIdx.x + blockIdx.x * blockDim.x;

	int _fsrID = _tid / (*GPUNumMOCGroups);

	/* Iterate over all FSRs */
	while (_fsrID < (*GPUNumFSRsLayer)) {

		_GPUSource[_tid] = _GPUMacroXS_nf[_tid];

        _tid += blockDim.x * gridDim.x;
		_fsrID = _tid / (*GPUNumMOCGroups);
	}
}

__device__ FP_PRECISION GPUComputeFSRScattSourceOnDevice_paraFsr_paraGroup_Layer(FP_PRECISION_FLUX *_GPUScalarFlux, \
	FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID,\
    FP_PRECISION *_GPUSpheHarmCoeff, int _fsrID, int _groupID, int _azimID, int _polID)
{
	int g1, _ind, g2, _ind1;
	int l, m, _ind5, _count1, _ind2, _numAng, _ind4, _ind3;
	FP_PRECISION _xs_s, _scattSour, _accu, _alm, _ylm;
	M_UINT64 _id;

	_scattSour = 0;

	_numAng = (*GPUNumAzimAng) * (*GPUNumPolarAng);
	_ind = _azimID * (*GPUNumPolarAng) + _polID;
	_ind4 = (*GPUNumFSRsLayer) * (*GPUNumMOCGroups);

	for (l=0;l<=(*GPUNumScattOrder);l++) {
		for (g1 = 0; g1 < (*GPUNumMOCGroups); g1++) {
			_ind1 = _fsrID * (*GPUNumMOCGroups) + g1;
			_ind5 = _fsrID * ((*GPUNumScattOrder) + 1) * (*GPUNumMOCGroups) + l * (*GPUNumMOCGroups) + g1;
			g2=_GPUMacroXS_s_NZColStartID[_ind5];
			for (_id=_GPUMacroXS_s_NZRowStartID_Layer[_ind5];_id<_GPUMacroXS_s_NZRowStartID_Layer[_ind5+1];_id++) {
				if (g2==_groupID) {
					_xs_s = (FP_PRECISION)(2 * l + 1) * _GPUMacroXS_s[_id];
					_count1=0;
					_accu=0.0;
					for (m=-l;m<=l;m++) {
						_ind2 = l * l + _count1;
						_alm = GPUCoeffA_HOScatt[_ind2];
						_ylm = _GPUSpheHarmCoeff[_ind2 * _numAng + _ind];
						_ind3 = _ind2 * _ind4 + _ind1;
						_accu += _alm * _GPUScalarFlux[_ind3] * _ylm;
						_count1++;
					}
					_scattSour += _xs_s * _accu;
					break;
				}
				g2++;
			}
		}
	}

#if defined(M_Use4PI)
    _scattSour *= ONE_OVER_FOUR_PI;
#endif
	return _scattSour;
}


#define IDX_2D(i, j, lda) ((i) * (lda) + (j))
#define IDX_3D(i, j, k, lda, ldb) ((i) * (lda) * (ldb) + (j) * (ldb) + (k))
#define IDX_4D(a, b, c, d, lda, ldb, ldc) ((a) * (lda)*(ldb)*(ldc) + (b)*(ldb)*(ldc) + (c)*(ldc) + (d))

__global__ void GPUInitFSRHOSourceOnDevice(FP_PRECISION *_GPUFSRAngScattSour, FP_PRECISION_FLUX *_GPUScalarFlux, \
	FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID,\
    FP_PRECISION *_GPUSpheHarmCoeff, int _polarBoundLowLim)
{
	int _tid, _numPol, _angMul1, _angMul2;
	int _fsrID, _groupID, _azimID, _polID;

	_tid = threadIdx.x + blockIdx.x * blockDim.x;

#if defined(M_2DMOC)
    _numPol = (*GPUNumPolarAng);
#else
    _numPol = (*GPUNumPolarAng)/2;
#endif 
	_angMul2=(*GPUNumAzimAng) * _numPol;
	_angMul1=(*GPUNumMOCGroups) * _angMul2;

	_fsrID = _tid/_angMul1;
	_groupID = _tid%_angMul1/_angMul2;
	_azimID = _tid%_angMul1%_angMul2/_numPol;
	_polID = _tid%_angMul1%_angMul2%_numPol + _polarBoundLowLim;

	while (_fsrID < (*GPUNumFSRsLayer)) {
		_GPUFSRAngScattSour[_tid] = 0.;
        _tid += blockDim.x * gridDim.x;
		_fsrID = _tid/_angMul1;
		_groupID = _tid%_angMul1/_angMul2;
		_azimID = _tid%_angMul1%_angMul2/_numPol;
		_polID = _tid%_angMul1%_angMul2%_numPol + _polarBoundLowLim;
	}
}


/* Experimental fused high-order scattering-source kernels used to compare
 * four thread decompositions while leaving the type-11 kernel unchanged.
 * The physical group direction is independent of TRANSPOSE_SCATT_MTX; this
 * helper maps (groupIn, groupOut) to the active CSR storage orientation. */
__device__ __forceinline__ FP_PRECISION GPUGetHOScattXS(
	FP_PRECISION *_GPUMacroXS_s,
	M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
	M_UINT64 *_GPUMacroXS_s_NZColStartID,
	size_t _fsrID, size_t _lID, size_t _groupInID, size_t _groupOutID)
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
	const size_t _rowID = _fsrID * (_numOrder + 1) * _numGrp + _lID * _numGrp + _rowGroupID;
	const M_UINT64 _rowStart = _GPUMacroXS_s_NZRowStartID_Layer[_rowID];
	const M_UINT64 _rowEnd = _GPUMacroXS_s_NZRowStartID_Layer[_rowID + 1];
	const M_UINT64 _colStart = _GPUMacroXS_s_NZColStartID[_rowID];

	if (_targetGroupID < (size_t) _colStart) {
		return 0.0;
	}
	const size_t _offset = _targetGroupID - (size_t) _colStart;
	if (_offset >= (size_t) (_rowEnd - _rowStart)) {
		return 0.0;
	}
	return _GPUMacroXS_s[_rowStart + _offset];
}

/* Same CSR lookup used by the per-azimuth source path.  The scan mode does
 * not decide whether the XS rows are clustered: this accessor receives that
 * choice explicitly and only changes the row-end lookup. */
__device__ __forceinline__ FP_PRECISION GPUGetHOScattXS_PerAzim(
	FP_PRECISION *_GPUMacroXS_s,
	M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
	M_UINT64 *_GPUMacroXS_s_NZRowEndID_Layer,
	M_UINT64 *_GPUMacroXS_s_NZColStartID,
	size_t _fsrID, size_t _lID, size_t _groupInID, size_t _groupOutID,
	int _useCluster)
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
	const size_t _rowID = _fsrID * (_numOrder + 1) * _numGrp +
		_lID * _numGrp + _rowGroupID;
	const M_UINT64 _rowStart = _GPUMacroXS_s_NZRowStartID_Layer[_rowID];
	const M_UINT64 _rowEnd = _useCluster
		? _GPUMacroXS_s_NZRowEndID_Layer[_rowID]
		: _GPUMacroXS_s_NZRowStartID_Layer[_rowID + 1];
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


__device__ __forceinline__ FP_PRECISION GPUProjectHOFluxForAngle(
	FP_PRECISION_FLUX *_GPUScalarFlux,
	FP_PRECISION *_GPUSpheHarmCoeff,
	size_t _fsrID, size_t _groupInID, int _lID, int _azimID, int _polID)
{
	const size_t _numGrp = *GPUNumMOCGroups;
	const size_t _numFsr = *GPUNumFSRsLayer;
	const size_t _numAng = (size_t) (*GPUNumAzimAng) * (size_t) (*GPUNumPolarAng);
	const size_t _angleID = (size_t) _azimID * (size_t) (*GPUNumPolarAng) + (size_t) _polID;
	const size_t _fluxGroupID = _fsrID * _numGrp + _groupInID;
	const int _lmStart = _lID * _lID;
	FP_PRECISION _projectedFlux = 0.0;

	for (int _mOffset = 0; _mOffset < 2 * _lID + 1; ++_mOffset) {
		const size_t _lmID = (size_t) (_lmStart + _mOffset);
		const size_t _fluxID = _lmID * _numFsr * _numGrp + _fluxGroupID;
		const size_t _ylmID = _lmID * _numAng + _angleID;
		_projectedFlux += GPUCoeffA_HOScatt[_lmID] *
			(FP_PRECISION) _GPUScalarFlux[_fluxID] * _GPUSpheHarmCoeff[_ylmID];
	}
	return _projectedFlux;
}


/* Type 15: one logical thread per (FSR, groupIn, azimuth, polar). */
__global__ void GPUComputeFSRScattSourceOnDevice_FsrGrpInAng(
	FP_PRECISION *_GPUFSRAngScattSour,
	FP_PRECISION_FLUX *_GPUScalarFlux,
	FP_PRECISION *_GPUMacroXS_s,
	M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
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

	for (size_t _tid = (size_t) threadIdx.x + (size_t) blockDim.x * blockIdx.x;
		 _tid < _total; _tid += (size_t) blockDim.x * gridDim.x) {
		const size_t _fsrID = _tid / (_numGrp * _numAng);
		const size_t _rem = _tid % (_numGrp * _numAng);
		const size_t _groupInID = _rem / _numAng;
		const size_t _angleID = _rem % _numAng;
		const int _azimID = (int) (_angleID / (size_t) _numPol);
		const int _localPolID = (int) (_angleID % (size_t) _numPol);
		const int _polID = _localPolID + _polarBoundLowLim;

		for (int _lID = 0; _lID <= *GPUNumScattOrder; ++_lID) {
			const FP_PRECISION _projectedFlux = GPUProjectHOFluxForAngle(
				_GPUScalarFlux, _GPUSpheHarmCoeff, _fsrID, _groupInID, _lID, _azimID, _polID);
#ifndef TRANSPOSE_SCATT_MTX
			const size_t _rowID = _fsrID * ((size_t) (*GPUNumScattOrder) + 1) * _numGrp +
				(size_t) _lID * _numGrp + _groupInID;
			const M_UINT64 _rowStart = _GPUMacroXS_s_NZRowStartID_Layer[_rowID];
			const M_UINT64 _rowEnd = _GPUMacroXS_s_NZRowStartID_Layer[_rowID + 1];
			size_t _groupOutID = (size_t) _GPUMacroXS_s_NZColStartID[_rowID];
			for (M_UINT64 _xsID = _rowStart; _xsID < _rowEnd; ++_xsID, ++_groupOutID) {
				const size_t _outID = _fsrID * _numGrp * _numAng + _groupOutID * _numAng + _angleID;
				const FP_PRECISION _source = (FP_PRECISION) (2 * _lID + 1) *
					_GPUMacroXS_s[_xsID] * _projectedFlux * ONE_OVER_FOUR_PI;
				atomicAdd(&_GPUFSRAngScattSour[_outID], _source);
			}
#else
			for (size_t _groupOutID = 0; _groupOutID < _numGrp; ++_groupOutID) {
				const FP_PRECISION _xs = GPUGetHOScattXS(_GPUMacroXS_s,
					_GPUMacroXS_s_NZRowStartID_Layer, _GPUMacroXS_s_NZColStartID,
					_fsrID, (size_t) _lID, _groupInID, _groupOutID);
				if (_xs != 0.0) {
					const size_t _outID = _fsrID * _numGrp * _numAng + _groupOutID * _numAng + _angleID;
					const FP_PRECISION _source = (FP_PRECISION) (2 * _lID + 1) *
						_xs * _projectedFlux * ONE_OVER_FOUR_PI;
					atomicAdd(&_GPUFSRAngScattSour[_outID], _source);
				}
			}
#endif
		}
	}
}


/* Type 16: one logical thread per (FSR, groupOut, azimuth, polar). */
__global__ void GPUComputeFSRScattSourceOnDevice_FsrGrpOutAng(
	FP_PRECISION *_GPUFSRAngScattSour,
	FP_PRECISION_FLUX *_GPUScalarFlux,
	FP_PRECISION *_GPUMacroXS_s,
	M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
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

	for (size_t _tid = (size_t) threadIdx.x + (size_t) blockDim.x * blockIdx.x;
		 _tid < _total; _tid += (size_t) blockDim.x * gridDim.x) {
		const size_t _fsrID = _tid / (_numGrp * _numAng);
		const size_t _rem = _tid % (_numGrp * _numAng);
		const size_t _groupOutID = _rem / _numAng;
		const size_t _angleID = _rem % _numAng;
		const int _azimID = (int) (_angleID / (size_t) _numPol);
		const int _localPolID = (int) (_angleID % (size_t) _numPol);
		const int _polID = _localPolID + _polarBoundLowLim;
		FP_PRECISION _scattSource = 0.0;

		for (int _lID = 0; _lID <= *GPUNumScattOrder; ++_lID) {
#ifdef TRANSPOSE_SCATT_MTX
			const size_t _rowID = _fsrID * ((size_t) (*GPUNumScattOrder) + 1) * _numGrp +
				(size_t) _lID * _numGrp + _groupOutID;
			const M_UINT64 _rowStart = _GPUMacroXS_s_NZRowStartID_Layer[_rowID];
			const M_UINT64 _rowEnd = _GPUMacroXS_s_NZRowStartID_Layer[_rowID + 1];
			size_t _groupInID = (size_t) _GPUMacroXS_s_NZColStartID[_rowID];
			for (M_UINT64 _xsID = _rowStart; _xsID < _rowEnd; ++_xsID, ++_groupInID) {
				const FP_PRECISION _projectedFlux = GPUProjectHOFluxForAngle(
					_GPUScalarFlux, _GPUSpheHarmCoeff, _fsrID, _groupInID, _lID, _azimID, _polID);
				_scattSource += (FP_PRECISION) (2 * _lID + 1) *
					_GPUMacroXS_s[_xsID] * _projectedFlux;
			}
#else
			for (size_t _groupInID = 0; _groupInID < _numGrp; ++_groupInID) {
				const FP_PRECISION _xs = GPUGetHOScattXS(_GPUMacroXS_s,
					_GPUMacroXS_s_NZRowStartID_Layer, _GPUMacroXS_s_NZColStartID,
					_fsrID, (size_t) _lID, _groupInID, _groupOutID);
				if (_xs != 0.0) {
					const FP_PRECISION _projectedFlux = GPUProjectHOFluxForAngle(
						_GPUScalarFlux, _GPUSpheHarmCoeff, _fsrID, _groupInID, _lID, _azimID, _polID);
					_scattSource += (FP_PRECISION) (2 * _lID + 1) * _xs * _projectedFlux;
				}
			}
#endif
		}
		_GPUFSRAngScattSour[_tid] = _scattSource * ONE_OVER_FOUR_PI;
	}
}



/**
 * LM stage, target-group-oriented (gather) variant.
 * Thread = (lmID, fsrID, grpID), grpID is the OUTGOING group.
 * For each incoming group grpIn, locate Sigma_s,l(grpIn -> grpID) by an O(1)
 * row-band check, and accumulate S_lm(grpID). Direct write, deterministic.
 * Note: _gOffset is clamped before the load to avoid an out-of-range read.
 */
__global__ void GPUComputeFSRScatterSourceLMOnDevice_grpOut(FP_PRECISION *_GPUFSRAngScatterSourceLM,
                                                            FP_PRECISION_FLUX *_GPUScalarFlux,
                                                            FP_PRECISION *_GPUMacroXS_s,
                                                            M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
                                                            M_UINT64 *_GPUMacroXS_s_NZColStartID) {
	size_t _numGrp = *GPUNumMOCGroups, _numFsr = *GPUNumFSRsLayer, _numOrder = *GPUNumScattOrder;
	size_t _lID, _grpInID, _ind11, _ind12, _ind21, _ind22, _ind31, _nnzInGroupIn;
	M_UINT64 _groupInNNZStartColID = 0, _groupInXsStartInd = 0;
	FP_PRECISION _flux = 0., _scaSource = 0., _scaXs = 0.;
	int _gOffset = 0, _flag;

	int L[25] = {0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4};

	size_t _tid = threadIdx.x + blockDim.x * blockIdx.x;
	size_t _lmID = _tid / (_numFsr * _numGrp);
	size_t _fsrID = _tid % (_numFsr * _numGrp) / _numGrp;
	size_t _grpID = _tid % (_numFsr * _numGrp) % _numGrp;

	while (_lmID < (_numOrder + 1) * (_numOrder + 1)) {
		_scaSource = 0.;
		_lID = L[_lmID];

		_ind11 = _lmID * _numFsr * _numGrp + _fsrID * _numGrp;          // _ind1x: flux相关索引
		_ind21 = _fsrID * (_numOrder + 1) * _numGrp + _lID * _numGrp;   // _ind2x: 散射截面相关索引
		_ind31 = _lmID * _numFsr * _numGrp + _fsrID * _numGrp + _grpID; // _ind3x: 散射源相关索引

		for (_grpInID = 0; _grpInID < _numGrp; ++_grpInID) {
			_ind12 = _ind11 + _grpInID;
			_flux = _GPUScalarFlux[_ind12];

			_ind22 = _ind21 + _grpInID;
			_groupInNNZStartColID = _GPUMacroXS_s_NZColStartID[_ind22];
			_groupInXsStartInd = _GPUMacroXS_s_NZRowStartID_Layer[_ind22];
			_nnzInGroupIn = _GPUMacroXS_s_NZRowStartID_Layer[_ind22 + 1] - _groupInXsStartInd;

			/* signed subtraction: _grpID and colStart both fit in int, so no wrap */
			_gOffset = (int) _grpID - (int) _groupInNNZStartColID;
			_flag = (_gOffset >= 0 && _gOffset < (int) _nnzInGroupIn) ? 1 : 0;
			if (_flag == 0) {
				_gOffset = 0;
			}

			_scaXs = _flag * _GPUMacroXS_s[_groupInXsStartInd + _gOffset];
			_scaSource += _scaXs * _flux * (2 * _lID + 1);
		}

		_GPUFSRAngScatterSourceLM[_ind31] = _scaSource;

		_tid += blockDim.x * gridDim.x;
		_lmID = _tid / (_numFsr * _numGrp);
		_fsrID = _tid % (_numFsr * _numGrp) / _numGrp;
		_grpID = _tid % (_numFsr * _numGrp) % _numGrp;
	}
}

/**
 * LM stage, source-group-oriented (scatter) variant.
 * Thread = (lmID, fsrID, grpInID), grpInID is the INCOMING group.
 * Iterate the row segment of the incoming group directly and atomically
 * accumulate S_lm(g2). Requires S_lm to be zeroed before launch.
 */
__global__ void GPUComputeFSRScatterSourceLMOnDevice_grpIn(FP_PRECISION *_GPUFSRAngScatterSourceLM,
                                                           FP_PRECISION_FLUX *_GPUScalarFlux,
                                                           FP_PRECISION *_GPUMacroXS_s,
                                                           M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
                                                           M_UINT64 *_GPUMacroXS_s_NZColStartID) {
	size_t _numGrp = *GPUNumMOCGroups, _numFsr = *GPUNumFSRsLayer, _numOrder = *GPUNumScattOrder;
	size_t _lmID, _lID, _grpInID, _grpOutID, _fsrID;
	size_t _ind12, _ind22, _ind31;
	M_UINT64 _grpOutStartColID = 0, _grpOutXsStartInd = 0, _nnz = 0;
	FP_PRECISION _flux = 0., _scaXs = 0.;

	int L[25] = {0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4};

	size_t _tid = threadIdx.x + blockDim.x * blockIdx.x;
	_lmID = _tid / (_numFsr * _numGrp);
	_fsrID = _tid % (_numFsr * _numGrp) / _numGrp;
	_grpInID = _tid % _numGrp;

	while (_lmID < (_numOrder + 1) * (_numOrder + 1)) {
		_lID = L[_lmID];

		_ind12 = _lmID * _numFsr * _numGrp + _fsrID * _numGrp + _grpInID;
		_flux = _GPUScalarFlux[_ind12];

		_ind22 = _fsrID * (_numOrder + 1) * _numGrp + _lID * _numGrp + _grpInID;
		_grpOutStartColID = _GPUMacroXS_s_NZColStartID[_ind22];
		_grpOutXsStartInd = _GPUMacroXS_s_NZRowStartID_Layer[_ind22];
		_nnz = _GPUMacroXS_s_NZRowStartID_Layer[_ind22 + 1] - _grpOutXsStartInd;

		_grpOutID = _grpOutStartColID;
		for (M_UINT64 _idx = 0; _idx < _nnz; ++_idx) {
			_scaXs = (2 * _lID + 1) * _flux * _GPUMacroXS_s[_grpOutXsStartInd + _idx];
			_ind31 = _lmID * _numFsr * _numGrp + _fsrID * _numGrp + _grpOutID;
			atomicAdd(&_GPUFSRAngScatterSourceLM[_ind31], _scaXs);
			++_grpOutID;
		}

		_tid += blockDim.x * gridDim.x;
		_lmID = _tid / (_numFsr * _numGrp);
		_fsrID = _tid % (_numFsr * _numGrp) / _numGrp;
		_grpInID = _tid % _numGrp;
	}
}

/**
 * AP stage, shared by both LM variants.
 * Thread = (fsrID, grpID); project the moment-resolved source S_lm onto all
 * discrete directions and accumulate into the angular scattering source.
 * The angular source must be zeroed before launch.
 */
__global__ void GPUComputeFSRScatterSourceAPOnDevice(FP_PRECISION *_GPUFSRAngScattSour,
                                                     FP_PRECISION *_GPUSpheHarmCoeff,
                                                     FP_PRECISION *_GPUFSRAngScatterSourceLM) {
    size_t _numGrp = *GPUNumMOCGroups, _numFsr = *GPUNumFSRsLayer, _numOrder = *GPUNumScattOrder;
    size_t _numAzim = *GPUNumAzimAng, _numPol = 0;

	#if defined(M_2DMOC)
		_numPol = *GPUNumPolarAng;
	#else
		_numPol = (*GPUNumPolarAng) / 2;
	#endif

    size_t _ind11, _ind12, _ind13, _ind14, _ind21, _ind22;
    int _ilInd, _imInd, _iaInd, _ipInd;
    FP_PRECISION _coeffALM = 0., _coefYLM = 0., _scatterSourceLM = 0., _scatterSourceAP = 0.;

    size_t _tid = threadIdx.x + blockDim.x * blockIdx.x;
    size_t _fsrID = _tid / _numGrp;
    size_t _grpID = _tid % _numGrp;

    while (_fsrID < _numFsr) {
        _ind21 = _fsrID * _numGrp * _numAzim * _numPol + _grpID * _numAzim * _numPol;

        for (_ilInd = 0; _ilInd <= (int) _numOrder; ++_ilInd) {
            _ind11 = _ilInd * _ilInd;
            for (_imInd = 0; _imInd < 2 * _ilInd + 1; ++_imInd) {
                _ind12 = _ind11 + _imInd;
                _coeffALM = GPUCoeffA_HOScatt[_ind12];

                _ind13 = (_ind11 + _imInd) * _numFsr * _numGrp + _fsrID * _numGrp + _grpID;
                _scatterSourceLM = _GPUFSRAngScatterSourceLM[_ind13];

                for (_iaInd = 0; _iaInd < (int) _numAzim; ++_iaInd) {
                    for (_ipInd = 0; _ipInd < (int) _numPol; ++_ipInd) {
                        _ind14 = (_ind11 + _imInd) * _numAzim * _numPol + _iaInd * _numPol + _ipInd;
                        _coefYLM = _GPUSpheHarmCoeff[_ind14];

                        _scatterSourceAP = ONE_OVER_FOUR_PI * _coeffALM * _coefYLM * _scatterSourceLM;
                        _ind22 = _ind21 + _iaInd * _numPol + _ipInd;
                        _GPUFSRAngScattSour[_ind22] += _scatterSourceAP;
                    }
                }
            }
        }

        _tid += blockDim.x * gridDim.x;
        _fsrID = _tid / _numGrp;
        _grpID = _tid % _numGrp;
    }
}

__global__ void GPUComputeFSRScatterSourceAPOnDeviceOpt1(FP_PRECISION *_GPUFSRAngScattSour,
                                                         FP_PRECISION *_GPUSpheHarmCoeff,
                                                         FP_PRECISION *_GPUFSRAngScatterSourceLM) {
	size_t _numGrp = *GPUNumMOCGroups, _numFsr = *GPUNumFSRsLayer, _numOrder = *GPUNumScattOrder;
	size_t _numAzim = *GPUNumAzimAng, _numPol = 0;

	#if defined(M_2DMOC)
		_numPol = *GPUNumPolarAng;
	#else
		_numPol = (*GPUNumPolarAng)/2;
	#endif

	int _ilInd, _imInd, _iaInd, _ipInd;

	size_t _tid = threadIdx.x + blockDim.x * blockIdx.x;

	size_t _fsrID = _tid / _numGrp;
	size_t _grpID = _tid % _numGrp;

	while (_fsrID < _numFsr) {

		for (_iaInd = 0; _iaInd < _numAzim; ++_iaInd) {
			for (_ipInd = 0; _ipInd < _numPol; ++_ipInd) {
				FP_PRECISION _acc = 0.f;  // ← 寄存器累加，零延迟

				for (_ilInd = 0; _ilInd <= _numOrder; ++_ilInd) {
					for (_imInd = 0; _imInd < 2 * _ilInd + 1; ++_imInd) {

						int _ind12 = _ilInd * _ilInd + _imInd;
						FP_PRECISION _coeffALM = GPUCoeffA_HOScatt[_ind12];  // constant cache

						int _ind13 = _ind12 * _numFsr * _numGrp + _fsrID * _numGrp + _grpID;
						FP_PRECISION _scatterSourceLM = _GPUFSRAngScatterSourceLM[_ind13];

						int _ind14 = _ind12 * _numAzim * _numPol + _iaInd * _numPol + _ipInd;
						FP_PRECISION _coefYLM = _GPUSpheHarmCoeff[_ind14];

						_acc += _coeffALM * _coefYLM * _scatterSourceLM;
					}
				}

				size_t _ind21 = _fsrID * _numGrp * _numAzim * _numPol + _grpID * _numAzim * _numPol + _iaInd * _numPol + _ipInd;
				_GPUFSRAngScattSour[_ind21] = ONE_OVER_FOUR_PI * _acc;
			}
		}

		_tid += blockDim.x * gridDim.x;
		_fsrID = _tid / _numGrp;
		_grpID = _tid % _numGrp;
	}
}

__global__ void GPUComputeFSRScatterSourceAPOnDeviceOpt2(FP_PRECISION *_GPUFSRAngScattSour,
                                                         FP_PRECISION *_GPUSpheHarmCoeff,
                                                         FP_PRECISION *_GPUFSRAngScatterSourceLM) {
	size_t _numGrp = *GPUNumMOCGroups, _numFsr = *GPUNumFSRsLayer, _numOrder = *GPUNumScattOrder;
	size_t _numAzim = *GPUNumAzimAng, _numPol = 0;

	#if defined(M_2DMOC)
		_numPol = *GPUNumPolarAng;
	#else
		_numPol = (*GPUNumPolarAng)/2;
	#endif

	size_t _numAng = _numAzim * _numPol, _totalPerFsr = _numGrp * _numAng;

	int _ilInd, _imInd;

	size_t _tid = threadIdx.x + blockDim.x * blockIdx.x;

	size_t _fsrID = _tid / _totalPerFsr;
	size_t _rem   = _tid % _totalPerFsr;
	size_t _grpID = _rem / _numAng;
	size_t _angID = _rem % _numAng;
	size_t _iaInd = _angID / _numPol;
	size_t _ipInd = _angID % _numPol;

	while ((_fsrID < _numFsr) && (_grpID < _numGrp) && (_iaInd < _numAzim) && (_ipInd < _numPol)) {

		FP_PRECISION _acc = 0.f;  // ← 寄存器累加，零延迟

		for (_ilInd = 0; _ilInd <= _numOrder; ++_ilInd) {
			for (_imInd = 0; _imInd < 2 * _ilInd + 1; ++_imInd) {

				int _ind12 = _ilInd * _ilInd + _imInd;
				FP_PRECISION _coeffALM = GPUCoeffA_HOScatt[_ind12];  // constant cache

				int _ind13 = _ind12 * _numFsr * _numGrp + _fsrID * _numGrp + _grpID;
				FP_PRECISION _scatterSourceLM = _GPUFSRAngScatterSourceLM[_ind13];

				int _ind14 = _ind12 * _numAzim * _numPol + _iaInd * _numPol + _ipInd;
				FP_PRECISION _coefYLM = _GPUSpheHarmCoeff[_ind14];

				_acc += _coeffALM * _coefYLM * _scatterSourceLM;
			}
		}

		size_t _ind21 = _fsrID * _numGrp * _numAzim * _numPol + _grpID * _numAzim * _numPol + _iaInd * _numPol + _ipInd;
		_GPUFSRAngScattSour[_ind21] = ONE_OVER_FOUR_PI * _acc;

		_tid += blockDim.x * gridDim.x;
		_fsrID = _tid / _totalPerFsr;
		_rem   = _tid % _totalPerFsr;
		_grpID = _rem / _numAng;
		_angID = _rem % _numAng;
		_iaInd = _angID / _numPol;
		_ipInd = _angID % _numPol;
	}
}

__global__ void GPUComputeFSRScatterSourceAPOnDeviceOpt3(FP_PRECISION *_GPUFSRAngScattSour,
                                                         FP_PRECISION *_GPUSpheHarmCoeff,
                                                         FP_PRECISION *_GPUFSRAngScatterSourceLM) {
	unsigned int _numGrp = *GPUNumMOCGroups, _numFsr = *GPUNumFSRsLayer, _numOrder = *GPUNumScattOrder;
	unsigned int _numAzim = *GPUNumAzimAng, _numPol = 0;

	#if defined(M_2DMOC)
		_numPol = *GPUNumPolarAng;
	#else
		_numPol = (*GPUNumPolarAng)/2;
	#endif

	unsigned int _numAng = _numAzim * _numPol, _totalPerFsr = _numGrp * _numAzim * _numPol;

	int _ilInd, _imInd;

	unsigned int _tid = threadIdx.x + blockDim.x * blockIdx.x;

	unsigned int _fsrID = _tid / _totalPerFsr;
	unsigned int _rem   = _tid % _totalPerFsr;
	unsigned int _grpID = _rem / _numAng;
	unsigned int _angID = _rem % _numAng;
	unsigned int _iaInd = _angID / _numPol;
	unsigned int _ipInd = _angID % _numPol;

	while (_fsrID < _numFsr) {

		FP_PRECISION _acc = 0.f;  // ← 寄存器累加，零延迟
		unsigned int _ind21 = _fsrID * _numGrp * _numAzim * _numPol + _grpID * _numAzim * _numPol + _iaInd * _numPol + _ipInd;

		for (_ilInd = 0; _ilInd <= _numOrder; ++_ilInd) {
			for (_imInd = 0; _imInd < 2 * _ilInd + 1; ++_imInd) {

				int _ind12 = _ilInd * _ilInd + _imInd;
				FP_PRECISION _coeffALM = GPUCoeffA_HOScatt[_ind12];  // constant cache

				int _ind13 = _ind12 * _numFsr * _numGrp + _fsrID * _numGrp + _grpID;
				FP_PRECISION _scatterSourceLM = _GPUFSRAngScatterSourceLM[_ind13];

				int _ind14 = _ind12 * _numAzim * _numPol + _iaInd * _numPol + _ipInd;
				FP_PRECISION _coefYLM = _GPUSpheHarmCoeff[_ind14];

				_acc += _coeffALM * _coefYLM * _scatterSourceLM;
			}
		}

		_GPUFSRAngScattSour[_ind21] = ONE_OVER_FOUR_PI * _acc;

		_tid += blockDim.x * gridDim.x;
		_fsrID = _tid / _totalPerFsr;
		_rem   = _tid % _totalPerFsr;
		_grpID = _rem / _numAng;
		_angID = _rem % _numAng;
		_iaInd = _angID / _numPol;
		_ipInd = _angID % _numPol;
	}
}

/* AP projection for one azimuth batch.  The source buffer is laid out as
 * FSR x group x batch-azimuth x polar; the harmonic table keeps its original
 * full azimuth stride so that batching does not alter the physics. */
__global__ void GPUComputeFSRScatterSourceAPOnDevice_PerAzim(
	FP_PRECISION *_GPUFSRAngScattSour,
	FP_PRECISION *_GPUSpheHarmCoeff,
	FP_PRECISION *_GPUFSRAngScatterSourceLM,
	int _polarBoundLowLim, int _azimBegin, int _azimEnd, int _apKernel)
{
	const size_t _numGrp = *GPUNumMOCGroups;
	const size_t _numFsr = *GPUNumFSRsLayer;
	const size_t _numOrder = *GPUNumScattOrder;
	const size_t _numPol = *GPUNumPolarAng;
	const size_t _fullNumAng = (size_t) (*GPUNumAzimAng) * _numPol;
	const size_t _numLM = (_numOrder + 1) * (_numOrder + 1);
	const size_t _numBatchAzim = (size_t) (_azimEnd - _azimBegin);
	const size_t _batchAng = _numBatchAzim * _numPol;
	const size_t _tid = (size_t) threadIdx.x + (size_t) blockIdx.x * blockDim.x;

	if (_apKernel >= 2) {
		const size_t _total = _numFsr * _numGrp * _batchAng;
		for (size_t _work = _tid; _work < _total; _work +=
			(size_t) blockDim.x * gridDim.x) {
			const size_t _fsrID = _work / (_numGrp * _batchAng);
			const size_t _rem = _work % (_numGrp * _batchAng);
			const size_t _groupID = _rem / _batchAng;
			const size_t _localAzimID = (_rem % _batchAng) / _numPol;
			const size_t _localPolID = _rem % _numPol;
			const size_t _azimID = (size_t) _azimBegin + _localAzimID;
			const size_t _harmonicAngleID = _azimID * _numPol +
				_localPolID + (size_t) _polarBoundLowLim;
			FP_PRECISION _acc = 0.0;
			for (size_t _lmID = 0; _lmID < _numLM; ++_lmID) {
				_acc += GPUCoeffA_HOScatt[_lmID] *
					_GPUSpheHarmCoeff[_lmID * _fullNumAng + _harmonicAngleID] *
					_GPUFSRAngScatterSourceLM[_lmID * _numFsr * _numGrp +
						_fsrID * _numGrp + _groupID];
			}
			_GPUFSRAngScattSour[_fsrID * _numGrp * _batchAng +
				_groupID * _batchAng + _localAzimID * _numPol + _localPolID] =
				_acc * ONE_OVER_FOUR_PI;
		}
	}
	else {
		const size_t _total = _numFsr * _numGrp * _numBatchAzim;
		for (size_t _work = _tid; _work < _total; _work +=
			(size_t) blockDim.x * gridDim.x) {
			const size_t _fsrID = _work / (_numGrp * _numBatchAzim);
			const size_t _rem = _work % (_numGrp * _numBatchAzim);
			const size_t _groupID = _rem / _numBatchAzim;
			const size_t _localAzimID = _rem % _numBatchAzim;
			const size_t _azimID = (size_t) _azimBegin + _localAzimID;
			for (size_t _localPolID = 0; _localPolID < _numPol; ++_localPolID) {
				const size_t _harmonicAngleID = _azimID * _numPol +
					_localPolID + (size_t) _polarBoundLowLim;
				FP_PRECISION _acc = 0.0;
				for (size_t _lmID = 0; _lmID < _numLM; ++_lmID) {
					_acc += GPUCoeffA_HOScatt[_lmID] *
						_GPUSpheHarmCoeff[_lmID * _fullNumAng + _harmonicAngleID] *
						_GPUFSRAngScatterSourceLM[_lmID * _numFsr * _numGrp +
							_fsrID * _numGrp + _groupID];
				}
				_GPUFSRAngScattSour[_fsrID * _numGrp * _batchAng +
					_groupID * _batchAng + _localAzimID * _numPol + _localPolID] =
					_acc * ONE_OVER_FOUR_PI;
			}
		}
	}
}

/* Direct one-azimuth source evaluation used by types 15-21.  Modes 0/1
 * produce each outgoing group directly (angle-oriented and group-oriented
 * launch shapes respectively); mode 2 keeps the source-group atomic form
 * used by Type 20.  The cluster flag only changes the CSR accessor. */
/*
 * Correct compact-output kernels for the 2D per-azimuth path.  The previous
 * generic kernel launched one thread per (FSR, group), but modes 0/1 made
 * every such thread loop over both group dimensions.  These kernels retain
 * the four original thread organizations while fixing the selected group
 * and azimuth mapping:
 *   - InAng:  (FSR, groupIn, polar), atomic accumulation to groupOut;
 *   - OutAng: (FSR, groupOut, polar), direct write;
 *   - In:     (FSR, groupIn), polar remains inside the thread;
 *   - Out:    (FSR, groupOut), polar remains inside the thread.
 *
 * The output layout is compact for one azimuth:
 *   FSR x groupOut x polar.
 */
__device__ __forceinline__ void GPUAddPerAzimSource(
	FP_PRECISION *_GPUFSRAngScattSour, size_t _fsrID, size_t _groupOutID,
	size_t _localPolID, size_t _numGrp, size_t _numPol, FP_PRECISION _source)
{
	const size_t _outID = _fsrID * _numGrp * _numPol +
		_groupOutID * _numPol + _localPolID;
#ifdef FP32
	atomicAdd(&_GPUFSRAngScattSour[_outID], _source);
#else
	atomicAdd_(&_GPUFSRAngScattSour[_outID], _source);
#endif
}

__global__ void GPUComputeFSRScattSourceOnDevice_FsrGrpInAng_PerAzim(
	FP_PRECISION *_GPUFSRAngScattSour, FP_PRECISION_FLUX *_GPUScalarFlux,
	FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
	M_UINT64 *_GPUMacroXS_s_NZRowEndID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID,
	FP_PRECISION *_GPUSpheHarmCoeff, int _polarBoundLowLim, int _azimID,
	int _useCluster)
{
	const size_t _numFsr = *GPUNumFSRsLayer;
	const size_t _numGrp = *GPUNumMOCGroups;
	const size_t _numOrder = *GPUNumScattOrder;
	const size_t _numPol = *GPUNumPolarAng;
	const size_t _total = _numFsr * _numGrp * _numPol;

	for (size_t _tid = (size_t) threadIdx.x + (size_t) blockIdx.x * blockDim.x;
		 _tid < _total; _tid += (size_t) blockDim.x * gridDim.x) {
		const size_t _fsrID = _tid / (_numGrp * _numPol);
		const size_t _rem = _tid % (_numGrp * _numPol);
		const size_t _groupInID = _rem / _numPol;
		const size_t _localPolID = _rem % _numPol;
		const int _polID = (int) _localPolID + _polarBoundLowLim;

		for (size_t _lID = 0; _lID <= _numOrder; ++_lID) {
			const FP_PRECISION _projected = GPUProjectHOFluxForAngle(
				_GPUScalarFlux, _GPUSpheHarmCoeff, _fsrID, _groupInID,
				(int) _lID, _azimID, _polID);

#ifndef TRANSPOSE_SCATT_MTX
			const size_t _rowID = _fsrID * (_numOrder + 1) * _numGrp +
				_lID * _numGrp + _groupInID;
			const M_UINT64 _rowStart = _GPUMacroXS_s_NZRowStartID_Layer[_rowID];
			const M_UINT64 _rowEnd = _useCluster
				? _GPUMacroXS_s_NZRowEndID_Layer[_rowID]
				: _GPUMacroXS_s_NZRowStartID_Layer[_rowID + 1];
			size_t _groupOutID = (size_t) _GPUMacroXS_s_NZColStartID[_rowID];
			for (M_UINT64 _xsID = _rowStart; _xsID < _rowEnd;
				 ++_xsID, ++_groupOutID) {
				const FP_PRECISION _source = (FP_PRECISION) (2 * _lID + 1) *
					_GPUMacroXS_s[_xsID] * _projected * ONE_OVER_FOUR_PI;
				GPUAddPerAzimSource(_GPUFSRAngScattSour, _fsrID, _groupOutID,
					_localPolID, _numGrp, _numPol, _source);
			}
#else
			for (size_t _groupOutID = 0; _groupOutID < _numGrp; ++_groupOutID) {
				const FP_PRECISION _xs = GPUGetHOScattXS_PerAzim(
					_GPUMacroXS_s, _GPUMacroXS_s_NZRowStartID_Layer,
					_GPUMacroXS_s_NZRowEndID_Layer, _GPUMacroXS_s_NZColStartID,
					_fsrID, _lID, _groupInID, _groupOutID, _useCluster);
				if (_xs != 0.0) {
					const FP_PRECISION _source = (FP_PRECISION) (2 * _lID + 1) *
						_xs * _projected * ONE_OVER_FOUR_PI;
					GPUAddPerAzimSource(_GPUFSRAngScattSour, _fsrID, _groupOutID,
						_localPolID, _numGrp, _numPol, _source);
				}
			}
#endif
		}
	}
}

__global__ void GPUComputeFSRScattSourceOnDevice_FsrGrpOutAng_PerAzim(
	FP_PRECISION *_GPUFSRAngScattSour, FP_PRECISION_FLUX *_GPUScalarFlux,
	FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
	M_UINT64 *_GPUMacroXS_s_NZRowEndID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID,
	FP_PRECISION *_GPUSpheHarmCoeff, int _polarBoundLowLim, int _azimID,
	int _useCluster)
{
	const size_t _numFsr = *GPUNumFSRsLayer;
	const size_t _numGrp = *GPUNumMOCGroups;
	const size_t _numOrder = *GPUNumScattOrder;
	const size_t _numPol = *GPUNumPolarAng;
	const size_t _total = _numFsr * _numGrp * _numPol;

	for (size_t _tid = (size_t) threadIdx.x + (size_t) blockIdx.x * blockDim.x;
		 _tid < _total; _tid += (size_t) blockDim.x * gridDim.x) {
		const size_t _fsrID = _tid / (_numGrp * _numPol);
		const size_t _rem = _tid % (_numGrp * _numPol);
		const size_t _groupOutID = _rem / _numPol;
		const size_t _localPolID = _rem % _numPol;
		const int _polID = (int) _localPolID + _polarBoundLowLim;
		FP_PRECISION _source = 0.0;

		for (size_t _lID = 0; _lID <= _numOrder; ++_lID) {
#ifdef TRANSPOSE_SCATT_MTX
			const size_t _rowID = _fsrID * (_numOrder + 1) * _numGrp +
				_lID * _numGrp + _groupOutID;
			const M_UINT64 _rowStart = _GPUMacroXS_s_NZRowStartID_Layer[_rowID];
			const M_UINT64 _rowEnd = _useCluster
				? _GPUMacroXS_s_NZRowEndID_Layer[_rowID]
				: _GPUMacroXS_s_NZRowStartID_Layer[_rowID + 1];
			size_t _groupInID = (size_t) _GPUMacroXS_s_NZColStartID[_rowID];
			for (M_UINT64 _xsID = _rowStart; _xsID < _rowEnd;
				 ++_xsID, ++_groupInID) {
				const FP_PRECISION _projected = GPUProjectHOFluxForAngle(
					_GPUScalarFlux, _GPUSpheHarmCoeff, _fsrID, _groupInID,
					(int) _lID, _azimID, _polID);
				_source += (FP_PRECISION) (2 * _lID + 1) *
					_GPUMacroXS_s[_xsID] * _projected;
			}
#else
			for (size_t _groupInID = 0; _groupInID < _numGrp; ++_groupInID) {
				const FP_PRECISION _xs = GPUGetHOScattXS_PerAzim(
					_GPUMacroXS_s, _GPUMacroXS_s_NZRowStartID_Layer,
					_GPUMacroXS_s_NZRowEndID_Layer, _GPUMacroXS_s_NZColStartID,
					_fsrID, _lID, _groupInID, _groupOutID, _useCluster);
				if (_xs != 0.0) {
					const FP_PRECISION _projected = GPUProjectHOFluxForAngle(
						_GPUScalarFlux, _GPUSpheHarmCoeff, _fsrID, _groupInID,
						(int) _lID, _azimID, _polID);
					_source += (FP_PRECISION) (2 * _lID + 1) * _xs * _projected;
				}
			}
#endif
		}

		const size_t _outID = _fsrID * _numGrp * _numPol +
			_groupOutID * _numPol + _localPolID;
		_GPUFSRAngScattSour[_outID] = _source * ONE_OVER_FOUR_PI;
	}
}


/**
 * @ Function: atomicAdd_()
 *   introduction: add a given value onto a parameter stored in a specific memory
 *                 this function is called by device function (__global__ function) and excuted on device
 * @ Parameter List: address - memory address where the value will be write in
 *                   val - value needs to be stored
 *   Return: void
 * @ Created     : 2018/05/25, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
 __device__ FP_PRECISION atomicAdd_(FP_PRECISION* _address, FP_PRECISION _val)
 {
     unsigned long long int* address_as_ull = (unsigned long long int*)_address;
     unsigned long long int old = *address_as_ull, assumed;
     do {
       assumed = old;
       old = atomicCAS(address_as_ull, assumed, __double_as_longlong(_val +
                       __longlong_as_double(assumed)));
     } while (assumed != old);
   
     return __longlong_as_double(old);
 }

 __device__ void GPUUpdateBoundaryFlux_3polar_1(int _azim, int _azimNext, int *_GPUNumLTsSum, int _ltIDNext, int _polEngID, \
	int _ltGlobID, FP_PRECISION track_flux_0, FP_PRECISION track_flux_1, FP_PRECISION track_flux_2, FP_PRECISION *_GPUBoundaryFlux, \
	FP_PRECISION *_GPUMPIFlux, FP_PRECISION _beta)
{
	if (_azim != _azimNext) {
		_GPUBoundaryFlux(_GPUNumLTsSum[_azimNext] + _ltIDNext, _polEngID) = _beta * track_flux_0;
		_GPUBoundaryFlux(_GPUNumLTsSum[_azimNext] + _ltIDNext, 1 + _polEngID) = _beta * track_flux_1;
		_GPUBoundaryFlux(_GPUNumLTsSum[_azimNext] + _ltIDNext, 2 + _polEngID) = _beta * track_flux_2;
	}
	else {
		_GPUMPIFlux(_ltGlobID, _polEngID) = track_flux_0;
		_GPUMPIFlux(_ltGlobID, 1 + _polEngID) = track_flux_1;
		_GPUMPIFlux(_ltGlobID, 2 + _polEngID) = track_flux_2;
	}
}

__device__ void GPUUpdateBoundaryFlux_3polar(int _azim, int _azimNext, int *_GPUNumLTsSum, int _ltIDNext, int _polEngID, \
	int _ltGlobID, FP_PRECISION track_flux_0, FP_PRECISION track_flux_1, FP_PRECISION track_flux_2, FP_PRECISION *_GPUStartFlux, \
	FP_PRECISION *_GPUMPIFlux, FP_PRECISION _beta)
{
	if (_azim != _azimNext) {
		_GPUStartFlux(_GPUNumLTsSum[_azimNext] + _ltIDNext, _polEngID) = _beta * track_flux_0;
		_GPUStartFlux(_GPUNumLTsSum[_azimNext] + _ltIDNext, 1 + _polEngID) = _beta * track_flux_1;
		_GPUStartFlux(_GPUNumLTsSum[_azimNext] + _ltIDNext, 2 + _polEngID) = _beta * track_flux_2;
	}
	else {
		_GPUMPIFlux(_ltGlobID, _polEngID) = track_flux_0;
		_GPUMPIFlux(_ltGlobID, 1 + _polEngID) = track_flux_1;
		_GPUMPIFlux(_ltGlobID, 2 + _polEngID) = track_flux_2;
	}
}

__device__ void GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme(int _ltInd_lay, int _azim, int _azimNext, int *_GPUNumLTsSum, int _ltIDNext, int _polEngID, \
	int _ltGlobID, FP_PRECISION *track_flux, FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUMPIFlux, FP_PRECISION _beta,\
	int _polarBoundLowLim,int _polarBoundUpLim, int polOffset)
{
	int/* _beta,*/ _pol;
	if (_azim != _azimNext) {
		for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
            _GPUBoundaryFlux(/*_ltInd_lay+*/_GPUNumLTsSum[_azimNext] + _ltIDNext, _pol + _polEngID) = _beta * track_flux[_pol-polOffset];
		}
	}
	else {
		for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
			_GPUMPIFlux(/*_ltInd_lay+*/_ltGlobID, _pol + _polEngID) = track_flux[_pol-polOffset];
		}
	}
}

__device__ void GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme_1(int _ltInd_lay, int _azim, int _azimNext, int _pol, int *_GPUNumLTsSum, int _ltIDNext, int _polEngID, \
	int _ltGlobID, FP_PRECISION track_flux, FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUMPIFlux, FP_PRECISION _beta,\
	int _polarBoundLowLim,int _polarBoundUpLim, int polOffset)
{
    if (_azim != _azimNext) {
        _GPUBoundaryFlux(/*_ltInd_lay+*/_GPUNumLTsSum[_azimNext] + _ltIDNext, _pol + _polEngID) = _beta * track_flux;
    }
    else {
        _GPUMPIFlux(/*_ltInd_lay+*/_ltGlobID, _pol + _polEngID) = track_flux;
    }
}


/**
* @ Function: GPUTallyScalarFlux_DDExpInte_OneDir_AngFSmem()
*   introduction: transport sweep solver on GPU for the polar loop of a segment, in GPUTS_Group_DDExpInte_OneDir_AngFSmem kernel
* @ Parameter List:
*   Return: void
* @ Created     : 2019/12/26, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
__device__ void GPUTallyScalarFlux_DDExpInte_OneDir_AngFSmem(int _index, int _indexLay, int _azim, FP_PRECISION _length, FP_PRECISION _oneOverVolume, \
	FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *_track_flux, FP_PRECISION *_GPUSource, FP_PRECISION_FLUX *_GPUScalarFlux, \
	FP_PRECISION *_GPULinearExpTable, int _polarBoundLowLim,int _polarBoundUpLim, int _polOffset, FP_PRECISION _twoOverWidth,\
	FP_PRECISION _oneOverVolbySpacing,M_SIZET _angId,FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update)
{
	int _x1, _pol;
	FP_PRECISION _xs_t, _sigmaTByLength, _fsrFlux,_delta, _mul2,_flatSour,_x2,_x3, _fluxTemp=0.0, _exp,_sourTerm;
#if defined(M_3DMOCex)
	FP_PRECISION _3d_sigma, _3d_oneOverSigma, _3d_Q, _3d_QtoSigma,_scale;
#endif // defined(M_3DMOCex)
	_xs_t = _GPUMacroXS_t[_indexLay];
	_flatSour = _GPUSource[_indexLay];

#if defined(M_2DMOC)
	_sigmaTByLength = _xs_t * _length;
	_sourTerm = _flatSour;
	_mul2 = (FP_PRECISION)(2.0) * _sourTerm * _sigmaTByLength;
	_fsrFlux = (FP_PRECISION)(0.0);
#endif // defined(M_2DMOC)

	for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
#if defined(M_3DMOCex)
		_scale=_twoOverWidth*GPUCosThetas[_pol];
		_3d_sigma=(_xs_t+_scale);
		_3d_oneOverSigma=1.0/_3d_sigma;
		_3d_Q=(_flatSour+_scale*_axialSurfAnfFlux_inp[_angId + _pol - _polOffset]);
		_3d_QtoSigma=_3d_Q*_3d_oneOverSigma;
		_sourTerm = _3d_QtoSigma;
		_sigmaTByLength = _3d_sigma * _length;
		_mul2 = (FP_PRECISION)(2.0) * _3d_Q * _length;
#endif // defined(M_3DMOCex)
		_fluxTemp = ((GPUSinThetasTimes2[_pol]-_sigmaTByLength)*_track_flux[_pol-_polOffset]+_mul2)/(GPUSinThetasTimes2[_pol]+_sigmaTByLength);
		if (_fluxTemp < 0.0) {
			_x2 = _sigmaTByLength * GPUSinThetaForExp[_pol];
			_x1 = (int)(_x2);
			_x3 = _x2 - _x1;
			_exp = _GPULinearExpTable[_x1] + _x3 * (_GPULinearExpTable[_x1 + 1] - _GPULinearExpTable[_x1]);
			// _exp=1-expf(-_sigmaTByLength/GPUSinThetas[_pol]);
			_delta = (_track_flux[_pol - _polOffset] - _sourTerm)*_exp;
			_track_flux[_pol - _polOffset] -= _delta;
		}
		else {
			_delta = _track_flux[_pol-_polOffset] - _fluxTemp;
			_track_flux[_pol - _polOffset] = _fluxTemp;
		}

#if defined(M_2DMOC)
		_fsrFlux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + _pol];
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
		_fsrFlux = _oneOverVolbySpacing*(_3d_QtoSigma*_length+GPUSinThetas[_pol]*_3d_oneOverSigma*_delta);
#ifdef DOUBLE_FLUX
    	atomicAdd_(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
    	atomicAdd(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
#endif
#endif // defined(M_3DMOCex)
	}

#if defined(M_2DMOC)
	_fsrFlux *= _oneOverVolume;
	/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
#endif
#endif // defined(M_2DMOC)
}

__device__ void GPUTallyScalarFlux_DDExpFunc_OneDir_AngFSmem(int _index, int _indexLay, int _azim, FP_PRECISION _length, FP_PRECISION _oneOverVolume, \
	FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *_track_flux, FP_PRECISION *_GPUSource, FP_PRECISION_FLUX *_GPUScalarFlux, \
	int _polarBoundLowLim,int _polarBoundUpLim, int _polOffset, FP_PRECISION _twoOverWidth,\
	FP_PRECISION _oneOverVolbySpacing, M_SIZET _angId,FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update)
{
	int _pol;
	FP_PRECISION _xs_t, _sigmaTByLength, _fsrFlux,_delta, _mul2,_flatSour, _fluxTemp=0.0, _exp,_sourTerm;
#if defined(M_3DMOCex)
	FP_PRECISION _3d_sigma, _3d_oneOverSigma, _3d_Q, _3d_QtoSigma,_scale;
#endif // defined(M_3DMOCex)
	_xs_t = _GPUMacroXS_t[_indexLay];
	_flatSour = _GPUSource[_indexLay];

#if defined(M_2DMOC)
	_sigmaTByLength = _xs_t * _length;
	_sourTerm = _flatSour;
	_mul2 = (FP_PRECISION)(2.0) * _sourTerm * _sigmaTByLength;
	_fsrFlux = (FP_PRECISION)(0.0);
#endif // defined(M_2DMOC)

	for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
#if defined(M_3DMOCex)
		_scale=_twoOverWidth*GPUCosThetas[_pol];
		_3d_sigma=(_xs_t+_scale);
		_3d_oneOverSigma=1.0/_3d_sigma;
		_3d_Q=(_flatSour+_scale*_axialSurfAnfFlux_inp[_angId + _pol - _polOffset]);
		_3d_QtoSigma=_3d_Q*_3d_oneOverSigma;
		_sourTerm = _3d_QtoSigma;
		_sigmaTByLength = _3d_sigma * _length;
		_mul2 = (FP_PRECISION)(2.0) * _3d_Q * _length;
#endif // defined(M_3DMOCex)
		_fluxTemp = ((GPUSinThetasTimes2[_pol]-_sigmaTByLength)*_track_flux[_pol-_polOffset]+_mul2)/(GPUSinThetasTimes2[_pol]+_sigmaTByLength);
		if (_fluxTemp < 0.0) {

			// _exp=1-expf(-_sigmaTByLength/GPUSinThetas[_pol]);
			// _exp=1.-expf(-_sigmaTByLength * GPUOneOverSinThetas[_pol]);
			_exp=1.-_ExpFunction_GPU(-_sigmaTByLength * GPUOneOverSinThetas[_pol]);
			// _exp=1.-expf(-_sigmaTByLength * GPUOneOverSinThetas[_pol]);

			_delta = (_track_flux[_pol - _polOffset] - _sourTerm)*_exp;
			_track_flux[_pol - _polOffset] -= _delta;
		}
		else {
			_delta = _track_flux[_pol-_polOffset] - _fluxTemp;
			_track_flux[_pol - _polOffset] = _fluxTemp;
		}

#if defined(M_2DMOC)
		_fsrFlux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + _pol];
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
		_fsrFlux = _oneOverVolbySpacing*(_3d_QtoSigma*_length+GPUSinThetas[_pol]*_3d_oneOverSigma*_delta);
#ifdef DOUBLE_FLUX
    	atomicAdd_(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
    	atomicAdd(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
#endif
#endif // defined(M_3DMOCex)
	}

#if defined(M_2DMOC)
	_fsrFlux *= _oneOverVolume;
	/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
#endif
#endif // defined(M_2DMOC)
}


/**
* @ Function: GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem()
*   introduction: transport sweep solver on GPU for the polar loop of a segment, in GPUTS_Group_ExpInte_OneDir_AngFSmem kernel
* @ Parameter List:
*   Return: void
* @ Created     : 2019/12/27, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
__device__ void GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem(int _index, int _indexLay, int _azim, FP_PRECISION _length, FP_PRECISION _oneOverVolume, \
	FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *_track_flux, FP_PRECISION *_GPUSource, FP_PRECISION_FLUX *_GPUScalarFlux, \
	FP_PRECISION *_GPULinearExpTable, int _polarBoundLowLim,int _polarBoundUpLim, int _polOffset, FP_PRECISION _twoOverWidth,\
	FP_PRECISION _oneOverVolbySpacing,M_SIZET _angId,FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update)
{
	int _x1, _pol;
	FP_PRECISION _xs_t, _sigmaTByLength, _fsrFlux,_delta,_flatSour,_x2,_x3, _exp,_sourTerm;
#if defined(M_3DMOCex)
	FP_PRECISION _3d_sigma, _3d_oneOverSigma, _3d_Q,_scale;
#endif // defined(M_3DMOCex)
	_xs_t = _GPUMacroXS_t[_indexLay];
	_flatSour = _GPUSource[_indexLay];

#if defined(M_2DMOC)
	_sigmaTByLength = _xs_t * _length;
	_sourTerm = _flatSour;
	
	_fsrFlux = (FP_PRECISION)(0.0);
#endif // defined(M_2DMOC)

	for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {

#if defined(M_3DMOCex)
		_scale=_twoOverWidth*GPUCosThetas[_pol];
		_3d_sigma=(_xs_t+_scale);
		_3d_oneOverSigma=1.0/_3d_sigma;
		_3d_Q=(_flatSour+_scale*_axialSurfAnfFlux_inp[_angId + _pol - _polOffset]);
		_sourTerm=_3d_Q*_3d_oneOverSigma;
		_sigmaTByLength = _3d_sigma * _length;
#endif // defined(M_3DMOCex)
		_x2 = _sigmaTByLength * GPUSinThetaForExp[_pol];
		_x1 = (int)(_x2);
		_x3 = _x2 - _x1;
		
		_exp = _GPULinearExpTable[_x1] + _x3 * (_GPULinearExpTable[_x1 + 1] - _GPULinearExpTable[_x1]);
		// _exp=1.-expf(-_sigmaTByLength/GPUSinThetas[_pol]);

        _delta = (_track_flux[_pol - _polOffset] - _sourTerm)*_exp;
		_track_flux[_pol - _polOffset] -= _delta;
		// if (_track_flux[_pol - _polOffset]<0.0) {
		// 	_track_flux[_pol - _polOffset] = 0.0;
		// }

#if defined(M_2DMOC)
		_fsrFlux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + _pol];
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
		_fsrFlux = _oneOverVolbySpacing*(_sourTerm*_length+GPUSinThetas[_pol]*_3d_oneOverSigma*_delta);
#ifdef DOUBLE_FLUX
    	atomicAdd_(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
    	atomicAdd(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
#endif
#endif // defined(M_3DMOCex)
	}

#if defined(M_2DMOC)
	// _fsrFlux *= 2.0 * _oneOverVolume;
	_fsrFlux *= /*2.0 **/ _oneOverVolume;
	/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
#endif
#endif // defined(M_2DMOC)
}

__device__ void GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_HOScatt(int _indexLay, int _azim, FP_PRECISION _length, \
	FP_PRECISION _oneOverVolume, FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *_track_flux, FP_PRECISION *_GPUSource, \
	FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPULinearExpTable, int _polarBoundLowLim,int _polarBoundUpLim, \
	int _polOffset, FP_PRECISION _twoOverWidth, FP_PRECISION _oneOverVolbySpacing,M_SIZET _angId,FP_PRECISION *_axialSurfAnfFlux_inp, \
	FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer, \
	M_UINT64 *_GPUMacroXS_s_NZColStartID, FP_PRECISION *_GPUSpheHarmCoeff, int _fsrID, int _groupID)
{
	int _x1, _pol;
	FP_PRECISION _xs_t, _sigmaTByLength, _fsrFlux,_delta,_flatSour,_x2,_x3, _exp,_sourTerm,_scattSour=0.0,_3d_oneOverSigma;
#if defined(M_3DMOCex)
	FP_PRECISION _3d_sigma, _3d_Q,_scale;
#endif // defined(M_3DMOCex)
	_xs_t = _GPUMacroXS_t[_indexLay];
	_flatSour = _GPUSource[_indexLay];

#if defined(M_2DMOC)
	_sigmaTByLength = _xs_t * _length;
	_3d_oneOverSigma = 1.0 / _xs_t;
	// _sourTerm = _flatSour;
	
	_fsrFlux = (FP_PRECISION)(0.0);
#endif // defined(M_2DMOC)

	for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {

		_scattSour = GPUComputeFSRScattSourceOnDevice_paraFsr_paraGroup_Layer(_GPUScalarFlux, \
			_GPUMacroXS_s, _GPUMacroXS_s_NZRowStartID_Layer, _GPUMacroXS_s_NZColStartID,\
			_GPUSpheHarmCoeff, _fsrID, _groupID, _azim, _pol);

#if defined(M_2DMOC)
	    _sourTerm = (_flatSour + _scattSour) * _3d_oneOverSigma;
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
		_scale=_twoOverWidth*GPUCosThetas[_pol];
		_3d_sigma=(_xs_t+_scale);
		_3d_oneOverSigma=1.0/_3d_sigma;
		_3d_Q=(_flatSour + _scattSour + _scale*_axialSurfAnfFlux_inp[_angId + _pol - _polOffset]);
		_sourTerm=_3d_Q*_3d_oneOverSigma;
		_sigmaTByLength = _3d_sigma * _length;
#endif // defined(M_3DMOCex)
		_x2 = _sigmaTByLength * GPUSinThetaForExp[_pol];
		_x1 = (int)(_x2);
		_x3 = _x2 - _x1;
		
		_exp = _GPULinearExpTable[_x1] + _x3 * (_GPULinearExpTable[_x1 + 1] - _GPULinearExpTable[_x1]);
        _delta = (_track_flux[_pol - _polOffset] - _sourTerm)*_exp;
		_track_flux[_pol - _polOffset] -= _delta;

// #if defined(M_2DMOC)
// 		_fsrFlux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + _pol];
// #endif // defined(M_2DMOC)

// #if defined(M_3DMOCex)
		_fsrFlux = _oneOverVolbySpacing*(_sourTerm*_length+GPUSinThetas[_pol]*_3d_oneOverSigma*_delta);
#ifdef DOUBLE_FLUX
    	atomicAdd_(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
    	atomicAdd(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
#endif
// #endif // defined(M_3DMOCex)
	}

// #if defined(M_2DMOC)
// 	_fsrFlux *= _oneOverVolume;
// 	/* Atomically increment the scalar flux for this FSR */
// #ifdef DOUBLE_FLUX
// 	atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
// #endif

// #ifdef SINGLE_FLUX
// 	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
// #endif
// #endif // defined(M_2DMOC)
}

__device__ void GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_HOScatt_1(int _indexLay, int _azim, FP_PRECISION _length, \
	FP_PRECISION _oneOverVolume, FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *_track_flux, FP_PRECISION *_GPUSource, \
	FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPULinearExpTable, int _polarBoundLowLim,int _polarBoundUpLim, \
	int _polOffset, FP_PRECISION _twoOverWidth, FP_PRECISION _oneOverVolbySpacing,M_SIZET _angId,FP_PRECISION *_axialSurfAnfFlux_inp, \
	FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_GPUFSRAngScattSour)
{
	int _x1, _pol;
	FP_PRECISION _xs_t, _sigmaTByLength, _fsrFlux,_delta,_flatSour,_x2,_x3, _exp,_sourTerm,_scattSour=0.0,_3d_oneOverSigma;
#if defined(M_3DMOCex)
	FP_PRECISION _3d_sigma, _3d_Q,_scale;
#endif // defined(M_3DMOCex)
	_xs_t = _GPUMacroXS_t[_indexLay];
	_flatSour = _GPUSource[_indexLay];

#if defined(M_2DMOC)
	_sigmaTByLength = _xs_t * _length;
	_3d_oneOverSigma = 1.0 / _xs_t;
	// _sourTerm = _flatSour;
	
	_fsrFlux = (FP_PRECISION)(0.0);
#endif // defined(M_2DMOC)

	for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {

		// _scattSour = GPUComputeFSRScattSourceOnDevice_paraFsr_paraGroup_Layer(_GPUScalarFlux, \
		// 	_GPUMacroXS_s, _GPUMacroXS_s_NZRowStartID_Layer, _GPUMacroXS_s_NZColStartID,\
		// 	_GPUSpheHarmCoeff, _fsrID, _groupID, _azim, _pol);

		_scattSour = _GPUFSRAngScattSour[_angId+_pol-_polOffset];
		// _scattSour = 0.0;

#if defined(M_2DMOC)
	    _sourTerm = (_flatSour + _scattSour) * _3d_oneOverSigma;
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
		_scale=_twoOverWidth*GPUCosThetas[_pol];
		_3d_sigma=(_xs_t+_scale);
		_3d_oneOverSigma=1.0/_3d_sigma;
		_3d_Q=(_flatSour + _scattSour + _scale*_axialSurfAnfFlux_inp[_angId + _pol - _polOffset]);
		_sourTerm=_3d_Q*_3d_oneOverSigma;
		_sigmaTByLength = _3d_sigma * _length;
#endif // defined(M_3DMOCex)
//		_x2 = _sigmaTByLength * GPUSinThetaForExp[_pol];
//		_x1 = (int)(_x2);
//		_x3 = _x2 - _x1;

//		_exp = _GPULinearExpTable[_x1] + _x3 * (_GPULinearExpTable[_x1 + 1] - _GPULinearExpTable[_x1]);

         _exp = 1. - _ExpFunction_GPU(-_sigmaTByLength * GPUOneOverSinThetas[_pol]);


        _delta = (_track_flux[_pol - _polOffset] - _sourTerm)*_exp;
		_track_flux[_pol - _polOffset] -= _delta;
//
		_fsrFlux = _oneOverVolbySpacing*(_sourTerm*_length+GPUSinThetas[_pol]*_3d_oneOverSigma*_delta);
#ifdef DOUBLE_FLUX
    	atomicAdd_(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
    	atomicAdd(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
#endif
	}
}

__device__ void GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_HOScatt_11(int _indexLay, int _azim, int _pol, FP_PRECISION _length, \
	FP_PRECISION _oneOverVolume, FP_PRECISION *_GPUMacroXS_t, FP_PRECISION _track_flux, FP_PRECISION *_GPUSource, \
	FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPULinearExpTable, int _polarBoundLowLim,int _polarBoundUpLim, \
	int _polOffset, FP_PRECISION _twoOverWidth, FP_PRECISION _oneOverVolbySpacing,M_SIZET _angId,FP_PRECISION *_axialSurfAnfFlux_inp, \
	FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_GPUFSRAngScattSour, int a)
{
    int _x1;
    a++;
    FP_PRECISION _xs_t, _sigmaTByLength, _fsrFlux,_delta,_flatSour,_x2,_x3, _exp,_sourTerm,_scattSour=0.0,_3d_oneOverSigma;
#if defined(M_3DMOCex)
    FP_PRECISION _3d_sigma, _3d_Q,_scale;
#endif // defined(M_3DMOCex)
    _xs_t = _GPUMacroXS_t[_indexLay];
    _flatSour = _GPUSource[_indexLay];

#if defined(M_2DMOC)
    _sigmaTByLength = _xs_t * _length;
    _3d_oneOverSigma = 1.0 / _xs_t;
    // _sourTerm = _flatSour;

    _fsrFlux = (FP_PRECISION) (0.0);
#endif // defined(M_2DMOC)


    // _scattSour = GPUComputeFSRScattSourceOnDevice_paraFsr_paraGroup_Layer(_GPUScalarFlux, \
		// 	_GPUMacroXS_s, _GPUMacroXS_s_NZRowStartID_Layer, _GPUMacroXS_s_NZColStartID,\
		// 	_GPUSpheHarmCoeff, _fsrID, _groupID, _azim, _pol);

    _scattSour = _GPUFSRAngScattSour[_angId + _pol - _polOffset];
    // _scattSour = 0.0;

#if defined(M_2DMOC)
    _sourTerm = (_flatSour + _scattSour) * _3d_oneOverSigma;
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
    _scale=_twoOverWidth*GPUCosThetas[_pol];
    _3d_sigma=(_xs_t+_scale);
    _3d_oneOverSigma=1.0/_3d_sigma;
    _3d_Q=(_flatSour + _scattSour + _scale*_axialSurfAnfFlux_inp[_angId + _pol - _polOffset]);
    _sourTerm=_3d_Q*_3d_oneOverSigma;
    _sigmaTByLength = _3d_sigma * _length;
#endif // defined(M_3DMOCex)
//		_x2 = _sigmaTByLength * GPUSinThetaForExp[_pol];
//		_x1 = (int)(_x2);
//		_x3 = _x2 - _x1;

//		_exp = _GPULinearExpTable[_x1] + _x3 * (_GPULinearExpTable[_x1 + 1] - _GPULinearExpTable[_x1]);

    _exp = 1. - _ExpFunction_GPU(-_sigmaTByLength * GPUOneOverSinThetas[_pol]);


    _delta = (_track_flux - _sourTerm) * _exp;
    _track_flux -= _delta;
//
    _fsrFlux = _oneOverVolbySpacing * (_sourTerm * _length + GPUSinThetas[_pol] * _3d_oneOverSigma * _delta);
#ifdef DOUBLE_FLUX
    atomicAdd_(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
    atomicAdd(&_axialSurfAngFlux_update[_angId + _pol - _polOffset], (FP_PRECISION_FLUX) _fsrFlux);
#endif

}

__device__ void GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_HOScatt_2(int _indexLay, int _azim, FP_PRECISION _length, \
	FP_PRECISION _oneOverVolume, FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *_GPUMacroXS_t_inv, FP_PRECISION *_track_flux, FP_PRECISION *_GPUSource, \
	FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPULinearExpTable, int _polarBoundLowLim,int _polarBoundUpLim, \
	int _polOffset, FP_PRECISION _twoOverWidth, FP_PRECISION _oneOverVolbySpacing,M_SIZET _angId,FP_PRECISION *_axialSurfAnfFlux_inp, \
	FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_GPUFSRAngScattSour)
{
	int _x1, _pol;
	FP_PRECISION _xs_t, _sigmaTByLength, _fsrFlux,_delta,_flatSour,_x2,_x3, _exp,_sourTerm,_scattSour=0.0,_3d_oneOverSigma;
#if defined(M_3DMOCex)
	FP_PRECISION _3d_sigma, _3d_Q,_scale;
#endif // defined(M_3DMOCex)
	_xs_t = _GPUMacroXS_t[_indexLay];
	// _flatSour = _GPUSource[_indexLay];

#if defined(M_2DMOC)
	_sigmaTByLength = _xs_t * _length;
	// _3d_oneOverSigma = 1.0 / _xs_t;
	_3d_oneOverSigma = _GPUMacroXS_t_inv[_indexLay];
	// _sourTerm = _flatSour;
	
	_fsrFlux = (FP_PRECISION)(0.0);
#endif // defined(M_2DMOC)

	for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {

		// _scattSour = GPUComputeFSRScattSourceOnDevice_paraFsr_paraGroup_Layer(_GPUScalarFlux, \
		// 	_GPUMacroXS_s, _GPUMacroXS_s_NZRowStartID_Layer, _GPUMacroXS_s_NZColStartID,\
		// 	_GPUSpheHarmCoeff, _fsrID, _groupID, _azim, _pol);

		// _scattSour = _GPUFSRAngScattSour[_angId+_pol-_polOffset];
		// _scattSour = 0.0;

#if defined(M_2DMOC)
	    // _sourTerm = (_flatSour + _scattSour) * _3d_oneOverSigma;
	    _sourTerm = _GPUFSRAngScattSour[_angId + _pol - _polOffset];
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
		_scale=_twoOverWidth*GPUCosThetas[_pol];
		_3d_sigma=(_xs_t+_scale);
		_3d_oneOverSigma=1.0/_3d_sigma;
		// _3d_Q=(_flatSour + _scattSour + _scale*_axialSurfAnfFlux_inp[_angId + _pol - _polOffset]);
		_3d_Q=_axialSurfAnfFlux_inp[_angId + _pol - _polOffset];
		_sourTerm=_3d_Q*_3d_oneOverSigma;
		_sigmaTByLength = _3d_sigma * _length;
#endif // defined(M_3DMOCex)
		_x2 = _sigmaTByLength * GPUSinThetaForExp[_pol];
		_x1 = (int)(_x2);
		_x3 = _x2 - _x1;
		
		_exp = _GPULinearExpTable[_x1] + _x3 * (_GPULinearExpTable[_x1 + 1] - _GPULinearExpTable[_x1]);
        _delta = (_track_flux[_pol - _polOffset] - _sourTerm)*_exp;
		_track_flux[_pol - _polOffset] -= _delta;

		_fsrFlux = _oneOverVolbySpacing*(_sourTerm*_length+GPUSinThetas[_pol]*_3d_oneOverSigma*_delta);
#ifdef DOUBLE_FLUX
    	atomicAdd_(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
    	atomicAdd(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
#endif
	}
}




__device__ void GPUTallyScalarFlux_ExpFunc_OneDir_AngFSmem(int _index, int _indexLay, int _azim, FP_PRECISION _length, FP_PRECISION _oneOverVolume, \
	FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *_track_flux, FP_PRECISION *_GPUSource, FP_PRECISION_FLUX *_GPUScalarFlux, \
	int _polarBoundLowLim,int _polarBoundUpLim, int _polOffset, FP_PRECISION _twoOverWidth,\
	FP_PRECISION _oneOverVolbySpacing, M_SIZET _angId,FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update)
{
	int _pol;
	FP_PRECISION _xs_t, _sigmaTByLength, _fsrFlux,_delta,_flatSour, _exp,_sourTerm;
#if defined(M_3DMOCex)
	FP_PRECISION _3d_sigma, _3d_oneOverSigma, _3d_Q,_scale;
#endif // defined(M_3DMOCex)
	_xs_t = _GPUMacroXS_t[_indexLay];
	_flatSour = _GPUSource[_indexLay];

#if defined(M_2DMOC)
	_sigmaTByLength = _xs_t * _length;
	_sourTerm = _flatSour;
	
	_fsrFlux = (FP_PRECISION)(0.0);
#endif // defined(M_2DMOC)

	for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {

#if defined(M_3DMOCex)
		_scale=_twoOverWidth*GPUCosThetas[_pol];
		_3d_sigma=(_xs_t+_scale);
		_3d_oneOverSigma=1.0/_3d_sigma;
		_3d_Q=(_flatSour+_scale*_axialSurfAnfFlux_inp[_angId + _pol - _polOffset]);
		_sourTerm=_3d_Q*_3d_oneOverSigma;
		_sigmaTByLength = _3d_sigma * _length;
#endif // defined(M_3DMOCex)

		_exp=1.-_ExpFunction_GPU(-_sigmaTByLength * GPUOneOverSinThetas[_pol]);
		_delta = (_track_flux[_pol - _polOffset] - _sourTerm)*_exp;
		_track_flux[_pol - _polOffset] -= _delta;

#if defined(M_2DMOC)
		_fsrFlux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + _pol];
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
		_fsrFlux = _oneOverVolbySpacing*(_sourTerm*_length+GPUSinThetas[_pol]*_3d_oneOverSigma*_delta);
#ifdef DOUBLE_FLUX
    	atomicAdd_(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
    	atomicAdd(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
#endif
#endif // defined(M_3DMOCex)
	}

#if defined(M_2DMOC)
	_fsrFlux *= _oneOverVolume;
	/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
#endif
#endif // defined(M_2DMOC)
}

__device__ void GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_opt2(int _index, int _azim, FP_PRECISION _length, FP_PRECISION _oneOverVolume, \
	FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *_GPUMacroXS_t_wScale, FP_PRECISION *_track_flux, FP_PRECISION *_GPUSource, FP_PRECISION_FLUX *_GPUScalarFlux, \
	FP_PRECISION *_GPULinearExpTable, int _polarBoundLowLim,int _polarBoundUpLim, int _polOffset,\
	FP_PRECISION _oneOverVolbySpacing,int _angId,FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update)
{
// 	int _x1, _pol;
// 	FP_PRECISION _xs_t, _sigmaTByLength, _fsrFlux,_delta,_flatSour,_x2,_x3, _exp,_sourTerm;
// #if defined(M_3DMOCex)
// 	FP_PRECISION _3d_sigma, _3d_oneOverSigma, _3d_Q, _3d_QtoSigma,_scale;
// #endif // defined(M_3DMOCex)
	
	

// #if defined(M_2DMOC)
// 	_xs_t = _GPUMacroXS_t[_index];
// 	_flatSour = _GPUSource[_index];
// 	_sigmaTByLength = _xs_t * _length;
// 	_sourTerm = _flatSour;
	
// 	_fsrFlux = (FP_PRECISION)(0.0);
// #endif // defined(M_2DMOC)

// 	for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {

// #if defined(M_3DMOCex)
// 		// _scale=_twoOverWidth*GPUCosThetas[_pol];
// 		// // _3d_sigma=(_xs_t+_scale);
// 		_3d_sigma=_GPUMacroXS_t_wScale[_index*(*GPUNumPolarAng)+_pol];
// 		_3d_oneOverSigma=1.0/_3d_sigma;

// 		// _3d_Q=(_flatSour+_scale*_axialSurfAnfFlux_inp[_angId + _pol - _polOffset]);
// 		// _sourTerm=_3d_Q*_3d_oneOverSigma;

// 		_sourTerm = _axialSurfAnfFlux_inp[_angId + _pol - _polOffset];

// 		_sigmaTByLength = _3d_sigma * _length;
// #endif // defined(M_3DMOCex)
// 		_x2 = _sigmaTByLength * GPUSinThetaForExp[_pol];
// 		_x1 = (int)(_x2);
// 		_x3 = _x2 - _x1;
// 		_exp = _GPULinearExpTable[_x1] + _x3 * (_GPULinearExpTable[_x1 + 1] - _GPULinearExpTable[_x1]);
// 		// _exp=1.-expf(-_sigmaTByLength/GPUSinThetas[_pol]);
//         _delta = (_track_flux[_pol - _polOffset] - _sourTerm)*_exp;
// 		_track_flux[_pol - _polOffset] -= _delta;
// 		// if (_track_flux[_pol - _polOffset]<0.0) {
// 		// 	_track_flux[_pol - _polOffset] = 0.0;
// 		// }

// #if defined(M_2DMOC)
// 		_fsrFlux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + _pol];
// #endif // defined(M_2DMOC)

// #if defined(M_3DMOCex)
// 		_fsrFlux = _oneOverVolbySpacing*(/*_3d_QtoSigma*/_sourTerm*_length+GPUSinThetas[_pol]*_3d_oneOverSigma*_delta);
// #ifdef DOUBLE_FLUX
//     	atomicAdd_(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
// #endif

// #ifdef SINGLE_FLUX
//     	atomicAdd(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
// #endif
// #endif // defined(M_3DMOCex)
// 	}

// #if defined(M_2DMOC)
// 	_fsrFlux *= _oneOverVolume;
// 	/* Atomically increment the scalar flux for this FSR */
// #ifdef DOUBLE_FLUX
// 	atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
// #endif

// #ifdef SINGLE_FLUX
// 	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
// #endif
// #endif // defined(M_2DMOC)
}


/**
* @ Function: GPUTS_Group_DDExpInte_OneDir_AngFSmem()
*   introduction: transport sweep solver on GPU, length and fsr ID are stored in the individual arries, not the structure array
* @ Parameter List:
*   Return: void
* @ Created     : 2019/12/25, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
__global__ void GPUTS_Group_DDExpInte_OneDir_AngFSmem(int _lay, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
	int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
    FP_PRECISION *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
    FP_PRECISION *_GPULinearExpTable,FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
    FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
	FP_PRECISION_GEO *_GPURTMWidth_Y, int _polarBoundLowLim,int _polarBoundUpLim, FP_PRECISION _twoOverLayWidth,\
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, int _polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC)
{
	int _tid, _ltGlobID, _groupID, _azim, _azimNext, _ltIDNext, _polEngID, _pol, track_flux_index, _cell, _ray, _globCellID, _trackIndex;
	int _seg, _reg, _fsrID, _track, _index,_numPol, _ltInd, _indexLay;
	FP_PRECISION _length, _oneOverVolume, _oneOverVolbySpacing;
	M_SIZET _angId;
#if defined(M_3DMOCex)
	FP_PRECISION _spacing;
	int _angId2,_angId3, _layFsrID;
	M_SIZET _angId1;
#endif // defined(M_3DMOCex)
	/* Shared memory buffer for each thread's angular flux */
	extern __shared__ FP_PRECISION temp_flux[];
	FP_PRECISION *_track_flux = NULL;
	FP_PRECISION _beta;

#if defined(M_MG_CMFD)
	int _surfIndex_Layer_X, _surfIndex_Layer_Y,_firstTrackInLT;
	_surfIndex_Layer_X = ((*GPUMOCNumPinsInSubDom_X) + 1) * (*GPUMOCNumPinsInSubDom_Y);
	_surfIndex_Layer_Y = (*GPUMOCNumPinsInSubDom_X) * ((*GPUMOCNumPinsInSubDom_Y) + 1);
#endif // M_MG_CMFD

    _ltInd = _lay * (*GPUNumLTs_Total);
	_numPol=(*GPUNumPolarAng);
	
#if defined(M_3DMOCex)
	_numPol=_numPol/2;
#endif // defined(M_3DMOCex)

	_tid = threadIdx.x + blockIdx.x * blockDim.x;
	_ltGlobID = _tid / (*GPUNumMOCGroups);
	_groupID = _tid % (*GPUNumMOCGroups);

	_polEngID = _groupID * (*GPUNumPolarAng);

	track_flux_index = threadIdx.x * _numPol;

#if defined(M_3DMOCex)
	_angId2 = _groupID*(*GPU_angMul2_HP);
#endif // defined(M_3DMOCex)

	if (_ltGlobID < (*GPUNumLTs_Total)) {
		/* Retrieve pointer to thread's shared memory buffer for angular flux */
		_track_flux = &temp_flux[track_flux_index];
		
		for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
    		_track_flux[_pol-_polOffset] = _GPUBoundaryFlux(/*_ltInd + */_ltGlobID, _pol + _polEngID);
		}
		_azim = _GPULTsIndex_AzimAngID[_ltGlobID];

#if defined(M_3DMOCex)
		_angId3 = _azim*_numPol;
		_spacing = GPUEffRaySpacing[_azim];
#endif // defined(M_3DMOCex)

#if defined(M_MG_CMFD)
        _firstTrackInLT = 0;
#endif // defined(M_MG_CMFD)
		/* Loop over each Track segment in forward direction */
		for (_track = _GPUTrackIndexInLTs[_ltGlobID]; _track < _GPUTrackIndexInLTs[_ltGlobID + 1]; _track++) {
			_cell = _GPUTracksInfoInLTs[_track].cell_ID;
			_ray = _GPUTracksInfoInLTs[_track].track_ID;
			_globCellID = _lay * _numCellsInDom + _cell;
            _trackIndex = (*GPUTotNumTracksInOneRTM_2PI)*_GPUCellRTMType_SubDom[_globCellID] + _GPUIndexTrackInRTM_SumAzim[_azim] + _ray;
#if defined(M_MG_CMFD)
            if (_firstTrackInLT == 0) {
                GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
                    _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 1, \
                    _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
                    _GPURTMWidth_X, _GPURTMWidth_Y,_polarBoundLowLim, _polarBoundUpLim,_polOffset,_GPUTotalWeights_2DMOC);
            }
#endif // defined(M_MG_CMFD)
			for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {
				_length = _GPUSegInfoInTrack_2PI_length[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];
				_reg = _GPUSegInfoInTrack_2PI_FSRID[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];

				_fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID] + _reg;
				// _oneOverVolume = /*(FP_PRECISION)(1.0) /*/ _GPUOneOverFSRsAreaInGlob[_fsrID];
				_oneOverVolume = _GPUOneOverFSRsAreaInRTM[_GPUSumNumFSRsInRTM[_GPUCellRTMType_SubDom[_globCellID]] + _reg];
#if defined(M_3DMOCex)
				_oneOverVolbySpacing = _oneOverVolume * _spacing;
				_layFsrID = _GPUSumNumFSRsBasedGlobCellID[_cell] + _reg;
				_angId1 = (M_SIZET)(_layFsrID) * (M_SIZET)(*GPU_angMul1_HP);
				_indexLay = _layFsrID * (*GPUNumMOCGroups) + _groupID;
#endif //  defined(M_3DMOCex)
				_index = _fsrID * (*GPUNumMOCGroups) + _groupID;

#if defined(M_2DMOC)
				_indexLay = _index;
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
				_angId = _angId1 + (M_SIZET)(_angId2 + _angId3);
#endif //  defined(M_3DMOCex)

			    GPUTallyScalarFlux_DDExpInte_OneDir_AngFSmem(_index, _indexLay, _azim, _length, _oneOverVolume, \
			    	_GPUMacroXS_t, _track_flux, _GPUSource, _GPUScalarFlux, _GPULinearExpTable, \
					_polarBoundLowLim, _polarBoundUpLim, _polOffset, _twoOverLayWidth,_oneOverVolbySpacing,\
					_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update);

            }
#if defined(M_MG_CMFD)
            GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
                _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 0, \
                _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
                _GPURTMWidth_X, _GPURTMWidth_Y,_polarBoundLowLim, _polarBoundUpLim,_polOffset,_GPUTotalWeights_2DMOC);
            _firstTrackInLT=1;
#endif // M_MG_CMFD
		}
		_azimNext = _GPULongTracksInfo[_ltGlobID].next_azim_angle_id;
		_ltIDNext = _GPULongTracksInfo[_ltGlobID].next_LongTrack_id_SubDom;
		_beta = _GPULongTracksInfo[_ltGlobID]._boundBeta;

#ifdef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme(_ltInd,_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUBoundaryFlux, _GPUMPIFlux, _beta, _polarBoundLowLim, _polarBoundUpLim, _polOffset);
#endif // MPI_CUDA_UpAngFImme

#ifndef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUStartFlux, _GPUMPIFlux, _beta);
#endif // !MPI_CUDA_UpAngFImme
	}
}

__global__ void GPUTS_Group_DDExpFunc_OneDir_AngFSmem(int _lay, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
	int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
    FP_PRECISION *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
    FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
    FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
	FP_PRECISION_GEO *_GPURTMWidth_Y, int _polarBoundLowLim,int _polarBoundUpLim, FP_PRECISION _twoOverLayWidth,\
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, int _polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC)
{
	int _tid, _ltGlobID, _groupID, _azim, _azimNext, _ltIDNext, _polEngID, _pol, track_flux_index, _cell, _ray, _globCellID, _trackIndex;
	int _seg, _reg, _fsrID, _track, _index,_numPol, _ltInd, _indexLay;
	FP_PRECISION _length, _oneOverVolume, _oneOverVolbySpacing;
	M_SIZET _angId;
#if defined(M_3DMOCex)
	FP_PRECISION _spacing;
	int _angId2,_angId3, _layFsrID;
	M_SIZET _angId1;
#endif // defined(M_3DMOCex)
	/* Shared memory buffer for each thread's angular flux */
	extern __shared__ FP_PRECISION temp_flux[];
	FP_PRECISION *_track_flux = NULL;
	FP_PRECISION _beta;

#if defined(M_MG_CMFD)
	int _surfIndex_Layer_X, _surfIndex_Layer_Y,_firstTrackInLT;
	_surfIndex_Layer_X = ((*GPUMOCNumPinsInSubDom_X) + 1) * (*GPUMOCNumPinsInSubDom_Y);
	_surfIndex_Layer_Y = (*GPUMOCNumPinsInSubDom_X) * ((*GPUMOCNumPinsInSubDom_Y) + 1);
#endif // M_MG_CMFD

    _ltInd = _lay * (*GPUNumLTs_Total);
	_numPol=(*GPUNumPolarAng);
	
#if defined(M_3DMOCex)
	_numPol=_numPol/2;
#endif // defined(M_3DMOCex)

	_tid = threadIdx.x + blockIdx.x * blockDim.x;
	_ltGlobID = _tid / (*GPUNumMOCGroups);
	_groupID = _tid % (*GPUNumMOCGroups);

	_polEngID = _groupID * (*GPUNumPolarAng);

	track_flux_index = threadIdx.x * _numPol;

#if defined(M_3DMOCex)
	_angId2 = _groupID*(*GPU_angMul2_HP);
#endif // defined(M_3DMOCex)

	if (_ltGlobID < (*GPUNumLTs_Total)) {
		/* Retrieve pointer to thread's shared memory buffer for angular flux */
		_track_flux = &temp_flux[track_flux_index];
		
		for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
    		_track_flux[_pol-_polOffset] = _GPUBoundaryFlux(/*_ltInd + */_ltGlobID, _pol + _polEngID);
		}
		_azim = _GPULTsIndex_AzimAngID[_ltGlobID];

#if defined(M_3DMOCex)
		_angId3 = _azim*_numPol;
		_spacing = GPUEffRaySpacing[_azim];
#endif // defined(M_3DMOCex)

#if defined(M_MG_CMFD)
        _firstTrackInLT = 0;
#endif // defined(M_MG_CMFD)
		/* Loop over each Track segment in forward direction */
		for (_track = _GPUTrackIndexInLTs[_ltGlobID]; _track < _GPUTrackIndexInLTs[_ltGlobID + 1]; _track++) {
			_cell = _GPUTracksInfoInLTs[_track].cell_ID;
			_ray = _GPUTracksInfoInLTs[_track].track_ID;
			_globCellID = _lay * _numCellsInDom + _cell;
            _trackIndex = (*GPUTotNumTracksInOneRTM_2PI)*_GPUCellRTMType_SubDom[_globCellID] + _GPUIndexTrackInRTM_SumAzim[_azim] + _ray;
#if defined(M_MG_CMFD)
            if (_firstTrackInLT == 0) {
                GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
                    _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 1, \
                    _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
                    _GPURTMWidth_X, _GPURTMWidth_Y,_polarBoundLowLim, _polarBoundUpLim,_polOffset,_GPUTotalWeights_2DMOC);
            }
#endif // defined(M_MG_CMFD)
			for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {
				_length = _GPUSegInfoInTrack_2PI_length[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];
				_reg = _GPUSegInfoInTrack_2PI_FSRID[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];

				_fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID] + _reg;
				// _oneOverVolume = /*(FP_PRECISION)(1.0) /*/ _GPUOneOverFSRsAreaInGlob[_fsrID];
				_oneOverVolume = _GPUOneOverFSRsAreaInRTM[_GPUSumNumFSRsInRTM[_GPUCellRTMType_SubDom[_globCellID]] + _reg];
#if defined(M_3DMOCex)
				_oneOverVolbySpacing = _oneOverVolume * _spacing;
				_layFsrID = _GPUSumNumFSRsBasedGlobCellID[_cell] + _reg;
				_angId1 = (M_SIZET)(_layFsrID) * (M_SIZET)(GPU_angMul1_HP);
				_indexLay = _layFsrID * (*GPUNumMOCGroups) + _groupID;
#endif //  defined(M_3DMOCex)
				_index = _fsrID * (*GPUNumMOCGroups) + _groupID;
				
#if defined(M_2DMOC)
				_indexLay = _index;
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
				_angId = _angId1 + (M_SIZET)(_angId2 + _angId3);
#endif //  defined(M_3DMOCex)

			    GPUTallyScalarFlux_DDExpFunc_OneDir_AngFSmem(_index, _indexLay, _azim, _length, _oneOverVolume, \
			    	_GPUMacroXS_t, _track_flux, _GPUSource, _GPUScalarFlux, \
					_polarBoundLowLim, _polarBoundUpLim, _polOffset, _twoOverLayWidth,_oneOverVolbySpacing,\
					_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update);

            }
#if defined(M_MG_CMFD)
            GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
                _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 0, \
                _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
                _GPURTMWidth_X, _GPURTMWidth_Y,_polarBoundLowLim, _polarBoundUpLim,_polOffset,_GPUTotalWeights_2DMOC);
            _firstTrackInLT=1;
#endif // M_MG_CMFD
		}
		_azimNext = _GPULongTracksInfo[_ltGlobID].next_azim_angle_id;
		_ltIDNext = _GPULongTracksInfo[_ltGlobID].next_LongTrack_id_SubDom;
		_beta = _GPULongTracksInfo[_ltGlobID]._boundBeta;

#ifdef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme(_ltInd,_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUBoundaryFlux, _GPUMPIFlux, _beta, _polarBoundLowLim, _polarBoundUpLim, _polOffset);
#endif // MPI_CUDA_UpAngFImme

#ifndef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUStartFlux, _GPUMPIFlux, _beta);
#endif // !MPI_CUDA_UpAngFImme
	}
}


/**
* @ Function: GPUTS_Group_ExpInte_OneDir_AngFSmem()
*   introduction: transport sweep solver on GPU, length and fsr ID are stored in the individual arries, not the structure array
* @ Parameter List:
*   Return: void
* @ Created     : 2019/12/27, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
__global__ void GPUTS_Group_ExpInte_OneDir_AngFSmem(int _lay, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
	int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
    FP_PRECISION *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
    FP_PRECISION *_GPULinearExpTable,FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
    FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
	FP_PRECISION_GEO *_GPURTMWidth_Y, int _polarBoundLowLim,int _polarBoundUpLim, FP_PRECISION _twoOverLayWidth,\
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, int _polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC, \
	FP_PRECISION *_GPUMacroXS_t_wScale)
{
	int _tid, _ltGlobID, _groupID, _azim, _azimNext, _ltIDNext, _polEngID, _pol, track_flux_index, _cell, _ray, _globCellID, _trackIndex;
	int _seg, _reg, _fsrID, _track, _index,_indexLay,_numPol,_ltInd;
	FP_PRECISION _length, _oneOverVolume, _oneOverVolbySpacing;
	M_SIZET _angId;
#if defined(M_3DMOCex)
	FP_PRECISION _spacing;
	int _angId2,_angId3, _layFsrID;
	M_SIZET _angId1;
#endif // defined(M_3DMOCex)
	/* Shared memory buffer for each thread's angular flux */
	extern __shared__ FP_PRECISION temp_flux[];
	FP_PRECISION *_track_flux = NULL;
	FP_PRECISION _beta;

#if defined(M_MG_CMFD)
	int _surfIndex_Layer_X, _surfIndex_Layer_Y,_firstTrackInLT;
	_surfIndex_Layer_X = ((*GPUMOCNumPinsInSubDom_X) + 1) * (*GPUMOCNumPinsInSubDom_Y);
	_surfIndex_Layer_Y = (*GPUMOCNumPinsInSubDom_X) * ((*GPUMOCNumPinsInSubDom_Y) + 1);
#endif // M_MG_CMFD

    _ltInd = _lay * (*GPUNumLTs_Total);
	_numPol=(*GPUNumPolarAng);

	// _numPol=1;
	
#if defined(M_3DMOCex)
	_numPol=_numPol/2;
#endif // defined(M_3DMOCex)

	_tid = threadIdx.x + blockIdx.x * blockDim.x;
	_ltGlobID = _tid / (*GPUNumMOCGroups);
	_groupID = _tid % (*GPUNumMOCGroups);

	// printf("\ntid: %10d, _ltGlobID: %5d, _groupID:%5d\n", _tid, _ltGlobID, _groupID);

	_polEngID = _groupID * (*GPUNumPolarAng);

	track_flux_index = threadIdx.x * _numPol;

#if defined(M_3DMOCex)
	_angId2 = _groupID*(*GPU_angMul2_HP);
#endif // defined(M_3DMOCex)

	// while (_ltGlobID < (*GPUNumLTs_Total)) {
	if (_ltGlobID < (*GPUNumLTs_Total)) { // NORMAL
		/* Retrieve pointer to thread's shared memory buffer for angular flux */
		_track_flux = &temp_flux[track_flux_index];
		
		for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
    		_track_flux[_pol-_polOffset] = _GPUBoundaryFlux(/*_ltInd + */_ltGlobID, _pol + _polEngID);
		}
		_azim = _GPULTsIndex_AzimAngID[_ltGlobID];

#if defined(M_3DMOCex)
		_angId3 = _azim*_numPol;
		_spacing = GPUEffRaySpacing[_azim];
#endif // defined(M_3DMOCex)

#if defined(M_MG_CMFD)
        _firstTrackInLT = 0;
#endif // defined(M_MG_CMFD)
		/* Loop over each Track segment in forward direction */
		for (_track = _GPUTrackIndexInLTs[_ltGlobID]; _track < _GPUTrackIndexInLTs[_ltGlobID + 1]; _track++) {
			_cell = _GPUTracksInfoInLTs[_track].cell_ID;
			_ray = _GPUTracksInfoInLTs[_track].track_ID;
			_globCellID = _lay * _numCellsInDom + _cell;
            _trackIndex = (*GPUTotNumTracksInOneRTM_2PI)*_GPUCellRTMType_SubDom[_globCellID] + _GPUIndexTrackInRTM_SumAzim[_azim] + _ray;
#if defined(M_MG_CMFD)
            if (_firstTrackInLT == 0) {
                GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
                    _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 1, \
                    _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
					_GPURTMWidth_X, _GPURTMWidth_Y, _polarBoundLowLim, _polarBoundUpLim,  _polOffset,_GPUTotalWeights_2DMOC);
            }
#endif // defined(M_MG_CMFD)
			for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {
				_length = _GPUSegInfoInTrack_2PI_length[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];
				_reg = _GPUSegInfoInTrack_2PI_FSRID[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];

				_fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID] + _reg;
				// _oneOverVolume = /*(FP_PRECISION)(1.0) /*/ _GPUOneOverFSRsAreaInGlob[_fsrID];
				_oneOverVolume = _GPUOneOverFSRsAreaInRTM[_GPUSumNumFSRsInRTM[_GPUCellRTMType_SubDom[_globCellID]] + _reg];
#if defined(M_3DMOCex)
				_oneOverVolbySpacing = _oneOverVolume * _spacing;
				_layFsrID = _GPUSumNumFSRsBasedGlobCellID[_cell] + _reg;
				_angId1 = (M_SIZET)(_layFsrID) * (M_SIZET)(*GPU_angMul1_HP);
				_indexLay = _layFsrID * (*GPUNumMOCGroups) + _groupID;
#endif //  defined(M_3DMOCex)
				_index = _fsrID * (*GPUNumMOCGroups) + _groupID;
				
#if defined(M_2DMOC)
				_indexLay = _index;
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
				_angId = _angId1 + (M_SIZET)(_angId2 + _angId3);
#endif //  defined(M_3DMOCex)

#if !defined(M_3DOpt)
			    GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem(_index,_indexLay, _azim, _length, _oneOverVolume, \
			    	_GPUMacroXS_t, _track_flux, _GPUSource, _GPUScalarFlux, _GPULinearExpTable, \
					_polarBoundLowLim, _polarBoundUpLim, _polOffset, _twoOverLayWidth,_oneOverVolbySpacing,\
					_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update);
#else
				// GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_opt2(_index, _azim, _length, _oneOverVolume, \
			    // 	_GPUMacroXS_t, _GPUMacroXS_t_wScale, _track_flux, _GPUSource, _GPUScalarFlux, _GPULinearExpTable, \
				// 	_polarBoundLowLim, _polarBoundUpLim, _polOffset,_oneOverVolbySpacing,\
				// 	_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update);
#endif // 
            }
#if defined(M_MG_CMFD)
            GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
                _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 0, \
                _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
                _GPURTMWidth_X, _GPURTMWidth_Y, _polarBoundLowLim, _polarBoundUpLim,  _polOffset,_GPUTotalWeights_2DMOC);
            _firstTrackInLT=1;
#endif // M_MG_CMFD
		}
		_azimNext = _GPULongTracksInfo[_ltGlobID].next_azim_angle_id;
		_ltIDNext = _GPULongTracksInfo[_ltGlobID].next_LongTrack_id_SubDom;
		_beta = _GPULongTracksInfo[_ltGlobID]._boundBeta;

#ifdef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme(_ltInd,_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUBoundaryFlux, _GPUMPIFlux, _beta, _polarBoundLowLim, _polarBoundUpLim, _polOffset);
#endif // MPI_CUDA_UpAngFImme

#ifndef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUStartFlux, _GPUMPIFlux, _beta);
#endif // !MPI_CUDA_UpAngFImme

		// // for pb project effect
		// {
		// 	_tid += blockDim.x * gridDim.x;
		// 	_ltGlobID = _tid / (*GPUNumMOCGroups);
		// 	_groupID = _tid % (*GPUNumMOCGroups);

		// 	_polEngID = _groupID * (*GPUNumPolarAng);

		// 	track_flux_index = threadIdx.x * _numPol;
		// }
	}
}

__global__ void GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt(int _lay, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
	int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
    FP_PRECISION *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
    FP_PRECISION *_GPULinearExpTable,FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
    FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
	FP_PRECISION_GEO *_GPURTMWidth_Y, int _polarBoundLowLim,int _polarBoundUpLim, FP_PRECISION _twoOverLayWidth,\
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, int _polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC, \
	FP_PRECISION *_GPUMacroXS_t_wScale, FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer, \
	M_UINT64 *_GPUMacroXS_s_NZColStartID, FP_PRECISION *_GPUSpheHarmCoeff)
{
	int _tid, _ltGlobID, _groupID, _azim, _azimNext, _ltIDNext, _polEngID, _pol, track_flux_index, _cell, _ray, _globCellID, _trackIndex;
	int _seg, _reg, _fsrID, _track, _index,_indexLay,_numPol,_ltInd;
	FP_PRECISION _length, _oneOverVolume, _oneOverVolbySpacing;
	M_SIZET _angId;
// #if defined(M_3DMOCex)
	FP_PRECISION _spacing;
	int _angId2,_angId3, _layFsrID;
	M_SIZET _angId1;
// #endif // defined(M_3DMOCex)
	/* Shared memory buffer for each thread's angular flux */
	extern __shared__ FP_PRECISION temp_flux[];
	FP_PRECISION *_track_flux = NULL;
	FP_PRECISION _beta;

#if defined(M_MG_CMFD)
	int _surfIndex_Layer_X, _surfIndex_Layer_Y,_firstTrackInLT;
	_surfIndex_Layer_X = ((*GPUMOCNumPinsInSubDom_X) + 1) * (*GPUMOCNumPinsInSubDom_Y);
	_surfIndex_Layer_Y = (*GPUMOCNumPinsInSubDom_X) * ((*GPUMOCNumPinsInSubDom_Y) + 1);
#endif // M_MG_CMFD

    _ltInd = _lay * (*GPUNumLTs_Total);
	_numPol=(*GPUNumPolarAng);

	// _numPol=1;
	
#if defined(M_3DMOCex)
	_numPol=_numPol/2;
#endif // defined(M_3DMOCex)

	_tid = threadIdx.x + blockIdx.x * blockDim.x;
	_ltGlobID = _tid / (*GPUNumMOCGroups);
	_groupID = _tid % (*GPUNumMOCGroups);

	_polEngID = _groupID * (*GPUNumPolarAng);

	track_flux_index = threadIdx.x * _numPol;

#if defined(M_2DMOC)
    _angId2 = _groupID*(*GPU_angMul2_FP);
#else
	// #if defined(M_3DMOCex)
	_angId2 = _groupID*(*GPU_angMul2_HP);
#endif

	if (_ltGlobID < (*GPUNumLTs_Total)) {
		/* Retrieve pointer to thread's shared memory buffer for angular flux */
		_track_flux = &temp_flux[track_flux_index];
		
		for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
    		_track_flux[_pol-_polOffset] = _GPUBoundaryFlux(/*_ltInd + */_ltGlobID, _pol + _polEngID);
		}
		_azim = _GPULTsIndex_AzimAngID[_ltGlobID];

// #if defined(M_3DMOCex)
		_angId3 = _azim*_numPol;
		_spacing = GPUEffRaySpacing[_azim];
// #endif // defined(M_3DMOCex)

#if defined(M_MG_CMFD)
        _firstTrackInLT = 0;
#endif // defined(M_MG_CMFD)
		/* Loop over each Track segment in forward direction */
		for (_track = _GPUTrackIndexInLTs[_ltGlobID]; _track < _GPUTrackIndexInLTs[_ltGlobID + 1]; _track++) {
			_cell = _GPUTracksInfoInLTs[_track].cell_ID;
			_ray = _GPUTracksInfoInLTs[_track].track_ID;
			_globCellID = _lay * _numCellsInDom + _cell;
            _trackIndex = (*GPUTotNumTracksInOneRTM_2PI)*_GPUCellRTMType_SubDom[_globCellID] + _GPUIndexTrackInRTM_SumAzim[_azim] + _ray;
#if defined(M_MG_CMFD)
            if (_firstTrackInLT == 0) {
                GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
                    _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 1, \
                    _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
					_GPURTMWidth_X, _GPURTMWidth_Y, _polarBoundLowLim, _polarBoundUpLim,  _polOffset,_GPUTotalWeights_2DMOC);
            }
#endif // defined(M_MG_CMFD)
			for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {
				_length = _GPUSegInfoInTrack_2PI_length[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];
				_reg = _GPUSegInfoInTrack_2PI_FSRID[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];

				_fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID] + _reg;
				// _oneOverVolume = /*(FP_PRECISION)(1.0) /*/ _GPUOneOverFSRsAreaInGlob[_fsrID];
				_oneOverVolume = _GPUOneOverFSRsAreaInRTM[_GPUSumNumFSRsInRTM[_GPUCellRTMType_SubDom[_globCellID]] + _reg];
// #if defined(M_3DMOCex)
				_oneOverVolbySpacing = _oneOverVolume * _spacing;
				_layFsrID = _GPUSumNumFSRsBasedGlobCellID[_cell] + _reg;
#if defined(M_2DMOC)
                _angId1 = (M_SIZET)(_layFsrID) * (M_SIZET)(*GPU_angMul1_FP);
#else
				_angId1 = (M_SIZET)(_layFsrID) * (M_SIZET)(*GPU_angMul1_HP);
#endif
				_indexLay = _layFsrID * (*GPUNumMOCGroups) + _groupID;
// #endif //  defined(M_3DMOCex)
				_index = _fsrID * (*GPUNumMOCGroups) + _groupID;
				
#if defined(M_2DMOC)
				_indexLay = _index;
#endif // defined(M_2DMOC)

// #if defined(M_3DMOCex)
				_angId = _angId1 + (M_SIZET)(_angId2 + _angId3);
// #endif //  defined(M_3DMOCex)

#if !defined(M_3DOpt)
			    GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_HOScatt(_indexLay, _azim, _length, _oneOverVolume, \
			    	_GPUMacroXS_t, _track_flux, _GPUSource, _GPUScalarFlux, _GPULinearExpTable, \
					_polarBoundLowLim, _polarBoundUpLim, _polOffset, _twoOverLayWidth,_oneOverVolbySpacing,\
					_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update, _GPUMacroXS_s, _GPUMacroXS_s_NZRowStartID_Layer, \
					_GPUMacroXS_s_NZColStartID, _GPUSpheHarmCoeff, _layFsrID, _groupID);
#else
				// GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_opt2(_index, _azim, _length, _oneOverVolume, \
			    // 	_GPUMacroXS_t, _GPUMacroXS_t_wScale, _track_flux, _GPUSource, _GPUScalarFlux, _GPULinearExpTable, \
				// 	_polarBoundLowLim, _polarBoundUpLim, _polOffset,_oneOverVolbySpacing,\
				// 	_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update);
#endif // 
            }
#if defined(M_MG_CMFD)
            GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
                _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 0, \
                _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
                _GPURTMWidth_X, _GPURTMWidth_Y, _polarBoundLowLim, _polarBoundUpLim,  _polOffset,_GPUTotalWeights_2DMOC);
            _firstTrackInLT=1;
#endif // M_MG_CMFD
		}
		_azimNext = _GPULongTracksInfo[_ltGlobID].next_azim_angle_id;
		_ltIDNext = _GPULongTracksInfo[_ltGlobID].next_LongTrack_id_SubDom;
		_beta = _GPULongTracksInfo[_ltGlobID]._boundBeta;

#ifdef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme(_ltInd,_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUBoundaryFlux, _GPUMPIFlux, _beta, _polarBoundLowLim, _polarBoundUpLim, _polOffset);
#endif // MPI_CUDA_UpAngFImme

#ifndef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUStartFlux, _GPUMPIFlux, _beta);
#endif // !MPI_CUDA_UpAngFImme
	}
}

__global__ void GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt_1(int _lay, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
	int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
    FP_PRECISION *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
    FP_PRECISION *_GPULinearExpTable,FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
    FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
	FP_PRECISION_GEO *_GPURTMWidth_Y, int _polarBoundLowLim,int _polarBoundUpLim, FP_PRECISION _twoOverLayWidth,\
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, int _polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC, \
	FP_PRECISION *_GPUMacroXS_t_wScale, FP_PRECISION *_GPUFSRAngScattSour)
{
	int _tid, _ltGlobID, _groupID, _azim, _azimNext, _ltIDNext, _polEngID, _pol, track_flux_index, _cell, _ray, _globCellID, _trackIndex;
	int _seg, _reg, _fsrID, _track, _index,_indexLay,_numPol,_ltInd;
	FP_PRECISION _length, _oneOverVolume, _oneOverVolbySpacing;
	M_SIZET _angId;
// #if defined(M_3DMOCex)
	FP_PRECISION _spacing;
	int _angId2,_angId3, _layFsrID;
	M_SIZET _angId1;
// #endif // defined(M_3DMOCex)
	/* Shared memory buffer for each thread's angular flux */
	extern __shared__ FP_PRECISION temp_flux[];
	FP_PRECISION *_track_flux = NULL;
	FP_PRECISION _beta;

#if defined(M_MG_CMFD)
	int _surfIndex_Layer_X, _surfIndex_Layer_Y,_firstTrackInLT;
	_surfIndex_Layer_X = ((*GPUMOCNumPinsInSubDom_X) + 1) * (*GPUMOCNumPinsInSubDom_Y);
	_surfIndex_Layer_Y = (*GPUMOCNumPinsInSubDom_X) * ((*GPUMOCNumPinsInSubDom_Y) + 1);
#endif // M_MG_CMFD

    _ltInd = _lay * (*GPUNumLTs_Total);
	_numPol=(*GPUNumPolarAng);

	// _numPol=1;
	
#if defined(M_3DMOCex)
	_numPol=_numPol/2;
#endif // defined(M_3DMOCex)

	_tid = threadIdx.x + blockIdx.x * blockDim.x;
	_ltGlobID = _tid / (*GPUNumMOCGroups);
	_groupID = _tid % (*GPUNumMOCGroups);

	_polEngID = _groupID * (*GPUNumPolarAng);

	track_flux_index = threadIdx.x * _numPol;

#if defined(M_2DMOC)
    _angId2 = _groupID*(*GPU_angMul2_FP);
#else
	// #if defined(M_3DMOCex)
	_angId2 = _groupID*(*GPU_angMul2_HP);
#endif

	if (_ltGlobID < (*GPUNumLTs_Total)) {
		/* Retrieve pointer to thread's shared memory buffer for angular flux */
		_track_flux = &temp_flux[track_flux_index];
		
		for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
    		_track_flux[_pol-_polOffset] = _GPUBoundaryFlux(/*_ltInd + */_ltGlobID, _pol + _polEngID);
		}
		_azim = _GPULTsIndex_AzimAngID[_ltGlobID];

// #if defined(M_3DMOCex)
		_angId3 = _azim*_numPol;
		_spacing = GPUEffRaySpacing[_azim];
// #endif // defined(M_3DMOCex)

#if defined(M_MG_CMFD)
        _firstTrackInLT = 0;
#endif // defined(M_MG_CMFD)
		/* Loop over each Track segment in forward direction */
		for (_track = _GPUTrackIndexInLTs[_ltGlobID]; _track < _GPUTrackIndexInLTs[_ltGlobID + 1]; _track++) {
			_cell = _GPUTracksInfoInLTs[_track].cell_ID;
			_ray = _GPUTracksInfoInLTs[_track].track_ID;
			_globCellID = _lay * _numCellsInDom + _cell;
            _trackIndex = (*GPUTotNumTracksInOneRTM_2PI)*_GPUCellRTMType_SubDom[_globCellID] + _GPUIndexTrackInRTM_SumAzim[_azim] + _ray;
#if defined(M_MG_CMFD)
            if (_firstTrackInLT == 0) {
                GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
                    _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 1, \
                    _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
					_GPURTMWidth_X, _GPURTMWidth_Y, _polarBoundLowLim, _polarBoundUpLim,  _polOffset,_GPUTotalWeights_2DMOC);
            }
#endif // defined(M_MG_CMFD)
			for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {
				_length = _GPUSegInfoInTrack_2PI_length[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];
				_reg = _GPUSegInfoInTrack_2PI_FSRID[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];

				_fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID] + _reg;
				// _oneOverVolume = /*(FP_PRECISION)(1.0) /*/ _GPUOneOverFSRsAreaInGlob[_fsrID];
				_oneOverVolume = _GPUOneOverFSRsAreaInRTM[_GPUSumNumFSRsInRTM[_GPUCellRTMType_SubDom[_globCellID]] + _reg];
// #if defined(M_3DMOCex)
				_oneOverVolbySpacing = _oneOverVolume * _spacing;
				_layFsrID = _GPUSumNumFSRsBasedGlobCellID[_cell] + _reg;
#if defined(M_2DMOC)
                _angId1 = (M_SIZET)(_layFsrID) * (M_SIZET)(*GPU_angMul1_FP);
#else
				_angId1 = (M_SIZET)(_layFsrID) * (M_SIZET)(*GPU_angMul1_HP);
#endif
				_indexLay = _layFsrID * (*GPUNumMOCGroups) + _groupID;
// #endif //  defined(M_3DMOCex)
				_index = _fsrID * (*GPUNumMOCGroups) + _groupID;
				
#if defined(M_2DMOC)
				_indexLay = _index;
#endif // defined(M_2DMOC)

// #if defined(M_3DMOCex)
				_angId = _angId1 + (M_SIZET)(_angId2 + _angId3);
// #endif //  defined(M_3DMOCex)

#if !defined(M_3DOpt)
			    GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_HOScatt_1(_indexLay, _azim, _length, _oneOverVolume, \
			    	_GPUMacroXS_t, _track_flux, _GPUSource, _GPUScalarFlux, _GPULinearExpTable, \
					_polarBoundLowLim, _polarBoundUpLim, _polOffset, _twoOverLayWidth,_oneOverVolbySpacing,\
					_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update,_GPUFSRAngScattSour);
#else
				// GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_opt2(_index, _azim, _length, _oneOverVolume, \
			    // 	_GPUMacroXS_t, _GPUMacroXS_t_wScale, _track_flux, _GPUSource, _GPUScalarFlux, _GPULinearExpTable, \
				// 	_polarBoundLowLim, _polarBoundUpLim, _polOffset,_oneOverVolbySpacing,\
				// 	_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update);
#endif // 
            }
#if defined(M_MG_CMFD)
            GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
                _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 0, \
                _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
                _GPURTMWidth_X, _GPURTMWidth_Y, _polarBoundLowLim, _polarBoundUpLim,  _polOffset,_GPUTotalWeights_2DMOC);
            _firstTrackInLT=1;
#endif // M_MG_CMFD
		}
		_azimNext = _GPULongTracksInfo[_ltGlobID].next_azim_angle_id;
		_ltIDNext = _GPULongTracksInfo[_ltGlobID].next_LongTrack_id_SubDom;
		_beta = _GPULongTracksInfo[_ltGlobID]._boundBeta;

#ifdef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme(_ltInd,_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUBoundaryFlux, _GPUMPIFlux, _beta, _polarBoundLowLim, _polarBoundUpLim, _polOffset);
#endif // MPI_CUDA_UpAngFImme

#ifndef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUStartFlux, _GPUMPIFlux, _beta);
#endif // !MPI_CUDA_UpAngFImme
	}
}

__global__ void GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt_11(int _lay, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
	int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
    FP_PRECISION *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
    FP_PRECISION *_GPULinearExpTable,FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
    FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
	FP_PRECISION_GEO *_GPURTMWidth_Y, int _polarBoundLowLim,int _polarBoundUpLim, FP_PRECISION _twoOverLayWidth,\
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, int _polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC, \
	FP_PRECISION *_GPUMacroXS_t_wScale, FP_PRECISION *_GPUFSRAngScattSour, int _azimBegin, int _azimEnd)
{
    int _tid, _ltGlobID, _groupID, _azim, _azimNext, _ltIDNext, _polEngID, _pol, track_flux_index, _cell, _ray, _globCellID, _trackIndex;
    int _seg, _reg, _fsrID, _track, _index,_indexLay,_numPol,_ltInd, a;
    FP_PRECISION _length, _oneOverVolume, _oneOverVolbySpacing;
    M_SIZET _angId;
// #if defined(M_3DMOCex)
    FP_PRECISION _spacing;
    int _angId2,_angId3, _layFsrID;
    M_SIZET _angId1;
// #endif // defined(M_3DMOCex)
    /* Shared memory buffer for each thread's angular flux */
//#ifdef M_MG_CMFD
//    extern __shared__ FP_PRECISION temp_flux[];
//    FP_PRECISION *_track_flux = NULL;
//#else
    FP_PRECISION _trk_flux = 0.0;
//#endif
    FP_PRECISION _beta;



    FP_PRECISION _xs_t, _sigmaTByLength, _fsrFlux,_delta,_flatSour,_x2,_x3, _exp,_sourTerm,_scattSour=0.0,_3d_oneOverSigma;


#if defined(M_MG_CMFD)
    int _surfIndex_Layer_X, _surfIndex_Layer_Y,_firstTrackInLT;
	_surfIndex_Layer_X = ((*GPUMOCNumPinsInSubDom_X) + 1) * (*GPUMOCNumPinsInSubDom_Y);
	_surfIndex_Layer_Y = (*GPUMOCNumPinsInSubDom_X) * ((*GPUMOCNumPinsInSubDom_Y) + 1);
#endif // M_MG_CMFD

    _ltInd = _lay * (*GPUNumLTs_Total);
    _numPol=(*GPUNumPolarAng);

    // _numPol=1;

#if defined(M_3DMOCex)
    _numPol=_numPol/2;
#endif // defined(M_3DMOCex)

    _tid = threadIdx.x + blockIdx.x * blockDim.x;
    // tid = lt * NG * NP + g * NP + p
    int NGxNPol = (*GPUNumMOCGroups) * _numPol;
    _ltGlobID = _tid / (NGxNPol);
    _groupID = _tid % (NGxNPol) / _numPol;
    _pol = _tid % (NGxNPol) % _numPol;

    _polEngID = _groupID * _numPol;

//    track_flux_index = threadIdx.x * _numPol;
//    track_flux_index = threadIdx.x;

#if defined(M_2DMOC)
	const int _batchAzim = (_azimBegin >= 0)
		? (_azimEnd - _azimBegin) : *GPUNumAzimAng;
	_angId2 = (_azimBegin >= 0) ? _groupID*_batchAzim*(*GPUNumPolarAng) :
		_groupID*(*GPU_angMul2_FP);  // local P or full A x P
#else
    // #if defined(M_3DMOCex)
	_angId2 = _groupID*(*GPU_angMul2_HP);
#endif

    if (_ltGlobID < (*GPUNumLTs_Total)) {
        /* Retrieve pointer to thread's shared memory buffer for angular flux */

//        _track_flux = &temp_flux[track_flux_index];
//        _track_flux[_pol-_polOffset] = _GPUBoundaryFlux(/*_ltInd + */_ltGlobID, _pol + _polEngID);

        _trk_flux = _GPUBoundaryFlux(_ltGlobID, _pol + _polEngID);

        _azim = _GPULTsIndex_AzimAngID[_ltGlobID];
		if (_azimBegin >= 0 && (_azim < _azimBegin || _azim >= _azimEnd)) {
			return;
		}

// #if defined(M_3DMOCex)
        _angId3 = (_azimBegin >= 0) ? (_azim - _azimBegin) * _numPol : _azim*_numPol;
        _spacing = GPUEffRaySpacing[_azim];
// #endif // defined(M_3DMOCex)

#if defined(M_MG_CMFD)
        _firstTrackInLT = 0;
#endif // defined(M_MG_CMFD)
        /* Loop over each Track segment in forward direction */
        for (_track = _GPUTrackIndexInLTs[_ltGlobID]; _track < _GPUTrackIndexInLTs[_ltGlobID + 1]; _track++) {
            _cell = _GPUTracksInfoInLTs[_track].cell_ID;
            _ray = _GPUTracksInfoInLTs[_track].track_ID;
            _globCellID = _lay * _numCellsInDom + _cell;
            _trackIndex = (*GPUTotNumTracksInOneRTM_2PI)*_GPUCellRTMType_SubDom[_globCellID] + _GPUIndexTrackInRTM_SumAzim[_azim] + _ray;

            for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {
                _length = _GPUSegInfoInTrack_2PI_length[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];
                _reg = _GPUSegInfoInTrack_2PI_FSRID[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];

                _fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID] + _reg;
                // _oneOverVolume = /*(FP_PRECISION)(1.0) /*/ _GPUOneOverFSRsAreaInGlob[_fsrID];
                _oneOverVolume = _GPUOneOverFSRsAreaInRTM[_GPUSumNumFSRsInRTM[_GPUCellRTMType_SubDom[_globCellID]] + _reg];
// #if defined(M_3DMOCex)
                _oneOverVolbySpacing = _oneOverVolume * _spacing;
                _layFsrID = _GPUSumNumFSRsBasedGlobCellID[_cell] + _reg;
#if defined(M_2DMOC)
                if (_azimBegin >= 0) {
                    _angId1 = (M_SIZET)(_layFsrID) * (M_SIZET)(*GPUNumMOCGroups) *
                        (M_SIZET)(_batchAzim) * (M_SIZET)(*GPUNumPolarAng);
                }
                else {
                    _angId1 = (M_SIZET)(_layFsrID) * (M_SIZET)(*GPU_angMul1_FP);  // NG x NA x NP
                }
#else
                _angId1 = (M_SIZET)(_layFsrID) * (M_SIZET)(*GPU_angMul1_HP);
#endif
                _indexLay = _layFsrID * (*GPUNumMOCGroups) + _groupID;
// #endif //  defined(M_3DMOCex)
                _index = _fsrID * (*GPUNumMOCGroups) + _groupID;

#if defined(M_2DMOC)
                _indexLay = _index;
#endif // defined(M_2DMOC)

// #if defined(M_3DMOCex)
                _angId = _angId1 + (M_SIZET)(_angId2 + _angId3); // (fsr, g, azim, *)
// #endif //  defined(M_3DMOCex)

#if !defined(M_3DOpt)

                _xs_t = _GPUMacroXS_t[_indexLay];
                _flatSour = _GPUSource[_indexLay];
                _sigmaTByLength = _xs_t * _length;
                _3d_oneOverSigma = 1.0 / _xs_t;
                _fsrFlux = (FP_PRECISION) (0.0);
                _scattSour = _GPUFSRAngScattSour[_angId + _pol - _polOffset];
                _sourTerm = (_flatSour + _scattSour) * _3d_oneOverSigma;

                _exp = 1. - _ExpFunction_GPU(-_sigmaTByLength * GPUOneOverSinThetas[_pol]);


                _delta = (_trk_flux - _sourTerm) * _exp;
                _trk_flux -= _delta;
                _fsrFlux = _oneOverVolbySpacing * (_sourTerm * _length + GPUSinThetas[_pol] * _3d_oneOverSigma * _delta);
#ifdef DOUBLE_FLUX
                atomicAdd_(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
                atomicAdd(&_axialSurfAngFlux_update[_angId + _pol - _polOffset], (FP_PRECISION_FLUX) _fsrFlux);
#endif


#else
                // GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_opt2(_index, _azim, _length, _oneOverVolume, \
			    // 	_GPUMacroXS_t, _GPUMacroXS_t_wScale, _track_flux, _GPUSource, _GPUScalarFlux, _GPULinearExpTable, \
				// 	_polarBoundLowLim, _polarBoundUpLim, _polOffset,_oneOverVolbySpacing,\
				// 	_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update);
#endif //


            }

        }
        _azimNext = _GPULongTracksInfo[_ltGlobID].next_azim_angle_id;
        _ltIDNext = _GPULongTracksInfo[_ltGlobID].next_LongTrack_id_SubDom;
        _beta = _GPULongTracksInfo[_ltGlobID]._boundBeta;

#ifdef MPI_CUDA_UpAngFImme
        GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme_1(_ltInd,_azim, _azimNext, _pol, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _trk_flux, _GPUBoundaryFlux, _GPUMPIFlux, _beta, _polarBoundLowLim, _polarBoundUpLim, _polOffset);
#endif // MPI_CUDA_UpAngFImme

#ifndef MPI_CUDA_UpAngFImme
        GPUUpdateBoundaryFlux_AngFluxOnMem(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUStartFlux, _GPUMPIFlux, _beta);
#endif // !MPI_CUDA_UpAngFImme
    }
}


__global__ void GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt_2(int _lay, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
	int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
	FP_PRECISION *_GPUMacroXS_t,FP_PRECISION *_GPUMacroXS_t_inv, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
    FP_PRECISION *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
    FP_PRECISION *_GPULinearExpTable,FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
    FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
	FP_PRECISION_GEO *_GPURTMWidth_Y, int _polarBoundLowLim,int _polarBoundUpLim, FP_PRECISION _twoOverLayWidth,\
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, int _polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC, \
	FP_PRECISION *_GPUMacroXS_t_wScale, FP_PRECISION *_GPUFSRAngScattSour)
{
	int _tid, _ltGlobID, _groupID, _azim, _azimNext, _ltIDNext, _polEngID, _pol, track_flux_index, _cell, _ray, _globCellID, _trackIndex;
	int _seg, _reg, _fsrID, _track, _index,_indexLay,_numPol,_ltInd;
	FP_PRECISION _length, _oneOverVolume, _oneOverVolbySpacing;
	M_SIZET _angId;
// #if defined(M_3DMOCex)
	FP_PRECISION _spacing;
	int _angId2,_angId3, _layFsrID;
	M_SIZET _angId1;
// #endif // defined(M_3DMOCex)
	/* Shared memory buffer for each thread's angular flux */
	extern __shared__ FP_PRECISION temp_flux[];
	FP_PRECISION *_track_flux = NULL;
	FP_PRECISION _beta;

#if defined(M_MG_CMFD)
	int _surfIndex_Layer_X, _surfIndex_Layer_Y,_firstTrackInLT;
	_surfIndex_Layer_X = ((*GPUMOCNumPinsInSubDom_X) + 1) * (*GPUMOCNumPinsInSubDom_Y);
	_surfIndex_Layer_Y = (*GPUMOCNumPinsInSubDom_X) * ((*GPUMOCNumPinsInSubDom_Y) + 1);
#endif // M_MG_CMFD

    _ltInd = _lay * (*GPUNumLTs_Total);
	_numPol=(*GPUNumPolarAng);

	// _numPol=1;
	
#if defined(M_3DMOCex)
	_numPol=_numPol/2;
#endif // defined(M_3DMOCex)

	_tid = threadIdx.x + blockIdx.x * blockDim.x;
	_ltGlobID = _tid / (*GPUNumMOCGroups);
	_groupID = _tid % (*GPUNumMOCGroups);

	_polEngID = _groupID * (*GPUNumPolarAng);

	track_flux_index = threadIdx.x * _numPol;

#if defined(M_2DMOC)
    _angId2 = _groupID*(*GPU_angMul2_FP);
#else
	// #if defined(M_3DMOCex)
	_angId2 = _groupID*(*GPU_angMul2_HP);
#endif

	if (_ltGlobID < (*GPUNumLTs_Total)) {
		/* Retrieve pointer to thread's shared memory buffer for angular flux */
		_track_flux = &temp_flux[track_flux_index];
		
		for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
    		_track_flux[_pol-_polOffset] = _GPUBoundaryFlux(/*_ltInd + */_ltGlobID, _pol + _polEngID);
		}
		_azim = _GPULTsIndex_AzimAngID[_ltGlobID];

// #if defined(M_3DMOCex)
		_angId3 = _azim*_numPol;
		_spacing = GPUEffRaySpacing[_azim];
// #endif // defined(M_3DMOCex)

#if defined(M_MG_CMFD)
        _firstTrackInLT = 0;
#endif // defined(M_MG_CMFD)
		/* Loop over each Track segment in forward direction */
		for (_track = _GPUTrackIndexInLTs[_ltGlobID]; _track < _GPUTrackIndexInLTs[_ltGlobID + 1]; _track++) {
			_cell = _GPUTracksInfoInLTs[_track].cell_ID;
			_ray = _GPUTracksInfoInLTs[_track].track_ID;
			_globCellID = _lay * _numCellsInDom + _cell;
            _trackIndex = (*GPUTotNumTracksInOneRTM_2PI)*_GPUCellRTMType_SubDom[_globCellID] + _GPUIndexTrackInRTM_SumAzim[_azim] + _ray;
#if defined(M_MG_CMFD)
            if (_firstTrackInLT == 0) {
                GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
                    _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 1, \
                    _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
					_GPURTMWidth_X, _GPURTMWidth_Y, _polarBoundLowLim, _polarBoundUpLim,  _polOffset,_GPUTotalWeights_2DMOC);
            }
#endif // defined(M_MG_CMFD)
			for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {
				_length = _GPUSegInfoInTrack_2PI_length[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];
				_reg = _GPUSegInfoInTrack_2PI_FSRID[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];

				_fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID] + _reg;
				// _oneOverVolume = /*(FP_PRECISION)(1.0) /*/ _GPUOneOverFSRsAreaInGlob[_fsrID];
				_oneOverVolume = _GPUOneOverFSRsAreaInRTM[_GPUSumNumFSRsInRTM[_GPUCellRTMType_SubDom[_globCellID]] + _reg];
// #if defined(M_3DMOCex)
				_oneOverVolbySpacing = _oneOverVolume * _spacing;
				_layFsrID = _GPUSumNumFSRsBasedGlobCellID[_cell] + _reg;
#if defined(M_2DMOC)
                _angId1 = (M_SIZET)(_layFsrID) * (M_SIZET)(*GPU_angMul1_FP);
#else
				_angId1 = (M_SIZET)(_layFsrID) * (M_SIZET)(*GPU_angMul1_HP);
#endif
				_indexLay = _layFsrID * (*GPUNumMOCGroups) + _groupID;
// #endif //  defined(M_3DMOCex)
				_index = _fsrID * (*GPUNumMOCGroups) + _groupID;
				
#if defined(M_2DMOC)
				_indexLay = _index;
#endif // defined(M_2DMOC)

// #if defined(M_3DMOCex)
				_angId = _angId1 + (M_SIZET)(_angId2 + _angId3);
// #endif //  defined(M_3DMOCex)

#if !defined(M_3DOpt)
			    GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_HOScatt_2(_indexLay, _azim, _length, _oneOverVolume, \
			    	_GPUMacroXS_t,_GPUMacroXS_t_inv, _track_flux, _GPUSource, _GPUScalarFlux, _GPULinearExpTable, \
					_polarBoundLowLim, _polarBoundUpLim, _polOffset, _twoOverLayWidth,_oneOverVolbySpacing,\
					_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update,_GPUFSRAngScattSour);
#else
				// GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_opt2(_index, _azim, _length, _oneOverVolume, \
			    // 	_GPUMacroXS_t, _GPUMacroXS_t_wScale, _track_flux, _GPUSource, _GPUScalarFlux, _GPULinearExpTable, \
				// 	_polarBoundLowLim, _polarBoundUpLim, _polOffset,_oneOverVolbySpacing,\
				// 	_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update);
#endif // 
            }
#if defined(M_MG_CMFD)
            GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
                _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 0, \
                _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
                _GPURTMWidth_X, _GPURTMWidth_Y, _polarBoundLowLim, _polarBoundUpLim,  _polOffset,_GPUTotalWeights_2DMOC);
            _firstTrackInLT=1;
#endif // M_MG_CMFD
		}
		_azimNext = _GPULongTracksInfo[_ltGlobID].next_azim_angle_id;
		_ltIDNext = _GPULongTracksInfo[_ltGlobID].next_LongTrack_id_SubDom;
		_beta = _GPULongTracksInfo[_ltGlobID]._boundBeta;

#ifdef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme(_ltInd,_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUBoundaryFlux, _GPUMPIFlux, _beta, _polarBoundLowLim, _polarBoundUpLim, _polOffset);
#endif // MPI_CUDA_UpAngFImme

#ifndef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUStartFlux, _GPUMPIFlux, _beta);
#endif // !MPI_CUDA_UpAngFImme
	}
}



__global__ void GPUTS_Group_ExpFunc_OneDir_AngFSmem(int _lay, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
	int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
    FP_PRECISION *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
    FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
    FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
	FP_PRECISION_GEO *_GPURTMWidth_Y, int _polarBoundLowLim,int _polarBoundUpLim, FP_PRECISION _twoOverLayWidth,\
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, int _polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC, \
	FP_PRECISION *_GPUMacroXS_t_wScale, int _azimBegin, int _azimEnd)
{
	int _tid, _ltGlobID, _groupID, _azim, _azimNext, _ltIDNext, _polEngID, _pol, track_flux_index, _cell, _ray, _globCellID, _trackIndex;
	int _seg, _reg, _fsrID, _track, _index,_numPol,_ltInd, _indexLay=0;
	FP_PRECISION _length, _oneOverVolume, _oneOverVolbySpacing;
	M_SIZET _angId;
#if defined(M_3DMOCex)
	FP_PRECISION _spacing;
	int _angId2,_angId3, _layFsrID;
	M_SIZET _angId1;
#endif // defined(M_3DMOCex)
	/* Shared memory buffer for each thread's angular flux */
	extern __shared__ FP_PRECISION temp_flux[];
	FP_PRECISION *_track_flux = NULL;
	FP_PRECISION _beta;

#if defined(M_MG_CMFD)
	int _surfIndex_Layer_X, _surfIndex_Layer_Y,_firstTrackInLT;
	_surfIndex_Layer_X = ((*GPUMOCNumPinsInSubDom_X) + 1) * (*GPUMOCNumPinsInSubDom_Y);
	_surfIndex_Layer_Y = (*GPUMOCNumPinsInSubDom_X) * ((*GPUMOCNumPinsInSubDom_Y) + 1);
#endif // M_MG_CMFD

    _ltInd = _lay * (*GPUNumLTs_Total);
	_numPol=(*GPUNumPolarAng);

	// _numPol=1;
	
#if defined(M_3DMOCex)
	_numPol=_numPol/2;
#endif // defined(M_3DMOCex)

	_tid = threadIdx.x + blockIdx.x * blockDim.x;
	_ltGlobID = _tid / (*GPUNumMOCGroups);
	_groupID = _tid % (*GPUNumMOCGroups);

	_polEngID = _groupID * (*GPUNumPolarAng);

	track_flux_index = threadIdx.x * _numPol;

#if defined(M_3DMOCex)
	_angId2 = _groupID*(*GPU_angMul2_HP);
#endif // defined(M_3DMOCex)

	if (_ltGlobID < (*GPUNumLTs_Total)) {
		/* Retrieve pointer to thread's shared memory buffer for angular flux */
		_track_flux = &temp_flux[track_flux_index];
		
		for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
    		_track_flux[_pol-_polOffset] = _GPUBoundaryFlux(/*_ltInd+*/_ltGlobID, _pol + _polEngID);
		}
		_azim = _GPULTsIndex_AzimAngID[_ltGlobID];
		/* Keep the original full-angle behavior for _azimID < 0. */
		if (_azimBegin >= 0 && (_azim < _azimBegin || _azim >= _azimEnd)) {
			return;
		}

#if defined(M_3DMOCex)
		_angId3 = _azim*_numPol;
		_spacing = GPUEffRaySpacing[_azim];
#endif // defined(M_3DMOCex)

#if defined(M_MG_CMFD)
        _firstTrackInLT = 0;
#endif // defined(M_MG_CMFD)
		/* Loop over each Track segment in forward direction */
		for (_track = _GPUTrackIndexInLTs[_ltGlobID]; _track < _GPUTrackIndexInLTs[_ltGlobID + 1]; _track++) {
			_cell = _GPUTracksInfoInLTs[_track].cell_ID;
			_ray = _GPUTracksInfoInLTs[_track].track_ID;
			_globCellID = _lay * _numCellsInDom + _cell;
            _trackIndex = (*GPUTotNumTracksInOneRTM_2PI)*_GPUCellRTMType_SubDom[_globCellID] + _GPUIndexTrackInRTM_SumAzim[_azim] + _ray;
#if defined(M_MG_CMFD)
            if (_firstTrackInLT == 0) {
                GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
                    _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 1, \
                    _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
					_GPURTMWidth_X, _GPURTMWidth_Y, _polarBoundLowLim, _polarBoundUpLim,  _polOffset,_GPUTotalWeights_2DMOC);
            }
#endif // defined(M_MG_CMFD)
			for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {
				_length = _GPUSegInfoInTrack_2PI_length[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];
				_reg = _GPUSegInfoInTrack_2PI_FSRID[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];

				_fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID] + _reg;
				// _oneOverVolume = /*(FP_PRECISION)(1.0) /*/ _GPUOneOverFSRsAreaInGlob[_fsrID];
				_oneOverVolume = (FP_PRECISION)(_GPUOneOverFSRsAreaInRTM[_GPUSumNumFSRsInRTM[_GPUCellRTMType_SubDom[_globCellID]] + _reg]);
#if defined(M_3DMOCex)
				_oneOverVolbySpacing = _oneOverVolume * _spacing;
				_layFsrID = _GPUSumNumFSRsBasedGlobCellID[_cell] + _reg;
				_angId1 = (M_SIZET)(_layFsrID) * (M_SIZET)(*GPU_angMul1_HP);
				_indexLay = _layFsrID * (*GPUNumMOCGroups) + _groupID;
#endif //  defined(M_3DMOCex)
				_index = _fsrID * (*GPUNumMOCGroups) + _groupID;
#if defined(M_2DMOC)
				_indexLay = _index;
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
				_angId = _angId1 + (M_SIZET)(_angId2 + _angId3);
#endif //  defined(M_3DMOCex)

			    GPUTallyScalarFlux_ExpFunc_OneDir_AngFSmem(_index, _indexLay, _azim, _length, _oneOverVolume, \
			    	_GPUMacroXS_t, _track_flux, _GPUSource, _GPUScalarFlux, \
					_polarBoundLowLim, _polarBoundUpLim, _polOffset, _twoOverLayWidth,_oneOverVolbySpacing,\
					_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update);

				// GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_opt2(_index, _azim, _length, _oneOverVolume, \
			    // 	_GPUMacroXS_t, _GPUMacroXS_t_wScale, _track_flux, _GPUSource, _GPUScalarFlux, _GPULinearExpTable, \
				// 	_polarBoundLowLim, _polarBoundUpLim, _polOffset,_oneOverVolbySpacing,\
				// 	_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update);
            }
#if defined(M_MG_CMFD)
            GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
                _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 0, \
                _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
                _GPURTMWidth_X, _GPURTMWidth_Y, _polarBoundLowLim, _polarBoundUpLim,  _polOffset,_GPUTotalWeights_2DMOC);
            _firstTrackInLT=1;
#endif // M_MG_CMFD
		}
		_azimNext = _GPULongTracksInfo[_ltGlobID].next_azim_angle_id;
		_ltIDNext = _GPULongTracksInfo[_ltGlobID].next_LongTrack_id_SubDom;
		_beta = _GPULongTracksInfo[_ltGlobID]._boundBeta;

#ifdef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme(_ltInd,_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUBoundaryFlux, _GPUMPIFlux, _beta, _polarBoundLowLim, _polarBoundUpLim, _polOffset);
#endif // MPI_CUDA_UpAngFImme

#ifndef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUStartFlux, _GPUMPIFlux, _beta);
#endif // !MPI_CUDA_UpAngFImme
	}
}


/**
* @ Function: GPUAccumAngFlux_ScalarFlux()
*   introduction: accumulate the average angular flux to scalar flux by numerical integration
* @ Parameter List: 
*   Return: void
* @ Created     : 2019/12/24, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
__global__ void GPUAccumAngFlux_ScalarFlux(int _lay, int *_GPUNumFSRs_Layer, int *_GPUSumNumFSRs_Layer, \
    int _polarBoundLowLim, int _polarBoundUpLim, FP_PRECISION_FLUX *_aveAngFlux, FP_PRECISION_FLUX *_scalarFlux, \
	int _polOffset)
{
	int _tid = threadIdx.x + blockIdx.x * blockDim.x, _fsrID, _groupID;
	int _layFSR, _numFSRs_lay,_fsrID_lay,_azim,_pol,_ind;
	int _angId2,_angId3,_angId4,_halfPol;
	M_SIZET _angId1;
	FP_PRECISION_FLUX _flux, _zero=(FP_PRECISION_FLUX)(0.0);

	_fsrID_lay = _tid / (*GPUNumMOCGroups);
	_groupID = _tid % (*GPUNumMOCGroups);

	_numFSRs_lay=_GPUNumFSRs_Layer[_lay];
	_layFSR=_GPUSumNumFSRs_Layer[_lay];
	_halfPol=(*GPUNumPolarAng)/2;
	// _halfPol=3;

	while(_fsrID_lay < _numFSRs_lay) {

		_angId1=(M_SIZET)(_fsrID_lay)*(M_SIZET)(*GPU_angMul1_HP);
		_angId2=_groupID*(*GPU_angMul2_HP);

		_fsrID=_layFSR+_fsrID_lay;
		_ind = _fsrID*(*GPUNumMOCGroups)+_groupID;
		_flux=_zero;
		for (_azim=0;_azim<(*GPUNumAzimAng);_azim++) {
			_angId3=_azim*_halfPol;
			_angId4=_azim*(*GPUNumPolarAng);
			for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
				_flux += _aveAngFlux[_angId1 + (M_SIZET)(_angId2 + _angId3 + _pol - _polOffset)] * GPUTotalWeights[_angId4 + _pol];
			}
		}
		_scalarFlux[_ind] += _flux;
#if defined(M_ZeroNegFlux)
		if (_scalarFlux[_ind]<0.0) {
			_scalarFlux[_ind]=0.0;
		}
#endif
		_tid += blockDim.x * gridDim.x;
		_fsrID_lay = _tid / (*GPUNumMOCGroups);
		_groupID = _tid % (*GPUNumMOCGroups);
	}
}

__global__ void GPUAccumAngFlux_ScalarFlux_HOScatt(int _lay, int *_GPUNumFSRs_Layer, int _polarBoundLowLim, \
	int _polarBoundUpLim, FP_PRECISION_FLUX *_aveAngFlux, FP_PRECISION_FLUX *_scalarFlux, int _polOffset, \
	FP_PRECISION *_GPUSpheHarmCoeff)
{
	int _tid = threadIdx.x + blockIdx.x * blockDim.x, _groupID;
	int _numFSRs_lay,_fsrID_lay,_azim,_pol,_ind;
	int _angId2,_angId3,_angId4, l,m,_ind1,_mul1,_count;
	M_SIZET _angId1;
	FP_PRECISION_FLUX _flux, _zero=(FP_PRECISION_FLUX)(0.0);
	int _mul, _angMul1, _angMul2;
#if defined(M_2DMOC)
	_mul = (*GPUNumPolarAng);
	_angMul1 = (*GPU_angMul1_FP);  // NumAzimAng * NumPolarAng * NumGroups
	_angMul2 = (*GPU_angMul2_FP);  // NumAzimAng * NumPolarAng
#else
	_mul = (*GPUNumPolarAng)/2;
	_angMul1 = (*GPU_angMul1_HP);
	_angMul2 = (*GPU_angMul2_HP);
#endif
	_fsrID_lay = _tid / (*GPUNumMOCGroups);
	_groupID = _tid % (*GPUNumMOCGroups);

	_numFSRs_lay=_GPUNumFSRs_Layer[_lay];
	_mul1 = (*GPUNumAzimAng) * (*GPUNumPolarAng);

	while(_fsrID_lay < _numFSRs_lay) {

		_angId1=(M_SIZET)(_fsrID_lay)*(M_SIZET)(_angMul1);
		_angId2=_groupID*(_angMul2);
		_count=0;
		for (l=0;l<=(*GPUNumScattOrder);l++) {
			for (m=-l;m<=l;m++) {
				_ind = _count * (*GPUNumFSRsLayer) * (*GPUNumMOCGroups) + _fsrID_lay * (*GPUNumMOCGroups) + _groupID;

                // accumulate scalar flux moment
				_flux=_zero;
				for (_azim=0;_azim<(*GPUNumAzimAng);_azim++) {
					_angId3=_azim*_mul;
					_angId4=_azim*(*GPUNumPolarAng);
					for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
						_ind1 = _count * _mul1 + _azim * (*GPUNumPolarAng) + _pol;
						_flux += _aveAngFlux[_angId1 + (M_SIZET)(_angId2 + _angId3 + _pol - _polOffset)] * \
							GPUTotalWeights[_angId4 + _pol] * _GPUSpheHarmCoeff[_ind1];
					}
				}

				_scalarFlux[_ind] = _flux;
		#if defined(M_ZeroNegFlux)
				if (_scalarFlux[_ind]<0.0) {
					_scalarFlux[_ind]=0.0;
				}
		#endif
		        _count++;
			}
		}

		_tid += blockDim.x * gridDim.x;
		_fsrID_lay = _tid / (*GPUNumMOCGroups);
		_groupID = _tid % (*GPUNumMOCGroups);
	}
}


// parallel by FSR-Group-Order, no atomic
__global__ void GPUAccumAngFlux_ScalarFlux_HOScatt_opt(int _lay, int *_GPUNumFSRs_Layer, int _polarBoundLowLim, \
	int _polarBoundUpLim, FP_PRECISION_FLUX *_aveAngFlux, FP_PRECISION_FLUX *_scalarFlux, int _polOffset, \
	FP_PRECISION *_GPUSpheHarmCoeff)
{
    size_t _tid = threadIdx.x + blockIdx.x * blockDim.x;
    int _groupID;
    int _numFSRs_lay,_fsrID_lay,_azim,_pol,_ind;
    int _angId2,_angId3,_angId4, l,m,_ind1,_mul1,_count, torder;
    size_t _angId1;
    FP_PRECISION_FLUX _flux, _zero=(FP_PRECISION_FLUX)(0.0);
    int _mul, _angMul1, _angMul2;

#if defined(M_2DMOC)
    _mul = (*GPUNumPolarAng);
    _angMul1 = (*GPU_angMul1_FP);  // NumAzimAng * NumPolarAng * NumGroups
    _angMul2 = (*GPU_angMul2_FP);  // NumAzimAng * NumPolarAng
#else
    _mul = (*GPUNumPolarAng)/2;
	_angMul1 = (*GPU_angMul1_HP);
	_angMul2 = (*GPU_angMul2_HP);
#endif
    _numFSRs_lay=_GPUNumFSRs_Layer[_lay];
    size_t NFSRxNG = (*GPUNumMOCGroups) * (*GPUNumFSRsLayer);
    _count = _tid / NFSRxNG;
    _fsrID_lay = (_tid % NFSRxNG) / (*GPUNumMOCGroups);
    _groupID = (_tid % NFSRxNG) % (*GPUNumMOCGroups);

    while(_count < (*GPUNumScattOrder+1)*(*GPUNumScattOrder+1)) {
        _angId1=(_fsrID_lay)*(_angMul1);
        _angId2=_groupID*(_angMul2);

        // accumulate scalar flux moment
        _flux=_zero;
        for (_azim=0;_azim<(*GPUNumAzimAng);_azim++) {
            _angId3=_azim*_mul;
            _angId4=_azim*(*GPUNumPolarAng);
            for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
                _ind1 = _count * _angMul2 + _azim * (*GPUNumPolarAng) + _pol;
                _flux += _aveAngFlux[_angId1 + (M_SIZET)(_angId2 + _angId3 + _pol - _polOffset)] * \
                    GPUTotalWeights[_angId4 + _pol] * _GPUSpheHarmCoeff[_ind1];
            }
        }

        _scalarFlux[_tid] = _flux;
#if defined(M_ZeroNegFlux)
                if (_scalarFlux[_ind]<0.0) {
					_scalarFlux[_ind]=0.0;
				}
#endif

        _tid += blockDim.x * gridDim.x;
        _count = _tid / NFSRxNG;
        _fsrID_lay = (_tid % NFSRxNG) / (*GPUNumMOCGroups);
        _groupID = (_tid % NFSRxNG) % (*GPUNumMOCGroups);
    }
}

// parallel by FSR-Group-Order, no atomic
__global__ void GPUAccumAngFlux_ScalarFlux_HOScatt_opt_sld(int _lay, int *_GPUNumFSRs_Layer, int _polarBoundLowLim, \
	int _polarBoundUpLim, FP_PRECISION_FLUX *_aveAngFlux, FP_PRECISION_FLUX *_scalarFlux, int _polOffset, \
	FP_PRECISION *_GPUSpheHarmCoeff, int sldOrder)
{
    size_t _tid = threadIdx.x + blockIdx.x * blockDim.x;
    int _groupID;
    int _numFSRs_lay,_fsrID_lay,_azim,_pol,_ind;
    int _angId2,_angId3,_angId4, l,m,_ind1,_mul1,_count, torder;
    size_t _angId1;
    FP_PRECISION_FLUX _flux, _zero=(FP_PRECISION_FLUX)(0.0);
    int _mul, _angMul1, _angMul2;

#if defined(M_2DMOC)
    _mul = (*GPUNumPolarAng);
    _angMul1 = (*GPU_angMul1_FP);  // NumAzimAng * NumPolarAng * NumGroups
    _angMul2 = (*GPU_angMul2_FP);  // NumAzimAng * NumPolarAng
#else
    _mul = (*GPUNumPolarAng)/2;
	_angMul1 = (*GPU_angMul1_HP);
	_angMul2 = (*GPU_angMul2_HP);
#endif
    _numFSRs_lay=_GPUNumFSRs_Layer[_lay];
    size_t NFSRxNG = (*GPUNumMOCGroups) * (*GPUNumFSRsLayer);
    _count = _tid / NFSRxNG;
    _fsrID_lay = (_tid % NFSRxNG) / (*GPUNumMOCGroups);
    _groupID = (_tid % NFSRxNG) % (*GPUNumMOCGroups);

    while(_count < (sldOrder+1)*(sldOrder+1)) {
        _angId1=(_fsrID_lay)*(_angMul1);
        _angId2=_groupID*(_angMul2);

        // accumulate scalar flux moment
        _flux=_zero;
        for (_azim=0;_azim<(*GPUNumAzimAng);_azim++) {
            _angId3=_azim*_mul;
            _angId4=_azim*(*GPUNumPolarAng);
            for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
                _ind1 = _count * _angMul2 + _azim * (*GPUNumPolarAng) + _pol;
                _flux += _aveAngFlux[_angId1 + (M_SIZET)(_angId2 + _angId3 + _pol - _polOffset)] * \
                    GPUTotalWeights[_angId4 + _pol] * _GPUSpheHarmCoeff[_ind1];
            }
        }

        _scalarFlux[_tid] = _flux;
#if defined(M_ZeroNegFlux)
                if (_scalarFlux[_ind]<0.0) {
					_scalarFlux[_ind]=0.0;
				}
#endif

        _tid += blockDim.x * gridDim.x;
        _count = _tid / NFSRxNG;
        _fsrID_lay = (_tid % NFSRxNG) / (*GPUNumMOCGroups);
        _groupID = (_tid % NFSRxNG) % (*GPUNumMOCGroups);
    }
}

// parallel by FSR-Group-Order-Azim, atomic
__global__ void GPUAccumAngFlux_ScalarFlux_HOScatt_opt2(int _lay, int *_GPUNumFSRs_Layer, int _polarBoundLowLim, \
	int _polarBoundUpLim, FP_PRECISION_FLUX *_aveAngFlux, FP_PRECISION_FLUX *_scalarFlux, int _polOffset, \
	FP_PRECISION *_GPUSpheHarmCoeff)
{
    int _tid = threadIdx.x + blockIdx.x * blockDim.x, _groupID;
    int _numFSRs_lay,_fsrID_lay,_azim,_pol,_ind;
    int _angId2,_angId3,_angId4, l,m,_ind1,_mul1,_count, torder;
    M_SIZET _angId1;
    FP_PRECISION_FLUX _flux, _zero=(FP_PRECISION_FLUX)(0.0);
    int _mul, _angMul1, _angMul2;

    int L[25] = {0,
                 1, 1, 1,
                 2, 2, 2, 2, 2,
                 3, 3, 3, 3, 3, 3, 3,
                 4, 4, 4, 4, 4, 4, 4, 4, 4};
    int M[25] = {0,
                 -1, 0, 1,
                 -2, -1, 0, 1, 2,
                 -3, -2, -1, 0, 1, 2, 3,
                 -4, -3, -2, -1, 0, 1, 2, 3, 4};

#if defined(M_2DMOC)
    _mul = (*GPUNumPolarAng);
    _angMul1 = (*GPU_angMul1_FP);  // NumAzimAng * NumPolarAng * NumGroups
    _angMul2 = (*GPU_angMul2_FP);  // NumAzimAng * NumPolarAng
#else
    _mul = (*GPUNumPolarAng)/2;
	_angMul1 = (*GPU_angMul1_HP);
	_angMul2 = (*GPU_angMul2_HP);
#endif
    // dimX = NumAzim
    // dimY = (NumOrder + 1)^2
    // dimZ = NumGroups
    // dimH = NumFSRs
    // tid = dimX*Y*Z * h + dimX*Y * z + dimX * y + x
    int NOrder2 = (*GPUNumScattOrder + 1) * (*GPUNumScattOrder + 1);
    int NAzimxNOrder2 = (*GPUNumAzimAng) * NOrder2;
    int NAzimxNOrder2xNG = (*GPUNumMOCGroups) * NAzimxNOrder2;

    _fsrID_lay = _tid / NAzimxNOrder2xNG;
    _groupID = _tid % NAzimxNOrder2xNG / NAzimxNOrder2;
    torder = _tid % NAzimxNOrder2xNG % NAzimxNOrder2 / (*GPUNumAzimAng);
    _azim = _tid % NAzimxNOrder2xNG % NAzimxNOrder2 % (*GPUNumAzimAng);
    l = L[torder];
    m = M[torder];

    _numFSRs_lay=_GPUNumFSRs_Layer[_lay];
    _mul1 = (*GPUNumAzimAng) * (*GPUNumPolarAng);

    while(_fsrID_lay < _numFSRs_lay) {

        _angId1=(M_SIZET)(_fsrID_lay)*(M_SIZET)(_angMul1);
        _angId2=_groupID*(_angMul2);
        _ind = torder * (*GPUNumFSRsLayer) * (*GPUNumMOCGroups) + _fsrID_lay * (*GPUNumMOCGroups) + _groupID;

        // accumulate scalar flux moment
        _flux=_zero;

        _angId3 = _azim * _mul;
        _angId4 = _azim * (*GPUNumPolarAng);
        for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
            _ind1 = torder * _mul1 + _azim * (*GPUNumPolarAng) + _pol;
            _flux += _aveAngFlux[_angId1 + (M_SIZET) (_angId2 + _angId3 + _pol - _polOffset)] * \
                    GPUTotalWeights[_angId4 + _pol] * _GPUSpheHarmCoeff[_ind1];
        }
#ifdef FP32
        atomicAdd(&_scalarFlux[_ind], _flux);
#endif
#ifdef FP64
        atomicAdd_(&_scalarFlux[_ind], _flux);
#endif

#if defined(M_ZeroNegFlux)
        if (_scalarFlux[_ind]<0.0) {
					_scalarFlux[_ind]=0.0;
				}
#endif

        _tid += blockDim.x * gridDim.x;
        _fsrID_lay = _tid / NAzimxNOrder2xNG;
        _groupID = _tid % NAzimxNOrder2xNG / NAzimxNOrder2;
        torder = _tid % NAzimxNOrder2xNG % NAzimxNOrder2 / (*GPUNumAzimAng);
        _azim = _tid % NAzimxNOrder2xNG % NAzimxNOrder2 % (*GPUNumAzimAng);
        l = L[torder];
        m = M[torder];
    }
}

// parallel by FSR-Group-Order-Pol, atomic
__global__ void GPUAccumAngFlux_ScalarFlux_HOScatt_opt3(int _lay, int *_GPUNumFSRs_Layer, int _polarBoundLowLim, \
	int _polarBoundUpLim, FP_PRECISION_FLUX *_aveAngFlux, FP_PRECISION_FLUX *_scalarFlux, int _polOffset, \
	FP_PRECISION *_GPUSpheHarmCoeff)
{
    int _tid = threadIdx.x + blockIdx.x * blockDim.x, _groupID;
    int _numFSRs_lay,_fsrID_lay,_azim,_pol,_ind;
    int _angId2,_angId3,_angId4, l,m,_ind1,_mul1,_count, torder;
    M_SIZET _angId1;
    FP_PRECISION_FLUX _flux, _zero=(FP_PRECISION_FLUX)(0.0);
    int _mul, _angMul1, _angMul2;

#if defined(M_2DMOC)
    _mul = (*GPUNumPolarAng);
    _angMul1 = (*GPU_angMul1_FP);  // NumAzimAng * NumPolarAng * NumGroups
    _angMul2 = (*GPU_angMul2_FP);  // NumAzimAng * NumPolarAng
#else
    _mul = (*GPUNumPolarAng)/2;
	_angMul1 = (*GPU_angMul1_HP);
	_angMul2 = (*GPU_angMul2_HP);
#endif
    // dimX = NumPol
    // dimY = (NumOrder + 1)^2
    // dimZ = NumGroups
    // dimH = NumFSRs
    // tid = dimX*Y*Z * h + dimX*Y * z + dimX * y + x

    int NOrder2 = (*GPUNumScattOrder + 1) * (*GPUNumScattOrder + 1);
    size_t NFSRxNGxNPol = (*GPUNumMOCGroups) * (*GPUNumFSRsLayer) * (*GPUNumPolarAng);
    int NGxNPol = (*GPUNumPolarAng) * (*GPUNumMOCGroups);
    torder = _tid / NFSRxNGxNPol;
    _fsrID_lay = (_tid % NFSRxNGxNPol) / NGxNPol;
    _groupID = (_tid % NFSRxNGxNPol) % NGxNPol;
    _pol = _tid % NFSRxNGxNPol % NGxNPol % (*GPUNumPolarAng);

//    __shared__ FP_PRECISION a[4*64*6];
//
//    for (int o = 0; o < 4; o++) {
//        for (_azim=0; _azim < (*GPUNumAzimAng); _azim++) {
//            for (int p=0; p < 6; p++) {
//                _ind1 = o * _angMul2 + _azim * (*GPUNumPolarAng) + p;
//                a[_ind1] = _GPUSpheHarmCoeff[_ind1];
//            }
//        }
//    }

    while(_fsrID_lay < *GPUNumFSRsLayer) {
        _angId1=(M_SIZET)(_fsrID_lay)*(M_SIZET)(_angMul1);
        _angId2=_groupID*(_angMul2);
        _ind = torder * (*GPUNumFSRsLayer) * (*GPUNumMOCGroups) + _fsrID_lay * (*GPUNumMOCGroups) + _groupID;

        // accumulate scalar flux moment
        _flux=_zero;

        for (_azim = 0; _azim < (*GPUNumAzimAng); _azim++) {
            _angId3 = _azim * _mul;
            _angId4 = _azim * (*GPUNumPolarAng);
            _ind1 = torder * _angMul2 + _azim * (*GPUNumPolarAng) + _pol;
            _flux += _aveAngFlux[_angId1 + (M_SIZET) (_angId2 + _angId3 + _pol - _polOffset)] * \
                    GPUTotalWeights[_angId4 + _pol] * _GPUSpheHarmCoeff[_ind1];
        }
#ifdef SINGLE_FLUX
        atomicAdd(&_scalarFlux[_ind], _flux);
#endif
#ifdef DOUBLE_FLUX
        atomicAdd_(&_scalarFlux[_ind], _flux);
#endif

#if defined(M_ZeroNegFlux)
        if (_scalarFlux[_ind]<0.0) {
					_scalarFlux[_ind]=0.0;
				}
#endif

        _tid += blockDim.x * gridDim.x;

        torder = _tid / NFSRxNGxNPol;
        _fsrID_lay = (_tid % NFSRxNGxNPol) / NGxNPol;
        _groupID = (_tid % NFSRxNGxNPol) % NGxNPol;
        _pol = _tid % NFSRxNGxNPol % NGxNPol % (*GPUNumPolarAng);
    }
}





/**
* @ Function: GPUUpdate_AxialSurfFlux()
*   introduction: update the axial surface angular flux
* @ Parameter List: 
*   Return: void
* @ Created     : 2019/12/24, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
__global__ void GPUUpdate_AxialSurfFlux(int _dim, FP_PRECISION *_aveAngFlux_inp, FP_PRECISION *_aveAngFlux_update)
{
	FP_PRECISION _two=(FP_PRECISION)(2.0), _temp;
	int _tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (_tid<_dim) {
		_temp = _two * _aveAngFlux_update[_tid] - _aveAngFlux_inp[_tid];
		if (_temp < 0.0) {
			_temp = 0.0;
		}
		_aveAngFlux_inp[_tid] = _temp;
		_aveAngFlux_update[_tid]=0.0;
		_tid += blockDim.x * gridDim.x;
	}
}

__global__ void GPUUpdate_AxialSurfFlux_1(int _dim, FP_PRECISION *_aveAngFlux_inp, FP_PRECISION *_aveAngFlux_update,\
	FP_PRECISION _twoOverLayWidth, FP_PRECISION *_GPUFSRAngScattSour, int _polarBoundLowLim, int _polarBoundUpLim, int _polOffset)
{
	FP_PRECISION _two=(FP_PRECISION)(2.0), _temp, _temp1;
	int _ind,_pol,_ind1;
	int _tid = threadIdx.x + blockIdx.x * blockDim.x;
	FP_PRECISION _scale;

	while (_tid<_dim) {
		_ind = _tid * (_polarBoundUpLim-_polarBoundLowLim);

		for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
			_scale = _twoOverLayWidth*GPUCosThetas[_pol];
			_ind1=_ind + _pol - _polOffset;
			_temp1 = (_aveAngFlux_inp[_ind1] - _GPUFSRAngScattSour[_ind1]) / _scale;

			_temp = _two * _aveAngFlux_update[_ind1] - _temp1;

			if (_temp < 0.0) {
				_temp = 0.0;
			}
			_aveAngFlux_inp[_ind1] = _temp;
			_aveAngFlux_update[_ind1]=0.0;
		}

		_tid += blockDim.x * gridDim.x;
	}
}

/**
* @ Function: GPUUpdate_AxialBoundCond()
*   introduction: update the axial surface angular flux
* @ Parameter List: _dim = TotalNumFSRs_Layer * NumGroups * NumAzims
*   Return: void
* @ Created     : 2019/12/26, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
__global__ void GPUUpdate_AxialBoundCond(FP_PRECISION _beta, int _dim, int _halfPol, \
	FP_PRECISION *_aveAngFlux_inp, FP_PRECISION *_aveAngFlux_bound)
{
	int _ind, _temp,_pol,_ind1,_ind2;
	int _tid = threadIdx.x + blockIdx.x * blockDim.x;
	_temp=_halfPol-1;

	while (_tid<_dim) {
		_ind = _tid * _halfPol;

		for (_pol = 0; _pol < _halfPol; _pol++) {
			_ind1=_ind + _pol;
			_ind2=_ind + _temp - _pol;
			_aveAngFlux_bound[_ind2] = _beta * _aveAngFlux_inp[_ind1];
			_aveAngFlux_inp[_ind1]=0.0;
		}
		_tid += blockDim.x * gridDim.x;
	}
}

__global__ void GPUManipulateAxialInciAngFlux(FP_PRECISION _twoOverLayWidth, int _dim, \
	FP_PRECISION *_aveAngFlux_inp, FP_PRECISION *_GPUFSRAngScattSour, \
	int _polarBoundLowLim, int _polarBoundUpLim, int _polOffset)
{
	int _ind,_pol,_ind1;
	int _tid = threadIdx.x + blockIdx.x * blockDim.x;
    FP_PRECISION _scale;
	

	while (_tid<_dim) {
		_ind = _tid * (_polarBoundUpLim - _polarBoundLowLim);

		for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
			_scale = _twoOverLayWidth*GPUCosThetas[_pol];
			_ind1=_ind + _pol - _polOffset;
			_aveAngFlux_inp[_ind1] = _GPUFSRAngScattSour[_ind1] + _scale * _aveAngFlux_inp[_ind1];
		}
		_tid += blockDim.x * gridDim.x;
	}
}

__global__ void GPUComputeSourceOverSigma_3DMOCex(int _lay, int *_GPUNumFSRs_Layer, int *_GPUSumNumFSRs_Layer, \
    int _polarBoundLowLim, int _polarBoundUpLim, FP_PRECISION_FLUX *_axialSurfAnfFlux_inp, \
	int _polOffset, FP_PRECISION _twoOverLayWidth, FP_PRECISION *_GPUMacroXS_t_wScale, FP_PRECISION *_GPUSource)
{
	int _tid = threadIdx.x + blockIdx.x * blockDim.x, _fsrID, _groupID;
	int _layFSR, _numFSRs_lay,_fsrID_lay,_azim,_pol,_ind;
	int _angId2,_angId3,_halfPol;
	M_SIZET _angId1,_angId;
	FP_PRECISION _xs_t, _flatSour,_3d_Q,_scale;

	_fsrID_lay = _tid / (*GPUNumMOCGroups);
	_groupID = _tid % (*GPUNumMOCGroups);

	_numFSRs_lay=_GPUNumFSRs_Layer[_lay];

	while(_fsrID_lay < _numFSRs_lay) {

		_layFSR=_GPUSumNumFSRs_Layer[_lay];
		_halfPol=(*GPUNumPolarAng)/2;

		_angId1=(M_SIZET)(_fsrID_lay)*(M_SIZET)(*GPU_angMul1_HP);
		_angId2=_groupID*(*GPU_angMul2_HP);

		_fsrID=_layFSR+_fsrID_lay;
		_ind = _fsrID*(*GPUNumMOCGroups)+_groupID;

		_flatSour = _GPUSource[_ind];

		for (_azim=0;_azim<(*GPUNumAzimAng);_azim++) {
			_angId3=_azim*_halfPol;
			_angId = _angId1 + (M_SIZET)(_angId2 + _angId3);

			for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
				_scale=_twoOverLayWidth*GPUCosThetas[_pol];
				_xs_t = _GPUMacroXS_t_wScale[_ind*(*GPUNumPolarAng) + _pol];
				_3d_Q = _flatSour+_scale*_axialSurfAnfFlux_inp[_angId + _pol - _polOffset];
				_axialSurfAnfFlux_inp[_angId + (M_SIZET)(_pol - _polOffset)] = _3d_Q/_xs_t;
			}
		}
		_tid += blockDim.x * gridDim.x;
		_fsrID_lay = _tid / (*GPUNumMOCGroups);
		_groupID = _tid % (*GPUNumMOCGroups);
	}
}


__global__ void GPUCompAxialAngF_FromSourceOverSigma_3DMOCex(int _lay, int *_GPUNumFSRs_Layer, int *_GPUSumNumFSRs_Layer, \
    int _polarBoundLowLim, int _polarBoundUpLim, FP_PRECISION_FLUX *_axialSurfAnfFlux_inp, \
	int _polOffset, FP_PRECISION _twoOverLayWidth, FP_PRECISION *_GPUMacroXS_t_wScale, FP_PRECISION *_GPUSource)
{
	int _tid = threadIdx.x + blockIdx.x * blockDim.x, _fsrID, _groupID;
	int _layFSR, _numFSRs_lay,_fsrID_lay,_azim,_pol,_ind;
	int _angId2,_angId3,_halfPol;
	M_SIZET _angId1,_angId;
	FP_PRECISION _xs_t, _flatSour, _temp,_scale;

	_fsrID_lay = _tid / (*GPUNumMOCGroups);
	_groupID = _tid % (*GPUNumMOCGroups);

	_numFSRs_lay=_GPUNumFSRs_Layer[_lay];

	while(_fsrID_lay < _numFSRs_lay) {

		_layFSR=_GPUSumNumFSRs_Layer[_lay];
		_halfPol=(*GPUNumPolarAng)/2;

		_angId1=(M_SIZET)(_fsrID_lay)*(M_SIZET)(*GPU_angMul1_HP);
		_angId2=_groupID*(*GPU_angMul2_HP);

		_fsrID=_layFSR+_fsrID_lay;
		_ind = _fsrID*(*GPUNumMOCGroups)+_groupID;

		_flatSour = _GPUSource[_ind];

		for (_azim=0;_azim<(*GPUNumAzimAng);_azim++) {
			_angId3=_azim*_halfPol;
			_angId = _angId1 + (M_SIZET)(_angId2 + _angId3);

			for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
				_scale=_twoOverLayWidth*GPUCosThetas[_pol];
				_xs_t = _GPUMacroXS_t_wScale[_ind*(*GPUNumPolarAng) + _pol];
				_temp = _axialSurfAnfFlux_inp[_angId + (M_SIZET)(_pol - _polOffset)] * _xs_t - _flatSour;
				_axialSurfAnfFlux_inp[_angId + (M_SIZET)(_pol - _polOffset)] = _temp/_scale;
			}
		}
		_tid += blockDim.x * gridDim.x;
		_fsrID_lay = _tid / (*GPUNumMOCGroups);
		_groupID = _tid % (*GPUNumMOCGroups);
	}
}

__device__ void GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_FixSourProb_2D(int _index, int _indexLay, int _azim, FP_PRECISION _length, \
	FP_PRECISION _oneOverVolume, FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *_track_flux, FP_PRECISION *_GPUSource, \
	FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPULinearExpTable, int _polarBoundLowLim,int _polarBoundUpLim, \
	int _polOffset, FP_PRECISION _twoOverWidth, FP_PRECISION _oneOverVolbySpacing,int _angId,FP_PRECISION *_axialSurfAnfFlux_inp, \
	FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_GPUTotalWeights_2DMOC)
{
	int _x1, _pol;
	FP_PRECISION _xs_t, _sigmaTByLength, _fsrFlux,_delta,_flatSour,_x2,_x3, _exp,_sourTerm;

	FP_PRECISION _mul=1.0;
#if defined(M_2DMOC)
    _mul=1.0;
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
    _mul=2.0;
#endif // defined(M_3DMOCex)
// #if defined(M_3DMOCex)
// 	FP_PRECISION _3d_sigma, _3d_oneOverSigma, _3d_Q,_scale;
// #endif // defined(M_3DMOCex)
	_xs_t = _GPUMacroXS_t[_indexLay];
	_flatSour = _GPUSource[_indexLay];

// #if defined(M_2DMOC)
	_sigmaTByLength = _xs_t * _length;
	_sourTerm = _flatSour;
	
	_fsrFlux = (FP_PRECISION)(0.0);
// #endif // defined(M_2DMOC)

	for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {

// #if defined(M_3DMOCex)
// 		_scale=_twoOverWidth*GPUCosThetas[_pol];
// 		_3d_sigma=(_xs_t+_scale);
// 		_3d_oneOverSigma=1.0/_3d_sigma;
// 		_3d_Q=(_flatSour+_scale*_axialSurfAnfFlux_inp[_angId + _pol - _polOffset]);
// 		_sourTerm=_3d_Q*_3d_oneOverSigma;
// 		_sigmaTByLength = _3d_sigma * _length;
// #endif // defined(M_3DMOCex)
		_x2 = _sigmaTByLength * GPUSinThetaForExp[_pol];
		_x1 = (int)(_x2);
		_x3 = _x2 - _x1;

		_exp = _GPULinearExpTable[_x1] + _x3 * (_GPULinearExpTable[_x1 + 1] - _GPULinearExpTable[_x1]);
		// _exp=1.-_ExpFunction_GPU(-_sigmaTByLength * GPUOneOverSinThetas[_pol]);
        _delta = (_track_flux[_pol - _polOffset] - _sourTerm)*_exp;
		_track_flux[_pol - _polOffset] -= _delta;

// #if defined(M_2DMOC)
		_fsrFlux += _delta * _GPUTotalWeights_2DMOC[_azim*(*GPUNumPolarAng) + _pol] * _mul;
// #endif // defined(M_2DMOC)

// #if defined(M_3DMOCex)
// 		_fsrFlux = _oneOverVolbySpacing*(_sourTerm*_length+GPUSinThetas[_pol]*_3d_oneOverSigma*_delta);
// #ifdef DOUBLE_FLUX
//     	atomicAdd_(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
// #endif

// #ifdef SINGLE_FLUX
//     	atomicAdd(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
// #endif
// #endif // defined(M_3DMOCex)
	}

// #if defined(M_2DMOC)
	_fsrFlux *= _oneOverVolume;
	/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	atomicAdd_(&_GPUScalarFlux[_indexLay], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_indexLay], (FP_PRECISION_FLUX)_fsrFlux);
#endif
// #endif // defined(M_2DMOC)
}



__global__ void GPUTS_Group_ExpInte_OneDir_AngFSmem_FixSourProb_2D(int _lay, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
	int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
    FP_PRECISION *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
    FP_PRECISION *_GPULinearExpTable,FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
    FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
	FP_PRECISION_GEO *_GPURTMWidth_Y, int _polarBoundLowLim,int _polarBoundUpLim, FP_PRECISION _twoOverLayWidth,\
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, int _polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC, \
	FP_PRECISION *_GPUMacroXS_t_wScale,int _groupBoundLowLim, int _groupBoundUpLim)
{
	int _tid, _ltGlobID, _groupID, _azim, _azimNext, _ltIDNext, _polEngID, _pol, track_flux_index, _cell, _ray, _globCellID, _trackIndex;
	int _angId, _seg, _reg, _fsrID, _track, _index,_numPol,_ltInd,_layFsrID,_indexLay;
	FP_PRECISION _length, _oneOverVolume, _oneOverVolbySpacing;
// #if defined(M_3DMOCex)
// 	FP_PRECISION _spacing; 
// 	int _angId1,_angId2,_angId3,_angMul1,_angMul2, _layFsrID;
// #endif // defined(M_3DMOCex)
	/* Shared memory buffer for each thread's angular flux */
	extern __shared__ FP_PRECISION temp_flux[];
	FP_PRECISION *_track_flux = NULL;
	FP_PRECISION _beta;
	int _numGroupCalc=_groupBoundUpLim-_groupBoundLowLim;

// #if defined(M_MG_CMFD)
// 	int _surfIndex_Layer_X, _surfIndex_Layer_Y,_firstTrackInLT;
// 	_surfIndex_Layer_X = ((*GPUMOCNumPinsInSubDom_X) + 1) * (*GPUMOCNumPinsInSubDom_Y);
// 	_surfIndex_Layer_Y = (*GPUMOCNumPinsInSubDom_X) * ((*GPUMOCNumPinsInSubDom_Y) + 1);
// #endif // M_MG_CMFD

    _ltInd = _lay * (*GPUNumLTs_Total);
	_numPol=(*GPUNumPolarAng);

	// _numPol=1;
	
#if defined(M_3DMOCex)
	_numPol=_numPol/2;
#endif // defined(M_3DMOCex)

	_tid = threadIdx.x + blockIdx.x * blockDim.x;
	// _ltGlobID = _tid / (*GPUNumMOCGroups);
	// _groupID = _tid % (*GPUNumMOCGroups);
	_ltGlobID = _tid / _numGroupCalc;
	_groupID = _groupBoundLowLim + _tid % _numGroupCalc;
	_polEngID = _groupID * (*GPUNumPolarAng);

	track_flux_index = threadIdx.x * _numPol;

// #if defined(M_3DMOCex)
// 	_angMul2=(*GPUNumAzimAng)*_numPol;
// 	_angMul1=(*GPUNumMOCGroups)*_angMul2;
// 	_angId2 = _groupID*_angMul2;
// #endif // defined(M_3DMOCex)

	if (_ltGlobID < (*GPUNumLTs_Total)) {
		/* Retrieve pointer to thread's shared memory buffer for angular flux */
		_track_flux = &temp_flux[track_flux_index];
		
		for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
    		_track_flux[_pol-_polOffset] = _GPUBoundaryFlux(/*_ltInd+*/_ltGlobID, _pol + _polEngID);
		}
		_azim = _GPULTsIndex_AzimAngID[_ltGlobID];

// #if defined(M_3DMOCex)
// 		_angId3 = _azim*_numPol;
// 		_spacing = GPUEffRaySpacing[_azim];
// #endif // defined(M_3DMOCex)

// #if defined(M_MG_CMFD)
//         _firstTrackInLT = 0;
// #endif // defined(M_MG_CMFD)
		/* Loop over each Track segment in forward direction */
		for (_track = _GPUTrackIndexInLTs[_ltGlobID]; _track < _GPUTrackIndexInLTs[_ltGlobID + 1]; _track++) {
			_cell = _GPUTracksInfoInLTs[_track].cell_ID;
			_ray = _GPUTracksInfoInLTs[_track].track_ID;
			_globCellID = _lay * _numCellsInDom + _cell;
            _trackIndex = (*GPUTotNumTracksInOneRTM_2PI)*_GPUCellRTMType_SubDom[_globCellID] + _GPUIndexTrackInRTM_SumAzim[_azim] + _ray;
// #if defined(M_MG_CMFD)
//             if (_firstTrackInLT == 0) {
//                 GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
//                     _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 1, \
//                     _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
// 					_GPURTMWidth_X, _GPURTMWidth_Y, _polarBoundLowLim, _polarBoundUpLim,  _polOffset,_GPUTotalWeights_2DMOC);
//             }
// #endif // defined(M_MG_CMFD)
			for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {
				_length = _GPUSegInfoInTrack_2PI_length[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];
				_reg = _GPUSegInfoInTrack_2PI_FSRID[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];

				_fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID] + _reg;
				_oneOverVolume = _GPUOneOverFSRsAreaInRTM[_GPUSumNumFSRsInRTM[_GPUCellRTMType_SubDom[_globCellID]] + _reg];
// #if defined(M_3DMOCex)
// 				_oneOverVolbySpacing = _oneOverVolume * _spacing;
				_layFsrID = _GPUSumNumFSRsBasedGlobCellID[_cell] + _reg;
// 				_angId1 = _layFsrID * _angMul1;
// #endif //  defined(M_3DMOCex)
				_index = _fsrID * (*GPUNumMOCGroups) + _groupID;
				_indexLay = _layFsrID * (*GPUNumMOCGroups) + _groupID;

// #if defined(M_3DMOCex)
// 				_angId = _angId1 + _angId2 + _angId3;
// #endif //  defined(M_3DMOCex)

// #if !defined(M_3DOpt)
			    GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_FixSourProb_2D(_index,_indexLay, _azim, _length, _oneOverVolume, \
			    	_GPUMacroXS_t, _track_flux, _GPUSource, _GPUScalarFlux, _GPULinearExpTable, \
					_polarBoundLowLim, _polarBoundUpLim, _polOffset, _twoOverLayWidth,_oneOverVolbySpacing,\
					_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update,_GPUTotalWeights_2DMOC);
// #else
// 				GPUTallyScalarFlux_ExpInte_OneDir_AngFSmem_opt2(_index, _azim, _length, _oneOverVolume, \
// 			    	_GPUMacroXS_t, _GPUMacroXS_t_wScale, _track_flux, _GPUSource, _GPUScalarFlux, _GPULinearExpTable, \
// 					_polarBoundLowLim, _polarBoundUpLim, _polOffset,_oneOverVolbySpacing,\
// 					_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update);
// #endif // 
            }
// #if defined(M_MG_CMFD)
//             GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
//                 _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 0, \
//                 _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
//                 _GPURTMWidth_X, _GPURTMWidth_Y, _polarBoundLowLim, _polarBoundUpLim,  _polOffset,_GPUTotalWeights_2DMOC);
//             _firstTrackInLT=1;
// #endif // M_MG_CMFD
		}
		_azimNext = _GPULongTracksInfo[_ltGlobID].next_azim_angle_id;
		_ltIDNext = _GPULongTracksInfo[_ltGlobID].next_LongTrack_id_SubDom;
		_beta = _GPULongTracksInfo[_ltGlobID]._boundBeta;

#ifdef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme(_ltInd,_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUBoundaryFlux, _GPUMPIFlux, _beta, _polarBoundLowLim, _polarBoundUpLim, _polOffset);
#endif // MPI_CUDA_UpAngFImme

#ifndef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUStartFlux, _GPUMPIFlux, _beta);
#endif // !MPI_CUDA_UpAngFImme
	}
}


__device__ void GPUTallyScalarFlux_ExpFunc_OneDir_AngFSmem_FixSourProb_2D(int _index, int _indexLay, int _azim, FP_PRECISION _length, \
	FP_PRECISION _oneOverVolume, FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *_track_flux, FP_PRECISION *_GPUSource, \
	FP_PRECISION_FLUX *_GPUScalarFlux, int _polarBoundLowLim,int _polarBoundUpLim, int _polOffset, FP_PRECISION _twoOverWidth,\
	FP_PRECISION _oneOverVolbySpacing,int _angId,FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update,\
	FP_PRECISION *_GPUTotalWeights_2DMOC)
{
	int _pol;
	FP_PRECISION _xs_t, _sigmaTByLength, _fsrFlux,_delta,_flatSour, _exp,_sourTerm;

	FP_PRECISION _mul=1.0;
#if defined(M_2DMOC)
    _mul=1.0;
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
    _mul=2.0;
#endif // defined(M_3DMOCex)
// #if defined(M_3DMOCex)
// 	FP_PRECISION _3d_sigma, _3d_oneOverSigma, _3d_Q,_scale;
// #endif // defined(M_3DMOCex)
	_xs_t = _GPUMacroXS_t[_indexLay];
	_flatSour = _GPUSource[_indexLay];

// #if defined(M_2DMOC)
	_sigmaTByLength = _xs_t * _length;
	_sourTerm = _flatSour;
	
	_fsrFlux = (FP_PRECISION)(0.0);
// #endif // defined(M_2DMOC)

	for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {

// #if defined(M_3DMOCex)
// 		_scale=_twoOverWidth*GPUCosThetas[_pol];
// 		_3d_sigma=(_xs_t+_scale);
// 		_3d_oneOverSigma=1.0/_3d_sigma;
// 		_3d_Q=(_flatSour+_scale*_axialSurfAnfFlux_inp[_angId + _pol - _polOffset]);
// 		_sourTerm=_3d_Q*_3d_oneOverSigma;
// 		_sigmaTByLength = _3d_sigma * _length;
// #endif // defined(M_3DMOCex)

		_exp=1.-_ExpFunction_GPU(-_sigmaTByLength * GPUOneOverSinThetas[_pol]);
		_delta = (_track_flux[_pol - _polOffset] - _sourTerm)*_exp;
		_track_flux[_pol - _polOffset] -= _delta;

// #if defined(M_2DMOC)
		_fsrFlux += _delta * _GPUTotalWeights_2DMOC[_azim*(*GPUNumPolarAng) + _pol]*_mul;
// #endif // defined(M_2DMOC)

// #if defined(M_3DMOCex)
// 		_fsrFlux = _oneOverVolbySpacing*(_sourTerm*_length+GPUSinThetas[_pol]*_3d_oneOverSigma*_delta);
// #ifdef DOUBLE_FLUX
//     	atomicAdd_(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
// #endif

// #ifdef SINGLE_FLUX
//     	atomicAdd(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
// #endif
// #endif // defined(M_3DMOCex)
	}

// #if defined(M_2DMOC)
	_fsrFlux *= _oneOverVolume;
	/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
#endif
// #endif // defined(M_2DMOC)
}

__device__ void GPUTallyScalarFlux_ExpFunc_OneDir_AngFSmem_FixSourProb_2D_sld(int _index, int _indexLay, int _azim, FP_PRECISION _length, \
	FP_PRECISION _oneOverVolume, FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *_track_flux, FP_PRECISION *_GPUSource, \
	FP_PRECISION_FLUX *_GPUScalarFlux, int _polarBoundLowLim,int _polarBoundUpLim, int _polOffset, FP_PRECISION _twoOverWidth,\
	FP_PRECISION _oneOverVolbySpacing,int _angId,FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update,\
	FP_PRECISION *_GPUTotalWeights_2DMOC)
{
	int _pol;
	FP_PRECISION _xs_t, _sigmaTByLength, _fsrFlux,_delta,_flatSour, _exp,_sourTerm, _fsrFluxAng, _3d_oneOverSigma;

	FP_PRECISION _mul=1.0;
#if defined(M_2DMOC)
    _mul=1.0;
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
    _mul=2.0;
#endif // defined(M_3DMOCex)

	_xs_t = _GPUMacroXS_t[_indexLay];
	_flatSour = _GPUSource[_indexLay];

	_3d_oneOverSigma = 1.0 / _xs_t;
	_sigmaTByLength = _xs_t * _length;
	_sourTerm = _flatSour;
	
	_fsrFlux = (FP_PRECISION)(0.0);

	for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {

		_exp=1.-_ExpFunction_GPU(-_sigmaTByLength * GPUOneOverSinThetas[_pol]);
		_delta = (_track_flux[_pol - _polOffset] - _sourTerm)*_exp;
		_track_flux[_pol - _polOffset] -= _delta;

		_fsrFlux += _delta * _GPUTotalWeights_2DMOC[_azim*(*GPUNumPolarAng) + _pol]*_mul;
		_fsrFluxAng = _oneOverVolbySpacing*(_sourTerm*_length+GPUSinThetas[_pol]*_3d_oneOverSigma*_delta);

#ifdef DOUBLE_FLUX
    	atomicAdd_(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFluxAng);
#endif

#ifdef SINGLE_FLUX
    	atomicAdd(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFluxAng);
#endif
	}

// #if defined(M_2DMOC)
	_fsrFlux *= _oneOverVolume;
	/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
#endif
// #endif // defined(M_2DMOC)
}

__global__ void GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt_11_sld(int _lay, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
	int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
    FP_PRECISION *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
    FP_PRECISION *_GPULinearExpTable,FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
    FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
	FP_PRECISION_GEO *_GPURTMWidth_Y, int _polarBoundLowLim,int _polarBoundUpLim, FP_PRECISION _twoOverLayWidth,\
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, int _polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC, \
	FP_PRECISION *_GPUMacroXS_t_wScale, FP_PRECISION *_GPUFSRAngScattSour,int _groupBoundLowLim, int _groupBoundUpLim)
{
    int _tid, _ltGlobID, _groupID, _azim, _azimNext, _ltIDNext, _polEngID, _pol, track_flux_index, _cell, _ray, _globCellID, _trackIndex;
    int _seg, _reg, _fsrID, _track, _index,_indexLay,_numPol,_ltInd, a;
    FP_PRECISION _length, _oneOverVolume, _oneOverVolbySpacing;
    M_SIZET _angId;
// #if defined(M_3DMOCex)
    FP_PRECISION _spacing;
    int _angId2,_angId3, _layFsrID;
    M_SIZET _angId1;
    FP_PRECISION _trk_flux = 0.0;
    FP_PRECISION _beta;

    FP_PRECISION _xs_t, _sigmaTByLength, _fsrFlux,_delta,_flatSour,_x2,_x3, _exp,_sourTerm,_scattSour=0.0,_3d_oneOverSigma;
	FP_PRECISION _fsrFluxS;

    _ltInd = _lay * (*GPUNumLTs_Total);
    _numPol=(*GPUNumPolarAng);

	FP_PRECISION _mul=1.0;
#if defined(M_2DMOC)
    _mul=1.0;
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
    _mul=2.0;
#endif // defined(M_3DMOCex)

    _tid = threadIdx.x + blockIdx.x * blockDim.x;
	int _numGroupCalc=_groupBoundUpLim-_groupBoundLowLim;
    // tid = lt * NG * NP + g * NP + p
    // int NGxNPol = (*GPUNumMOCGroups) * _numPol;
    int NGxNPol = _numGroupCalc * _numPol;
    _ltGlobID = _tid / (NGxNPol);
    _groupID = _tid % (NGxNPol) / _numPol;
    _pol = _tid % (NGxNPol) % _numPol;

    _polEngID = _groupID * _numPol;

    _angId2 = _groupID*(*GPU_angMul2_FP);  // NA x NP

    if (_ltGlobID < (*GPUNumLTs_Total)) {
        /* Retrieve pointer to thread's shared memory buffer for angular flux */

        _trk_flux = _GPUBoundaryFlux(_ltGlobID, _pol + _polEngID);

        _azim = _GPULTsIndex_AzimAngID[_ltGlobID];

        _angId3 = _azim*_numPol;
        _spacing = GPUEffRaySpacing[_azim];

        /* Loop over each Track segment in forward direction */
        for (_track = _GPUTrackIndexInLTs[_ltGlobID]; _track < _GPUTrackIndexInLTs[_ltGlobID + 1]; _track++) {
            _cell = _GPUTracksInfoInLTs[_track].cell_ID;
            _ray = _GPUTracksInfoInLTs[_track].track_ID;
            _globCellID = _lay * _numCellsInDom + _cell;
            _trackIndex = (*GPUTotNumTracksInOneRTM_2PI)*_GPUCellRTMType_SubDom[_globCellID] + _GPUIndexTrackInRTM_SumAzim[_azim] + _ray;

            for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {
                _length = _GPUSegInfoInTrack_2PI_length[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];
                _reg = _GPUSegInfoInTrack_2PI_FSRID[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];

                _fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID] + _reg;
				
                _oneOverVolume = _GPUOneOverFSRsAreaInRTM[_GPUSumNumFSRsInRTM[_GPUCellRTMType_SubDom[_globCellID]] + _reg];
				
                _oneOverVolbySpacing = _oneOverVolume * _spacing;
                _layFsrID = _GPUSumNumFSRsBasedGlobCellID[_cell] + _reg;
                _angId1 = (M_SIZET)(_layFsrID) * (M_SIZET)(*GPU_angMul1_FP);  // NG x NA x NP

                _indexLay = _layFsrID * (*GPUNumMOCGroups) + _groupID;
				
                _index = _fsrID * (*GPUNumMOCGroups) + _groupID;

                _indexLay = _index;

                _angId = _angId1 + (M_SIZET)(_angId2 + _angId3); // (fsr, g, azim, *)

                _xs_t = _GPUMacroXS_t[_indexLay];
                _flatSour = _GPUSource[_indexLay];
                _sigmaTByLength = _xs_t * _length;
                _3d_oneOverSigma = 1.0 / _xs_t;
                _sourTerm = _flatSour * _3d_oneOverSigma;

                _exp = 1. - _ExpFunction_GPU(-_sigmaTByLength * GPUOneOverSinThetas[_pol]);


                _delta = (_trk_flux - _sourTerm) * _exp;
                _trk_flux -= _delta;
                _fsrFlux = _oneOverVolbySpacing * (_sourTerm * _length + GPUSinThetas[_pol] * _3d_oneOverSigma * _delta);
				_fsrFluxS = _delta * _GPUTotalWeights_2DMOC[_azim*(*GPUNumPolarAng) + _pol]*_mul*_oneOverVolume;
#ifdef DOUBLE_FLUX
                atomicAdd_(&_axialSurfAngFlux_update[_angId + _pol - _polOffset], (FP_PRECISION_FLUX)_fsrFlux);
				atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX) _fsrFluxS);
#endif

#ifdef SINGLE_FLUX
                atomicAdd(&_axialSurfAngFlux_update[_angId + _pol - _polOffset], (FP_PRECISION_FLUX) _fsrFlux);
				atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX) _fsrFluxS);
#endif
            }

        }
        _azimNext = _GPULongTracksInfo[_ltGlobID].next_azim_angle_id;
        _ltIDNext = _GPULongTracksInfo[_ltGlobID].next_LongTrack_id_SubDom;
        _beta = _GPULongTracksInfo[_ltGlobID]._boundBeta;

#ifdef MPI_CUDA_UpAngFImme
        GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme_1(_ltInd,_azim, _azimNext, _pol, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _trk_flux, _GPUBoundaryFlux, _GPUMPIFlux, _beta, _polarBoundLowLim, _polarBoundUpLim, _polOffset);
#endif // MPI_CUDA_UpAngFImme

#ifndef MPI_CUDA_UpAngFImme
        GPUUpdateBoundaryFlux_AngFluxOnMem(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUStartFlux, _GPUMPIFlux, _beta);
#endif // !MPI_CUDA_UpAngFImme
    }
}

__global__ void GPUTS_Group_ExpFunc_OneDir_AngFSmem_FixSourProb_2D(int _lay, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
	int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
    FP_PRECISION *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
    FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
    FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
	FP_PRECISION_GEO *_GPURTMWidth_Y, int _polarBoundLowLim,int _polarBoundUpLim, FP_PRECISION _twoOverLayWidth,\
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, int _polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC, \
	FP_PRECISION *_GPUMacroXS_t_wScale,int _groupBoundLowLim, int _groupBoundUpLim)
{
	int _tid, _ltGlobID, _groupID, _azim, _azimNext, _ltIDNext, _polEngID, _pol, track_flux_index, _cell, _ray, _globCellID, _trackIndex;
	int _angId, _seg, _reg, _fsrID, _track, _index,_numPol,_ltInd,_layFsrID,_indexLay;
	FP_PRECISION _length, _oneOverVolume, _oneOverVolbySpacing;
// #if defined(M_3DMOCex)
// 	FP_PRECISION _spacing;
// 	int _angId1,_angId2,_angId3,_angMul1,_angMul2, _layFsrID;
// #endif // defined(M_3DMOCex)
	/* Shared memory buffer for each thread's angular flux */
	extern __shared__ FP_PRECISION temp_flux[];
	FP_PRECISION *_track_flux = NULL;
	FP_PRECISION _beta;
	int _numGroupCalc=_groupBoundUpLim-_groupBoundLowLim;

// #if defined(M_MG_CMFD)
// 	int _surfIndex_Layer_X, _surfIndex_Layer_Y,_firstTrackInLT;
// 	_surfIndex_Layer_X = ((*GPUMOCNumPinsInSubDom_X) + 1) * (*GPUMOCNumPinsInSubDom_Y);
// 	_surfIndex_Layer_Y = (*GPUMOCNumPinsInSubDom_X) * ((*GPUMOCNumPinsInSubDom_Y) + 1);
// #endif // M_MG_CMFD

    _ltInd = _lay * (*GPUNumLTs_Total);
	_numPol=(*GPUNumPolarAng);

#if defined(M_3DMOCex)
	_numPol=_numPol/2;
#endif // defined(M_3DMOCex)

	_tid = threadIdx.x + blockIdx.x * blockDim.x;
	// _ltGlobID = _tid / (*GPUNumMOCGroups);
	// _groupID = _tid % (*GPUNumMOCGroups);
	_ltGlobID = _tid / _numGroupCalc;
	_groupID = _groupBoundLowLim + _tid % _numGroupCalc;
	_polEngID = _groupID * (*GPUNumPolarAng);
	

	track_flux_index = threadIdx.x * _numPol;

// #if defined(M_3DMOCex)
// 	_angMul2=(*GPUNumAzimAng)*_numPol;
// 	_angMul1=(*GPUNumMOCGroups)*_angMul2;
// 	_angId2 = _groupID*_angMul2;
// #endif // defined(M_3DMOCex)

	if (_ltGlobID < (*GPUNumLTs_Total)) {
		/* Retrieve pointer to thread's shared memory buffer for angular flux */
		_track_flux = &temp_flux[track_flux_index];
		
		for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
    		_track_flux[_pol-_polOffset] = _GPUBoundaryFlux(/*_ltInd+*/_ltGlobID, _pol + _polEngID);
		}
		_azim = _GPULTsIndex_AzimAngID[_ltGlobID];

// #if defined(M_3DMOCex)
// 		_angId3 = _azim*_numPol;
// 		_spacing = GPUEffRaySpacing[_azim];
// #endif // defined(M_3DMOCex)

// #if defined(M_MG_CMFD)
//         _firstTrackInLT = 0;
// #endif // defined(M_MG_CMFD)
		/* Loop over each Track segment in forward direction */
		for (_track = _GPUTrackIndexInLTs[_ltGlobID]; _track < _GPUTrackIndexInLTs[_ltGlobID + 1]; _track++) {
			_cell = _GPUTracksInfoInLTs[_track].cell_ID;
			_ray = _GPUTracksInfoInLTs[_track].track_ID;
			_globCellID = _lay * _numCellsInDom + _cell;
            _trackIndex = (*GPUTotNumTracksInOneRTM_2PI)*_GPUCellRTMType_SubDom[_globCellID] + _GPUIndexTrackInRTM_SumAzim[_azim] + _ray;
// #if defined(M_MG_CMFD)
//             if (_firstTrackInLT == 0) {
//                 GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
//                     _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 1, \
//                     _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
// 					_GPURTMWidth_X, _GPURTMWidth_Y, _polarBoundLowLim, _polarBoundUpLim,  _polOffset,_GPUTotalWeights_2DMOC);
//             }
// #endif // defined(M_MG_CMFD)
			for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {
				_length = _GPUSegInfoInTrack_2PI_length[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];
				_reg = _GPUSegInfoInTrack_2PI_FSRID[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];

				_fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID] + _reg;
				_oneOverVolume = (FP_PRECISION)(_GPUOneOverFSRsAreaInRTM[_GPUSumNumFSRsInRTM[_GPUCellRTMType_SubDom[_globCellID]] + _reg]);
// #if defined(M_3DMOCex)
// 				_oneOverVolbySpacing = _oneOverVolume * _spacing;
				_layFsrID = _GPUSumNumFSRsBasedGlobCellID[_cell] + _reg;
// 				_angId1 = _layFsrID * _angMul1;
// #endif //  defined(M_3DMOCex)
				_index = _fsrID * (*GPUNumMOCGroups) + _groupID;
				_indexLay = _layFsrID * (*GPUNumMOCGroups) + _groupID;

// #if defined(M_3DMOCex)
// 				_angId = _angId1 + _angId2 + _angId3;
// #endif //  defined(M_3DMOCex)

			    GPUTallyScalarFlux_ExpFunc_OneDir_AngFSmem_FixSourProb_2D(_index,_indexLay, _azim, _length, _oneOverVolume, \
			    	_GPUMacroXS_t, _track_flux, _GPUSource, _GPUScalarFlux, \
					_polarBoundLowLim, _polarBoundUpLim, _polOffset, _twoOverLayWidth,_oneOverVolbySpacing,\
					_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update,_GPUTotalWeights_2DMOC);
            }
// #if defined(M_MG_CMFD)
//             GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
//                 _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 0, \
//                 _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
//                 _GPURTMWidth_X, _GPURTMWidth_Y, _polarBoundLowLim, _polarBoundUpLim,  _polOffset,_GPUTotalWeights_2DMOC);
//             _firstTrackInLT=1;
// #endif // M_MG_CMFD
		}
		_azimNext = _GPULongTracksInfo[_ltGlobID].next_azim_angle_id;
		_ltIDNext = _GPULongTracksInfo[_ltGlobID].next_LongTrack_id_SubDom;
		_beta = _GPULongTracksInfo[_ltGlobID]._boundBeta;

#ifdef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme(_ltInd,_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUBoundaryFlux, _GPUMPIFlux, _beta, _polarBoundLowLim, _polarBoundUpLim, _polOffset);
#endif // MPI_CUDA_UpAngFImme

#ifndef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUStartFlux, _GPUMPIFlux, _beta);
#endif // !MPI_CUDA_UpAngFImme

        /* Increment the thread id */
	    // _tid += blockDim.x * gridDim.x;
	    // _ltGlobID = _tid / _numGroupCalc;
	    // _groupID = _groupBoundLowLim + _tid % _numGroupCalc;
		// _polEngID = _groupID * (*GPUNumPolarAng);
		// track_flux_index = threadIdx.x * _numPol;
	}
}

__global__ void GPUTS_Group_ExpFunc_OneDir_AngFSmem_FixSourProb_2D_sld(int _lay, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
	int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
    FP_PRECISION *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
    FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
    FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
	FP_PRECISION_GEO *_GPURTMWidth_Y, int _polarBoundLowLim,int _polarBoundUpLim, FP_PRECISION _twoOverLayWidth,\
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, int _polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC, \
	FP_PRECISION *_GPUMacroXS_t_wScale,int _groupBoundLowLim, int _groupBoundUpLim)
{
	int _tid, _ltGlobID, _groupID, _azim, _azimNext, _ltIDNext, _polEngID, _pol, track_flux_index, _cell, _ray, _globCellID, _trackIndex;
	int _angId, _seg, _reg, _fsrID, _track, _index,_numPol,_ltInd,_layFsrID,_indexLay;
	FP_PRECISION _length, _oneOverVolume, _oneOverVolbySpacing;

	FP_PRECISION _spacing;
	int _angId1, _angId2,_angId3;

	/* Shared memory buffer for each thread's angular flux */
	extern __shared__ FP_PRECISION temp_flux[];
	FP_PRECISION *_track_flux = NULL;
	FP_PRECISION _beta;
	int _numGroupCalc=_groupBoundUpLim-_groupBoundLowLim;

    _ltInd = _lay * (*GPUNumLTs_Total);
	_numPol=(*GPUNumPolarAng);

#if defined(M_3DMOCex)
	_numPol=_numPol/2;
#endif // defined(M_3DMOCex)

	_tid = threadIdx.x + blockIdx.x * blockDim.x;
	_ltGlobID = _tid / _numGroupCalc;
	_groupID = _groupBoundLowLim + _tid % _numGroupCalc;
	_polEngID = _groupID * (*GPUNumPolarAng);
	
#if defined(M_2DMOC)
    _angId2 = _groupID*(*GPU_angMul2_FP);  // NA x NP
#else
    // #if defined(M_3DMOCex)
	_angId2 = _groupID*(*GPU_angMul2_HP);
#endif

	track_flux_index = threadIdx.x * _numPol;

	if (_ltGlobID < (*GPUNumLTs_Total)) {
		/* Retrieve pointer to thread's shared memory buffer for angular flux */
		_track_flux = &temp_flux[track_flux_index];
		
		for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {

    		_track_flux[_pol-_polOffset] = _GPUBoundaryFlux(/*_ltInd+*/_ltGlobID, _pol + _polEngID);

		}
		_azim = _GPULTsIndex_AzimAngID[_ltGlobID];

		_angId3 = _azim*_numPol;
        _spacing = GPUEffRaySpacing[_azim];

		/* Loop over each Track segment in forward direction */
		for (_track = _GPUTrackIndexInLTs[_ltGlobID]; _track < _GPUTrackIndexInLTs[_ltGlobID + 1]; _track++) {

			_cell = _GPUTracksInfoInLTs[_track].cell_ID;
			_ray = _GPUTracksInfoInLTs[_track].track_ID;
			_globCellID = _lay * _numCellsInDom + _cell;
            _trackIndex = (*GPUTotNumTracksInOneRTM_2PI)*_GPUCellRTMType_SubDom[_globCellID] + _GPUIndexTrackInRTM_SumAzim[_azim] + _ray;

			for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {

				_length = _GPUSegInfoInTrack_2PI_length[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];
				_reg = _GPUSegInfoInTrack_2PI_FSRID[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];

				_fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID] + _reg;
				
				_oneOverVolume = (FP_PRECISION)(_GPUOneOverFSRsAreaInRTM[_GPUSumNumFSRsInRTM[_GPUCellRTMType_SubDom[_globCellID]] + _reg]);
				_oneOverVolbySpacing = _oneOverVolume * _spacing;

				_layFsrID = _GPUSumNumFSRsBasedGlobCellID[_cell] + _reg;

			#if defined(M_2DMOC)
                _angId1 = (M_SIZET)(_layFsrID) * (M_SIZET)(*GPU_angMul1_FP);  // NG x NA x NP
			#else
                _angId1 = (M_SIZET)(_layFsrID) * (M_SIZET)(*GPU_angMul1_HP);
			#endif

				_index = _fsrID * (*GPUNumMOCGroups) + _groupID;
				_indexLay = _layFsrID * (*GPUNumMOCGroups) + _groupID;

				_angId = _angId1 + (M_SIZET)(_angId2 + _angId3); // (fsr, g, azim, *)

			    GPUTallyScalarFlux_ExpFunc_OneDir_AngFSmem_FixSourProb_2D_sld(_index,_indexLay, _azim, _length, _oneOverVolume, \
			    	_GPUMacroXS_t, _track_flux, _GPUSource, _GPUScalarFlux, \
					_polarBoundLowLim, _polarBoundUpLim, _polOffset, _twoOverLayWidth,_oneOverVolbySpacing,\
					_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update,_GPUTotalWeights_2DMOC);
            }
		}
		_azimNext = _GPULongTracksInfo[_ltGlobID].next_azim_angle_id;
		_ltIDNext = _GPULongTracksInfo[_ltGlobID].next_LongTrack_id_SubDom;
		_beta = _GPULongTracksInfo[_ltGlobID]._boundBeta;

#ifdef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme(_ltInd,_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUBoundaryFlux, _GPUMPIFlux, _beta, _polarBoundLowLim, _polarBoundUpLim, _polOffset);
#endif // MPI_CUDA_UpAngFImme

#ifndef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUStartFlux, _GPUMPIFlux, _beta);
#endif // !MPI_CUDA_UpAngFImme
	}
}

__device__ void GPUTallyScalarFlux_DDExpInte_OneDir_AngFSmem_FixSourProb_2D(int _index, int _indexLay, int _azim, FP_PRECISION _length, \
	FP_PRECISION _oneOverVolume, FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *_track_flux, FP_PRECISION *_GPUSource, \
	FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPULinearExpTable, int _polarBoundLowLim,int _polarBoundUpLim, \
	int _polOffset, FP_PRECISION _twoOverWidth, FP_PRECISION _oneOverVolbySpacing,int _angId,FP_PRECISION *_axialSurfAnfFlux_inp, \
	FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_GPUTotalWeights_2DMOC)
{
	int _x1, _pol;
	FP_PRECISION _xs_t, _sigmaTByLength, _fsrFlux,_delta, _mul2,_flatSour,_x2,_x3, _fluxTemp=0.0, _exp,_sourTerm;

	FP_PRECISION _mul=1.0;
#if defined(M_2DMOC)
    _mul=1.0;
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
    _mul=2.0;
#endif // defined(M_3DMOCex)
// #if defined(M_3DMOCex)
// 	FP_PRECISION _3d_sigma, _3d_oneOverSigma, _3d_Q, _3d_QtoSigma,_scale;
// #endif // defined(M_3DMOCex)
	_xs_t = _GPUMacroXS_t[_indexLay];
	_flatSour = _GPUSource[_indexLay];

// #if defined(M_2DMOC)
	_sigmaTByLength = _xs_t * _length;
	_sourTerm = _flatSour;
	_mul2 = (FP_PRECISION)(2.0) * _sourTerm * _sigmaTByLength;
	_fsrFlux = (FP_PRECISION)(0.0);
// #endif // defined(M_2DMOC)

	for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
// #if defined(M_3DMOCex)
// 		_scale=_twoOverWidth*GPUCosThetas[_pol];
// 		_3d_sigma=(_xs_t+_scale);
// 		_3d_oneOverSigma=1.0/_3d_sigma;
// 		_3d_Q=(_flatSour+_scale*_axialSurfAnfFlux_inp[_angId + _pol - _polOffset]);
// 		_3d_QtoSigma=_3d_Q*_3d_oneOverSigma;
// 		_sourTerm = _3d_QtoSigma;
// 		_sigmaTByLength = _3d_sigma * _length;
// 		_mul2 = (FP_PRECISION)(2.0) * _3d_Q * _length;
// #endif // defined(M_3DMOCex)
		_fluxTemp = ((GPUSinThetasTimes2[_pol]-_sigmaTByLength)*_track_flux[_pol-_polOffset]+_mul2)/(GPUSinThetasTimes2[_pol]+_sigmaTByLength);
		if (_fluxTemp < 0.0) {
			_x2 = _sigmaTByLength * GPUSinThetaForExp[_pol];
			_x1 = (int)(_x2);
			_x3 = _x2 - _x1;
			_exp = _GPULinearExpTable[_x1] + _x3 * (_GPULinearExpTable[_x1 + 1] - _GPULinearExpTable[_x1]);
			_delta = (_track_flux[_pol - _polOffset] - _sourTerm)*_exp;
			_track_flux[_pol - _polOffset] -= _delta;
		}
		else {
			_delta = _track_flux[_pol-_polOffset] - _fluxTemp;
			_track_flux[_pol - _polOffset] = _fluxTemp;
		}

// #if defined(M_2DMOC)
		_fsrFlux += _delta * _GPUTotalWeights_2DMOC[_azim*(*GPUNumPolarAng) + _pol]* _mul;
// #endif // defined(M_2DMOC)

// #if defined(M_3DMOCex)
// 		_fsrFlux = _oneOverVolbySpacing*(_3d_QtoSigma*_length+GPUSinThetas[_pol]*_3d_oneOverSigma*_delta);
// #ifdef DOUBLE_FLUX
//     	atomicAdd_(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
// #endif

// #ifdef SINGLE_FLUX
//     	atomicAdd(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
// #endif
// #endif // defined(M_3DMOCex)
	}

// #if defined(M_2DMOC)
	_fsrFlux *= _oneOverVolume;
	/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
#endif
// #endif // defined(M_2DMOC)
}


__global__ void GPUTS_Group_DDExpInte_OneDir_AngFSmem_FixSourProb_2D(int _lay, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
	int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
    FP_PRECISION *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
    FP_PRECISION *_GPULinearExpTable,FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
    FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
	FP_PRECISION_GEO *_GPURTMWidth_Y, int _polarBoundLowLim,int _polarBoundUpLim, FP_PRECISION _twoOverLayWidth,\
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, int _polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC,\
	int _groupBoundLowLim, int _groupBoundUpLim)
{
	int _tid, _ltGlobID, _groupID, _azim, _azimNext, _ltIDNext, _polEngID, _pol, track_flux_index, _cell, _ray, _globCellID, _trackIndex;
	int _angId, _seg, _reg, _fsrID, _track, _index,_numPol, _ltInd,_layFsrID,_indexLay;
	FP_PRECISION _length, _oneOverVolume, _oneOverVolbySpacing;
// #if defined(M_3DMOCex)
// 	FP_PRECISION _spacing;
// 	int _angId1,_angId2,_angId3,_angMul1,_angMul2, _layFsrID;
// #endif // defined(M_3DMOCex)
	/* Shared memory buffer for each thread's angular flux */
	extern __shared__ FP_PRECISION temp_flux[];
	FP_PRECISION *_track_flux = NULL;
	FP_PRECISION _beta;
	int _numGroupCalc=_groupBoundUpLim-_groupBoundLowLim;

// #if defined(M_MG_CMFD)
// 	int _surfIndex_Layer_X, _surfIndex_Layer_Y,_firstTrackInLT;
// 	_surfIndex_Layer_X = ((*GPUMOCNumPinsInSubDom_X) + 1) * (*GPUMOCNumPinsInSubDom_Y);
// 	_surfIndex_Layer_Y = (*GPUMOCNumPinsInSubDom_X) * ((*GPUMOCNumPinsInSubDom_Y) + 1);
// #endif // M_MG_CMFD

    _ltInd = _lay * (*GPUNumLTs_Total);
	_numPol=(*GPUNumPolarAng);
	
#if defined(M_3DMOCex)
	_numPol=_numPol/2;
#endif // defined(M_3DMOCex)

	_tid = threadIdx.x + blockIdx.x * blockDim.x;
	// _ltGlobID = _tid / (*GPUNumMOCGroups);
	// _groupID = _tid % (*GPUNumMOCGroups);
	_ltGlobID = _tid / _numGroupCalc;
	_groupID = _groupBoundLowLim + _tid % _numGroupCalc;
	_polEngID = _groupID * (*GPUNumPolarAng);

	track_flux_index = threadIdx.x * _numPol;

// #if defined(M_3DMOCex)
// 	_angMul2=(*GPUNumAzimAng)*_numPol;
// 	_angMul1=(*GPUNumMOCGroups)*_angMul2;
// 	_angId2 = _groupID*_angMul2;
// #endif // defined(M_3DMOCex)

	if (_ltGlobID < (*GPUNumLTs_Total)) {
		/* Retrieve pointer to thread's shared memory buffer for angular flux */
		_track_flux = &temp_flux[track_flux_index];
		
		for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
    		_track_flux[_pol-_polOffset] = _GPUBoundaryFlux(/*_ltInd + */_ltGlobID, _pol + _polEngID);
		}
		_azim = _GPULTsIndex_AzimAngID[_ltGlobID];

// #if defined(M_3DMOCex)
// 		_angId3 = _azim*_numPol;
// 		_spacing = GPUEffRaySpacing[_azim];
// #endif // defined(M_3DMOCex)

// #if defined(M_MG_CMFD)
//         _firstTrackInLT = 0;
// #endif // defined(M_MG_CMFD)
		/* Loop over each Track segment in forward direction */
		for (_track = _GPUTrackIndexInLTs[_ltGlobID]; _track < _GPUTrackIndexInLTs[_ltGlobID + 1]; _track++) {
			_cell = _GPUTracksInfoInLTs[_track].cell_ID;
			_ray = _GPUTracksInfoInLTs[_track].track_ID;
			_globCellID = _lay * _numCellsInDom + _cell;
            _trackIndex = (*GPUTotNumTracksInOneRTM_2PI)*_GPUCellRTMType_SubDom[_globCellID] + _GPUIndexTrackInRTM_SumAzim[_azim] + _ray;
// #if defined(M_MG_CMFD)
//             if (_firstTrackInLT == 0) {
//                 GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
//                     _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 1, \
//                     _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
//                     _GPURTMWidth_X, _GPURTMWidth_Y,_polarBoundLowLim, _polarBoundUpLim,_polOffset,_GPUTotalWeights_2DMOC);
//             }
// #endif // defined(M_MG_CMFD)
			for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {
				_length = _GPUSegInfoInTrack_2PI_length[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];
				_reg = _GPUSegInfoInTrack_2PI_FSRID[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];

				_fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID] + _reg;
				_oneOverVolume = _GPUOneOverFSRsAreaInRTM[_GPUSumNumFSRsInRTM[_GPUCellRTMType_SubDom[_globCellID]] + _reg];
// #if defined(M_3DMOCex)
// 				_oneOverVolbySpacing = _oneOverVolume * _spacing;
				_layFsrID = _GPUSumNumFSRsBasedGlobCellID[_cell] + _reg;
// 				_angId1 = _layFsrID * _angMul1;
// #endif //  defined(M_3DMOCex)
				_index = _fsrID * (*GPUNumMOCGroups) + _groupID;
				_indexLay = _layFsrID * (*GPUNumMOCGroups) + _groupID;

// #if defined(M_3DMOCex)
// 				_angId = _angId1 + _angId2 + _angId3;
// #endif //  defined(M_3DMOCex)

			    GPUTallyScalarFlux_DDExpInte_OneDir_AngFSmem_FixSourProb_2D(_index, _indexLay, _azim, _length, _oneOverVolume, \
			    	_GPUMacroXS_t, _track_flux, _GPUSource, _GPUScalarFlux, _GPULinearExpTable, \
					_polarBoundLowLim, _polarBoundUpLim, _polOffset, _twoOverLayWidth,_oneOverVolbySpacing,\
					_angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update,_GPUTotalWeights_2DMOC);

            }
// #if defined(M_MG_CMFD)
//             GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
//                 _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 0, \
//                 _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
//                 _GPURTMWidth_X, _GPURTMWidth_Y,_polarBoundLowLim, _polarBoundUpLim,_polOffset,_GPUTotalWeights_2DMOC);
//             _firstTrackInLT=1;
// #endif // M_MG_CMFD
		}
		_azimNext = _GPULongTracksInfo[_ltGlobID].next_azim_angle_id;
		_ltIDNext = _GPULongTracksInfo[_ltGlobID].next_LongTrack_id_SubDom;
		_beta = _GPULongTracksInfo[_ltGlobID]._boundBeta;

#ifdef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme(_ltInd,_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUBoundaryFlux, _GPUMPIFlux, _beta, _polarBoundLowLim, _polarBoundUpLim, _polOffset);
#endif // MPI_CUDA_UpAngFImme

#ifndef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUStartFlux, _GPUMPIFlux, _beta);
#endif // !MPI_CUDA_UpAngFImme
	}
}

__device__ void GPUTallyScalarFlux_DDExpFunc_OneDir_AngFSmem_FixSourProb_2D(int _index, int _indexLay, int _azim, FP_PRECISION _length, \
	FP_PRECISION _oneOverVolume, FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *_track_flux, FP_PRECISION *_GPUSource, \
	FP_PRECISION_FLUX *_GPUScalarFlux, int _polarBoundLowLim,int _polarBoundUpLim, int _polOffset, FP_PRECISION _twoOverWidth,\
	FP_PRECISION _oneOverVolbySpacing,int _angId,FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update,\
	FP_PRECISION *_GPUTotalWeights_2DMOC)
{
	int _pol;
	FP_PRECISION _xs_t, _sigmaTByLength, _fsrFlux,_delta, _mul2,_flatSour, _fluxTemp=0.0, _exp,_sourTerm;

	FP_PRECISION _mul=1.0;
#if defined(M_2DMOC)
    _mul=1.0;
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
    _mul=2.0;
#endif // defined(M_3DMOCex)
// #if defined(M_3DMOCex)
// 	FP_PRECISION _3d_sigma, _3d_oneOverSigma, _3d_Q, _3d_QtoSigma,_scale;
// #endif // defined(M_3DMOCex)
	_xs_t = _GPUMacroXS_t[_indexLay];
	_flatSour = _GPUSource[_indexLay];

// #if defined(M_2DMOC)
	_sigmaTByLength = _xs_t * _length;
	_sourTerm = _flatSour;
	_mul2 = (FP_PRECISION)(2.0) * _sourTerm * _sigmaTByLength;
	_fsrFlux = (FP_PRECISION)(0.0);
// #endif // defined(M_2DMOC)

	for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
// #if defined(M_3DMOCex)
// 		_scale=_twoOverWidth*GPUCosThetas[_pol];
// 		_3d_sigma=(_xs_t+_scale);
// 		_3d_oneOverSigma=1.0/_3d_sigma;
// 		_3d_Q=(_flatSour+_scale*_axialSurfAnfFlux_inp[_angId + _pol - _polOffset]);
// 		_3d_QtoSigma=_3d_Q*_3d_oneOverSigma;
// 		_sourTerm = _3d_QtoSigma;
// 		_sigmaTByLength = _3d_sigma * _length;
// 		_mul2 = (FP_PRECISION)(2.0) * _3d_Q * _length;
// #endif // defined(M_3DMOCex)
		_fluxTemp = ((GPUSinThetasTimes2[_pol]-_sigmaTByLength)*_track_flux[_pol-_polOffset]+_mul2)/(GPUSinThetasTimes2[_pol]+_sigmaTByLength);
		if (_fluxTemp < 0.0) {
			_exp=1.-_ExpFunction_GPU(-_sigmaTByLength * GPUOneOverSinThetas[_pol]);
			_delta = (_track_flux[_pol - _polOffset] - _sourTerm)*_exp;
			_track_flux[_pol - _polOffset] -= _delta;
		}
		else {
			_delta = _track_flux[_pol-_polOffset] - _fluxTemp;
			_track_flux[_pol - _polOffset] = _fluxTemp;
		}

// #if defined(M_2DMOC)
		_fsrFlux += _delta * _GPUTotalWeights_2DMOC[_azim*(*GPUNumPolarAng) + _pol]* _mul;
// #endif // defined(M_2DMOC)

// #if defined(M_3DMOCex)
// 		_fsrFlux = _oneOverVolbySpacing*(_3d_QtoSigma*_length+GPUSinThetas[_pol]*_3d_oneOverSigma*_delta);
// #ifdef DOUBLE_FLUX
//     	atomicAdd_(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
// #endif

// #ifdef SINGLE_FLUX
//     	atomicAdd(&_axialSurfAngFlux_update[_angId+_pol-_polOffset], (FP_PRECISION_FLUX)_fsrFlux);
// #endif
// #endif // defined(M_3DMOCex)
	}

// #if defined(M_2DMOC)
	_fsrFlux *= _oneOverVolume;
	/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)_fsrFlux);
#endif
// #endif // defined(M_2DMOC)
}


__global__ void GPUTS_Group_DDExpFunc_OneDir_AngFSmem_FixSourProb_2D(int _lay, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
	int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
    FP_PRECISION *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
    FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
    FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
	FP_PRECISION_GEO *_GPURTMWidth_Y, int _polarBoundLowLim,int _polarBoundUpLim, FP_PRECISION _twoOverLayWidth,\
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, int _polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC,\
	int _groupBoundLowLim, int _groupBoundUpLim)
{
	int _tid, _ltGlobID, _groupID, _azim, _azimNext, _ltIDNext, _polEngID, _pol, track_flux_index, _cell, _ray, _globCellID, _trackIndex;
	int _angId, _seg, _reg, _fsrID, _track, _index,_numPol, _ltInd,_layFsrID,_indexLay;
	FP_PRECISION _length, _oneOverVolume, _oneOverVolbySpacing;
// #if defined(M_3DMOCex)
// 	FP_PRECISION _spacing;
// 	int _angId1,_angId2,_angId3,_angMul1,_angMul2, _layFsrID;
// #endif // defined(M_3DMOCex)
	/* Shared memory buffer for each thread's angular flux */
	extern __shared__ FP_PRECISION temp_flux[];
	FP_PRECISION *_track_flux = NULL;
	FP_PRECISION _beta;
	int _numGroupCalc=_groupBoundUpLim-_groupBoundLowLim;

// #if defined(M_MG_CMFD)
// 	int _surfIndex_Layer_X, _surfIndex_Layer_Y,_firstTrackInLT;
// 	_surfIndex_Layer_X = ((*GPUMOCNumPinsInSubDom_X) + 1) * (*GPUMOCNumPinsInSubDom_Y);
// 	_surfIndex_Layer_Y = (*GPUMOCNumPinsInSubDom_X) * ((*GPUMOCNumPinsInSubDom_Y) + 1);
// #endif // M_MG_CMFD

    _ltInd = _lay * (*GPUNumLTs_Total);
	_numPol=(*GPUNumPolarAng);
	
#if defined(M_3DMOCex)
	_numPol=_numPol/2;
#endif // defined(M_3DMOCex)

	_tid = threadIdx.x + blockIdx.x * blockDim.x;
	// _ltGlobID = _tid / (*GPUNumMOCGroups);
	// _groupID = _tid % (*GPUNumMOCGroups);
	_ltGlobID = _tid / _numGroupCalc;
	_groupID = _groupBoundLowLim + _tid % _numGroupCalc;
	_polEngID = _groupID * (*GPUNumPolarAng);

	track_flux_index = threadIdx.x * _numPol;

// #if defined(M_3DMOCex)
// 	_angMul2=(*GPUNumAzimAng)*_numPol;
// 	_angMul1=(*GPUNumMOCGroups)*_angMul2;
// 	_angId2 = _groupID*_angMul2;
// #endif // defined(M_3DMOCex)

	if (_ltGlobID < (*GPUNumLTs_Total)) {
		/* Retrieve pointer to thread's shared memory buffer for angular flux */
		_track_flux = &temp_flux[track_flux_index];
		
		for (_pol = _polarBoundLowLim; _pol < _polarBoundUpLim; _pol++) {
    		_track_flux[_pol-_polOffset] = _GPUBoundaryFlux(/*_ltInd +*/ _ltGlobID, _pol + _polEngID);
		}
		_azim = _GPULTsIndex_AzimAngID[_ltGlobID];

// #if defined(M_3DMOCex)
// 		_angId3 = _azim*_numPol;
// 		_spacing = GPUEffRaySpacing[_azim];
// #endif // defined(M_3DMOCex)

// #if defined(M_MG_CMFD)
//         _firstTrackInLT = 0;
// #endif // defined(M_MG_CMFD)
		/* Loop over each Track segment in forward direction */
		for (_track = _GPUTrackIndexInLTs[_ltGlobID]; _track < _GPUTrackIndexInLTs[_ltGlobID + 1]; _track++) {
			_cell = _GPUTracksInfoInLTs[_track].cell_ID;
			_ray = _GPUTracksInfoInLTs[_track].track_ID;
			_globCellID = _lay * _numCellsInDom + _cell;
            _trackIndex = (*GPUTotNumTracksInOneRTM_2PI)*_GPUCellRTMType_SubDom[_globCellID] + _GPUIndexTrackInRTM_SumAzim[_azim] + _ray;
// #if defined(M_MG_CMFD)
//             if (_firstTrackInLT == 0) {
//                 GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
//                     _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 1, \
//                     _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
//                     _GPURTMWidth_X, _GPURTMWidth_Y,_polarBoundLowLim, _polarBoundUpLim,_polOffset,_GPUTotalWeights_2DMOC);
//             }
// #endif // defined(M_MG_CMFD)
			for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {
				_length = _GPUSegInfoInTrack_2PI_length[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];
				_reg = _GPUSegInfoInTrack_2PI_FSRID[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg];

				_fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID] + _reg;
				_oneOverVolume = _GPUOneOverFSRsAreaInRTM[_GPUSumNumFSRsInRTM[_GPUCellRTMType_SubDom[_globCellID]] + _reg];
// #if defined(M_3DMOCex)
// 				_oneOverVolbySpacing = _oneOverVolume * _spacing;
				_layFsrID = _GPUSumNumFSRsBasedGlobCellID[_cell] + _reg;
// 				_angId1 = _layFsrID * _angMul1;
// #endif //  defined(M_3DMOCex)
				_index = _fsrID * (*GPUNumMOCGroups) + _groupID;
				_indexLay = _layFsrID * (*GPUNumMOCGroups) + _groupID;

// #if defined(M_3DMOCex)
// 				_angId = _angId1 + _angId2 + _angId3;
// #endif //  defined(M_3DMOCex)

			    GPUTallyScalarFlux_DDExpFunc_OneDir_AngFSmem_FixSourProb_2D(_index,_indexLay, _azim, _length, _oneOverVolume, \
					_GPUMacroXS_t, _track_flux, _GPUSource, _GPUScalarFlux, _polarBoundLowLim, _polarBoundUpLim, \
					_polOffset, _twoOverLayWidth,_oneOverVolbySpacing, _angId,_axialSurfAnfFlux_inp,_axialSurfAngFlux_update,\
					_GPUTotalWeights_2DMOC);

            }
// #if defined(M_MG_CMFD)
//             GPUGetParCurr_2DMOC_AngFSmem(_GPUMOCParCurr_X_Pos, _GPUMOCParCurr_X_Neg, _GPUMOCParCurr_Y_Pos, _GPUMOCParCurr_Y_Neg, \
//                 _track_flux, _ray, _azim, _groupID,_GPUNumTracksX, _GPUNumTracksY, 0, \
//                 _GPUCellRTMType_SubDom[_globCellID], _cell, _lay*_surfIndex_Layer_X, _lay*_surfIndex_Layer_Y,\
//                 _GPURTMWidth_X, _GPURTMWidth_Y,_polarBoundLowLim, _polarBoundUpLim,_polOffset,_GPUTotalWeights_2DMOC);
//             _firstTrackInLT=1;
// #endif // M_MG_CMFD
		}
		_azimNext = _GPULongTracksInfo[_ltGlobID].next_azim_angle_id;
		_ltIDNext = _GPULongTracksInfo[_ltGlobID].next_LongTrack_id_SubDom;
		_beta = _GPULongTracksInfo[_ltGlobID]._boundBeta;

#ifdef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme(_ltInd,_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUBoundaryFlux, _GPUMPIFlux, _beta, _polarBoundLowLim, _polarBoundUpLim, _polOffset);
#endif // MPI_CUDA_UpAngFImme

#ifndef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_AngFluxOnMem(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, _track_flux, _GPUStartFlux, _GPUMPIFlux, _beta);
#endif // !MPI_CUDA_UpAngFImme
	}
}

