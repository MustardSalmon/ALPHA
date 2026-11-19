/********************************************************************************/
/* @ File: MOC_GPUSolver.c
/*   introduction: This file contains functions that relate to solving the moc
/*                 neutron transport equation on GPU
/* @ Created     : 2018/05/25, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*   1st modified:
/********************************************************************************/
// #include <corecrt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <math.h>
//#include <math_functions.h>
//#include <math_functions.h>
#include <curand_kernel.h>
#include <time.h>
#include <cuda_occupancy.h>
#include "cublas_v2.h"

  #include "cusparse_v2.h"

#include "MOC_CommonMacro.h"

// #if defined(M_MG_CMFD)
  #include "MOC_GPUCmfd.cuh"
  #include "MOC_CMFD.h"
// #endif // defined(M_MG_CMFD)

#include "MOC_GPU_TSSolver.cuh"
#include "MOC_GPUConstants.cuh"
#include "MOC_GPUMatVec.cuh"
#include "MOC_Input.h"
#include "MOC_Output.h"
#include "MOC_GPUQuery.cuh"

#include "R_Sub_Fix.h"

#ifdef TESTR
    #include "condition.h"
    #include "zres_common_func.h"
#endif

extern "C" {
    #include "MOC_GPUSolver.cuh"
}
//extern int NumPinsInSubDomPerLayer;
extern cudaDeviceProp DeviceProp;

//int GPU_TSKernelType = 15; // 0-group level iteration-diam; 1-group level iteration-expCalcNoRec; 2-group level iteration-expCalcRec
//				 // 3-group level iteration-expTableNoRec; 4-group level iteration-expTableRec;

int GPUNumThreadBlocks=1536; /** The number of thread blocks */
int GPUNumThreadPerBlock=1024;/** The number of threads per thread block */ // 512(56regs)
//int GPUBlockDim = 128;
int GPUGridDim_group_biDir, GPUGridDim_group_oneDir, GPUGridDim_polar_biDir, GPUGridDim_polar_oneDir, GPUGridDim_track_biDir, GPUGridDim_track_oneDir;
int GPUGridDim_source_eval, GPUGridDim_source_para, GPUGridDim_LT_CMFDGroup, TotNumLTs_CMFDGroup;

int GPUParallelLevel=1; // 0-track, 1-group, 2-segment


//FP_PRECISION *GPUMacroXS_t=NULL; // MacroXS_t stored on GPU Mempry
FP_PRECISION *GPUMacroXS_t_wScale=NULL; // MacroXS_t + 2*cos(pol) / deltZ
FP_PRECISION *GPUMacroXS_a=NULL;
FP_PRECISION *GPUMacroXS_f=NULL;
FP_PRECISION *GPUMacroXS_nf=NULL;
FP_PRECISION *GPUMacroXS_s=NULL;
int *GPUMacroXS_s_NZRowStartID = NULL;
M_UINT64 *GPUMacroXS_s_NZRowStartID_Layer = NULL;
M_UINT64 *GPUMacroXS_s_NZRowEndID_Layer = NULL;
M_UINT64 *GPUMacroXS_s_NZColStartID = NULL;
FP_PRECISION *GPUMacroXS_chi=NULL;
// track information
int *GPULTsIndex_AzimAngID_HS=NULL;
int *GPULTsIndex_LTID_HS=NULL;
int *GPULTsIndex_AzimAngID_Total = NULL;
int *GPULTsIndex_LTID_Total = NULL;
int *GPUNumFSRs_Layer=NULL,*GPUSumNumFSRs_Layer=NULL;
FP_PRECISION_GEO *GPUFSRVol_SubDom=NULL;

struct SUBDOM_LONGTRACK_INFO *GPULongTracksInfo_Total = NULL; // long tracks information
// int GPUNumTracksInHalfSpace;
struct SUBDOM_TRACK_IN_LONGTRACK *GPUTracksInfoInLTs_HS = NULL; // tracks information in longtracks for [0,PI]
struct SUBDOM_TRACK_IN_LONGTRACK *GPUTracksInfoInLTs_Total = NULL; // tracks information in longtracks for [0,2PI]

//int *GPUNumTracksInLTs_HS=NULL;
//int __ldg(const int * GPUNumTracksInLTs_HS);
int *GPUTrackIndexInLTs_HS=NULL, *GPUTrackIndexInLTs_Total = NULL;
int *GPUNumSegsInTrack_HS=NULL, *GPUNumSegsInTrack_Total = NULL;

int *GPUIndexTrackInRTM_SumAzim_HS=NULL, *GPUIndexTrackInRTM_SumAzim_Total = NULL;
struct SUBDOM_SEG_IN_TRACK_SINGLE *GPUSegInfoInTrack_HS = NULL, *GPUSegInfoInTrack_Total = NULL;
int *GPUNumLTsSum=NULL;
int *GPUNumLongTracks=NULL;
// geometry information
int *GPUSumNumFSRsBasedGlobCellID=NULL;
int *GPUSumNumFSRsInRTM=NULL;
FP_PRECISION *GPUFSRsAreaInRTM=NULL;
// int GPUNumCellsPerLayer;
int *GPUCellRTMType_SubDom=NULL;
int *GPUSumNumSegsInRTMTrack_HS=NULL, *GPUSumNumSegsInRTMTrack_2PI = NULL;
FP_PRECISION /**GPUFSRsAreaInGlob=NULL, */*GPUOneOverFSRsAreaInGlob = NULL;
// exp table
FP_PRECISION *GPULinearExpTable;
FP_PRECISION *GPUExpTab_pitch;
FP_PRECISION *GPUExpTab_y0xk;


// flux array
//FP_PRECISION *GPUMOCSource=NULL;
FP_PRECISION *GPUFSRAveAngFlux_Top=NULL;
FP_PRECISION *GPUFSRAveAngFlux_Bott=NULL;
FP_PRECISION *GPUFSRAngScattSour=NULL;
FP_PRECISION *GPUFSRAngScatterSourceLM=NULL;
FP_PRECISION_FLUX *GPUScalarFluxAzimAccum=NULL;

/* Reusable CUDA events for the two kernel-level diagnostics requested for
 * the 2D high-order transport path.  The events are synchronized only after
 * the measured kernel, so the reported values are device execution times. */
static cudaEvent_t _timeKernelFlux103Start = NULL;
static cudaEvent_t _timeKernelFlux103Stop = NULL;
static cudaEvent_t _timeKernelFluxPerAzimStart = NULL;
static cudaEvent_t _timeKernelFluxPerAzimStop = NULL;
static cudaEvent_t _timeKernelFluxScalarInitStart = NULL;
static cudaEvent_t _timeKernelFluxScalarInitStop = NULL;
static cudaEvent_t _timeKernelHOSweep1Start = NULL;
static cudaEvent_t _timeKernelHOSweep1Stop = NULL;
static cudaEvent_t _timeKernelHOSweep11Start = NULL;
static cudaEvent_t _timeKernelHOSweep11Stop = NULL;

static void GPUKernelTimeStart(cudaEvent_t *_start, cudaEvent_t *_stop)
{
	if (*_start == NULL) {
		cudaEventCreate(_start);
		cudaEventCreate(_stop);
	}
	cudaEventRecord(*_start, 0);
}

static FP_PRECISION_TIME GPUKernelTimeStop(cudaEvent_t *_start, cudaEvent_t *_stop)
{
	float _elapsedMs = 0.0f;
	cudaEventRecord(*_stop, 0);
	cudaEventSynchronize(*_stop);
	cudaEventElapsedTime(&_elapsedMs, *_start, *_stop);
	return (FP_PRECISION_TIME) _elapsedMs * 1.0e-3;
}

static int GPUUseFsrClusterScattSource(void)
{
#ifdef TESTR
	ZRES_GI *_gi = zres_get_gen_info();
	return _gi != NULL && _gi->ifFsrCluster && zres_is_fsr_cluster_ready()
		&& IfUseExplicitScaRowEnd && MacroXS_s_NZRowEndID_Layer != NULL;
#else
	return 0;
#endif
}

#ifdef TESTR
#ifdef SAVE_ANGLE_FLUX
FP_PRECISION *GPUSldSpheHarmCoeff = NULL;
#endif
#endif

// float *GPUScalarFlux_test=NULL;
FP_PRECISION_FLUX *GPUScalarFluxOld=NULL;
FP_PRECISION_FLUX *GPUScalarFluxTemp=NULL;
FP_PRECISION *GPUBoundaryFlux=NULL;
FP_PRECISION *GPUStartFlux=NULL;
// FP_PRECISION *GPUFluxResidual=NULL;
FP_PRECISION *FluxResidual=NULL;
FP_PRECISION *GPUMPIFlux=NULL;
// calc

FP_PRECISION *GPUReacRateCollect=NULL;
FP_PRECISION *ReacRateCollect=NULL;

// FP_PRECISION GPUGlobFissOld,GPUKeff,GPUKeffOld;
// FP_PRECISION *HostScalarFlux_output=NULL;
//FP_PRECISION *GPUKeffRec=NULL;
//FP_PRECISION *GPUFluxResRec=NULL;
FP_PRECISION *host;
float TSKernel_Time = 0.;

#ifdef GPU_STREAM_USED
  FP_PRECISION *CGBoundaryFlux = NULL;
  FP_PRECISION *CGLTInciAngFlux = NULL;
  //int GPUNumStreams=3;
  int *NumLTsInStream = NULL;
  int *StartPosInLTs_Stream = NULL;
  //int *GPUStartPosInLTs_Stream = NULL;
  int *StartPosInFlux_Stream = NULL;
  int *AzimBound_Stream = NULL;
  cudaStream_t *GPUStreams = NULL;
  int *GPUNumBlocks_Stream;
#endif // GPU_STREAM_USED


/*modified 2019-06-15 by peitao song*/
FP_PRECISION *GPUSegInfoInTrack_2PI_length=NULL;
int *GPUSegInfoInTrack_2PI_RayWisedFSR = NULL, *GPUSegInfoInTrack_2PI_AntiClockFSR = NULL, *GPUFsrIdMap_RayWisedToAntiClock = NULL;
FP_PRECISION_FLUX *GPUScalarFlux_RayWisedFSR=NULL;
FP_PRECISION *GPUSource_RayWisedFSR=NULL, *GPUSource_RayWisedFSR_allAzim = NULL;
FP_PRECISION *GPUMacroXS_t_RayWisedFSR=NULL, *GPUMacroXS_t_RayWisedFSR_allAzim = NULL;

int *GPUNumFSRsInCell=NULL;

FP_PRECISION_GEO *GPUOneOverFSRsAreaInRTM_AntiClockFSR = NULL,*GPUOneOverFSRsAreaInRTM_RayWisedFSR=NULL;

cublasHandle_t GlobCublasHandle;

/*modified 2019-06-15 by peitao song - end*/

FP_PRECISION MOC_SORFactor=0.9;

/**
 * @ Function: GPUComputeFSRSourceOnDevice()
 *   introduction: calculate the source item of each FRSs in current domian
 *                 this function is called by host and excuted on device
 * @ Parameter List: _GPUScalarFlux - scalar flux for each FSR
 *                   _GPUSource - source iterm for each FSR
 *                   _GPUMacroXS_t - total XSs for each FSR
 *                   _GPUMacroXS_nf - n*sigmaF XSs for each FSR
 *                   _GPUMacroXS_s - scattering XSs for each FSR
 *                   _GPUMacroXS_chi - fission spectrum for each FSR
 *                   _inverseKeff - inverse of keff, =1.0/keff
 *   Return: void
 * @ Created     : 2018/05/25, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
/**
 * @ Function: GPUComputeFSRSourceOnDevice_paraPin_paraGroup()
 *   introduction: calculate the source item of each FRSs in current domian
 *                 this function is called by host and excuted on device
 * @ Parameter List: _GPUScalarFlux - scalar flux for each FSR
 *                   _GPUSource - source iterm for each FSR
 *                   _GPUMacroXS_t - total XSs for each FSR
 *                   _GPUMacroXS_nf - n*sigmaF XSs for each FSR
 *                   _GPUMacroXS_s - scattering XSs for each FSR
 *                   _GPUMacroXS_chi - fission spectrum for each FSR
 *                   _inverseKeff - inverse of keff, =1.0/keff
 *   Return: void
 * @ Created     : 2018/05/25, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
/**
 * GPUComputeFSRFissSourceOnDevice_paraFSR_paraGroup_Layer()
 * compute fission source on GPU
 * @param _GPUScalarFlux
 * @param _GPUSource
 * @param _GPUMacroXS_nf
 * @param _GPUMacroXS_chi
 * @param _inverseKeff
 */
// if a > b, return 0
// if a < b, return 1
__forceinline__ __device__ int magic(int a, int b) {
    int sign = (a-b) >> 31;
    return (sign+1);
}

// 需要测试裂变矩阵是否有效 [  WTDEBUG  ]
// 分离更新散射源和裂变源时使用的通量
__device__ FP_PRECISION GPURadialBoundary(int _angleInID, int _angleOutID)
{
	FP_PRECISION _beta=0.0;
	int _qurtNumAzim=(*GPUNumAzimAng)/4;

	if (_angleInID>=0&&_angleInID<_qurtNumAzim)
	{
		if (_angleOutID>=3*_qurtNumAzim && _angleOutID<4*_qurtNumAzim)
		{
			if ((*GPUNorthBC) == 0)
				_beta = 0.0;
			else if ((*GPUNorthBC) == 1)
				_beta = 1.0;
			else {}
		}
		else if (_angleOutID>=_qurtNumAzim && _angleOutID<2*_qurtNumAzim)
		{
			if ((*GPUEastBC) == 0)
				_beta = 0.0;
			else if ((*GPUEastBC) == 1)
				_beta = 1.0;
			else {}
		}
	}
	else if (_angleInID>=_qurtNumAzim && _angleInID<2*_qurtNumAzim)
	{
		if (_angleOutID>=2*_qurtNumAzim && _angleOutID<3*_qurtNumAzim)
		{
			if ((*GPUNorthBC) == 0)
				_beta = 0.0;
			else if ((*GPUNorthBC) == 1)
				_beta = 1.0;
			else {}
		}
		else if (_angleOutID>=0 && _angleOutID<_qurtNumAzim)
		{
			if ((*GPUWestBC) == 0)
				_beta = 0.0;
			else if ((*GPUWestBC) == 1)
				_beta = 1.0;
			else {}
		}
	}
	else if (_angleInID>=2*_qurtNumAzim && _angleInID<3*_qurtNumAzim)
	{
		if (_angleOutID>=_qurtNumAzim && _angleOutID<2*_qurtNumAzim)
		{
			if ((*GPUSouthBC) == 0)
				_beta = 0.0;
			else if ((*GPUSouthBC) == 1)
				_beta = 1.0;
			else {}
		}
		else if (_angleOutID>=3*_qurtNumAzim && _angleOutID<4*_qurtNumAzim)
		{
			if ((*GPUWestBC) == 0)
				_beta = 0.0;
			else if ((*GPUWestBC) == 1)
				_beta = 1.0;
			else {}
		}
	}
	else if (_angleInID>=3*_qurtNumAzim && _angleInID<4*_qurtNumAzim)
	{
		if (_angleOutID>=0 && _angleOutID<_qurtNumAzim)
		{
			if ((*GPUSouthBC) == 0)
				_beta = 0.0;
			else if ((*GPUSouthBC) == 1)
				_beta = 1.0;
			else {}
		}
		else if (_angleOutID>=2*_qurtNumAzim && _angleOutID<3*_qurtNumAzim)
		{
			if ((*GPUEastBC) == 0)
				_beta = 0.0;
			else if ((*GPUEastBC) == 1)
				_beta = 1.0;
			else {}
		}
	}
	return(_beta);
}

__device__ int GPUGetTrackInfo(int _lay,int _ltGlobID,int _azim,int _groupID,int _numCellsInDom,int *_GPUSumNumSegsInRTMTrack,\
    struct SUBDOM_SEG_IN_TRACK_SINGLE *_GPUSegInfoInTrack,struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs,\
    FP_PRECISION *_GPUOneOverFSRsAreaInGlob,FP_PRECISION *_GPUMacroXS_t,int *_GPUSumNumFSRsBasedGlobCellID,\
    int *_GPUTrackIndexInLTs,int *_GPUNumSegsInTrack,int *_GPUCellRTMType_SubDom,int *_GPUIndexTrackInRTM_SumAzim_HS,\
    int *_GPUGlobFSRIDTempJacobi,FP_PRECISION *_GPUOneOverVolTempJacobi, FP_PRECISION *_GPUOptLengthTempJacobi,int *_expX1,FP_PRECISION *_expX3)
{
    int _track,_cell,_ray,_globCellID,_trackIndex,_seg,_expCount,_index,_reg,_fsrID,_x1;
    FP_PRECISION _length,_oneOverVolume,_xs_t,_sigmaTByLength,_x2,_x3;
    _expCount = 0;
    for (_track=_GPUTrackIndexInLTs[_ltGlobID];_track<_GPUTrackIndexInLTs[_ltGlobID+1];_track++) {
        _cell = _GPUTracksInfoInLTs[_track].cell_ID;
        _ray = _GPUTracksInfoInLTs[_track].track_ID;
        _globCellID=_lay*_numCellsInDom+_cell;
        _trackIndex=(*GPUTotNumTracksInOneRTM_HS)*_GPUCellRTMType_SubDom[_globCellID]+_GPUIndexTrackInRTM_SumAzim_HS[_azim]+_ray;
        for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {
            _length = _GPUSegInfoInTrack[_GPUSumNumSegsInRTMTrack[_trackIndex]+_seg].length;
            _reg = _GPUSegInfoInTrack[_GPUSumNumSegsInRTMTrack[_trackIndex]+_seg].FSR_ID;
            _fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID]+_reg;
            _oneOverVolume = 1. /_GPUOneOverFSRsAreaInGlob[_fsrID];
            _index=_fsrID*(*GPUNumMOCGroups)+_groupID;
            _GPUOneOverVolTempJacobi[_expCount]=_oneOverVolume;
            _GPUGlobFSRIDTempJacobi[_expCount]=_index;
            _xs_t=_GPUMacroXS_t[_index];
            _sigmaTByLength = _xs_t * _length;
            _GPUOptLengthTempJacobi[_expCount]=_sigmaTByLength;
            _x2 = _sigmaTByLength * 8192;
            _x1 = (int)(_x2);
            _x3 = _x2 - _x1;
            _expX1[_expCount]=_x1;
            _expX3[_expCount]=_x3;
            _expCount++;
        }
    }
    return _expCount;
}

__device__ void GPUTallyScalarFlux_group_diam_LI(int _index, int _azim, FP_PRECISION _length, FP_PRECISION _oneOverVolume, \
	FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *track_flux, \
	FP_PRECISION *_GPUSource, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPULinearExpTable)
{
	int _x1, _pol;
	FP_PRECISION _xs_t, _sigmaTByLength, fsr_flux, _delta, _mul2, _fluxTemp,_x2,_x3, _exp/*_mul1, ,_angFlux*/;
	_xs_t = _GPUMacroXS_t[_index];
	_sigmaTByLength = _xs_t * _length;
	// _mul1 = (FP_PRECISION)(2.0)*_sigmaTByLength;
	_mul2 = (FP_PRECISION)(2.0)*_GPUSource[_index] * _sigmaTByLength;
	fsr_flux = (FP_PRECISION)(0.0);

	for (_pol = 0; _pol < (*GPUNumPolarAng); _pol++) {

		_fluxTemp = ((GPUSinThetasTimes2[_pol] - _sigmaTByLength)*track_flux[_pol] + _mul2) / (GPUSinThetasTimes2[_pol] + _sigmaTByLength);
		if (_fluxTemp <  (FP_PRECISION)(0.0)) {
			_x2 = _sigmaTByLength * GPUSinThetaForExp[_pol];
			_x1 = (int)(_x2);
			_x3 = _x2 - (FP_PRECISION)(_x1);
			_exp = _GPULinearExpTable[_x1] + _x3 * (_GPULinearExpTable[_x1 + 1] - _GPULinearExpTable[_x1]);
			_delta = (track_flux[ + _pol] - _GPUSource[_index])*_exp;
			track_flux[_pol] -= _delta;
		}
		else {
			_delta = track_flux[_pol] - _fluxTemp;
			track_flux[_pol] = _fluxTemp;
		}
		fsr_flux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + _pol];
	}
	fsr_flux *= _oneOverVolume;
	/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif
}


__device__ void GPUTallyScalarFlux_track_diam(int _fsrID, int _azim, FP_PRECISION _length, FP_PRECISION _oneOverVolume, \
	FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *track_flux, FP_PRECISION *_GPUSource, FP_PRECISION_FLUX *_GPUScalarFlux)
{
// 	int _groupID, _polEngID, _index, _pol;
// 	FP_PRECISION _xs_t, _sigmaTByLength, fsr_flux, _delta, _mul1, _mul2/*, _fluxTemp, _x2, _x3, _exp, _angFlux*/;
// 	for (_groupID = 0; _groupID < (*GPUNumMOCGroups); _groupID++) {
// 		_polEngID = _groupID * (*GPUNumPolarAng);
// 		_index = _fsrID * (*GPUNumMOCGroups) + _groupID;
// 		_xs_t = _GPUMacroXS_t[_index];
// 		_sigmaTByLength = _xs_t * _length;

// 		_mul1 = (FP_PRECISION)(2.0)*_sigmaTByLength;
// 		_mul2 = (FP_PRECISION)(2.0)*_GPUSource[_index] * _sigmaTByLength;
// 		fsr_flux = (FP_PRECISION)(0.0);

// 		for (_pol = 0; _pol < (*GPUNumPolarAng); _pol++) {
// 			_delta = (_mul1*track_flux[_polEngID + _pol] - _mul2) / (GPUSinThetasTimes2[_pol] + _sigmaTByLength);
// 			fsr_flux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + _pol];
// 			track_flux[_polEngID + _pol] -= _delta;
// 		}
// 		fsr_flux *= _oneOverVolume;
// 		/* Atomically increment the scalar flux for this FSR */
// #ifdef DOUBLE_FLUX
// 		atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
// #endif

// #ifdef SINGLE_FLUX
// 		atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
// #endif

// 	}
}

__device__ void GPUTallyScalarFlux_track_expNoRec(int _fsrID, int _azim, FP_PRECISION _length, FP_PRECISION _oneOverVolume, \
	FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *track_flux, FP_PRECISION *_GPUSource, FP_PRECISION_FLUX *_GPUScalarFlux)
{
// 	int _groupID, _polEngID, _index, _pol;
// 	FP_PRECISION _xs_t, _sigmaTByLength, fsr_flux, _delta/*, _mul1, _mul2*/, _exp;
// 	for (_groupID = 0; _groupID < (*GPUNumMOCGroups); _groupID++) {
// 		_polEngID = _groupID * (*GPUNumPolarAng);
// 		_index = _fsrID * (*GPUNumMOCGroups) + _groupID;
// 		_xs_t = _GPUMacroXS_t[_index];
// 		_sigmaTByLength = _xs_t * _length;

// 		fsr_flux = (FP_PRECISION)(0.0);
// 		for (_pol = 0; _pol < (*GPUNumPolarAng); _pol++) {
// 			_exp = 1.0 - exp(-_sigmaTByLength * GPUOneOverSinThetas[_pol]);
// 			_delta = (track_flux[_polEngID + _pol] - _GPUSource[_index])*_exp;
// 			fsr_flux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + _pol];
// 			track_flux[_polEngID + _pol] -= _delta;
// 		}
// 		fsr_flux *= _oneOverVolume;
// 		/* Atomically increment the scalar flux for this FSR */
// #ifdef DOUBLE_FLUX
// 		atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
// #endif

// #ifdef SINGLE_FLUX
// 		atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
// #endif

// 	}
}


__device__ void GPUTallyScalarFlux_track_expTabNoRec(int _fsrID, int _azim, FP_PRECISION _length, FP_PRECISION _oneOverVolume, \
	FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *track_flux, FP_PRECISION *_GPUSource, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPULinearExpTable)
{
// 	int _groupID, _polEngID, _index, _pol, _x1;
// 	FP_PRECISION _xs_t, _sigmaTByLength, fsr_flux, _delta/*, _mul1, _mul2*/, _exp, _x2,_x3;
// 	for (_groupID = 0; _groupID < (*GPUNumMOCGroups); _groupID++) {
// 		_polEngID = _groupID * (*GPUNumPolarAng);
// 		_index = _fsrID * (*GPUNumMOCGroups) + _groupID;
// 		_xs_t = _GPUMacroXS_t[_index];
// 		_sigmaTByLength = _xs_t * _length;

// 		fsr_flux = (FP_PRECISION)(0.0);
// 		for (_pol = 0; _pol < (*GPUNumPolarAng); _pol++) {
// 			_x2 = _sigmaTByLength * GPUSinThetaForExp[_pol];
// 			_x1 = (int)(_x2);
// 			_x3 = _x2 - (FP_PRECISION)(_x1);
// 			_exp = _GPULinearExpTable[_x1] + _x3 * (_GPULinearExpTable[_x1 + 1] - _GPULinearExpTable[_x1]);
// 			_delta = (track_flux[_polEngID + _pol] - _GPUSource[_index])*_exp;
// 			fsr_flux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + _pol];
// 			track_flux[_polEngID + _pol] -= _delta;
// 		}
// 		fsr_flux *= _oneOverVolume;
// 		/* Atomically increment the scalar flux for this FSR */
// #ifdef DOUBLE_FLUX
// 		atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
// #endif

// #ifdef SINGLE_FLUX
// 		atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
// #endif

// 	}
}



__device__ void Experimrnt_Diam_CalDelta(int _index, FP_PRECISION _length,FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *track_flux, FP_PRECISION *_GPUSource)
{
	//int _pol;
	//FP_PRECISION _xs_t, _sigmaTByLength/*, _delta*/, _mul1, _mul2;
	//_xs_t = _GPUMacroXS_t[_index];
	//_sigmaTByLength = _xs_t * _length;
	//_mul1 = 2.0*_sigmaTByLength;
	//_mul2 = 2.0*_GPUSource[_index] * _sigmaTByLength;
	//for (_pol = 0; _pol < (*GPUNumPolarAng); _pol++) {
	//	_delta = (_mul1*track_flux[_pol] - _mul2) / (2.0* GPUSinThetas[_pol] + _sigmaTByLength);
	//}
}

__device__ FP_PRECISION GPUTallyScalarFlux_polar_diam(int _index, int _azim,int _pol, FP_PRECISION _length, FP_PRECISION _oneOverVolume, \
	FP_PRECISION *_GPUMacroXS_t, FP_PRECISION track_flux, \
	FP_PRECISION *_GPUSource, FP_PRECISION_FLUX *_GPUScalarFlux)
{
	FP_PRECISION _xs_t, _sigmaTByLength, fsr_flux, _delta/*, _exp*/,_flux/*,_const*/;
	//_const = (FP_PRECISION)(2.0);
	_flux = track_flux;
	_xs_t = _GPUMacroXS_t[_index];
	_sigmaTByLength = _xs_t * _length;
	_delta = (/*_const*/(FP_PRECISION)(2.0)*_sigmaTByLength*track_flux - /*_const*/(FP_PRECISION)(2.0) *_GPUSource[_index] * _sigmaTByLength) / (GPUSinThetasTimes2[_pol] + _sigmaTByLength);
	fsr_flux = _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + _pol] * _oneOverVolume;
	_flux -= _delta;
	/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif
	return _flux;
}

__device__ FP_PRECISION GPUTallyScalarFlux_polar_expNoRec(int _index, int _azim, int _pol, FP_PRECISION _length, FP_PRECISION _oneOverVolume, \
	FP_PRECISION *_GPUMacroXS_t, FP_PRECISION track_flux, FP_PRECISION *_GPUSource, FP_PRECISION_FLUX *_GPUScalarFlux)
{
	FP_PRECISION _xs_t, _sigmaTByLength, fsr_flux, _delta, _exp, _flux/*,_const*/;
	//_const = (FP_PRECISION)(2.0);
	_flux = track_flux;
	_xs_t = _GPUMacroXS_t[_index];
	_sigmaTByLength = _xs_t * _length;
	_exp = 1.0 - _ExpFunction_GPU(-_sigmaTByLength * GPUOneOverSinThetas[_pol]);
	_delta = (_flux - _GPUSource[_index])*_exp;
	fsr_flux = _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + _pol] * _oneOverVolume;
	_flux -= _delta;
	/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif
	return _flux;
}

__device__ FP_PRECISION GPUTallyScalarFlux_polar_expTabNoRec(int _index, int _azim, int _pol, FP_PRECISION _length, FP_PRECISION _oneOverVolume, \
	FP_PRECISION *_GPUMacroXS_t, FP_PRECISION track_flux, FP_PRECISION *_GPUSource, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPULinearExpTable)
{
	FP_PRECISION _xs_t, _sigmaTByLength, fsr_flux, _delta, _exp, _flux, _x2, _x3/*,_const*/;
	int _x1;
	//_const = (FP_PRECISION)(2.0);
	_flux = track_flux;
	_xs_t = _GPUMacroXS_t[_index];
	_sigmaTByLength = _xs_t * _length;
	_x2 = _sigmaTByLength * GPUSinThetaForExp[_pol];
	_x1 = (int)(_x2);
	_x3 = _x2 - (FP_PRECISION)(_x1);
	_exp = _GPULinearExpTable[_x1] + _x3 * (_GPULinearExpTable[_x1 + 1] - _GPULinearExpTable[_x1]);
	_delta = (_flux - _GPUSource[_index])*_exp;
	fsr_flux = _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + _pol] * _oneOverVolume;
	_flux -= _delta;
	/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif
	return _flux;
}


__device__ void GPUTallyScalarFluxReverse_expRec_AngFluxOnMem(int _index, int _azim, FP_PRECISION _length, FP_PRECISION _oneOverVolume, \
	FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *track_flux, \
	FP_PRECISION *_GPUSource, FP_PRECISION_EXP *_GPUExpTempJacobi, int _expCount, FP_PRECISION_FLUX *_GPUScalarFlux)
{
	int _pol;
	FP_PRECISION fsr_flux, _delta;
	FP_PRECISION_EXP _exp;

	fsr_flux = 0.0;
	for (_pol = 0; _pol < (*GPUNumPolarAng); _pol++) {
		_exp = _GPUExpTempJacobi[_expCount*(*GPUNumPolarAng) + _pol];
		_delta = (track_flux[_pol] - _GPUSource[_index])*_exp;
		fsr_flux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + _pol]/* * _oneOverVolume*/;
		track_flux[_pol] -= _delta;
	}
	fsr_flux *= _oneOverVolume;
	/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	// atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif
}


__device__ void GPUTallyScalarFlux_expTabNoRec_AngFSmem(int _index,int _azim,FP_PRECISION _length,FP_PRECISION _oneOverVolume,\
    FP_PRECISION *_GPUMacroXS_t,FP_PRECISION *track_flux,FP_PRECISION *_GPULinearExpTable,\
    FP_PRECISION *_GPUSource,FP_PRECISION_FLUX *_GPUScalarFlux)
{
    int _x1,_pol;
    FP_PRECISION _xs_t,_sigmaTByLength,fsr_flux,_x2,_x3,_delta,_exp;
    _xs_t=_GPUMacroXS_t[_index];
    _sigmaTByLength = _xs_t * _length;
    fsr_flux=0.0;
    for (_pol = 0; _pol < (*GPUNumPolarAng); _pol++) {
		_x2 = _sigmaTByLength * GPUSinThetaForExp[_pol];
		_x1 = (int)(_x2);
		_x3 = _x2 - _x1;
		_exp = _GPULinearExpTable[_x1] + _x3 * (_GPULinearExpTable[_x1 + 1] - _GPULinearExpTable[_x1]);
        _delta = (track_flux[_pol] - _GPUSource[_index])*_exp;
        fsr_flux += _delta *GPUTotalWeights[_azim*(*GPUNumPolarAng)+_pol];
        track_flux[_pol] -= _delta;
    }
	fsr_flux *= _oneOverVolume;
    /* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	// atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif
}

__device__ void GPUTallyScalarFlux_NewExpTab_noRec(int _index, int _azim, FP_PRECISION _length, FP_PRECISION _oneOverVolume, \
	FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *track_flux, FP_PRECISION *_GPUExpTab_pitch, FP_PRECISION *_GPUExpTab_y0xk, \
	FP_PRECISION *_GPUSource, FP_PRECISION_FLUX *_GPUScalarFlux)
{
	int _x1, _pol;
	FP_PRECISION _xs_t, _sigmaTByLength, fsr_flux, _x2/*, _x3*/, _delta, _exp;
	_xs_t = _GPUMacroXS_t[_index];
	_sigmaTByLength = _xs_t * _length;
	fsr_flux = 0.0;
	for (_pol = 0; _pol < (*GPUNumPolarAng); _pol++) {
		_x2 = _sigmaTByLength * GPUOneOverSinThetas[_pol];
		_x1 = (int)(_x2*NumExpPoints_One_Mod1);
		//_x3 = _x2 - _x1;
		_exp = _GPUExpTab_y0xk[_x1] + _x2 * _GPUExpTab_pitch[_x1];

		_delta = (track_flux[_pol] - _GPUSource[_index])*(1.0-_exp);
		fsr_flux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + _pol];
		track_flux[_pol] -= _delta;
	}
	fsr_flux *= _oneOverVolume;
	/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	// atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif
}


__device__ void GPUTallyScalarFlux_expNoRec_AngFluxOnMem(int _index, int _azim, FP_PRECISION _length, FP_PRECISION _oneOverVolume, \
	FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *track_flux, \
	FP_PRECISION *_GPUSource, FP_PRECISION_FLUX *_GPUScalarFlux)
{
	int _pol;
	FP_PRECISION _xs_t, _sigmaTByLength, fsr_flux, _delta;
	FP_PRECISION_EXP _exp;
	_xs_t = _GPUMacroXS_t[_index];
	_sigmaTByLength = _xs_t * _length;
	fsr_flux = 0.0;
	for (_pol = 0; _pol < (*GPUNumPolarAng); _pol++) {
		_exp = (FP_PRECISION_EXP)1.0 - _ExpFunction_GPU((FP_PRECISION_EXP)(-_sigmaTByLength * GPUOneOverSinThetas[_pol]));
		_delta = (track_flux[_pol] - _GPUSource[_index])*_exp;
		fsr_flux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + _pol]/* * _oneOverVolume*/;
		track_flux[_pol] -= _delta;
	}
	fsr_flux *= _oneOverVolume;
	/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
	// atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif

#ifdef SINGLE_FLUX
	atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif
}

__device__ void GPUTallyScalarFluxReverse(int _expCount,int _groupID,int _azimReverse,int *_GPUGlobFSRIDTempJacobi,\
    FP_PRECISION *_GPUOneOverVolTempJacobi,FP_PRECISION *_GPUExpTempJacobi,FP_PRECISION *track_flux,\
    FP_PRECISION *_GPUSource,FP_PRECISION_FLUX *_GPUScalarFlux)
{
    int _fsrID,_index,_pol/*,_numSeg*/;
    FP_PRECISION _oneOverVolume,_delta,fsr_flux;
    _expCount--;
    while (_expCount>=0) {
        fsr_flux=0.0;
        _fsrID=_GPUGlobFSRIDTempJacobi[_expCount];
        _oneOverVolume=_GPUOneOverVolTempJacobi[_expCount];
        _index=_fsrID*(*GPUNumMOCGroups)+_groupID;
        for (_pol=0;_pol<(*GPUNumPolarAng);_pol++) {
            _delta = (track_flux[_pol] - _GPUSource[_index])*_GPUExpTempJacobi[_expCount*(*GPUNumPolarAng)+_pol];
            fsr_flux += _delta*GPUTotalWeights[_azimReverse*(*GPUNumPolarAng)+_pol] * _oneOverVolume;
            track_flux[_pol] -= _delta;
        }
#ifdef DOUBLE_FLUX
		// atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif

#ifdef SINGLE_FLUX
		atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif
        _expCount--;
    }
}

__device__ void GPUUpdateBoundaryFlux_AngFluxOnMem(int _azim,int _azimNext,int *_GPUNumLTsSum,int _ltIDNext,int _polEngID,\
    int _ltGlobID,FP_PRECISION *track_flux,FP_PRECISION *_GPUStartFlux,FP_PRECISION *_GPUMPIFlux, FP_PRECISION _beta)
{
    int/* _beta,*/ _pol;
     if (_azim != _azimNext) {
        //_beta = GPURadialBoundary(_azim, _azimNext);
        for (_pol=0;_pol<(*GPUNumPolarAng);_pol++) {
            _GPUStartFlux(_GPUNumLTsSum[_azimNext]+_ltIDNext,_pol+_polEngID)=_beta*track_flux[_pol];
        }
     }
     else {
         for (_pol=0;_pol<(*GPUNumPolarAng);_pol++) {
             _GPUMPIFlux(_ltGlobID,_pol+_polEngID)=track_flux[_pol];
         }
     }
}



__device__ void GPUUpdateBoundaryFlux_track(int _azim, int _azimNext, int *_GPUNumLTsSum, int _ltIDNext, \
	int _ltGlobID, FP_PRECISION *track_flux, FP_PRECISION *_GPUStartFlux, FP_PRECISION *_GPUMPIFlux,FP_PRECISION _beta)
{
	int /*_beta, */_index;
	if (_azim != _azimNext) {
		//_beta = GPURadialBoundary(_azim, _azimNext);
		for (_index = 0; _index<(*GPUPolarTimesGroups); _index++) {
			_GPUStartFlux(_GPUNumLTsSum[_azimNext] + _ltIDNext, _index) = _beta * track_flux[_index];
		}
	}
	else {
		for (_index = 0; _index<(*GPUPolarTimesGroups); _index++) {
			_GPUMPIFlux(_ltGlobID, _index) = track_flux[_index];
		}
	}
}


__device__ void GPUUpdateBoundaryFlux_polar(int _azim,int _pol, int _azimNext, int *_GPUNumLTsSum, int _ltIDNext, int _polEngID, \
	int _ltGlobID, FP_PRECISION track_flux, FP_PRECISION *_GPUStartFlux, FP_PRECISION *_GPUMPIFlux, FP_PRECISION _beta)
{
	//FP_PRECISION _beta;
	if (_azim != _azimNext) {
		//_beta = GPURadialBoundary(_azim, _azimNext);
		_GPUStartFlux(_GPUNumLTsSum[_azimNext] + _ltIDNext, _pol + _polEngID) = _beta * track_flux;
	}
	else {
		_GPUMPIFlux(_ltGlobID, _pol + _polEngID) = track_flux;
	}
}




__global__ void GPUTransportSweepOnDevice_group_diam_oneDir_AngFReg_Stream(FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, FP_PRECISION *_GPUBoundaryFlux, \
	FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, \
	struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, int *_GPUTrackIndexInLTs, int _lay, int _numCellsInDom/* , int _GPUTotNumTracksInOneRTM_HS */, \
	int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, int *_GPUNumSegsInTrack, struct SUBDOM_SEG_IN_TRACK_SINGLE *_GPUSegInfoInTrack, \
	int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, FP_PRECISION *_GPUOneOverFSRsAreaInGlob, int *_GPUSumNumFSRsInRTM, \
	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, int _LTIDOffset, int _numLTsUpBound_stream)
{
	int _tid, _ltGlobID/*, _ltID*/, _groupID, _azim, _azimNext, _ltIDNext, _polEngID,/* _pol, track_flux_index,*/ _cell, _ray, _globCellID, _trackIndex;
	int _seg, _reg, _fsrID, _track/*, _expCount*/, _index/*, _azimReverse, _ltIDReverse, _ltGlobIDReverse ,_x1 */;
	FP_PRECISION _length, _oneOverVolume;
	FP_PRECISION _xs_t, _sigmaTByLength, fsr_flux, _delta, _mul1, _mul2;
	/* Shared memory buffer for each thread's angular flux */
	//extern __shared__ FP_PRECISION temp_flux[];
	FP_PRECISION angFlux1, angFlux2, angFlux0, _beta;
	_tid = threadIdx.x + blockIdx.x * blockDim.x;
	_ltGlobID = _tid / (*GPUNumMOCGroups) + _LTIDOffset;
	_groupID = _tid % (*GPUNumMOCGroups);
	_polEngID = _groupID * (*GPUNumPolarAng);
	//track_flux_index = threadIdx.x * (*GPUNumPolarAng);

	while (_ltGlobID < _numLTsUpBound_stream/*(*GPUNumLTs_Total)*/) {
		/* Retrieve pointer to thread's shared memory buffer for angular flux */
		_azim = _GPULTsIndex_AzimAngID[_ltGlobID];

		angFlux0 = _GPUBoundaryFlux(_ltGlobID, _polEngID);
		angFlux1 = _GPUBoundaryFlux(_ltGlobID, 1 + _polEngID);
		angFlux2 = _GPUBoundaryFlux(_ltGlobID, 2 + _polEngID);

		/* Loop over each Track segment in forward direction */
		for (_track = _GPUTrackIndexInLTs[_ltGlobID]; _track < _GPUTrackIndexInLTs[_ltGlobID + 1]; _track++) {
			_cell = _GPUTracksInfoInLTs[_track].cell_ID;
			_ray = _GPUTracksInfoInLTs[_track].track_ID;
			_globCellID = _lay * _numCellsInDom + _cell;
			_trackIndex = (*GPUTotNumTracksInOneRTM_2PI)*_GPUCellRTMType_SubDom[_globCellID] + _GPUIndexTrackInRTM_SumAzim[_azim] + _ray;
			for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {
				_length = _GPUSegInfoInTrack[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg].length;
				_reg = _GPUSegInfoInTrack[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg].FSR_ID;
				_fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID] + _reg;
				_oneOverVolume = /*(FP_PRECISION)(1.0) /*/ _GPUOneOverFSRsAreaInGlob[_fsrID];
				_index = _fsrID * (*GPUNumMOCGroups) + _groupID;

				_xs_t = _GPUMacroXS_t[_index];
				_sigmaTByLength = _xs_t * _length;
				_mul1 = (FP_PRECISION)(2.0)*_sigmaTByLength;
				_mul2 = (FP_PRECISION)(2.0)*_GPUSource[_index] * _sigmaTByLength;
				fsr_flux = (FP_PRECISION)(0.0);

				_delta = (_mul1*angFlux0 - _mul2) / (GPUSinThetasTimes2[0] + _sigmaTByLength);
				fsr_flux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + 0];
				angFlux0 -= _delta;
				_delta = (_mul1*angFlux1 - _mul2) / (GPUSinThetasTimes2[1] + _sigmaTByLength);
				fsr_flux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + 1];
				angFlux1 -= _delta;
				_delta = (_mul1*angFlux2 - _mul2) / (GPUSinThetasTimes2[2] + _sigmaTByLength);
				fsr_flux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + 2];
				angFlux2 -= _delta;
				fsr_flux *= _oneOverVolume;

				/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
				atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif

#ifdef SINGLE_FLUX
				atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif

			}
		}
		_azimNext = _GPULongTracksInfo[_ltGlobID].next_azim_angle_id;
		_ltIDNext = _GPULongTracksInfo[_ltGlobID].next_LongTrack_id_SubDom;
		_beta = _GPULongTracksInfo[_ltGlobID]._boundBeta;

#ifdef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_3polar_1(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, angFlux0, angFlux1, angFlux2, _GPUBoundaryFlux, _GPUMPIFlux, _beta);
#endif // MPI_CUDA_UpAngFImme

#ifndef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_3polar(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, angFlux0, angFlux1, angFlux2/*0*/, _GPUStartFlux, _GPUMPIFlux, _beta);
#endif // !MPI_CUDA_UpAngFImme

        _ltGlobID += blockDim.x * gridDim.x;
	}
}


__global__ void GPUTransportSweepOnDevice_group_diam_oneDir_AngFReg_Azim(FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, FP_PRECISION *_GPUBoundaryFlux, \
	FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, \
	struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, int *_GPUTrackIndexInLTs, int _lay, int _numCellsInDom/* , int _GPUTotNumTracksInOneRTM_HS */, \
	int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, int *_GPUNumSegsInTrack, struct SUBDOM_SEG_IN_TRACK_SINGLE *_GPUSegInfoInTrack, \
	int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, FP_PRECISION *_GPUOneOverFSRsAreaInGlob, int *_GPUSumNumFSRsInRTM, \
	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux,int _GPUSumNumLTs_Total_Azim)
{
	int _tid, _ltGlobID/*, _ltID*/, _groupID, _azim, _azimNext, _ltIDNext, _polEngID,/* _pol, track_flux_index,*/ _cell, _ray, _globCellID, _trackIndex;
	int _seg, _reg, _fsrID, _track/*, _expCount*/, _index/*, _azimReverse, _ltIDReverse, _ltGlobIDReverse ,_x1 */;
	FP_PRECISION _length, _oneOverVolume;
	FP_PRECISION _xs_t, _sigmaTByLength, fsr_flux, _delta, _mul1, _mul2;
	/* Shared memory buffer for each thread's angular flux */
	//extern __shared__ FP_PRECISION temp_flux[];
	FP_PRECISION angFlux1, angFlux2, angFlux0, _beta;
	_tid = threadIdx.x + blockIdx.x * blockDim.x;
	_ltGlobID = _tid / (*GPUNumMOCGroups) + _GPUSumNumLTs_Total_Azim;
	_groupID = _tid % (*GPUNumMOCGroups);
	_polEngID = _groupID * (*GPUNumPolarAng);
	//track_flux_index = threadIdx.x * (*GPUNumPolarAng);

	while (_ltGlobID < (*GPUNumLTs_Total)) {
		/* Retrieve pointer to thread's shared memory buffer for angular flux */
		_azim = _GPULTsIndex_AzimAngID[_ltGlobID];

		angFlux0 = _GPUBoundaryFlux(_ltGlobID, _polEngID);
		angFlux1 = _GPUBoundaryFlux(_ltGlobID, 1 + _polEngID);
		angFlux2 = _GPUBoundaryFlux(_ltGlobID, 2 + _polEngID);

		/* Loop over each Track segment in forward direction */
		for (_track = _GPUTrackIndexInLTs[_ltGlobID]; _track < _GPUTrackIndexInLTs[_ltGlobID + 1]; _track++) {
			_cell = _GPUTracksInfoInLTs[_track].cell_ID;
			_ray = _GPUTracksInfoInLTs[_track].track_ID;
			_globCellID = _lay * _numCellsInDom + _cell;
			_trackIndex = (*GPUTotNumTracksInOneRTM_2PI)*_GPUCellRTMType_SubDom[_globCellID] + _GPUIndexTrackInRTM_SumAzim[_azim] + _ray;
			for (_seg = 0; _seg < _GPUNumSegsInTrack[_trackIndex]; _seg++) {
				_length = _GPUSegInfoInTrack[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg].length;
				_reg = _GPUSegInfoInTrack[_GPUSumNumSegsInRTMTrack[_trackIndex] + _seg].FSR_ID;
				_fsrID = _GPUSumNumFSRsBasedGlobCellID[_globCellID] + _reg;
				_oneOverVolume = /*(FP_PRECISION)(1.0) /*/ _GPUOneOverFSRsAreaInGlob[_fsrID];
				_index = _fsrID * (*GPUNumMOCGroups) + _groupID;

				_xs_t = _GPUMacroXS_t[_index];
				_sigmaTByLength = _xs_t * _length;
				_mul1 = (FP_PRECISION)(2.0)*_sigmaTByLength;
				_mul2 = (FP_PRECISION)(2.0)*_GPUSource[_index] * _sigmaTByLength;
				fsr_flux = (FP_PRECISION)(0.0);

				_delta = (_mul1*angFlux0 - _mul2) / (GPUSinThetasTimes2[0] + _sigmaTByLength);
				fsr_flux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + 0];
				angFlux0 -= _delta;
				_delta = (_mul1*angFlux1 - _mul2) / (GPUSinThetasTimes2[1] + _sigmaTByLength);
				fsr_flux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + 1];
				angFlux1 -= _delta;
				_delta = (_mul1*angFlux2 - _mul2) / (GPUSinThetasTimes2[2] + _sigmaTByLength);
				fsr_flux += _delta * GPUTotalWeights[_azim*(*GPUNumPolarAng) + 2];
				angFlux2 -= _delta;
				fsr_flux *= _oneOverVolume;

				/* Atomically increment the scalar flux for this FSR */
#ifdef DOUBLE_FLUX
				atomicAdd_(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif

#ifdef SINGLE_FLUX
				atomicAdd(&_GPUScalarFlux[_index], (FP_PRECISION_FLUX)fsr_flux);
#endif

			}
		}
		_azimNext = _GPULongTracksInfo[_ltGlobID].next_azim_angle_id;
		_ltIDNext = _GPULongTracksInfo[_ltGlobID].next_LongTrack_id_SubDom;
		_beta = _GPULongTracksInfo[_ltGlobID]._boundBeta;

#ifdef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_3polar_1(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, angFlux0, angFlux1, angFlux2, _GPUBoundaryFlux, _GPUMPIFlux, _beta);
#endif // MPI_CUDA_UpAngFImme

#ifndef MPI_CUDA_UpAngFImme
		GPUUpdateBoundaryFlux_3polar(_azim, _azimNext, _GPUNumLTsSum, _ltIDNext, _polEngID, \
			_ltGlobID, angFlux0, angFlux1, angFlux2/*0*/, _GPUStartFlux, _GPUMPIFlux, _beta);
#endif // !MPI_CUDA_UpAngFImme

        _ltGlobID += blockDim.x * gridDim.x;
	}
}



/**
 * @ Function: GPUAddSourceToScalarFluxOnDevice()
 *   introduction: add the FSR source to scalar flux
 *                 this function is called by host and excuted on device
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2018/05/29, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
__global__ void GPUAddSourceToScalarFluxOnDevice(FP_PRECISION_FLUX *_GPUScalarFlux,FP_PRECISION *_GPUSource,FP_PRECISION *_GPUMacroXS_t)
{
    int _tid = threadIdx.x + blockIdx.x * blockDim.x;
    int g,_numG,_numFSRs,_tIDbyNG,_id;
    FP_PRECISION_FLUX _flux;
    _numFSRs=(*GPUTotNumFSRs);
    _numG=(*GPUNumMOCGroups);
    while (_tid< _numFSRs) {
        _tIDbyNG=_tid*_numG;
        for (g=0;g<_numG;g++) {
            _id=_tIDbyNG+g;
            _flux = _GPUScalarFlux[_id]/_GPUMacroXS_t[_id];
            // _flux = 2.0 * _GPUScalarFlux[_id]/_GPUMacroXS_t[_id];
			_GPUScalarFlux[_id]=(FP_PRECISION_FLUX)(_flux+FOUR_PI*_GPUSource[_id]);
#if defined(M_ZeroNegFlux)
			if (_GPUScalarFlux[_id]<0.0) {
				_GPUScalarFlux[_id]=0.0;
				}
				#endif
				}
        /* Increment thread id */
        _tid += blockDim.x * gridDim.x;
    }
}

__global__ void GPUAddSourceToScalarFluxOnDevice_lay_g(int _numFSRsPerLay, int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, \
	int _groupBoundUpLim, FP_PRECISION_FLUX *_GPUScalarFlux,FP_PRECISION *_GPUSource,FP_PRECISION *_GPUMacroXS_t)
{
    int _tid = threadIdx.x + blockIdx.x * blockDim.x;
    int _id, _calcFSRs, _calcGroup, _totCalc, _fsrID, _groupID,_fsrIDLay, _idLay;
	FP_PRECISION_FLUX _flux;
	_calcFSRs = _numFSRsPerLay*(_layBoundUpLim-_layBoundLowLim);
	_calcGroup = _groupBoundUpLim-_groupBoundLowLim;
	_totCalc = _calcFSRs*_calcGroup;

	while (_tid< _totCalc) {
		_fsrID=_layBoundLowLim*_numFSRsPerLay + _tid/_calcGroup;
		_fsrIDLay = _tid/_calcGroup;
		_groupID = _tid%_calcGroup;
		_id=_fsrID*(*GPUNumMOCGroups)+_groupID;
		_idLay=_fsrIDLay*(*GPUNumMOCGroups)+_groupID;
		_flux=_GPUScalarFlux[_idLay]/_GPUMacroXS_t[_idLay];
		_GPUScalarFlux[_idLay]=(FP_PRECISION_FLUX)(_flux+FOUR_PI*_GPUSource[_idLay]);
		
		_tid += blockDim.x * gridDim.x;
	}
}

/**
 * @ Function: GPUUpdateScalarFluxOnDevice()
 *   introduction: store the latest scalar flux talled by transport sweep into the ScalarFluxOld array
 *                 this is used for calculating the residual of scalar flux
 *                 this function is called by host and excuted on device
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2018/05/30, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
__global__ void GPUUpdateScalarFluxOnDevice(FP_PRECISION_FLUX *_GPUScalarFlux,FP_PRECISION *_GPUScalarFluxOld)
{
    int _tid = threadIdx.x + blockIdx.x * blockDim.x,_ind;
    int g;
    while (_tid< (*GPUTotNumFSRs)) {
        for (g=0;g<(*GPUNumMOCGroups);g++) {
            _ind=_tid*(*GPUNumMOCGroups)+g;
            _GPUScalarFluxOld[_ind]=_GPUScalarFlux[_ind];
        }
        /* Increment thread id */
        _tid += blockDim.x * gridDim.x;
    }
}

__global__ void GPUVectorInit_1DFP_float(float *_vector, int _dim,float _value)
{
    int _tid = threadIdx.x + blockIdx.x * blockDim.x;
    // int _tid = threadIdx.x + blockIdx.x * GPUNumThreadPerBlock;

    while (_tid < _dim) {
        _vector[_tid]=_value;
        /* Increment the thread id */
        _tid += blockDim.x * gridDim.x;
        // _tid += GPUNumThreadPerBlock * GPUNumThreadBlocks;

    }

}


/**
 * @ Function: GPUComputeReacRatesOnDevice()
 *   introduction: calculate the reaction rate of the whole domain
 *                 this function is called by host and excuted on device
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2018/05/30, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified: 2019/12/25, change the FSR area to volume
 */
__global__ void GPUComputeReacRatesOnDevice(FP_PRECISION_FLUX *_GPUScalarFlux,FP_PRECISION *_GPUMacroXS,\
    FP_PRECISION_GEO *_GPUFSRVol_SubDom,FP_PRECISION *fission)
{
    int _tid = threadIdx.x + blockIdx.x * blockDim.x,g;
    FP_PRECISION curr_fiss = 0.;
    FP_PRECISION fiss = 0.;
    /* Iterate over all FSRs */
    while (_tid<(*GPUTotNumFSRs)) {
        curr_fiss = 0.;
        for (g=0;g<(*GPUNumMOCGroups);g++) {
            curr_fiss+=_GPUScalarFlux[_tid*(*GPUNumMOCGroups)+g]*_GPUMacroXS[_tid*(*GPUNumMOCGroups)+g];
        }
        fiss+=curr_fiss*(FP_PRECISION)(_GPUFSRVol_SubDom[_tid]);
        /* Increment thread id */
        _tid += blockDim.x * gridDim.x;
    }
    /* Copy this thread's fission to global memory */
    _tid = threadIdx.x + blockIdx.x * blockDim.x;
    fission[_tid] = fiss;
}

/**
 * @ Function: GPUComputeFSRInfNormOnDevice()
 *   introduction: calculate the residual of scalar flux
 *                 this function is called by host and excuted on device
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2018/05/30, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
__global__ void GPUComputeFSRInfNormOnDevice(FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUScalarFluxOld,\
    FP_PRECISION *_GPUFluxResidual)
{
    int _tid = threadIdx.x + blockIdx.x * blockDim.x,g,_ind;
    /* Iterate over all FSRs */
    while (_tid < (*GPUTotNumFSRs)) {
        for (g=0;g<(*GPUNumMOCGroups);g++) {
            _ind=_tid*(*GPUNumMOCGroups)+g;
            _GPUFluxResidual[_ind]=fabs((_GPUScalarFlux[_ind]-_GPUScalarFluxOld[_ind])/_GPUScalarFluxOld[_ind]);
        }
        /* Increment the thread id */
        _tid += blockDim.x * gridDim.x;
    }
}

/**
 * @ Function: GPUInitMacroXSs()
 *   introduction: copy the XSs variable from host to device
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2018/05/30, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
extern "C"
void GPUInitMacroXSs(int _procID, int _TotalNumFSRs_subDom, int _NumGroups, int _MOCNumLayersZ, \
	M_UINT64 *_MacroXS_s_NZRowStartID_Layer, M_UINT64 *_MacroXS_s_NZColStartID, size_t *_GPUPreMemXSs,\
	M_INT32 _IfResonance, M_INT32 _ProblemType, M_UINT64 _MacroXS_s_MaxNNZ_Layer, FILE *_LogFile)
{
	if (_procID==0) {
		printf("\n [GPU-INFO ]  Initializing MacroXSs on GPU ...");
		fflush(stdout);
		fprintf(_LogFile, "\n [GPU-INFO ]  Initializing MacroXSs on GPU ...");
    }
	// copy macroscopic cross sections from host to device
    GPUCopyMacroXSsHostToDevice_Init(_TotalNumFSRs_subDom, _NumGroups, _MOCNumLayersZ, \
		_MacroXS_s_NZRowStartID_Layer, _MacroXS_s_NZColStartID, _GPUPreMemXSs,\
		_IfResonance, _ProblemType, _MacroXS_s_MaxNNZ_Layer);

	CUDA_CheckError((char *)("GPUInitMacroXSs"));
}

/**
 * @ Function: GPUInitQuadratureSet()
 *   introduction: copy the quatrature and exponential table variables from host to device
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2018/06/05, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
extern "C"
void GPUInitQuadratureSet(int _procID, size_t *_GPUPreMemQuadrSet, size_t *_GPUPreMemExpTab, FILE *_LogFile, \
	M_INT32 _NumAzimAng, M_INT32 _NumPolarAng, FP_PRECISION *_SinTheta, FP_PRECISION *_CosTheta, \
	FP_PRECISION *_GC_SinThetasTimes2, FP_PRECISION *_GC_OneOverSinThetas, FP_PRECISION *_SinThetaForExp, \
	FP_PRECISION *_AzimWeights, FP_PRECISION *_PolarWeights, FP_PRECISION *_TotalWeights, FP_PRECISION *_TotalWeights_2DMOC, \
	FP_PRECISION *_ExpTable, int _NumGroups, M_INT32 _NumScattOrder, FP_PRECISION *_CoeffA_HOScatt, FP_PRECISION *_SpheHarmCoeff)
{
	size_t _dim=0;
// #if defined(M_3DMOCex)
	int _angMul1, _angMul2;
// #endif 
	if (_procID==0) {
		printf("\n [GPU-INFO ]  Initializing Quadrature Set on GPU ...");
		fflush(stdout);
		fprintf(_LogFile, "\n [GPU-INFO ]  Initializing Quadrature Set on GPU ...");
    }
    /* Copy the number of azimuthal angles to constant memory on the GPU */
	cudaMemcpyToSymbol(GPUNumAzimAng, (void*)&_NumAzimAng, sizeof(int), 0, cudaMemcpyHostToDevice);
	CUDA_CheckError((char *)("GPUInitQuadratureSet-cudaMemcpyToSymbol-GPUNumAzimAng"));
	/* Copy the number of polar angles to constant memory on the GPU */
	cudaMemcpyToSymbol(GPUNumPolarAng, (void*)&_NumPolarAng, sizeof(int), 0, cudaMemcpyHostToDevice);
	CUDA_CheckError((char *)("GPUInitQuadratureSet-cudaMemcpyToSymbol-GPUNumPolarAng"));
	cudaMemcpyToSymbol(GPUNumScattOrder, (void*)&_NumScattOrder, sizeof(int), 0, cudaMemcpyHostToDevice);
    CUDA_CheckError((char *)("GPUInitQuadratureSet-cudaMemcpyToSymbol-GPUNumScattOrder1"));
	cudaMemcpyToSymbol(GPUCoeffA_HOScatt, (void*)_CoeffA_HOScatt, (size_t)((_NumScattOrder+1)*(_NumScattOrder+1)) * sizeof(FP_PRECISION), 0, cudaMemcpyHostToDevice);
	CUDA_CheckError((char *)("GPUInitQuadratureSet-cudaMemcpyToSymbol-GPUNumScattOrder"));
	_dim = (size_t)(_NumAzimAng * _NumPolarAng * (_NumScattOrder + 1) * (_NumScattOrder + 1)) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUSpheHarmCoeff, _dim);
	CUDA_CheckError((char *)("GPUInitQuadratureSet-cudaMalloc-GPUSpheHarmCoeff"));

	_GPUPreMemQuadrSet[0] += _dim;
	cudaMemcpy((void*)GPUSpheHarmCoeff, (void*)_SpheHarmCoeff, _dim, cudaMemcpyHostToDevice);
	CUDA_CheckError((char *)("GPUInitQuadratureSet-cudaMemcpy-GPUSpheHarmCoeff"));
    /* Copy the sines of the polar angles which is needed if the user
	 * requested the use of the exp intrinsic to evaluate exponentials */
	
	cudaMemcpyToSymbol(GPUSinThetas, (void*)_SinTheta, (size_t)(_NumPolarAng) * sizeof(FP_PRECISION), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUCosThetas, (void*)_CosTheta, (size_t)(_NumPolarAng) * sizeof(FP_PRECISION), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUSinThetasTimes2, (void*)_GC_SinThetasTimes2, (size_t)(_NumPolarAng) * sizeof(FP_PRECISION), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUOneOverSinThetas, (void*)_GC_OneOverSinThetas, (size_t)(_NumPolarAng) * sizeof(FP_PRECISION), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUSinThetaForExp, (void*)_SinThetaForExp, (size_t)(_NumPolarAng) * sizeof(FP_PRECISION), 0, cudaMemcpyHostToDevice);
	CUDA_CheckError((char *)("GPUInitQuadratureSet-cudaMemcpyToSymbol-GPUSinThetaForExp"));
    /* Copy the weights to constant memory on the GPU */
    cudaMemcpyToSymbol(GPUAzimWeights, (void*)_AzimWeights, (size_t)(_NumAzimAng) * sizeof(FP_PRECISION), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(GPUPolarWeights, (void*)_PolarWeights, (size_t)(_NumPolarAng) * sizeof(FP_PRECISION), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(GPUTotalWeights, (void*)_TotalWeights, (size_t)(_NumAzimAng* _NumPolarAng) * sizeof(FP_PRECISION), 0, cudaMemcpyHostToDevice);
	CUDA_CheckError((char *)("GPUInitQuadratureSet-cudaMemcpyToSymbol-GPUTotalWeights"));

	// printf("\n _NumAzimAng=%d, _NumPolarAng=%d",_NumAzimAng, _NumPolarAng);
	// PrintFPArrayOnDevice(_NumAzimAng* _NumPolarAng, GPUTotalWeights, (char *)("GPUTotalWeights"));
	
	// int i;
	// for (i=0;i<_NumAzimAng* _NumPolarAng;i++) {
	// 	printf("\n _TotalWeights[%d]=%e",i,_TotalWeights[i]);
	// }
	_dim = (size_t)(_NumAzimAng * _NumPolarAng) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUTotalWeights_2DMOC, _dim);
	CUDA_CheckError((char *)("GPUInitQuadratureSet-cudaMalloc-GPUTotalWeights_2DMOC"));

	_GPUPreMemQuadrSet[0] += _dim;
	cudaMemcpy((void*)GPUTotalWeights_2DMOC, (void*)_TotalWeights_2DMOC, _dim, cudaMemcpyHostToDevice);
	CUDA_CheckError((char *)("GPUInitQuadratureSet-cudaMemcpy-GPUTotalWeights_2DMOC"));
	// PrintFPArrayOnDevice(_NumAzimAng* _NumPolarAng, GPUTotalWeights_2DMOC, (char *)("GPUTotalWeights_2DMOC"));

if (_procID==0) {
	printf("\n [GPU-INFO ]  Initializing ExpInte Table on GPU ...");
	fflush(stdout);
	fprintf(_LogFile, "\n [GPU-INFO ]  Initializing ExpInte Table on GPU ...");
}
	cudaMalloc((void**)&GPULinearExpTable, (size_t)(NumExpPoints) * sizeof(FP_PRECISION));
	_GPUPreMemExpTab[0] += (size_t)(NumExpPoints) * sizeof(FP_PRECISION);

    cudaMemcpy((void*)GPULinearExpTable, (void*)_ExpTable, (size_t)(NumExpPoints) *sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
	CUDA_CheckError((char *)("GPUInitQuadratureSet"));
// #if defined(M_3DMOCex)
	_angMul2 = _NumAzimAng * _NumPolarAng/2;
	_angMul1 = _NumGroups * _angMul2;
    cudaMemcpyToSymbol(GPU_angMul1_HP, (void*)&_angMul1, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPU_angMul2_HP, (void*)&_angMul2, sizeof(int), 0, cudaMemcpyHostToDevice);
	
	_angMul2 = _NumAzimAng * _NumPolarAng;
	_angMul1 = _NumGroups * _angMul2;
    cudaMemcpyToSymbol(GPU_angMul1_FP, (void*)&_angMul1, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(GPU_angMul2_FP, (void*)&_angMul2, sizeof(int), 0, cudaMemcpyHostToDevice);
// #endif
}

/**
 * @ Function: GPUCopyMacroXSsHostToDevice_Layer()
 *   introduction: Allocate memory on device for MacroXS data arrays and copy XSs data from host to arrays on the device
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2018/06/05, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
void GPUCopyMacroXSsHostToDevice_Layer(int _layID, int _TotalNumFSRs_subDom, int _TotalNumFSRs_perLayer, int _NumGroups, M_UINT64 *_MacroXS_s_NZStartID_Layer, \
	M_UINT64 *_MacroXS_s_NNZ_Layer, M_INT32 _ProblemType, FP_PRECISION *_MacroXS_t, FP_PRECISION *_MacroXS_a, FP_PRECISION *_MacroXS_f, \
	FP_PRECISION *_MacroXS_nf, FP_PRECISION *_Chi, FP_PRECISION *_MacroXS_s, FP_PRECISION *_FixedSour, M_INT32 _NumScattOrder,\
	M_UINT64 *_MacroXS_s_NZRowStartID_Layer, M_UINT64 *_MacroXS_s_NZColStartID, int _isCMFD, FP_PRECISION_FLUX *_ScalarFluxOld)
{
	int _numFRSsTimesNumGroup, _index1D, _index_rowID1, _index_rowID2, _dim;
	M_UINT64 _index2D;
	int l, m,_count,_ind,_ind1,_ind2, _ind3;

	_numFRSsTimesNumGroup=_TotalNumFSRs_perLayer * _NumGroups;
	_index1D = _layID * _TotalNumFSRs_perLayer * _NumGroups;
	
	if (_ProblemType == 0) {
		_index2D = _MacroXS_s_NZStartID_Layer[_layID];
		_index_rowID1 = _layID * _TotalNumFSRs_perLayer * (_NumScattOrder + 1) * _NumGroups + _layID;
		_index_rowID2 = _layID * _TotalNumFSRs_perLayer * (_NumScattOrder + 1) * _NumGroups;
	}
    _dim = _TotalNumFSRs_perLayer * _NumGroups;

    /* Copy Material data from host to arrays on the device */
	cudaMemcpy((void*)GPUMacroXS_t, (void*)&_MacroXS_t[_index1D],(size_t)(_numFRSsTimesNumGroup) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
	CUDA_CheckError((char *)("cudaMemcpy-GPUMacroXS_t"));

	_count=0;
	_ind1 = _TotalNumFSRs_subDom * _NumGroups;
	_ind2 = _layID * _TotalNumFSRs_perLayer * _NumGroups;
    for (l=0;l<=_NumScattOrder;l++) {
		for (m=-l;m<=l;m++) {
			_ind = _count * _ind1 + _ind2;
			_ind3 = _count * _dim;
			cudaMemcpy((void*)&GPUScalarFlux[_ind3], (void*)&_ScalarFluxOld[_ind],(size_t)(_dim) * sizeof(FP_PRECISION_FLUX), cudaMemcpyHostToDevice);
			CUDA_CheckError((char *)("cudaMemcpy-GPUScalarFlux"));
			_count++;
		}
	}

	if (_ProblemType ==0) {
		if (GPUMacroXS_a != NULL && _isCMFD==1) {
			// printf("\n GPUMacroXS_a != NULL");
			// printf("\n _MacroXS_a[_index1D=%d]=%e",_index1D,_MacroXS_a[_index1D]);
			// printf("\n _MacroXS_a[_index1D=%d]=%e",23,_MacroXS_a[23]);
			// printf("\n _numFRSsTimesNumGroup=%d",_numFRSsTimesNumGroup);
			// exit(90);
			cudaMemcpy((void*)GPUMacroXS_a, (void*)&_MacroXS_a[_index1D],(size_t)(_numFRSsTimesNumGroup) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
			CUDA_CheckError((char *)("cudaMemcpy-GPUMacroXS_a"));
		}
		
		if (GPUMacroXS_f != NULL && _isCMFD==1) {
			cudaMemcpy((void*)GPUMacroXS_f, (void*)&_MacroXS_f[_index1D],(size_t)(_numFRSsTimesNumGroup) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
	        CUDA_CheckError((char *)("cudaMemcpy-GPUMacroXS_f"));
		}

		cudaMemcpy((void*)GPUMacroXS_nf, (void*)&_MacroXS_nf[_index1D],(size_t)(_numFRSsTimesNumGroup) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
	    CUDA_CheckError((char *)("cudaMemcpy-GPUMacroXS_nf"));
    #ifdef FIS_MTX
        cudaMemcpy((void*)GPUMacroXS_chi, (void*)&_Chi[_index1D],(size_t)(_numFRSsTimesNumGroup * _NumGroups) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
    #else
	    cudaMemcpy((void*)GPUMacroXS_chi, (void*)&_Chi[_index1D],(size_t)(_numFRSsTimesNumGroup) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
	#endif
        CUDA_CheckError((char *)("cudaMemcpy-GPUMacroXS_chi"));
	    cudaMemcpy((void*)GPUMacroXS_s, (void*)&_MacroXS_s[_index2D], (size_t)(_MacroXS_s_NNZ_Layer[_layID]) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
		CUDA_CheckError((char *)("cudaMemcpy-GPUMacroXS_s"));
		
		cudaMemcpy((void*)GPUMacroXS_s_NZRowStartID_Layer, (void*)&_MacroXS_s_NZRowStartID_Layer[_index_rowID1],(size_t)(_TotalNumFSRs_perLayer*(_NumScattOrder + 1) *_NumGroups+1)*sizeof(M_UINT64), cudaMemcpyHostToDevice);
		CUDA_CheckError((char *)("cudaMemcpy-GPUMacroXS_s_NZRowStartID_Layer"));
        M_UINT64 *rowEndHost = (IfUseExplicitScaRowEnd && MacroXS_s_NZRowEndID_Layer != NULL)
            ? &MacroXS_s_NZRowEndID_Layer[_index_rowID1]
            : &_MacroXS_s_NZRowStartID_Layer[_index_rowID1 + 1];
        cudaMemcpy((void*)GPUMacroXS_s_NZRowEndID_Layer, (void*)rowEndHost,(size_t)(_TotalNumFSRs_perLayer*(_NumScattOrder + 1)*_NumGroups)*sizeof(M_UINT64), cudaMemcpyHostToDevice);
        CUDA_CheckError((char *)("cudaMemcpy-GPUMacroXS_s_NZRowEndID_Layer"));

		cudaMemcpy((void*)GPUMacroXS_s_NZColStartID, (void*)&_MacroXS_s_NZColStartID[_index_rowID2],(size_t)(_TotalNumFSRs_perLayer*(_NumScattOrder + 1) *_NumGroups)*sizeof(M_UINT64), cudaMemcpyHostToDevice);
		CUDA_CheckError((char *)("cudaMemcpy-GPUMacroXS_s_NZColStartID"));
		// printf("\n _layID=%d",_layID);
		// printf("\n _NumScattOrder=%d",_NumScattOrder);
		// printf("\n _index_rowID1=%d",_index_rowID1);
		// printf("\n _index_rowID2=%d",_index_rowID2);

		// GPUPrintf1DArray_UI64(_TotalNumFSRs_perLayer*(_NumScattOrder + 1) *_NumGroups+1, GPUMacroXS_s_NZRowStartID_Layer, "GPUMacroXS_s_NZRowStartID_Layer",NULL);
		// GPUPrintf1DArray_I32(_TotalNumFSRs_perLayer*(_NumScattOrder + 1) *_NumGroups, GPUMacroXS_s_NZColStartID, "GPUMacroXS_s_NZColStartID",NULL);
		// GPUPrintf1DArray_FP(_MacroXS_s_NNZ_Layer[_layID], GPUMacroXS_s, "b4 ts GPUMacroXS_s",NULL);
		// // exit(90);	
	}
	else if (_ProblemType==1) {
		cudaMemcpy((void*)GPUMOCSource, (void*)&_FixedSour[_index1D],(size_t)(_numFRSsTimesNumGroup) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
	}
	else {}

	//printf("\n _lay = %d, _index1D=%d, _numFRSsTimesNumGroup=%d, _index2D=%d",_layID,_index1D,_numFRSsTimesNumGroup,_index2D);
	//int ii=0;
	//for (ii=0;ii<_TotalNumFSRs_perLayer*_NumGroups*2;ii++) {
	//	printf("\n MacroXS_t[%d]=%e",ii,MacroXS_t[ii]);
	//}

//    GPUPrintf1DArray_FP(_numFRSsTimesNumGroup, GPUMacroXS_t, "b4 ts GPUMacroXS_t",NULL);
	// GPUPrintf1DArray_FP(_numFRSsTimesNumGroup, GPUMOCSource, "b4 ts GPUMOCSource",NULL);
	// printf("\n _lay = %d",_layID);
	// GPUPrintf1DArray_FP(_numFRSsTimesNumGroup, GPUMacroXS_nf, "b4 ts GPUMacroXS_nf",LogFile);
	// printf("\n _lay = %d",_layID);
	// GPUPrintf1DArray_FP(_numFRSsTimesNumGroup, GPUMacroXS_chi, "b4 ts GPUMacroXS_chi",LogFile);
	// printf("\n _lay = %d",_layID);
	// GPUPrintf1DArray_FP(_MacroXS_s_NNZ_Layer[_layID], GPUMacroXS_s, "b4 ts GPUMacroXS_s",LogFile);
}

/**
 * @ Function: GPUCopyTrackInfoHostToDevice()
 *   introduction: Allocate memory on device for track data arrays and copy track data from host to arrays on the device
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2018/06/05, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
extern "C"
void GPUCopyTrackInfoHostToDevice(int _procID, M_INT32 _NumGroups, M_INT32 _TotalNumFSRs_subDom, M_INT32 _TotalNumFSRs_perLayer,\
	M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumPinsInSubDomPerLayer, M_INT32 _TotalNumFSRsInRTM, M_INT32 _NumPolarAng, \
	M_INT32 _NumLTs_HS, M_INT32 _NumLTs_2PI, M_INT32 *_LTsIndex_AzimAngID_HS, M_INT32 *_LTsIndex_LTID_HS, \
	M_INT32 *_LTsIndex_AzimAngID_2PI, M_INT32 *_LTsIndex_LTID_2PI, struct SUBDOM_TRACK_IN_LONGTRACK *_TracksInfoInLTs_2PI,\
	M_INT32 *_TrackIndexInLTs_2PI, M_INT32 _TotNumTracksInOneRTM_2PI, M_INT32 *_IndexTrackInRTM_SumAzim_2PI,\
	M_INT32 _TotalNumTracksInRTM_Total, M_INT32 *_NumSegsInTrack_2PI, M_INT32 _TotNumSegInRTM_Total,\
	struct SUBDOM_SEG_IN_TRACK_SINGLE *_SegInfoInTrack_SINGLE_2PI, M_INT32 *_SumNumSegsInRTMTrack_2PI,\
	FP_PRECISION *_SegInfoInTrack_2PI_length, M_INT32 *_SegInfoInTrack_2PI_RayWisedFSR, M_INT32 *_SegInfoInTrack_2PI_AntiClockFSR,\
	M_INT32 *_FsrIdMap_RayWisedToAntiClock, struct SUBDOM_TRACK_IN_LONGTRACK *_TracksInfoInLTs_HS, M_INT32 _NumTracks_HS,\
	struct SUBDOM_LONGTRACK_INFO *_LongTracksInfo_Total, M_INT32 *_TrackIndexInLTs_HS, M_INT32 _TotalNumTracksInRTM_HS,\
	M_INT32 *_NumSegsInTrack_HS, M_INT32 _TotNumTracksInOneRTM_HS, M_INT32 _NumAzimAng, M_INT32 *_IndexTrackInRTM_SumAzim_HS,\
	M_INT32 _TotNumSegInRTM_HS, struct SUBDOM_SEG_IN_TRACK_SINGLE *_SegInfoInTrack_SINGLE_HS, M_INT32 _MOCNumLayersZ,\
	M_INT32 *_SumNumFSRsBasedGlobCellID, M_INT32 _NumRTMGeoType, M_INT32 *_SumNumFSRsInRTM, M_INT32 *_CellRTMType_SubDom,\
	M_INT32 *_SumNumSegsInRTMTrack_HS, M_INT32 *_NumLTsSum, M_INT32 *_NumLongTracks,\
	FP_PRECISION_GEO *_OneOverFSRArea_SubDom, FP_PRECISION_GEO *_FSRVol_SubDom, M_INT32 _NorthBC, M_INT32 _SouthBC, M_INT32 _EastBC, \
	M_INT32 _WestBC, M_INT32 _BottomBC, M_INT32 _TopBC, M_INT32 *_NumFSRsInCell, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR,\
	FP_PRECISION_GEO *_EffRaySpacing, M_INT32 *_NumFSRs_Layer, M_INT32 *_SumNumFSRs_Layer, FP_PRECISION_GEO *_WidthLayersInCore, \
	M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, FP_PRECISION_GEO *_RayTracModDim, M_INT32 _NumSurfaceX, M_INT32 _NumSurfaceY,\
	M_INT32 *_NumTracksX, M_INT32 *_NumTracksY, FP_PRECISION_GEO *_RTMWidth_X, FP_PRECISION_GEO *_RTMWidth_Y, M_INT32 *_NumFSRsInRTM, \
	FP_PRECISION_GEO *_FSRsAreaInRTM_AntiClockFSR, FP_PRECISION_GEO *_FSRArea_SubDom, size_t *_GPUPreMemTrack, size_t *_GPUPreMemXSs, \
	M_INT32 _NumTracks_Total, FILE *_LogFile)
{
	int _totNumLTs,_polarTimesGroup,_dimSurfX,_dimSurfY,_dimSurfZ;
	size_t _dim=0;

    if (_procID==0) {
		printf("\n [GPU-INFO ]  Initializing Track Information on GPU ...");
		fflush(stdout);
		fprintf(_LogFile, "\n [GPU-INFO ]  Initializing Track Information on GPU ...");
    }
	_totNumLTs= _NumLTs_HS*2;
	
	cudaMemcpyToSymbol(GPUNumMOCGroups, (void*)&_NumGroups, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUTotNumFSRs, (void*)&_TotalNumFSRs_subDom, sizeof(M_INT32), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUNumFSRsLayer, (void*)&_TotalNumFSRs_perLayer, sizeof(M_INT32), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUTotNumPins, (void*)&_MOC_NumPinsInSubDom, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUNumPinsInSubDomPerLayer, (void*)&_NumPinsInSubDomPerLayer, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUTotalNumFSRsInRTM, (void*)&_TotalNumFSRsInRTM, sizeof(int), 0, cudaMemcpyHostToDevice);
    //
    _polarTimesGroup= _NumPolarAng * _NumGroups;
    cudaMemcpyToSymbol(GPUPolarTimesGroups, (void*)&_polarTimesGroup, sizeof(int), 0, cudaMemcpyHostToDevice);
    /* Copy the number of long tracks in halfspace[0,PI] on the GPU */
    cudaMemcpyToSymbol(GPUNumLTs_HS, (void*)&_NumLTs_HS, sizeof(int), 0, cudaMemcpyHostToDevice);
#ifdef GeoTwoPIRegion
	cudaMemcpyToSymbol(GPUNumLTs_Total, (void*)&_NumLTs_2PI, sizeof(int), 0, cudaMemcpyHostToDevice);
#endif // GeoTwoPIRegion

    _dim = (size_t)(_NumLTs_HS) * sizeof(int);
    cudaMalloc((void**)&GPULTsIndex_AzimAngID_HS, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPULTsIndex_AzimAngID_HS, (void*)_LTsIndex_AzimAngID_HS,_dim, cudaMemcpyHostToDevice);
	
	_dim = (size_t)(_NumLTs_HS) * sizeof(int);
	cudaMalloc((void**)&GPULTsIndex_LTID_HS, _dim);
	_GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPULTsIndex_LTID_HS, (void*)_LTsIndex_LTID_HS,_dim, cudaMemcpyHostToDevice);
	//
#ifdef GeoTwoPIRegion
    _dim = (size_t)(_NumLTs_2PI) * sizeof(int);
	cudaMalloc((void**)&GPULTsIndex_AzimAngID_Total, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPULTsIndex_AzimAngID_Total, (void*)_LTsIndex_AzimAngID_2PI, _dim, cudaMemcpyHostToDevice);

	_dim = (size_t)(_NumLTs_2PI) * sizeof(int);
	cudaMalloc((void**)&GPULTsIndex_LTID_Total, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPULTsIndex_LTID_Total, (void*)_LTsIndex_LTID_2PI, _dim, cudaMemcpyHostToDevice);

	_dim = (size_t)(_NumTracks_Total) * sizeof(struct SUBDOM_TRACK_IN_LONGTRACK);
	cudaMalloc((void**)&GPUTracksInfoInLTs_Total, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUTracksInfoInLTs_Total, (void*)_TracksInfoInLTs_2PI, _dim, cudaMemcpyHostToDevice);
	
	_dim = (size_t)(_NumLTs_2PI + 1) * sizeof(int);
	cudaMalloc((void**)&GPUTrackIndexInLTs_Total, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUTrackIndexInLTs_Total, (void*)_TrackIndexInLTs_2PI, _dim, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(GPUTotNumTracksInOneRTM_2PI, (void*)&_TotNumTracksInOneRTM_2PI, sizeof(int), 0, cudaMemcpyHostToDevice);
	
	_dim = (size_t)(_NumAzimAng) * sizeof(int);
	cudaMalloc((void**)&GPUIndexTrackInRTM_SumAzim_Total, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUIndexTrackInRTM_SumAzim_Total, (void*)_IndexTrackInRTM_SumAzim_2PI, _dim, cudaMemcpyHostToDevice);
	
	_dim = (size_t)(_TotalNumTracksInRTM_Total) * sizeof(int);
	cudaMalloc((void**)&GPUNumSegsInTrack_Total, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUNumSegsInTrack_Total, (void*)_NumSegsInTrack_2PI, _dim, cudaMemcpyHostToDevice);
	
	_dim = (size_t)(_TotNumSegInRTM_Total) * sizeof(struct SUBDOM_SEG_IN_TRACK_SINGLE);
	cudaMalloc((void**)&GPUSegInfoInTrack_Total, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUSegInfoInTrack_Total, (void*)_SegInfoInTrack_SINGLE_2PI, _dim, cudaMemcpyHostToDevice);
	
	_dim = (size_t)(_TotalNumTracksInRTM_Total) * sizeof(int);
	cudaMalloc((void**)&GPUSumNumSegsInRTMTrack_2PI, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUSumNumSegsInRTMTrack_2PI, (void*)_SumNumSegsInRTMTrack_2PI, _dim, cudaMemcpyHostToDevice);
	CUDA_CheckError((char *)("GPUSumNumSegsInRTMTrack_2PI"));
/*modified 2019-06-15 by peitao song*/
    _dim = (size_t)(_TotNumSegInRTM_Total) * sizeof(FP_PRECISION);
    cudaMalloc((void**)&GPUSegInfoInTrack_2PI_length, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUSegInfoInTrack_2PI_length, (void*)_SegInfoInTrack_2PI_length, _dim, cudaMemcpyHostToDevice);
	
	_dim = (size_t)(_TotNumSegInRTM_Total) * sizeof(int);
	cudaMalloc((void**)&GPUSegInfoInTrack_2PI_RayWisedFSR, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUSegInfoInTrack_2PI_RayWisedFSR, (void*)_SegInfoInTrack_2PI_RayWisedFSR, _dim, cudaMemcpyHostToDevice);
	
	_dim = (size_t)(_TotNumSegInRTM_Total) * sizeof(int);
	cudaMalloc((void**)&GPUSegInfoInTrack_2PI_AntiClockFSR, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUSegInfoInTrack_2PI_AntiClockFSR, (void*)_SegInfoInTrack_2PI_AntiClockFSR, _dim, cudaMemcpyHostToDevice);
	
	_dim = (size_t)(_TotalNumFSRsInRTM * _NumAzimAng) * sizeof(int);
	cudaMalloc((void**)&GPUFsrIdMap_RayWisedToAntiClock, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUFsrIdMap_RayWisedToAntiClock, (void*)_FsrIdMap_RayWisedToAntiClock, _dim, cudaMemcpyHostToDevice);
    /*modified 2019-06-15 by peitao song - end*/
    CUDA_CheckError((char *)("GPUFsrIdMap_RayWisedToAntiClock"));

#endif // GeoTwoPIRegion

	// copy tracks info in LTs from host to device
	_dim = (size_t)(_NumTracks_HS) * sizeof(struct SUBDOM_TRACK_IN_LONGTRACK);
    cudaMalloc((void**)&GPUTracksInfoInLTs_HS, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUTracksInfoInLTs_HS, (void*)_TracksInfoInLTs_HS,_dim, cudaMemcpyHostToDevice);

    /* Allocate memory on device for GPULongTracksInfo_Total data arrays */
	_dim = (size_t)(_totNumLTs) * sizeof(struct SUBDOM_LONGTRACK_INFO);
	cudaMalloc((void**)&GPULongTracksInfo_Total, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPULongTracksInfo_Total, (void*)_LongTracksInfo_Total, _dim, cudaMemcpyHostToDevice);
	//
	_dim = (size_t)((_NumLTs_HS+1)) * sizeof(int);
    cudaMalloc((void**)&GPUTrackIndexInLTs_HS, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUTrackIndexInLTs_HS, (void*)_TrackIndexInLTs_HS,_dim, cudaMemcpyHostToDevice);

    // cudaMemcpyToSymbol(&GPUTotalNumTracksInRTM, (void*)TotalNumTracksInRTM_HS, sizeof(int), 0, cudaMemcpyHostToDevice);
	_dim = (size_t)(_TotalNumTracksInRTM_HS) * sizeof(int);
	cudaMalloc((void**)&GPUNumSegsInTrack_HS, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUNumSegsInTrack_HS, (void*)_NumSegsInTrack_HS,_dim, cudaMemcpyHostToDevice);
    //
    cudaMemcpyToSymbol(GPUTotNumTracksInOneRTM_HS, (void*)&_TotNumTracksInOneRTM_HS, sizeof(int), 0, cudaMemcpyHostToDevice);
	
	_dim = (size_t)(_NumAzimAng/2) * sizeof(int);
	cudaMalloc((void**)&GPUIndexTrackInRTM_SumAzim_HS, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUIndexTrackInRTM_SumAzim_HS, (void*)_IndexTrackInRTM_SumAzim_HS, _dim, cudaMemcpyHostToDevice);
	CUDA_CheckError((char *)("GPUIndexTrackInRTM_SumAzim_HS"));
	//
	_dim = (size_t)(_TotNumSegInRTM_HS) * sizeof(struct SUBDOM_SEG_IN_TRACK_SINGLE);
    cudaMalloc((void**)&GPUSegInfoInTrack_HS, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUSegInfoInTrack_HS, (void*)_SegInfoInTrack_SINGLE_HS,_dim, cudaMemcpyHostToDevice);
	//
	_dim = (size_t)(_MOCNumLayersZ*_NumPinsInSubDomPerLayer) * sizeof(int);
    cudaMalloc((void**)&GPUSumNumFSRsBasedGlobCellID, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUSumNumFSRsBasedGlobCellID, (void*)_SumNumFSRsBasedGlobCellID,_dim, cudaMemcpyHostToDevice);
    //
	_dim = (size_t)(_NumRTMGeoType) * sizeof(int);
	cudaMalloc((void**)&GPUSumNumFSRsInRTM, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUSumNumFSRsInRTM, (void*)_SumNumFSRsInRTM, _dim, cudaMemcpyHostToDevice);
    //
    _dim = (size_t)(_MOCNumLayersZ*_NumPinsInSubDomPerLayer) * sizeof(int);
    cudaMalloc((void**)&GPUCellRTMType_SubDom, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUCellRTMType_SubDom, (void*)_CellRTMType_SubDom, _dim, cudaMemcpyHostToDevice);
	//
	_dim = (size_t)(_TotalNumTracksInRTM_HS) * sizeof(int);
    cudaMalloc((void**)&GPUSumNumSegsInRTMTrack_HS, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUSumNumSegsInRTMTrack_HS, (void*)_SumNumSegsInRTMTrack_HS,_dim, cudaMemcpyHostToDevice);
	//
	_dim = (size_t)(_NumAzimAng) * sizeof(int);
    cudaMalloc((void**)&GPUNumLTsSum, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUNumLTsSum, (void*)_NumLTsSum, _dim, cudaMemcpyHostToDevice);
	//
	_dim = (size_t)(_NumAzimAng) * sizeof(int);
    cudaMalloc((void**)&GPUNumLongTracks, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUNumLongTracks, (void*)_NumLongTracks, _dim, cudaMemcpyHostToDevice);
    //
    _dim = (size_t)(_TotalNumFSRs_subDom) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUOneOverFSRsAreaInGlob, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUOneOverFSRsAreaInGlob, (void*)_OneOverFSRArea_SubDom, _dim, cudaMemcpyHostToDevice);
	
	_dim = (size_t)(_TotalNumFSRs_subDom) * sizeof(FP_PRECISION_GEO);
	cudaMalloc((void**)&GPUFSRVol_SubDom, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUFSRVol_SubDom, (void*)_FSRVol_SubDom, _dim, cudaMemcpyHostToDevice);

	CUDA_CheckError((char *)("GPUOneOverFSRsAreaInGlob"));
	// 
    cudaMemcpyToSymbol(GPUNorthBC, (void*)&_NorthBC, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(GPUSouthBC, (void*)&_SouthBC, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(GPUEastBC, (void*)&_EastBC, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUWestBC, (void*)&_WestBC, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUBottomBC, (void*)&_BottomBC, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUTopBC, (void*)&_TopBC, sizeof(int), 0, cudaMemcpyHostToDevice);

	_dim = (size_t)(_MOC_NumPinsInSubDom) * sizeof(int);
	cudaMalloc((void**)&GPUNumFSRsInCell, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUNumFSRsInCell, (void*)_NumFSRsInCell, _dim, cudaMemcpyHostToDevice);

	_dim = (size_t)(_TotalNumFSRsInRTM) * sizeof(FP_PRECISION_GEO);
	cudaMalloc((void**)&GPUOneOverFSRsAreaInRTM_AntiClockFSR, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUOneOverFSRsAreaInRTM_AntiClockFSR, (void*)_OneOverFSRsAreaInRTM_AntiClockFSR, _dim, cudaMemcpyHostToDevice);
	// 
	cudaMemcpyToSymbol(GPUEffRaySpacing, (void*)_EffRaySpacing, (size_t)(_NumAzimAng) * sizeof(FP_PRECISION_GEO), 0, cudaMemcpyHostToDevice);

	_dim = (size_t)(_MOCNumLayersZ) * sizeof(int);
	cudaMalloc((void**)&GPUNumFSRs_Layer, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUNumFSRs_Layer, (void*)_NumFSRs_Layer, _dim, cudaMemcpyHostToDevice);
	
	_dim = (size_t)(_MOCNumLayersZ) * sizeof(int);
	cudaMalloc((void**)&GPUSumNumFSRs_Layer, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUSumNumFSRs_Layer, (void*)_SumNumFSRs_Layer, _dim, cudaMemcpyHostToDevice);

	_dim = (size_t)(_MOCNumLayersZ) * sizeof(FP_PRECISION_GEO);
	cudaMalloc((void**)&GPUWidthLayersInCore, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUWidthLayersInCore, (void*)_WidthLayersInCore, _dim, cudaMemcpyHostToDevice);

	/* copy geometry related info */
	cudaMemcpyToSymbol(GPUMOCNumLayersZ, (void*)&_MOCNumLayersZ, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUMOCNumPinsInSubDom_X, (void*)&_MOCNumPinsInSubDom_X, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUMOCNumPinsInSubDom_Y, (void*)&_MOCNumPinsInSubDom_Y, sizeof(int), 0, cudaMemcpyHostToDevice);

	_dimSurfX = (_MOCNumPinsInSubDom_X + 1) * _MOCNumPinsInSubDom_Y;
    _dimSurfY = _MOCNumPinsInSubDom_X * (_MOCNumPinsInSubDom_Y + 1);
	_dimSurfZ = _MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y;
	cudaMemcpyToSymbol(GPUNumSurfLayer_X, (void*)&_dimSurfX, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUNumSurfLayer_Y, (void*)&_dimSurfY, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUNumSurfLayer_Z, (void*)&_dimSurfZ, sizeof(int), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(GPURayTracModDim, (void*)_RayTracModDim, (size_t)(3) * sizeof(FP_PRECISION_GEO), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUNumSurfaceX, (void*)&_NumSurfaceX, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUNumSurfaceY, (void*)&_NumSurfaceY, sizeof(int), 0, cudaMemcpyHostToDevice);

	_dim = (size_t)(_NumAzimAng) * sizeof(int);
	cudaMalloc((void**)&GPUNumTracksX, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUNumTracksX, (void*)_NumTracksX, _dim, cudaMemcpyHostToDevice);
	// 
	_dim = (size_t)(_NumAzimAng) * sizeof(int);
	cudaMalloc((void**)&GPUNumTracksY, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUNumTracksY, (void*)_NumTracksY, _dim, cudaMemcpyHostToDevice);

	_dim = (size_t)(_NumRTMGeoType) * sizeof(FP_PRECISION_GEO);
	cudaMalloc((void**)&GPURTMWidth_X, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPURTMWidth_X, (void*)_RTMWidth_X, _dim, cudaMemcpyHostToDevice);
	
	_dim = (size_t)(_NumRTMGeoType) * sizeof(FP_PRECISION_GEO);
	cudaMalloc((void**)&GPURTMWidth_Y, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPURTMWidth_Y, (void*)_RTMWidth_Y, _dim, cudaMemcpyHostToDevice);

	_dim = (size_t)(_NumRTMGeoType) * sizeof(int);
	cudaMalloc((void**)&GPUNumFSRsInRTM, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUNumFSRsInRTM, (void*)_NumFSRsInRTM, _dim, cudaMemcpyHostToDevice);
	
	_dim = (size_t)(_TotalNumFSRsInRTM) * sizeof(FP_PRECISION_GEO);
	cudaMalloc((void**)&GPUFSRsAreaInRTM_AntiClockFSR, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUFSRsAreaInRTM_AntiClockFSR, (void*)_FSRsAreaInRTM_AntiClockFSR, _dim, cudaMemcpyHostToDevice);
	
	_dim = (size_t)(_TotalNumFSRs_subDom) * sizeof(FP_PRECISION_GEO);
	cudaMalloc((void**)&GPUFSRArea_SubDom, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUFSRArea_SubDom, (void*)_FSRArea_SubDom, _dim, cudaMemcpyHostToDevice);
	CUDA_CheckError((char *)("GPUCopyTrackInfoHostToDevice"));
}

extern "C"
void GPU_MainAllocMem(int _procID, int _numProcs, M_INT32 _TotalNumFSRs_subDom, int _NumGroups, \
	int _totNumLongTracks,int _NumPolarAng, size_t *_GPUPreMemScalF, size_t *_GPUPreMemRadF,\
	size_t *_GPUPreMemAxiF, size_t *_GPUPreMemMGCMFD, size_t *_GPUPreMemXSs, M_INT32 _TotalNumFSRs_perLayer,\
	M_INT32 _IfResonance, M_INT32 _ProblemType, M_UINT64 _MacroXS_s_MaxNNZ_Layer, M_INT32 _MOCNumLayersZ,\
	M_INT32 _NumAzimAng, size_t *_GPUPreMemCGCMFD, M_INT32 _MOC_NumPinsInSubDom, \
	M_INT32 _NumCMFDGroup, M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, \
	M_INT32 *_CMFDGroupBound, FP_PRECISION *_CMFD_NodeVol, M_INT32 *_NumSubDomains, M_INT32 _MGCMFD_MOCParCurrInnerBound_Num, \
	M_INT32 *_MGCMFD_MOCParCurrInnerBound_Index, M_INT32 *_MGCMFD_MOCParCurrInnerBound_Dir, M_INT32 *_CMFDComm_TarProc, \
	M_INT32 _MGCMFDData_MPIComm_Num, M_INT32 *_MGCMFDData_MPIComm_SumInd, M_INT32 *_MGCMFDMPI_InBoundSourCoeff_Index, \
	M_INT32 *_MGCMFDMPI_InBoundSourCoeff_Dir, FP_PRECISION *_MGCMFDMPI_InBoundSourCoeff_GeoDim, M_INT32 *_MGCMFDData_MPIComm_Index, \
	M_INT32 _CGCMFDData_MPIComm_Num, M_INT32 *_CGCMFDMPI_InBoundSourCoeff_Index, M_INT32 *_CGCMFDMPI_InBoundSourCoeff_Dir, \
	FP_PRECISION *_CGCMFDMPI_InBoundSourCoeff_GeoDim, M_INT32 *_CGCMFDData_MPIComm_Index, M_INT32 _NumCoarseCMFDGroup, \
	M_INT32 *_CGCMFDGroupBound, M_INT32 *_CGCMFDData_MPIComm_SumInd, M_INT32 _MGCMFD_LinearSolver, M_INT32 _CGCMFD_LinearSolver, \
	FILE *_LogFile, M_INT32 _GmresKrylovDim_MG, M_INT32 _GmresKrylovDim_CG, M_INT32 _OUTPUT_CMFDGroupConstant, \
	M_INT32 _OUTPUT_CoarseMesh_GroupConstant, M_INT32 _NumScattOrder, M_INT32 _GPU_TSKernelType, int _isTransient)
{
	M_UINT64 _numFRSsTimesNumGroup, _ind;
	size_t _dim=0;
// #if defined(M_3DMOCex)
// 	int _num;
// #endif // defined(M_3DMOCex)
if (_procID==0) {
	printf("\n [GPU-INFO ]  Allocating Memory on GPU ...");
	fflush(stdout);
	fprintf(_LogFile, "\n [GPU-INFO ]  Allocating Memory on GPU ...");
}
	_numFRSsTimesNumGroup=(M_UINT64)(_TotalNumFSRs_subDom*_NumGroups);

	_ind= _TotalNumFSRs_perLayer * _NumGroups;

    /* Allocate memory on device for MacroXS data arrays */
	_dim = (size_t)(_ind) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUMacroXS_t, _dim);
	_GPUPreMemXSs[0] += _dim;
	CUDA_CheckError((char *)("GPUMacroXS_t"));
#if defined(M_2DMOC)
    if (_GPU_TSKernelType == 12) {
		cudaMalloc((void**)&GPUMacroXS_t_inv, _dim);
	_GPUPreMemXSs[0] += _dim;
	CUDA_CheckError((char *)("GPUMacroXS_t_inv"));
	}
#endif

	if (_OUTPUT_CMFDGroupConstant == 1 || _OUTPUT_CoarseMesh_GroupConstant == 1) {
		printf("\n _ind=%lld",_ind);
		cudaMalloc((void**)&GPUMacroXS_a, _dim);
		_GPUPreMemXSs[0] += _dim;
		CUDA_CheckError((char *)("GPUMacroXS_a"));
		cudaMalloc((void**)&GPUMacroXS_f, _dim);
		_GPUPreMemXSs[0] += _dim;
		CUDA_CheckError((char *)("GPUMacroXS_f"));
		
	}
	cudaMalloc((void**)&GPUMacroXS_nf, _dim);
	_GPUPreMemXSs[0] += _dim;
	CUDA_CheckError((char *)("GPUMacroXS_nf"));
#ifdef FIS_MTX
	cudaMalloc((void**)&GPUMacroXS_chi, _dim * _NumGroups);
	_GPUPreMemXSs[0] += _dim * _NumGroups;
#else
	cudaMalloc((void**)&GPUMacroXS_chi, _dim);
	_GPUPreMemXSs[0] += _dim;
#endif
	CUDA_CheckError((char *)("GPUMacroXS_chi"));

	if (_IfResonance == 0 && (_ProblemType == 0 || _ProblemType == 2)) {
		_dim = (size_t)(_MacroXS_s_MaxNNZ_Layer) * sizeof(FP_PRECISION);
	    cudaMalloc((void**)&GPUMacroXS_s, _dim);
	    _GPUPreMemXSs[0] += _dim;
	    CUDA_CheckError((char *)("cudaMalloc-GPUMacroXS_s-1"));
	}

	// _dim = (size_t)((_TotalNumFSRs_subDom*_NumGroups+_MOCNumLayersZ)) * sizeof(M_UINT64);
	_dim = (size_t)((_TotalNumFSRs_perLayer * (_NumScattOrder + 1) * _NumGroups + 1)) * sizeof(M_UINT64);
	cudaMalloc((void**)&GPUMacroXS_s_NZRowStartID_Layer, _dim);
	_GPUPreMemXSs[0] += _dim;
	CUDA_CheckError((char *)("GPUMacroXS_s_NZRowStartID_Layer"));

    cudaMalloc((void**)&GPUMacroXS_s_NZRowEndID_Layer, _dim - sizeof(M_UINT64));
    _GPUPreMemXSs[0] += (_dim - sizeof (M_UINT64));
    CUDA_CheckError((char *)("GPUMacroXS_s_NZRowEndID_Layer"));
	
	_dim = (size_t)(_TotalNumFSRs_perLayer * (_NumScattOrder + 1) * _NumGroups) * sizeof(M_UINT64);
	cudaMalloc((void**)&GPUMacroXS_s_NZColStartID, _dim);
	_GPUPreMemXSs[0] += _dim;
	CUDA_CheckError((char *)("GPUMacroXS_s_NZColStartID"));

	/* Allocate memory on device for source and flux data arrays */
	_dim = (size_t)(_ind) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUMOCSource, _dim);
	_GPUPreMemScalF[0] += _dim;
	CUDA_CheckError((char *)("cudaMalloc-GPUMOCSource"));

	// _dim = (size_t)(_numFRSsTimesNumGroup) * sizeof(FP_PRECISION_FLUX);
  #ifdef SAVE_ANGLE_FLUX
	if (IfResonance == 1 && _NumScattOrder == 0) {
		_ind= _TotalNumFSRs_perLayer * (SAVE_ORDER + 1) * (SAVE_ORDER + 1) * _NumGroups;
	}
	else {
		_ind= _TotalNumFSRs_perLayer * (_NumScattOrder + 1) * (_NumScattOrder + 1) * _NumGroups;
	}
  #else
	_ind= _TotalNumFSRs_perLayer * (_NumScattOrder + 1) * (_NumScattOrder + 1) * _NumGroups;
  #endif
	_dim = (size_t)(_ind) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUScalarFlux, _dim);
	_GPUPreMemScalF[0] += _dim;
	CUDA_CheckError((char *)("cudaMalloc-GPUScalarFlux"));

    cudaMalloc((void **)&GPUFSRFissRate, _TotalNumFSRs_subDom * _NumGroups * sizeof(FP_DOUBLE));
    CUDA_CheckError((char *)("cudaMalloc-GPUScalarFissRate"));
    _GPUPreMemScalF[0] += _TotalNumFSRs_subDom * _NumGroups * sizeof(FP_DOUBLE);

    cudaMalloc((void **)&GPUScalarFlux_RelErr, _dim);
    CUDA_CheckError((char *)("cudaMalloc-GPUScalarFlux_RelErr"));
    _GPUPreMemScalF[0] += _dim;

//	 _dim = (size_t)(_numFRSsTimesNumGroup) * sizeof(FP_PRECISION_FLUX);
	 cudaMalloc((void**)&GPUScalarFluxOld, _dim);
	 _GPUPreMemScalF[0] += _dim;
	 CUDA_CheckError((char *)("cudaMalloc-GPUScalarFluxOld"));
	// cudaMalloc((void**)&GPUScalarFluxTemp, _dim);
	// _GPUPreMemScalF[0] += _dim;
	// CUDA_CheckError((char *)("cudaMalloc-GPUScalarFluxTemp"));

	_dim = (size_t)(_totNumLongTracks*_NumPolarAng*_NumGroups) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUBoundaryFlux, _dim);
	_GPUPreMemRadF[0] += _dim;
	CUDA_CheckError((char *)("cudaMalloc-GPUBoundaryFlux"));
	
#ifndef MPI_CUDA_UpAngFImme
    _dim = (size_t)(_totNumLongTracks*_NumPolarAng*_NumGroups) * sizeof(FP_PRECISION);
    cudaMalloc((void**)&GPUStartFlux, _dim);
	_GPUPreMemRadF[0] += _dim;
	CUDA_CheckError((char *)("cudaMalloc-GPUStartFlux"));
#endif // !MPI_CUDA_UpAngFImme

#ifdef MPI_USED
    if (_numProcs > 1) {
		_dim = (size_t)(_totNumLongTracks*_NumPolarAng*_NumGroups) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPUMPIFlux, _dim);
		_GPUPreMemRadF[0] += _dim;
		CUDA_CheckError((char *)("cudaMalloc-GPUMPIFlux"));
	}
    
#endif //MPI_USED

#ifdef GPU_STREAM_USED
    // _dim = (size_t)(_totNumLongTracks*_numPolarAng*_numGroups) * sizeof(FP_PRECISION);
    // cudaHostAlloc((void **)&CGBoundaryFlux, _dim, cudaHostAllocDefault);
    // CUDA_CheckError((char *)("cudaMalloc-CGBoundaryFlux"));
    // cudaHostAlloc((void **)&CGLTInciAngFlux, _dim, cudaHostAllocDefault);
    // CUDA_CheckError((char *)("cudaMalloc-CGLTInciAngFlux"));
    // cudaMemcpy((void*)GPUBoundaryFlux, (void*)CGBoundaryFlux, _dim, cudaMemcpyHostToDevice);
    // CUDA_CheckError((char *)("cudaMalloc-GPUBoundaryFlux"));
#else
// BoundaryFlux = AllocMemInciAngFlux(_NumTrackSet, _NumLTs_HS, _NumGroups, _NumPolarAng, _MOCNumLayersZ, (char *)("BoundaryFlux"));	
#endif // GPU_STREAM_USED

#if defined(M_2DMOC)
    if (_NumScattOrder > 0 || _GPU_TSKernelType >= 10) {
		const M_INT32 _numAzimAlloc = GPUAzimSweepMode > 0
			? (GPUAzimSweepMode < _NumAzimAng ? GPUAzimSweepMode : _NumAzimAng)
			: _NumAzimAng;
		_dim = (size_t)(_TotalNumFSRs_perLayer)*(size_t)(_NumGroups*_numAzimAlloc*_NumPolarAng) * sizeof(FP_PRECISION_FLUX);
		cudaMalloc((void**)&GPUFSRAveAngFlux_Top, _dim);
		_GPUPreMemAxiF[0] += _dim;
		CUDA_CheckError((char *)("cudaMalloc-GPUFSRAveAngFlux_Top"));

		if (_GPU_TSKernelType >= 11) {
			cudaMalloc((void**)&GPUFSRAngScattSour, _dim);
			_GPUPreMemAxiF[0] += _dim;
			CUDA_CheckError((char *)("cudaMalloc-GPUFSRAngScattSour"));	
		}
		if (GPUAzimSweepMode) {
			const size_t _scalarDim = (size_t)(_TotalNumFSRs_perLayer) *
				(size_t)(_NumGroups) * (size_t)(_NumScattOrder + 1) *
				(size_t)(_NumScattOrder + 1) * sizeof(FP_PRECISION_FLUX);
			cudaMalloc((void**)&GPUScalarFluxAzimAccum, _scalarDim);
			_GPUPreMemScalF[0] += _scalarDim;
			CUDA_CheckError((char *)("cudaMalloc-GPUScalarFluxAzimAccum"));
		}
	}

  #ifdef SAVE_ANGLE_FLUX
    if (IfResonance == 1 && _NumScattOrder == 0) {
		_dim = (size_t)(_TotalNumFSRs_perLayer)*(size_t)(_NumGroups*_NumAzimAng*_NumPolarAng) * sizeof(FP_PRECISION_FLUX);
		cudaMalloc((void**)&GPUFSRAveAngFlux_Top, _dim);
		_GPUPreMemAxiF[0] += _dim;
		CUDA_CheckError((char *)("cudaMalloc-GPUFSRAveAngFlux_Top"));
		_dim = (size_t)(_NumAzimAng*_NumPolarAng*(SAVE_ORDER + 1) * (SAVE_ORDER + 1)) * sizeof(FP_PRECISION_FLUX);
		cudaMalloc((void**)&GPUSldSpheHarmCoeff, _dim);
		_GPUPreMemAxiF[0] += _dim;
		CUDA_CheckError((char *)("cudaMalloc-GPUSldSpheHarmCoeff"));
	}
  #endif
#else
	// _num = _TotalNumFSRs_perLayer * _NumGroups*_NumAzimAng*(_NumPolarAng/2);
	_dim = (size_t)(_TotalNumFSRs_perLayer)*(size_t)(_NumGroups*_NumAzimAng*(_NumPolarAng/2)) * sizeof(FP_PRECISION_FLUX);
	cudaMalloc((void**)&GPUFSRAveAngFlux_Top, _dim);
	_GPUPreMemAxiF[0] += _dim;
	CUDA_CheckError((char *)("cudaMalloc-GPUFSRAveAngFlux_Top"));
	cudaMalloc((void**)&GPUFSRAveAngFlux_Bott, _dim);
	_GPUPreMemAxiF[0] += _dim;
	CUDA_CheckError((char *)("cudaMalloc-GPUFSRAveAngFlux_Bott"));
	if (_NumScattOrder > 0 || _GPU_TSKernelType >= 11) {
		cudaMalloc((void**)&GPUFSRAngScattSour, _dim);
		_GPUPreMemAxiF[0] += _dim;
		CUDA_CheckError((char *)("cudaMalloc-GPUFSRAngScattSour"));	
	}
#endif // defined(M_3DMOCex)

    /* S_lm intermediate for the LM/AP high-order scattering source (TS types 13/14).
     * Layout: (order+1)^2 x FSR_layer x group, same moment ordering as GPUScalarFlux. */
    if (_GPU_TSKernelType == 13 || _GPU_TSKernelType == 14) {
        _dim = (size_t) (_TotalNumFSRs_perLayer) * (size_t) ((_NumScattOrder + 1) * (_NumScattOrder + 1) * _NumGroups) * sizeof (FP_PRECISION);
        cudaMalloc((void **) &GPUFSRAngScatterSourceLM, _dim);
        _GPUPreMemAxiF[0] += _dim;
        CUDA_CheckError((char *) ("cudaMalloc-GPUFSRAngScatterSourceLM"));
    }

    // _dim = (size_t)(_numFRSsTimesNumGroup) * sizeof(FP_PRECISION);
    // cudaMalloc((void**)&GPUFluxResidual, _dim);
	// CUDA_CheckError((char *)("cudaMalloc-GPUFluxResidual"));


#if defined(M_MG_CMFD)
     GPUCMFD_AllocMem(_procID, _GPUPreMemMGCMFD, _GPUPreMemCGCMFD, _MOC_NumPinsInSubDom, \
		 _NumGroups, _NumCMFDGroup, _MOCNumLayersZ, _MOCNumPinsInSubDom_X, _MOCNumPinsInSubDom_Y, \
		 _CMFDGroupBound, _CMFD_NodeVol, _NumSubDomains, _MGCMFD_MOCParCurrInnerBound_Num, \
		 _MGCMFD_MOCParCurrInnerBound_Index, _MGCMFD_MOCParCurrInnerBound_Dir, _CMFDComm_TarProc, \
		 _NumPolarAng, _MGCMFDData_MPIComm_Num, _MGCMFDData_MPIComm_SumInd, _MGCMFDMPI_InBoundSourCoeff_Index, \
		 _MGCMFDMPI_InBoundSourCoeff_Dir, _MGCMFDMPI_InBoundSourCoeff_GeoDim, _MGCMFDData_MPIComm_Index, \
		 _CGCMFDData_MPIComm_Num, _CGCMFDMPI_InBoundSourCoeff_Index, _CGCMFDMPI_InBoundSourCoeff_Dir, \
		 _CGCMFDMPI_InBoundSourCoeff_GeoDim, _CGCMFDData_MPIComm_Index, _NumCoarseCMFDGroup, \
		 _CGCMFDGroupBound, _CGCMFDData_MPIComm_SumInd, _MGCMFD_LinearSolver, _CGCMFD_LinearSolver, \
		 _LogFile, _GmresKrylovDim_MG, _GmresKrylovDim_CG, _OUTPUT_CMFDGroupConstant, _OUTPUT_CoarseMesh_GroupConstant, _isTransient, _numProcs);
#endif
    _dim = (size_t)(GPUNumThreadBlocks*GPUNumThreadPerBlock)* sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUReacRateCollect, _dim);
	CUDA_CheckError((char *)("cudaMalloc-GPUReacRateCollect"));

    ReacRateCollect=GPUAllocMem_FP_I((size_t)(GPUNumThreadBlocks*GPUNumThreadPerBlock), (char *)("ReacRateCollect"));
    FluxResidual=GPUAllocMem_FP_I((size_t)(_numFRSsTimesNumGroup), (char *)("ReacRateCollect"));

 //   if (_procID==0) {
 //       GPUKeffRec = GPUAllocMem_FP_I(MaxNumOuterIter_MOC, (char *)("GPUKeffRec"));
 //       GPUFluxResRec = GPUAllocMem_FP_I(MaxNumOuterIter_MOC, (char *)("GPUFluxResRec"));
	//}

	CUDA_CheckError((char *)("GPU_MainAllocMem"));
}



/**
 * @ Function: GPUTransportSweepInit()
 *   introduction: initialize the transport sweep
 *                 allocate memory for flux arrays on device
 *                 set the initial values, like keff=1.0
 * @ Parameter List: _totalNumFSRs - total number of FSRs in current domain
 *   Return: void
 * @ Created     : 2018/06/05, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
extern "C"
void GPUInitScalarFlux_Dev(int _dim, FP_PRECISION_FLUX _val)
{
	GPUVectorInit_1DFP_Flux<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(GPUScalarFlux, _dim, _val);
	CUDA_CheckError((char *)("GPUVectorInit_1DFP_Flux-GPUScalarFlux"));
}

extern "C"
void GPUInitAngFlux_Dev(int _dim, FP_PRECISION_FLUX _val, int _id)
{
	if (_id == 0) {
		GPUVectorInit_1DFP_Flux<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(GPUFSRAveAngFlux_Top, _dim, _val);
		CUDA_CheckError((char *)("GPUVectorInit_1DFP_Flux-GPUFSRAveAngFlux_Top"));	
	}
	else if(_id == 1) {
		GPUVectorInit_1DFP_Flux<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(GPUFSRAveAngFlux_Bott, _dim, _val);
		CUDA_CheckError((char *)("GPUVectorInit_1DFP_Flux-GPUFSRAveAngFlux_Bott"));	
	}
	else{}
}

extern "C"
void GPUInitAzimAccumFlux(int _dim, FP_PRECISION_FLUX _value)
{
	if (GPUScalarFluxAzimAccum == NULL) {
		return;
	}
	GPUVectorInit_1DFP_Flux<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(
		GPUScalarFluxAzimAccum, _dim, _value);
	CUDA_CheckError((char *)("GPUInitAzimAccumFlux"));
}

extern "C"
void GPUFinalizeAzimAccumFlux(int _dim)
{
	if (GPUScalarFluxAzimAccum == NULL) {
		return;
	}
	cudaMemcpy(GPUScalarFlux, GPUScalarFluxAzimAccum,
		(size_t) _dim * sizeof(FP_PRECISION_FLUX), cudaMemcpyDeviceToDevice);
	CUDA_CheckError((char *)("GPUFinalizeAzimAccumFlux"));
}

extern "C"
extern "C"
void ScalarFluxCopy_Dev2Host(FP_PRECISION_FLUX *_ScalarFlux_host, size_t _dim, int _pointer, char *_para)
{
	cudaMemcpy((void*)&_ScalarFlux_host[_pointer], (void*)GPUScalarFlux, _dim * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
	CUDA_CheckError((char *)(_para));
}

extern "C"
extern "C"
extern "C"
void GPUComputeFSRSource_Layer(int _layID, int _TotalNumFSRs_subDom, int _TotalNumFSRs_perLayer, int _NumGroups,
                               M_UINT64 *_MacroXS_s_NZStartID_Layer, M_UINT64 *_MacroXS_s_NNZ_Layer,
                               M_INT32 _ProblemType, FP_PRECISION *_MacroXS_t, FP_PRECISION *_MacroXS_nf,
                               FP_PRECISION *_Chi, FP_PRECISION *_MacroXS_s, FP_PRECISION *_FixedSour,
                               FP_PRECISION _invKeff, M_INT32 _NumScattOrder,
                               M_INT32 _GPU_TSKernelType, M_UINT64 *_MacroXS_s_NZRowStartID_Layer,
                               M_UINT64 *_MacroXS_s_NZColStartID, FP_PRECISION_FLUX *_ScalarFluxOld,
                               int _polarBoundLowLim,
                               int _isTransient, FP_PRECISION *_fissSource, FP_PRECISION *_tranConstA,
                               FP_PRECISION *_tranConstB, FP_PRECISION *_tranConstC)
{
    cudaEvent_t start, stop;
    float fiss_time, scatt_time, merge_time;
    FP_PRECISION_TIME _fissionStageStart = 0.0, _fissionStageEnd = 0.0;
    FP_PRECISION_TIME _scattStageStart = 0.0, _scattStageEnd = 0.0;
    const int _measureLMAP = (_GPU_TSKernelType == 13 || _GPU_TSKernelType == 14);
    cudaEvent_t _lmStart = NULL, _lmStop = NULL;
    cudaEvent_t _apStart = NULL, _apStop = NULL;
    cudaEvent_t _auxStart = NULL, _auxStop = NULL;
    int _lmRecorded = 0, _apRecorded = 0, _auxRecorded = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    if (_measureLMAP) {
        cudaEventCreate(&_lmStart);
        cudaEventCreate(&_lmStop);
        cudaEventCreate(&_apStart);
        cudaEventCreate(&_apStop);
        cudaEventCreate(&_auxStart);
        cudaEventCreate(&_auxStop);
    }

	// printf("\n [  DEBUG  ]  DDD:%d\n", _GPU_TSKernelType);

#ifdef M_3DMOCex
    GPUCopyMacroXSsHostToDevice_Layer(_layID,_TotalNumFSRs_subDom, _TotalNumFSRs_perLayer, _NumGroups, _MacroXS_s_NZStartID_Layer, \
		_MacroXS_s_NNZ_Layer, _ProblemType, _MacroXS_t, NULL, NULL, _MacroXS_nf, _Chi, _MacroXS_s, _FixedSour, \
		_NumScattOrder, _MacroXS_s_NZRowStartID_Layer, _MacroXS_s_NZColStartID, 0, _ScalarFluxOld);
#endif

#if defined(M_2DMOC)
    if (_GPU_TSKernelType == 12) {
		GPUVectorInv_FP_I<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(GPUMacroXS_t,GPUMacroXS_t_inv, _TotalNumFSRs_perLayer*_NumGroups);
	}
#endif

	if (_NumScattOrder == 0 && _GPU_TSKernelType < 10) {
		/* P0 uses one fused source kernel for fission and scattering.  Keep
		 * its timing separate from the characteristic-line sweep. */
		cudaDeviceSynchronize();
		const FP_PRECISION_TIME _p0SourceStart = TallyMonotonicTime_CPU();
		#ifdef SPLIT_FIS_S_SCA_S
            compute_source_on_device_inner_iteration<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(_layID,
                                                                                    GPUScalarFlux, GPUScalarFluxOld, GPUMOCSource,
                                                                                    GPUMacroXS_t, GPUMacroXS_nf,
                                                                                    GPUMacroXS_s, GPUMacroXS_chi,
                                                                                    _invKeff,
                                                                                    GPUMacroXS_s_NZRowStartID_Layer,
                                                                                    GPUMacroXS_s_NZRowEndID_Layer,
                                                                                    GPUMacroXS_s_NZColStartID);
		#else
            compute_source_on_device<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(_layID,
                                                                                    GPUScalarFlux, GPUMOCSource,
                                                                                    GPUMacroXS_t, GPUMacroXS_nf,
                                                                                    GPUMacroXS_s, GPUMacroXS_chi,
                                                                                    _invKeff,
                                                                                    GPUMacroXS_s_NZRowStartID_Layer,
                                                                                    GPUMacroXS_s_NZRowEndID_Layer,
                                                                                    GPUMacroXS_s_NZColStartID);
		#endif

/*
            GPUComputeFSRFissSourceOnDevice_paraFSR_paraGroup_Layer<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(
                    GPUScalarFlux, GPUMOCSource, GPUMacroXS_nf, GPUMacroXS_chi, _invKeff);

            GPUComputeFSRScattSourceOnDevice_paraFSR_paraGroup_Layer_opt1<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(
                    _layID, GPUScalarFlux,
                    GPUMOCSource,
                    GPUMacroXS_s,
                    GPUMacroXS_s_NZRowStartID_Layer,
                    GPUMacroXS_s_NZRowEndID_Layer,
                    GPUMacroXS_s_NZColStartID);

            GPUComputeFSRTotalSourceOnDevice_paraFSR_paraGroup_Layer<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(
                    _layID, GPUScalarFlux, GPUMOCSource, GPUMacroXS_t);
*/

		cudaDeviceSynchronize();
		TimeRec_P0Source += TallyMonotonicTime_CPU() - _p0SourceStart;
	}
	else {
        const int _useClusterScattSource = GPUUseFsrClusterScattSource();
        cudaDeviceSynchronize();
        _fissionStageStart = TallyMonotonicTime_CPU();
//        cudaEventRecord(start, NULL);
		GPUComputeFSRFissSourceOnDevice<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(GPUScalarFlux, \
			GPUMOCSource, GPUMacroXS_nf, GPUMacroXS_chi, _invKeff);
//        cudaEventRecord(stop, NULL);
//        cudaEventSynchronize(stop);
//        cudaEventElapsedTime(&fiss_time, start, stop);

	#ifdef HO_FIXED_SOURCE
		GPUOverWriteFissSourceWithGivenSource<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(GPUMOCSource, GPUMacroXS_nf);
	#endif
		cudaDeviceSynchronize();
		_fissionStageEnd = TallyMonotonicTime_CPU();
		TimeRec_FissionSource += _fissionStageEnd - _fissionStageStart;

		cudaDeviceSynchronize();
		_scattStageStart = TallyMonotonicTime_CPU();


#if 0
			if (_GPU_TSKernelType == 11) {
//            cudaEventRecord(start, NULL);
            GPUInitFSRHOSourceOnDevice<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(GPUFSRAngScattSour, \
                GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID, \
                GPUSpheHarmCoeff, _polarBoundLowLim);

            size_t gridSize = (_TotalNumFSRs_perLayer*_NumGroups - 1) / GPUBlockDim + 1;
//            size_t gridSize = (_TotalNumFSRs_perLayer*_NumGroups*32*6 - 1) / GPUBlockDim + 1;
	            if (_useClusterScattSource) {
                GPUComputeFSRScattSourceOnDevice_Cluster<<<gridSize, GPUBlockDim>>>(GPUFSRAngScattSour, \
                    GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer, \
                    GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID, \
                    GPUSpheHarmCoeff, _polarBoundLowLim);
            }
            else {
                GPUComputeFSRScattSourceOnDevice<<<gridSize, GPUBlockDim>>>(GPUFSRAngScattSour, \
                    GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID, \
                    GPUSpheHarmCoeff, _polarBoundLowLim);
            }
//            cudaEventRecord(stop, NULL);
//            cudaEventSynchronize(stop);
//            cudaEventElapsedTime(&scatt_time, start, stop);
//int GPUGridDim_AniSource_para_layer = (_TotalNumFSRs_perLayer*_NumGroups*48*3- 1) / GPUBlockDim+1;
//            GPUComputeFSRScattSourceOnDevice<<<GPUGridDim_AniSource_para_layer, GPUBlockDim>>>(GPUFSRAngScattSour, \
//				GPUScalarFlux,GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID,\
//				GPUSpheHarmCoeff, _polarBoundLowLim);
// int _pol;
// #if defined(M_2DMOC)
// _pol=NumPolarAng[0];
// #else
// _pol=NumPolarAng[0]/2;
// #endif
// 	GPUPrintf1DArray_FP(_TotalNumFSRs_perLayer * _NumGroups * NumAzimAng[0] * _pol, GPUFSRAngScattSour, (char *)("b4 ts GPUFSRAngScattSour"),NULL);
			}
#endif
#if 0
			if (_GPU_TSKernelType == 12) {
            cudaEventRecord(start, NULL);
			GPUComputeFSRScattSourceOnDevice_addFissSour<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(GPUFSRAngScattSour, \
				GPUScalarFlux,GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID,\
				GPUSpheHarmCoeff, GPUMOCSource, _polarBoundLowLim, GPUMacroXS_t_inv);
			CUDA_CheckError((char *)("GPUComputeFSRScattSourceOnDevice_addFissSour"));
            cudaEventRecord(stop, NULL);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&scatt_time, start, stop);
// int _pol;
// #if defined(M_2DMOC)
// _pol=NumPolarAng[0];
// #else
// _pol=NumPolarAng[0]/2;
// #endif
// 	GPUPrintf1DArray_FP(_TotalNumFSRs_perLayer * _NumGroups * NumAzimAng[0] * _pol, GPUFSRAngScattSour, (char *)("b4 ts GPUFSRAngScattSour"),NULL);
		}
#endif
		if (_GPU_TSKernelType == 13 || _GPU_TSKernelType == 14) {
			/* LM/AP two-stage high-order scattering source (full-angle version).
			 * Stage 1 (LM): moment-resolved, angle-independent group coupling.
			 * Stage 2 (AP): angular projection, shared by both LM variants.
			 * Interface notes:
			 *  - S_lm is angle-independent; the future azimuth/polar-batch mode
			 *    only needs to slice the AP launch and the angular source buffer;
			 *  - 3D support is kept through the M_2DMOC/M_3DMOCex branches. */
			/* select the AP kernel variant through the APKernel environment
			 * variable: 0 = base (accumulate), 1/2/3 = Opt1/Opt2/Opt3 (direct write) */
			int apKernel = 0;
			const char* apKernelEnv = getenv("APKernel");
			if (apKernelEnv != NULL) {
				apKernel = atoi(apKernelEnv);
				if (apKernel < 0 || apKernel > 3) {
					apKernel = 0;
				}
			}

			if (_measureLMAP && (apKernel == 0 || _GPU_TSKernelType == 14)) {
				cudaEventRecord(_auxStart, 0);
				_auxRecorded = 1;
			}

			if (apKernel == 0) {
				/* base AP accumulates into the angular source: zero it first */
				GPUInitFSRHOSourceOnDevice<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(GPUFSRAngScattSour, \
					GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID, \
					GPUSpheHarmCoeff, _polarBoundLowLim);
			}

			size_t lmGridSize = ((size_t) (_TotalNumFSRs_perLayer) * (size_t) (_NumGroups) * \
				(size_t) ((_NumScattOrder + 1) * (_NumScattOrder + 1)) - 1) / GPUBlockDim + 1;

			if (_GPU_TSKernelType == 13) {
				/* target-group-oriented (gather) LM: direct write, deterministic */
				if (_measureLMAP && _auxRecorded) {
					cudaEventRecord(_auxStop, 0);
				}
				if (_measureLMAP) {
					cudaEventRecord(_lmStart, 0);
				}
				if (_useClusterScattSource) {
					GPUComputeFSRScatterSourceLMOnDevice_grpOut_Cluster<<<lmGridSize, GPUBlockDim>>>(GPUFSRAngScatterSourceLM, \
						GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer, \
						GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID);
				}
				else {
					GPUComputeFSRScatterSourceLMOnDevice_grpOut<<<lmGridSize, GPUBlockDim>>>(GPUFSRAngScatterSourceLM, \
						GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID);
				}
				if (_measureLMAP) {
					cudaEventRecord(_lmStop, 0);
					_lmRecorded = 1;
				}
			}
			else {
				/* source-group-oriented (scatter) LM: atomic accumulation,
				 * S_lm must be zeroed before launch */
				cudaMemset(GPUFSRAngScatterSourceLM, 0, (size_t) (_TotalNumFSRs_perLayer) * (size_t) (_NumGroups) * \
					(size_t) ((_NumScattOrder + 1) * (_NumScattOrder + 1)) * sizeof (FP_PRECISION));
				if (_measureLMAP && _auxRecorded) {
					cudaEventRecord(_auxStop, 0);
				}
				if (_measureLMAP) {
					cudaEventRecord(_lmStart, 0);
				}
				if (_useClusterScattSource) {
					GPUComputeFSRScatterSourceLMOnDevice_grpIn_Cluster<<<lmGridSize, GPUBlockDim>>>(GPUFSRAngScatterSourceLM, \
						GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer, \
						GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID);
				}
				else {
					GPUComputeFSRScatterSourceLMOnDevice_grpIn<<<lmGridSize, GPUBlockDim>>>(GPUFSRAngScatterSourceLM, \
						GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID);
				}
				if (_measureLMAP) {
					cudaEventRecord(_lmStop, 0);
					_lmRecorded = 1;
				}
			}

			if (_measureLMAP) {
				cudaEventRecord(_apStart, 0);
			}

			/* AP stage: Opt1 uses one thread per (FSR, group); Opt2/3 use one
			 * thread per (FSR, group, angle) and need a full angle grid */
			if (apKernel == 0) {
				GPUComputeFSRScatterSourceAPOnDevice<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(GPUFSRAngScattSour,
					GPUSpheHarmCoeff,
					GPUFSRAngScatterSourceLM);
			} else if (apKernel == 1) {
				GPUComputeFSRScatterSourceAPOnDeviceOpt1<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(
					 GPUFSRAngScattSour,
					 GPUSpheHarmCoeff,
					 GPUFSRAngScatterSourceLM);
			} else {
				size_t apGridSize; {
					int _apNumPol = NumPolarAng[0];
					#if defined(M_3DMOCex)
						_apNumPol = NumPolarAng[0] / 2;
					#endif
						apGridSize = ((size_t) (_TotalNumFSRs_perLayer) * (size_t) (_NumGroups) *
						              (size_t) (NumAzimAng[0]) * (size_t) (_apNumPol) - 1) / GPUBlockDim + 1;
				}
				if (apKernel == 2) {
					GPUComputeFSRScatterSourceAPOnDeviceOpt2<<<apGridSize, GPUBlockDim>>>(GPUFSRAngScattSour,
						GPUSpheHarmCoeff,
						GPUFSRAngScatterSourceLM);
				} else {
					GPUComputeFSRScatterSourceAPOnDeviceOpt3<<<apGridSize, GPUBlockDim>>>(GPUFSRAngScattSour,
						GPUSpheHarmCoeff,
						GPUFSRAngScatterSourceLM);
				}
			}
			if (_measureLMAP) {
				cudaEventRecord(_apStop, 0);
				_apRecorded = 1;
			}
			CUDA_CheckError((char *) ("LM-AP high-order scattering source"));
		}
		if (_GPU_TSKernelType == 15 || _GPU_TSKernelType == 16) {
			GPUInitFSRHOSourceOnDevice<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(GPUFSRAngScattSour, \
				GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID, \
				GPUSpheHarmCoeff, _polarBoundLowLim);

			int _sourceNumPol = NumPolarAng[0];
			#if defined(M_3DMOCex)
				_sourceNumPol = NumPolarAng[0] / 2;
			#endif
			const size_t _groupWork = (size_t) _TotalNumFSRs_perLayer * (size_t) _NumGroups;
			const size_t _angleWork = _groupWork * (size_t) NumAzimAng[0] * (size_t) _sourceNumPol;
			const size_t _groupGridSize = (_groupWork - 1) / GPUBlockDim + 1;
			const size_t _angleGridSize = (_angleWork - 1) / GPUBlockDim + 1;

			if (_GPU_TSKernelType == 15) {
				if (_useClusterScattSource) {
					GPUComputeFSRScattSourceOnDevice_FsrGrpInAng_Cluster<<<_angleGridSize, GPUBlockDim>>>(GPUFSRAngScattSour,
						GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
						GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
						GPUSpheHarmCoeff, _polarBoundLowLim);
				}
				else {
					GPUComputeFSRScattSourceOnDevice_FsrGrpInAng<<<_angleGridSize, GPUBlockDim>>>(GPUFSRAngScattSour,
						GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
						GPUMacroXS_s_NZColStartID, GPUSpheHarmCoeff, _polarBoundLowLim);
				}
			}
			else if (_GPU_TSKernelType == 16) {
				if (_useClusterScattSource) {
					GPUComputeFSRScattSourceOnDevice_FsrGrpOutAng_Cluster<<<_angleGridSize, GPUBlockDim>>>(GPUFSRAngScattSour,
						GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
						GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
						GPUSpheHarmCoeff, _polarBoundLowLim);
				}
				else {
					GPUComputeFSRScattSourceOnDevice_FsrGrpOutAng<<<_angleGridSize, GPUBlockDim>>>(GPUFSRAngScattSour,
						GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
						GPUMacroXS_s_NZColStartID, GPUSpheHarmCoeff, _polarBoundLowLim);
				}
			}
			#if !defined(PAPER_TRANSPORT_ONLY)
			else if (_GPU_TSKernelType == 17) {
				if (_useClusterScattSource) {
					GPUComputeFSRScattSourceOnDevice_FsrGrpIn_Cluster<<<_groupGridSize, GPUBlockDim>>>(GPUFSRAngScattSour,
						GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
						GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
						GPUSpheHarmCoeff, _polarBoundLowLim);
				}
				else {
					GPUComputeFSRScattSourceOnDevice_FsrGrpIn<<<_groupGridSize, GPUBlockDim>>>(GPUFSRAngScattSour,
						GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
						GPUMacroXS_s_NZColStartID, GPUSpheHarmCoeff, _polarBoundLowLim);
				}
			}
			else {
				if (_useClusterScattSource) {
					GPUComputeFSRScattSourceOnDevice_FsrGrpOut_Cluster<<<_groupGridSize, GPUBlockDim>>>(GPUFSRAngScattSour,
						GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
						GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
						GPUSpheHarmCoeff, _polarBoundLowLim);
				}
				else {
					GPUComputeFSRScattSourceOnDevice_FsrGrpOut<<<_groupGridSize, GPUBlockDim>>>(GPUFSRAngScattSour,
						GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
						GPUMacroXS_s_NZColStartID, GPUSpheHarmCoeff, _polarBoundLowLim);
				}
			}
			#endif
			CUDA_CheckError((char *) ("experimental fused high-order scattering source"));
		}
#if defined(M_2DMOC) && !defined(PAPER_TRANSPORT_ONLY)
		if (_GPU_TSKernelType == 19 || _GPU_TSKernelType == 20) {
			/* 2D fused LM/AP kernels with thread-private moment storage.
			 * Type 19 writes each outgoing-group source directly. Type 20
			 * scatters from incoming groups and therefore needs a cleared
			 * angular source plus atomic accumulation. */
			const size_t _groupWork = (size_t) _TotalNumFSRs_perLayer * (size_t) _NumGroups;
			const size_t _groupGridSize = (_groupWork - 1) / GPUBlockDim + 1;

			if (_GPU_TSKernelType == 19) {
				switch (_NumScattOrder) {
				case 0:
					if (_useClusterScattSource) {
						GPUComputeFSRScattSourceOnDevice_FsrGrpOutLocalLM_Cluster_P0<<<_groupGridSize, GPUBlockDim>>>(
							GPUFSRAngScattSour, GPUScalarFlux, GPUMacroXS_s,
							GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZRowEndID_Layer,
							GPUMacroXS_s_NZColStartID, GPUSpheHarmCoeff, _polarBoundLowLim);
					}
					else {
						GPUComputeFSRScattSourceOnDevice_FsrGrpOutLocalLM_P0<<<_groupGridSize, GPUBlockDim>>>(
							GPUFSRAngScattSour, GPUScalarFlux, GPUMacroXS_s,
							GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID,
							GPUSpheHarmCoeff, _polarBoundLowLim);
					}
					break;
				case 1:
					if (_useClusterScattSource) {
						GPUComputeFSRScattSourceOnDevice_FsrGrpOutLocalLM_Cluster_P1<<<_groupGridSize, GPUBlockDim>>>(
							GPUFSRAngScattSour, GPUScalarFlux, GPUMacroXS_s,
							GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZRowEndID_Layer,
							GPUMacroXS_s_NZColStartID, GPUSpheHarmCoeff, _polarBoundLowLim);
					}
					else {
						GPUComputeFSRScattSourceOnDevice_FsrGrpOutLocalLM_P1<<<_groupGridSize, GPUBlockDim>>>(
							GPUFSRAngScattSour, GPUScalarFlux, GPUMacroXS_s,
							GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID,
							GPUSpheHarmCoeff, _polarBoundLowLim);
					}
					break;
				case 2:
					if (_useClusterScattSource) {
						GPUComputeFSRScattSourceOnDevice_FsrGrpOutLocalLM_Cluster_P2<<<_groupGridSize, GPUBlockDim>>>(
							GPUFSRAngScattSour, GPUScalarFlux, GPUMacroXS_s,
							GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZRowEndID_Layer,
							GPUMacroXS_s_NZColStartID, GPUSpheHarmCoeff, _polarBoundLowLim);
					}
					else {
						GPUComputeFSRScattSourceOnDevice_FsrGrpOutLocalLM_P2<<<_groupGridSize, GPUBlockDim>>>(
							GPUFSRAngScattSour, GPUScalarFlux, GPUMacroXS_s,
							GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID,
							GPUSpheHarmCoeff, _polarBoundLowLim);
					}
					break;
				case 3:
					if (_useClusterScattSource) {
						GPUComputeFSRScattSourceOnDevice_FsrGrpOutLocalLM_Cluster_P3<<<_groupGridSize, GPUBlockDim>>>(
							GPUFSRAngScattSour, GPUScalarFlux, GPUMacroXS_s,
							GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZRowEndID_Layer,
							GPUMacroXS_s_NZColStartID, GPUSpheHarmCoeff, _polarBoundLowLim);
					}
					else {
						GPUComputeFSRScattSourceOnDevice_FsrGrpOutLocalLM_P3<<<_groupGridSize, GPUBlockDim>>>(
							GPUFSRAngScattSour, GPUScalarFlux, GPUMacroXS_s,
							GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID,
							GPUSpheHarmCoeff, _polarBoundLowLim);
					}
					break;
				default:
					if (_useClusterScattSource) {
						GPUComputeFSRScattSourceOnDevice_FsrGrpOutLocalLM_Cluster<<<_groupGridSize, GPUBlockDim>>>(
							GPUFSRAngScattSour, GPUScalarFlux, GPUMacroXS_s,
							GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZRowEndID_Layer,
							GPUMacroXS_s_NZColStartID, GPUSpheHarmCoeff, _polarBoundLowLim);
					}
					else {
						GPUComputeFSRScattSourceOnDevice_FsrGrpOutLocalLM<<<_groupGridSize, GPUBlockDim>>>(
							GPUFSRAngScattSour, GPUScalarFlux, GPUMacroXS_s,
							GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID,
							GPUSpheHarmCoeff, _polarBoundLowLim);
					}
					break;
				}
			}
			else {
				GPUInitFSRHOSourceOnDevice<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(GPUFSRAngScattSour,
					GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
					GPUMacroXS_s_NZColStartID, GPUSpheHarmCoeff, _polarBoundLowLim);
				if (_useClusterScattSource) {
					GPUComputeFSRScattSourceOnDevice_FsrGrpInLocalLM_Cluster<<<_groupGridSize, GPUBlockDim>>>(
						GPUFSRAngScattSour, GPUScalarFlux, GPUMacroXS_s,
						GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZRowEndID_Layer,
						GPUMacroXS_s_NZColStartID, GPUSpheHarmCoeff, _polarBoundLowLim);
				}
				else {
					GPUComputeFSRScattSourceOnDevice_FsrGrpInLocalLM<<<_groupGridSize, GPUBlockDim>>>(
						GPUFSRAngScattSour, GPUScalarFlux, GPUMacroXS_s,
						GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID,
						GPUSpheHarmCoeff, _polarBoundLowLim);
				}
			}
			CUDA_CheckError((char *) ("fused local-LM high-order scattering source"));
		}
		if (_GPU_TSKernelType == 21) {
			if (GPUBlockDim < 32 || GPUBlockDim % 32 != 0) {
				printf("\n[ERROR] GPUBlockDim must be a positive multiple of 32 for Type 21.\n");
				exit(70);
			}

			const size_t _groupWork = (size_t) _TotalNumFSRs_perLayer * (size_t) _NumGroups;
			const size_t _warpsPerBlock = (size_t) GPUBlockDim / 32u;
			const size_t _warpGridSize = (_groupWork - 1) / _warpsPerBlock + 1;
			const size_t _numLM = (size_t) (_NumScattOrder + 1) *
				(size_t) (_NumScattOrder + 1);
			const size_t _warpSharedBytes = _warpsPerBlock * _numLM * sizeof(FP_PRECISION);

			if (_useClusterScattSource) {
				GPUComputeFSRScattSourceOnDevice_FsrGrpOutWarpLM_Cluster<<<
					_warpGridSize, GPUBlockDim, _warpSharedBytes>>>(
					GPUFSRAngScattSour, GPUScalarFlux, GPUMacroXS_s,
					GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZRowEndID_Layer,
					GPUMacroXS_s_NZColStartID, GPUSpheHarmCoeff, _polarBoundLowLim);
			}
			else {
				GPUComputeFSRScattSourceOnDevice_FsrGrpOutWarpLM<<<
					_warpGridSize, GPUBlockDim, _warpSharedBytes>>>(
					GPUFSRAngScattSour, GPUScalarFlux, GPUMacroXS_s,
					GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID,
					GPUSpheHarmCoeff, _polarBoundLowLim);
			}
			CUDA_CheckError((char *) ("fused warp-LM high-order scattering source"));
		}
#endif
#if defined(PAPER_TRANSPORT_ONLY)
		/* Type 17--21 are experimental paths and are intentionally omitted
		 * from the publication transport snapshot. */
#endif
//        GPU_FISS_CALC_TIME += fiss_time;
//        GPU_SCAT_CALC_TIME += scatt_time;
	}

	if (_scattStageStart > 0.0) {
		cudaDeviceSynchronize();
		if (_measureLMAP) {
			float _elapsedMs = 0.0f;
			if (_lmRecorded) {
				cudaEventElapsedTime(&_elapsedMs, _lmStart, _lmStop);
				TimeRec_HigherOrderSourceLM += (FP_PRECISION_TIME) _elapsedMs * 1.0e-3;
			}
			if (_apRecorded) {
				cudaEventElapsedTime(&_elapsedMs, _apStart, _apStop);
				TimeRec_HigherOrderSourceAP += (FP_PRECISION_TIME) _elapsedMs * 1.0e-3;
			}
			if (_auxRecorded) {
				cudaEventElapsedTime(&_elapsedMs, _auxStart, _auxStop);
				TimeRec_HigherOrderSourceAuxiliary += (FP_PRECISION_TIME) _elapsedMs * 1.0e-3;
			}
		}
		_scattStageEnd = TallyMonotonicTime_CPU();
		TimeRec_HigherOrderSource += _scattStageEnd - _scattStageStart;
	}
	if (_measureLMAP) {
		cudaEventDestroy(_lmStart);
		cudaEventDestroy(_lmStop);
		cudaEventDestroy(_apStart);
		cudaEventDestroy(_apStop);
		cudaEventDestroy(_auxStart);
		cudaEventDestroy(_auxStop);
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

//    printf("\nGPU_FISS_CALC_TIME = %e", GPU_FISS_CALC_TIME);
//    printf("\nGPU_SCAT_CALC_TIME = %e", GPU_SCAT_CALC_TIME);
//    printf("\nGPU_TOTAL_SOURCE_CALC_TIME = %e\n", GPU_TOTAL_SOURCE_CALC_TIME);

	// GPUPrintf1DArray_FP(_TotalNumFSRs_perLayer * _NumGroups, GPUMOCSource, (char *)("b4 ts GPUMOCSource"),NULL);
	// GPUPrintf1DArray_FP(_TotalNumFSRs_perLayer * _NumGroups, GPUScalarFluxTemp, (char *)("b4 ts GPUScalarFluxTemp"),LogFile);
	// GPUPrintf1DArray_FP(_TotalNumFSRs_perLayer * _NumGroups, GPUMacroXS_t, (char *)("b4 ts GPUMacroXS_t"),LogFile);
	// GPUPrintf1DArray_FP(_TotalNumFSRs_perLayer * _NumGroups, GPUMacroXS_nf, (char *)("b4 ts GPUMacroXS_nf"),LogFile);
	// GPUPrintf1DArray_FP(_TotalNumFSRs_perLayer * _NumGroups, GPUMacroXS_chi, (char *)("b4 ts GPUMacroXS_chi"),LogFile);
	// GPUPrintf1DArray_FP(MacroXS_s_NNZ_Layer[_layID], GPUMacroXS_s, (char *)("b4 ts GPUMacroXS_s"),LogFile);
	// GPUPrintf1DArray_I32(_TotalNumFSRs_perLayer * _NumGroups, GPUMacroXS_s_NZColStartID, (char *)("GPUMacroXS_s_NZColStartID"),LogFile);
	// GPUPrintf1DArray_UI64(_TotalNumFSRs_perLayer * _NumGroups + _MOCNumLayersZ, GPUMacroXS_s_NZRowStartID_Layer, (char *)("GPUMacroXS_s_NZRowStartID_Layer"),LogFile);
	// exit(11);
#ifdef TIME_REC
	cudaDeviceSynchronize();
#endif // TIME_REC
#ifdef GPU_STREAM_USED
	cudaDeviceSynchronize();
#endif // GPU_STREAM_USED
    CUDA_CheckError((char *)("GPUComputeFSRSource_Layer"));
}

extern "C"
void GPUComputeFSRSource_Layer_PerAzim(int _layID, int _TotalNumFSRs_subDom, int _TotalNumFSRs_perLayer, int _NumGroups,
                               M_UINT64 *_MacroXS_s_NZStartID_Layer, M_UINT64 *_MacroXS_s_NNZ_Layer,
                               M_INT32 _ProblemType, FP_PRECISION *_MacroXS_t, FP_PRECISION *_MacroXS_nf,
                               FP_PRECISION *_Chi, FP_PRECISION *_MacroXS_s, FP_PRECISION *_FixedSour,
                               FP_PRECISION _invKeff, M_INT32 _NumScattOrder,
                               M_INT32 _GPU_TSKernelType, M_UINT64 *_MacroXS_s_NZRowStartID_Layer,
                               M_UINT64 *_MacroXS_s_NZColStartID, FP_PRECISION_FLUX *_ScalarFluxOld,
                               int _polarBoundLowLim, int _azimBegin, int _azimEnd,
                               int _reuseAngleIndependentSource,
                               int _isTransient, FP_PRECISION *_fissSource, FP_PRECISION *_tranConstA,
                               FP_PRECISION *_tranConstB, FP_PRECISION *_tranConstC)
{
	FP_PRECISION_TIME _fissionStageStart = 0.0, _fissionStageEnd = 0.0;
	FP_PRECISION_TIME _scattStageStart = 0.0, _scattStageEnd = 0.0;
	const int _measureLMAP = (_GPU_TSKernelType == 13 || _GPU_TSKernelType == 14);
	cudaEvent_t _lmStart = NULL, _lmStop = NULL;
	cudaEvent_t _apStart = NULL, _apStop = NULL;
	cudaEvent_t _auxStart = NULL, _auxStop = NULL;
	int _lmRecorded = 0, _apRecorded = 0, _auxRecorded = 0;
	int _hasFissionStage = (_reuseAngleIndependentSource == 0);
	const int _azimID = _azimBegin;

	if (_measureLMAP) {
		cudaEventCreate(&_lmStart);
		cudaEventCreate(&_lmStop);
		cudaEventCreate(&_apStart);
		cudaEventCreate(&_apStop);
		cudaEventCreate(&_auxStart);
		cudaEventCreate(&_auxStop);
	}

	#ifdef HO_FIXED_SOURCE
	_hasFissionStage = 1;
	#endif

	(void) _TotalNumFSRs_subDom;
	(void) _MacroXS_s_NZStartID_Layer;
	(void) _MacroXS_s_NNZ_Layer;
	(void) _ProblemType;
	(void) _MacroXS_t;
	(void) _FixedSour;
	(void) _ScalarFluxOld;
	(void) _isTransient;
	(void) _fissSource;
	(void) _tranConstA;
	(void) _tranConstB;
	(void) _tranConstC;

	/* P0 does not use the high-order angular-source buffers.  Evaluate
	 * the isotropic source once for the first azimuth and reuse it for
	 * the remaining azimuths. */
	if (_NumScattOrder == 0 && _GPU_TSKernelType < 10) {
		if (!_reuseAngleIndependentSource) {
			GPUComputeFSRSource_Layer(_layID, _TotalNumFSRs_subDom,
				_TotalNumFSRs_perLayer, _NumGroups,
				_MacroXS_s_NZStartID_Layer, _MacroXS_s_NNZ_Layer,
				_ProblemType, _MacroXS_t, _MacroXS_nf, _Chi, _MacroXS_s,
				_FixedSour, _invKeff, _NumScattOrder, _GPU_TSKernelType,
				_MacroXS_s_NZRowStartID_Layer, _MacroXS_s_NZColStartID,
				_ScalarFluxOld, _polarBoundLowLim, _isTransient,
				_fissSource, _tranConstA, _tranConstB, _tranConstC);
		}
		return;
	}

	if (_hasFissionStage) {
		cudaDeviceSynchronize();
		_fissionStageStart = TallyMonotonicTime_CPU();
		if (!_reuseAngleIndependentSource) {
			GPUComputeFSRFissSourceOnDevice<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(
				GPUScalarFlux, GPUMOCSource, GPUMacroXS_nf, GPUMacroXS_chi, _invKeff);
		}
#ifdef HO_FIXED_SOURCE
		GPUOverWriteFissSourceWithGivenSource<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(
			GPUMOCSource, GPUMacroXS_nf);
#endif
		cudaDeviceSynchronize();
		_fissionStageEnd = TallyMonotonicTime_CPU();
		TimeRec_FissionSource += _fissionStageEnd - _fissionStageStart;
	}

	const int _useClusterScattSource = GPUUseFsrClusterScattSource();
	const size_t _groupWork = (size_t) _TotalNumFSRs_perLayer * (size_t) _NumGroups;
	const size_t _groupGridSize = (_groupWork - 1) / GPUBlockDim + 1;
	cudaDeviceSynchronize();
	_scattStageStart = TallyMonotonicTime_CPU();

	if (_GPU_TSKernelType == 13 || _GPU_TSKernelType == 14) {
		/* LM is angle independent; AP is evaluated only for the selected
		 * azimuth and writes the local FSR x group x polar buffer. */
		const size_t _numLM = (size_t) (_NumScattOrder + 1) *
			(size_t) (_NumScattOrder + 1);
		const size_t _lmBytes = (size_t) _TotalNumFSRs_perLayer *
			(size_t) _NumGroups * _numLM * sizeof(FP_PRECISION);
		const size_t _lmGridSize = (_groupWork * _numLM - 1) / GPUBlockDim + 1;

		if (!_reuseAngleIndependentSource) {
		if (_GPU_TSKernelType == 13) {
			if (_measureLMAP) {
				cudaEventRecord(_lmStart, 0);
			}
			if (_useClusterScattSource) {
				GPUComputeFSRScatterSourceLMOnDevice_grpOut_Cluster<<<_lmGridSize, GPUBlockDim>>>(
					GPUFSRAngScatterSourceLM, GPUScalarFlux, GPUMacroXS_s,
					GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZRowEndID_Layer,
					GPUMacroXS_s_NZColStartID);
			}
			else {
				GPUComputeFSRScatterSourceLMOnDevice_grpOut<<<_lmGridSize, GPUBlockDim>>>(
					GPUFSRAngScatterSourceLM, GPUScalarFlux, GPUMacroXS_s,
					GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID);
			}
			if (_measureLMAP) {
				cudaEventRecord(_lmStop, 0);
				_lmRecorded = 1;
			}
		}
		else {
			if (_measureLMAP) {
				cudaEventRecord(_auxStart, 0);
				_auxRecorded = 1;
			}
			cudaMemset(GPUFSRAngScatterSourceLM, 0, _lmBytes);
			if (_measureLMAP) {
				cudaEventRecord(_auxStop, 0);
				cudaEventRecord(_lmStart, 0);
			}
			if (_useClusterScattSource) {
				GPUComputeFSRScatterSourceLMOnDevice_grpIn_Cluster<<<_lmGridSize, GPUBlockDim>>>(
					GPUFSRAngScatterSourceLM, GPUScalarFlux, GPUMacroXS_s,
					GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZRowEndID_Layer,
					GPUMacroXS_s_NZColStartID);
			}
			else {
				GPUComputeFSRScatterSourceLMOnDevice_grpIn<<<_lmGridSize, GPUBlockDim>>>(
					GPUFSRAngScatterSourceLM, GPUScalarFlux, GPUMacroXS_s,
					GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID);
			}
			if (_measureLMAP) {
				cudaEventRecord(_lmStop, 0);
				_lmRecorded = 1;
			}
		}
		}

		int _apKernel = 0;
		const char *_apEnv = getenv("APKernel");
		if (_apEnv != NULL) {
			_apKernel = atoi(_apEnv);
			if (_apKernel < 0 || _apKernel > 3) {
				_apKernel = 0;
			}
		}
		const size_t _azimWork = (size_t) (_azimEnd - _azimBegin);
		const size_t _apWork = (_apKernel >= 2)
			? _groupWork * _azimWork * (size_t) NumPolarAng[0]
			: _groupWork * _azimWork;
		const size_t _apGridSize = (_apWork - 1) / GPUBlockDim + 1;
		if (_measureLMAP) {
			cudaEventRecord(_apStart, 0);
		}
		GPUComputeFSRScatterSourceAPOnDevice_PerAzim<<<_apGridSize, GPUBlockDim>>>(
			GPUFSRAngScattSour, GPUSpheHarmCoeff, GPUFSRAngScatterSourceLM,
			_polarBoundLowLim, _azimBegin, _azimEnd, _apKernel);
		if (_measureLMAP) {
			cudaEventRecord(_apStop, 0);
			_apRecorded = 1;
		}
	}
		else if (_GPU_TSKernelType == 15 || _GPU_TSKernelType == 16) {
#if defined(M_2DMOC)
			const size_t _sourceWork = (size_t) _TotalNumFSRs_perLayer *
				(size_t) _NumGroups;
			const size_t _sourceBytes = _sourceWork * (size_t) NumPolarAng[0] *
				sizeof(FP_PRECISION);
			const size_t _angleWork = _sourceWork * (size_t) NumPolarAng[0];
			const size_t _angleGridSize = (_angleWork - 1) / GPUBlockDim + 1;

			if (_GPU_TSKernelType == 15) {
				/* Incoming-group organization uses atomic accumulation. */
				cudaMemset(GPUFSRAngScattSour, 0, _sourceBytes);
			}

			if (_GPU_TSKernelType == 15) {
				const char *_type15ReferenceEnv = getenv("GPU_SCATT_TYPE15_REFERENCE");
				const int _type15Reference = _type15ReferenceEnv != NULL &&
					atoi(_type15ReferenceEnv) != 0;
				if (_type15Reference) {
					/* Diagnostic fallback: use the deterministic Type 16
					 * outgoing-group reduction while retaining Type15 dispatch. */
					GPUComputeFSRScattSourceOnDevice_FsrGrpOutAng_PerAzim<<<
						_angleGridSize, GPUBlockDim>>>(GPUFSRAngScattSour, GPUScalarFlux,
						GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
						GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
						GPUSpheHarmCoeff, _polarBoundLowLim, _azimID,
						_useClusterScattSource);
				}
				else if (_useClusterScattSource) {
					GPUComputeFSRScattSourceOnDevice_FsrGrpInAng_PerAzim<<<
						_angleGridSize, GPUBlockDim>>>(GPUFSRAngScattSour, GPUScalarFlux,
						GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
						GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
						GPUSpheHarmCoeff, _polarBoundLowLim, _azimID,
						_useClusterScattSource);
				}
				else {
					GPUComputeFSRScattSourceOnDevice_FsrGrpInAng_PerAzim<<<
						_angleGridSize, GPUBlockDim>>>(GPUFSRAngScattSour, GPUScalarFlux,
						GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
						GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
						GPUSpheHarmCoeff, _polarBoundLowLim, _azimID,
						_useClusterScattSource);
				}

				#if !defined(PAPER_TRANSPORT_ONLY)
				/*
				 * Type 15 accumulates the incoming-group contributions with
				 * FP32 atomics.  Keep the production path unchanged, but allow
				 * one controlled comparison against the deterministic outgoing-
				 * group path (Type 16) and the incoming-group path without the
				 * polar dimension in its thread index (Type 17).  This is useful
				 * for separating a source-term discrepancy from a transport-sweep
				 * discrepancy.  The check is disabled unless explicitly enabled.
				 */
				static int _type15SourceValidationDone = 0;
				const char *_validateSourceEnv = getenv("GPU_SCATT_SOURCE_VALIDATE");
				if (!_type15Reference && !_type15SourceValidationDone &&
					_validateSourceEnv != NULL &&
					atoi(_validateSourceEnv) != 0) {
					FP_PRECISION *_validationType16 = NULL;
					FP_PRECISION *_validationType17 = NULL;
					FP_PRECISION *_hostType15 = NULL;
					FP_PRECISION *_hostType16 = NULL;
					FP_PRECISION *_hostType17 = NULL;
					const cudaError_t _alloc16 = cudaMalloc(
						(void **) &_validationType16, _sourceBytes);
					const cudaError_t _alloc17 = cudaMalloc(
						(void **) &_validationType17, _sourceBytes);
					if (_alloc16 == cudaSuccess && _alloc17 == cudaSuccess) {
						cudaMemset(_validationType17, 0, _sourceBytes);
						GPUComputeFSRScattSourceOnDevice_FsrGrpOutAng_PerAzim<<<
							_angleGridSize, GPUBlockDim>>>(_validationType16, GPUScalarFlux,
							GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
							GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
							GPUSpheHarmCoeff, _polarBoundLowLim, _azimID,
							_useClusterScattSource);
						GPUComputeFSRScattSourceOnDevice_FsrGrpIn_PerAzim<<<
							_groupGridSize, GPUBlockDim>>>(_validationType17, GPUScalarFlux,
							GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
							GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
							GPUSpheHarmCoeff, _polarBoundLowLim, _azimID,
							_useClusterScattSource);
						cudaDeviceSynchronize();

						_hostType15 = (FP_PRECISION *) malloc(_sourceBytes);
						_hostType16 = (FP_PRECISION *) malloc(_sourceBytes);
						_hostType17 = (FP_PRECISION *) malloc(_sourceBytes);
						if (_hostType15 != NULL && _hostType16 != NULL &&
							_hostType17 != NULL) {
							cudaMemcpy(_hostType15, GPUFSRAngScattSour, _sourceBytes,
								cudaMemcpyDeviceToHost);
							cudaMemcpy(_hostType16, _validationType16, _sourceBytes,
								cudaMemcpyDeviceToHost);
							cudaMemcpy(_hostType17, _validationType17, _sourceBytes,
								cudaMemcpyDeviceToHost);

							const size_t _sourceCount = _sourceWork *
								(size_t) NumPolarAng[0];
							FP_PRECISION _maxAbs16 = 0.0;
							FP_PRECISION _maxRel16 = 0.0;
							FP_PRECISION _maxAbs17 = 0.0;
							FP_PRECISION _maxRel17 = 0.0;
							size_t _maxAbs16ID = 0;
							size_t _maxRel16ID = 0;
							size_t _maxAbs17ID = 0;
							size_t _maxRel17ID = 0;
							for (size_t _id = 0; _id < _sourceCount; ++_id) {
								const FP_PRECISION _diff16 = fabs(
									_hostType15[_id] - _hostType16[_id]);
								const FP_PRECISION _diff17 = fabs(
									_hostType15[_id] - _hostType17[_id]);
								const FP_PRECISION _rel16 = _diff16 /
									(fabs(_hostType16[_id]) + (FP_PRECISION) 1.0e-30);
								const FP_PRECISION _rel17 = _diff17 /
									(fabs(_hostType17[_id]) + (FP_PRECISION) 1.0e-30);
								if (_diff16 > _maxAbs16) {
									_maxAbs16 = _diff16;
									_maxAbs16ID = _id;
								}
								if (_rel16 > _maxRel16) {
									_maxRel16 = _rel16;
									_maxRel16ID = _id;
								}
								if (_diff17 > _maxAbs17) {
									_maxAbs17 = _diff17;
									_maxAbs17ID = _id;
								}
								if (_rel17 > _maxRel17) {
									_maxRel17 = _rel17;
									_maxRel17ID = _id;
								}
							}

							const size_t _numPol = (size_t) NumPolarAng[0];
							const size_t _numGrp = (size_t) _NumGroups;
							printf("\n[SCATT-VALIDATE] layer=%d azim=%d\n",
								_layID, _azimID);
							printf("[SCATT-VALIDATE] Type15 vs Type16: max_abs=%e "
								"(id=%zu, v15=%e, v16=%e), max_rel=%e "
								"(id=%zu)\n", (double) _maxAbs16, _maxAbs16ID,
								(double) _hostType15[_maxAbs16ID],
								(double) _hostType16[_maxAbs16ID], (double) _maxRel16,
								_maxRel16ID);
							printf("[SCATT-VALIDATE] Type15 vs Type17: max_abs=%e "
								"(id=%zu, v15=%e, v17=%e), max_rel=%e "
								"(id=%zu)\n", (double) _maxAbs17, _maxAbs17ID,
								(double) _hostType15[_maxAbs17ID],
								(double) _hostType17[_maxAbs17ID], (double) _maxRel17,
								_maxRel17ID);
							printf("[SCATT-VALIDATE] layout: maxAbs16=(fsr=%zu, "
								"group=%zu, polar=%zu), maxRel16=(fsr=%zu, group=%zu, "
								"polar=%zu)\n", _maxAbs16ID / (_numGrp * _numPol),
								(_maxAbs16ID / _numPol) % _numGrp,
								_maxAbs16ID % _numPol, _maxRel16ID / (_numGrp * _numPol),
								(_maxRel16ID / _numPol) % _numGrp,
								_maxRel16ID % _numPol);
						}
						_type15SourceValidationDone = 1;
					}
					else {
						printf("\n[SCATT-VALIDATE] cudaMalloc failed for reference "
							"source buffers: %s / %s\n", cudaGetErrorString(_alloc16),
							cudaGetErrorString(_alloc17));
					}
					if (_hostType15 != NULL) free(_hostType15);
					if (_hostType16 != NULL) free(_hostType16);
					if (_hostType17 != NULL) free(_hostType17);
					if (_validationType16 != NULL) cudaFree(_validationType16);
					if (_validationType17 != NULL) cudaFree(_validationType17);
				}
			#endif
			}
			else if (_GPU_TSKernelType == 16) {
				GPUComputeFSRScattSourceOnDevice_FsrGrpOutAng_PerAzim<<<
					_angleGridSize, GPUBlockDim>>>(GPUFSRAngScattSour, GPUScalarFlux,
					GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
					GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
					GPUSpheHarmCoeff, _polarBoundLowLim, _azimID,
					_useClusterScattSource);
			}
			#if !defined(PAPER_TRANSPORT_ONLY)
			else if (_GPU_TSKernelType == 17) {
				GPUComputeFSRScattSourceOnDevice_FsrGrpIn_PerAzim<<<
					_groupGridSize, GPUBlockDim>>>(GPUFSRAngScattSour, GPUScalarFlux,
					GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
					GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
					GPUSpheHarmCoeff, _polarBoundLowLim, _azimID,
					_useClusterScattSource);
			}
			else if (_GPU_TSKernelType == 18) {
				GPUComputeFSRScattSourceOnDevice_FsrGrpOut_PerAzim<<<
					_groupGridSize, GPUBlockDim>>>(GPUFSRAngScattSour, GPUScalarFlux,
					GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
					GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
					GPUSpheHarmCoeff, _polarBoundLowLim, _azimID,
					_useClusterScattSource);
			}
			else if (_GPU_TSKernelType == 19) {
				switch (_NumScattOrder) {
				case 0:
					GPUComputeFSRScattSourceOnDevice_FsrGrpOutLocalLM_P0_PerAzim<<<
						_groupGridSize, GPUBlockDim>>>(GPUFSRAngScattSour, GPUScalarFlux,
						GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
						GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
						GPUSpheHarmCoeff, _polarBoundLowLim, _azimID,
						_useClusterScattSource);
					break;
				case 1:
					GPUComputeFSRScattSourceOnDevice_FsrGrpOutLocalLM_P1_PerAzim<<<
						_groupGridSize, GPUBlockDim>>>(GPUFSRAngScattSour, GPUScalarFlux,
						GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
						GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
						GPUSpheHarmCoeff, _polarBoundLowLim, _azimID,
						_useClusterScattSource);
					break;
				case 2:
					GPUComputeFSRScattSourceOnDevice_FsrGrpOutLocalLM_P2_PerAzim<<<
						_groupGridSize, GPUBlockDim>>>(GPUFSRAngScattSour, GPUScalarFlux,
						GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
						GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
						GPUSpheHarmCoeff, _polarBoundLowLim, _azimID,
						_useClusterScattSource);
					break;
				case 3:
					GPUComputeFSRScattSourceOnDevice_FsrGrpOutLocalLM_P3_PerAzim<<<
						_groupGridSize, GPUBlockDim>>>(GPUFSRAngScattSour, GPUScalarFlux,
						GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
						GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
						GPUSpheHarmCoeff, _polarBoundLowLim, _azimID,
						_useClusterScattSource);
					break;
				default:
					GPUComputeFSRScattSourceOnDevice_FsrGrpOutLocalLM_P4_PerAzim<<<
						_groupGridSize, GPUBlockDim>>>(GPUFSRAngScattSour, GPUScalarFlux,
						GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
						GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
						GPUSpheHarmCoeff, _polarBoundLowLim, _azimID,
						_useClusterScattSource);
					break;
				}
			}
			else if (_GPU_TSKernelType == 20) {
				cudaMemset(GPUFSRAngScattSour, 0, _sourceBytes);
				GPUComputeFSRScattSourceOnDevice_FsrGrpInLocalLM_PerAzim<<<
					_groupGridSize, GPUBlockDim>>>(GPUFSRAngScattSour, GPUScalarFlux,
					GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer,
					GPUMacroXS_s_NZRowEndID_Layer, GPUMacroXS_s_NZColStartID,
					GPUSpheHarmCoeff, _polarBoundLowLim, _azimID,
					_useClusterScattSource);
			}
			else {
				if (GPUBlockDim < 32 || GPUBlockDim % 32 != 0) {
					printf("\n[ERROR] GPUBlockDim must be a positive multiple of 32 for Type 21.\n");
					exit(70);
				}
				const size_t _warpsPerBlock = (size_t) GPUBlockDim / 32u;
				const size_t _warpGridSize = (_sourceWork - 1) / _warpsPerBlock + 1;
				const size_t _numLM = (size_t) (_NumScattOrder + 1) *
					(size_t) (_NumScattOrder + 1);
				const size_t _warpSharedBytes = _warpsPerBlock * _numLM *
					sizeof(FP_PRECISION);
				GPUComputeFSRScattSourceOnDevice_FsrGrpOutWarpLM_PerAzim<<<
					_warpGridSize, GPUBlockDim, _warpSharedBytes>>>(
					GPUFSRAngScattSour, GPUScalarFlux, GPUMacroXS_s,
					GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID,
					GPUSpheHarmCoeff, _polarBoundLowLim, _azimID,
					_useClusterScattSource, GPUMacroXS_s_NZRowEndID_Layer);
			}
			#endif
#else
			/* Keep the existing 3D path unchanged. */
			GPUInitFSRHOSourceOnDevice<<<GPUGridDim_source_para_Layer, GPUBlockDim>>>(GPUFSRAngScattSour, \
				GPUScalarFlux, GPUMacroXS_s, GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID, \
				GPUSpheHarmCoeff, _polarBoundLowLim);
			GPUComputeFSRScattSourceOnDevice_PerAzim<<<_groupGridSize, GPUBlockDim>>>(
				GPUFSRAngScattSour, GPUScalarFlux, GPUMacroXS_s,
				GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZRowEndID_Layer,
				GPUMacroXS_s_NZColStartID, GPUSpheHarmCoeff, _polarBoundLowLim,
				_azimID, _GPU_TSKernelType == 20 ? 2 : 1, _useClusterScattSource);
#endif
		}
	else {
		printf("\n[ERROR] GPU_AzimSweepMode=1 supports P0 GPU_TSKernelType < 10 or high-order GPU_TSKernelType 13-16.\n");
		exit(72);
	}

	cudaDeviceSynchronize();
	if (_measureLMAP) {
		float _elapsedMs = 0.0f;
		if (_lmRecorded) {
			cudaEventElapsedTime(&_elapsedMs, _lmStart, _lmStop);
			TimeRec_HigherOrderSourceLM += (FP_PRECISION_TIME) _elapsedMs * 1.0e-3;
		}
		if (_apRecorded) {
			cudaEventElapsedTime(&_elapsedMs, _apStart, _apStop);
			TimeRec_HigherOrderSourceAP += (FP_PRECISION_TIME) _elapsedMs * 1.0e-3;
		}
		if (_auxRecorded) {
			cudaEventElapsedTime(&_elapsedMs, _auxStart, _auxStop);
			TimeRec_HigherOrderSourceAuxiliary += (FP_PRECISION_TIME) _elapsedMs * 1.0e-3;
		}
		cudaEventDestroy(_lmStart);
		cudaEventDestroy(_lmStop);
		cudaEventDestroy(_apStart);
		cudaEventDestroy(_apStop);
		cudaEventDestroy(_auxStart);
		cudaEventDestroy(_auxStop);
	}
	_scattStageEnd = TallyMonotonicTime_CPU();
	TimeRec_HigherOrderSource += _scattStageEnd - _scattStageStart;
	CUDA_CheckError((char *) ("GPUComputeFSRSource_Layer_PerAzim"));
}

extern "C"
extern "C"
/**
 * @ Function: GPUInitScalarFlux()
 *   introduction: initialize the scalar flux with _value
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2019/12/24, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
void GPUInitScalarFlux(M_INT32 _dim, FP_PRECISION _value)
{
	GPUVectorInit_1DFP_Flux<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(GPUScalarFlux, _dim,_value);
	CUDA_CheckError((char *)("GPUInitScalarFlux"));
}



/**
 * @ Function: GPUTransportSweep()
 *   introduction: excute the transport sweep on device
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2018/06/05, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
void GPUTransportSweep(int _lay, FP_PRECISION _twoOverLayWidth, int _polarBoundLowLim, int _polarBoundUpLim, \
	FP_PRECISION *_GPUFSRAveAngFlux_inp,FP_PRECISION *_GPUFSRAveAngFlux_update,int _polOffset, \
	M_INT32 _GPU_TSKernelType, M_INT32 _NumPinsInSubDomPerLayer, M_INT32 _NumPolarAng, \
	int _azimBegin, int _azimEnd)
{
	int shared_mem, _numPol/*, share_int_mem*/;
	FP_PRECISION_TIME _transportSweepStart;
	int _hoSweepKernelVariant = 0;
#ifdef REC_TIME_SOLVER
	FP_PRECISION_TIME t_start, t_end;
#endif // REC_TIME_SOLVER

#if defined(M_2DMOC)
    _numPol = _NumPolarAng;
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
    _numPol = _NumPolarAng /2;
#endif // defined(M_3DMOCex)

	// if (_GPU_TSKernelType == 1) {
	shared_mem = GPUBlockDim * _numPol * sizeof(FP_PRECISION);
		// shared_mem = GPUBlockDim * _numPol * 4;
	// }


#ifdef CUDA_CLOCK
	cudaEvent_t _ts_gpu, _te_gpu;
	float _tGPUTS;
	cudaEventCreate(&_ts_gpu);
	cudaEventCreate(&_te_gpu);
	cudaEventRecord(_ts_gpu, 0);
#endif // CUDA_CLOCK
// printf("\n _GPU_TSKernelType=%d",_GPU_TSKernelType);
	/* Synchronize before starting the phase timer so that this phase contains
	 * only the characteristic-line transport kernels launched below. */
	cudaDeviceSynchronize();
	_transportSweepStart = TallyMonotonicTime_CPU();
		//cudaFuncSetCacheConfig(GPUTransportSweepOnDevice_group_diam_oneDir_AngFReg, cudaFuncCachePreferL1);
	if (_GPU_TSKernelType == 0) {

//#ifdef REC_TIME_SOLVER
//		cudaDeviceSynchronize();
//		t_start = TIME_RECORD;
//#endif // REC_TIME_SOLVER

#if defined(M_MG_CMFD)
        GPUTS_Group_ExpInte_OneDir_AngFSmem<<<GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_lay, GPUScalarFlux, \
	        GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, GPULTsIndex_LTID_Total, \
	        GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, _NumPinsInSubDomPerLayer, \
	        GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, GPUSumNumSegsInRTMTrack_2PI, \
	        GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, GPUNumLTsSum, GPUNumLongTracks, \
	        GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, GPUOneOverFSRsAreaInRTM_AntiClockFSR,\
	        GPULinearExpTable, GPUMOCParCurr_X_Pos, GPUMOCParCurr_X_Neg, GPUMOCParCurr_Y_Pos, GPUMOCParCurr_Y_Neg, GPUNumTracksX, \
	        GPUNumTracksY, GPURTMWidth_X, GPURTMWidth_Y,_polarBoundLowLim,_polarBoundUpLim,_twoOverLayWidth,\
			_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC,GPUMacroXS_t_wScale);
#else
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(NumPolarAng[0]/2), _GPUFSRAveAngFlux_inp, "B4 inp");
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(NumPolarAng[0]/2), _GPUFSRAveAngFlux_update, "B4 update");
// printf("\n lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*(NumScattOrder+1)*(NumScattOrder+1), GPUScalarFlux, (char *)("B4 TS GPUScalarFlux"));
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups, GPUMOCSource, (char *)("0000 B4 TS GPUMOCSource"));
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(_polarBoundUpLim-_polarBoundLowLim), _GPUFSRAveAngFlux_inp, "_GPUFSRAveAngFlux_inp");

        GPUTS_Group_ExpInte_OneDir_AngFSmem<<<GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_lay, GPUScalarFlux, 
        // GPUTS_Group_ExpInte_OneDir_AngFSmem<<<100, 1, shared_mem>>>(_lay, GPUScalarFlux, 
        	GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, GPULTsIndex_LTID_Total, \
	        GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, _NumPinsInSubDomPerLayer, \
        	GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, GPUSumNumSegsInRTMTrack_2PI, \
        	GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, GPUNumLTsSum, GPUNumLongTracks, \
        	GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, GPUOneOverFSRsAreaInRTM_AntiClockFSR,\
        	GPULinearExpTable, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,_polarBoundLowLim,_polarBoundUpLim,\
			_twoOverLayWidth,_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC,GPUMacroXS_t_wScale);
// printf("\n lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*(NumScattOrder+1)*(NumScattOrder+1), GPUScalarFlux, (char *)("after TS GPUScalarFlux"));
// PrintFPArrayOnDevice(10, GPUScalarFlux, (char *)("0000 GPUScalarFlux"));
// CUDA_CheckError((char *)("0000 GPUScalarFlux"));
// printf("\n lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(_polarBoundUpLim-_polarBoundLowLim), _GPUFSRAveAngFlux_update, "0000 update");
// CUDA_CheckError((char *)("after TS _GPUFSRAveAngFlux_update"));
//
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(NumPolarAng[0]/2), _GPUFSRAveAngFlux_inp, "after inp");
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(NumPolarAng[0]/2), _GPUFSRAveAngFlux_update, "after update");
#endif

//#ifdef REC_TIME_SOLVER
//		cudaDeviceSynchronize();
//		t_end = TIME_RECORD;
//		RecTime_RayWisedSolver += (t_end - t_start) / (FP_PRECISION)(CLOCKS_PER_SEC);
//#endif // REC_TIME_SOLVER
	}
	else if (_GPU_TSKernelType == 10) {
/// otf calc anisotropic source
#if defined(M_MG_CMFD)
        GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt<<<GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_lay, GPUScalarFlux, \
	        GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, GPULTsIndex_LTID_Total, \
	        GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, _NumPinsInSubDomPerLayer, \
	        GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, GPUSumNumSegsInRTMTrack_2PI, \
	        GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, GPUNumLTsSum, GPUNumLongTracks, \
	        GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, GPUOneOverFSRsAreaInRTM_AntiClockFSR,\
	        GPULinearExpTable, GPUMOCParCurr_X_Pos, GPUMOCParCurr_X_Neg, GPUMOCParCurr_Y_Pos, GPUMOCParCurr_Y_Neg, GPUNumTracksX, \
	        GPUNumTracksY, GPURTMWidth_X, GPURTMWidth_Y,_polarBoundLowLim,_polarBoundUpLim,_twoOverLayWidth,\
			_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC,GPUMacroXS_t_wScale,\
			GPUMacroXS_s,GPUMacroXS_s_NZRowStartID_Layer,GPUMacroXS_s_NZColStartID, GPUSpheHarmCoeff);
#else
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups, GPUScalarFlux, (char *)("GPUScalarFlux"));
// printf("\n lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups, GPUMOCSource, (char *)("B4 TS GPUMOCSource"));
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(_polarBoundUpLim-_polarBoundLowLim), _GPUFSRAveAngFlux_inp, "_GPUFSRAveAngFlux_inp");
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(_polarBoundUpLim-_polarBoundLowLim), _GPUFSRAveAngFlux_update, "before update");

        GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt<<<GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_lay, GPUScalarFlux, \
        	GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, GPULTsIndex_LTID_Total, \
	        GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, _NumPinsInSubDomPerLayer, \
        	GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, GPUSumNumSegsInRTMTrack_2PI, \
        	GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, GPUNumLTsSum, GPUNumLongTracks, \
        	GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, GPUOneOverFSRsAreaInRTM_AntiClockFSR,\
        	GPULinearExpTable, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,_polarBoundLowLim,_polarBoundUpLim,\
			_twoOverLayWidth,_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC,GPUMacroXS_t_wScale,\
			GPUMacroXS_s,GPUMacroXS_s_NZRowStartID_Layer,GPUMacroXS_s_NZColStartID, GPUSpheHarmCoeff);
// printf("\n lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);

// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(_polarBoundUpLim-_polarBoundLowLim), _GPUFSRAveAngFlux_update, "after update");
// CUDA_CheckError((char *)("_GPUFSRAveAngFlux_update"));
// printf("\n lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);

// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups, GPUScalarFlux, (char *)("A4 GPUScalarFlux"));

#endif
	}
	else if (_GPU_TSKernelType == 11 || _GPU_TSKernelType == 13 || _GPU_TSKernelType == 14 ||
	         (_GPU_TSKernelType >= 15 && _GPU_TSKernelType <= 18)
#if defined(M_2DMOC)
	         || (_GPU_TSKernelType >= 19 && _GPU_TSKernelType <= 21)
#endif
	) {
/// pre calc anisotropic source
		/* Types 13/14 reuse the type-11 sweep: the LM/AP source kernels fill
		 * GPUFSRAngScattSour with the same pure-scattering angular source
		 * (no fission, no 1/Sigma_t) that the type-11 sweep expects. */
			#if defined(M_MG_CMFD)
					GPUKernelTimeStart(&_timeKernelHOSweep1Start, &_timeKernelHOSweep1Stop);
					_hoSweepKernelVariant = 1;
					GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt_1<<<GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_lay, GPUScalarFlux, \
					GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, GPULTsIndex_LTID_Total, \
					GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, _NumPinsInSubDomPerLayer, \
					GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, GPUSumNumSegsInRTMTrack_2PI, \
					GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, GPUNumLTsSum, GPUNumLongTracks, \
					GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, GPUOneOverFSRsAreaInRTM_AntiClockFSR,\
					GPULinearExpTable, GPUMOCParCurr_X_Pos, GPUMOCParCurr_X_Neg, GPUMOCParCurr_Y_Pos, GPUMOCParCurr_Y_Neg, GPUNumTracksX, \
					GPUNumTracksY, GPURTMWidth_X, GPURTMWidth_Y,_polarBoundLowLim,_polarBoundUpLim,_twoOverLayWidth,\
					_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC,GPUMacroXS_t_wScale, GPUFSRAngScattSour);
		#else
		// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups, GPUScalarFlux, (char *)("GPUScalarFlux"));
		// printf("\n lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);
		// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups, GPUMOCSource, (char *)("B4 TS GPUMOCSource"));
		// PrintFPArrayOnkDevice(10, _GPUFSRAveAngFlux_inp, "_GPUFSRAveAngFlux_inp");
		// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(_polarBoundUpLim-_polarBoundLowLim), _GPUFSRAveAngFlux_update, "before update");
		// int _pol;
		// #if defined(M_2DMOC)
		// _pol=NumPolarAng[0];
		// #else
		// _pol=NumPolarAng[0]/2;
		// #endif
		// PrintFPArrayOnDevice(TotalNumFSRs_perLayer * NumGroups * NumAzimAng[0] * _pol, GPUFSRAngScattSour, (char *)("b4 ts GPUFSRAngScattSour"));

					GPUKernelTimeStart(&_timeKernelHOSweep11Start, &_timeKernelHOSweep11Stop);
					_hoSweepKernelVariant = 11;
					GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt_11<<<GPUGridDim_polar_oneDir, GPUBlockDim>>>(_lay, GPUScalarFlux, \
					GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, GPULTsIndex_LTID_Total, \
					GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, _NumPinsInSubDomPerLayer, \
					GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, GPUSumNumSegsInRTMTrack_2PI, \
					GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, GPUNumLTsSum, GPUNumLongTracks, \
					GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, GPUOneOverFSRsAreaInRTM_AntiClockFSR,\
					GPULinearExpTable, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,_polarBoundLowLim,_polarBoundUpLim,\
					_twoOverLayWidth,_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC,GPUMacroXS_t_wScale, \
					GPUFSRAngScattSour, _azimBegin, _azimEnd);
    CUDA_CheckError((char *)("GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt_1"));

// printf("\n lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);
// PrintFPArrayOnDevice(20, _GPUFSRAveAngFlux_update, "11 update");
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(_polarBoundUpLim-_polarBoundLowLim), _GPUFSRAveAngFlux_update, "after update");
		// CUDA_CheckError((char *)("_GPUFSRAveAngFlux_update"));
		// printf("\n lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);
		
		// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups, GPUScalarFlux, (char *)("A4 GPUScalarFlux"));
		
		#endif
	}
	else if (_GPU_TSKernelType == 12) {

		#if defined(M_MG_CMFD)
				GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt_2<<<GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_lay, GPUScalarFlux, \
					GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, GPULTsIndex_LTID_Total, \
					GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, _NumPinsInSubDomPerLayer, \
					GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, GPUSumNumSegsInRTMTrack_2PI, \
					GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, GPUMacroXS_t_inv, GPUNumLTsSum, GPUNumLongTracks, \
					GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, GPUOneOverFSRsAreaInRTM_AntiClockFSR,\
					GPULinearExpTable, GPUMOCParCurr_X_Pos, GPUMOCParCurr_X_Neg, GPUMOCParCurr_Y_Pos, GPUMOCParCurr_Y_Neg, GPUNumTracksX, \
					GPUNumTracksY, GPURTMWidth_X, GPURTMWidth_Y,_polarBoundLowLim,_polarBoundUpLim,_twoOverLayWidth,\
					_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC,GPUMacroXS_t_wScale, GPUFSRAngScattSour);
		#else
				// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups, GPUScalarFlux, (char *)("GPUScalarFlux"));
		// printf("\n lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);
		// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups, GPUMOCSource, (char *)("B4 TS GPUMOCSource"));
		// // PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(_polarBoundUpLim-_polarBoundLowLim), _GPUFSRAveAngFlux_inp, "_GPUFSRAveAngFlux_inp");
		// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(_polarBoundUpLim-_polarBoundLowLim), _GPUFSRAveAngFlux_update, "before update");
		// int _pol;
		// #if defined(M_2DMOC)
		// _pol=NumPolarAng[0];
		// #else
		// _pol=NumPolarAng[0]/2;
		// #endif
		// PrintFPArrayOnDevice(TotalNumFSRs_perLayer * NumGroups * NumAzimAng[0] * _pol, GPUFSRAngScattSour, (char *)("b4 ts GPUFSRAngScattSour"));

				GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt_2<<<GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_lay, GPUScalarFlux, \
					GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, GPULTsIndex_LTID_Total, \
					GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, _NumPinsInSubDomPerLayer, \
					GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, GPUSumNumSegsInRTMTrack_2PI, \
					GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, GPUMacroXS_t_inv, GPUNumLTsSum, GPUNumLongTracks, \
					GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, GPUOneOverFSRsAreaInRTM_AntiClockFSR,\
					GPULinearExpTable, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,_polarBoundLowLim,_polarBoundUpLim,\
					_twoOverLayWidth,_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC,GPUMacroXS_t_wScale, \
					GPUFSRAngScattSour);
	CUDA_CheckError((char *)("GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt_2"));
// printf("\n lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);
// PrintFPArrayOnDevice(20, _GPUFSRAveAngFlux_update, "12 update");
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(_polarBoundUpLim-_polarBoundLowLim), _GPUFSRAveAngFlux_update, "after update");
		// CUDA_CheckError((char *)("_GPUFSRAveAngFlux_update"));
		// printf("\n lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);
		
		// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups, GPUScalarFlux, (char *)("A4 GPUScalarFlux"));
		

		#endif
	}
	else if (_GPU_TSKernelType == 1) {

//#ifdef REC_TIME_SOLVER
//		cudaDeviceSynchronize();
//		t_start = TIME_RECORD;
//#endif // REC_TIME_SOLVER
// GPUPrintf1DArray_FP(100, GPUBoundaryFlux, "before GPUTS GPUBoundaryFlux", LogFile);
#if defined(M_MG_CMFD)
        GPUTS_Group_ExpFunc_OneDir_AngFSmem<<<GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_lay, GPUScalarFlux, \
	        GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, GPULTsIndex_LTID_Total, \
	        GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, _NumPinsInSubDomPerLayer, \
	        GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, GPUSumNumSegsInRTMTrack_2PI, \
	        GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, GPUNumLTsSum, GPUNumLongTracks, \
	        GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, GPUOneOverFSRsAreaInRTM_AntiClockFSR,\
	        GPUMOCParCurr_X_Pos, GPUMOCParCurr_X_Neg, GPUMOCParCurr_Y_Pos, GPUMOCParCurr_Y_Neg, GPUNumTracksX, \
	        GPUNumTracksY, GPURTMWidth_X, GPURTMWidth_Y,_polarBoundLowLim,_polarBoundUpLim,_twoOverLayWidth,\
			_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC,GPUMacroXS_t_wScale,_azimBegin,_azimEnd);
#else
        GPUTS_Group_ExpFunc_OneDir_AngFSmem<<<GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_lay, GPUScalarFlux, \
        	GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, GPULTsIndex_LTID_Total, \
	        GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, _NumPinsInSubDomPerLayer, \
        	GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, GPUSumNumSegsInRTMTrack_2PI, \
        	GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, GPUNumLTsSum, GPUNumLongTracks, \
        	GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, GPUOneOverFSRsAreaInRTM_AntiClockFSR,\
        	NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,_polarBoundLowLim,_polarBoundUpLim,\
			_twoOverLayWidth,_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC,GPUMacroXS_t_wScale,_azimBegin,_azimEnd);
#endif

//#ifdef REC_TIME_SOLVER
//		cudaDeviceSynchronize();
//		t_end = TIME_RECORD;
//		RecTime_RayWisedSolver += (t_end - t_start) / (FP_PRECISION)(CLOCKS_PER_SEC);
//#endif // REC_TIME_SOLVER
	}
	else if (_GPU_TSKernelType == 2) {
#if defined(M_MG_CMFD)
		GPUTS_Group_DDExpInte_OneDir_AngFSmem<<<GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_lay, GPUScalarFlux, \
			GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, GPULTsIndex_LTID_Total, \
			GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, _NumPinsInSubDomPerLayer, \
			GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, GPUSumNumSegsInRTMTrack_2PI, \
			GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, GPUNumLTsSum, GPUNumLongTracks, \
			GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, GPUOneOverFSRsAreaInRTM_AntiClockFSR,\
			GPULinearExpTable, GPUMOCParCurr_X_Pos, GPUMOCParCurr_X_Neg, GPUMOCParCurr_Y_Pos, GPUMOCParCurr_Y_Neg, GPUNumTracksX, \
			GPUNumTracksY, GPURTMWidth_X, GPURTMWidth_Y,_polarBoundLowLim,_polarBoundUpLim,_twoOverLayWidth,\
			_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC);
#else
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(NumPolarAng[0]/2), _GPUFSRAveAngFlux_inp, "B4 inp");
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(NumPolarAng[0]/2), _GPUFSRAveAngFlux_update, "B4 update");
        GPUTS_Group_DDExpInte_OneDir_AngFSmem<<<GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_lay, GPUScalarFlux, \
			GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, GPULTsIndex_LTID_Total, \
			GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, _NumPinsInSubDomPerLayer, \
			GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, GPUSumNumSegsInRTMTrack_2PI, \
			GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, GPUNumLTsSum, GPUNumLongTracks, \
			GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, GPUOneOverFSRsAreaInRTM_AntiClockFSR,\
			GPULinearExpTable, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,_polarBoundLowLim,_polarBoundUpLim,\
			_twoOverLayWidth,_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC);
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(NumPolarAng[0]/2), _GPUFSRAveAngFlux_inp, "after inp");
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(NumPolarAng[0]/2), _GPUFSRAveAngFlux_update, "after update");

#endif
	}
	else if (_GPU_TSKernelType == 3) {
		#if defined(M_MG_CMFD)
				GPUTS_Group_DDExpFunc_OneDir_AngFSmem<<<GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_lay, GPUScalarFlux, \
					GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, GPULTsIndex_LTID_Total, \
					GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, _NumPinsInSubDomPerLayer, \
					GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, GPUSumNumSegsInRTMTrack_2PI, \
					GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, GPUNumLTsSum, GPUNumLongTracks, \
					GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, GPUOneOverFSRsAreaInRTM_AntiClockFSR,\
					GPUMOCParCurr_X_Pos, GPUMOCParCurr_X_Neg, GPUMOCParCurr_Y_Pos, GPUMOCParCurr_Y_Neg, GPUNumTracksX, \
					GPUNumTracksY, GPURTMWidth_X, GPURTMWidth_Y,_polarBoundLowLim,_polarBoundUpLim,_twoOverLayWidth,\
					_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC);
		#else
		// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(NumPolarAng[0]/2), _GPUFSRAveAngFlux_inp, "B4 inp");
		// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(NumPolarAng[0]/2), _GPUFSRAveAngFlux_update, "B4 update");
				GPUTS_Group_DDExpFunc_OneDir_AngFSmem<<<GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_lay, GPUScalarFlux, \
					GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, GPULTsIndex_LTID_Total, \
					GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, _NumPinsInSubDomPerLayer, \
					GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, GPUSumNumSegsInRTMTrack_2PI, \
					GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, GPUNumLTsSum, GPUNumLongTracks, \
					GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, GPUOneOverFSRsAreaInRTM_AntiClockFSR,\
					NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,_polarBoundLowLim,_polarBoundUpLim,\
					_twoOverLayWidth,_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC);
		// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(NumPolarAng[0]/2), _GPUFSRAveAngFlux_inp, "after inp");
		// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(NumPolarAng[0]/2), _GPUFSRAveAngFlux_update, "after update");
		
		#endif
			}
	else {}

	/* Stop before GPUAddSourceToScalarFlux().  That operation belongs to the
	 * scalar-flux/source assembly, not to the pure characteristic-line sweep. */
	if (_hoSweepKernelVariant == 1) {
		TimeRec_HOTransportKernel1 +=
			GPUKernelTimeStop(&_timeKernelHOSweep1Start, &_timeKernelHOSweep1Stop);
		TimeRec_HOTransportKernel1Calls++;
	}
	else if (_hoSweepKernelVariant == 11) {
		TimeRec_HOTransportKernel11 +=
			GPUKernelTimeStop(&_timeKernelHOSweep11Start, &_timeKernelHOSweep11Stop);
		TimeRec_HOTransportKernel11Calls++;
	}
	else {
		cudaDeviceSynchronize();
	}
	TimeRec_TransportSweep += TallyMonotonicTime_CPU() - _transportSweepStart;
	#ifdef CUDA_CLOCK
	//cudaEventSynchronize(_te_gpu);
	cudaEventRecord(_te_gpu, 0);
	cudaEventSynchronize(_te_gpu);
	cudaEventElapsedTime(&_tGPUTS, _ts_gpu, _te_gpu);
	TSKernel_Time += _tGPUTS;
	//printf("\n _tGPUTS=%.3e",_tGPUTS*0.001);
	cudaEventDestroy(_ts_gpu);
	cudaEventDestroy(_te_gpu);
#endif // CUDA_CLOCK

    CUDA_CheckError((char *)("GPUTransportSweep"));
	/* add source */
#if defined(M_2DMOC)
	if (_GPU_TSKernelType < 10 &&
		(_azimBegin < 0 || _azimEnd == NumAzimAng[0])) {
		GPUAddSourceToScalarFlux();
// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups, GPUScalarFlux, (char *)("add GPUScalarFlux"));

	}
#endif // defined(M_2DMOC)

#ifndef CPUMPI_GPUTS_ASYNCH
    cudaDeviceSynchronize();
#endif // !CPUMPI_GPUTS_ASYNCH
#ifdef TIME_REC
	cudaDeviceSynchronize();
#endif // TIME_REC
}

void GPUMOC_TransSweep_FixSourProb_lay_g(FP_PRECISION _twoOverLayWidth, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim, FP_PRECISION *_GPUFSRAveAngFlux_inp, \
	FP_PRECISION *_GPUFSRAveAngFlux_update, int _polOffset, int _layBoundLowLim, int _layBoundUpLim,\
	int _numFSRsPerLay, M_INT32 _NumPinsInSubDomPerLayer, M_INT32 _NumLTs_HS, M_INT32 _NumPolarAng,\
	M_INT32 _GPU_FixSourSolver)
{
	int shared_mem, _numPol/*, share_int_mem*/;
	int _GPUGridDim_group_oneDir=1;
#if defined(M_2DMOC)
    _numPol = _NumPolarAng;
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
    _numPol = _NumPolarAng/2;
#endif // defined(M_3DMOCex)

	shared_mem = GPUBlockDim * _numPol * sizeof(FP_PRECISION);
	_GPUGridDim_group_oneDir= (_NumLTs_HS * (_groupBoundUpLim-_groupBoundLowLim) * 2 - 1) / GPUBlockDim + 1;

	if (_GPU_FixSourSolver == 0) {
		GPUTS_Group_ExpInte_OneDir_AngFSmem_FixSourProb_2D<<<_GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_layBoundLowLim, \
			GPUScalarFlux, GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, \
			GPULTsIndex_LTID_Total, GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, \
			_NumPinsInSubDomPerLayer, GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, \
			GPUSumNumSegsInRTMTrack_2PI, GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, \
			GPUNumLTsSum, GPUNumLongTracks, GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, \
			GPUOneOverFSRsAreaInRTM_AntiClockFSR, GPULinearExpTable, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,\
			_polarBoundLowLim,_polarBoundUpLim, _twoOverLayWidth,_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,\
			GPUTotalWeights_2DMOC,GPUMacroXS_t_wScale,_groupBoundLowLim, _groupBoundUpLim);
	}
	else if (_GPU_FixSourSolver == 1) {
		GPUTS_Group_ExpFunc_OneDir_AngFSmem_FixSourProb_2D<<<_GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_layBoundLowLim, \
			GPUScalarFlux, GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, \
			GPULTsIndex_LTID_Total, GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, \
			_NumPinsInSubDomPerLayer, GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, \
			GPUSumNumSegsInRTMTrack_2PI, GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, \
			GPUNumLTsSum, GPUNumLongTracks, GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, \
			GPUOneOverFSRsAreaInRTM_AntiClockFSR, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,_polarBoundLowLim, \
			_polarBoundUpLim, _twoOverLayWidth,_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC,\
			GPUMacroXS_t_wScale,_groupBoundLowLim, _groupBoundUpLim);
	}
	else if (_GPU_FixSourSolver == 2) {
		GPUTS_Group_DDExpInte_OneDir_AngFSmem_FixSourProb_2D<<<_GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_layBoundLowLim, \
			GPUScalarFlux, GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, \
			GPULTsIndex_LTID_Total, GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, \
			_NumPinsInSubDomPerLayer, GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, \
			GPUSumNumSegsInRTMTrack_2PI, GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, \
			GPUNumLTsSum, GPUNumLongTracks, GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, \
			GPUOneOverFSRsAreaInRTM_AntiClockFSR, GPULinearExpTable, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,\
			_polarBoundLowLim,_polarBoundUpLim, _twoOverLayWidth,_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,\
			GPUTotalWeights_2DMOC,_groupBoundLowLim, _groupBoundUpLim);
	}
	else if (_GPU_FixSourSolver == 3) {
		GPUTS_Group_DDExpFunc_OneDir_AngFSmem_FixSourProb_2D<<<_GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_layBoundLowLim, \
			GPUScalarFlux, GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, \
			GPULTsIndex_LTID_Total, GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, \
			_NumPinsInSubDomPerLayer, GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, \
			GPUSumNumSegsInRTMTrack_2PI, GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, \
			GPUNumLTsSum, GPUNumLongTracks, GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, \
			GPUOneOverFSRsAreaInRTM_AntiClockFSR, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,_polarBoundLowLim, \
			_polarBoundUpLim, _twoOverLayWidth,_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC,\
			_groupBoundLowLim, _groupBoundUpLim);
		}
	else {}
	/* add source */
    CUDA_CheckError((char *)("GPUTS_Group_ExpFunc_OneDir_AngFSmem_FixSourProb_2D"));
    GPUAddSourceToScalarFlux_lay_g(_numFSRsPerLay, _layBoundLowLim, _layBoundUpLim, \
		_groupBoundLowLim, _groupBoundUpLim);
	CUDA_CheckError((char *)("GPUAddSourceToScalarFlux_lay_g"));
}

void GPUMOC_TransSweep_FixSourProb_lay_g_sld(FP_PRECISION _twoOverLayWidth, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim, FP_PRECISION *_GPUFSRAveAngFlux_inp, \
	FP_PRECISION *_GPUFSRAveAngFlux_update, int _polOffset, int _layBoundLowLim, int _layBoundUpLim,\
	int _numFSRsPerLay, M_INT32 _NumPinsInSubDomPerLayer, M_INT32 _NumLTs_HS, M_INT32 _NumPolarAng,\
	M_INT32 _GPU_FixSourSolver)
{
	int shared_mem, _numPol/*, share_int_mem*/;
	int _GPUGridDim_group_oneDir=1;
#if defined(M_2DMOC)
    _numPol = _NumPolarAng;
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
    _numPol = _NumPolarAng/2;
#endif // defined(M_3DMOCex)

	// shared_mem = GPUBlockDim * _numPol * sizeof(FP_PRECISION);
	// _GPUGridDim_group_oneDir= (_NumLTs_HS * (_groupBoundUpLim-_groupBoundLowLim) * 2 - 1) / GPUBlockDim + 1;

	// GPUTS_Group_ExpFunc_OneDir_AngFSmem_FixSourProb_2D_sld<<<_GPUGridDim_group_oneDir, GPUBlockDim, shared_mem>>>(_layBoundLowLim, \
	// 	GPUScalarFlux, GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, \
	// 	GPULTsIndex_LTID_Total, GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, \
	// 	_NumPinsInSubDomPerLayer, GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, \
	// 	GPUSumNumSegsInRTMTrack_2PI, GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, \
	// 	GPUNumLTsSum, GPUNumLongTracks, GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, \
	// 	GPUOneOverFSRsAreaInRTM_AntiClockFSR, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,_polarBoundLowLim, \
	// 	_polarBoundUpLim, _twoOverLayWidth,_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC,\
	// 	GPUMacroXS_t_wScale,_groupBoundLowLim, _groupBoundUpLim);
	
	int totThread = _NumLTs_HS * (_groupBoundUpLim - _groupBoundLowLim);

	int LocalGPUGridDim_polar_oneDir = (totThread*_NumPolarAng*2 - 1) / GPUBlockDim + 1;

	GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt_11_sld<<<LocalGPUGridDim_polar_oneDir, GPUBlockDim>>>(_layBoundLowLim, GPUScalarFlux, \
					GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, GPULTsIndex_LTID_Total, \
					GPULongTracksInfo_Total, GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, _NumPinsInSubDomPerLayer, \
					GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, GPUNumSegsInTrack_Total, GPUSumNumSegsInRTMTrack_2PI, \
					GPUSumNumFSRsBasedGlobCellID, GPUSumNumFSRsInRTM, GPUMacroXS_t, GPUNumLTsSum, GPUNumLongTracks, \
					GPUMPIFlux, GPUSegInfoInTrack_2PI_length, GPUSegInfoInTrack_2PI_AntiClockFSR, GPUOneOverFSRsAreaInRTM_AntiClockFSR,\
					GPULinearExpTable, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,_polarBoundLowLim,_polarBoundUpLim,\
					_twoOverLayWidth,_GPUFSRAveAngFlux_inp, _GPUFSRAveAngFlux_update,_polOffset,GPUTotalWeights_2DMOC,GPUMacroXS_t_wScale, \
					GPUFSRAngScattSour, _groupBoundLowLim, _groupBoundUpLim);

	/* add source */
    CUDA_CheckError((char *)("GPUTS_Group_ExpFunc_OneDir_AngFSmem_FixSourProb_2D_sld"));
    GPUAddSourceToScalarFlux_lay_g(_numFSRsPerLay, _layBoundLowLim, _layBoundUpLim, \
		_groupBoundLowLim, _groupBoundUpLim);
	CUDA_CheckError((char *)("GPUAddSourceToScalarFlux_lay_g"));
}

void GPUAccumAngFlux(int _lay,int _polarBoundLowLim,int _polarBoundUpLim,int _polOffset,\
	FP_PRECISION *_aveAngFlux_update, int _GPU_TSKernelType, int _dim, int _numFSRs, int _numGroups, int _numAzim, int _numPol, int _scattOrder)
{
    int _GPUGridDim, kernel_type;
	// printf("\n _lay=%d",_lay);
	// PrintFPArrayOnDevice(TotalNumFSRs_subDom*NumGroups, GPUScalarFlux, "B4 GPUScalarFlux");
	// PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(NumPolarAng[0]/2), _aveAngFlux_update, "B4 update");
    // if (_GPU_TSKernelType < 10) {
	// 	GPUAccumAngFlux_ScalarFlux<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(_lay, GPUNumFSRs_Layer, GPUSumNumFSRs_Layer, \
	// 		_polarBoundLowLim, _polarBoundUpLim, _aveAngFlux_update, GPUScalarFlux, _polOffset);
	// 	CUDA_CheckError((char *)("GPUAccumAngFlux-GPUAccumAngFlux_ScalarFlux"));
	// }
	// else if (_GPU_TSKernelType == 10) {
		// printf("\n _polarBoundUpLim=%d",_polarBoundUpLim);
    // PrintFPArrayOnDevice(8, GPUSpheHarmCoeff, (char *)("GPUSpheHarmCoeff"));
	GPUKernelTimeStart(&_timeKernelFluxScalarInitStart, &_timeKernelFluxScalarInitStop);
	GPUVectorInit_1DFP_Flux<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(GPUScalarFlux, _dim, 0.0);
	TimeRec_FluxMomentScalarFluxInit +=
		GPUKernelTimeStop(&_timeKernelFluxScalarInitStart, &_timeKernelFluxScalarInitStop);

#ifdef M_2DMOC
    kernel_type = 103;
#else
    kernel_type = 100;
#endif

    if (kernel_type == 100) {
        _GPUGridDim = (_numFSRs * _numGroups - 1) / GPUBlockDim + 1;
        GPUAccumAngFlux_ScalarFlux_HOScatt<<<_GPUGridDim, GPUBlockDim>>>(_lay, GPUNumFSRs_Layer,
                                                                         _polarBoundLowLim, _polarBoundUpLim,
                                                                         _aveAngFlux_update, GPUScalarFlux, _polOffset,
                                                                         GPUSpheHarmCoeff);
	} else if (kernel_type == 103) {
	        // PARALLEL SCHEME: FSR - GROUP - ORDER, no atomic
		GPUKernelTimeStart(&_timeKernelFlux103Start, &_timeKernelFlux103Stop);
	        _GPUGridDim = (_numFSRs * _numGroups * (_scattOrder + 1) * (_scattOrder + 1) - 1) / GPUBlockDim + 1;
	        GPUAccumAngFlux_ScalarFlux_HOScatt_opt<<<_GPUGridDim, GPUBlockDim>>>(_lay, GPUNumFSRs_Layer,
                                                                              _polarBoundLowLim,
                                                                              _polarBoundUpLim,
	                                                                              _aveAngFlux_update,
	                                                                              GPUScalarFlux, _polOffset,
	                                                                              GPUSpheHarmCoeff);
		TimeRec_FluxMomentKernel103 +=
			GPUKernelTimeStop(&_timeKernelFlux103Start, &_timeKernelFlux103Stop);
		TimeRec_FluxMomentKernel103Calls++;
	    } else if (kernel_type == 104) {
        // PARALLEL SCHEME: FSR - GROUP - ORDER - Azim
        _GPUGridDim = (_numFSRs * _numGroups * (_scattOrder + 1) * (_scattOrder + 1) * _numAzim - 1) / GPUBlockDim + 1;
        GPUAccumAngFlux_ScalarFlux_HOScatt_opt2<<<_GPUGridDim, GPUBlockDim>>>(_lay, GPUNumFSRs_Layer,
                                                                             _polarBoundLowLim,
                                                                             _polarBoundUpLim,
                                                                             _aveAngFlux_update,
                                                                             GPUScalarFlux, _polOffset,
                                                                             GPUSpheHarmCoeff);
    } else if (kernel_type == 105) {
        // PARALLEL SCHEME: FSR - GROUP - ORDER - Pol
        _GPUGridDim = (_numFSRs * _numGroups * (_scattOrder + 1) * (_scattOrder + 1) * _numPol - 1) / GPUBlockDim + 1;
        GPUAccumAngFlux_ScalarFlux_HOScatt_opt3<<<_GPUGridDim, GPUBlockDim>>>(_lay, GPUNumFSRs_Layer,
                                                                              _polarBoundLowLim,
                                                                              _polarBoundUpLim,
                                                                              _aveAngFlux_update,
                                                                              GPUScalarFlux, _polOffset,
                                                                              GPUSpheHarmCoeff);
    }

    CUDA_CheckError((char *) ("GPUAccumAngFlux"));
    // printf("\n NumScattOrder=%d",NumScattOrder);
    // printf("\n _lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);
    // PrintFPArrayOnDevice(TotalNumFSRs_perLayer*NumGroups*(NumScattOrder+1)*(NumScattOrder+1), GPUScalarFlux, (char *)("ACCU GPUScalarFlux"));

	// }
	// else {}
}

extern "C"
void GPUAccumAngFlux_PerAzim(int _lay, int _polarBoundLowLim, int _polarBoundUpLim,
	int _polOffset, FP_PRECISION_FLUX *_aveAngFlux_update, int _dim,
	int _azimID)
{
	GPUAccumAngFlux_PerAzimBatch(_lay, _polarBoundLowLim, _polarBoundUpLim,
		_polOffset, _aveAngFlux_update, _dim, _azimID, _azimID + 1);
}

extern "C"
void GPUAccumAngFlux_PerAzimBatch(int _lay, int _polarBoundLowLim, int _polarBoundUpLim,
	int _polOffset, FP_PRECISION_FLUX *_aveAngFlux_update, int _dim,
	int _azimBegin, int _azimEnd)
{
	const int _grid = (_dim - 1) / GPUBlockDim + 1;
	GPUKernelTimeStart(&_timeKernelFluxPerAzimStart, &_timeKernelFluxPerAzimStop);
	GPUAccumAngFlux_ScalarFlux_HOScatt_PerAzim<<<_grid, GPUBlockDim>>>(
		_lay, GPUNumFSRs_Layer, _polarBoundLowLim, _polarBoundUpLim,
		_aveAngFlux_update, GPUScalarFluxAzimAccum, _polOffset,
		GPUSpheHarmCoeff, _azimBegin, _azimEnd);
	TimeRec_FluxMomentKernelPerAzim +=
		GPUKernelTimeStop(&_timeKernelFluxPerAzimStart, &_timeKernelFluxPerAzimStop);
	TimeRec_FluxMomentKernelPerAzimCalls++;
	CUDA_CheckError((char *)("GPUAccumAngFlux_PerAzimBatch"));
}

/**
* @ Function: GPUJacobi_TransFSRPara_RayWisedAndAntiClock()
*   introduction: transform the original FSR related value to a new assigned array with ray-wised ID
* @ Parameter List:
*   Return: void
* @ Created     : 2019/06/15, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
__global__ void GPUJacobi_TransFSRPara_RayWisedAndAntiClock(int _azimID, int *_GPUNumFSRsInCell, \
	int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUFsrIdMap_rayWisedToAntiClock, int *_GPUCellRTMType_SubDom, \
	FP_PRECISION *_GPUPara, FP_PRECISION *_GPUParaOrig, int *_GPUSumNumFSRsInRTM)
{
	int g, _reg, _ind, _fsrID, _cellRTMType,_fsrID_local, _ind1, _fsrID1, _mapID;
	int _tid = threadIdx.x + blockIdx.x * blockDim.x;
    /* Iterate over all FSRs */
    while (_tid < (*GPUTotNumPins)) {
		_cellRTMType = _GPUCellRTMType_SubDom[_tid];
		for (_reg = 0; _reg < _GPUNumFSRsInCell[_tid]; _reg++) {
			_mapID = _azimID*(*GPUTotalNumFSRsInRTM) + _GPUSumNumFSRsInRTM[_cellRTMType] + _reg;
			_fsrID_local = _GPUFsrIdMap_rayWisedToAntiClock[_mapID];
			_fsrID = _GPUSumNumFSRsBasedGlobCellID[_tid] + _fsrID_local;
			_fsrID1 = _GPUSumNumFSRsBasedGlobCellID[_tid] + _reg;
			for (g = 0; g < (*GPUNumMOCGroups); g++) {
				_ind = _fsrID*(*GPUNumMOCGroups) + g;
				_ind1 = _fsrID1*(*GPUNumMOCGroups) + g;
				_GPUPara[_ind1] = _GPUParaOrig[_ind];
			}
		}
        /* Increment the thread id */
        _tid += blockDim.x * gridDim.x;
    }
}

/**
* @ Function: GPUJacobi_TransFSRPara_RayWisedAndAntiClock_group()
*   introduction: transform the original FSR related value to a new assigned array with ray-wised ID
* @ Parameter List:
*   Return: void
* @ Created     : 2019/06/15, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
__global__ void GPUJacobi_TransFSRPara_RayWisedAndAntiClock_group(int _azimID, int *_GPUNumFSRsInCell, \
	int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUFsrIdMap_rayWisedToAntiClock, int *_GPUCellRTMType_SubDom, \
	FP_PRECISION *_GPUPara, FP_PRECISION *_GPUParaOrig, int *_GPUSumNumFSRsInRTM)
{
	int g, _reg, _ind, _fsrID, _cellRTMType, _fsrID_local, _ind1, _fsrID1, _mapID;
	int _tid = threadIdx.x + blockIdx.x * blockDim.x;
	/* Iterate over all FSRs */
	while (_tid < (*GPUTotNumPins)) {
		_cellRTMType = _GPUCellRTMType_SubDom[_tid];
		for (_reg = 0; _reg < _GPUNumFSRsInCell[_tid]; _reg++) {
			_mapID = _azimID * (*GPUTotalNumFSRsInRTM) + _GPUSumNumFSRsInRTM[_cellRTMType] + _reg;
			_fsrID_local = _GPUFsrIdMap_rayWisedToAntiClock[_mapID];
			_fsrID = _GPUSumNumFSRsBasedGlobCellID[_tid] + _fsrID_local;
			_fsrID1 = _GPUSumNumFSRsBasedGlobCellID[_tid] + _reg;
			for (g = 0; g < (*GPUNumMOCGroups); g++) {
				_ind = _fsrID * (*GPUNumMOCGroups) + g;
				//_ind1 = _fsrID1 * (*GPUNumMOCGroups) + g;
				_ind1 = g * (*GPUTotNumFSRs) + _fsrID1;
				_GPUPara[_ind1] = _GPUParaOrig[_ind];
			}
		}
		/* Increment the thread id */
		_tid += blockDim.x * gridDim.x;
	}
}

/**
* @ Function: GPUJacobi_TransFSRPara_RayWisedAndAntiClock_group_1()
*   introduction: transform the original FSR related value to a new assigned array with ray-wised ID
* @ Parameter List:
*   Return: void
* @ Created     : 2019/07/02, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
__global__ void GPUJacobi_TransFSRPara_RayWisedAndAntiClock_group_1(int _azimID, int *_GPUNumFSRsInCell, \
	int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUFsrIdMap_rayWisedToAntiClock, int *_GPUCellRTMType_SubDom, \
	FP_PRECISION *_GPUPara, FP_PRECISION *_GPUParaOrig, int *_GPUSumNumFSRsInRTM)
{
	int /*g,*/ _reg, _ind, _fsrID, _cellRTMType, _fsrID_local, _ind1, _fsrID1, _mapID;
	int _locPinID, _groupID;
	int _tid = threadIdx.x + blockIdx.x * blockDim.x;
	// _tid = threadIdx.x + blockIdx.x * blockDim.x;
	_locPinID = _tid / (*GPUNumMOCGroups);
	_groupID = _tid % (*GPUNumMOCGroups);
	//_polEngID = _groupID * (*GPUNumPolarAng);
	/* Iterate over all FSRs */
	if (_locPinID < (*GPUTotNumPins)) {
		_cellRTMType = _GPUCellRTMType_SubDom[_locPinID];
		for (_reg = 0; _reg < _GPUNumFSRsInCell[_locPinID]; _reg++) {
			_mapID = _azimID * (*GPUTotalNumFSRsInRTM) + _GPUSumNumFSRsInRTM[_cellRTMType] + _reg;
			_fsrID_local = _GPUFsrIdMap_rayWisedToAntiClock[_mapID];
			_fsrID = _GPUSumNumFSRsBasedGlobCellID[_locPinID] + _fsrID_local;
			_fsrID1 = _GPUSumNumFSRsBasedGlobCellID[_locPinID] + _reg;

			_ind = _fsrID * (*GPUNumMOCGroups) + _groupID;
			_ind1 = _groupID * (*GPUTotNumFSRs) + _fsrID1;
			_GPUPara[_ind1] = _GPUParaOrig[_ind];

			//for (g = 0; g < (*GPUNumMOCGroups); g++) {
			//	_ind = _fsrID * (*GPUNumMOCGroups) + g;
			//	//_ind1 = _fsrID1 * (*GPUNumMOCGroups) + g;
			//	_ind1 = g * (*GPUTotNumFSRs) + _fsrID1;
			//	_GPUPara[_ind1] = _GPUParaOrig[_ind];
			//}
		}
		/* Increment the thread id */
		// _tid += blockDim.x * gridDim.x;
	}
}

/**
* @ Function: GPUTransOneOverFSRVolInTRM_RayWised()
*   introduction: transform the original FSR related volume to a new assigned array with ray-wised ID
* @ Parameter List:
*   Return: void
* @ Created     : 2019/06/15, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
__global__ void GPUTransOneOverFSRVolInTRM_RayWised(FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM_AntiClockFSR, \
	FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM_RayWisedFSR, int _azimID, \
	int *_GPUFsrIdMap_rayWisedToAntiClock, int *_GPUSumNumFSRsInRTM)
{
	int _mapID, _origFsrId;
	int _tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (_tid < (*GPUTotalNumFSRsInRTM)) {
		_mapID= _azimID*(*GPUTotalNumFSRsInRTM) + _tid;
		_origFsrId = _GPUFsrIdMap_rayWisedToAntiClock[_mapID];
		_GPUOneOverFSRsAreaInRTM_RayWisedFSR[_tid] = _GPUOneOverFSRsAreaInRTM_AntiClockFSR[_GPUSumNumFSRsInRTM[_tid]+ _origFsrId];
		/* Increment the thread id */
        _tid += blockDim.x * gridDim.x;
	}
}
/**
* @ Function: GPUJacobi_ScalarFlux_RayWisedSumToAntiClock()
*   introduction: add the ray-wised scalar flux to anti-clock scalar flux by azimuthal angle
* @ Parameter List:
*   Return: void
* @ Created     : 2019/06/17, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
__global__ void GPUJacobi_ScalarFlux_RayWisedSumToAntiClock(int _azimID, int *_GPUNumFSRsInCell, \
	int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUFsrIdMap_rayWisedToAntiClock, int *_GPUCellRTMType_SubDom, \
	FP_PRECISION_FLUX *_GPUflux_rayWisedFSR, FP_PRECISION_FLUX *_GPUflux_antiClockFSR, int *_GPUSumNumFSRsInRTM)
{
	int g, _reg, /*_lay,*/ _ind, _fsrID, _cellRTMType, _fsrID_local, _ind1, _fsrID1, _mapID;
	int _tid = threadIdx.x + blockIdx.x * blockDim.x;
	/* Iterate over all FSRs */
	while (_tid < (*GPUTotNumPins)) {
		_cellRTMType = _GPUCellRTMType_SubDom[_tid];
		for (_reg = 0; _reg < _GPUNumFSRsInCell[_tid]; _reg++) {
			_mapID = _azimID * (*GPUTotalNumFSRsInRTM) + _GPUSumNumFSRsInRTM[_cellRTMType] + _reg;
			_fsrID_local = _GPUFsrIdMap_rayWisedToAntiClock[_mapID];
			_fsrID = _GPUSumNumFSRsBasedGlobCellID[_tid] + _fsrID_local;
			_fsrID1 = _GPUSumNumFSRsBasedGlobCellID[_tid] + _reg;
			for (g = 0; g < (*GPUNumMOCGroups); g++) {
				_ind = _fsrID * (*GPUNumMOCGroups) + g;
				_ind1 = _fsrID1 * (*GPUNumMOCGroups) + g;
				_GPUflux_antiClockFSR[_ind] += _GPUflux_rayWisedFSR[_ind1];
				//_GPUflux_rayWisedFSR[_ind1] = (FP_PRECISION_FLUX)(0.0);
			}
		}
		/* Increment the thread id */
		_tid += blockDim.x * gridDim.x;
	}
}

/**
* @ Function: GPUJacobi_ScalarFlux_RayWisedSumToAntiClock()
*   introduction: add the ray-wised scalar flux to anti-clock scalar flux by azimuthal angle
* @ Parameter List:
*   Return: void
* @ Created     : 2019/06/17, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
__global__ void GPUJacobi_ScalarFlux_RayWisedSumToAntiClock_group(int _azimID, int *_GPUNumFSRsInCell, \
	int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUFsrIdMap_rayWisedToAntiClock, int *_GPUCellRTMType_SubDom, \
	FP_PRECISION_FLUX *_GPUflux_rayWisedFSR, FP_PRECISION_FLUX *_GPUflux_antiClockFSR, int *_GPUSumNumFSRsInRTM)
{
	int g, _reg, /*_lay,*/ _ind, _fsrID, _cellRTMType, _fsrID_local, _ind1, _fsrID1, _mapID;
	int _tid = threadIdx.x + blockIdx.x * blockDim.x;
	/* Iterate over all FSRs */
	while (_tid < (*GPUTotNumPins)) {
		_cellRTMType = _GPUCellRTMType_SubDom[_tid];
		for (_reg = 0; _reg < _GPUNumFSRsInCell[_tid]; _reg++) {
			_mapID = _azimID * (*GPUTotalNumFSRsInRTM) + _GPUSumNumFSRsInRTM[_cellRTMType] + _reg;
			_fsrID_local = _GPUFsrIdMap_rayWisedToAntiClock[_mapID];
			_fsrID = _GPUSumNumFSRsBasedGlobCellID[_tid] + _fsrID_local;
			_fsrID1 = _GPUSumNumFSRsBasedGlobCellID[_tid] + _reg;
			for (g = 0; g < (*GPUNumMOCGroups); g++) {
				_ind = _fsrID * (*GPUNumMOCGroups) + g;
				//_ind1 = _fsrID1 * (*GPUNumMOCGroups) + g;
				_ind1 = g * (*GPUTotNumFSRs) + _fsrID1;
				_GPUflux_antiClockFSR[_ind] += _GPUflux_rayWisedFSR[_ind1];
				//_GPUflux_rayWisedFSR[_ind1] = (FP_PRECISION_FLUX)(0.0);
			}
		}
		/* Increment the thread id */
		_tid += blockDim.x * gridDim.x;
	}
}

__global__ void GPUJacobi_ScalarFlux_RayWisedSumToAntiClock_group_1(int _azimID, int *_GPUNumFSRsInCell, \
	int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUFsrIdMap_rayWisedToAntiClock, int *_GPUCellRTMType_SubDom, \
	FP_PRECISION_FLUX *_GPUflux_rayWisedFSR, FP_PRECISION_FLUX *_GPUflux_antiClockFSR, int *_GPUSumNumFSRsInRTM)
{
	int /*g, _lay,*/ _reg, _ind, _fsrID, _cellRTMType, _fsrID_local, _ind1, _fsrID1, _mapID;
	int _tid = threadIdx.x + blockIdx.x * blockDim.x;
	int _locPinID, _groupID;

	_locPinID = _tid / (*GPUNumMOCGroups);
	_groupID = _tid % (*GPUNumMOCGroups);

	/* Iterate over all FSRs */
	if (_locPinID < (*GPUTotNumPins)) {
		_cellRTMType = _GPUCellRTMType_SubDom[_locPinID];
		for (_reg = 0; _reg < _GPUNumFSRsInCell[_locPinID]; _reg++) {
			_mapID = _azimID * (*GPUTotalNumFSRsInRTM) + _GPUSumNumFSRsInRTM[_cellRTMType] + _reg;
			_fsrID_local = _GPUFsrIdMap_rayWisedToAntiClock[_mapID];
			_fsrID = _GPUSumNumFSRsBasedGlobCellID[_locPinID] + _fsrID_local;
			_fsrID1 = _GPUSumNumFSRsBasedGlobCellID[_locPinID] + _reg;

			_ind = _fsrID * (*GPUNumMOCGroups) + _groupID;
			_ind1 = _groupID * (*GPUTotNumFSRs) + _fsrID1;
			_GPUflux_antiClockFSR[_ind] += _GPUflux_rayWisedFSR[_ind1];

			//for (g = 0; g < (*GPUNumMOCGroups); g++) {
			//	_ind = _fsrID * (*GPUNumMOCGroups) + g;
			//	//_ind1 = _fsrID1 * (*GPUNumMOCGroups) + g;
			//	_ind1 = g * (*GPUTotNumFSRs) + _fsrID1;
			//	_GPUflux_antiClockFSR[_ind] += _GPUflux_rayWisedFSR[_ind1];
			//	//_GPUflux_rayWisedFSR[_ind1] = (FP_PRECISION_FLUX)(0.0);
			//}
		}
		/* Increment the thread id */
		//_tid += blockDim.x * gridDim.x;
	}
}

/**
 * @ Function: GPUAddSourceToScalarFlux()
 *   introduction: add the source to scalar flux after transport sweep
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2018/06/05, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
void GPUAddSourceToScalarFlux()
{
	//cudaError_t CUDAStatus;
	GPUAddSourceToScalarFluxOnDevice<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(GPUScalarFlux,GPUMOCSource,GPUMacroXS_t);
	CUDA_CheckError((char *)("GPUAddSourceToScalarFlux-GPUAddSourceToScalarFluxOnDevice"));
}

void GPUAddSourceToScalarFlux_lay_g(int _numFSRsPerLay, int _layBoundLowLim, int _layBoundUpLim, \
	int _groupBoundLowLim, int _groupBoundUpLim)
{
	GPUAddSourceToScalarFluxOnDevice_lay_g<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(_numFSRsPerLay, _layBoundLowLim, \
		_layBoundUpLim, _groupBoundLowLim, _groupBoundUpLim, GPUScalarFlux,GPUMOCSource,GPUMacroXS_t);
}

/**
 * @ Function: GPUConvergenceCheck()
 *   introduction: evaluate the kerr and flux residual, and check the convergence
 * @ Parameter List:
 *   Return:
 * @ Created     : 2018/06/05, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
int GPUConvergenceCheck(int _eigenValueIter,int _procID,int _numProcs)
{
    //FP_PRECISION _kerr;
    int _bool=0;
    //_kerr=fabs((GPUKeff-GPUKeffOld)/GPUKeffOld);
    //if (_procID==0) {
    //    GPUKeffRec[_eigenValueIter]=GPUKeff;
    //    GPUFluxResRec[_eigenValueIter]=GPUFluxRes_Glob;
    //    printf("    k_eff = %.10lf    k_err = %.1e    FluxInfNorm = %.1e",GPUKeff,_kerr,GPUFluxRes_Glob);
    //    fprintf(LogFile,"    k_eff = %.10lf    k_err = %.6e    FluxInfNorm = %.6e",GPUKeff,_kerr,GPUFluxRes_Glob);
    //}
    //if (_kerr<ConvCritK && GPUFluxRes_Glob<ConvCritFlux) {//
    //    if (_procID==0) {
    //        printf("\n [  RESULT ]  The Abslute Error of Eigenvalue is %d pcm", ((int)((GPUKeff - KRef)*1.e5)));
    //        fprintf(LogFile,"\n [  RESULT ]  The Abslute Error of Eigenvalue is %d pcm", ((int)((GPUKeff - KRef)*1.e5)));
    //    }
    //    _bool=1;
    //}
    return _bool;
}


FP_PRECISION *GPUAllocMem_FP_I(size_t _dim1, char *_parameter)
{
	FP_PRECISION *_pointer = NULL;
	if ((_pointer = (FP_PRECISION *)calloc(_dim1, sizeof(FP_PRECISION))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"GPUAllocMem_FP_I\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"GPUAllocMem_FP_I\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	return _pointer;
}

M_INT32 *GPUAllocMem_INT32_I(size_t _dim1, char *_parameter)
{
	M_INT32 *_pointer = NULL;
	if ((_pointer = (M_INT32 *)calloc(_dim1, sizeof(M_INT32))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"GPUAllocMem_INT32_I\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"GPUAllocMem_INT32_I\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	return _pointer;
}

M_UINT64 *GPUAllocMem_UINT64_I(size_t _dim1, char *_parameter)
{
	M_UINT64 *_pointer = NULL;
	if ((_pointer = (M_UINT64 *)calloc(_dim1, sizeof(M_UINT64))) == NULL) {
		if (strcmp(_parameter,CHAR_NONE) != 0) {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"GPUAllocMem_UINT64_I\" - %s\n", _parameter);
		}
		else {
			printf("\n [**ERROR**]  Memory Allocation Failed in \"GPUAllocMem_UINT64_I\" - local variable\n");
		}
		fflush(stdout);
		exit(69);
	}
	return _pointer;
}

extern "C"
void GPUScalarFluxUpdate_TS(int _TotalNumFSRs_subDom, int _NumGroups)
{
	cudaMemcpy((void*)GPUScalarFluxOld, (void*)GPUScalarFlux, (size_t)(_TotalNumFSRs_subDom*_NumGroups) * sizeof(FP_PRECISION), cudaMemcpyDeviceToDevice);
	CUDA_CheckError((char *)("GPUScalarFluxUpdate_TS-cudaMemcpy-GPUScalarFluxOld"));
	cudaMemcpy((void*)GPUScalarFluxTemp, (void*)GPUScalarFlux, (size_t)(_TotalNumFSRs_subDom*_NumGroups) * sizeof(FP_PRECISION), cudaMemcpyDeviceToDevice);
	CUDA_CheckError((char *)("GPUScalarFluxUpdate_TS-cudaMemcpy-GPUScalarFluxTemp"));
}

void GPUScalarFluxUpdate_FixSour1(int _TotalNumFSRs_subDom, int _NumGroups, FP_PRECISION_FLUX *_scalarFlux_MOC)
{
//	cudaMemcpy((void*)ScalarFlux_MOC, (void*)GPUScalarFlux, (size_t)(_TotalNumFSRs_subDom*_NumGroups) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)_scalarFlux_MOC, (void*)GPUScalarFlux, (size_t)(_TotalNumFSRs_subDom*_NumGroups) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
	CUDA_CheckError((char *)("GPUScalarFluxUpdate_FixSour1"));
}

void GPUScalarFluxUpdate_FixSour2(int _TotalNumFSRs_subDom, int _NumGroups, FP_PRECISION_FLUX *_scalarFluxOld)
{
//	cudaMemcpy((void*)ScalarFluxOld, (void*)GPUScalarFlux, (size_t)(_TotalNumFSRs_subDom*_NumGroups) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)_scalarFluxOld, (void*)GPUScalarFlux, (size_t)(_TotalNumFSRs_subDom*_NumGroups) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
    CUDA_CheckError((char *)("GPUScalarFluxUpdate_FixSour2"));
}

void GPUCopyBoundAngFluxHostToDevice(M_INT32 _MOCNumLayersZ, M_INT32 _NumLTs_HS, M_INT32 _NumPolarAng, M_INT32 _NumGroups,\
	FP_PRECISION *_BoundaryFlux, M_INT32 _NumAzimAng, M_INT32 **_GCCopy_NumDataInAngF, M_INT32 **_GCCopy_StartPInAngF)
{
	int _lay, _azim;
	//#ifdef MPI_USED
	//if (NumProcs > 1) {
#ifndef GC_COPY_AZIM
  #ifndef CUDA_AWARE_MPI
    #ifndef OMP_ASYN
		cudaMemcpy((void*)GPUBoundaryFlux, (void*)_BoundaryFlux, (size_t)(_MOCNumLayersZ*_NumLTs_HS * 2 * _NumPolarAng * _NumGroups) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
    #endif // OMP_ASYN 
//#ifndef HYBRID_ITER
//#ifdef OMP_ASYN
//		cudaMemcpy((void*)GPUBoundaryFlux, (void*)/*DFlux*/_BoundaryFlux[0], _MOCNumLayersZ*_NumLTs_HS[0] * 2 * _NumPolarAng[0] * _NumGroups * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
//#endif // OMP_ASYN
//#endif // HYBRID_ITER
  #endif // !CUDA_AWARE_MPI
#endif // !GC_COPY_AZIM

#ifdef GC_COPY_AZIM
// if (ProcID==0)
// GPUPrintf1DArray_FP(_MOCNumLayersZ*_NumLTs_HS * 2 * _NumPolarAng * _NumGroups, GPUBoundaryFlux, "B4 cudaMemcpy GPUBoundaryFlux", LogFile);
		for (_lay = 0; _lay < _MOCNumLayersZ; _lay++) {
			for (_azim = 0; _azim < _NumAzimAng; _azim++) {
				if (_GCCopy_NumDataInAngF[_lay][_azim] > 0) {
					cudaMemcpy((void*)&GPUBoundaryFlux[_GCCopy_StartPInAngF[_lay][_azim]], (void*)&_BoundaryFlux[_GCCopy_StartPInAngF[_lay][_azim]], \
					(size_t)(_GCCopy_NumDataInAngF[_lay][_azim]) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
				}
			}
		}

#endif // GC_COPY_AZIM
	//}
#ifndef GPU_STREAM_USED
  #ifdef TIME_REC
		cudaDeviceSynchronize();
  #endif // TIME_REC
#endif // !GPU_STREAM_USED

#ifdef CUDAErrorCheck
	CUDAStatus = cudaGetLastError();
	if (CUDAStatus != cudaSuccess) {
		printf("\n [GPU-ERROR]  Lunch Failed: GPUCopyBoundAngFluxHostToDevice with Info: %s\n", cudaGetErrorString(CUDAStatus));
		exit(69);
	}
#endif // CUDAErrorCheck
}

/**
 * @ Function: GPUGetFluxResidualFromDevice()
 *   introduction: evaluate the flux residual
 * @ Parameter List:
 *   Return:
 * @ Created     : 2018/06/05, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
FP_PRECISION GPUGetFluxResidualFromDevice()
{
//    int _dim=TotalNumFSRs_subDom*NumGroups;
    FP_PRECISION _infNorm=0.0;
//	//cudaError_t CUDAStatus;
//    GPUComputeFSRInfNormOnDevice<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(GPUScalarFlux,GPUScalarFluxOld,\
//        GPUFluxResidual);
//
//    cudaMemcpy((void*)FluxResidual, (void*)GPUFluxResidual,(size_t)(_dim) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
//	// _infNorm=GPUVectorGetMaxValue(FluxResidual,_dim);
//	_infNorm=GetRMSE_Vector(_dim, FluxResidual);
//	/*cudaDeviceSynchronize();*/
//#ifdef CUDAErrorCheck
//	CUDAStatus = cudaGetLastError();
//	if (CUDAStatus != cudaSuccess) {
//		printf("\n [GPU-ERROR]  Lunch Failed: GPUGetFluxResidualFromDevice with Info: %s\n", cudaGetErrorString(CUDAStatus));
//		exit(69);
//	}
//#endif // CUDAErrorCheck
//#ifdef TIME_REC
//	cudaDeviceSynchronize();
//#endif // TIME_REC
    return _infNorm;
}

extern "C"
// void GPUFinalOutputs(M_INT32 _dim, M_INT32 _OUTPUT_flux, M_INT32 _OUTPUT_flux_mat, M_INT32 _OUTPUT_normPinPower, FP_PRECISION *_ScalarFlux_MOC)
// {
// // #if defined(CUDA_USED)
// 	if (_OUTPUT_flux==1 || _OUTPUT_flux_mat==1 || _OUTPUT_normPinPower==1) {
// 		cudaMemcpy((void*)ScalarFlux_MOC, (void*)GPUScalarFlux,(size_t)(_dim) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
// 	}
// // #endif // defined(CUDA_USED)
// }

extern "C"
void GPUCopyBoundAngFluxDeviceToHost_Synch(FP_PRECISION *_CPUMPIFlux, size_t _dim)
{
	cudaMemcpy((void*)_CPUMPIFlux,(void*)GPUMPIFlux, _dim *sizeof(FP_PRECISION),cudaMemcpyDeviceToHost);
}


void SetKernelCallingPara(M_INT32 _NumLTs_HS, M_INT32 _NumGroups, M_INT32 _NumPolarAng,\
	M_INT32 _TotalNumFSRs_subDom, M_INT32 _TotalNumFSRs_perLayer, M_INT32 _MOCNumLayersZ, \
	M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, M_INT32 _NumCMFDGroup, \
	M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumPinsInSubDomPerLayer, M_INT32 _NumCoarseCMFDGroup)
{
	// int dimy = 8;
	int totThread = _NumLTs_HS * _NumGroups;
	int _polarTimesGroup = _NumPolarAng * _NumGroups;
	int _angMul2 = NumAzimAng[0] * _NumPolarAng / 2;
	int _angMul1 = _NumGroups * _angMul2;
#if defined(M_MG_CMFD)
	int _numEle_X,_numEle_Y,_numEle_Z, _numEle,_numEleLay;
	// int _numPinsOneDir;
	int _lay, _numRed=0, _numBlack=0;
#endif //defined(M_MG_CMFD)

	cudaMemcpyToSymbol(GPUNumMOCGroups, (void*)&_NumGroups, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPUPolarTimesGroups, (void*)&_polarTimesGroup, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPU_angMul1_HP, (void*)&_angMul1, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPU_angMul2_HP, (void*)&_angMul2, sizeof(int), 0, cudaMemcpyHostToDevice);
	_angMul2 = NumAzimAng[0] * _NumPolarAng;
	_angMul1 = _NumGroups * _angMul2;
	cudaMemcpyToSymbol(GPU_angMul1_FP, (void*)&_angMul1, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPU_angMul2_FP, (void*)&_angMul2, sizeof(int), 0, cudaMemcpyHostToDevice);

	GPUGridDim_group_biDir = (totThread - 1) / GPUBlockDim + 1;
	GPUGridDim_group_oneDir= (totThread*2 - 1) / GPUBlockDim + 1;
	GPUGridDim_polar_biDir = (totThread*_NumPolarAng - 1) / GPUBlockDim + 1;
	GPUGridDim_polar_oneDir = (totThread*_NumPolarAng*2 - 1) / GPUBlockDim + 1;
	GPUGridDim_track_biDir= (_NumLTs_HS - 1) / GPUBlockDim + 1;
	GPUGridDim_track_oneDir = (_NumLTs_HS*2 - 1) / GPUBlockDim + 1;

	GPUGridDim_source_eval = (_TotalNumFSRs_subDom -1)/ GPUBlockDim + 1;
	GPUGridDim_source_para = (_TotalNumFSRs_subDom*_NumGroups - 1) / GPUBlockDim + 1;
	GPUGridDim_source_para_Layer = (_TotalNumFSRs_perLayer*_NumGroups - 1) / GPUBlockDim + 1;
#if defined(M_MG_CMFD)
	_numEle_X = _MOCNumLayersZ * (_MOCNumPinsInSubDom_X + 1) * _MOCNumPinsInSubDom_Y * _NumGroups;
	_numEle_Y = _MOCNumLayersZ * _MOCNumPinsInSubDom_X * (_MOCNumPinsInSubDom_Y + 1) * _NumGroups;
	_numEle_Z = (_MOCNumLayersZ + 1) * _MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y * _NumGroups;

	GPUGridDim_MOCPartCurr_X = (_numEle_X - 1) / GPUBlockDim + 1;
	GPUGridDim_MOCPartCurr_Y = (_numEle_Y - 1) / GPUBlockDim + 1;
	GPUGridDim_MOCNetCurr_Z = (_numEle_Z - 1) / GPUBlockDim + 1;

	_numEle_X = _MOCNumLayersZ * (_MOCNumPinsInSubDom_X + 1) * _MOCNumPinsInSubDom_Y * _NumCMFDGroup;
	_numEle_Y = _MOCNumLayersZ * _MOCNumPinsInSubDom_X * (_MOCNumPinsInSubDom_Y + 1) * _NumCMFDGroup;
	_numEle_Z = (_MOCNumLayersZ + 1) * _MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y * _NumCMFDGroup;
	GPUGridDim_CMFDSurf_X = (_numEle_X - 1) / GPUBlockDim + 1;
	GPUGridDim_CMFDSurf_Y = (_numEle_Y - 1) / GPUBlockDim + 1;
	GPUGridDim_CMFDSurf_Z = (_numEle_Z - 1) / GPUBlockDim + 1;

	_numEle = _MOC_NumPinsInSubDom * _NumCMFDGroup;
	_numEleLay = _NumPinsInSubDomPerLayer * _NumCMFDGroup;
	GPUGridDim_CMFDNode = (_numEle - 1) / GPUBlockDim + 1;
	GPUGridDim_CMFDNode_Layer = (_numEleLay - 1) / GPUBlockDim + 1;

	_numEle = _MOC_NumPinsInSubDom * _NumCMFDGroup * _NumCMFDGroup;
	_numEleLay = _NumPinsInSubDomPerLayer * _NumCMFDGroup * _NumCMFDGroup;
	GPUGridDim_CMFDXSs = (_numEleLay - 1) / GPUBlockDim + 1;
	//printf("\n GPUBlock.x=%d, GPUBlock.y=%d, GPUBlock.z=%d", GPUBlock.x, GPUBlock.y, GPUBlock.z);
	//printf("\n GPUGrid.x=%d, GPUGrid.y=%d, GPUGrid.z=%d", GPUGrid.x, GPUGrid.y, GPUGrid.z);
	//printf("\n GPUGridDim_group_oneDir=%d, GPUBlockDim=%d, GPUGridDim_polar_biDir=%d", GPUGridDim_group_oneDir, GPUBlockDim, GPUGridDim_polar_biDir);
	//cudaOccMaxPotentialOccupancyBlockSize(&minGridSize, &blockSize, (CUfunction)GPUTransportSweepOnDevice_group_diam_oneDir_AngFReg,NULL, 0, N);
	TotNumLTs_CMFDGroup = _MOCNumLayersZ * _NumLTs_HS * _NumCMFDGroup * 2;
	GPUGridDim_LT_CMFDGroup = (TotNumLTs_CMFDGroup - 1) / GPUBlockDim + 1;

	_numRed = 0;
	_numBlack = 0;
	if (_MOCNumPinsInSubDom_X % 2 == 0 || _MOCNumPinsInSubDom_Y % 2 == 0) {
		_numRed = (_MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y) / 2 * _MOCNumLayersZ;
		_numBlack = _numRed;
	}
	else {
		for (_lay=0; _lay<_MOCNumLayersZ; _lay++) {
			if (_lay % 2 ==0) {
				_numRed += (_MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y) / 2 + 1;
				_numBlack += (_MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y) / 2;
			}
			else {
				_numRed += (_MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y) / 2;
				_numBlack += (_MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y) / 2 + 1;
			}
		}
	}
	GPU_MGCMFDNumRedThreads = _numRed * _NumCMFDGroup;
	GPU_MGCMFDNumBlackThreads = _numBlack * _NumCMFDGroup;
	GPUGridDim_MGCMFDRed = (GPU_MGCMFDNumRedThreads - 1) / GPUBlockDim + 1;
	GPUGridDim_MGCMFDBlack = (GPU_MGCMFDNumBlackThreads - 1) / GPUBlockDim + 1;
	cudaMemcpyToSymbol(GPU_MGCMFDNumRedThreads_Constant, (void*)&GPU_MGCMFDNumRedThreads, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(GPU_MGCMFDNumBlackThreads_Constant, (void*)&GPU_MGCMFDNumBlackThreads, sizeof(int), 0, cudaMemcpyHostToDevice);


	if (_MOCNumPinsInSubDom_Y % 2 == 0) {
		GPUNumRedLay_OddNx = _MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y / 2;
		GPUNumBlackLay_OddNx = GPUNumRedLay_OddNx;
	}
	else {
		GPUNumRedLay_OddNx = (_MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y) / 2 + 1;
		GPUNumBlackLay_OddNx = (_MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y) / 2;
	}

#if defined(M_CG_CMFD)
	_numEle = _MOC_NumPinsInSubDom * _NumCoarseCMFDGroup;
	GPUGridDim_CGCMFDNode = (_numEle - 1) / GPUBlockDim + 1;

	_numEle = _MOC_NumPinsInSubDom * _NumCoarseCMFDGroup * _NumCoarseCMFDGroup;
	GPUGridDim_CGCMFDXSs = (_numEle - 1) / GPUBlockDim + 1;

	_numEle_X = _MOCNumLayersZ * (_MOCNumPinsInSubDom_X + 1) * _MOCNumPinsInSubDom_Y * _NumCoarseCMFDGroup;
	_numEle_Y = _MOCNumLayersZ * _MOCNumPinsInSubDom_X * (_MOCNumPinsInSubDom_Y + 1) * _NumCoarseCMFDGroup;
	_numEle_Z = (_MOCNumLayersZ + 1) * _MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y * _NumCoarseCMFDGroup;
	GPUGridDim_CGCMFDSurf_X = (_numEle_X - 1) / GPUBlockDim + 1;
	GPUGridDim_CGCMFDSurf_Y = (_numEle_Y - 1) / GPUBlockDim + 1;
	GPUGridDim_CGCMFDSurf_Z = (_numEle_Z - 1) / GPUBlockDim + 1;

	GPU_CGCMFDNumRedThreads = _numRed * _NumCoarseCMFDGroup;
	GPU_CGCMFDNumBlackThreads = _numBlack * _NumCoarseCMFDGroup;
	GPUGridDim_CGCMFDRed = (GPU_CGCMFDNumRedThreads - 1) / GPUBlockDim + 1;
	GPUGridDim_CGCMFDBlack = (GPU_CGCMFDNumBlackThreads - 1) / GPUBlockDim + 1;

	cudaMemcpyToSymbol(GPU_CGCMFDNumRedThreads_Constant, (void*)&GPU_CGCMFDNumRedThreads, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GPU_CGCMFDNumBlackThreads_Constant, (void*)&GPU_CGCMFDNumBlackThreads, sizeof(int), 0, cudaMemcpyHostToDevice);

#endif // defined(M_CG_CMFD)

#endif // defined(M_MG_CMFD)
#ifdef CUDAErrorCheck
	CUDAStatus = cudaGetLastError();
	if (CUDAStatus != cudaSuccess) {
		printf("\n [GPU-ERROR]  Lunch Failed: after with Info: %s\n", cudaGetErrorString(CUDAStatus));
		exit(69);
	}
#endif // CUDAErrorCheck
}

__global__ void CUDATestPTX_ExpFunction(FP_PRECISION_TEST *_gpuExpFunTest)
{
	int _tid = threadIdx.x + blockIdx.x * blockDim.x;
	_gpuExpFunTest[_tid] = _ExpFunction_GPU((FP_PRECISION_TEST)_tid);
	//_gpuExpFunTest[_tid] = expf((FP_PRECISION_TEST)_tid);
	//_gpuExpFunTest[_tid] = exp(0.0001/*f*//**_tid*/);
	//_gpuExpFunTest[_tid] = __expf(0.0001f/**_tid*/);
}

__global__ void CUDATestPTX_ExpTab(FP_PRECISION_TEST *_gpuExpTabTest, FP_PRECISION_TEST *_GPULinearExpTable, FP_PRECISION_TEST *_gpuRandFP)
{
	int _tid = threadIdx.x + blockIdx.x * blockDim.x;
	int _num = 1000,i,_x1;
	FP_PRECISION _sum, _exp, _x2,_x3;
	_sum = (FP_PRECISION)(0.0);
	for (i = 0; i < _num; i++) {
		//_x2 = ((FP_PRECISION)(_tid+i))*(FP_PRECISION)(9.14017E-06);
		_x2 = _gpuRandFP[_tid * _num + i];
		_x1 = (int)(_x2);
		_x3 = _x2 - _x1;
		_exp = _GPULinearExpTable[_x1] + _x3 * (_GPULinearExpTable[_x1 + 1] - _GPULinearExpTable[_x1]);
		_sum += _exp;
	}
	_gpuExpTabTest[_tid] = _sum;
}


FP_PRECISION *CPUGenerateRandFP(int _blockDim,int _gridDim,int _num)
{
	int i;
	/*unsigned long*/int _dim = _blockDim * _gridDim * _num;
	FP_PRECISION *_randFP = NULL;
	_randFP = (FP_PRECISION *)calloc(_dim, sizeof(FP_PRECISION));
	//srand((unsigned)time(NULL));
	for (i = 0; i < _dim; i++) {
		_randFP[i] = (FP_PRECISION)(rand() % 18)+ (FP_PRECISION)(rand() % 10)*(FP_PRECISION)(0.1) + (FP_PRECISION)(rand() % 10)*(FP_PRECISION)(0.01)+\
			+ (FP_PRECISION)(rand() % 10)*(FP_PRECISION)(0.001) + (FP_PRECISION)(rand() % 10)*(FP_PRECISION)(0.0001)+\
			+ (FP_PRECISION)(rand() % 10)*(FP_PRECISION)(0.00001) + (FP_PRECISION)(rand() % 10)*(FP_PRECISION)(0.000001);
		//_randFP[i] = (FP_PRECISION)((i + 1)*18)/(FP_PRECISION)(_blockDim*_gridDim*_num);
	}
	return _randFP;
}

#ifdef GPU_STREAM_USED
void GPUStreamInitialize()
{
	int i,_iquo,_irem,_add, /*_totNumLongTracks,*/ _numGroups, _numPolarAng, _numAzimAng,_azim;
	int _GPUNumStreams,totThread;
	_GPUNumStreams = GPUNumStreams;
	//_totNumLongTracks = NumLTs_HS[0] * 2;
	_numGroups = NumGroups;
	_numPolarAng = NumPolarAng[0];
	_numAzimAng = NumAzimAng[0];

	AzimBound_Stream = (int *)calloc(_GPUNumStreams + 1, sizeof(int));
	_iquo = _numAzimAng / _GPUNumStreams;
	for (i = 0; i < _GPUNumStreams + 1; i++) {
		if (i == 0) {
			AzimBound_Stream[0] = 0;
		}
		else if (i > 0 && i < _GPUNumStreams) {
			AzimBound_Stream[i] = AzimBound_Stream[i - 1] + _iquo;
		}
		else {
			AzimBound_Stream[i] = _numAzimAng;
		}
	}

	NumLTsInStream = (int *)calloc(_GPUNumStreams, sizeof(int));
	//_iquo = _totNumLongTracks / _GPUNumStreams;
	//_irem = _totNumLongTracks % _GPUNumStreams;
	for (i = 0; i < _GPUNumStreams; i++) {//NumLongTracks
		NumLTsInStream[i] = 0;
		for (_azim = AzimBound_Stream[i]; _azim < AzimBound_Stream[i + 1]; _azim++) {
			NumLTsInStream[i] += NumLongTracks[0][_azim];
		}
		//if (i != (_GPUNumStreams - 1)) {
		//	NumLTsInStream[i] = _iquo;
		//}
		//else {
		//	NumLTsInStream[i] = _iquo + _irem;
		//}
	}

	GPUStreams = (cudaStream_t *)malloc(_GPUNumStreams * sizeof(cudaStream_t));
	for (i = 0; i < _GPUNumStreams; i++) {
		cudaStreamCreate(&GPUStreams[i]);
	}

	StartPosInLTs_Stream = (int *)calloc(_GPUNumStreams, sizeof(int));
	_add = 0;
	for (i = 0; i < _GPUNumStreams; i++) {
		StartPosInLTs_Stream[i] = _add;
		_add += NumLTsInStream[i];
	}

	StartPosInFlux_Stream = (int *)calloc(_GPUNumStreams, sizeof(int));
	_add = 0;
	for (i = 0; i < _GPUNumStreams; i++) {
		StartPosInFlux_Stream[i] = _add * _numGroups*_numPolarAng;
		_add += NumLTsInStream[i];
	}

	GPUNumBlocks_Stream = (int *)calloc(_GPUNumStreams, sizeof(int));
	for (i = 0; i < _GPUNumStreams; i++) {
		totThread = NumLTsInStream[i] * _numGroups;
		GPUNumBlocks_Stream[i] = (totThread - 1) / GPUBlockDim + 1;
	}


}

void GPUTransportSweep_GCMemcpy_Streams()
{
// 	int i, _GPUNumStreams, *_numLTsInStream, *_startPosInFlux_Stream, _numGroups, _numPolarAng,_azim,_lay;
// 	cudaStream_t *_GPUStreams = NULL;
// 	_GPUNumStreams = GPUNumStreams;
// 	_numLTsInStream = NumLTsInStream;
// 	_startPosInFlux_Stream = StartPosInFlux_Stream;
// 	_numGroups = NumGroups;
// 	_numPolarAng = NumPolarAng[0];
// 	_GPUStreams = GPUStreams;
// 	_lay = 0;

// 	GPUVectorInit_1DFP_Flux<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(GPUScalarFlux, TotalNumFSRs_subDom*NumGroups, 0.0);

// 	//GPUCopyBoundAngFluxDeviceToDevice();

// 	for (i = 0; i < _GPUNumStreams; i++) {
// 		if (NumProcs > 1) {
// #ifndef GC_COPY_AZIM
// 			cudaMemcpyAsync((void *)&GPUBoundaryFlux[_startPosInFlux_Stream[i]], (void*)&CGBoundaryFlux[_startPosInFlux_Stream[i]], \
// 				_numLTsInStream[i] * _numGroups*_numPolarAng * sizeof(FP_PRECISION), cudaMemcpyHostToDevice, _GPUStreams[i]);
// #endif // !GC_COPY_AZIM

// #ifdef GC_COPY_AZIM
// 			for (_azim = AzimBound_Stream[i]; _azim < AzimBound_Stream[i+1]; _azim++) {
// 				if (GCCopy_NumDataInAngF[_lay][_azim] > 0) {
// 					cudaMemcpyAsync((void*)&GPUBoundaryFlux[GCCopy_StartPInAngF[_lay][_azim]], (void*)&CGBoundaryFlux[GCCopy_StartPInAngF[_lay][_azim]], \
// 						GCCopy_NumDataInAngF[_lay][_azim] * sizeof(FP_PRECISION), cudaMemcpyHostToDevice, _GPUStreams[i]);
// 				}
// 			}
// #endif // GC_COPY_AZIM
// 		}

// 		GPUTransportSweepOnDevice_group_diam_oneDir_AngFReg_Stream<<<GPUNumBlocks_Stream[i], GPUBlockDim, 0, _GPUStreams[i]>>>(GPUScalarFlux, \
// 			GPUMOCSource, GPUBoundaryFlux, GPUStartFlux, GPULTsIndex_AzimAngID_Total, GPULTsIndex_LTID_Total, GPULongTracksInfo_Total, \
// 			GPUTracksInfoInLTs_Total, GPUTrackIndexInLTs_Total, 0, NumPinsInSubDomPerLayer, GPUCellRTMType_SubDom, GPUIndexTrackInRTM_SumAzim_Total, \
// 			GPUNumSegsInTrack_Total, GPUSegInfoInTrack_Total, GPUSumNumSegsInRTMTrack_2PI, GPUSumNumFSRsBasedGlobCellID, GPUOneOverFSRsAreaInGlob, \
// 			GPUSumNumFSRsInRTM, GPUMacroXS_t, GPUNumLTsSum, GPUNumLongTracks, GPUMPIFlux, StartPosInLTs_Stream[i], StartPosInLTs_Stream[i]+NumLTsInStream[i]/* GPUStartPosInLTs_Stream[i]*/);

// 		if (NumProcs > 1) {
// 			cudaMemcpyAsync((void *)&CGLTInciAngFlux[_startPosInFlux_Stream[i]], (void*)&GPUMPIFlux[_startPosInFlux_Stream[i]], \
// 				_numLTsInStream[i] * _numGroups*_numPolarAng * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost, _GPUStreams[i]);
// 		}
// 	}
// 	//cudaDeviceSynchronize();
// #if defined(M_2DMOC)
// 	GPUAddSourceToScalarFlux();
// #endif // defined(M_2DMOC)
// #ifndef CPUMPI_GPUTS_ASYNCH
// 	cudaDeviceSynchronize();
// #endif // !CPUMPI_GPUTS_ASYNCH
}

#endif // GPU_STREAM_USED

#if defined(M_MG_CMFD)
/**
 * @ Function: GPUCMFD_AllocMem()
 *   introduction: allocate the memory of M_MG_CMFD arraies on device
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2019/08/14, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
void GPUCMFD_AllocMem(int _procID, size_t *_GPUPreMemMGCMFD, size_t *_GPUPreMemCGCMFD, M_INT32 _MOC_NumPinsInSubDom,\
	M_INT32 _NumGroups, M_INT32 _NumCMFDGroup, M_INT32 _MOCNumLayersZ, M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y,\
	M_INT32 *_CMFDGroupBound, FP_PRECISION *_CMFD_NodeVol, M_INT32 *_NumSubDomains, M_INT32 _MGCMFD_MOCParCurrInnerBound_Num,\
	M_INT32 *_MGCMFD_MOCParCurrInnerBound_Index, M_INT32 *_MGCMFD_MOCParCurrInnerBound_Dir, M_INT32 *_CMFDComm_TarProc, \
	M_INT32 _NumPolarAng, M_INT32 _MGCMFDData_MPIComm_Num, M_INT32 *_MGCMFDData_MPIComm_SumInd, M_INT32 *_MGCMFDMPI_InBoundSourCoeff_Index, \
	M_INT32 *_MGCMFDMPI_InBoundSourCoeff_Dir, FP_PRECISION *_MGCMFDMPI_InBoundSourCoeff_GeoDim, M_INT32 *_MGCMFDData_MPIComm_Index, \
	M_INT32 _CGCMFDData_MPIComm_Num, M_INT32 *_CGCMFDMPI_InBoundSourCoeff_Index, M_INT32 *_CGCMFDMPI_InBoundSourCoeff_Dir, \
	FP_PRECISION *_CGCMFDMPI_InBoundSourCoeff_GeoDim, M_INT32 *_CGCMFDData_MPIComm_Index, M_INT32 _NumCoarseCMFDGroup, \
	M_INT32 *_CGCMFDGroupBound, M_INT32 *_CGCMFDData_MPIComm_SumInd, M_INT32 _MGCMFD_LinearSolver, M_INT32 _CGCMFD_LinearSolver, \
	FILE *_LogFile, M_INT32 _GmresKrylovDim_MG, M_INT32 _GmresKrylovDim_CG, M_INT32 _OUTPUT_CMFDGroupConstant, M_INT32 _OUTPUT_CoarseMesh_GroupConstant, int _isTransient, M_INT32 _NumProcs)
{
	int _numPinsTimesNumCMFDGroup, _dimXs_s, _dimSurfX, _dimSurfY, _dimSurfZ;
	size_t _dim;

	_numPinsTimesNumCMFDGroup = _MOC_NumPinsInSubDom * _NumCMFDGroup;
    _dimXs_s = _numPinsTimesNumCMFDGroup * _NumCMFDGroup;

    if (_procID==0) {
        printf("\n [GPU-INFO ]  Initializing Arraies of CMFD on GPU ...");
		fprintf(_LogFile, "\n [GPU-INFO ]  Initializing Arraies of CMFD on GPU ...");
    }

    cudaMemcpyToSymbol(GPUNumMGCMFDGroups, (void*)&_NumCMFDGroup, sizeof(int), 0, cudaMemcpyHostToDevice);
	_dim = (size_t)((_NumCMFDGroup+1)) * sizeof(int);
	cudaMalloc((void**)&GPU_MGCMFDGroupBound, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	cudaMemcpy((void*)GPU_MGCMFDGroupBound, (void*)_CMFDGroupBound, _dim, cudaMemcpyHostToDevice);//

	_dim = (size_t)(_MOC_NumPinsInSubDom) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUCMFDNodeVol, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	cudaMemcpy((void*)GPUCMFDNodeVol, (void*)_CMFD_NodeVol, _dim, cudaMemcpyHostToDevice);

	/* allocate memory */
	_dim = (size_t)(_numPinsTimesNumCMFDGroup) * sizeof(FP_PRECISION);
    cudaMalloc((void**)&GPUCMFDXs_t, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	if (_OUTPUT_CMFDGroupConstant == 1 || _OUTPUT_CoarseMesh_GroupConstant == 1) {
		cudaMalloc((void**)&GPUCMFDXs_a, _dim);
		_GPUPreMemMGCMFD[0] += _dim;
		cudaMalloc((void**)&GPUCMFDXs_f, _dim);
		_GPUPreMemMGCMFD[0] += _dim;	
	}
    cudaMalloc((void**)&GPUCMFDXs_nf, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
    cudaMalloc((void**)&GPUCMFD_Chi, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	_dim = (size_t)(_dimXs_s) * sizeof(FP_PRECISION);
    cudaMalloc((void**)&GPUCMFDXs_s, _dim);
	_GPUPreMemMGCMFD[0] += _dim;

    if (_isTransient == 1) {
        cudaMalloc((void**) &GPUCMFDTranCoeffA, _dim);
        _GPUPreMemMGCMFD[0] += _dim;
        cudaMalloc((void**) &GPUCMFDTranCoeffB, _dim);
        _GPUPreMemMGCMFD[0] += _dim;
        cudaMalloc((void**) &GPUCMFDTranCoeffC, _dim);
        _GPUPreMemMGCMFD[0] += _dim;
    }

	_dim = (size_t)(_numPinsTimesNumCMFDGroup) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUCMFDHetScalFlux, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	cudaMalloc((void**)&GPUCMFDNodeDiffCoeff, _dim);
	_GPUPreMemMGCMFD[0] += _dim;

	_dimSurfX = _MOCNumLayersZ * (_MOCNumPinsInSubDom_X + 1) * _MOCNumPinsInSubDom_Y * _NumCMFDGroup;
    _dimSurfY = _MOCNumLayersZ * _MOCNumPinsInSubDom_X * (_MOCNumPinsInSubDom_Y + 1) * _NumCMFDGroup;
	_dimSurfZ = (_MOCNumLayersZ + 1) * _MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y * _NumCMFDGroup;
	_dim = (size_t)(_dimSurfX) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUCMFDSurfDiffCoeff_X, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	cudaMalloc((void**)&GPUCMFDSurfNDF_X, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	cudaMalloc((void**)&GPUMOCtoCMFD_ParCurr_X_Pos, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
    cudaMalloc((void**)&GPUMOCtoCMFD_ParCurr_X_Neg, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	cudaMalloc((void**)&GPUMOCtoCMFD_NetCurr_X, _dim);
	_GPUPreMemMGCMFD[0] += _dim;

	_dim = (size_t)(_dimSurfY) * sizeof(FP_PRECISION);
    cudaMalloc((void**)&GPUCMFDSurfDiffCoeff_Y, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	cudaMalloc((void**)&GPUCMFDSurfNDF_Y, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	cudaMalloc((void**)&GPUMOCtoCMFD_ParCurr_Y_Pos, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
    cudaMalloc((void**)&GPUMOCtoCMFD_ParCurr_Y_Neg, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	cudaMalloc((void**)&GPUMOCtoCMFD_NetCurr_Y, _dim);
	_GPUPreMemMGCMFD[0] += _dim;

	_dim = (size_t)(_dimSurfZ) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUCMFDSurfDiffCoeff_Z, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
    cudaMalloc((void**)&GPUCMFDSurfNDF_Z, _dim);
	_GPUPreMemMGCMFD[0] += _dim;

#if defined(M_3DMOCex)
    _dim = (size_t)(_dimSurfZ) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUMOCtoCMFD_NetCurr_Z, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
#endif // defined(M_3DMOCex)

#if defined(M_YQZ_CMFD_Damp)
    _dim = (size_t)(_dimSurfX) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUCMFDSurfNDF_XOld, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	_dim = (size_t)(_dimSurfY) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUCMFDSurfNDF_YOld, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	_dim = (size_t)(_dimSurfZ) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUCMFDSurfNDF_ZOld, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
#endif //defined(M_YQZ_CMFD_Damp)

	_dimSurfX = _MOCNumLayersZ * (_MOCNumPinsInSubDom_X + 1) * _MOCNumPinsInSubDom_Y * _NumGroups;
    _dimSurfY = _MOCNumLayersZ * _MOCNumPinsInSubDom_X * (_MOCNumPinsInSubDom_Y + 1) * _NumGroups;
	_dimSurfZ = (_MOCNumLayersZ + 1) * _MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y * _NumGroups;
	_dim = (size_t)(_dimSurfX) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUMOCParCurr_X_Pos, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
    cudaMalloc((void**)&GPUMOCParCurr_X_Neg, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	_dim = (size_t)(_dimSurfY) * sizeof(FP_PRECISION);
    cudaMalloc((void**)&GPUMOCParCurr_Y_Pos, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	cudaMalloc((void**)&GPUMOCParCurr_Y_Neg, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
#if defined(M_3DMOCex)
    _dim = (size_t)(_dimSurfZ) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUMOCNetCurr_Z, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
#endif // defined(M_3DMOCex)

    _dim = (size_t)(_numPinsTimesNumCMFDGroup) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUCMFDTotalSource, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	cudaMalloc((void**)&GPUCMFDScalarFlux, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	cudaMalloc((void**)&GPUCMFDScalarFlux_Old, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	cudaMalloc((void**)&GPUCMFDScalarFlux_Old_LS, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	cudaMalloc((void**)&GPUCMFDFluxRelErr, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	cudaMalloc((void**)&GPUCMFDNiuFissRate, _dim);
	_GPUPreMemMGCMFD[0] += _dim;

	_dim = (size_t)(_MOC_NumPinsInSubDom) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPU_MGCMFDTotalFissQ, _dim);
	_GPUPreMemMGCMFD[0] += _dim;

	if (_MGCMFD_LinearSolver == 2) { /* pgmres solver*/
		GPU_MGGmres_blocksPerGrid = ((_numPinsTimesNumCMFDGroup + GPU_Gmres_threadsPerBlock - 1) / GPU_Gmres_threadsPerBlock);
		_dim = (size_t)((_numPinsTimesNumCMFDGroup)) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPU_MGGMRES_r0, _dim);
		cudaMalloc((void**)&GPU_MGGMRES_r0Z, _dim);
		CUDA_CheckError((char *)("GPUCMFD_AllocMem-cudaMalloc-GPU_MGGMRES_r0"));
		_GPUPreMemMGCMFD[0] += _dim;
		_dim = (size_t)((GPU_MGGmres_blocksPerGrid)) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPU_MGGMRES_res, _dim);
        _GPUPreMemMGCMFD[0] += _dim;
		cudaMalloc((void**)&GPU_MGGMRES_beta, sizeof(FP_PRECISION));
		_GPUPreMemMGCMFD[0] += sizeof(FP_PRECISION);
		_dim = (size_t)((_numPinsTimesNumCMFDGroup*_GmresKrylovDim_MG)) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPU_MGGMRES_u_base, _dim);
		_GPUPreMemMGCMFD[0] += _dim;
		_dim = (size_t)((_GmresKrylovDim_MG + 1)) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPU_MGGMRES_g, _dim);
        _GPUPreMemMGCMFD[0] += _dim;
		cudaMalloc((void**)&GPU_MGGMRES_e, _dim);
		_GPUPreMemMGCMFD[0] += _dim;
		_dim = (size_t)(((_GmresKrylovDim_MG + 1)*_GmresKrylovDim_MG)) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPU_MGGMRES_Hm, _dim);
		_GPUPreMemMGCMFD[0] += _dim;
		_dim = (size_t)((_numPinsTimesNumCMFDGroup)) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPU_MGGMRES_w, _dim);
		_GPUPreMemMGCMFD[0] += _dim;
		_dim = (size_t)((_GmresKrylovDim_MG)) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPU_MGGMRES_y, _dim);
		_GPUPreMemMGCMFD[0] += _dim;
		_dim = (size_t)(((_GmresKrylovDim_MG + 1)*(_GmresKrylovDim_MG + 1))) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPU_MGGMRES_Wm, _dim);
        _GPUPreMemMGCMFD[0] += _dim;
	}

	//
	cudaMalloc((void**)&GPU_MGCMFDNegFluxCount, sizeof(float));
	_GPUPreMemMGCMFD[0] += sizeof(float);

	_dim = (size_t)(3) * sizeof(int);
	cudaMalloc((void**)&GPUNumSubDomains, _dim);
	_GPUPreMemMGCMFD[0] += _dim;
	cudaMemcpy((void*)GPUNumSubDomains, (void*)_NumSubDomains, _dim, cudaMemcpyHostToDevice);

#if defined(MPI_USED)
	if (_NumProcs > 1) {//
		_dim = (size_t)(_MGCMFD_MOCParCurrInnerBound_Num) * sizeof(int);
		cudaMalloc((void**)&GPUMGCMFD_MOCParCurrInnerBound_Index, _dim);
	    _GPUPreMemMGCMFD[0] += _dim;
		cudaMalloc((void**)&GPUMGCMFD_MOCParCurrInnerBound_Dir, _dim);
		_GPUPreMemMGCMFD[0] += _dim;
		cudaMemcpy((void*)GPUMGCMFD_MOCParCurrInnerBound_Index, (void*)_MGCMFD_MOCParCurrInnerBound_Index, _dim, cudaMemcpyHostToDevice);
		cudaMemcpy((void*)GPUMGCMFD_MOCParCurrInnerBound_Dir, (void*)_MGCMFD_MOCParCurrInnerBound_Dir, _dim, cudaMemcpyHostToDevice);
		_dim = (size_t)(6) * sizeof(int);
		cudaMalloc((void**)&GPUCMFDComm_TarProc, _dim);
		_GPUPreMemMGCMFD[0] += _dim;
		cudaMemcpy((void*)GPUCMFDComm_TarProc, (void*)_CMFDComm_TarProc, _dim, cudaMemcpyHostToDevice);
		//
		_dim = (size_t)(_NumPolarAng) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPUMGCMFD_MOCParCurrInnerBound_AngF, _dim);
	    _GPUPreMemMGCMFD[0] += _dim;

		_dim = (size_t)(_MGCMFDData_MPIComm_Num) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPUMGCMFDFlux_MPIRecv, _dim);
	    _GPUPreMemMGCMFD[0] += _dim;
		cudaMalloc((void**)&GPUMGCMFDNodeD_MPIRecv, _dim);
	    _GPUPreMemMGCMFD[0] += _dim;

		_dim = (size_t)(6) * sizeof(int);
		cudaMalloc((void**)&GPUMGCMFDData_MPIComm_SumInd, _dim);
	    _GPUPreMemMGCMFD[0] += _dim;
		cudaMemcpy((void*)GPUMGCMFDData_MPIComm_SumInd, (void*)_MGCMFDData_MPIComm_SumInd, _dim, cudaMemcpyHostToDevice);
	    _GPUPreMemMGCMFD[0] += _dim;

		_dim = (size_t)(_MGCMFDData_MPIComm_Num) * sizeof(int);
		cudaMalloc((void**)&GPUMGCMFDMPI_InBoundSourCoeff_Index, _dim);
	    _GPUPreMemMGCMFD[0] += _dim;
		cudaMalloc((void**)&GPUMGCMFDMPI_InBoundSourCoeff_Dir, _dim);
		_GPUPreMemMGCMFD[0] += _dim;
		cudaMemcpy((void*)GPUMGCMFDMPI_InBoundSourCoeff_Index, (void*)_MGCMFDMPI_InBoundSourCoeff_Index, _dim, cudaMemcpyHostToDevice);
		cudaMemcpy((void*)GPUMGCMFDMPI_InBoundSourCoeff_Dir, (void*)_MGCMFDMPI_InBoundSourCoeff_Dir, _dim, cudaMemcpyHostToDevice);

	    _dim = (size_t)(_MGCMFDData_MPIComm_Num) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPUMGCMFDMPI_InBoundSourCoeff_GeoDim, _dim);
	    _GPUPreMemMGCMFD[0] += _dim;
		cudaMemcpy((void*)GPUMGCMFDMPI_InBoundSourCoeff_GeoDim, (void*)_MGCMFDMPI_InBoundSourCoeff_GeoDim, _dim, cudaMemcpyHostToDevice);

		cudaMalloc((void**)&GPUMGCMFDMPI_InBoundSourCoeff, _dim);
	    _GPUPreMemMGCMFD[0] += _dim;
		
		_dim = (size_t)(_MGCMFDData_MPIComm_Num) * sizeof(int);
		cudaMalloc((void**)&GPUMGCMFDData_MPIComm_Index, _dim);
	    _GPUPreMemMGCMFD[0] += _dim;
		cudaMemcpy((void*)GPUMGCMFDData_MPIComm_Index, (void*)_MGCMFDData_MPIComm_Index, _dim, cudaMemcpyHostToDevice);
	}
#endif // defined(MPI_USED)

#if defined(M_CG_CMFD)
   _dimSurfX = _MOCNumLayersZ * (_MOCNumPinsInSubDom_X + 1) * _MOCNumPinsInSubDom_Y * _NumCMFDGroup;
   _dimSurfY = _MOCNumLayersZ * _MOCNumPinsInSubDom_X * (_MOCNumPinsInSubDom_Y + 1) * _NumCMFDGroup;
   _dimSurfZ = (_MOCNumLayersZ + 1) * _MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y * _NumCMFDGroup;

    _dim = (size_t)(_dimSurfX) * sizeof(FP_PRECISION);
    cudaMalloc((void**)&GPU_MGCMFDNetCurr_X, _dim);
    _GPUPreMemCGCMFD[0] += _dim;
	_dim = (size_t)(_dimSurfY) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPU_MGCMFDNetCurr_Y, _dim);
	_GPUPreMemCGCMFD[0] += _dim;

#if defined(M_3DMOCex)
    _dim = (size_t)(_dimSurfZ) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPU_MGCMFDNetCurr_Z, _dim);
	_GPUPreMemCGCMFD[0] += _dim;
	
    if (NumProcs > 1) {
		_dim = (size_t)(_CGCMFDData_MPIComm_Num) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPUCGCMFDFlux_MPIRecv, _dim);
	    _GPUPreMemCGCMFD[0] += _dim;
		cudaMalloc((void**)&GPUCGCMFDNodeD_MPIRecv, _dim);
	    _GPUPreMemCGCMFD[0] += _dim;
		cudaMalloc((void**)&GPUCGCMFDMPI_InBoundSourCoeff, _dim);
		_GPUPreMemCGCMFD[0] += _dim;
		
		_dim = (size_t)(_CGCMFDData_MPIComm_Num) * sizeof(int);
		cudaMalloc((void**)&GPUCGCMFDMPI_InBoundSourCoeff_Index, _dim);
	    _GPUPreMemCGCMFD[0] += _dim;
		cudaMalloc((void**)&GPUCGCMFDMPI_InBoundSourCoeff_Dir, _dim);
		_GPUPreMemCGCMFD[0] += _dim;
		cudaMemcpy((void*)GPUCGCMFDMPI_InBoundSourCoeff_Index, (void*)_CGCMFDMPI_InBoundSourCoeff_Index, _dim, cudaMemcpyHostToDevice);
		cudaMemcpy((void*)GPUCGCMFDMPI_InBoundSourCoeff_Dir, (void*)_CGCMFDMPI_InBoundSourCoeff_Dir, _dim, cudaMemcpyHostToDevice);
        _dim = (size_t)(_CGCMFDData_MPIComm_Num) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPUCGCMFDMPI_InBoundSourCoeff_GeoDim, _dim);
	    _GPUPreMemCGCMFD[0] += _dim;
		cudaMemcpy((void*)GPUCGCMFDMPI_InBoundSourCoeff_GeoDim, (void*)_CGCMFDMPI_InBoundSourCoeff_GeoDim, _dim, cudaMemcpyHostToDevice);

		_dim = (size_t)(_CGCMFDData_MPIComm_Num) * sizeof(int);
		cudaMalloc((void**)&GPUCGCMFDData_MPIComm_Index, _dim);
	    _GPUPreMemCGCMFD[0] += _dim;
		cudaMemcpy((void*)GPUCGCMFDData_MPIComm_Index, (void*)_CGCMFDData_MPIComm_Index, _dim, cudaMemcpyHostToDevice);
	}

#endif //defined(M_3DMOCex)

	_numPinsTimesNumCMFDGroup = _MOC_NumPinsInSubDom * _NumCoarseCMFDGroup;
	_dimXs_s = _numPinsTimesNumCMFDGroup * _NumCoarseCMFDGroup;
	_dimSurfX = _MOCNumLayersZ * (_MOCNumPinsInSubDom_X + 1) * _MOCNumPinsInSubDom_Y * _NumCoarseCMFDGroup;
    _dimSurfY = _MOCNumLayersZ * _MOCNumPinsInSubDom_X * (_MOCNumPinsInSubDom_Y + 1) * _NumCoarseCMFDGroup;
	_dimSurfZ = (_MOCNumLayersZ + 1) * _MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y * _NumCoarseCMFDGroup;

   cudaMemcpyToSymbol(GPUNumCGCMFDGroups, (void*)&_NumCoarseCMFDGroup, sizeof(int), 0, cudaMemcpyHostToDevice);
   _dim = (size_t)((_NumCoarseCMFDGroup + 1)) * sizeof(int);
   cudaMalloc((void**)&GPU_CGCMFDGroupBound, _dim);
   _GPUPreMemCGCMFD[0] += _dim;//
   cudaMemcpy((void*)GPU_CGCMFDGroupBound, (void*)_CGCMFDGroupBound, _dim, cudaMemcpyHostToDevice);

    _dim = (size_t)(_numPinsTimesNumCMFDGroup) * sizeof(FP_PRECISION);
    cudaMalloc((void**)&GPU_CGCMFDHetScalFlux, _dim);
	_GPUPreMemCGCMFD[0] += _dim;
	cudaMalloc((void**)&GPU_CGCMFDXs_t, _dim);
	_GPUPreMemCGCMFD[0] += _dim;
	cudaMalloc((void**)&GPU_CGCMFDXs_nf, _dim);
	_GPUPreMemCGCMFD[0] += _dim;
	cudaMalloc((void**)&GPU_CGCMFD_Chi, _dim);
	_GPUPreMemCGCMFD[0] += _dim;
	cudaMalloc((void**)&GPU_CGCMFDNodeDiffCoeff, _dim);
	_GPUPreMemCGCMFD[0] += _dim;

	_dim = (size_t)(_dimXs_s) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPU_CGCMFDXs_s, _dim);
	_GPUPreMemCGCMFD[0] += _dim;
	
	_dim = (size_t)(_dimSurfX) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPU_CGMGtoCGCMFD_NetCurr_X, _dim);
	_GPUPreMemCGCMFD[0] += _dim;
	cudaMalloc((void**)&GPU_CGCMFDSurfDiffCoeff_X, _dim);
	_GPUPreMemCGCMFD[0] += _dim;
	cudaMalloc((void**)&GPU_CGCMFDSurfNDF_X, _dim);
    _GPUPreMemCGCMFD[0] += _dim;
	
	_dim = (size_t)(_dimSurfY) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPU_CGMGtoCGCMFD_NetCurr_Y, _dim);
	_GPUPreMemCGCMFD[0] += _dim;
	cudaMalloc((void**)&GPU_CGCMFDSurfDiffCoeff_Y, _dim);
	_GPUPreMemCGCMFD[0] += _dim;
    cudaMalloc((void**)&GPU_CGCMFDSurfNDF_Y, _dim);
    _GPUPreMemCGCMFD[0] += _dim;
#if defined(M_3DMOCex)
    _dim = (size_t)(_dimSurfZ) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPU_CGMGtoCGCMFD_NetCurr_Z, _dim);
	_GPUPreMemCGCMFD[0] += _dim;
#endif //defined(M_3DMOCex)
	
    _dim = (size_t)(_dimSurfZ) * sizeof(FP_PRECISION);
    cudaMalloc((void**)&GPU_CGCMFDSurfDiffCoeff_Z, _dim);
    _GPUPreMemCGCMFD[0] += _dim;
    cudaMalloc((void**)&GPU_CGCMFDSurfNDF_Z, _dim);
    _GPUPreMemCGCMFD[0] += _dim;

	_dim = (size_t)(_numPinsTimesNumCMFDGroup) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPU_CGCMFDTotalSource, _dim);
	_GPUPreMemCGCMFD[0] += _dim;
	cudaMalloc((void**)&GPU_CGCMFDScalarFlux, _dim);
	_GPUPreMemCGCMFD[0] += _dim;
	cudaMalloc((void**)&GPU_CGCMFDScalarFlux_Old, _dim);
	_GPUPreMemCGCMFD[0] += _dim;
	cudaMalloc((void**)&GPU_CGCMFDScalarFlux_Old_LS, _dim);
	_GPUPreMemCGCMFD[0] += _dim;

	cudaMalloc((void**)&GPU_CGCMFDFluxRelErr, _dim);
	_GPUPreMemCGCMFD[0] += _dim;
	cudaMalloc((void**)&GPU_CGCMFDNiuFissRate, _dim);
	_GPUPreMemCGCMFD[0] += _dim;

	_dim = (size_t)(_MOC_NumPinsInSubDom) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPU_CGCMFDTotalFissQ, _dim);
	_GPUPreMemCGCMFD[0] += _dim;

	if (_CGCMFD_LinearSolver == 2) {
		GPU_CGGmres_blocksPerGrid=((_numPinsTimesNumCMFDGroup+GPU_Gmres_threadsPerBlock-1)/GPU_Gmres_threadsPerBlock);
		_dim = (size_t)((_numPinsTimesNumCMFDGroup)) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPU_CGGMRES_r0, _dim);
		_GPUPreMemCGCMFD[0] += _dim;
		_dim = (size_t)((GPU_CGGmres_blocksPerGrid)) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPU_CGGMRES_res, _dim);
	    _GPUPreMemCGCMFD[0] += _dim;
		cudaMalloc((void**)&GPU_CGGMRES_beta,sizeof(FP_PRECISION));
		_GPUPreMemCGCMFD[0] += sizeof(FP_PRECISION);
		_dim = (size_t)((_numPinsTimesNumCMFDGroup*_GmresKrylovDim_CG)) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPU_CGGMRES_u_base, _dim);
		_GPUPreMemCGCMFD[0] += _dim;
		_dim = (size_t)((_GmresKrylovDim_CG+1)) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPU_CGGMRES_g,_dim);
	    _GPUPreMemCGCMFD[0] += _dim;
		cudaMalloc((void**)&GPU_CGGMRES_e,_dim);
		_GPUPreMemCGCMFD[0] += _dim;
		_dim = (size_t)(((_GmresKrylovDim_CG+1)*_GmresKrylovDim_CG)) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPU_CGGMRES_Hm, _dim);
		_GPUPreMemCGCMFD[0] += _dim;
		_dim = (size_t)((_numPinsTimesNumCMFDGroup)) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPU_CGGMRES_w, _dim);
		_GPUPreMemCGCMFD[0] += _dim;
		_dim = (size_t)((_GmresKrylovDim_CG)) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPU_CGGMRES_y, _dim);
		_GPUPreMemCGCMFD[0] += _dim;
		_dim = (size_t)(((_GmresKrylovDim_CG+1)*(_GmresKrylovDim_CG+1))) * sizeof(FP_PRECISION);
		cudaMalloc((void**)&GPU_CGGMRES_Wm, _dim);
	    _GPUPreMemCGCMFD[0] += _dim;
	}

	cudaMalloc((void**)&GPU_CGCMFDNegFluxCount, sizeof(float));
	_GPUPreMemCGCMFD[0] += sizeof(float);

#if defined(MPI_USED)
    if (NumProcs > 1) {
		_dim = (size_t)(6) * sizeof(int);
		cudaMalloc((void**)&GPUCGCMFDData_MPIComm_SumInd, _dim);
	    _GPUPreMemCGCMFD[0] += _dim;
	    cudaMemcpy((void*)GPUCGCMFDData_MPIComm_SumInd, (void*)_CGCMFDData_MPIComm_SumInd, _dim, cudaMemcpyHostToDevice);
	}
#endif // defined(MPI_USED)
#endif // defined(M_CG_CMFD)

#ifdef CUDAErrorCheck
	CUDAStatus = cudaGetLastError();
	if (CUDAStatus != cudaSuccess) {
		printf("\n [GPU-ERROR]  Lunch Failed: after with Info: %s\n", cudaGetErrorString(CUDAStatus));
		exit(69);
	}
#endif // CUDAErrorCheck
}

/**
 * @ Function: GPUCMFD_InitMOCCurr()
 *   introduction: allocate the memory of M_MG_CMFD arraies on device
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2019/08/14, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
extern "C"
void GPUCMFD_InitMOCCurr(FP_PRECISION _partCurr2D, FP_PRECISION _netCurrZ, M_INT32 _MOCNumLayersZ, \
	M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, M_INT32 _NumGroups)
{
	int _numEleSurf_X, _numEleSurf_Y;
#if defined(M_3DMOCex)
	int _numEleSurf_Z;
#endif

	_numEleSurf_X = _MOCNumLayersZ * (_MOCNumPinsInSubDom_X + 1) * _MOCNumPinsInSubDom_Y * _NumGroups;
	_numEleSurf_Y = _MOCNumLayersZ * _MOCNumPinsInSubDom_X * (_MOCNumPinsInSubDom_Y + 1) * _NumGroups;

	GPUVectorInit_FP_I<<<GPUGridDim_MOCPartCurr_X, GPUBlockDim>>>(GPUMOCParCurr_X_Pos, _numEleSurf_X, _partCurr2D);
	GPUVectorInit_FP_I<<<GPUGridDim_MOCPartCurr_X, GPUBlockDim>>>(GPUMOCParCurr_X_Neg, _numEleSurf_X, _partCurr2D);
	GPUVectorInit_FP_I<<<GPUGridDim_MOCPartCurr_Y, GPUBlockDim>>>(GPUMOCParCurr_Y_Pos, _numEleSurf_Y, _partCurr2D);
	GPUVectorInit_FP_I<<<GPUGridDim_MOCPartCurr_Y, GPUBlockDim>>>(GPUMOCParCurr_Y_Neg, _numEleSurf_Y, _partCurr2D);
#if defined(M_3DMOCex)
    _numEleSurf_Z = (_MOCNumLayersZ + 1) * _MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y * _NumGroups;
	GPUVectorInit_FP_I<<<GPUGridDim_MOCNetCurr_Z, GPUBlockDim>>>(GPUMOCNetCurr_Z, _numEleSurf_Z, _netCurrZ);
#endif // defined(M_3DMOCex)
}


extern "C"
void GPUCMFD_CheckCollapse(int _isFirstInvoke, M_INT32 _OUTPUT_CMFDGroupConstant, M_INT32 _OUTPUT_CoarseMesh_GroupConstant, M_INT32 _MOC_NumPinsInSubDom, M_INT32 _MOCNumLayersZ, \
	M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, M_INT32 _NumCMFDGroup, M_INT32 _CMFD_NumNoneZeroValue_Matrix,\
	FP_PRECISION *_CMFDHetScalFlux, FP_PRECISION *_CMFDXs_t, FP_PRECISION *_CMFDXs_a, FP_PRECISION *_CMFDXs_f, FP_PRECISION *_CMFDXs_nf, FP_PRECISION *_CMFD_Chi, FP_PRECISION *_CMFDXs_s, \
	FP_PRECISION *_CMFDNodeDiffCoeff, FP_PRECISION *_MOCtoCMFD_ParCurr_X_Pos, FP_PRECISION *_MOCtoCMFD_ParCurr_X_Neg, \
	FP_PRECISION *_MOCtoCMFD_ParCurr_Y_Pos, FP_PRECISION *_MOCtoCMFD_ParCurr_Y_Neg, FP_PRECISION *_MGCMFDNetCurr_X, FP_PRECISION *_MGCMFDNetCurr_Y,\
	FP_PRECISION *_CMFDSurfDiffCoeff_X, FP_PRECISION *_CMFDSurfNDF_X, FP_PRECISION *_CMFDSurfDiffCoeff_Y, FP_PRECISION *_CMFDSurfNDF_Y, \
	M_INT32 _MGCMFD_LinearSolver, FP_PRECISION *_MatrixCSR, FP_PRECISION *_MG_ILUMatrix,M_INT32 *_MatrixCSR_RowStartInd, M_INT32 *_MatrixCSR_ColInd, \
	FP_PRECISION *_MGCMFDMPI_InBoundSourCoeff, M_INT32 _MGCMFDData_MPIComm_Num)
{
	if (_OUTPUT_CMFDGroupConstant == 1 || _OUTPUT_CoarseMesh_GroupConstant == 1) {
		int _numEleCmfdNode, _numEleSurf_X, _numEleSurf_Y, _numEleCmfdXSs, _nnz;
		// int i;
		_numEleCmfdNode = _MOC_NumPinsInSubDom * _NumCMFDGroup;
		_numEleCmfdXSs = _numEleCmfdNode * _NumCMFDGroup;
		_numEleSurf_X = _MOCNumLayersZ * (_MOCNumPinsInSubDom_X + 1) * _MOCNumPinsInSubDom_Y * _NumCMFDGroup;
		_numEleSurf_Y = _MOCNumLayersZ * _MOCNumPinsInSubDom_X * (_MOCNumPinsInSubDom_Y + 1) * _NumCMFDGroup;
		_nnz = _CMFD_NumNoneZeroValue_Matrix;

		cudaMemcpy((void*)_CMFDHetScalFlux, (void*)GPUCMFDHetScalFlux, (size_t)(_numEleCmfdNode) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUCMFDHetScalFlux"));

		cudaMemcpy((void*)_CMFDXs_t, (void*)GPUCMFDXs_t, (size_t)(_numEleCmfdNode) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUCMFDXs_t"));

		cudaMemcpy((void*)_CMFDXs_a, (void*)GPUCMFDXs_a, (size_t)(_numEleCmfdNode) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUCMFDXs_a"));

		cudaMemcpy((void*)_CMFDXs_f, (void*)GPUCMFDXs_f, (size_t)(_numEleCmfdNode) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUCMFDXs_f"));

		cudaMemcpy((void*)_CMFDXs_nf, (void*)GPUCMFDXs_nf, (size_t)(_numEleCmfdNode) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUCMFDXs_nf"));

		cudaMemcpy((void*)_CMFD_Chi, (void*)GPUCMFD_Chi, (size_t)(_numEleCmfdNode) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUCMFD_Chi"));

		cudaMemcpy((void*)_CMFDXs_s, (void*)GPUCMFDXs_s, (size_t)(_numEleCmfdXSs) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUCMFDXs_s"));

		cudaMemcpy((void*)_CMFDNodeDiffCoeff, (void*)GPUCMFDNodeDiffCoeff, (size_t)(_numEleCmfdNode) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUCMFDNodeDiffCoeff"));

		cudaMemcpy((void*)_MOCtoCMFD_ParCurr_X_Pos, (void*)GPUMOCtoCMFD_ParCurr_X_Pos, (size_t)(_numEleSurf_X) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUMOCtoCMFD_ParCurr_X_Pos"));
		cudaMemcpy((void*)_MOCtoCMFD_ParCurr_X_Neg, (void*)GPUMOCtoCMFD_ParCurr_X_Neg, (size_t)(_numEleSurf_X) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUMOCtoCMFD_ParCurr_X_Neg"));

		cudaMemcpy((void*)_MOCtoCMFD_ParCurr_Y_Pos, (void*)GPUMOCtoCMFD_ParCurr_Y_Pos, (size_t)(_numEleSurf_Y) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUMOCtoCMFD_ParCurr_Y_Pos"));

		cudaMemcpy((void*)_MOCtoCMFD_ParCurr_Y_Neg, (void*)GPUMOCtoCMFD_ParCurr_Y_Neg, (size_t)(_numEleSurf_Y) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUMOCtoCMFD_ParCurr_Y_Neg"));

		cudaMemcpy((void*)_MGCMFDNetCurr_X, (void*)GPUMOCtoCMFD_NetCurr_X, (size_t)(_numEleSurf_X) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUMOCtoCMFD_NetCurr_X"));
		cudaMemcpy((void*)_MGCMFDNetCurr_Y, (void*)GPUMOCtoCMFD_NetCurr_Y, (size_t)(_numEleSurf_Y) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUMOCtoCMFD_NetCurr_Y"));
		
		cudaMemcpy((void*)_CMFDSurfDiffCoeff_X, (void*)GPUCMFDSurfDiffCoeff_X, (size_t)(_numEleSurf_X) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUCMFDSurfDiffCoeff_X"));
		cudaMemcpy((void*)_CMFDSurfNDF_X, (void*)GPUCMFDSurfNDF_X, (size_t)(_numEleSurf_X) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUCMFDSurfNDF_X"));
		cudaMemcpy((void*)_CMFDSurfDiffCoeff_Y, (void*)GPUCMFDSurfDiffCoeff_Y, (size_t)(_numEleSurf_Y) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUCMFDSurfDiffCoeff_Y"));
		cudaMemcpy((void*)_CMFDSurfNDF_Y, (void*)GPUCMFDSurfNDF_Y, (size_t)(_numEleSurf_Y) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUCMFDSurfNDF_Y"));


		if (GPU_MGCMFDMatrixCSR == NULL) {
			printf("\n GPU_MGCMFDMatrixCSR==NULL");
		}
		cudaMemcpy((void*)_MatrixCSR, (void*)GPU_MGCMFDMatrixCSR, (size_t)(_nnz) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPU_MGCMFDMatrixCSR"));
		if (_MGCMFD_LinearSolver == 2) {
			cudaMemcpy((void*)_MG_ILUMatrix, (void*)GPU_MGCMFDMatrixCSR_ILUFactor, (size_t)(_nnz) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
			CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPU_MGCMFDMatrixCSR_ILUFactor"));
		}

		cudaMemcpy((void*)_MatrixCSR_RowStartInd, (void*)GPU_MGCMFDMatrixCSR_RowStartInd, (size_t)(_numEleCmfdNode+1) * sizeof(int), cudaMemcpyDeviceToHost);
	    cudaMemcpy((void*)_MatrixCSR_ColInd, (void*)GPU_MGCMFDMatrixCSR_ColInd, (size_t)(_nnz) * sizeof(int), cudaMemcpyDeviceToHost);
		
		cudaMemcpy((void*)_MGCMFDMPI_InBoundSourCoeff, (void*)GPUMGCMFDMPI_InBoundSourCoeff, (size_t)(_MGCMFDData_MPIComm_Num) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPU_MGCMFDMatrixCSR_ILUFactor"));
		// for (i=0;i<_numEleCmfdNode;i++) {
		// 	printf("\n GPU_MGCMFDMatrixCSR_Diag[pin=%d, g=%d]=%e",i/_NumCMFDGroup, i%_NumCMFDGroup, MatrixCSR_Diag[i]);
		// }
		// for (i=0;i<_nnz;i++) {
		// 	printf("\n GPU_MGCMFDMatrixCSR[%d]=%e",i, MatrixCSR[i]);
		// }
		// printf("\n MatrixCSR_Diag=%e",1.0/MatrixCSR_Diag[0]);

//#if defined(M_CG_CMFD)
//
//		_numEleCmfdNode = _MOC_NumPinsInSubDom * NumCoarseCMFDGroup;
//		_numEleCmfdXSs = _numEleCmfdNode * NumCoarseCMFDGroup;
//		_numEleSurf_X = _MOCNumLayersZ * (_MOCNumPinsInSubDom_X + 1) * _MOCNumPinsInSubDom_X * NumCoarseCMFDGroup;
//		_numEleSurf_Y = _MOCNumLayersZ * _MOCNumPinsInSubDom_X * (_MOCNumPinsInSubDom_X + 1) * NumCoarseCMFDGroup;
//#if defined(M_CGLS_GMRES)
//		_nnz = CGCMFD_NumNoneZeroValue_Matrix;
//#else
//		_nnz = CGCMFD_NumNoneZeroValueLU_Matrix;
//#endif // defined(M_CGLS_GMRES)
//		cudaMemcpy((void*)CGCMFDHetScalFlux, (void*)GPU_CGCMFDHetScalFlux, _numEleCmfdNode * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
//		// for (i=0;i<_numEleCmfdNode;i++) {
//		// 	printf("\n GPU_CGCMFDHetScalFlux[pin=%d, g=%d]=%e",i/NumCoarseCMFDGroup, i%NumCoarseCMFDGroup, CGCMFDHetScalFlux[i]);
//		// 	fflush(stdout);
//		// }
//
//		cudaMemcpy((void*)CGCMFDXs_t, (void*)GPU_CGCMFDXs_t, _numEleCmfdNode * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
//		// for (i=0;i<_numEleCmfdNode;i++) {
//		// 	printf("\n GPU_CGCMFDXs_t[pin=%d, g=%d]=%e",i/NumCoarseCMFDGroup, i%NumCoarseCMFDGroup, CGCMFDXs_t[i]);
//		// 	fflush(stdout);
//		// }
//
//		cudaMemcpy((void*)CGCMFDXs_nf, (void*)GPU_CGCMFDXs_nf, _numEleCmfdNode * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
//		// for (i=0;i<_numEleCmfdNode;i++) {
//		// 	printf("\n GPUCMFDXs_nf[pin=%d, g=%d]=%e",i/NumCoarseCMFDGroup, i%NumCoarseCMFDGroup, CGCMFDXs_nf[i]);
//		// 	fflush(stdout);
//		// }
//
//		cudaMemcpy((void*)CGCMFD_Chi, (void*)GPU_CGCMFD_Chi, _numEleCmfdNode * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
//		// for (i=0;i<_numEleCmfdNode;i++) {
//		// 	printf("\n GPU_CGCMFD_Chi[pin=%d, g=%d]=%e",i/NumCoarseCMFDGroup, i%NumCoarseCMFDGroup, CGCMFD_Chi[i]);
//		// 	fflush(stdout);
//		// }
//
//		cudaMemcpy((void*)CGCMFDXs_s, (void*)GPU_CGCMFDXs_s, _numEleCmfdXSs * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
//		// for (i=0;i<_numEleCmfdXSs;i++) {
//		// 	printf("\n GPU_CGCMFDXs_s[pin=%d, g=%d -> g1=%d]=%e",i/(NumCoarseCMFDGroup*NumCoarseCMFDGroup), \
//		// 		i%(NumCoarseCMFDGroup*NumCoarseCMFDGroup)/NumCoarseCMFDGroup,\
//		// 		i%(NumCoarseCMFDGroup*NumCoarseCMFDGroup)%NumCoarseCMFDGroup, CGCMFDXs_s[i]);
//		// }
//
//		cudaMemcpy((void*)CGCMFDNodeDiffCoeff, (void*)GPU_CGCMFDNodeDiffCoeff, _numEleCmfdNode * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
//		// for (i=0;i<_numEleCmfdNode;i++) {
//		// 	printf("\n GPU_CGCMFDNodeDiffCoeff[pin=%d, g=%d]=%e",i/NumCoarseCMFDGroup, i%NumCoarseCMFDGroup, CGCMFDNodeDiffCoeff[i]);
//		// 	fflush(stdout);
//		// }
//
//		cudaMemcpy((void*)CGCMFDNetCurr_X, (void*)GPU_CGMGtoCGCMFD_NetCurr_X, _numEleSurf_X * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
//		// for (i=0;i<_numEleSurf_X;i++) {
//		// 	printf("\n GPU_CGMGtoCGCMFD_NetCurr_X[surf=%d, g=%d]=%e",i/NumCoarseCMFDGroup, i%NumCoarseCMFDGroup, CGCMFDNetCurr_X[i]);
//		// }
//		cudaMemcpy((void*)CGCMFDNetCurr_Y, (void*)GPU_CGMGtoCGCMFD_NetCurr_Y, _numEleSurf_Y * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
//		// for (i=0;i<_numEleSurf_Y;i++) {
//		// 	printf("\n GPU_CGMGtoCGCMFD_NetCurr_Y[surf=%d, g=%d]=%e",i/NumCoarseCMFDGroup, i%NumCoarseCMFDGroup, CGCMFDNetCurr_Y[i]);
//		// }
//
//		cudaMemcpy((void*)CGCMFDSurfDiffCoeff_X, (void*)GPU_CGCMFDSurfDiffCoeff_X, _numEleSurf_X * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
//		cudaMemcpy((void*)CGCMFDSurfNDF_X, (void*)GPU_CGCMFDSurfNDF_X, _numEleSurf_X * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
//		// for (i=0;i<_numEleSurf_X;i++) {
//		// 	printf("\n GPU_CGCMFDSurfDiffCoeff_X[surf=%d, g=%d]=%e, GPU_CGCMFDSurfNDF_X=%e",i/NumCoarseCMFDGroup, \
//		//     	i%NumCoarseCMFDGroup, CGCMFDSurfDiffCoeff_X[i], CGCMFDSurfNDF_X[i]);
//		// }
//		cudaMemcpy((void*)CGCMFDSurfDiffCoeff_Y, (void*)GPU_CGCMFDSurfDiffCoeff_Y, _numEleSurf_Y * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
//		cudaMemcpy((void*)CGCMFDSurfNDF_Y, (void*)GPU_CGCMFDSurfNDF_Y, _numEleSurf_Y * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
//		// for (i=0;i<_numEleSurf_Y;i++) {
//		// 	printf("\n GPU_CGCMFDSurfDiffCoeff_X[surf=%d, g=%d]=%e, GPU_CGCMFDSurfNDF_Y=%e",i/NumCoarseCMFDGroup, \
//		//     	i%NumCoarseCMFDGroup, CGCMFDSurfDiffCoeff_Y[i], CGCMFDSurfNDF_Y[i]);
//		// }
//
//		if (_isFirstInvoke) {
//			CGMatrixCSR_Diag = (FP_PRECISION *)calloc(_numEleCmfdNode, sizeof(FP_PRECISION));
//#if defined(M_GMRES_PRE_cuSparseLib)
//			CGMatrixCSR = (FP_PRECISION *)calloc(_nnz, sizeof(FP_PRECISION));
//#endif // defined(M_GMRES_PRE_cuSparseLib)
//#if defined(M_CGLS_GMRES) && defined(M_GMRES_PRE_cuSparseLib)
//			CG_ILUMatrix = (FP_PRECISION *)calloc(_nnz, sizeof(FP_PRECISION));
//#endif // defined(M_CGLS_GMRES) && defined(M_GMRES_PRE_cuSparseLib)
//		}
//		cudaMemcpy((void*)CGMatrixCSR_Diag, (void*)GPU_CGCMFDMatrixCSR_Diag, _numEleCmfdNode * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
//		cudaMemcpy((void*)CGMatrixCSR, (void*)GPU_CGCMFDMatrixCSR, _nnz * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
//#if defined(M_CGLS_GMRES)
//		cudaMemcpy((void*)CG_ILUMatrix, (void*)GPU_CGCMFDMatrixCSR_ILUFactor, _nnz * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
//#endif // defined(M_CGLS_GMRES)
//		// for (i=0;i<_numEleCmfdNode;i++) {
//		// 	printf("\n GPU_CGCMFDMatrixCSR_Diag[pin=%d, g=%d]=%e",i/NumCoarseCMFDGroup, i%NumCoarseCMFDGroup, CGMatrixCSR_Diag[i]);
//		// }
//		// for (i=0;i<CGCMFD_NumNoneZeroValueLU_Matrix;i++) {
//		// 	printf("\n GPU_CGCMFDMatrixCSR[%d]=%e",i, CGMatrixCSR[i]);
//		// }
//
//#endif // defined(M_CG_CMFD)

#ifdef CUDAErrorCheck
		CUDAStatus = cudaGetLastError();
		if (CUDAStatus != cudaSuccess) {
			printf("\n [GPU-ERROR]  Lunch Failed: GPUCMFD_CheckCollapse with Info: %s\n", cudaGetErrorString(CUDAStatus));
			exit(69);
		}
#endif // CUDAErrorCheck
	}
}

extern "C"
void GPUCoarseMeshGroupConst_CheckCollapse(M_INT32 _OUTPUT_CoarseMesh_GroupConstant, M_INT32 _MOC_NumPinsInSubDom, \
	M_INT32 _NumCMFDGroup, FP_PRECISION *_CMFDXs_t, FP_PRECISION *_CMFDXs_a, FP_PRECISION *_CMFDXs_f, FP_PRECISION *_CMFDXs_nf, \
	FP_PRECISION *_CMFD_Chi, FP_PRECISION *_CMFDXs_s)
{
	if (_OUTPUT_CoarseMesh_GroupConstant == 1) {
		int _numEleCmfdNode, _numEleCmfdXSs;
		// int i;
		_numEleCmfdNode = _MOC_NumPinsInSubDom * _NumCMFDGroup;
		_numEleCmfdXSs = _numEleCmfdNode * _NumCMFDGroup;

		cudaMemcpy((void*)_CMFDXs_t, (void*)GPUCMFDXs_t, (size_t)(_numEleCmfdNode) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCoarseMeshGroupConst_CheckCollapse - cudaMemcpy - GPUCMFDXs_t"));

		cudaMemcpy((void*)_CMFDXs_a, (void*)GPUCMFDXs_a, (size_t)(_numEleCmfdNode) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCoarseMeshGroupConst_CheckCollapse - cudaMemcpy - GPUCMFDXs_a"));

		cudaMemcpy((void*)_CMFDXs_f, (void*)GPUCMFDXs_f, (size_t)(_numEleCmfdNode) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCoarseMeshGroupConst_CheckCollapse - cudaMemcpy - GPUCMFDXs_f"));

		cudaMemcpy((void*)_CMFDXs_nf, (void*)GPUCMFDXs_nf, (size_t)(_numEleCmfdNode) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCoarseMeshGroupConst_CheckCollapse - cudaMemcpy - GPUCMFDXs_nf"));

		cudaMemcpy((void*)_CMFD_Chi, (void*)GPUCMFD_Chi, (size_t)(_numEleCmfdNode) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCoarseMeshGroupConst_CheckCollapse - cudaMemcpy - GPUCMFD_Chi"));

		cudaMemcpy((void*)_CMFDXs_s, (void*)GPUCMFDXs_s, (size_t)(_numEleCmfdXSs) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCoarseMeshGroupConst_CheckCollapse - cudaMemcpy - GPUCMFDXs_s"));

#ifdef CUDAErrorCheck
		CUDAStatus = cudaGetLastError();
		if (CUDAStatus != cudaSuccess) {
			printf("\n [GPU-ERROR]  Lunch Failed: GPUCoarseMeshGroupConst_CheckCollapse with Info: %s\n", cudaGetErrorString(CUDAStatus));
			exit(69);
		}
#endif // CUDAErrorCheck
	}
}



/**
* @ Function: GPUCMFD_GCEvaluation_Part1()
*   introduction:
* @ Parameter List:
*   Return:
* @ Created     : 2019/08/16, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
extern "C"
void GPUCMFD_GCEvaluation_Part1(M_INT32 _outIter, M_INT32 _NumProcs, M_INT32 _MOC_NumPinsInSubDom, \
	M_INT32 _NumPinsInSubDomPerLayer, M_INT32 _MOCNumLayersZ, M_INT32 _MOCNumPinsInSubDom_X, \
	M_INT32 _MOCNumPinsInSubDom_Y, M_INT32 _NumCMFDGroup, M_INT32 _TotalNumFSRs_subDom, M_INT32 _TotalNumFSRs_perLayer, \
	M_INT32 _NumGroups, M_UINT64 *_MacroXS_s_NZStartID_Layer, M_UINT64 *_MacroXS_s_NNZ_Layer, \
	M_INT32 _ProblemType, FP_PRECISION *_MacroXS_t, FP_PRECISION *_MacroXS_a, FP_PRECISION *_MacroXS_f, \
	FP_PRECISION *_MacroXS_nf, FP_PRECISION *_Chi, FP_PRECISION *_MacroXS_s, FP_PRECISION *_FixedSour, \
	M_INT32 _MGCMFD_MOCParCurrInnerBound_Num, FP_PRECISION *_CMFDHetScalFlux, FP_PRECISION *_CMFDNodeDiffCoeff, \
	M_INT32 _NumPolarAng, M_INT32 _NumLTs_HS, FP_PRECISION_TIME *_TimeRec_MGCMFD_XSCollapse, \
	M_INT32 _OUTPUT_CMFDGroupConstant,M_INT32 _OUTPUT_CoarseMesh_GroupConstant, M_INT32 _NumScattOrder,\
	M_UINT64 *_MacroXS_s_NZRowStartID_Layer, M_UINT64 *_MacroXS_s_NZColStartID, FP_PRECISION_FLUX *_ScalarFluxOld,\
    M_INT32 _isTransient, FP_PRECISION *_tranSourceConstA, FP_PRECISION *_tranSourceConstB, FP_PRECISION *_tranSourceConstC,
    FP_PRECISION **_boundaryFlux, FP_PRECISION **_CPUMPIFlux)
{
	int _lay;
	// int _probType = 0;
	// FP_PRECISION _zero = (FP_PRECISION)(M_Zero_FP);
#if defined(DEFAULT_TIME)
	FP_PRECISION_TIME _default_time_start, _default_time_end;
	cudaDeviceSynchronize();
	_default_time_start = TIME_RECORD;
#endif // defined(DEFAULT_TIME)
	int _numEleCmfdNode, _numEleCmfdNodeLay, _numEleCmfdXSs,_numEleSurf_X,_numEleSurf_Y/*,_numEleSurf_Z*/;
	// int /*_numCMFDGroup,*/ _numLayerZ, _numCellsInSubDom_X, _numCellsInSubDom_Y;
	// _numCMFDGroup = _NumCMFDGroup;
	// _numLayerZ = _MOCNumLayersZ;
	// _numCellsInSubDom_X = _MOCNumPinsInSubDom_X;
	// _numCellsInSubDom_Y = _MOCNumPinsInSubDom_Y;

	_numEleCmfdNode = _MOC_NumPinsInSubDom * _NumCMFDGroup;
	_numEleCmfdNodeLay = _NumPinsInSubDomPerLayer * _NumCMFDGroup;
	_numEleCmfdXSs = _NumPinsInSubDomPerLayer * _NumCMFDGroup * _NumCMFDGroup;
	_numEleSurf_X = _MOCNumLayersZ * (_MOCNumPinsInSubDom_X + 1) * _MOCNumPinsInSubDom_Y * _NumCMFDGroup;
	_numEleSurf_Y = _MOCNumLayersZ * _MOCNumPinsInSubDom_X * (_MOCNumPinsInSubDom_Y + 1) * _NumCMFDGroup;
	// _numEleSurf_Z = (_MOCNumLayersZ + 1) * _MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y * _NumCMFDGroup;

	
	
for (_lay=0; _lay<_MOCNumLayersZ; _lay++) {
	GPUCopyMacroXSsHostToDevice_Layer(_lay,_TotalNumFSRs_subDom, _TotalNumFSRs_perLayer, _NumGroups, _MacroXS_s_NZStartID_Layer, \
		_MacroXS_s_NNZ_Layer, _ProblemType, _MacroXS_t, _MacroXS_a, _MacroXS_f, _MacroXS_nf, _Chi, _MacroXS_s, \
		_FixedSour, _NumScattOrder, _MacroXS_s_NZRowStartID_Layer, _MacroXS_s_NZColStartID, 1, _ScalarFluxOld);

	GPUCMFD_Restriction_Flux<<<GPUGridDim_CMFDNode, GPUBlockDim>>>(_lay, _numEleCmfdNodeLay, GPUCMFDHetScalFlux, \
		GPUScalarFlux, GPU_MGCMFDGroupBound, GPUFSRsAreaInRTM_AntiClockFSR, GPUNumFSRsInRTM, \
		GPUCellRTMType_SubDom, GPUSumNumFSRsInRTM, GPUSumNumFSRsBasedGlobCellID);
		CUDA_CheckError((char *)("GPUCMFD_Restriction_Flux"));

	GPUCMFD_Restriction_1DXSs<<<GPUGridDim_CMFDNode_Layer, GPUBlockDim>>>(_lay, _numEleCmfdNodeLay, GPUCMFDXs_t, \
		GPUMacroXS_t, GPUScalarFlux, GPU_MGCMFDGroupBound, GPUFSRsAreaInRTM_AntiClockFSR, GPUNumFSRsInRTM, \
		GPUCellRTMType_SubDom, GPUSumNumFSRsInRTM, GPUSumNumFSRsBasedGlobCellID);
	CUDA_CheckError((char *)("GPUCMFD_Restriction_1DXSs - Xs_t"));

    if (_OUTPUT_CMFDGroupConstant == 1 || _OUTPUT_CoarseMesh_GroupConstant == 1) {
		GPUCMFD_Restriction_1DXSs<<<GPUGridDim_CMFDNode_Layer, GPUBlockDim>>>(_lay, _numEleCmfdNodeLay, GPUCMFDXs_a, \
			GPUMacroXS_a, GPUScalarFlux, GPU_MGCMFDGroupBound, GPUFSRsAreaInRTM_AntiClockFSR, GPUNumFSRsInRTM, \
			GPUCellRTMType_SubDom, GPUSumNumFSRsInRTM, GPUSumNumFSRsBasedGlobCellID);
		CUDA_CheckError((char *)("GPUCMFD_Restriction_1DXSs - Xs_t"));

		GPUCMFD_Restriction_1DXSs<<<GPUGridDim_CMFDNode_Layer, GPUBlockDim>>>(_lay, _numEleCmfdNodeLay, GPUCMFDXs_f, \
			GPUMacroXS_f, GPUScalarFlux, GPU_MGCMFDGroupBound, GPUFSRsAreaInRTM_AntiClockFSR, GPUNumFSRsInRTM, \
			GPUCellRTMType_SubDom, GPUSumNumFSRsInRTM, GPUSumNumFSRsBasedGlobCellID);
		CUDA_CheckError((char *)("GPUCMFD_Restriction_1DXSs - Xs_t"));
	}

	GPUCMFD_Restriction_1DXSs<<<GPUGridDim_CMFDNode_Layer, GPUBlockDim>>>(_lay, _numEleCmfdNodeLay, GPUCMFDXs_nf, \
		GPUMacroXS_nf, GPUScalarFlux, GPU_MGCMFDGroupBound, GPUFSRsAreaInRTM_AntiClockFSR, GPUNumFSRsInRTM, \
		GPUCellRTMType_SubDom, GPUSumNumFSRsInRTM, GPUSumNumFSRsBasedGlobCellID);
	CUDA_CheckError((char *)("GPUCMFD_Restriction_1DXSs - Xs_nf"));

	GPUCMFD_Restriction_Chi<<<GPUGridDim_CMFDNode_Layer, GPUBlockDim>>>(_lay,_numEleCmfdNodeLay, GPUCMFD_Chi, \
		GPUMacroXS_chi, GPUMacroXS_nf, GPUScalarFlux, GPU_MGCMFDGroupBound, GPUFSRsAreaInRTM_AntiClockFSR, \
		GPUNumFSRsInRTM, GPUCellRTMType_SubDom, GPUSumNumFSRsInRTM, GPUSumNumFSRsBasedGlobCellID);
	CUDA_CheckError((char *)("GPUCMFD_Restriction_Chi"));

	GPUCMFD_Restriction_2DXSs<<<GPUGridDim_CMFDXSs, GPUBlockDim>>>(_lay,_numEleCmfdXSs, GPUCMFDXs_s, \
		GPUMacroXS_s, GPUScalarFlux, GPU_MGCMFDGroupBound, GPUFSRsAreaInRTM_AntiClockFSR, \
		GPUNumFSRsInRTM, GPUCellRTMType_SubDom, GPUSumNumFSRsInRTM, GPUSumNumFSRsBasedGlobCellID,\
		GPUMacroXS_s_NZRowStartID_Layer, GPUMacroXS_s_NZColStartID);
	CUDA_CheckError((char *)("GPUCMFD_Restriction_2DXSs"));

	GPUCMFD_Restriction_DiffCoeff<<<GPUGridDim_CMFDNode_Layer, GPUBlockDim>>>(_lay,_numEleCmfdNodeLay, \
		GPUCMFDNodeDiffCoeff, GPUMacroXS_t, GPUScalarFlux, GPUCMFDHetScalFlux, GPU_MGCMFDGroupBound, \
		GPUFSRsAreaInRTM_AntiClockFSR, GPUNumFSRsInRTM, GPUCellRTMType_SubDom, GPUSumNumFSRsInRTM, \
		GPUSumNumFSRsBasedGlobCellID);
	CUDA_CheckError((char *)("GPUCMFD_Restriction_DiffCoeff"));

}

#if defined(MPI_USED)
	if ((_outIter > 0) && (_NumProcs > 1)) {
		GPUSetMOCPartCurrToZero_InnerBound<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(GPUMOCParCurr_X_Pos, \
			GPUMOCParCurr_X_Neg, GPUMOCParCurr_Y_Pos, GPUMOCParCurr_Y_Neg, \
			_MGCMFD_MOCParCurrInnerBound_Num, GPUMGCMFD_MOCParCurrInnerBound_Index, GPUMGCMFD_MOCParCurrInnerBound_Dir);

		GPUTallyMOCPartialCurr_InnerBound(_NumProcs ,_MOCNumLayersZ, _NumPolarAng, _NumLTs_HS, \
			_NumGroups, _NumPinsInSubDomPerLayer, _boundaryFlux, _CPUMPIFlux);
	}
#endif

	GPUCMFD_RestrictionCurr_GroupCond<<<GPUGridDim_CMFDSurf_X, GPUBlockDim>>>(_numEleSurf_X, GPUMOCtoCMFD_ParCurr_X_Pos, \
		GPUMOCParCurr_X_Pos, GPU_MGCMFDGroupBound);
	GPUCMFD_RestrictionCurr_GroupCond<<<GPUGridDim_CMFDSurf_X, GPUBlockDim>>>(_numEleSurf_X, GPUMOCtoCMFD_ParCurr_X_Neg, \
		GPUMOCParCurr_X_Neg, GPU_MGCMFDGroupBound);
	GPUCMFD_RestrictionCurr_GroupCond<<<GPUGridDim_CMFDSurf_Y, GPUBlockDim>>>(_numEleSurf_Y, GPUMOCtoCMFD_ParCurr_Y_Pos, \
		GPUMOCParCurr_Y_Pos, GPU_MGCMFDGroupBound);
	GPUCMFD_RestrictionCurr_GroupCond<<<GPUGridDim_CMFDSurf_Y, GPUBlockDim>>>(_numEleSurf_Y, GPUMOCtoCMFD_ParCurr_Y_Neg, \
		GPUMOCParCurr_Y_Neg, GPU_MGCMFDGroupBound);
	CUDA_CheckError((char *)("GPUCMFD_RestrictionCurr_GroupCond_XY"));

	GPUCMFD_RestrictionNetCurr<<<GPUGridDim_CMFDSurf_X, GPUBlockDim>>>(_numEleSurf_X, GPUMOCtoCMFD_NetCurr_X, \
		GPUMOCtoCMFD_ParCurr_X_Pos, GPUMOCtoCMFD_ParCurr_X_Neg);
	GPUCMFD_RestrictionNetCurr<<<GPUGridDim_CMFDSurf_Y, GPUBlockDim>>>(_numEleSurf_Y, GPUMOCtoCMFD_NetCurr_Y, \
		GPUMOCtoCMFD_ParCurr_Y_Pos, GPUMOCtoCMFD_ParCurr_Y_Neg);
	CUDA_CheckError((char *)("GPUCMFD_RestrictionNetCurr"));

#if defined(MPI_USED)
    if (_NumProcs > 1) {
	    cudaMemcpy((void*)_CMFDHetScalFlux, (void*)GPUCMFDHetScalFlux, (size_t)(_numEleCmfdNode) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
	    CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part1 - cudaMemcpy - GPUCMFDHetScalFlux"));
	    cudaMemcpy((void*)_CMFDNodeDiffCoeff, (void*)GPUCMFDNodeDiffCoeff, (size_t)(_numEleCmfdNode) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPUCMFD_CheckCollapse - cudaMemcpy - GPUCMFDNodeDiffCoeff"));
    }
#endif // defined(MPI_USED)

#if defined(DEFAULT_TIME)
	cudaDeviceSynchronize();
	_default_time_end = TIME_RECORD;
	_TimeRec_MGCMFD_XSCollapse[0] += _default_time_end - _default_time_start;
	_default_time_start = TIME_RECORD;
#endif // defined(DEFAULT_TIME)

#ifdef CUDAErrorCheck
	CUDAStatus = cudaGetLastError();
	if (CUDAStatus != cudaSuccess) {
		printf("\n [GPU-ERROR]  Lunch Failed: GPUCMFD_GCEvaluation_Part1 with Info: %s\n", cudaGetErrorString(CUDAStatus));
		exit(69);
	}
#endif // CUDAErrorCheck
}

extern "C"
void GPUCMFD_GCEvaluation_Part2(M_INT32 _ProcID,M_INT32 _NumProcs,M_INT32 _isFirstInvoke, M_INT32 _MOC_NumPinsInSubDom, \
	M_INT32 _MOCNumLayersZ, M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, M_INT32 _NumCMFDGroup, \
	FP_PRECISION *_MGCMFDFlux_MPIRecv, FP_PRECISION *_MGCMFDNodeD_MPIRecv, M_INT32 _MGCMFDData_MPIComm_Num, \
	M_INT32 _SubDomLocID_2D_X, M_INT32 _SubDomLocID_2D_Y, FP_PRECISION _MGCMFD_damp, FP_PRECISION *_CMFDXs_s, \
	M_INT32 *_CMFD_NumNoneZeroValue_Matrix, M_INT32 _MGCMFD_LinearSolver, M_INT32 _GmresPrecond_MG, M_INT32 *_MG_DiagPointer, \
	M_INT32 _ProblemType, FP_PRECISION_TIME *_TimeRec_MGCMFD_XSCollapse, FP_PRECISION_TIME *_TimeRec_MGCMFD_MatrixConst)
{
	int _probType = 0;
	// FP_PRECISION _zero = (FP_PRECISION)(M_Zero_FP);
#if defined(DEFAULT_TIME)
	FP_PRECISION_TIME _default_time_start, _default_time_end;
	cudaDeviceSynchronize();
	_default_time_start = TIME_RECORD;
#endif // defined(DEFAULT_TIME)
	int _numEleCmfdNode,_numEleSurf_X,_numEleSurf_Y,_numEleSurf_Z;
#if defined(M_CMFDMatrixIncScatt)
	int _numEleCmfdXSs;
	_numEleCmfdXSs = _MOC_NumPinsInSubDom * _NumCMFDGroup * _NumCMFDGroup;
#endif
	cusparseStatus_t _cudaStatus;
int _pBufferSize_M;
int _pBufferSize_L;
int _pBufferSize_U;
int _pBufferSize;
int _structural_zero;
int _numerical_zero;

	_numEleCmfdNode = _MOC_NumPinsInSubDom * _NumCMFDGroup;
	
	_numEleSurf_X = _MOCNumLayersZ * (_MOCNumPinsInSubDom_X + 1) * _MOCNumPinsInSubDom_Y * _NumCMFDGroup;
	_numEleSurf_Y = _MOCNumLayersZ * _MOCNumPinsInSubDom_X * (_MOCNumPinsInSubDom_Y + 1) * _NumCMFDGroup;
	_numEleSurf_Z = (_MOCNumLayersZ + 1) * _MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y * _NumCMFDGroup;
	
#if defined(MPI_USED)
    if (_NumProcs>1) {
		cudaMemcpy((void*)GPUMGCMFDFlux_MPIRecv, (void*)_MGCMFDFlux_MPIRecv, (size_t)(_MGCMFDData_MPIComm_Num) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
	    CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2 - cudaMemcpy - GPUMGCMFDFlux_MPIRecv"));
	    cudaMemcpy((void*)GPUMGCMFDNodeD_MPIRecv, (void*)_MGCMFDNodeD_MPIRecv, (size_t)(_MGCMFDData_MPIComm_Num) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
		CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2 - cudaMemcpy - GPUMGCMFDNodeD_MPIRecv"));
	}
	
#endif // defined(MPI_USED)

#if defined(M_YQZ_CMFD_Damp)
	cudaMemcpy((void*)GPUCMFDSurfNDF_XOld, (void*)GPUCMFDSurfNDF_X, (size_t)(_numEleSurf_X) * sizeof(FP_PRECISION), cudaMemcpyDeviceToDevice);
	cudaMemcpy((void*)GPUCMFDSurfNDF_YOld, (void*)GPUCMFDSurfNDF_Y, (size_t)(_numEleSurf_Y) * sizeof(FP_PRECISION), cudaMemcpyDeviceToDevice);
	cudaMemcpy((void*)GPUCMFDSurfNDF_ZOld, (void*)GPUCMFDSurfNDF_Z, (size_t)(_numEleSurf_Z) * sizeof(FP_PRECISION), cudaMemcpyDeviceToDevice);
	CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2  - cudaMemcpy - GPUCMFDSurfNDF_XYZ"));
#endif // defined(M_YQZ_CMFD_Damp)
	GPUCMFD_RestrictionSurfDiffCoef_X<<<GPUGridDim_CMFDSurf_X, GPUBlockDim>>>(_numEleSurf_X, \
		GPUCMFDSurfDiffCoeff_X,	GPUCMFDSurfNDF_X, GPUCMFDNodeDiffCoeff, GPUCMFDHetScalFlux, \
		GPUMOCtoCMFD_NetCurr_X, GPUCellRTMType_SubDom, GPURTMWidth_X, _NumCMFDGroup, \
		_SubDomLocID_2D_X, GPUNumSubDomains, GPUMGCMFDFlux_MPIRecv, GPUMGCMFDNodeD_MPIRecv, GPUMGCMFDData_MPIComm_SumInd);
	CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2 - GPUCMFD_RestrictionSurfDiffCoef_X"));
	GPUCMFD_RestrictionSurfDiffCoef_Y<<<GPUGridDim_CMFDSurf_Y, GPUBlockDim>>>(_numEleSurf_Y, \
		GPUCMFDSurfDiffCoeff_Y, GPUCMFDSurfNDF_Y, GPUCMFDNodeDiffCoeff, GPUCMFDHetScalFlux, \
		GPUMOCtoCMFD_NetCurr_Y, GPUCellRTMType_SubDom, GPURTMWidth_Y, _NumCMFDGroup, \
		_SubDomLocID_2D_Y, GPUNumSubDomains, GPUMGCMFDFlux_MPIRecv, GPUMGCMFDNodeD_MPIRecv, GPUMGCMFDData_MPIComm_SumInd);
	CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2 - GPUCMFD_RestrictionSurfDiffCoef_Y"));

#if defined(M_3DMOCex)
    GPUCMFD_RestrictionCurr_GroupCond<<<GPUGridDim_CMFDSurf_Z, GPUBlockDim>>>(_numEleSurf_Z, GPUMOCtoCMFD_NetCurr_Z, \
		GPUMOCNetCurr_Z, GPU_MGCMFDGroupBound);
	CUDA_CheckError((char *)("GPUCMFD_RestrictionCurr_GroupCond_Z"));

	GPUCMFD_RestrictionSurfDiffCoef_Z<<<GPUGridDim_CMFDSurf_Z, GPUBlockDim>>>(_numEleSurf_Z, \
		GPUCMFDSurfDiffCoeff_Z, GPUCMFDSurfNDF_Z, GPUCMFDNodeDiffCoeff, GPUCMFDHetScalFlux, \
		GPUMOCtoCMFD_NetCurr_Z, GPUCellRTMType_SubDom, GPUWidthLayersInCore, _NumCMFDGroup);
	CUDA_CheckError((char *)("GPUCMFD_RestrictionSurfDiffCoef_Z"));
#endif //#if defiend(M_3DMOCex)
#if defined(M_YQZ_CMFD_Damp) && !defined(MOCthenCMFD)
	GPU_VecDamp_FP<<<GPUGridDim_CMFDSurf_X, GPUBlockDim>>>(_numEleSurf_X, _MGCMFD_damp, GPUCMFDSurfNDF_X, GPUCMFDSurfNDF_XOld);
	GPU_VecDamp_FP<<<GPUGridDim_CMFDSurf_Y, GPUBlockDim>>>(_numEleSurf_Y, _MGCMFD_damp, GPUCMFDSurfNDF_Y, GPUCMFDSurfNDF_YOld);
#if defined(M_3DMOCex)
	GPU_VecDamp_FP<<<GPUGridDim_CMFDSurf_Z, GPUBlockDim>>>(_numEleSurf_Z, _MGCMFD_damp, GPUCMFDSurfNDF_Z, GPUCMFDSurfNDF_ZOld);
#endif //#if defiend(M_3DMOCex)
	CUDA_CheckError((char *)("GPU_VecDamp_FP-XYZ"));
#endif // defined(M_YQZ_CMFD_Damp) && !defined(MOCthenCMFD)

#if defined(MPI_USED)
	if (_NumProcs > 1) {
		GPUSetCMFDMPI_InBoundSourCoeff<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(_ProcID, _NumProcs, \
			GPUMGCMFDMPI_InBoundSourCoeff, GPUCMFDSurfDiffCoeff_X, GPUCMFDSurfNDF_X, GPUCMFDSurfDiffCoeff_Y, \
			GPUCMFDSurfNDF_Y, GPUCMFDSurfDiffCoeff_Z, GPUCMFDSurfNDF_Z, _MGCMFDData_MPIComm_Num, \
			GPUMGCMFDMPI_InBoundSourCoeff_Index, GPUMGCMFDMPI_InBoundSourCoeff_Dir, GPUMGCMFDMPI_InBoundSourCoeff_GeoDim);
	}
#endif // defined(MPI_USED)


#if defined(DEFAULT_TIME)
	cudaDeviceSynchronize();
	_default_time_end = TIME_RECORD;
	_TimeRec_MGCMFD_XSCollapse[0] += _default_time_end - _default_time_start;
	_default_time_start = TIME_RECORD;
#endif // defined(DEFAULT_TIME)

#if defined(DIRECT_CSR)
	if (_isFirstInvoke == 1) {
#if defined(M_2DMOC)
		_probType = 0;
#endif //#if defiend(M_2DMOC)

#if defined(M_3DMOCex)
		_probType = 1;
#endif //#if defiend(M_3DMOCex)
		
#if defined(M_CMFDMatrixIncScatt)
		cudaMemcpy((void*)_CMFDXs_s, (void*)GPUCMFDXs_s, (size_t)(_numEleCmfdXSs) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
#endif // #if defined(M_CMFDMatrixIncScatt)
		_CMFD_NumNoneZeroValue_Matrix[0] = GPUCMFD_GetNumNoneZeroEle_MatrixA(_MOCNumLayersZ, \
			_MOCNumPinsInSubDom_X, _MOCNumPinsInSubDom_Y, _NumCMFDGroup, _CMFDXs_s, _probType);
		CUDA_CheckError((char *)("GPUCMFD_GetNumNoneZeroEle_MatrixA"));

		//cudaMemcpyToSymbol(GPU_MGCMFDNumNoneZeroValueLU_Matrix, (void*)&CMFD_NumNoneZeroValueLU_Matrix, sizeof(int), 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(GPU_MGCMFDNumNoneZeroValue_Matrix, (void*)_CMFD_NumNoneZeroValue_Matrix, sizeof(int), 0, cudaMemcpyHostToDevice);
	    cudaMemcpyToSymbol(GPU_MGCMFDNumValueDiag_Matrix, (void*)&_numEleCmfdNode, sizeof(int), 0, cudaMemcpyHostToDevice);
		GPU_MGCMFDMatrixCSR_AllocMem(_numEleCmfdNode, _CMFD_NumNoneZeroValue_Matrix[0], _MGCMFD_LinearSolver);

		GPUCMFD_GetNoneZeroIndexInfo_MatrixCSR(_MOCNumLayersZ, _MOCNumPinsInSubDom_X, _MOCNumPinsInSubDom_Y, \
			_NumCMFDGroup, _CMFDXs_s, Dev_MGCMFDMatrixCSR_RowStartInd, Dev_MGCMFDMatrixCSR_ColInd, _ProblemType);
		CUDA_CheckError((char *)("GPUCMFD_GetNoneZeroIndexInfo_MatrixCSR"));

		cudaMemcpy((void*)GPU_MGCMFDMatrixCSR_RowStartInd, (void*)Dev_MGCMFDMatrixCSR_RowStartInd, (size_t)((_numEleCmfdNode + 1)) * sizeof(int), cudaMemcpyHostToDevice);
		

		cudaMemcpy((void*)GPU_MGCMFDMatrixCSR_ColInd, (void*)Dev_MGCMFDMatrixCSR_ColInd, (size_t)(_CMFD_NumNoneZeroValue_Matrix[0]) * sizeof(int), cudaMemcpyHostToDevice);
		GPU_DiagPointer_CSRMatrix<<<GPUGridDim_CMFDNode, GPUBlockDim>>>(_numEleCmfdNode, GPU_MGCMFDMatrixCSR_RowStartInd, \
			GPU_MGCMFDMatrixCSR_ColInd, GPU_MGCMFDDiagPointer);
		CUDA_CheckError((char *)("GPU_DiagPointer_CSRMatrix"));
	}

	GPUCMFD_ConstructMatrixA_CSR<<<GPUGridDim_CMFDNode, GPUBlockDim>>>(GPUCellRTMType_SubDom,GPUCMFDSurfDiffCoeff_X, \
		GPUCMFDSurfDiffCoeff_Y, GPUCMFDSurfDiffCoeff_Z, GPUCMFDSurfNDF_X, GPUCMFDSurfNDF_Y, GPUCMFDSurfNDF_Z, \
		GPUCMFDXs_t, GPUCMFDXs_s, GPURTMWidth_X, GPURTMWidth_Y, GPUWidthLayersInCore, GPU_MGCMFDMatrixCSR_RowStartInd, \
		GPU_MGCMFDMatrixCSR, _numEleCmfdNode, _NumCMFDGroup);
	CUDA_CheckError((char *)("GPUCMFD_ConstructMatrixA_CSR"));
#endif // defined(DIRECT_CSR)
	
	if (_MGCMFD_LinearSolver == 2) {
		if (_GmresPrecond_MG) {

#if defined(M_GMRES_JacobiPRE)
			//GPU_VecInverse <<<GPUGridDim_CMFDNode, GPUBlockDim>>>(_numEleCmfdNode, GPU_MGCMFDMatrixCSR_Diag);
#endif //defined(M_GMRES_JacobiPRE)

#if defined(M_GMRES_IluPRE)
#if defined(M_GMRES_PRE_cuSparseLib)
			if (_GmresPrecond_MG) {
				cudaMemcpy((void*)GPU_MGCMFDMatrixCSR_ILUFactor, (void*)GPU_MGCMFDMatrixCSR, \
				(size_t)(_CMFD_NumNoneZeroValue_Matrix[0]) * sizeof(FP_PRECISION), cudaMemcpyDeviceToDevice);
				CUDA_CheckError((char *)("cudaMemcpy-GPU_MGCMFDMatrixCSR_ILUFactor"));
			}

			if (_GmresPrecond_MG && _isFirstInvoke == 1) {
				_cudaStatus = cusparseCreate(&MG_cusparseHandle);
				// step 1: create a descriptor which contains
				// - matrix M is base-0
				// - matrix L is base-0
				// - matrix L is lower triangular
				// - matrix L has unit diagonal
				// - matrix U is base-0
				// - matrix U is upper triangular
				// - matrix U has non-unit diagonal
				cusparseCreateMatDescr(&MG_descr_M);
				cusparseSetMatIndexBase(MG_descr_M, CUSPARSE_INDEX_BASE_ZERO);
				// cusparseSetMatIndexBase(MG_descr_M, CUSPARSE_INDEX_BASE_ONE);
				cusparseSetMatType(MG_descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
				CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseCreateMatDescr-MG_descr_M"));

				cusparseCreateMatDescr(&MG_descr_L);
				cusparseSetMatIndexBase(MG_descr_L, CUSPARSE_INDEX_BASE_ZERO);
				// cusparseSetMatIndexBase(MG_descr_L, CUSPARSE_INDEX_BASE_ONE);
				cusparseSetMatType(MG_descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
				cusparseSetMatFillMode(MG_descr_L, CUSPARSE_FILL_MODE_LOWER);
				cusparseSetMatDiagType(MG_descr_L, CUSPARSE_DIAG_TYPE_UNIT);
				CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseCreateMatDescr-MG_descr_L"));

				cusparseCreateMatDescr(&MG_descr_U);
				cusparseSetMatIndexBase(MG_descr_U, CUSPARSE_INDEX_BASE_ZERO);
				// cusparseSetMatIndexBase(MG_descr_U, CUSPARSE_INDEX_BASE_ONE);
				cusparseSetMatType(MG_descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
				cusparseSetMatFillMode(MG_descr_U, CUSPARSE_FILL_MODE_UPPER);
				cusparseSetMatDiagType(MG_descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);
				CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseCreateMatDescr-MG_descr_U"));

				// step 2: create a empty info structure
				// we need one info for csrilu02 and two info's for csrsv2
				cusparseCreateCsrilu02Info(&MG_info_M);
				cusparseCreateCsrsv2Info(&MG_info_L);
				cusparseCreateCsrsv2Info(&MG_info_U);
				CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseCreateCsrilu02Info-MG_info_M"));

				// step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
#if defined(DOUBLE)
				cusparseDcsrilu02_bufferSize(MG_cusparseHandle, _numEleCmfdNode, _CMFD_NumNoneZeroValue_Matrix[0],
					MG_descr_M, GPU_MGCMFDMatrixCSR_ILUFactor, GPU_MGCMFDMatrixCSR_RowStartInd, GPU_MGCMFDMatrixCSR_ColInd, \
					MG_info_M, &_pBufferSize_M);
				cusparseDcsrsv2_bufferSize(MG_cusparseHandle, MG_trans_L, _numEleCmfdNode, _CMFD_NumNoneZeroValue_Matrix[0],
					MG_descr_L, GPU_MGCMFDMatrixCSR_ILUFactor, GPU_MGCMFDMatrixCSR_RowStartInd, GPU_MGCMFDMatrixCSR_ColInd, \
					MG_info_L, &_pBufferSize_L);
				cusparseDcsrsv2_bufferSize(MG_cusparseHandle, MG_trans_U, _numEleCmfdNode, _CMFD_NumNoneZeroValue_Matrix[0],
					MG_descr_U, GPU_MGCMFDMatrixCSR_ILUFactor, GPU_MGCMFDMatrixCSR_RowStartInd, GPU_MGCMFDMatrixCSR_ColInd, \
					MG_info_U, &_pBufferSize_U);
#endif //defined(DOUBLE)

#if defined(SINGLE)
                cusparseScsrilu02_bufferSize(MG_cusparseHandle, _numEleCmfdNode, _CMFD_NumNoneZeroValue_Matrix[0],
					MG_descr_M, GPU_MGCMFDMatrixCSR_ILUFactor, GPU_MGCMFDMatrixCSR_RowStartInd, GPU_MGCMFDMatrixCSR_ColInd, \
					MG_info_M, &_pBufferSize_M);
				cusparseScsrsv2_bufferSize(MG_cusparseHandle, MG_trans_L, _numEleCmfdNode, _CMFD_NumNoneZeroValue_Matrix[0],
						MG_descr_L, GPU_MGCMFDMatrixCSR_ILUFactor, GPU_MGCMFDMatrixCSR_RowStartInd, GPU_MGCMFDMatrixCSR_ColInd, \
						MG_info_L, &_pBufferSize_L);
				cusparseScsrsv2_bufferSize(MG_cusparseHandle, MG_trans_U, _numEleCmfdNode, _CMFD_NumNoneZeroValue_Matrix[0],
						MG_descr_U, GPU_MGCMFDMatrixCSR_ILUFactor, GPU_MGCMFDMatrixCSR_RowStartInd, GPU_MGCMFDMatrixCSR_ColInd, \
						MG_info_U, &_pBufferSize_U);
#endif //defined(SINGLE)
				CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseDcsrilu02_bufferSize"));

				_pBufferSize = max(_pBufferSize_M, max(_pBufferSize_L, _pBufferSize_U));
	            // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
	            cudaMalloc((void**)&MG_pBuffer, _pBufferSize);
	            CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cudaMalloc-MG_pBuffer"));

	            // step 4: perform analysis of incomplete Cholesky on M
	            //         perform analysis of triangular solve on L
	            //         perform analysis of triangular solve on U
	            // The lower(upper) triangular part of M has the same sparsity pattern as L(U),
	            // we can do analysis of csrilu0 and csrsv2 simultaneously.
#if defined(DOUBLE)
	            cusparseDcsrilu02_analysis(MG_cusparseHandle, _numEleCmfdNode, _CMFD_NumNoneZeroValue_Matrix[0], MG_descr_M,
    	            GPU_MGCMFDMatrixCSR_ILUFactor, GPU_MGCMFDMatrixCSR_RowStartInd, GPU_MGCMFDMatrixCSR_ColInd, MG_info_M,
		            MG_policy_M, MG_pBuffer);
		        CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseDcsrilu02_analysis"));

	            _cudaStatus = cusparseXcsrilu02_zeroPivot(MG_cusparseHandle, MG_info_M, &_structural_zero);
		        if (CUSPARSE_STATUS_ZERO_PIVOT == _cudaStatus){
					printf("A(%d,%d) is missing\n", _structural_zero, _structural_zero);
		        }

		        cusparseDcsrsv2_analysis(MG_cusparseHandle, MG_trans_L, _numEleCmfdNode, _CMFD_NumNoneZeroValue_Matrix[0], MG_descr_L,
    		        GPU_MGCMFDMatrixCSR_ILUFactor, GPU_MGCMFDMatrixCSR_RowStartInd, GPU_MGCMFDMatrixCSR_ColInd,
    		        MG_info_L, MG_policy_L, MG_pBuffer);
			    CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseDcsrsv2_analysis-MG_trans_L"));

				cusparseDcsrsv2_analysis(MG_cusparseHandle, MG_trans_U, _numEleCmfdNode, _CMFD_NumNoneZeroValue_Matrix[0], MG_descr_U,
					GPU_MGCMFDMatrixCSR_ILUFactor, GPU_MGCMFDMatrixCSR_RowStartInd, GPU_MGCMFDMatrixCSR_ColInd,
					MG_info_U, MG_policy_U, MG_pBuffer);
				CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseDcsrsv2_analysis-MG_trans_U"));
#endif //defined(DOUBLE)

#if defined(SINGLE)
	            cusparseScsrilu02_analysis(MG_cusparseHandle, _numEleCmfdNode, _CMFD_NumNoneZeroValue_Matrix[0], MG_descr_M,
    	            GPU_MGCMFDMatrixCSR_ILUFactor, GPU_MGCMFDMatrixCSR_RowStartInd, GPU_MGCMFDMatrixCSR_ColInd, MG_info_M,
		            MG_policy_M, MG_pBuffer);
		        CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseDcsrilu02_analysis"));

	            _cudaStatus = cusparseXcsrilu02_zeroPivot(MG_cusparseHandle, MG_info_M, &_structural_zero);
		        if (CUSPARSE_STATUS_ZERO_PIVOT == _cudaStatus){
					printf("A(%d,%d) is missing\n", _structural_zero, _structural_zero);
		        }

		        cusparseScsrsv2_analysis(MG_cusparseHandle, MG_trans_L, _numEleCmfdNode, _CMFD_NumNoneZeroValue_Matrix[0], MG_descr_L,
    		        GPU_MGCMFDMatrixCSR_ILUFactor, GPU_MGCMFDMatrixCSR_RowStartInd, GPU_MGCMFDMatrixCSR_ColInd,
    		        MG_info_L, MG_policy_L, MG_pBuffer);
			    CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseScsrsv2_analysis-MG_trans_L"));

				cusparseScsrsv2_analysis(MG_cusparseHandle, MG_trans_U, _numEleCmfdNode, _CMFD_NumNoneZeroValue_Matrix[0], MG_descr_U,
					GPU_MGCMFDMatrixCSR_ILUFactor, GPU_MGCMFDMatrixCSR_RowStartInd, GPU_MGCMFDMatrixCSR_ColInd,
					MG_info_U, MG_policy_U, MG_pBuffer);
				CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseScsrsv2_analysis-MG_trans_U"));
#endif //defined(SINGLE)
			}
			if (_GmresPrecond_MG) {
				// step 5: M = L * U
#if defined(DOUBLE)
				cusparseDcsrilu02(MG_cusparseHandle, _numEleCmfdNode, _CMFD_NumNoneZeroValue_Matrix[0], MG_descr_M,
					GPU_MGCMFDMatrixCSR_ILUFactor, GPU_MGCMFDMatrixCSR_RowStartInd, GPU_MGCMFDMatrixCSR_ColInd, \
					MG_info_M, MG_policy_M, MG_pBuffer);
#endif // defined(DOUBLE)
#if defined(SINGLE)
				cusparseScsrilu02(MG_cusparseHandle, _numEleCmfdNode, _CMFD_NumNoneZeroValue_Matrix[0], MG_descr_M,
					GPU_MGCMFDMatrixCSR_ILUFactor, GPU_MGCMFDMatrixCSR_RowStartInd, GPU_MGCMFDMatrixCSR_ColInd, \
					MG_info_M, MG_policy_M, MG_pBuffer);
#endif // defined(DOUBLE)
				_cudaStatus = cusparseXcsrilu02_zeroPivot(MG_cusparseHandle, MG_info_M, &_numerical_zero);
				if (CUSPARSE_STATUS_ZERO_PIVOT == _cudaStatus){
				   printf("U(%d,%d) is zero\n", _numerical_zero, _numerical_zero);
				}
			}

#else //!defined(M_GMRES_PRE_cuSparseLib), the preconditioning is performed on CPU
			if (_GmresPrecond_MG && _isFirstInvoke == 1) {
				_MG_ILUMatrix = (FP_PRECISION *)calloc(_CMFD_NumNoneZeroValue_Matrix[0], sizeof(FP_PRECISION));
				_MatrixCSR = (FP_PRECISION *)calloc(_CMFD_NumNoneZeroValue_Matrix[0], sizeof(FP_PRECISION));
			}
			if (_GmresPrecond_MG) {
				cudaMemcpy((void*)_MatrixCSR, (void*)GPU_MGCMFDMatrixCSR, \
				(size_t)(_CMFD_NumNoneZeroValue_Matrix[0]) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);

				cudaMemcpy((void*)_MG_DiagPointer, (void*)GPU_MGCMFDDiagPointer, (size_t)(_numEleCmfdNode) * sizeof(int), cudaMemcpyDeviceToHost);

				GPU_ILUFactPrecond(_numEleCmfdNode, _CMFD_NumNoneZeroValue_Matrix[0], Dev_MGCMFDMatrixCSR_RowStartInd, \
					Dev_MGCMFDMatrixCSR_ColInd, _MatrixCSR, _MG_DiagPointer, _MG_ILUMatrix, _MG_IW);
				CUDA_CheckError((char *)("GPU_ILUFactPrecond"));
			}
#endif //defined(M_GMRES_PRE_cuSparseLib)
#endif // defined(M_GMRES_IluPRE)
		}
	}

#if defined(DEFAULT_TIME)
    cudaDeviceSynchronize();
	_default_time_end = TIME_RECORD;
	_TimeRec_MGCMFD_MatrixConst[0] += _default_time_end - _default_time_start;
#endif // defined(DEFAULT_TIME)

    cudaMemcpy((void*)GPUCMFDScalarFlux, (void*)GPUCMFDHetScalFlux, \
	(size_t)(_numEleCmfdNode) * sizeof(FP_PRECISION), cudaMemcpyDeviceToDevice);
	CUDA_CheckError((char *)("cudaMemcpy-GPUCMFDScalarFlux"));
    CUDA_CheckError((char *)("Lunch Failed: GPUCMFD_GCEvaluation_Part2 with Info:"));
}


/**
* @ Function: GPUCGCMFD_GCEvaluation_Part1()
*   introduction:
* @ Parameter List:
*   Return:
* @ Created     : 2019/08/16, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
extern "C"
void GPUCGCMFD_GCEvaluation_Part1(M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCoarseCMFDGroup, FP_PRECISION_TIME *_TimeRec_CGCMFD_XSCollapse)
{
#if defined(DEFAULT_TIME)
	FP_PRECISION_TIME _default_time_start, _default_time_end;
	cudaDeviceSynchronize();
	_default_time_start = TIME_RECORD;
#endif // defined(DEFAULT_TIME)
	int _numEleCmfdNode, _numEleCmfdXSs;
	_numEleCmfdNode = _MOC_NumPinsInSubDom * _NumCoarseCMFDGroup;
	_numEleCmfdXSs = _MOC_NumPinsInSubDom * _NumCoarseCMFDGroup * _NumCoarseCMFDGroup;

	GPU_CGCMFDRestriction_Sumation<<<GPUGridDim_CGCMFDNode, GPUBlockDim>>>(GPUCMFDScalarFlux, \
		GPU_CGCMFDHetScalFlux, GPU_CGCMFDGroupBound, _numEleCmfdNode);

	GPU_CGCMFDRestriction_FluxWeighted<<<GPUGridDim_CGCMFDNode, GPUBlockDim>>>(GPUCMFDXs_t, \
		GPU_CGCMFDXs_t, GPUCMFDScalarFlux, GPU_CGCMFDHetScalFlux, GPU_CGCMFDGroupBound, _numEleCmfdNode);

	GPU_CGCMFDRestriction_FluxWeighted<<<GPUGridDim_CGCMFDNode, GPUBlockDim>>>(GPUCMFDXs_nf, \
		GPU_CGCMFDXs_nf, GPUCMFDScalarFlux, GPU_CGCMFDHetScalFlux, GPU_CGCMFDGroupBound, _numEleCmfdNode);

	GPU_CGCMFDRestriction_Sumation<<<GPUGridDim_CGCMFDNode, GPUBlockDim>>>(GPUCMFD_Chi, \
		GPU_CGCMFD_Chi, GPU_CGCMFDGroupBound, _numEleCmfdNode);

	GPU_CGCMFDRestriction_2DXSs<<<GPUGridDim_CGCMFDXSs, GPUBlockDim>>>(GPUCMFDXs_s, GPU_CGCMFDXs_s, GPUCMFDScalarFlux, \
		GPU_CGCMFDHetScalFlux, GPU_CGCMFDGroupBound, _numEleCmfdXSs);

	GPU_CGCMFDRestriction_FluxWeighted<<<GPUGridDim_CGCMFDNode, GPUBlockDim>>>(GPUCMFDNodeDiffCoeff, \
		GPU_CGCMFDNodeDiffCoeff, GPUCMFDScalarFlux, GPU_CGCMFDHetScalFlux, GPU_CGCMFDGroupBound, _numEleCmfdNode);
	
#if defined(DEFAULT_TIME)
	cudaDeviceSynchronize();
	_default_time_end = TIME_RECORD;
	_TimeRec_CGCMFD_XSCollapse[0] += _default_time_end - _default_time_start;
	_default_time_start = TIME_RECORD;
#endif // defined(DEFAULT_TIME)
    CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part1"));
}


extern "C"
void GPUCGCMFD_GCEvaluation_Part2(M_INT32 _ProcID, M_INT32 _NumProcs, M_INT32 _isFirstInvoke, M_INT32 _MOC_NumPinsInSubDom, \
	M_INT32 _MOCNumLayersZ, M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, M_INT32 _NumCMFDGroup, M_INT32 _NumCoarseCMFDGroup,\
	M_INT32 _CGCMFDData_MPIComm_Num, M_INT32 _MGCMFDData_MPIComm_Num, FP_PRECISION *_CGCMFDFlux_MPIRecv, FP_PRECISION *_CGCMFDNodeD_MPIRecv,\
	FP_PRECISION *_MGCMFDFlux_MPIRecv, M_INT32 _SubDomLocID_2D_X, M_INT32 _SubDomLocID_2D_Y, FP_PRECISION *_CGCMFDXs_s, \
	M_INT32 *_CGCMFD_NumNoneZeroValue_Matrix, M_INT32 _CGCMFD_LinearSolver, \
	M_INT32 _GmresPrecond_CG, FP_PRECISION *_CG_ILUMatrix, FP_PRECISION *_CGMatrixCSR, M_INT32 *_CG_DiagPointer, M_INT32 *_CG_IW,\
	M_INT32 _ProblemType, FP_PRECISION_TIME *_TimeRec_CGCMFD_XSCollapse, FP_PRECISION_TIME *_TimeRec_CGCMFD_MatrixConst)
{
	// FP_PRECISION _zero = (FP_PRECISION)(M_Zero_FP);
#if defined(DEFAULT_TIME)
	FP_PRECISION_TIME _default_time_start, _default_time_end;
	cudaDeviceSynchronize();
	_default_time_start = TIME_RECORD;
#endif // defined(DEFAULT_TIME)
	int _numEleCmfdNode,_numEleSurf_X,_numEleSurf_Y,_probType;
#if defined(M_CMFDMatrixIncScatt)
	int _numEleCmfdXSs;
	_numEleCmfdXSs = _MOC_NumPinsInSubDom * _NumCoarseCMFDGroup * _NumCoarseCMFDGroup;
#endif 
	cusparseStatus_t _cudaStatus;
	int _pBufferSize_M;
int _pBufferSize_L;
int _pBufferSize_U;
int _pBufferSize;
int _structural_zero;
int _numerical_zero;

#if defined(M_3DMOCex)
	int _numEleSurf_Z;
#endif
	_numEleCmfdNode = _MOC_NumPinsInSubDom * _NumCoarseCMFDGroup;
	
	_numEleSurf_X = _MOCNumLayersZ * (_MOCNumPinsInSubDom_X + 1) * _MOCNumPinsInSubDom_Y * _NumCMFDGroup;
	_numEleSurf_Y = _MOCNumLayersZ * _MOCNumPinsInSubDom_X * (_MOCNumPinsInSubDom_Y + 1) * _NumCMFDGroup;

#if defined(MPI_USED)
    if (_NumProcs>1) {
		cudaMemcpy((void*)GPUCGCMFDFlux_MPIRecv, (void*)_CGCMFDFlux_MPIRecv, (size_t)(_CGCMFDData_MPIComm_Num) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
	    CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - cudaMemcpy - GPUCGCMFDFlux_MPIRecv"));
	    cudaMemcpy((void*)GPUCGCMFDNodeD_MPIRecv, (void*)_CGCMFDNodeD_MPIRecv, (size_t)(_CGCMFDData_MPIComm_Num) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
		CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - cudaMemcpy - GPUCGCMFDNodeD_MPIRecv"));
		cudaMemcpy((void*)GPUMGCMFDFlux_MPIRecv, (void*)_MGCMFDFlux_MPIRecv, (size_t)(_MGCMFDData_MPIComm_Num) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
	    CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - cudaMemcpy - GPUMGCMFDFlux_MPIRecv"));
	}
#endif // defined(MPI_USED)
	
	GPUCMFD_GetNetCurrent_X<<<GPUGridDim_CMFDSurf_X, GPUBlockDim>>>(_numEleSurf_X, GPUCMFDSurfDiffCoeff_X, \
		GPUCMFDSurfNDF_X, GPUCMFDScalarFlux, GPU_MGCMFDNetCurr_X, _SubDomLocID_2D_X, GPUNumSubDomains, \
		GPUMGCMFDFlux_MPIRecv, GPUMGCMFDData_MPIComm_SumInd);
	CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - GPUCMFD_GetNetCurrent_X"));

	GPUCMFD_GetNetCurrent_Y<<<GPUGridDim_CMFDSurf_Y, GPUBlockDim>>>(_numEleSurf_Y, GPUCMFDSurfDiffCoeff_Y, \
		GPUCMFDSurfNDF_Y, GPUCMFDScalarFlux, GPU_MGCMFDNetCurr_Y, _SubDomLocID_2D_Y, GPUNumSubDomains, \
		GPUMGCMFDFlux_MPIRecv, GPUMGCMFDData_MPIComm_SumInd);
	CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - GPUCMFD_GetNetCurrent_Y"));

#if defined(M_3DMOCex)
    _numEleSurf_Z = (_MOCNumLayersZ + 1) * _MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y * _NumCMFDGroup;
	GPUCMFD_GetNetCurrent_Z<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(_numEleSurf_Z, GPUCMFDSurfDiffCoeff_Z, \
		GPUCMFDSurfNDF_Z, GPUCMFDScalarFlux, GPU_MGCMFDNetCurr_Z);
	CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - GPUCMFD_GetNetCurrent_Z"));
#endif // defined(M_3DMOCex)

	_numEleSurf_X = _MOCNumLayersZ * (_MOCNumPinsInSubDom_X + 1) * _MOCNumPinsInSubDom_Y * _NumCoarseCMFDGroup;
	_numEleSurf_Y = _MOCNumLayersZ * _MOCNumPinsInSubDom_X * (_MOCNumPinsInSubDom_Y + 1) * _NumCoarseCMFDGroup;

	GPU_CGCMFDRestriction_Sumation<<<GPUGridDim_CGCMFDSurf_X, GPUBlockDim>>>(GPU_MGCMFDNetCurr_X, \
		GPU_CGMGtoCGCMFD_NetCurr_X, GPU_CGCMFDGroupBound, _numEleSurf_X);
	CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - GPU_CGCMFDRestriction_Sumation-NetCurrX"));
	GPU_CGCMFDRestriction_Sumation<<<GPUGridDim_CGCMFDSurf_Y, GPUBlockDim>>>(GPU_MGCMFDNetCurr_Y, \
		GPU_CGMGtoCGCMFD_NetCurr_Y, GPU_CGCMFDGroupBound, _numEleSurf_Y);
	CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - GPU_CGCMFDRestriction_Sumation-NetCurrY"));

#if defined(M_3DMOCex)
    _numEleSurf_Z = (_MOCNumLayersZ + 1) * _MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y * _NumCoarseCMFDGroup;
    GPU_CGCMFDRestriction_Sumation<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(GPU_MGCMFDNetCurr_Z, \
		GPU_CGMGtoCGCMFD_NetCurr_Z, GPU_CGCMFDGroupBound, _numEleSurf_Z);
	CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - GPU_CGCMFDRestriction_Sumation-NetCurrZ"));
#endif // defined(M_3DMOCex)

	GPUCMFD_RestrictionSurfDiffCoef_X<<<GPUGridDim_CGCMFDSurf_X, GPUBlockDim>>>(_numEleSurf_X, \
		GPU_CGCMFDSurfDiffCoeff_X, GPU_CGCMFDSurfNDF_X, GPU_CGCMFDNodeDiffCoeff, GPU_CGCMFDHetScalFlux, \
		GPU_CGMGtoCGCMFD_NetCurr_X, GPUCellRTMType_SubDom, GPURTMWidth_X, _NumCoarseCMFDGroup, \
		_SubDomLocID_2D_X, GPUNumSubDomains, GPUCGCMFDFlux_MPIRecv, GPUCGCMFDNodeD_MPIRecv, GPUCGCMFDData_MPIComm_SumInd);
	CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - GPUCMFD_RestrictionSurfDiffCoef_X"));

	GPUCMFD_RestrictionSurfDiffCoef_Y<<<GPUGridDim_CGCMFDSurf_Y, GPUBlockDim>>>(_numEleSurf_Y, \
		GPU_CGCMFDSurfDiffCoeff_Y, GPU_CGCMFDSurfNDF_Y, GPU_CGCMFDNodeDiffCoeff, GPU_CGCMFDHetScalFlux, \
		GPU_CGMGtoCGCMFD_NetCurr_Y, GPUCellRTMType_SubDom, GPURTMWidth_Y, _NumCoarseCMFDGroup, \
		_SubDomLocID_2D_Y, GPUNumSubDomains, GPUCGCMFDFlux_MPIRecv, GPUCGCMFDNodeD_MPIRecv, GPUCGCMFDData_MPIComm_SumInd);
	CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - GPUCMFD_RestrictionSurfDiffCoef_Y"));
	
#if defined(M_3DMOCex)
	GPUCMFD_RestrictionSurfDiffCoef_Z<<<GPUGridDim_CGCMFDSurf_Z, GPUBlockDim>>>(_numEleSurf_Z, \
		GPU_CGCMFDSurfDiffCoeff_Z, GPU_CGCMFDSurfNDF_Z, GPU_CGCMFDNodeDiffCoeff, GPU_CGCMFDHetScalFlux, \
		GPU_CGMGtoCGCMFD_NetCurr_Z, GPUCellRTMType_SubDom, GPUWidthLayersInCore, _NumCoarseCMFDGroup);
	CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - GPUCMFD_RestrictionSurfDiffCoef_Z"));
#endif //#if defiend(M_3DMOCex)

#if defined(MPI_USED)
	if (_NumProcs > 1) {
		GPUSetCMFDMPI_InBoundSourCoeff<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(_ProcID, _NumProcs, \
			GPUCGCMFDMPI_InBoundSourCoeff, GPU_CGCMFDSurfDiffCoeff_X, GPU_CGCMFDSurfNDF_X, GPU_CGCMFDSurfDiffCoeff_Y, \
			GPU_CGCMFDSurfNDF_Y, GPU_CGCMFDSurfDiffCoeff_Z, GPU_CGCMFDSurfNDF_Z, _CGCMFDData_MPIComm_Num, \
			GPUCGCMFDMPI_InBoundSourCoeff_Index, GPUCGCMFDMPI_InBoundSourCoeff_Dir, GPUCGCMFDMPI_InBoundSourCoeff_GeoDim);
		CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - GPUSetCMFDMPI_InBoundSourCoeff"));
	}
#endif // defined(MPI_USED)

#if defined(DEFAULT_TIME)
	cudaDeviceSynchronize();
	_default_time_end = TIME_RECORD;
	_TimeRec_CGCMFD_XSCollapse[0] += _default_time_end - _default_time_start;
	_default_time_start = TIME_RECORD;
#endif // defined(DEFAULT_TIME)

#if defined(DIRECT_CSR)
	if (_isFirstInvoke == 1) {

#if defined(M_2DMOC)
		_probType = 0;
#endif //#if defiend(M_2DMOC)

#if defined(M_3DMOCex)
		_probType = 1;
#endif //#if defiend(M_3DMOCex)
		// 
#if defined(M_CMFDMatrixIncScatt)
		cudaMemcpy((void*)_CGCMFDXs_s, (void*)GPU_CGCMFDXs_s, (size_t)(_numEleCmfdXSs) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
#endif // #if defined(M_CMFDMatrixIncScatt)

		_CGCMFD_NumNoneZeroValue_Matrix[0] = GPUCMFD_GetNumNoneZeroEle_MatrixA(_MOCNumLayersZ, \
			_MOCNumPinsInSubDom_X, _MOCNumPinsInSubDom_Y, _NumCoarseCMFDGroup, _CGCMFDXs_s, _probType);
		CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - GPUCMFD_GetNumNoneZeroEle_MatrixA"));

		cudaMemcpyToSymbol(GPU_CGCMFDNumNoneZeroValue_Matrix, (void*)_CGCMFD_NumNoneZeroValue_Matrix, sizeof(int), 0, cudaMemcpyHostToDevice);
	    cudaMemcpyToSymbol(GPU_CGCMFDNumValueDiag_Matrix, (void*)&_numEleCmfdNode, sizeof(int), 0, cudaMemcpyHostToDevice);

		GPU_CGCMFDMatrixCSR_AllocMem(_numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0], _CGCMFD_LinearSolver);
		
		GPUCMFD_GetNoneZeroIndexInfo_MatrixCSR(_MOCNumLayersZ, _MOCNumPinsInSubDom_X, _MOCNumPinsInSubDom_Y, \
			_NumCoarseCMFDGroup, _CGCMFDXs_s, Dev_CGCMFDMatrixCSR_RowStartInd, Dev_CGCMFDMatrixCSR_ColInd, _ProblemType);
		CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - GPUCMFD_GetNoneZeroIndexInfo_MatrixCSR"));

		cudaMemcpy((void*)GPU_CGCMFDMatrixCSR_RowStartInd, (void*)Dev_CGCMFDMatrixCSR_RowStartInd, (size_t)((_numEleCmfdNode + 1)) * sizeof(int), cudaMemcpyHostToDevice);

		cudaMemcpy((void*)GPU_CGCMFDMatrixCSR_ColInd, (void*)Dev_CGCMFDMatrixCSR_ColInd, (size_t)(_CGCMFD_NumNoneZeroValue_Matrix[0]) * sizeof(int), cudaMemcpyHostToDevice);

		GPU_DiagPointer_CSRMatrix<<<GPUGridDim_CGCMFDNode, GPUBlockDim>>>(_numEleCmfdNode, GPU_CGCMFDMatrixCSR_RowStartInd, \
			GPU_CGCMFDMatrixCSR_ColInd, GPU_CGCMFDDiagPointer);
		CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - GPU_DiagPointer_CSRMatrix"));
	}

	GPUCMFD_ConstructMatrixA_CSR<<<GPUGridDim_CGCMFDNode, GPUBlockDim>>>(GPUCellRTMType_SubDom,GPU_CGCMFDSurfDiffCoeff_X, \
	    GPU_CGCMFDSurfDiffCoeff_Y, GPU_CGCMFDSurfDiffCoeff_Z, GPU_CGCMFDSurfNDF_X, GPU_CGCMFDSurfNDF_Y, GPU_CGCMFDSurfNDF_Z, \
		GPU_CGCMFDXs_t, GPU_CGCMFDXs_s, GPURTMWidth_X, GPURTMWidth_Y,GPUWidthLayersInCore, GPU_CGCMFDMatrixCSR_RowStartInd, \
		GPU_CGCMFDMatrixCSR, _numEleCmfdNode, _NumCoarseCMFDGroup);
	CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - GPUCMFD_ConstructMatrixA_CSR"));
#endif // defined(DIRECT_CSR)
	
    if (_CGCMFD_LinearSolver == 2) {
	    if (_GmresPrecond_CG) {
// #if defined(M_GMRES_JacobiPRE)
// 			GPU_VecInverse<<<GPUGridDim_CGCMFDNode, GPUBlockDim>>>(_numEleCmfdNode,GPU_CGCMFDMatrixCSR_Diag);
// #endif //defined(M_GMRES_JacobiPRE)
#if defined(M_GMRES_IluPRE)
 #if defined(M_GMRES_PRE_cuSparseLib)
   if (_GmresPrecond_CG) {
	    cudaMemcpy((void*)GPU_CGCMFDMatrixCSR_ILUFactor, (void*)GPU_CGCMFDMatrixCSR, \
		(size_t)(_CGCMFD_NumNoneZeroValue_Matrix[0]) * sizeof(FP_PRECISION), cudaMemcpyDeviceToDevice);
	}
	if (_GmresPrecond_CG && _isFirstInvoke == 1) {
		_cudaStatus = cusparseCreate(&CG_cusparseHandle);
		// step 1: create a descriptor which contains
		// - matrix M is base-0
		// - matrix L is base-0
		// - matrix L is lower triangular
		// - matrix L has unit diagonal
		// - matrix U is base-0
		// - matrix U is upper triangular
		// - matrix U has non-unit diagonal
		cusparseCreateMatDescr(&CG_descr_M);
		cusparseSetMatIndexBase(CG_descr_M, CUSPARSE_INDEX_BASE_ZERO);
		// cusparseSetMatIndexBase(CG_descr_M, CUSPARSE_INDEX_BASE_ONE);
		cusparseSetMatType(CG_descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);
		CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseCreateMatDescr-CG_descr_M"));

		cusparseCreateMatDescr(&CG_descr_L);
		cusparseSetMatIndexBase(CG_descr_L, CUSPARSE_INDEX_BASE_ZERO);
		// cusparseSetMatIndexBase(CG_descr_L, CUSPARSE_INDEX_BASE_ONE);
		cusparseSetMatType(CG_descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatFillMode(CG_descr_L, CUSPARSE_FILL_MODE_LOWER);
		cusparseSetMatDiagType(CG_descr_L, CUSPARSE_DIAG_TYPE_UNIT);
		CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseCreateMatDescr-CG_descr_L"));

		cusparseCreateMatDescr(&CG_descr_U);
		cusparseSetMatIndexBase(CG_descr_U, CUSPARSE_INDEX_BASE_ZERO);
		// cusparseSetMatIndexBase(CG_descr_U, CUSPARSE_INDEX_BASE_ONE);
		cusparseSetMatType(CG_descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatFillMode(CG_descr_U, CUSPARSE_FILL_MODE_UPPER);
		cusparseSetMatDiagType(CG_descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);
		CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseCreateMatDescr-CG_descr_U"));

		// step 2: create a empty info structure
		// we need one info for csrilu02 and two info's for csrsv2
		cusparseCreateCsrilu02Info(&CG_info_M);
		cusparseCreateCsrsv2Info(&CG_info_L);
		cusparseCreateCsrsv2Info(&CG_info_U);
		CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseCreateCsrilu02Info-CG_info_M"));

		// step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
#if defined(DOUBLE)
		cusparseDcsrilu02_bufferSize(CG_cusparseHandle, _numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0],
			CG_descr_M, GPU_CGCMFDMatrixCSR_ILUFactor, GPU_CGCMFDMatrixCSR_RowStartInd, GPU_CGCMFDMatrixCSR_ColInd, \
			CG_info_M, &_pBufferSize_M);
		cusparseDcsrsv2_bufferSize(CG_cusparseHandle, CG_trans_L, _numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0],
			CG_descr_L, GPU_CGCMFDMatrixCSR_ILUFactor, GPU_CGCMFDMatrixCSR_RowStartInd, GPU_CGCMFDMatrixCSR_ColInd, \
			CG_info_L, &_pBufferSize_L);
		cusparseDcsrsv2_bufferSize(CG_cusparseHandle, CG_trans_U, _numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0],
			CG_descr_U, GPU_CGCMFDMatrixCSR_ILUFactor, GPU_CGCMFDMatrixCSR_RowStartInd, GPU_CGCMFDMatrixCSR_ColInd, \
			CG_info_U, &_pBufferSize_U);
#endif //defined(DOUBLE)

#if defined(SINGLE)
		cusparseScsrilu02_bufferSize(CG_cusparseHandle, _numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0],
			CG_descr_M, GPU_CGCMFDMatrixCSR_ILUFactor, GPU_CGCMFDMatrixCSR_RowStartInd, GPU_CGCMFDMatrixCSR_ColInd, \
			CG_info_M, &_pBufferSize_M);
		cusparseScsrsv2_bufferSize(CG_cusparseHandle, CG_trans_L, _numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0],
				CG_descr_L, GPU_CGCMFDMatrixCSR_ILUFactor, GPU_CGCMFDMatrixCSR_RowStartInd, GPU_CGCMFDMatrixCSR_ColInd, \
				CG_info_L, &_pBufferSize_L);
		cusparseScsrsv2_bufferSize(CG_cusparseHandle, CG_trans_U, _numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0],
				CG_descr_U, GPU_CGCMFDMatrixCSR_ILUFactor, GPU_CGCMFDMatrixCSR_RowStartInd, GPU_CGCMFDMatrixCSR_ColInd, \
				CG_info_U, &_pBufferSize_U);
#endif //defined(SINGLE)
		CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseDcsrilu02_bufferSize"));

		_pBufferSize = max(_pBufferSize_M, max(_pBufferSize_L, _pBufferSize_U));
		// pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
		cudaMalloc((void**)&CG_pBuffer, _pBufferSize);
		CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cudaMalloc-CG_pBuffer"));

		// step 4: perform analysis of incomplete Cholesky on M
		//         perform analysis of triangular solve on L
		//         perform analysis of triangular solve on U
		// The lower(upper) triangular part of M has the same sparsity pattern as L(U),
		// we can do analysis of csrilu0 and csrsv2 simultaneously.
#if defined(DOUBLE)
		cusparseDcsrilu02_analysis(CG_cusparseHandle, _numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0], CG_descr_M,
			GPU_CGCMFDMatrixCSR_ILUFactor, GPU_CGCMFDMatrixCSR_RowStartInd, GPU_CGCMFDMatrixCSR_ColInd, CG_info_M,
			CG_policy_M, CG_pBuffer);
		CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseDcsrilu02_analysis"));

		_cudaStatus = cusparseXcsrilu02_zeroPivot(CG_cusparseHandle, CG_info_M, &_structural_zero);
		if (CUSPARSE_STATUS_ZERO_PIVOT == _cudaStatus){
			printf("A(%d,%d) is missing\n", _structural_zero, _structural_zero);
		}

		cusparseDcsrsv2_analysis(CG_cusparseHandle, CG_trans_L, _numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0], CG_descr_L,
			GPU_CGCMFDMatrixCSR_ILUFactor, GPU_CGCMFDMatrixCSR_RowStartInd, GPU_CGCMFDMatrixCSR_ColInd,
			CG_info_L, CG_policy_L, CG_pBuffer);
		CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseDcsrsv2_analysis-CG_trans_L"));

		cusparseDcsrsv2_analysis(CG_cusparseHandle, CG_trans_U, _numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0], CG_descr_U,
			GPU_CGCMFDMatrixCSR_ILUFactor, GPU_CGCMFDMatrixCSR_RowStartInd, GPU_CGCMFDMatrixCSR_ColInd,
			CG_info_U, CG_policy_U, CG_pBuffer);
		CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseDcsrsv2_analysis-CG_trans_U"));
#endif //defined(DOUBLE)

#if defined(SINGLE)
		cusparseScsrilu02_analysis(CG_cusparseHandle, _numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0], CG_descr_M,
			GPU_CGCMFDMatrixCSR_ILUFactor, GPU_CGCMFDMatrixCSR_RowStartInd, GPU_CGCMFDMatrixCSR_ColInd, CG_info_M,
			CG_policy_M, CG_pBuffer);
		CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseDcsrilu02_analysis"));

		_cudaStatus = cusparseXcsrilu02_zeroPivot(CG_cusparseHandle, CG_info_M, &_structural_zero);
		if (CUSPARSE_STATUS_ZERO_PIVOT == _cudaStatus){
			printf("A(%d,%d) is missing\n", _structural_zero, _structural_zero);
		}

		cusparseScsrsv2_analysis(CG_cusparseHandle, CG_trans_L, _numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0], CG_descr_L,
			GPU_CGCMFDMatrixCSR_ILUFactor, GPU_CGCMFDMatrixCSR_RowStartInd, GPU_CGCMFDMatrixCSR_ColInd,
			CG_info_L, CG_policy_L, CG_pBuffer);
		CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseScsrsv2_analysis-CG_trans_L"));

		cusparseScsrsv2_analysis(CG_cusparseHandle, CG_trans_U, _numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0], CG_descr_U,
			GPU_CGCMFDMatrixCSR_ILUFactor, GPU_CGCMFDMatrixCSR_RowStartInd, GPU_CGCMFDMatrixCSR_ColInd,
			CG_info_U, CG_policy_U, CG_pBuffer);
		CUDA_CheckError((char *)("GPUCMFD_GCEvaluation_Part2-cusparseScsrsv2_analysis-CG_trans_U"));
#endif //defined(SINGLE)
// 	    cusparseCreate(&CG_cusparseHandle);
// 	    cusparseCreateMatDescr(&CG_descra);
// 	    cusparseCreateMatDescr(&CG_descrm);
// 	    cusparseSetMatType(CG_descra,CUSPARSE_MATRIX_TYPE_GENERAL);
// 	    cusparseSetMatType(CG_descrm,CUSPARSE_MATRIX_TYPE_GENERAL);
// 	    cusparseCreateSolveAnalysisInfo(&CG_info_l);
// 	    cusparseCreateSolveAnalysisInfo(&CG_info_u);

// 	    cusparseSetMatFillMode(CG_descrm,CUSPARSE_FILL_MODE_LOWER);
// 		cusparseSetMatDiagType(CG_descrm,CUSPARSE_DIAG_TYPE_UNIT);
// #if defined(DOUBLE)
// 	    cusparseDcsrsv_analysis(CG_cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,_numEleCmfdNode,\
// 	    	_CGCMFD_NumNoneZeroValue_Matrix,CG_descrm,GPU_CGCMFDMatrixCSR_ILUFactor, \
// 			GPU_CGCMFDMatrixCSR_RowStartInd,GPU_CGCMFDMatrixCSR_ColInd,CG_info_l);
// #endif

// #if defined(SINGLE)
// 	    cusparseScsrsv_analysis(CG_cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,_numEleCmfdNode,\
// 	    	_CGCMFD_NumNoneZeroValue_Matrix,CG_descrm,GPU_CGCMFDMatrixCSR_ILUFactor, \
// 			GPU_CGCMFDMatrixCSR_RowStartInd,GPU_CGCMFDMatrixCSR_ColInd,CG_info_l);
// #endif
// 	    cudaDeviceSynchronize();

// 	    cusparseSetMatFillMode(CG_descrm,CUSPARSE_FILL_MODE_UPPER);
// 	   cusparseSetMatDiagType(CG_descrm,CUSPARSE_DIAG_TYPE_NON_UNIT);
// #if defined(DOUBLE)
//        cusparseDcsrsv_analysis(CG_cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,_numEleCmfdNode,\
// 		    _CGCMFD_NumNoneZeroValue_Matrix,CG_descrm,GPU_CGCMFDMatrixCSR_ILUFactor, \
// 			GPU_CGCMFDMatrixCSR_RowStartInd,GPU_CGCMFDMatrixCSR_ColInd,CG_info_u);
// #endif

// #if defined(SINGLE)
//        cusparseScsrsv_analysis(CG_cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,_numEleCmfdNode,\
// 		    _CGCMFD_NumNoneZeroValue_Matrix,CG_descrm,GPU_CGCMFDMatrixCSR_ILUFactor, \
// 			GPU_CGCMFDMatrixCSR_RowStartInd,GPU_CGCMFDMatrixCSR_ColInd,CG_info_u);
// #endif
// 	    cudaDeviceSynchronize();
	}
	if (_GmresPrecond_CG) {
		// step 5: M = L * U
#if defined(DOUBLE)
		cusparseDcsrilu02(CG_cusparseHandle, _numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0], CG_descr_M,
			GPU_CGCMFDMatrixCSR_ILUFactor, GPU_CGCMFDMatrixCSR_RowStartInd, GPU_CGCMFDMatrixCSR_ColInd, \
			CG_info_M, CG_policy_M, CG_pBuffer);
#endif // defined(DOUBLE)
#if defined(SINGLE)
		cusparseScsrilu02(CG_cusparseHandle, _numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0], CG_descr_M,
			GPU_CGCMFDMatrixCSR_ILUFactor, GPU_CGCMFDMatrixCSR_RowStartInd, GPU_CGCMFDMatrixCSR_ColInd, \
			CG_info_M, CG_policy_M, CG_pBuffer);
#endif // defined(DOUBLE)
	    _cudaStatus = cusparseXcsrilu02_zeroPivot(CG_cusparseHandle, CG_info_M, &_numerical_zero);
		if (CUSPARSE_STATUS_ZERO_PIVOT == _cudaStatus){
			printf("U(%d,%d) is zero\n", _numerical_zero, _numerical_zero);
		}
// #if defined(DOUBLE)
// 		cusparseDcsrilu0(CG_cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,_numEleCmfdNode, CG_descra, \
// 			GPU_CGCMFDMatrixCSR_ILUFactor, GPU_CGCMFDMatrixCSR_RowStartInd,GPU_CGCMFDMatrixCSR_ColInd,CG_info_l);
// #endif

// #if defined(SINGLE)
// 		cusparseScsrilu0(CG_cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,_numEleCmfdNode, CG_descra, \
// 			GPU_CGCMFDMatrixCSR_ILUFactor, GPU_CGCMFDMatrixCSR_RowStartInd,GPU_CGCMFDMatrixCSR_ColInd,CG_info_l);
// #endif
	}
	
 #else //!defined(M_GMRES_PRE_cuSparseLib), the preconditioning is performed on CPU
   if (_GmresPrecond_CG && _isFirstInvoke == 1) {
		_CG_ILUMatrix = (FP_PRECISION *)calloc(_CGCMFD_NumNoneZeroValue_Matrix[0], sizeof(FP_PRECISION));
		_CGMatrixCSR = (FP_PRECISION *)calloc(_CGCMFD_NumNoneZeroValue_Matrix[0], sizeof(FP_PRECISION));
	}
   if (_GmresPrecond_CG) {
		cudaMemcpy((void*)_CGMatrixCSR, (void*)GPU_CGCMFDMatrixCSR, \
		(size_t)(_CGCMFD_NumNoneZeroValue_Matrix[0]) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		GPU_DiagPointer_CSRMatrix(_numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0], Dev_CGCMFDMatrixCSR_RowStartInd, \
			Dev_CGCMFDMatrixCSR_ColInd, _CG_DiagPointer);
	    GPU_ILUFactPrecond(_numEleCmfdNode, _CGCMFD_NumNoneZeroValue_Matrix[0], Dev_CGCMFDMatrixCSR_RowStartInd, \
			Dev_CGCMFDMatrixCSR_ColInd, _CGMatrixCSR, _CG_DiagPointer, _CG_ILUMatrix, _CG_IW);
   }
 #endif //defined(M_GMRES_PRE_cuSparseLib)
#endif // defined(M_GMRES_IluPRE)
		}
	}

#if defined(DEFAULT_TIME)
   cudaDeviceSynchronize();
	_default_time_end = TIME_RECORD;
	_TimeRec_CGCMFD_MatrixConst[0] += _default_time_end - _default_time_start;
#endif // defined(DEFAULT_TIME)

	cudaMemcpy((void*)GPU_CGCMFDScalarFlux, (void*)GPU_CGCMFDHetScalFlux, \
	(size_t)(_numEleCmfdNode) * sizeof(FP_PRECISION), cudaMemcpyDeviceToDevice);
	CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2 - cudaMemcpy-GPU_CGCMFDScalarFlux"));

	CUDA_CheckError((char *)("GPUCGCMFD_GCEvaluation_Part2"));
}

extern "C"
void GPU_MGCMFDSolver_Part1(M_INT32 *_MGCMFD_Diver, M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCMFDGroup)
{
	_MGCMFD_Diver[0] = 0;

	GPUCMFD_EvaluateTotalFissQ<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(GPU_MGCMFDTotalFissQ, \
		GPUCMFDXs_nf, GPUCMFDScalarFlux, _MOC_NumPinsInSubDom, _NumCMFDGroup, GPUCMFDNodeVol);
}

extern "C"
void GPU_MGCMFDSolver_Part2(M_INT32 _procID, M_INT32 _iter_moc, M_INT32 _iter_mg, FP_PRECISION _k, M_INT32 _ProblemType,\
	M_INT32 *_MGCMFD_FluxFlag, M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCMFDGroup, FP_PRECISION *_CMFDScalarFlux, int _isTransient, M_INT32 _NumProcs)
{
	int _number;

#if defined(TIME_REC)
    FP_PRECISION_TIME _time_start, _time_end;
#endif // TIME_REC

	_number = _MOC_NumPinsInSubDom * _NumCMFDGroup;

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
	 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#if defined(TIME_REC)
	cudaDeviceSynchronize();
	_time_start = TIME_RECORD;
#endif // defined(TIME_REC)
	 _MGCMFD_FluxFlag[0] = 0;

	 cudaMemcpy((void*)GPUCMFDScalarFlux_Old, (void*)GPUCMFDScalarFlux, \
	     (size_t)(_number) * sizeof(FP_PRECISION), cudaMemcpyDeviceToDevice);
	 CUDA_CheckError((char *)("cudaMemcpy - update GPUCMFDScalarFlux"));
#if defined(TIME_REC)
	 cudaDeviceSynchronize();
	 if (_iter_moc == 0 && _iter_mg==0) {
		 TimeMGStamp4=TimeCountMGCMFD;
		 strcpy(TimeMetricsMG[TimeMGStamp4], "MG - old flux update");
		 TimeCountMGCMFD++;
	 }
	 _time_end = TIME_RECORD;
	 TimeRecMG[TimeMGStamp4] += _time_end-_time_start;
#endif // TIME_REC
#if defined(TIME_REC)
	cudaDeviceSynchronize();
	_time_start = TIME_RECORD;
#endif // defined(TIME_REC)
	 if (_ProblemType == 0) {
#if defined(M_CMFDMatrixIncScatt)
	     // todo: calc total source of transient CMFD
         if (_isTransient < 2){
             GPUCMFD_EvaluateFissionSource_onFissQ<<<GPUGridDim_CMFDNode, GPUBlockDim>>>(GPU_MGCMFDTotalFissQ, \
			 GPUCMFDTotalSource, GPUCMFD_Chi, 1.0 / _k, _number, _NumCMFDGroup);
             CUDA_CheckError((char *)("GPU_MGCMFDSolver_Part2 - GPUCMFD_EvaluateFissionSource_onFissQ"));
         } else{
             tranGPUCMFD_EvaluateFissionSource_onFissQ<<<GPUGridDim_CMFDNode, GPUBlockDim>>>(GPU_MGCMFDTotalFissQ,
             GPUCMFDTotalSource, GPUCMFD_Chi, GPUCMFDTranCoeffA, GPUCMFDTranCoeffB, GPUCMFDTranCoeffC, GPUCMFDScalarFlux,
             GPUCMFDNodeVol, 1.0 / _k, _number, _NumCMFDGroup);
             CUDA_CheckError((char *) ("GPU_MGCMFDSolver_Part2 - tranGPUCMFD_EvaluateFissionSource_onFissQ"));
         }

#else
		 GPUCMFD_EvaluateTotalSource_onFissQ<<<GPUGridDim_CMFDNode, GPUBlockDim>>>(GPU_MGCMFDTotalFissQ, GPUCMFDTotalSource, \
			 GPUCMFD_Chi, GPUCMFDXs_s, GPUCMFDScalarFlux, 1.0 / _k, _number, _NumCMFDGroup, GPUCMFDNodeVol);
		 CUDA_CheckError((char *)("GPU_MGCMFDSolver_Part2 - GPUCMFD_EvaluateFissionSource_onFissQ"));
#endif //#if defined(M_CMFDMatrixIncScatt)
	 }
#if defined(TIME_REC)
	 cudaDeviceSynchronize();
	 if (_iter_moc == 0 && _iter_mg==0) {
		 TimeMGStamp5=TimeCountMGCMFD;
		 strcpy(TimeMetricsMG[TimeMGStamp5], "MG - SourEva-1");
		 TimeCountMGCMFD++;
	 }
	 _time_end = TIME_RECORD;
	 TimeRecMG[TimeMGStamp5] += _time_end-_time_start;
#endif // TIME_REC

#if defined(MPI_USED)
#if defined(TIME_REC)
	cudaDeviceSynchronize();
	_time_start = TIME_RECORD;
#endif // defined(TIME_REC)
    if (_NumProcs>1) {
		cudaMemcpy((void*)_CMFDScalarFlux, (void*)GPUCMFDScalarFlux, (size_t)(_number) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPU_MGCMFDSolver_Part2 - cudaMemcpy - GPUCMFDScalarFlux"));
	}
#if defined(TIME_REC)
	cudaDeviceSynchronize();
	if (_iter_moc == 0 && _iter_mg==0) {
		TimeMGStamp6=TimeCountMGCMFD;
		strcpy(TimeMetricsMG[TimeMGStamp6], "MG - MPI Comm flux G->C");
		TimeCountMGCMFD++;
	}
	_time_end = TIME_RECORD;
	TimeRecMG[TimeMGStamp6] += _time_end-_time_start;
#endif // TIME_REC
#endif // defined(MPI_USED)

// #if defined(DEFAULT_TIME)
// 	cudaDeviceSynchronize();
// #endif // TIME_REC
}

extern "C"
int GPU_MGCMFDSolver_Part3(M_INT32 _iter_moc, M_INT32 _iter_mg, FP_PRECISION _k, M_INT32 _ProcID, M_INT32 _NumProcs, \
	M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCMFDGroup, FP_PRECISION *_MGCMFDFlux_MPIRecv, M_INT32 _MGCMFDData_MPIComm_Num, \
	M_INT32 _MGCMFD_LinearSolver, M_INT32 _MaxNumIter_MGLS, FP_PRECISION _ConvCritLinearSolver_MGCMFD, M_INT32 _MGCMFD_checkFluxNormInf, \
	M_INT32 _MGCMFD_checkFluxNorm2, M_INT32 _MGCMFD_checkFluxRMSE, FP_PRECISION *_MGCMFDLS_NormInf, FP_PRECISION *_MGCMFDLS_Norm2, \
	FP_PRECISION *_MGCMFDLS_RMSE, FP_PRECISION *_MG_ILUMatrix, M_INT32 *_MG_DiagPointer, M_INT32 _GmresPrecond_MG,\
	FP_PRECISION _SOR_Factor_MG, M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, FP_PRECISION_TIME *_Time_MGCMFD_GMRES_Precond, \
	FP_PRECISION_TIME *_Time_MGCMFD_GMRES_LeastSqr, FP_PRECISION_TIME *_Time_MGCMFD_GMRES_MemCpy, FP_PRECISION_TIME *_Time_MGCMFD_GMRES_SpMV, \
	FP_PRECISION_TIME *_Time_MGCMFD_GMRES_BLAS1, float *_MGCMFD_NegFluxCount, M_INT32 *_MGCMFD_FluxFlag, FP_PRECISION_TIME *_TimeRec_MGCMFD_LinearSolver,\
	M_INT32 _CMFD_NumNoneZeroValue_Matrix, FP_PRECISION *_MG_Residual, M_INT32 _GmresKrylovDim_MG)
{
	int _number, _numLSIter, _numCMFDGroup;

#if defined(TIME_REC)
    FP_PRECISION_TIME _time_start, _time_end;
#endif // TIME_REC

	cublasHandle_t _cublasHandle;
	// cusparseOperation_t _transA=CUSPARSE_OPERATION_TRANSPOSE;
	// CUDA_CheckError((char *)("cudaMemcpy-cusparseOperation_t-_transA"));

#if defined(DEFAULT_TIME)
    FP_PRECISION_TIME _default_time_start, _default_time_end;
	cudaDeviceSynchronize();
#endif // TIME_REC
	cublasCreate(&_cublasHandle);

	_numCMFDGroup = _NumCMFDGroup;

	_number = _MOC_NumPinsInSubDom * _numCMFDGroup;

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#if defined(MPI_USED)
    if (_NumProcs > 1) {
#if defined(TIME_REC)
		cudaDeviceSynchronize();
		_time_start = TIME_RECORD;
#endif // defined(TIME_REC)
		cudaMemcpy((void*)GPUMGCMFDFlux_MPIRecv, (void*)_MGCMFDFlux_MPIRecv, (size_t)(_MGCMFDData_MPIComm_Num) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
		CUDA_CheckError((char *)("GPU_MGCMFDSolver_Part3 - cudaMemcpy - GPUMGCMFDFlux_MPIRecv"));
#if defined(TIME_REC)
		cudaDeviceSynchronize();
		if (_iter_moc == 0 && _iter_mg==0) {
			TimeMGStamp8=TimeCountMGCMFD;
			strcpy(TimeMetricsMG[TimeMGStamp8], "MG - MPI Comm flux C->G");
			TimeCountMGCMFD++;
		}
		_time_end = TIME_RECORD;
		TimeRecMG[TimeMGStamp8] += _time_end-_time_start;
#endif // TIME_REC
#if defined(TIME_REC)
		cudaDeviceSynchronize();
		_time_start = TIME_RECORD;
#endif // defined(TIME_REC)
		GPUSetCMFDTotSour_wInBoundInfo<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(_ProcID, _NumProcs, _MGCMFDData_MPIComm_Num, \
			GPUMGCMFDData_MPIComm_Index, GPUCMFDTotalSource, GPUMGCMFDMPI_InBoundSourCoeff, GPUMGCMFDFlux_MPIRecv);
		
#if defined(TIME_REC)
		cudaDeviceSynchronize();
		if (_iter_moc == 0 && _iter_mg==0) {
			TimeMGStamp9=TimeCountMGCMFD;
			strcpy(TimeMetricsMG[TimeMGStamp9], "MG - SourEva-2");
			TimeCountMGCMFD++;
		}
		_time_end = TIME_RECORD;
		TimeRecMG[TimeMGStamp9] += _time_end-_time_start;
#endif // TIME_REC
	}
#endif // defined(MPI_USED)
	
// }
#if defined(DEFAULT_TIME)
	 cudaDeviceSynchronize();
	 _default_time_start = TIME_RECORD;
#endif // defined(DEFAULT_TIME)
#if defined(TIME_REC)
	cudaDeviceSynchronize();
	_time_start = TIME_RECORD;
#endif // defined(TIME_REC)
	 if (_MGCMFD_LinearSolver == 0) {//
		 _numLSIter = GPULinearSolver_Jacobi(GPUCMFDScalarFlux, GPUCMFDScalarFlux_Old_LS, GPUCMFDTotalSource, \
			 GPU_MGCMFDMatrixCSR, GPU_MGCMFDDiagPointer, GPU_MGCMFDMatrixCSR_RowStartInd, GPU_MGCMFDMatrixCSR_ColInd, \
			 _number, GPUCMFDFluxRelErr, _MaxNumIter_MGLS, _ConvCritLinearSolver_MGCMFD, _cublasHandle, GPUGridDim_CMFDNode,\
			 _MGCMFD_checkFluxNormInf, _MGCMFD_checkFluxNorm2, _MGCMFD_checkFluxRMSE, _MGCMFDLS_NormInf, _MGCMFDLS_Norm2, \
	         _MGCMFDLS_RMSE);
	 }
	 else if (_MGCMFD_LinearSolver == 1) {
		 _numLSIter = GPULinearSolver_BSOR(_numCMFDGroup, GPUCMFDScalarFlux, GPUCMFDScalarFlux_Old_LS, GPUCMFDTotalSource, \
			 GPU_MGCMFDMatrixCSR, GPU_MGCMFDDiagPointer, GPU_MGCMFDMatrixCSR_RowStartInd, GPU_MGCMFDMatrixCSR_ColInd, _number, \
			 GPUCMFDFluxRelErr, _MaxNumIter_MGLS, _ConvCritLinearSolver_MGCMFD, _cublasHandle, _SOR_Factor_MG, GPU_MGCMFDNumRedThreads, \
			 GPU_MGCMFDNumBlackThreads, GPUGridDim_MGCMFDRed, GPUGridDim_MGCMFDBlack, GPUGridDim_CMFDNode, _MOCNumPinsInSubDom_X, \
			 _MOCNumPinsInSubDom_Y, GPUNumThreadBlocks, GPUNumThreadPerBlock, GPUNumRedLay_OddNx, GPUNumBlackLay_OddNx,\
			 _MGCMFD_checkFluxNormInf, _MGCMFD_checkFluxNorm2, _MGCMFD_checkFluxRMSE, _MGCMFDLS_NormInf, _MGCMFDLS_Norm2, \
	         _MGCMFDLS_RMSE);
	 }
	 else if (_MGCMFD_LinearSolver == 2) {
		 _numLSIter = GPULinearSolver_GMRES(_number, _CMFD_NumNoneZeroValue_Matrix, _GmresKrylovDim_MG, _MaxNumIter_MGLS, \
			 GPU_MGCMFDMatrixCSR, /*GPU_MGCMFDMatrixCSR_Diag*/NULL, GPU_MGCMFDMatrixCSR_ILUFactor, _MG_ILUMatrix, GPU_MGCMFDMatrixCSR_ColInd, \
			 Dev_MGCMFDMatrixCSR_ColInd, GPU_MGCMFDMatrixCSR_RowStartInd, Dev_MGCMFDMatrixCSR_RowStartInd, _MG_DiagPointer, \
			 GPUCMFDScalarFlux, GPUCMFDTotalSource, _ConvCritLinearSolver_MGCMFD, _GmresPrecond_MG, _cublasHandle, MG_cusparseHandle, \
			 GPU_MGGMRES_r0, _MG_Residual, GPU_MGGMRES_res, GPU_MGGMRES_beta, GPU_MGGMRES_u_base, GPU_MGGMRES_g, GPU_MGGMRES_e, \
			 GPU_MGGMRES_Hm, GPU_MGGMRES_w, GPU_MGGMRES_y, GPU_MGGMRES_Wm, GPU_MGGmres_blocksPerGrid, _Time_MGCMFD_GMRES_Precond, \
			 _Time_MGCMFD_GMRES_LeastSqr, _Time_MGCMFD_GMRES_MemCpy, _Time_MGCMFD_GMRES_SpMV, _Time_MGCMFD_GMRES_BLAS1,_MGCMFDLS_RMSE, \
			 MG_trans_L, MG_trans_U, MG_descr_L, MG_descr_U, MG_info_L, MG_info_U, MG_policy_L, MG_policy_U, MG_pBuffer);
		 CUDA_CheckError((char *)("GPU_MGCMFDSolver_Part3-GPULinearSolver_GMRES"));
	 }
	 else {}

	 GPU_SetNegCMFDFlux_MG(_number, GPUNumThreadBlocks, GPUNumThreadPerBlock, _MGCMFD_NegFluxCount, _MGCMFD_FluxFlag);
	 CUDA_CheckError((char *)("GPU_MGCMFDSolver_Part3 - GPU_SetNegCMFDFlux_MG"));

#if defined(TIME_REC)
	 cudaDeviceSynchronize();
	 if (_iter_moc == 0 && _iter_mg==0) {
		 TimeMGStamp10=TimeCountMGCMFD;
		 strcpy(TimeMetricsMG[TimeMGStamp10], "MG - LS solver");
		 TimeCountMGCMFD++;
	 }
	 _time_end = TIME_RECORD;
	 TimeRecMG[TimeMGStamp10] += _time_end-_time_start;
#endif // TIME_REC

#if defined(DEFAULT_TIME)
	 cudaDeviceSynchronize();
	 _default_time_end = TIME_RECORD;
	 _TimeRec_MGCMFD_LinearSolver[0] += _default_time_end - _default_time_start;
#endif // defined(DEFAULT_TIME)

	cublasDestroy(_cublasHandle);
	CUDA_CheckError((char *)("GPU_MGCMFDSolver_Part3 - cublasDestroy"));

#if defined(DEFAULT_TIME)
	cudaDeviceSynchronize();
#endif // TIME_REC

    return (_numLSIter);
}

extern "C"
void GPU_CGCMFDSolver_Part1(M_INT32 *_CGCMFD_Diver, M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCoarseCMFDGroup)
{
	_CGCMFD_Diver[0] = 0;

	GPUCMFD_EvaluateTotalFissQ<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(GPU_CGCMFDTotalFissQ, \
		GPU_CGCMFDXs_nf, GPU_CGCMFDScalarFlux, _MOC_NumPinsInSubDom, _NumCoarseCMFDGroup, GPUCMFDNodeVol);
}

extern "C"
void GPU_CGCMFDSolver_Part2(M_INT32 _procID, M_INT32 _iter_moc, M_INT32 _iter_mg, M_INT32 _iter_cg, FP_PRECISION _k,\
	M_INT32 _ProblemType, M_INT32 *_CGCMFD_FluxFlag, M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCoarseCMFDGroup, \
	FP_PRECISION *_CGCMFDScalarFlux, M_INT32 _NumProcs)
{
	int _number, _numCMFDGroup;
#if defined(TIME_REC)
	FP_PRECISION_TIME _time_start, _time_end;
	cudaDeviceSynchronize();
	_time_start = TIME_RECORD;
#endif // TIME_REC
	_numCMFDGroup = _NumCoarseCMFDGroup;

	_number = _MOC_NumPinsInSubDom * _numCMFDGroup;

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
	 _CGCMFD_FluxFlag[0] = 0;
	 cudaMemcpy((void*)GPU_CGCMFDScalarFlux_Old, (void*)GPU_CGCMFDScalarFlux, (size_t)(_number) * sizeof(FP_PRECISION), cudaMemcpyDeviceToDevice);
	 CUDA_CheckError((char *)("GPU_CGCMFDSolver_Part2 - update GPU_CGCMFDScalarFlux"));
#if defined(TIME_REC)
	 if (_iter_moc == 0 && _iter_mg==0 && _iter_cg==0) {
		 TimeCGStamp2=TimeCountCGCMFD;
		 strcpy(TimeMetricsCG[TimeCGStamp2], "CG - old flux update");
		 TimeCountCGCMFD++;
	 }
	 cudaDeviceSynchronize();
	 _time_end = TIME_RECORD;
	 TimeRecCG[TimeCGStamp2] += _time_end-_time_start;
 #endif // DEFAULT_TIME

#if defined(TIME_REC)
    cudaDeviceSynchronize();
    _time_start = TIME_RECORD;
#endif // defined(TIME_REC)
	 if (_ProblemType == 0) {
#if defined(M_CMFDMatrixIncScatt)
		 GPUCMFD_EvaluateFissionSource_onFissQ<<<GPUGridDim_CGCMFDNode, GPUBlockDim>>>(GPU_CGCMFDTotalFissQ, \
			 GPU_CGCMFDTotalSource, GPU_CGCMFD_Chi, 1.0 / _k, _number, _numCMFDGroup);
		 CUDA_CheckError((char *)("GPU_MGCMFDSolver_Part2 - GPUCMFD_EvaluateFissionSource_onFissQ"));
#else
		 GPUCMFD_EvaluateTotalSource_onFissQ<<<GPUGridDim_CGCMFDNode, GPUBlockDim>>>(GPU_CGCMFDTotalFissQ, GPU_CGCMFDTotalSource, \
			GPU_CGCMFD_Chi, GPU_CGCMFDXs_s, GPU_CGCMFDScalarFlux, 1.0 / _k, _number, _numCMFDGroup, GPUCMFDNodeVol);
		 CUDA_CheckError((char *)("GPU_MGCMFDSolver_Part2 - GPUCMFD_EvaluateFissionSource_onFissQ"));
#endif //#if defined(M_CMFDMatrixIncScatt)
	 }
#if defined(TIME_REC)
	 if (_iter_moc == 0 && _iter_mg==0 && _iter_cg==0) {
		 TimeCGStamp3=TimeCountCGCMFD;
		 strcpy(TimeMetricsCG[TimeCGStamp3], "CG - SourEva-1");
		 TimeCountCGCMFD++;
	 }
	 cudaDeviceSynchronize();
	 _time_end = TIME_RECORD;
	 TimeRecCG[TimeCGStamp3] += _time_end-_time_start;
 #endif // DEFAULT_TIME

#if defined(MPI_USED)
#if defined(TIME_REC)
	cudaDeviceSynchronize();
	_time_start = TIME_RECORD;
#endif // defined(TIME_REC)
    if (_NumProcs>1) {
		cudaMemcpy((void*)_CGCMFDScalarFlux, (void*)GPU_CGCMFDScalarFlux, (size_t)(_number) * sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
		CUDA_CheckError((char *)("GPU_MGCMFDSolver_Part2 - cudaMemcpy - GPUCMFDScalarFlux"));
	}
#if defined(TIME_REC)
	 if (_iter_moc == 0 && _iter_mg==0 && _iter_cg==0) {
		 TimeCGStamp4=TimeCountCGCMFD;
		 strcpy(TimeMetricsCG[TimeCGStamp4], "CG - MPI Comm G->C");
		 TimeCountCGCMFD++;
	 }
	 cudaDeviceSynchronize();
	 _time_end = TIME_RECORD;
	 TimeRecCG[TimeCGStamp4] += _time_end-_time_start;
 #endif // DEFAULT_TIME
#endif // defined(MPI_USED)

// #if defined(DEFAULT_TIME)
// 	cudaDeviceSynchronize();
// #endif // TIME_REC
}

extern "C"
int GPU_CGCMFDSolver_Part3(M_INT32 _iter_moc, M_INT32 _iter_mg, M_INT32 _iter_cg, FP_PRECISION _k, M_INT32 _ProcID, M_INT32 _NumProcs, \
	M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCoarseCMFDGroup, FP_PRECISION *_CGCMFDFlux_MPIRecv, M_INT32 _CGCMFDData_MPIComm_Num,\
	M_INT32 _CGCMFD_LinearSolver, M_INT32 _MaxNumIter_CGLS, FP_PRECISION _ConvCritLinearSolver_CGCMFD, M_INT32 _CGCMFD_checkFluxNormInf, \
	M_INT32 _CGCMFD_checkFluxNorm2, M_INT32 _CGCMFD_checkFluxRMSE, FP_PRECISION *_CGCMFDLS_NormInf, FP_PRECISION *_CGCMFDLS_Norm2, \
	FP_PRECISION *_CGCMFDLS_RMSE, FP_PRECISION _SOR_Factor_CG, M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, \
	M_INT32 _CGCMFD_NumNoneZeroValue_Matrix, M_INT32 _GmresKrylovDim_CG, FP_PRECISION *_CG_ILUMatrix, M_INT32 *_CG_DiagPointer, \
	M_INT32 _GmresPrecond_CG, FP_PRECISION_TIME *_Time_CGCMFD_GMRES_Precond, FP_PRECISION_TIME *_Time_CGCMFD_GMRES_LeastSqr, \
	FP_PRECISION_TIME *_Time_CGCMFD_GMRES_MemCpy, FP_PRECISION_TIME *_Time_CGCMFD_GMRES_SpMV, FP_PRECISION_TIME *_Time_CGCMFD_GMRES_BLAS1,\
	float *_CGCMFD_NegFluxCount, M_INT32 *_CGCMFD_FluxFlag, FP_PRECISION_TIME *_TimeRec_CGCMFD_LinearSolver, FP_PRECISION *_CG_Residual)
{
	int _number, _numLSIter, _numCMFDGroup;
#if defined(TIME_REC)
	FP_PRECISION_TIME _time_start, _time_end;
	cudaDeviceSynchronize();
	_time_start = TIME_RECORD;
#endif // TIME_REC
	cublasHandle_t _cublasHandle;

#if defined(DEFAULT_TIME)
    FP_PRECISION_TIME _default_time_start, _default_time_end;
#endif // TIME_REC
	cublasCreate(&_cublasHandle);

	_numCMFDGroup = _NumCoarseCMFDGroup;

	_number = _MOC_NumPinsInSubDom * _numCMFDGroup;

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#if defined(MPI_USED)
    if (_NumProcs > 1) {
		cudaMemcpy((void*)GPUCGCMFDFlux_MPIRecv, (void*)_CGCMFDFlux_MPIRecv, (size_t)(_CGCMFDData_MPIComm_Num) * sizeof(FP_PRECISION), cudaMemcpyHostToDevice);
		CUDA_CheckError((char *)("GPU_CGCMFDSolver_Part3 - cudaMemcpy - GPUCGCMFDFlux_MPIRecv"));
#if defined(TIME_REC)
		cudaDeviceSynchronize();
		if (_iter_moc == 0 && _iter_mg==0 && _iter_cg==0) {
			TimeCGStamp6=TimeCountCGCMFD;
			strcpy(TimeMetricsCG[TimeCGStamp6], "CG - MPI Comm flux C->G");
			TimeCountCGCMFD++;
		}
		_time_end = TIME_RECORD;
		TimeRecCG[TimeCGStamp6] += _time_end-_time_start;
#endif // DEFAULT_TIME
#if defined(TIME_REC)
		cudaDeviceSynchronize();
		_time_start = TIME_RECORD;
#endif // defined(TIME_REC)
		GPUSetCMFDTotSour_wInBoundInfo<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(_ProcID, _NumProcs, _CGCMFDData_MPIComm_Num, \
			GPUCGCMFDData_MPIComm_Index, GPU_CGCMFDTotalSource, GPUCGCMFDMPI_InBoundSourCoeff, GPUCGCMFDFlux_MPIRecv);
#if defined(TIME_REC)
		if (_iter_moc == 0 && _iter_mg==0 && _iter_cg==0) {
			TimeCGStamp7=TimeCountCGCMFD;
			strcpy(TimeMetricsCG[TimeCGStamp7], "CG - SourEva-2");
			TimeCountCGCMFD++;
		}
		cudaDeviceSynchronize();
		_time_end = TIME_RECORD;
		TimeRecCG[TimeCGStamp7] += _time_end-_time_start;
#endif // DEFAULT_TIME
	}
#endif // defined(MPI_USED)

#if defined(DEFAULT_TIME)
	 cudaDeviceSynchronize();
	 _default_time_start = TIME_RECORD;
#endif // defined(DEFAULT_TIME)
#if defined(TIME_REC)
	cudaDeviceSynchronize();
	_time_start = TIME_RECORD;
#endif // defined(TIME_REC)
	 if (_CGCMFD_LinearSolver == 0) {
		 _numLSIter = GPULinearSolver_Jacobi(GPU_CGCMFDScalarFlux, GPU_CGCMFDScalarFlux_Old_LS, GPU_CGCMFDTotalSource, \
			 GPU_CGCMFDMatrixCSR, GPU_CGCMFDDiagPointer, GPU_CGCMFDMatrixCSR_RowStartInd, GPU_CGCMFDMatrixCSR_ColInd, \
			 _number, GPU_CGCMFDFluxRelErr, _MaxNumIter_CGLS, _ConvCritLinearSolver_CGCMFD, _cublasHandle, GPUGridDim_CGCMFDNode,\
			 _CGCMFD_checkFluxNormInf, _CGCMFD_checkFluxNorm2, _CGCMFD_checkFluxRMSE, _CGCMFDLS_NormInf, _CGCMFDLS_Norm2, \
	         _CGCMFDLS_RMSE);
	 }
	 else if (_CGCMFD_LinearSolver == 1) {
		 _numLSIter = GPULinearSolver_BSOR(_numCMFDGroup, GPU_CGCMFDScalarFlux, GPU_CGCMFDScalarFlux_Old_LS, GPU_CGCMFDTotalSource, \
			 GPU_CGCMFDMatrixCSR, GPU_CGCMFDDiagPointer, GPU_CGCMFDMatrixCSR_RowStartInd, GPU_CGCMFDMatrixCSR_ColInd, _number, \
			 GPU_CGCMFDFluxRelErr, _MaxNumIter_CGLS, _ConvCritLinearSolver_CGCMFD, _cublasHandle, _SOR_Factor_CG, GPU_CGCMFDNumRedThreads, \
			 GPU_CGCMFDNumBlackThreads, GPUGridDim_CGCMFDRed, GPUGridDim_CGCMFDBlack, GPUGridDim_CGCMFDNode, _MOCNumPinsInSubDom_X, \
			 _MOCNumPinsInSubDom_Y, GPUNumThreadBlocks, GPUNumThreadPerBlock, GPUNumRedLay_OddNx, GPUNumBlackLay_OddNx,\
			 _CGCMFD_checkFluxNormInf, _CGCMFD_checkFluxNorm2, _CGCMFD_checkFluxRMSE, _CGCMFDLS_NormInf, _CGCMFDLS_Norm2, _CGCMFDLS_RMSE);
	 }
	 else if (_CGCMFD_LinearSolver == 2) {
		_numLSIter = GPULinearSolver_GMRES(_number,_CGCMFD_NumNoneZeroValue_Matrix,_GmresKrylovDim_CG,_MaxNumIter_CGLS, \
			GPU_CGCMFDMatrixCSR, NULL,GPU_CGCMFDMatrixCSR_ILUFactor,_CG_ILUMatrix,GPU_CGCMFDMatrixCSR_ColInd,\
			Dev_CGCMFDMatrixCSR_ColInd, GPU_CGCMFDMatrixCSR_RowStartInd, Dev_CGCMFDMatrixCSR_RowStartInd, _CG_DiagPointer, \
			GPU_CGCMFDScalarFlux,GPU_CGCMFDTotalSource,_ConvCritLinearSolver_CGCMFD, _GmresPrecond_CG, _cublasHandle, \
			CG_cusparseHandle,GPU_CGGMRES_r0, _CG_Residual, GPU_CGGMRES_res, GPU_CGGMRES_beta, GPU_CGGMRES_u_base, \
			GPU_CGGMRES_g, GPU_CGGMRES_e, GPU_CGGMRES_Hm, GPU_CGGMRES_w, GPU_CGGMRES_y, GPU_CGGMRES_Wm, GPU_CGGmres_blocksPerGrid, \
			_Time_CGCMFD_GMRES_Precond, _Time_CGCMFD_GMRES_LeastSqr,_Time_CGCMFD_GMRES_MemCpy,_Time_CGCMFD_GMRES_SpMV, \
			_Time_CGCMFD_GMRES_BLAS1, _CGCMFDLS_RMSE, CG_trans_L, CG_trans_U, CG_descr_L, CG_descr_U, CG_info_L, CG_info_U, \
			CG_policy_L, CG_policy_U, CG_pBuffer);
		CUDA_CheckError((char *)("GPU_CGCMFDSolver_Part3 - GPULinearSolver_GMRES"));
	 }
	 else {}

	 GPU_SetNegCMFDFlux_CG(_number, GPUNumThreadBlocks, GPUNumThreadPerBlock, _CGCMFD_NegFluxCount, _CGCMFD_FluxFlag);
	 CUDA_CheckError((char *)("GPU_CGCMFDSolver_Part3 - GPU_SetNegCMFDFlux_CG"));
	 
	// if (_ProcID==0) {
	// 	GPUPrintf1DArray_FP(_MOC_NumPinsInSubDom * NumCMFDGroup, GPUCMFDScalarFlux, "GPUCMFDScalarFlux",LogFile);
	// }


	cublasDestroy(_cublasHandle);
	CUDA_CheckError((char *)("GPU_CGCMFDSolver_Part3 - cublasDestroy"));
	
#if defined(TIME_REC)
	if (_iter_moc == 0 && _iter_mg==0 && _iter_cg==0) {
		TimeCGStamp8=TimeCountCGCMFD;
		strcpy(TimeMetricsCG[TimeCGStamp8], "CG - LS solver");
		TimeCountCGCMFD++;
	}
	cudaDeviceSynchronize();
	_time_end = TIME_RECORD;
	TimeRecCG[TimeCGStamp8] += _time_end-_time_start;
#endif // DEFAULT_TIME

#if defined(DEFAULT_TIME)
	 cudaDeviceSynchronize();
	 _default_time_end = TIME_RECORD;
	 _TimeRec_CGCMFD_LinearSolver[0] += _default_time_end - _default_time_start;
#endif // defined(DEFAULT_TIME)
#if defined(DEFAULT_TIME)
	cudaDeviceSynchronize();
#endif // TIME_REC

    return (_numLSIter);
}


/**
* @ Function: GPUCMFD_ParaUpdate()
*   introduction: update the M_MG_CMFD iteration
* @ Parameter List:
*   Return:
* @ Created     : 2019/08/23, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void GPUCMFD_ParaUpdate()
{
//	CMFDKeff_Old = CMFDKeff;
//
//#ifdef CUDAErrorCheck
//	CUDAStatus = cudaGetLastError();
//	if (CUDAStatus != cudaSuccess) {
//		printf("\n [GPU-ERROR]  Lunch Failed: GPUCMFD_ParaUpdate with Info: %s\n", cudaGetErrorString(CUDAStatus));
//		exit(69);
//	}
//#endif // CUDAErrorCheck
}


/**
* @ Function: GPUUpdateFlux_CMFDtoMOC()
*   introduction: update the MOC scalar flux and angular flux by M_MG_CMFD solution
* @ Parameter List:
*   Return:
* @ Created     : 2019/08/23, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void GPUUpdateFlux_CMFDtoMOC(int _MOC_NumPinsInSubDom ,int _NumCMFDGroup, \
	FP_PRECISION_FLUX *_CMFDScalarFLux, FP_PRECISION_FLUX *_MOCtoCMFD_HetScalFlux)
{
	int _dim;
#if defined(DEFAULT_TIME)
	cudaDeviceSynchronize();
#endif // TIME_REC

	_dim = (size_t)(_MOC_NumPinsInSubDom * _NumCMFDGroup) * sizeof(FP_PRECISION_FLUX);
	cudaMemcpy((void*)_CMFDScalarFLux, (void*)GPUCMFDScalarFlux, _dim, cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)_MOCtoCMFD_HetScalFlux, (void*)GPUCMFDHetScalFlux, _dim, cudaMemcpyDeviceToHost);
	CUDA_CheckError((char *)("GPUUpdateFlux_CMFDtoMOC-MemCpy"));

	// GPUUpdateScalFlux_CMFDtoMOC<<<GPUGridDim_CMFDNode, GPUBlockDim>>>(GPUCMFDScalarFlux, \
	// 	GPUCMFDHetScalFlux, GPUScalarFluxTemp, GPU_MGCMFDGroupBound, GPUCellRTMType_SubDom, GPUNumFSRsInRTM, \
	// 	GPUSumNumFSRsBasedGlobCellID, _MGCMFD_damp);



#if defined(DEFAULT_TIME)
	cudaDeviceSynchronize();
#endif // TIME_REC
}
#endif // M_MG_CMFD



void PrintFPArrayOnDevice(int _dim, FP_PRECISION *_devArray, char *_parameter)
{
	int i;
	FP_PRECISION *_host=NULL;
	if ((_host=(FP_PRECISION *)calloc(_dim,sizeof(FP_PRECISION))) == NULL) {
		printf("\n _dim=%d in PrintFPArrayOnDevice, with _parameter = %s",_dim,_parameter);
		fflush(stdout);
		exit(90);
	}

	cudaMemcpy((void*)_host, (void*)_devArray,(size_t)(_dim)*sizeof(FP_PRECISION), cudaMemcpyDeviceToHost);
	printf("\n");
    for(i=0;i<_dim;i++) {
        printf("\n %s[%d]=%.6e",_parameter,i,_host[i]);
    }
    free(_host);
    _host=NULL;
}

/**
* @ Function: GPUGetNetCurr_Axial()
*   introduction: accumulate the axial net current for CMFD coarse mesh
* @ Parameter List: _dir = 1.0 (z pos) or -1.0 (z neg)
*   Return: void
* @ Created     : 2020/02/03, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void GPUGetNetCurr_Axial_1(int _lay, int _currLay, FP_PRECISION _dir, \
	int _polarBoundLowLim, int _polarBoundUpLim, FP_PRECISION_FLUX *_surfAngFlux, \
	int _polOffset)
{
#if defined(M_3DMOCex) && defined(M_MG_CMFD)
	GPUGetNetCurr_Axial<<<GPUNumThreadBlocks, GPUNumThreadPerBlock>>>(_lay,_currLay,_dir,\
		GPUSumNumFSRsBasedGlobCellID, GPUNumFSRsInCell,_polarBoundLowLim,_polarBoundUpLim,\
		_surfAngFlux, GPUMOCNetCurr_Z,_polOffset,GPUFSRArea_SubDom);
#endif // defined(M_3DMOCex) && defined(M_MG_CMFD)

#ifdef CUDAErrorCheck
	CUDAStatus = cudaGetLastError();
	if (CUDAStatus != cudaSuccess) {
		printf("\n [GPU-ERROR]  Lunch Failed: GPUGetNetCurr_Axial_1 with Info: %s\n", cudaGetErrorString(CUDAStatus));
		exit(69);
	}
#endif // CUDAErrorCheck
}



/**
* @ Function: GPUCompareTwo1DArray_FP()
*   introduction: print the 1D array on GPU with FP_PRECISION, compare two arraies
* @ Parameter List:
*   Return: void
* @ Created     : 2020/02/10, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
extern "C"
extern "C"
__global__ void GPUSetXSt_wScale(FP_PRECISION *_GPUMacroXS_t_wScale, FP_PRECISION *_GPUMacroXS_t,\
    FP_PRECISION_GEO *_GPUWidthLayersInCore, int _TotalNumFSRs_perLayer)
{
    int _tid = threadIdx.x + blockIdx.x * blockDim.x, _pol, _lay;
	FP_PRECISION _scale, _xs;
    /* Iterate over all FSRs and all groups */
    while (_tid < (*GPUTotNumFSRs)*(*GPUNumMOCGroups)) {
		_xs=_GPUMacroXS_t[_tid];
		_lay = _tid/(*GPUNumMOCGroups)/_TotalNumFSRs_perLayer;
		for (_pol = 0; _pol < (*GPUNumPolarAng); _pol++) {
			_scale = 2.0 * GPUCosThetas[_pol] / _GPUWidthLayersInCore[_lay];
			_GPUMacroXS_t_wScale[_tid*(*GPUNumPolarAng)+_pol]=_xs+_scale;
		}
        /* Increment the thread id */
        _tid += blockDim.x * gridDim.x;
    }
}

void cudaDeviceSynchronize_cpu()
{
#if defined(CUDA_USED)
	cudaDeviceSynchronize();
#endif // defined(CUDA_USED)
}

void GetConstantMemUsage_GPU(int _procID, int _numAzimAng, int _numPolAng, FILE *_LogFile)
{
#if defined(CUDA_USED)
	int _totalByte=0;

	if (MAX_AZIM_ANGLES_GPU<_numAzimAng) {
		if (_procID==0) {
			printf("\n [GPU-ERROR]  _numAzimAng = %d > MAX_AZIM_ANGLES_GPU = %d, please reset!", _numAzimAng, MAX_AZIM_ANGLES_GPU);
			fprintf(_LogFile, "\n [GPU-ERROR]  _numAzimAng = %d > MAX_AZIM_ANGLES_GPU = %d, please reset!", _numAzimAng, MAX_AZIM_ANGLES_GPU);
		}
		exit(90);
	}

	if (MAX_POLAR_ANGLES_GPU<_numPolAng) {
		if (_procID==0) {
			printf("\n [GPU-ERROR]  _numPolAng = %d > MAX_POLAR_ANGLES_GPU = %d, please reset!", _numPolAng, MAX_POLAR_ANGLES_GPU);
			fprintf(_LogFile, "\n [GPU-ERROR]  _numPolAng = %d > MAX_POLAR_ANGLES_GPU = %d, please reset!", _numPolAng, MAX_POLAR_ANGLES_GPU);
		}
		exit(90);
	}

	_totalByte+=1*sizeof(int); // int GPUNumAzimAng[1]
	_totalByte+=1*sizeof(int); // int GPUNumPolarAng[1]
	_totalByte+=1*sizeof(int); // int GPUNumMOCGroups[1]
	_totalByte+=1*sizeof(int); // int GPUNumLTs_Total[1]
	_totalByte+=1*sizeof(int); // int GPUTotNumTracksInOneRTM_2PI[1]
	_totalByte+=1*sizeof(int); // int GPUPolarTimesGroups[1]
	_totalByte+=MAX_POLAR_ANGLES_GPU*sizeof(FP_PRECISION); // FP_PRECISION GPUSinThetasTimes2[MAX_POLAR_ANGLES_GPU]
	_totalByte+=MAX_AZIM_ANGLES_GPU*sizeof(FP_PRECISION); // FP_PRECISION GPUAzimWeights[MAX_AZIM_ANGLES_GPU];
	_totalByte+=MAX_POLAR_ANGLES_GPU*sizeof(FP_PRECISION); // GPUPolarWeights[MAX_POLAR_ANGLES_GPU];
	_totalByte+=MAX_POLAR_ANGLES_GPU*MAX_AZIM_ANGLES_GPU*sizeof(FP_PRECISION); // FP_PRECISION GPUTotalWeights[MAX_POLAR_ANGLES_GPU*MAX_AZIM_ANGLES_GPU];
	_totalByte+=12*sizeof(int); 
	// int GPUMOCNumLayersZ[1],  GPUMOCNumPinsInSubDom_X[1], GPUMOCNumPinsInSubDom_Y[1];
    // int GPUNumSurfLayer_X[1],  GPUNumSurfLayer_Y[1], GPUNumSurfLayer_Z[1];
	// int GPUNorthBC[1],GPUSouthBC[1],GPUEastBC[1],GPUWestBC[1],GPUBottomBC[1],GPUTopBC[1];
	_totalByte+=MAX_POLAR_ANGLES_GPU*sizeof(FP_PRECISION); // FP_PRECISION GPUSinThetaForExp[MAX_POLAR_ANGLES_GPU];
	_totalByte+=MAX_POLAR_ANGLES_GPU*sizeof(FP_PRECISION); // FP_PRECISION GPUSinThetas[MAX_POLAR_ANGLES_GPU];
	_totalByte+=MAX_POLAR_ANGLES_GPU*sizeof(FP_PRECISION); // FP_PRECISION GPUCosThetas[MAX_POLAR_ANGLES_GPU];
	_totalByte+=MAX_POLAR_ANGLES_GPU*sizeof(FP_PRECISION); // FP_PRECISION GPUCosThetas[GPUOneOverSinThetas];
	_totalByte+=MAX_POLAR_ANGLES_GPU*sizeof(FP_PRECISION_GEO); // FP_PRECISION GPUEffRaySpacing[GPUOneOverSinThetas];
	_totalByte+=1*sizeof(int); // int GPUNumPinsInSubDomPerLayer[1]

	_totalByte+=1*sizeof(int); // int GPUTotNumFSRs[1]; /** The number of FSRs */
	_totalByte+=1*sizeof(int); //  int GPUNumLTs_HS[1];  // number of tracks in [0,PI]
	_totalByte+=1*sizeof(int); //  int GPUTotNumTracksInOneRTM_HS[1];
	//_totalByte+=1*sizeof(FP_PRECISION); //  FP_PRECISION GPUGlobFissRate[1];
	_totalByte+=1*sizeof(int); //  int GPUTotNumPins[1]; /** The number of pins */
	_totalByte+=1*sizeof(int); //  int GPUTotalNumFSRsInRTM[1];

	if (_procID==0) {
		printf("\n [GPU-INFO ]  Constant memory usage on GPU: %d bytes of %llu bytes", _totalByte, (unsigned long long)DeviceProp.totalConstMem);
		fprintf(_LogFile, "\n [GPU-INFO ]  Constant memory usage on GPU: %d bytes of %llu bytes", _totalByte, (unsigned long long)DeviceProp.totalConstMem);
	}
	if (_totalByte > DeviceProp.totalConstMem) {
		if (_procID==0) {
			printf("\n [GPU-ERROR]  Constant memory usage on GPU is out of support, please reset!");
			fprintf(_LogFile, "\n [GPU-ERROR]  Constant memory usage on GPU is out of support, please reset!");
		}
		exit(90);
	}
#endif // defined(CUDA_USED)
}

extern "C"
void CopyInpInfo_CPU2GPU(M_INT32 _GPUBlockDimInp, M_INT32 _GPUNumStreamsInp)
{
	//cudaMemcpy((void*)&GPUBlockDim, (void*)&_GPUBlockDimInp, sizeof(M_INT32), cudaMemcpyHostToDevice);
	//CUDA_CheckError((char *)("CopyInpInfo_CPU2GPU-cudaMemcpy-GPUBlockDim"));
	//cudaMemcpy((void*)&GPUNumStreams, (void*)&_GPUNumStreamsInp, sizeof(M_INT32), cudaMemcpyHostToDevice);
	//CUDA_CheckError((char *)("CopyInpInfo_CPU2GPU-cudaMemcpy-GPUNumStreams"));
	GPUBlockDim = _GPUBlockDimInp;
	GPUNumStreams = _GPUNumStreamsInp;
}


extern "C"
void GPUCopyTrackInfoHostToDevice_trackFile(int _procID, \
                                            M_INT32 _NumGroups,\
                                            M_INT32 _TotalNumFSRs_subDom,\
                                            M_INT32 _TotalNumFSRs_perLayer, \
	                                        M_INT32 _MOC_NumPinsInSubDom,\
	                                        M_INT32 _NumPinsInSubDomPerLayer,\
                                            M_INT32 _TotalNumFSRsInRTM,\
                                            M_INT32 _NumPolarAng, \
                                            M_INT32 _NumLTs_HS,\
                                            M_INT32 _NumLTs_2PI,\
                                            M_INT32 *_LTsIndex_AzimAngID_HS,\
                                            M_INT32 *_LTsIndex_LTID_HS, \
                                            M_INT32 *_LTsIndex_AzimAngID_2PI,\
                                            M_INT32 *_LTsIndex_LTID_2PI,\
                                            struct SUBDOM_TRACK_IN_LONGTRACK *_TracksInfoInLTs_2PI, \
                                            M_INT32 *_TrackIndexInLTs_2PI,\
                                            M_INT32 _TotNumTracksInOneRTM_2PI,\
                                            M_INT32 *_IndexTrackInRTM_SumAzim_2PI, \
                                            M_INT32 _TotalNumTracksInRTM_Total,\
                                            M_INT32 *_NumSegsInTrack_2PI,\
                                            M_INT32 _TotNumSegInRTM_Total, \
	                                        /*struct SUBDOM_SEG_IN_TRACK_SINGLE *_SegInfoInTrack_SINGLE_2PI,*/
                                            M_INT32 *_SumNumSegsInRTMTrack_2PI, \
	                                        FP_PRECISION *_SegInfoInTrack_2PI_length,\
	                                        /*M_INT32 *_SegInfoInTrack_2PI_RayWisedFSR,*/
                                            M_INT32 *_SegInfoInTrack_2PI_AntiClockFSR, \
	                                        /*M_INT32 *_FsrIdMap_RayWisedToAntiClock,*/
                                            struct SUBDOM_TRACK_IN_LONGTRACK *_TracksInfoInLTs_HS,\
                                            M_INT32 _NumTracks_HS, \
	                                        struct SUBDOM_LONGTRACK_INFO *_LongTracksInfo_Total,\
	                                        /*M_INT32 *_TrackIndexInLTs_HS, M_INT32 _TotalNumTracksInRTM_HS, \
	                                        M_INT32 *_NumSegsInTrack_HS, M_INT32 _TotNumTracksInOneRTM_HS,*/
	                                        M_INT32 _NumAzimAng,\
	                                        /*M_INT32 *_IndexTrackInRTM_SumAzim_HS, \
	                                        M_INT32 _TotNumSegInRTM_HS,\
	                                        struct SUBDOM_SEG_IN_TRACK_SINGLE *_SegInfoInTrack_SINGLE_HS,*/
	                                        M_INT32 _MOCNumLayersZ, \
	                                        M_INT32 *_SumNumFSRsBasedGlobCellID,\
	                                        M_INT32 _NumRTMGeoType,\
	                                        M_INT32 *_SumNumFSRsInRTM,\
	                                        M_INT32 *_CellRTMType_SubDom, \
	                                        /*M_INT32 *_SumNumSegsInRTMTrack_HS,*/
	                                        M_INT32 *_NumLTsSum, M_INT32 *_NumLongTracks, \
	                                        /*FP_PRECISION_GEO *_OneOverFSRArea_SubDom,*/
	                                        FP_PRECISION_GEO *_FSRVol_SubDom,\
	                                        M_INT32 _NorthBC,\
	                                        M_INT32 _SouthBC,\
	                                        M_INT32 _EastBC, \
	                                        M_INT32 _WestBC, \
	                                        M_INT32 _BottomBC,\
	                                        M_INT32 _TopBC,\
	                                        M_INT32 *_NumFSRsInCell, \
	                                        FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	                                        FP_PRECISION_GEO *_EffRaySpacing,\
	                                        M_INT32 *_NumFSRs_Layer, M_INT32 *_SumNumFSRs_Layer,/**/
	                                        FP_PRECISION_GEO *_WidthLayersInCore, \
	                                        M_INT32 _MOCNumPinsInSubDom_X,\
	                                        M_INT32 _MOCNumPinsInSubDom_Y,\
	                                        /*FP_PRECISION_GEO *_RayTracModDim,*/
	                                        M_INT32 _NumSurfaceX, M_INT32 _NumSurfaceY, \
	                                        M_INT32 *_NumTracksX, M_INT32 *_NumTracksY,\
	                                        FP_PRECISION_GEO *_RTMWidth_X, FP_PRECISION_GEO *_RTMWidth_Y,
	                                        M_INT32 *_NumFSRsInRTM, \
	                                        FP_PRECISION_GEO *_FSRsAreaInRTM_AntiClockFSR,\
	                                        FP_PRECISION_GEO *_FSRArea_SubDom,\
	                                        size_t *_GPUPreMemTrack, size_t *_GPUPreMemXSs, \
	                                        M_INT32 _NumTracks_Total, FILE *_LogFile)
{
    int _totNumLTs, _polarTimesGroup, _dimSurfX, _dimSurfY, _dimSurfZ;
    // int i;
    size_t _dim = 0;
    if (_procID == 0) {
        printf("\n [GPU-INFO ]  Initializing Track Information on GPU from TrackFile ...");
        fflush(stdout);
        fprintf(_LogFile, "\n [GPU-INFO ]  Initializing Track Information on GPU from TrackFile ...");
    }
    _totNumLTs = _NumLTs_HS * 2;

    cudaMemcpyToSymbol(GPUNumMOCGroups, (void*)&_NumGroups, sizeof(int), 0, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUNumMOCGroups"));

    cudaMemcpyToSymbol(GPUTotNumFSRs, (void*)&_TotalNumFSRs_subDom, sizeof(M_INT32), 0, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUTotNumFSRs"));

    cudaMemcpyToSymbol(GPUNumFSRsLayer, (void*)&_TotalNumFSRs_perLayer, sizeof(M_INT32), 0, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUNumFSRsLayer"));

    cudaMemcpyToSymbol(GPUTotNumPins, (void*)&_MOC_NumPinsInSubDom, sizeof(int), 0, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUTotNumPins"));

    cudaMemcpyToSymbol(GPUNumPinsInSubDomPerLayer, (void*)&_NumPinsInSubDomPerLayer, sizeof(int), 0, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUNumPinsInSubDomPerLayer"));

    cudaMemcpyToSymbol(GPUTotalNumFSRsInRTM, (void*)&_TotalNumFSRsInRTM, sizeof(int), 0, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUTotalNumFSRsInRTM"));
    //
    _polarTimesGroup = _NumPolarAng * _NumGroups;
    cudaMemcpyToSymbol(GPUPolarTimesGroups, (void*)&_polarTimesGroup, sizeof(int), 0, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUPolarTimesGroups"));
    /* Copy the number of long tracks in halfspace[0,PI] on the GPU */
    cudaMemcpyToSymbol(GPUNumLTs_HS, (void*)&_NumLTs_HS, sizeof(int), 0, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUNumLTs_HS"));
#ifdef GeoTwoPIRegion
    cudaMemcpyToSymbol(GPUNumLTs_Total, (void*)&_NumLTs_2PI, sizeof(int), 0, cudaMemcpyHostToDevice);
	//CUDA_CheckError((char *)("GPUNumLTs_Total"));
#endif // GeoTwoPIRegion

    _dim = (size_t)(_NumLTs_HS) * sizeof(int);
    cudaMalloc((void**)&GPULTsIndex_AzimAngID_HS, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPULTsIndex_AzimAngID_HS, (void*)_LTsIndex_AzimAngID_HS, _dim, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPULTsIndex_AzimAngID_HS"));

    _dim = (size_t)(_NumLTs_HS) * sizeof(int);
    cudaMalloc((void**)&GPULTsIndex_LTID_HS, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPULTsIndex_LTID_HS, (void*)_LTsIndex_LTID_HS, _dim, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPULTsIndex_LTID_HS"));
    //
#ifdef GeoTwoPIRegion
    _dim = (size_t)(_NumLTs_2PI) * sizeof(int);
	cudaMalloc((void**)&GPULTsIndex_AzimAngID_Total, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPULTsIndex_AzimAngID_Total, (void*)_LTsIndex_AzimAngID_2PI, _dim, cudaMemcpyHostToDevice);
	//CUDA_CheckError((char *)("GPULTsIndex_AzimAngID_Total"));

	_dim = (size_t)(_NumLTs_2PI) * sizeof(int);
	cudaMalloc((void**)&GPULTsIndex_LTID_Total, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPULTsIndex_LTID_Total, (void*)_LTsIndex_LTID_2PI, _dim, cudaMemcpyHostToDevice);
	//CUDA_CheckError((char *)("GPULTsIndex_LTID_Total"));

	_dim = (size_t)(_NumTracks_Total) * sizeof(struct SUBDOM_TRACK_IN_LONGTRACK);
	cudaMalloc((void**)&GPUTracksInfoInLTs_Total, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUTracksInfoInLTs_Total, (void*)_TracksInfoInLTs_2PI, _dim, cudaMemcpyHostToDevice);
	//CUDA_CheckError((char *)("GPUTracksInfoInLTs_Total"));

	_dim = (size_t)(_NumLTs_2PI + 1) * sizeof(int);
	cudaMalloc((void**)&GPUTrackIndexInLTs_Total, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUTrackIndexInLTs_Total, (void*)_TrackIndexInLTs_2PI, _dim, cudaMemcpyHostToDevice);
	//CUDA_CheckError((char *)("GPUTrackIndexInLTs_Total"));

	cudaMemcpyToSymbol(GPUTotNumTracksInOneRTM_2PI, (void*)&_TotNumTracksInOneRTM_2PI, sizeof(int), 0, cudaMemcpyHostToDevice);
	//CUDA_CheckError((char *)("GPUTotNumTracksInOneRTM_2PI"));

	_dim = (size_t)(_NumAzimAng) * sizeof(int);
	cudaMalloc((void**)&GPUIndexTrackInRTM_SumAzim_Total, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUIndexTrackInRTM_SumAzim_Total, (void*)_IndexTrackInRTM_SumAzim_2PI, _dim, cudaMemcpyHostToDevice);
	//CUDA_CheckError((char *)("GPUIndexTrackInRTM_SumAzim_Total"));

	_dim = (size_t)(_TotalNumTracksInRTM_Total) * sizeof(int);
	cudaMalloc((void**)&GPUNumSegsInTrack_Total, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUNumSegsInTrack_Total, (void*)_NumSegsInTrack_2PI, _dim, cudaMemcpyHostToDevice);
	//CUDA_CheckError((char *)("GPUNumSegsInTrack_Total"));

	//_dim = (size_t)(_TotNumSegInRTM_Total) * sizeof(struct SUBDOM_SEG_IN_TRACK_SINGLE);
	//cudaMalloc((void**)&GPUSegInfoInTrack_Total, _dim);
	//_GPUPreMemTrack[0] += _dim;
	//cudaMemcpy((void*)GPUSegInfoInTrack_Total, (void*)_SegInfoInTrack_SINGLE_2PI, _dim, cudaMemcpyHostToDevice);

	_dim = (size_t)(_TotalNumTracksInRTM_Total) * sizeof(int);
	cudaMalloc((void**)&GPUSumNumSegsInRTMTrack_2PI, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUSumNumSegsInRTMTrack_2PI, (void*)_SumNumSegsInRTMTrack_2PI, _dim, cudaMemcpyHostToDevice);
	//CUDA_CheckError((char *)("GPUSumNumSegsInRTMTrack_2PI"));
	/*modified 2019-06-15 by peitao song*/
	_dim = (size_t)(_TotNumSegInRTM_Total) * sizeof(FP_PRECISION);
	cudaMalloc((void**)&GPUSegInfoInTrack_2PI_length, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUSegInfoInTrack_2PI_length, (void*)_SegInfoInTrack_2PI_length, _dim, cudaMemcpyHostToDevice);
	//CUDA_CheckError((char *)("GPUSegInfoInTrack_2PI_length"));

	//_dim = (size_t)(_TotNumSegInRTM_Total) * sizeof(int);
	//cudaMalloc((void**)&GPUSegInfoInTrack_2PI_RayWisedFSR, _dim);
	//_GPUPreMemTrack[0] += _dim;
	//cudaMemcpy((void*)GPUSegInfoInTrack_2PI_RayWisedFSR, (void*)_SegInfoInTrack_2PI_RayWisedFSR, _dim, cudaMemcpyHostToDevice);

	_dim = (size_t)(_TotNumSegInRTM_Total) * sizeof(int);
	cudaMalloc((void**)&GPUSegInfoInTrack_2PI_AntiClockFSR, _dim);
	_GPUPreMemTrack[0] += _dim;
	cudaMemcpy((void*)GPUSegInfoInTrack_2PI_AntiClockFSR, (void*)_SegInfoInTrack_2PI_AntiClockFSR, _dim, cudaMemcpyHostToDevice);
	//CUDA_CheckError((char *)("GPUSegInfoInTrack_2PI_AntiClockFSR"));

	//_dim = (size_t)(_TotalNumFSRsInRTM * _NumAzimAng) * sizeof(int);
	//cudaMalloc((void**)&GPUFsrIdMap_RayWisedToAntiClock, _dim);
	//_GPUPreMemTrack[0] += _dim;
	//cudaMemcpy((void*)GPUFsrIdMap_RayWisedToAntiClock, (void*)_FsrIdMap_RayWisedToAntiClock, _dim, cudaMemcpyHostToDevice);
	/*modified 2019-06-15 by peitao song - end*/
	//CUDA_CheckError((char *)("GPUFsrIdMap_RayWisedToAntiClock"));

#endif // GeoTwoPIRegion

    // copy tracks info in LTs from host to device
    _dim = (size_t)(_NumTracks_HS) * sizeof(struct SUBDOM_TRACK_IN_LONGTRACK);
    cudaMalloc((void**)&GPUTracksInfoInLTs_HS, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUTracksInfoInLTs_HS, (void*)_TracksInfoInLTs_HS, _dim, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUTracksInfoInLTs_HS"));

    /* Allocate memory on device for GPULongTracksInfo_Total data arrays */
    _dim = (size_t)(_totNumLTs) * sizeof(struct SUBDOM_LONGTRACK_INFO);
    cudaMalloc((void**)&GPULongTracksInfo_Total, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPULongTracksInfo_Total, (void*)_LongTracksInfo_Total, _dim, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPULongTracksInfo_Total"));
    //
    //_dim = (size_t)((_NumLTs_HS + 1)) * sizeof(int);
    //cudaMalloc((void**)&GPUTrackIndexInLTs_HS, _dim);
    //_GPUPreMemTrack[0] += _dim;
    //cudaMemcpy((void*)GPUTrackIndexInLTs_HS, (void*)_TrackIndexInLTs_HS, _dim, cudaMemcpyHostToDevice);

    // cudaMemcpyToSymbol(&GPUTotalNumTracksInRTM, (void*)TotalNumTracksInRTM_HS, sizeof(int), 0, cudaMemcpyHostToDevice);
    //_dim = (size_t)(_TotalNumTracksInRTM_HS) * sizeof(int);
    //cudaMalloc((void**)&GPUNumSegsInTrack_HS, _dim);
    //_GPUPreMemTrack[0] += _dim;
    //cudaMemcpy((void*)GPUNumSegsInTrack_HS, (void*)_NumSegsInTrack_HS, _dim, cudaMemcpyHostToDevice);
    //
    //cudaMemcpyToSymbol(GPUTotNumTracksInOneRTM_HS, (void*)&_TotNumTracksInOneRTM_HS, sizeof(int), 0, cudaMemcpyHostToDevice);

    //_dim = (size_t)(_NumAzimAng / 2) * sizeof(int);
    //cudaMalloc((void**)&GPUIndexTrackInRTM_SumAzim_HS, _dim);
    //_GPUPreMemTrack[0] += _dim;
    //cudaMemcpy((void*)GPUIndexTrackInRTM_SumAzim_HS, (void*)_IndexTrackInRTM_SumAzim_HS, _dim, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUIndexTrackInRTM_SumAzim_HS"));
    //
    //_dim = (size_t)(_TotNumSegInRTM_HS) * sizeof(struct SUBDOM_SEG_IN_TRACK_SINGLE);
    //cudaMalloc((void**)&GPUSegInfoInTrack_HS, _dim);
    //_GPUPreMemTrack[0] += _dim;
    //cudaMemcpy((void*)GPUSegInfoInTrack_HS, (void*)_SegInfoInTrack_SINGLE_HS, _dim, cudaMemcpyHostToDevice);
    //
    _dim = (size_t)(_MOCNumLayersZ*_NumPinsInSubDomPerLayer) * sizeof(int);
    cudaMalloc((void**)&GPUSumNumFSRsBasedGlobCellID, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUSumNumFSRsBasedGlobCellID, (void*)_SumNumFSRsBasedGlobCellID, _dim, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUSumNumFSRsBasedGlobCellID"));
    //
    _dim = (size_t)(_NumRTMGeoType) * sizeof(int);
    cudaMalloc((void**)&GPUSumNumFSRsInRTM, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUSumNumFSRsInRTM, (void*)_SumNumFSRsInRTM, _dim, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUSumNumFSRsInRTM"));
    //
    _dim = (size_t)(_MOCNumLayersZ*_NumPinsInSubDomPerLayer) * sizeof(int);
    cudaMalloc((void**)&GPUCellRTMType_SubDom, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUCellRTMType_SubDom, (void*)_CellRTMType_SubDom, _dim, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUCellRTMType_SubDom"));
    //
    //_dim = (size_t)(_TotalNumTracksInRTM_HS) * sizeof(int);
    //cudaMalloc((void**)&GPUSumNumSegsInRTMTrack_HS, _dim);
    //_GPUPreMemTrack[0] += _dim;
    //cudaMemcpy((void*)GPUSumNumSegsInRTMTrack_HS, (void*)_SumNumSegsInRTMTrack_HS, _dim, cudaMemcpyHostToDevice);
    //
    _dim = (size_t)(_NumAzimAng) * sizeof(int);
    cudaMalloc((void**)&GPUNumLTsSum, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUNumLTsSum, (void*)_NumLTsSum, _dim, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUNumLTsSum"));
    //
    _dim = (size_t)(_NumAzimAng) * sizeof(int);
    cudaMalloc((void**)&GPUNumLongTracks, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUNumLongTracks, (void*)_NumLongTracks, _dim, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUNumLongTracks"));
    //
    //_dim = (size_t)(_TotalNumFSRs_subDom) * sizeof(FP_PRECISION);
    //cudaMalloc((void**)&GPUOneOverFSRsAreaInGlob, _dim);
    //_GPUPreMemTrack[0] += _dim;
    //cudaMemcpy((void*)GPUOneOverFSRsAreaInGlob, (void*)_OneOverFSRArea_SubDom, _dim, cudaMemcpyHostToDevice);

    _dim = (size_t)(_TotalNumFSRs_subDom) * sizeof(FP_PRECISION_GEO);
    cudaMalloc((void**)&GPUFSRVol_SubDom, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUFSRVol_SubDom, (void*)_FSRVol_SubDom, _dim, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUFSRVol_SubDom"));

    //CUDA_CheckError((char *)("GPUOneOverFSRsAreaInGlob"));
    //
    cudaMemcpyToSymbol(GPUNorthBC, (void*)&_NorthBC, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(GPUSouthBC, (void*)&_SouthBC, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(GPUEastBC, (void*)&_EastBC, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(GPUWestBC, (void*)&_WestBC, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(GPUBottomBC, (void*)&_BottomBC, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(GPUTopBC, (void*)&_TopBC, sizeof(int), 0, cudaMemcpyHostToDevice);

    _dim = (size_t)(_MOC_NumPinsInSubDom) * sizeof(int);
    cudaMalloc((void**)&GPUNumFSRsInCell, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUNumFSRsInCell, (void*)_NumFSRsInCell, _dim, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUNumFSRsInCell"));

    _dim = (size_t)(_TotalNumFSRsInRTM) * sizeof(FP_PRECISION_GEO);
    cudaMalloc((void**)&GPUOneOverFSRsAreaInRTM_AntiClockFSR, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUOneOverFSRsAreaInRTM_AntiClockFSR, (void*)_OneOverFSRsAreaInRTM_AntiClockFSR, _dim, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUOneOverFSRsAreaInRTM_AntiClockFSR"));
    //
    cudaMemcpyToSymbol(GPUEffRaySpacing, (void*)_EffRaySpacing, (size_t)(_NumAzimAng) * sizeof(FP_PRECISION_GEO), 0, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPUEffRaySpacing"));

    _dim = (size_t)(_MOCNumLayersZ) * sizeof(int);
    cudaMalloc((void**)&GPUNumFSRs_Layer, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUNumFSRs_Layer, (void*)_NumFSRs_Layer, _dim, cudaMemcpyHostToDevice);
    CUDA_CheckError((char *)("GPUNumFSRs_Layer"));

    _dim = (size_t)(_MOCNumLayersZ) * sizeof(int);
    cudaMalloc((void**)&GPUSumNumFSRs_Layer, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUSumNumFSRs_Layer, (void*)_SumNumFSRs_Layer, _dim, cudaMemcpyHostToDevice);
    CUDA_CheckError((char *)("GPUSumNumFSRs_Layer"));

    _dim = (size_t)(_MOCNumLayersZ) * sizeof(FP_PRECISION_GEO);
    cudaMalloc((void**)&GPUWidthLayersInCore, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUWidthLayersInCore, (void*)_WidthLayersInCore, _dim, cudaMemcpyHostToDevice);
    CUDA_CheckError((char *)("GPUWidthLayersInCore"));

    /* copy geometry related info */
    cudaMemcpyToSymbol(GPUMOCNumLayersZ, (void*)&_MOCNumLayersZ, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(GPUMOCNumPinsInSubDom_X, (void*)&_MOCNumPinsInSubDom_X, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(GPUMOCNumPinsInSubDom_Y, (void*)&_MOCNumPinsInSubDom_Y, sizeof(int), 0, cudaMemcpyHostToDevice);

    _dimSurfX = (_MOCNumPinsInSubDom_X + 1) * _MOCNumPinsInSubDom_Y;
    _dimSurfY = _MOCNumPinsInSubDom_X * (_MOCNumPinsInSubDom_Y + 1);
    _dimSurfZ = _MOCNumPinsInSubDom_X * _MOCNumPinsInSubDom_Y;
    cudaMemcpyToSymbol(GPUNumSurfLayer_X, (void*)&_dimSurfX, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(GPUNumSurfLayer_Y, (void*)&_dimSurfY, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(GPUNumSurfLayer_Z, (void*)&_dimSurfZ, sizeof(int), 0, cudaMemcpyHostToDevice);

    //cudaMemcpyToSymbol(GPURayTracModDim, (void*)_RayTracModDim, (size_t)(3) * sizeof(FP_PRECISION_GEO), 0, cudaMemcpyHostToDevice);
    //CUDA_CheckError((char *)("GPURayTranModDim"));
    cudaMemcpyToSymbol(GPUNumSurfaceX, (void*)&_NumSurfaceX, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(GPUNumSurfaceY, (void*)&_NumSurfaceY, sizeof(int), 0, cudaMemcpyHostToDevice);
    CUDA_CheckError((char *)("GPUNumSurfaceY"));

    _dim = (size_t)(_NumAzimAng) * sizeof(int);
    cudaMalloc((void**)&GPUNumTracksX, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUNumTracksX, (void*)_NumTracksX, _dim, cudaMemcpyHostToDevice);
    CUDA_CheckError((char *)("GPUNumTracksX"));

    _dim = (size_t)(_NumAzimAng) * sizeof(int);
    cudaMalloc((void**)&GPUNumTracksY, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUNumTracksY, (void*)_NumTracksY, _dim, cudaMemcpyHostToDevice);
    CUDA_CheckError((char *)("GPUNumTracksY"));

    _dim = (size_t)(_NumRTMGeoType) * sizeof(FP_PRECISION_GEO);
    cudaMalloc((void**)&GPURTMWidth_X, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPURTMWidth_X, (void*)_RTMWidth_X, _dim, cudaMemcpyHostToDevice);
    CUDA_CheckError((char *)("GPURTMWidth_X"));

    _dim = (size_t)(_NumRTMGeoType) * sizeof(FP_PRECISION_GEO);
    cudaMalloc((void**)&GPURTMWidth_Y, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPURTMWidth_Y, (void*)_RTMWidth_Y, _dim, cudaMemcpyHostToDevice);
    CUDA_CheckError((char *)("GPURTMWidth_Y"));

    _dim = (size_t)(_NumRTMGeoType) * sizeof(int);
    //for (i = 0; i < _NumRTMGeoType; i++)
    //	printf("\n_NumFSRsInRTM[%d]=%d", i, _NumFSRsInRTM[i]);
    cudaMalloc((void**)&GPUNumFSRsInRTM, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUNumFSRsInRTM, (void*)_NumFSRsInRTM, _dim, cudaMemcpyHostToDevice);
    //GPUPrintf1DArray_I32(_NumRTMGeoType, GPUNumFSRsInRTM, "GPUNumFSRsInRTM", NULL);
    CUDA_CheckError((char *)("GPUNumFSRsInRTM"));

    _dim = (size_t)(_TotalNumFSRsInRTM) * sizeof(FP_PRECISION_GEO);
    cudaMalloc((void**)&GPUFSRsAreaInRTM_AntiClockFSR, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUFSRsAreaInRTM_AntiClockFSR, (void*)_FSRsAreaInRTM_AntiClockFSR, _dim, cudaMemcpyHostToDevice);
    CUDA_CheckError((char *)("GPUFSRsAreaInRTM_AntiClockFSR"));

    _dim = (size_t)(_TotalNumFSRs_subDom) * sizeof(FP_PRECISION_GEO);
    cudaMalloc((void**)&GPUFSRArea_SubDom, _dim);
    _GPUPreMemTrack[0] += _dim;
    cudaMemcpy((void*)GPUFSRArea_SubDom, (void*)_FSRArea_SubDom, _dim, cudaMemcpyHostToDevice);
    CUDA_CheckError((char *)("GPUFSRArea_SubDom"));

    CUDA_CheckError((char *)("GPUCopyTrackInfoHostToDevice"));
}


__global__ void react_rate_calc( FP_PRECISION_FLUX *_GPUScalarFlux,
                                  FP_PRECISION_GEO *_FSRVolumes,
                                  FP_PRECISION *_GPU1DXS,
                                  FP_DOUBLE *_Rate) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t fsr = tid / (*GPUNumMOCGroups);
    size_t g = tid % (*GPUNumMOCGroups);

    if (tid < (*GPUNumMOCGroups) * (*GPUTotNumFSRs)) {
        FP_PRECISION_GEO fsrvol = _FSRVolumes[fsr];
        FP_PRECISION xs = _GPU1DXS[tid];
        FP_PRECISION_FLUX flux = _GPUScalarFlux[tid];
        _Rate[tid] = fsrvol * xs * flux;
    }
}

__global__ void flux_relerr2(FP_PRECISION_FLUX *_newFlux,
                             FP_PRECISION_FLUX *_oldFlux,
                             FP_PRECISION_FLUX *_fluxRelErr2) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t fsr = tid / (*GPUNumMOCGroups);
    size_t g = tid % (*GPUNumMOCGroups);

    FP_PRECISION_FLUX relerr;
    if (tid < (*GPUNumMOCGroups) * (*GPUTotNumFSRs)) {
        relerr = (_newFlux[tid] - _oldFlux[tid]) / _newFlux[tid];
        if (!isnan(relerr))
            _fluxRelErr2[tid] = relerr * relerr;
    }
}


FP_DOUBLE *GPUFSRFissRate;
FP_PRECISION_FLUX *GPUScalarFlux_RelErr;
extern "C"
FP_DOUBLE GPUCalcReacRate_SubDom(int _NumGroups, int _NumFSRs) {
    FP_DOUBLE rate = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);

    int dim = _NumGroups * _NumFSRs;
//    size_t gpugriddim = dim % GPUBlockDim == 0 ? dim / GPUBlockDim : dim / GPUBlockDim + 1;
    size_t gpugriddim = (dim - 1) / GPUBlockDim + 1;

    react_rate_calc<<<gpugriddim, GPUBlockDim>>>(GPUScalarFlux,
                                                 GPUFSRVol_SubDom,
                                                 GPUMacroXS_nf,
                                                 GPUFSRFissRate);

    CUDA_CheckError((char *) "react_rate_calc");

    cublasDasum(handle, dim, GPUFSRFissRate, 1, &rate);

    cublasDestroy(handle);


    return rate;
}

// CUDA 核函数
__global__ void rowSumKernel(FP_PRECISION_FLUX* F, 
							 FP_PRECISION_FLUX* xs,
							 FP_PRECISION_FLUX* sum_f, 
							 int cols) {
    // 动态共享内存，大小在启动时指定
    extern __shared__ FP_PRECISION_FLUX sdata[];

    int i = blockIdx.x; // 对应原代码的 i (1 到 50000)
    int tid = threadIdx.x;
    
    FP_PRECISION_FLUX temp_sum = 0.0;

    // 1. 跨步循环：处理当 cols > blockDim.x 时的所有列元素
    // 这种模式能保证合并访问 (Coalesced Access)
    for (int j = tid; j < cols; j += blockDim.x) {
        // 严格遵循你提供的索引逻辑：idx = i * 200 + j
        // 注意：在大规模下，请确保你的 F 数组是按照这个步长分配的
        // 这里假设步长为 cols (2000)
        long long idx = (long long)i * cols + j; 
		// printf("\n [  DEBUG  ]  i:%d, idx:%12ld, value:%13.5e\n", i, idx, temp_sum);
        temp_sum += F[idx] * xs[idx];
    }

    // 2. 将每个线程的局部和存入共享内存
    sdata[tid] = temp_sum;
    __syncthreads();

    // 3. 树状归约：在 Block 内部将 256 个局部和相加
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 4. 将该 Block 的最终结果写入对应的 sum_f[i]
    if (tid == 0) {
        sum_f[i] = sdata[0];
		// if (sum_f[i] > 0.) 
		//     printf(" \n [  DEBUG  ]  i:%5d, Tid:%12d, Value:%13.5e\n", i, tid, sum_f[i]);
    }
}

__global__ void computeRelativeErrorKernel(FP_PRECISION_FLUX* f1, 
                                           FP_PRECISION_FLUX* f2, 
                                           FP_PRECISION_FLUX* rel_err_array, 
                                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	FP_PRECISION_FLUX relerr;
    if (idx < n) {
        FP_PRECISION_FLUX ref = f2[idx];
		relerr = (f1[idx] - ref) / ref;
		if (!isnan(relerr)) {
			rel_err_array[idx] = relerr * relerr;
			// printf("\n IDx:%5d, ITE1:%13.5e, ITE2:%13.5e, relErr:%13.5e \n", idx, f1[idx], f2[idx], relerr);
		}
		else {
			rel_err_array[idx] = 0.;
		}
    }
}

FP_PRECISION GPUComputeScalarFluxRelErrDot(int _NumGroups, int _NumFSRs, int _ScatterOrder) {
    FP_PRECISION _dot = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);

    int dim = _NumGroups * _NumFSRs;
    size_t gpudriddim = (dim - 1) / GPUBlockDim + 1;

    flux_relerr2<<<gpudriddim, GPUBlockDim>>>(GPUScalarFlux, GPUScalarFluxOld, GPUScalarFlux_RelErr);

    CUDA_CheckError((char *) "GPUGetRelErrOfTwoVectors_FP_I");

	size_t localThread_Per_Block = 1024;
	size_t localSharedMemSize = GPUBlockDim * sizeof (FP_PRECISION_FLUX);
	size_t ROWS = _NumFSRs;
	int COLS = _NumGroups;

    size_t sum_elements = ROWS;

    FP_PRECISION_FLUX *d_sum_f1, *d_sum_f2, *d_err_f3;

    cudaMalloc(&d_sum_f1, sum_elements * sizeof(FP_PRECISION_FLUX));
    cudaMalloc(&d_sum_f2, sum_elements * sizeof(FP_PRECISION_FLUX));
    cudaMalloc(&d_err_f3, sum_elements * sizeof(FP_PRECISION_FLUX));

    CUDA_CheckError((char *) "GPU d_sum_f allocate");

	rowSumKernel<<<ROWS, GPUBlockDim, localSharedMemSize>>> (GPUScalarFluxOld, GPUMacroXS_t, d_sum_f1, COLS);
	CUDA_CheckError((char *) "GPU Cal d_sum_f1");

    dim = _NumGroups * _NumFSRs * (_ScatterOrder + 1) * (_ScatterOrder + 1);
    cudaMemcpy(GPUScalarFluxOld, GPUScalarFlux, dim * sizeof (FP_PRECISION_FLUX), cudaMemcpyDeviceToDevice);
    CUDA_CheckError((char *) "GPUMemcpy-scalarflux transfer");

	rowSumKernel<<<ROWS, GPUBlockDim, localSharedMemSize>>> (GPUScalarFluxOld, GPUMacroXS_t, d_sum_f2, COLS);
	CUDA_CheckError((char *) "GPU Cal d_sum_f2");

#if defined(FP32)
    cublasSasum(handle, dim, GPUScalarFlux_RelErr, 1, &_dot);
#else
    cublasDasum(handle, dim, GPUScalarFlux_RelErr, 1, &_dot);
#endif // defined(FP32)

	// 2. 启动核函数
	int blocksPerGrid = (ROWS + GPUBlockDim - 1) / GPUBlockDim;
	computeRelativeErrorKernel<<<blocksPerGrid, GPUBlockDim>>>(d_sum_f1, d_sum_f2, d_err_f3, ROWS);

	dim = ROWS;

	int max_idx;
	FP_PRECISION_FLUX max_val = 0;

#if defined(FP32)
    cublasIsamax(handle, dim, d_err_f3, 1, &max_idx);
#else
    cublasIdamax(handle, dim, d_err_f3, 1, &max_idx);
#endif

	// cudaMemcpy(&_dot, d_err_f3 + (max_idx - 1), sizeof(FP_PRECISION_FLUX), cudaMemcpyDeviceToHost);
	// _dot *= _NumFSRs * _NumGroups;

	cudaFree(d_sum_f1);
	cudaFree(d_sum_f2);
	cudaFree(d_err_f3);

    cublasDestroy(handle);

    return _dot;
}

__global__ void warm_up_gpu() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

void warmup() {
    warm_up_gpu<<<65535, 1024>>>();
}

__global__ void compute_source_on_device_inner_iteration(int _layID,
                                         FP_PRECISION_FLUX *_GPUScalarFlux,
                                         FP_PRECISION_FLUX *_GPUScalarFluxOld,
                                         FP_PRECISION *_GPUSource,
                                         FP_PRECISION *_GPUMacroXS_t,
                                         FP_PRECISION *_GPUMacroXS_nf,
                                         FP_PRECISION *_GPUMacroXS_s,
                                         FP_PRECISION *_GPUMacroXS_chi,
                                         FP_PRECISION _inverseKeff,
                                         M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
                                         M_UINT64 *_GPUMacroXS_s_NZRowEndID_Layer, 
                                         M_UINT64 *_GPUMacroXS_s_NZColStartID) {
    int _tid = threadIdx.x + blockIdx.x * blockDim.x, g1, _fsrID, _groupID, _ind, g2, nnz, goffset, flag1, flag2;
    FP_PRECISION _chi, _xs_nf, _xs_s, _fissSour, _scattSour;
#if defined(R_AD)
    int _ind2;
    FP_PRECISION_FLUX _flux2;
#endif // resonance adjoint calculation
    FP_PRECISION_FLUX _flux;
	FP_PRECISION_FLUX _fluxOld;
    M_UINT64 _id, _iid;
#if defined(M_2DMOC)
    FP_PRECISION _xs_t;
#endif // defined(M_2DMOC)
    _fsrID = _tid / (*GPUNumMOCGroups);
    _groupID = _tid % (*GPUNumMOCGroups);

#ifdef FIS_MTX
    int _indChi;
#endif

    /* Iterate over all FSRs */
    while (_fsrID < (*GPUNumFSRsLayer)) {
#if defined(M_2DMOC)
        _xs_t = _GPUMacroXS_t[_tid];
#endif // defined(M_2DMOC)
        _chi = _GPUMacroXS_chi[_tid];
        _fissSour = 0.0;
        _scattSour = 0.0;

        for (g1 = 0; g1 < (*GPUNumMOCGroups); g1++) {
            _ind = _fsrID*(*GPUNumMOCGroups) + g1;
            _flux = _GPUScalarFlux[_ind];
			_fluxOld = _GPUScalarFluxOld[_ind];
        #if defined(R_AD)
            _ind2 = _fsrID * (*GPUNumMOCGroups) + _groupID;
            g2 = _GPUMacroXS_s_NZColStartID[_ind2];
            _id = _GPUMacroXS_s_NZRowStartID_Layer[_ind2];
            _iid = _GPUMacroXS_s_NZRowEndID_Layer[_ind2];
            goffset = g1 - g2;
        #else
            g2 = _GPUMacroXS_s_NZColStartID[_ind];
            _id = _GPUMacroXS_s_NZRowStartID_Layer[_ind];
            _iid = _GPUMacroXS_s_NZRowEndID_Layer[_ind];
            goffset = _groupID - g2;
        #endif

            _xs_nf = _GPUMacroXS_nf[_ind];

            nnz = _iid - _id;
            
            flag1 = magic(goffset, 0);
            flag2 = magic(nnz-1, goffset);
            _xs_s = flag1 * flag2 * _GPUMacroXS_s[_id + goffset];

            _scattSour += _xs_s * _flux;

        #ifdef FIS_MTX
            _indChi = _fsrID * (*GPUNumMOCGroups) * (*GPUNumMOCGroups) + g1 * (*GPUNumMOCGroups) + _groupID;
            _fissSour += _GPUMacroXS_chi[_indChi] * _xs_nf * _flux;
        #else
            _fissSour += _xs_nf * _fluxOld;
        #endif
        }

#if defined(M_ZeroNegFissSour)
        if (_scattSour<0.0) {
			_scattSour = 0.0;
		}
#endif
        #ifdef FIS_MTX
            _fissSour *= _inverseKeff;
        #else
            _fissSour *= (_chi*_inverseKeff);
        #endif

	#ifdef FLAT_FIS_SOURCE
		_GPUSource[_tid] = _chi + _scattSour;
	#else
		_GPUSource[_tid] = _fissSour + _scattSour;
	#endif

#if defined(M_2DMOC)
        // _GPUSource[_tid] = _GPUSource[_tid] * ONE_OVER_FOUR_PI / _xs_t;
        _GPUSource[_tid] /= _xs_t;
#endif // defined(M_2DMOC)

// #if defined(M_3DMOCex)
#if defined(M_Use4PI)
        _GPUSource[_tid] *= ONE_OVER_FOUR_PI;
#endif // defined(M_Use4PI)
// #endif // defined(M_3DMOCex)
        _tid += blockDim.x * gridDim.x;
        _fsrID = _tid / (*GPUNumMOCGroups);
        _groupID = _tid % (*GPUNumMOCGroups);
    }

}

void GPUCopyMacroXSsHostToDevice_Init(int _TotalNumFSRs_subDom, int _NumGroups ,int _MOCNumLayersZ,\
    M_UINT64 *_MacroXS_s_NZRowStartID_Layer, M_UINT64 *_MacroXS_s_NZColStartID, size_t *_GPUPreMemXSs,\
	M_INT32 _IfResonance, M_INT32 _ProblemType, M_UINT64 _MacroXS_s_MaxNNZ_Layer)
{
	if (_IfResonance == 1 && _ProblemType == 0) {
		size_t _dim = 0;
		_dim = (size_t)(_MacroXS_s_MaxNNZ_Layer) * sizeof(FP_PRECISION);
	    cudaMalloc((void**)&GPUMacroXS_s, _dim);
		_GPUPreMemXSs[0] += _dim;
	    CUDA_CheckError((char *)("cudaMalloc-GPUMacroXS_s-1"));
	}
	// cudaMemcpy((void*)GPUMacroXS_s_NZRowStartID_Layer, (void*)_MacroXS_s_NZRowStartID_Layer,(size_t)(_TotalNumFSRs_subDom*_NumGroups+_MOCNumLayersZ)*sizeof(M_UINT64), cudaMemcpyHostToDevice);
	// CUDA_CheckError((char *)("cudaMemcpy-GPUMacroXS_s_NZRowStartID_Layer"));
	// cudaMemcpy((void*)GPUMacroXS_s_NZColStartID, (void*)_MacroXS_s_NZColStartID,(size_t)(_TotalNumFSRs_subDom*_NumGroups)*sizeof(M_INT32), cudaMemcpyHostToDevice);
	// CUDA_CheckError((char *)("cudaMemcpy-GPUMacroXS_s_NZColStartID"));
}
