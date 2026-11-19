/********************************************************************************/
/* @ File: MOC_GPUCPUInterface.c
/*   introduction: This file contains functions that transfer multidimensional data 
/*                 array allocated on CPU to one-dimension array, and the 1-D array 
/*                 will be copied from host to GPU device
/* @ Created     : 2018/05/20, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*   1st modified:
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "MOC_AllocAndFreeMem.h"
#include "MOC_CommonMacro.h"
#include "MOC_EigenValueEvaluation.h"
#include "MOC_ExpEvaluation.h"
#include "MOC_FluxEvaluation.h"
#include "MOC_Geometry.h"
#include "MOC_GPUCPUInterface.h"
#include "MOC_Input.h"
#include "MOC_main.h"
#include "MOC_Macro_XSs.h"
#include "MOC_Output.h"
#include "MOC_Quadrature_Set.h"
#include "MOC_ReacRateEvaluation.h"
#include "MOC_Track.h"
#include "MOC_VectorOperation.h"
#ifdef MPI_USED
  #include <mpi.h>
#endif // MPI_USED
#ifdef CUDA_USED
  #include "MOC_GPUQuery.cuh"
  #include "MOC_GPUSolver.cuh"
  #include "MOC_GPU_TSSolver.cuh"
  #include "MOC_GPUCmfd.cuh"
	
  #include "cuda_runtime.h"
#endif

// #if defined(M_MG_CMFD)
  #include "MOC_CMFD.h"
// #endif // defined(M_MG_CMFD)
#include "paper_transport_compat.h"

#ifdef TESTR
  #include "zres_common_func.h"
#endif


//FP_PRECISION *GC_TotalWeights=NULL;
//FP_PRECISION *GC_FSRsAreaInRTM=NULL;
//int *GC_SumNumFSRsInRTM=NULL;
//int GC_TotNumFSRsInRTM;
int *GC_SumNumLTs=NULL;

FP_PRECISION *CPUReacRateCollect=NULL;
FP_PRECISION GPUTotReacRate_subDom;
FP_PRECISION GPUGlobFissTemp;
FP_PRECISION GPUGlobFissOld,GPUKeff,GPUKeffOld;
FP_PRECISION *GPUFluxResRColl=NULL;
FP_PRECISION GPUFluxRes_subDom;
FP_PRECISION GPUFluxRes_Glob;

FP_PRECISION *GC_SinThetasTimes2 = NULL;
FP_PRECISION *GC_OneOverSinThetas = NULL;
FP_PRECISION_TIME CudaMemUse[1];
int GPUAzimSweepMode = 0;

void GPUInitAzimSweepMode(void)
{
	static int initialized = 0;
	const char *envMode;
	int mode;

	if (initialized) {
		return;
	}

	envMode = getenv("GPU_AzimSweepMode");
	if (envMode != NULL && envMode[0] != '\0') {
		mode = atoi(envMode);
		if (mode < 0) {
			fprintf(stderr, "\n[ERROR] GPU_AzimSweepMode must be non-negative (0: full-angle, positive: azimuth batch size).\n");
			exit(71);
		}
		GPUAzimSweepMode = mode;
	}

	initialized = 1;
	if (GPUAzimSweepMode == 0) {
		printf("\n [GPU-INFO ] 2D azimuth sweep mode: 0 (full-angle)");
	}
	else {
		printf("\n [GPU-INFO ] 2D azimuth sweep mode: %d (batched azimuths)",
			GPUAzimSweepMode);
	}
	fflush(stdout);
}
/**
 * @ Function: GC_getTotWeights()
 *   introduction: transfer 3-D CPU parameter _totalWeights to 1-D array
 * @ Parameter List: _totalWeights - total weights [trackSetID][_AzimAngID][_PolAngID]
 *   Return: void
 * @ Created     : 2018/05/20, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
void GC_getSinThetasTimes2(FP_PRECISION **_sinTheta)
{
    int i;
    GC_SinThetasTimes2 =AllocMem_FP_I_1(NumPolarAng[0],(char *)("GC_SinThetasTimes2"));
	GC_OneOverSinThetas = AllocMem_FP_I_1(NumPolarAng[0], (char *)("GC_OneOverSinThetas"));
    for (i=0;i<NumPolarAng[0];i++) {
		GC_SinThetasTimes2[i] = (FP_PRECISION)(2.0)*_sinTheta[0][i];
		GC_OneOverSinThetas[i] = (FP_PRECISION)(1.0)/_sinTheta[0][i];
	}
}


/**
 * @ Function: GC_getSumNumLTs_Azim()
 *   introduction: get the number of long tracks accumulated by azimuthal angle
 *                 GC_SumNumLTs[_azimAngID]=sumation of _numLongTracks[_aazim<_azimAngID]
 * @ Parameter List: _numLongTracks - number of long tracks in current domain
 *                   _numAzimAng - number of azimuthal angles
 *                   
 *   Return: void
 * @ Created     : 2018/05/25, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
#ifdef CUDA_USED
/**
 * @ Function: GPUComputeKeff()
 *   introduction: evaluate the keff
 * @ Parameter List: 
 *   Return: 
 * @ Created     : 2018/06/05, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
void InitCudaDevice_1(int _CPUProcID, int _numProcs)
{
#ifdef TIME_REC
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
	#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
	#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC

	InitCudaDevice(_CPUProcID, _numProcs, ProcessorName);

#if defined(MPI_USED)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
        GPUQuery_1(_CPUProcID, CUDADeviceID[_CPUProcID]);
#ifdef TIME_REC
	#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
	#endif // defined(CUDA_USED)
	_time_end = TallyCurrentTime_CPU();
	TimePrePart[TimeCountPrePart] = _time_end - _time_start;
	strcpy(TimeMetricsPrePart[TimeCountPrePart], "Set CUDA device");
	TimeCountPrePart++;
#endif // TIME_REC
}

void CopyInitInfoFromCPUtoGPU(int _procID)
{
#ifdef TIME_REC
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC

	/* initialize quadrature set on device */
	GPUInitQuadratureSet(_procID, GPUPreMemQuadrSet, GPUPreMemExpTab, LogFile,NumAzimAng[0], \
		NumPolarAng[0], SinTheta[0], CosTheta[0], GC_SinThetasTimes2, \
		GC_OneOverSinThetas, SinThetaForExp, AzimWeights[0], PolarWeights[0], \
		TotalWeights[0], TotalWeights_2DMOC[0], ExpTable, NumGroups, NumScattOrder, \
		CoeffA_HOScatt, SpheHarmCoeff[0]);
	/* initialize track information on device */
    if (InputTrackFile == 0) {
        GPUCopyTrackInfoHostToDevice(_procID, NumGroups, TotalNumFSRs_subDom, TotalNumFSRs_perLayer,
            MOC_NumPinsInSubDom, NumPinsInSubDomPerLayer, \
            TotalNumFSRsInRTM, NumPolarAng[0], NumLTs_HS[0], NumLTs_2PI[0], LTsIndex_AzimAngID_HS[0],
            LTsIndex_LTID_HS[0], LTsIndex_AzimAngID_2PI[0], \
            LTsIndex_LTID_2PI[0], TracksInfoInLTs_2PI[0], TrackIndexInLTs_2PI[0], TotNumTracksInOneRTM_2PI[0],
                                     IndexTrackInRTM_SumAzim_2PI[0], \
            TotalNumTracksInRTM_Total[0], NumSegsInTrack_2PI[0], TotNumSegInRTM_Total[0], SegInfoInTrack_SINGLE_2PI[0],
                                     SumNumSegsInRTMTrack_2PI[0], \
            SegInfoInTrack_2PI_length[0], SegInfoInTrack_2PI_RayWisedFSR[0], SegInfoInTrack_2PI_AntiClockFSR[0],
                                     FsrIdMap_RayWisedToAntiClock[0], \
            TracksInfoInLTs_HS[0], NumTracks_HS[0], LongTracksInfo_Total[0], TrackIndexInLTs_HS[0],
                                     TotalNumTracksInRTM_HS[0], NumSegsInTrack_HS[0], \
            TotNumTracksInOneRTM_HS[0], NumAzimAng[0], IndexTrackInRTM_SumAzim_HS[0], TotNumSegInRTM_HS[0],
                                     SegInfoInTrack_SINGLE_HS[0], MOCNumLayersZ, \
            SumNumFSRsBasedGlobCellID, NumRTMGeoType, SumNumFSRsInRTM, CellRTMType_SubDom, SumNumSegsInRTMTrack_HS[0], \
            NumLTsSum[0], NumLongTracks[0], OneOverFSRArea_SubDom, FSRVol_SubDom, NorthBC, SouthBC, EastBC, WestBC,
                                     BottomBC, TopBC, NumFSRsInCell, \
            OneOverFSRsAreaInRTM_AntiClockFSR, EffRaySpacing[0], NumFSRs_Layer, SumNumFSRs_Layer, WidthLayersInCore,
                                     MOCNumPinsInSubDom_X, \
            MOCNumPinsInSubDom_Y, RayTracModDim, NumSurfaceX, NumSurfaceY, NumTracksX[0], NumTracksY[0], RTMWidth_X,
                                     RTMWidth_Y, NumFSRsInRTM, \
            FSRsAreaInRTM_AntiClockFSR, FSRArea_SubDom, GPUPreMemTrack, GPUPreMemXSs, NumTracks_Total[0], LogFile);
    } else{
        GPUCopyTrackInfoHostToDevice_trackFile(_procID, NumGroups, TotalNumFSRs_subDom, TotalNumFSRs_perLayer, MOC_NumPinsInSubDom, NumPinsInSubDomPerLayer, \
			TotalNumFSRsInRTM, NumPolarAng[0], NumLTs_HS[0], NumLTs_2PI[0], LTsIndex_AzimAngID_HS[0], LTsIndex_LTID_HS[0], LTsIndex_AzimAngID_2PI[0], \
			LTsIndex_LTID_2PI[0], TracksInfoInLTs_2PI[0], TrackIndexInLTs_2PI[0], TotNumTracksInOneRTM_2PI[0], IndexTrackInRTM_SumAzim_2PI[0], \
			TotalNumTracksInRTM_Total[0], NumSegsInTrack_2PI[0], TotNumSegInRTM_Total[0], /*SegInfoInTrack_SINGLE_2PI[0],*/ SumNumSegsInRTMTrack_2PI[0], \
			SegInfoInTrack_2PI_length[0], /*SegInfoInTrack_2PI_RayWisedFSR[0],*/ SegInfoInTrack_2PI_AntiClockFSR[0], /*FsrIdMap_RayWisedToAntiClock[0],*/ \
			TracksInfoInLTs_HS[0], NumTracks_HS[0], LongTracksInfo_Total[0], /*TrackIndexInLTs_HS[0], TotalNumTracksInRTM_HS[0], NumSegsInTrack_HS[0], \
			TotNumTracksInOneRTM_HS[0],*/ NumAzimAng[0], /*IndexTrackInRTM_SumAzim_HS[0], TotNumSegInRTM_HS[0], SegInfoInTrack_SINGLE_HS[0],*/ MOCNumLayersZ, \
			SumNumFSRsBasedGlobCellID, NumRTMGeoType, SumNumFSRsInRTM, CellRTMType_SubDom, /*SumNumSegsInRTMTrack_HS[0],*/ \
			NumLTsSum[0], NumLongTracks[0], /*OneOverFSRArea_SubDom,*/ FSRVol_SubDom, NorthBC, SouthBC, EastBC, WestBC, BottomBC, TopBC, NumFSRsInCell, \
			OneOverFSRsAreaInRTM_AntiClockFSR, EffRaySpacing[0],NumFSRs_Layer, SumNumFSRs_Layer, /**/ WidthLayersInCore, MOCNumPinsInSubDom_X, \
			MOCNumPinsInSubDom_Y, /*RayTracModDim,*/ NumSurfaceX, NumSurfaceY, NumTracksX[0], NumTracksY[0], RTMWidth_X, RTMWidth_Y, NumFSRsInRTM, \
			FSRsAreaInRTM_AntiClockFSR, FSRArea_SubDom, GPUPreMemTrack, GPUPreMemXSs, NumTracks_Total[0], LogFile);
    }

#if defined(M_3DOpt) && defined(M_3DMOCex)
	GPUSetXSt_wScale_1();
#endif //defined(M_3DOpt) && defined(M_3DMOCex)

#ifdef TIME_REC
	#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
	#endif // defined(CUDA_USED)
	_time_end = TallyCurrentTime_CPU();
	TimePrePart[TimeCountPrePart] = _time_end - _time_start;
	strcpy(TimeMetricsPrePart[TimeCountPrePart], "Copy Init Info Host-->Device");
	TimeCountPrePart++;
#endif // TIME_REC
}

#ifdef GPU_STREAM_USED
void GPUStreamInitialize_1()
{
#ifdef TIME_REC
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
	#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
	#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC

	GPUStreamInitialize();

#ifdef TIME_REC
	#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
	#endif // defined(CUDA_USED)
	_time_end = TallyCurrentTime_CPU();
	TimeOverall[TimeCount] = _time_end - _time_start;
	strcpy(TimeMetrics[TimeCount], "Time for Preparing Stream Info");
	TimeCount++;
	TimeCountPrePart++;
#endif // TIME_REC
}
#endif //GPU_STREAM_USED

void GPUTransportSweepInit_1(int _procID, int _numProcs, int _totalNumFSRs, int _numGroups, int _totNumLongTracks, int _numPolarAng)
{
#ifdef TIME_REC
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC

	/* initialize stream invoking */
#ifdef GPU_STREAM_USED
	GPUStreamInitialize();
#endif // GPU_STREAM_USED

	SetKernelCallingPara(NumLTs_HS[0], _numGroups, NumPolarAng[0], TotalNumFSRs_subDom, TotalNumFSRs_perLayer, \
		MOCNumLayersZ, MOCNumPinsInSubDom_X, MOCNumPinsInSubDom_Y, NumCMFDGroup, MOC_NumPinsInSubDom, NumPinsInSubDomPerLayer, \
		NumCoarseCMFDGroup);

	/* Initialize transport sweep*/
	//GPUTransportSweepInit(_procID, _numProcs, _totalNumFSRs, _numGroups, _totNumLongTracks, _numPolarAng);
	MOC_InitTransSweep(_procID);
// #if defined(CUDA_USED)
//     // if (ProblemType == 1) {
// 		VectorCopy_FP_Flux(ScalarFluxOld, ScalarFlux_MOC, TotalNumFSRs_subDom*NumGroups);
// 	// }
// #endif // defined(CUDA_USED)

#ifdef TIME_REC
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_end = TallyCurrentTime_CPU();
	TimePrePart[TimeCountPrePart] = _time_end - _time_start;
	strcpy(TimeMetricsPrePart[TimeCountPrePart], "Init TransSweep on GPU");
	TimeCountPrePart++;
#endif // TIME_REC
}

void GPU_MainAllocMem_1(int _procID, int _numProcs, M_INT32 _totalNumFSRs, int _numGroups, int _totNumLongTracks, int _numPolarAng)
{
	GPUInitAzimSweepMode();
#ifdef TIME_REC
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC
	FP_PRECISION_TIME  _cudaMemTot[1];

	/* initialize kernel invoking parameters */
	SetKernelCallingPara(NumLTs_HS[0], _numGroups, NumPolarAng[0], TotalNumFSRs_subDom, TotalNumFSRs_perLayer, \
		MOCNumLayersZ, MOCNumPinsInSubDom_X, MOCNumPinsInSubDom_Y, NumCMFDGroup, MOC_NumPinsInSubDom, NumPinsInSubDomPerLayer, \
		NumCoarseCMFDGroup);

	/* allocate Mem on GPU for transport sweep*/
	GPU_MainAllocMem(_procID, _numProcs, _totalNumFSRs, _numGroups, _totNumLongTracks, _numPolarAng, \
		GPUPreMemScalF, GPUPreMemRadF, GPUPreMemAxiF, GPUPreMemMGCMFD, GPUPreMemXSs, TotalNumFSRs_perLayer, \
		IfResonance, ProblemType, MacroXS_s_MaxNNZ_Layer, MOCNumLayersZ, NumAzimAng[0], GPUPreMemCGCMFD, MOC_NumPinsInSubDom, \
		NumCMFDGroup, MOCNumPinsInSubDom_X, MOCNumPinsInSubDom_Y, CMFDGroupBound, CMFD_NodeVol, NumSubDomains, \
		MGCMFD_MOCParCurrInnerBound_Num, MGCMFD_MOCParCurrInnerBound_Index, MGCMFD_MOCParCurrInnerBound_Dir, \
		CMFDComm_TarProc, MGCMFDData_MPIComm_Num, MGCMFDData_MPIComm_SumInd, MGCMFDMPI_InBoundSourCoeff_Index, \
		MGCMFDMPI_InBoundSourCoeff_Dir, MGCMFDMPI_InBoundSourCoeff_GeoDim, MGCMFDData_MPIComm_Index, CGCMFDData_MPIComm_Num, \
		CGCMFDMPI_InBoundSourCoeff_Index, CGCMFDMPI_InBoundSourCoeff_Dir, CGCMFDMPI_InBoundSourCoeff_GeoDim, \
		CGCMFDData_MPIComm_Index, NumCoarseCMFDGroup, CGCMFDGroupBound, CGCMFDData_MPIComm_SumInd, MGCMFD_LinearSolver, \
		CGCMFD_LinearSolver, LogFile, GmresKrylovDim_MG, GmresKrylovDim_CG, OUTPUT_CMFDGroupConstant, \
		OUTPUT_CoarseMesh_GroupConstant, NumScattOrder, GPU_TSKernelType, isTransient);
	/* check constant mem usage on GPU */
	GetConstantMemUsage_GPU(_procID, NumAzimAng[0], NumPolarAng[0], LogFile);
    if (ProcID==0) {
		GPUMemUsage(ProcID, CudaMemUse, _cudaMemTot);
		printf("\n [GPU-INFO ]  Memory Usage on GPU: %8.2lf GB of %8.2lf GB",CudaMemUse[0]/1024., _cudaMemTot[0]/1024.);
		fflush(stdout);
		fprintf(LogFile, "\n [GPU-INFO ]  Memory Usage on GPU: %8.2lf GB of %8.2lf GB",CudaMemUse[0]/1024., _cudaMemTot[0]/1024.);
	}

#ifdef TIME_REC
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_end = TallyCurrentTime_CPU();
	TimePrePart[TimeCountPrePart] = _time_end - _time_start;
	strcpy(TimeMetricsPrePart[TimeCountPrePart], "Alloc Mem on GPU");
	TimeCountPrePart++;
#endif // TIME_REC
}

/*
 * P0 has no persistent angular-flux or moment buffer.  Consequently, the
 * azimuthal loop must be inside the layer loop: GPUScalarFlux is the
 * accumulator for the current layer only.  Reusing the high-order
 * azimuth-outer driver here would overwrite the current-layer accumulator
 * before all azimuths had contributed to that layer.
 */
static void GPUMOC_TransSweep_P0_PerAzim(int _iter, int _innerIter)
{
	const int _dim2 = TotalNumFSRs_perLayer * NumGroups;
	const int _numAzim = NumAzimAng[0];
	const int _azimBatch = GPUAzimSweepMode > 0 && GPUAzimSweepMode < _numAzim
		? GPUAzimSweepMode : _numAzim;
	int _lay, _azimBegin;

	for (_lay = 0; _lay < MOCNumLayersZ; ++_lay) {
		const FP_PRECISION _twoOverLayWidth =
			(FP_PRECISION) 2.0 / WidthLayersInCore[_lay];
		FP_PRECISION _keff;

		if (isTransient < 2) {
			_keff = MOCKeff;
		}
		else {
			_keff = sKeff;
		}

		for (_azimBegin = 0; _azimBegin < _numAzim; _azimBegin += _azimBatch) {
			const int _azimEnd =
				_azimBegin + _azimBatch < _numAzim
				? _azimBegin + _azimBatch : _numAzim;

			/* P0 source is angle independent and is evaluated once per layer. */
			GPUComputeFSRSource_Layer_PerAzim(
				_lay, TotalNumFSRs_subDom, TotalNumFSRs_perLayer, NumGroups,
				MacroXS_s_NZStartID_Layer, MacroXS_s_NNZ_Layer, ProblemType,
				MacroXS_t, MacroXS_nf, Chi, MacroXS_s, FixedSour,
				(FP_PRECISION) 1.0 / _keff, NumScattOrder, GPU_TSKernelType,
				MacroXS_s_NZRowStartID_Layer, MacroXS_s_NZColStartID,
				ScalarFluxOld, 0, _azimBegin, _azimEnd, _azimBegin == 0 ? 0 : 1,
				isTransient, FissSource, tranSrcConstA, tranSrcConstB,
				tranSrcConstC);

			/* Start the angular accumulation for this layer only once. */
			if (_azimBegin == 0) {
				GPUInitScalarFlux_Dev(_dim2, 0.0);
			}

			GPUTransportSweep(_lay, _twoOverLayWidth, 0, NumPolarAng[0],
				NULL, NULL, 0, GPU_TSKernelType, NumPinsInSubDomPerLayer,
				NumPolarAng[0], _azimBegin, _azimEnd);
		}
		/* GPUTransportSweep adds the source after the last azimuth. */
	}

	(void) _iter;
	(void) _innerIter;
}

static void GPUTransportSweep_1_PerAzim(int _iter)
{
	const int _polarBoundLowLim = 0;
	const int _polarBoundUpLim = NumPolarAng[0];
	const int _polOffset = 0;
	const int _isP0 = (NumScattOrder == 0 && GPU_TSKernelType < 10);
	const int _localAngDim = TotalNumFSRs_perLayer * NumGroups * NumPolarAng[0];
	const int _momentDim = TotalNumFSRs_perLayer * NumGroups *
		(NumScattOrder + 1) * (NumScattOrder + 1);
	const int _numAzim = NumAzimAng[0];
	const int _azimBatch = GPUAzimSweepMode > 0 && GPUAzimSweepMode < _numAzim
		? GPUAzimSweepMode : _numAzim;
	FP_PRECISION_TIME _fluxStageStart = 0.0, _fluxStageEnd = 0.0;

#if defined(TIME_REC) || defined(DynamicWorkLoad)
	FP_PRECISION_TIME _time_start = 0.0, _time_end = 0.0;
#if defined(CUDA_USED)
	cudaDeviceSynchronize_cpu();
#endif
	_time_start = TallyCurrentTime_CPU();
#endif

	for (int _inner = 0; _inner < MaxNumInnerIter; ++_inner) {
		if (_isP0) {
			GPUMOC_TransSweep_P0_PerAzim(_iter, _inner);
			continue;
		}

		/* Keep the old scalar flux unchanged while all azimuths are swept.
		 * The new moments are accumulated in a separate LM-sized buffer. */
		cudaDeviceSynchronize();
		_fluxStageStart = TallyMonotonicTime_CPU();
		GPUInitAzimAccumFlux(_momentDim, (FP_PRECISION_FLUX) 0.0);
		cudaDeviceSynchronize();
		_fluxStageEnd = TallyMonotonicTime_CPU();
		TimeRec_FluxAccumulation += _fluxStageEnd - _fluxStageStart;
		TimeRec_FluxMomentAzimAccumInit += _fluxStageEnd - _fluxStageStart;

		for (int _azimBegin = 0; _azimBegin < _numAzim; _azimBegin += _azimBatch) {
			const int _azimEnd = _azimBegin + _azimBatch < _numAzim
				? _azimBegin + _azimBatch : _numAzim;
			GPUMOC_TransSweep_UporDown(_localAngDim, 0, 1, 0,
				_polarBoundLowLim, _polarBoundUpLim, _polOffset, NULL,
				_isP0 ? NULL : GPUFSRAveAngFlux_Top, 0, 0,
				(FP_PRECISION) 1.0, 0,
				_iter, (_inner == 0 && _azimBegin == 0) ? 1 : 0, _inner,
				_azimBegin, _azimEnd);
		}

		cudaDeviceSynchronize();
		_fluxStageStart = TallyMonotonicTime_CPU();
		GPUFinalizeAzimAccumFlux(_momentDim);
		cudaDeviceSynchronize();
		_fluxStageEnd = TallyMonotonicTime_CPU();
		TimeRec_FluxAccumulation += _fluxStageEnd - _fluxStageStart;
		TimeRec_FluxMomentAzimAccumFinalize += _fluxStageEnd - _fluxStageStart;
	}

#if defined(TIME_REC) || defined(DynamicWorkLoad)
#if defined(CUDA_USED)
	cudaDeviceSynchronize_cpu();
#endif
	_time_end = TallyCurrentTime_CPU();
#ifdef TIME_REC
	if (_iter == 0) {
		TimeStamp_TS = TimeCount;
		strcpy(TimeMetrics[TimeStamp_TS], "Transport Sweep");
		TimeCount++;
	}
	TimeOverall[TimeStamp_TS] += _time_end - _time_start;
#endif
#ifdef DynamicWorkLoad
	TimeTS_DWL += _time_end - _time_start;
#endif
#endif
}

void GPUTransportSweep_1(int _iter)
{
	FP_PRECISION_TIME _transportCalculationStart;
	#if defined(CUDA_USED)
	cudaDeviceSynchronize();
	#endif
	_transportCalculationStart = TallyMonotonicTime_CPU();
#if defined(M_2DMOC)
	GPUInitAzimSweepMode();
	if (GPUAzimSweepMode) {
		GPUTransportSweep_1_PerAzim(_iter);
		#if defined(CUDA_USED)
		cudaDeviceSynchronize();
		#endif
		TimeRec_TransportCalculation +=
			TallyMonotonicTime_CPU() - _transportCalculationStart;
		return;
	}
#endif
    //FP_PRECISION _twoOverLayWidth;
	int /*_lay,*/_polarBoundLowLim=0, _polarBoundUpLim=0,_polOffset=0,_dim,_halfNumPol, _inner;
	FP_PRECISION *_axiFlux_inp = NULL, *_axiFlux_avg = NULL, _JzDir = (FP_PRECISION)(1.0);
    int _layStart=0, _layEnd=0, _layAdd=0, _nodeLay=0, _surfLay=0, _dir=0, _surfLayInc=0;
    _dim=TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0]*(NumPolarAng[0]/2);

#if defined(TIME_REC) || defined(DynamicWorkLoad)
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // #if defined(TIME_REC) || defined(DynamicWorkLoad)

	_halfNumPol = NumPolarAng[0] / 2;
#if defined(M_2DMOC)
	_layStart = 0;
	_layEnd = MOCNumLayersZ;
	_layAdd = 1;
	_polarBoundLowLim = 0;
	_polarBoundUpLim = NumPolarAng[0];
	_polOffset = 0;
	_axiFlux_inp = NULL;
	if (NumScattOrder > 0 || GPU_TSKernelType >= 10) {
		_axiFlux_avg = GPUFSRAveAngFlux_Top;
		// _polarBoundUpLim = NumPolarAng[0];
	}
	else {
		_axiFlux_avg = NULL;
		// _polarBoundUpLim = NumPolarAng[0]/2;
	}
#endif // defined(M_2DMOC)

for (_inner=0;_inner<MaxNumInnerIter;_inner++) {

//   GPUInitScalarFlux(TotalNumFSRs_subDom*NumGroups, (FP_PRECISION)(0.0));

#if defined(M_MG_CMFD)
    GPUCMFD_InitMOCCurr((FP_PRECISION)(0.0),(FP_PRECISION)(0.0), MOCNumLayersZ, \
		MOCNumPinsInSubDom_X, MOCNumPinsInSubDom_Y, NumGroups);
#endif // defined(M_MG_CMFD)


#if defined(M_3DMOCex)
	if ((BottomBC == TopBC) || (BottomBC == 0 && TopBC == 1)) {
		_layStart = 0;
		_layAdd = 1;
		_polarBoundLowLim = 0;
		_polarBoundUpLim = _halfNumPol;
		_axiFlux_inp = GPUFSRAveAngFlux_Bott;
		_axiFlux_avg = GPUFSRAveAngFlux_Top;
		_polOffset = 0;
		_nodeLay = MOCNumLayersZ - 1;
		_surfLay = MOCNumLayersZ;
		_JzDir = (FP_PRECISION)(1.0);
		_dir = 0;
		_surfLayInc = 0;
	}
	else if (BottomBC == 1 && TopBC == 0) {
		_layStart = MOCNumLayersZ - 1;
		_layAdd = -1;
		_polarBoundLowLim = _halfNumPol;
		_polarBoundUpLim = NumPolarAng[0];
		_axiFlux_inp = GPUFSRAveAngFlux_Top;
		_axiFlux_avg = GPUFSRAveAngFlux_Bott;
		_polOffset = _halfNumPol;
		_nodeLay = 0;
		_surfLay = 0;
		_JzDir = (FP_PRECISION)(-1.0);
		_dir = 1;
		_surfLayInc = 1;
	}
	else {}
#endif // defined(M_3DMOCex)

    GPUMOC_TransSweep_UporDown(_dim, _layStart, _layAdd, _surfLayInc, \
	    _polarBoundLowLim, _polarBoundUpLim, _polOffset, _axiFlux_inp, \
	    _axiFlux_avg, _nodeLay, _surfLay, _JzDir, _dir, _iter, 1,_inner, -1, -1);

#if defined(M_3DMOCex)
	 if ((BottomBC == TopBC) || (BottomBC == 0 && TopBC == 1)) {
		 _layStart = MOCNumLayersZ - 1;
		 _layAdd = -1;
		 _polarBoundLowLim = _halfNumPol;
		 _polarBoundUpLim = NumPolarAng[0];
		 _axiFlux_inp = GPUFSRAveAngFlux_Top;
		 _axiFlux_avg = GPUFSRAveAngFlux_Bott;
		 _polOffset = _halfNumPol;
		 _nodeLay = 0;
		 _surfLay = 0;
		 _JzDir = (FP_PRECISION)(-1.0);
		 _dir = 1;
		 _surfLayInc = 1;
	 }
	 else if (BottomBC == 1 && TopBC == 0) {
		 _layStart = 0;
		 _layAdd = 1;
		 _polarBoundLowLim = 0;
		 _polarBoundUpLim = _halfNumPol;
		 _axiFlux_inp = GPUFSRAveAngFlux_Bott;
		 _axiFlux_avg = GPUFSRAveAngFlux_Top;
		 _polOffset = 0;
		 _nodeLay = MOCNumLayersZ - 1;
		 _surfLay = MOCNumLayersZ;
		 _JzDir = (FP_PRECISION)(1.0);
		 _dir = 0;
		 _surfLayInc = 0;
	 }
	 else {}

	GPUMOC_TransSweep_UporDown(_dim, _layStart, _layAdd, _surfLayInc, \
	    _polarBoundLowLim, _polarBoundUpLim, _polOffset, _axiFlux_inp, \
	    _axiFlux_avg, _nodeLay, _surfLay, _JzDir, _dir,_iter, 0,_inner, -1, -1);
#endif // defined(M_3DMOCex)

}

#ifdef TIME_REC // barrier for transfor angFlux data from gpu to cpu
#if defined(CUDA_USED) 	
   cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
    _time_end = TallyCurrentTime_CPU();
	if (_iter == 0) {
		TimeStamp_TS = TimeCount;
		strcpy(TimeMetrics[TimeStamp_TS], "Transport Sweep");
		TimeCount++;
	}
	TimeOverall[TimeStamp_TS] += _time_end - _time_start;
#endif // TIME_REC

#ifdef DynamicWorkLoad
	#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
	#endif // defined(CUDA_USED)
	_time_end = TallyCurrentTime_CPU();
	TimeTS_DWL += _time_end - _time_start;
#endif // DynamicWorkLoad

	#if defined(CUDA_USED)
	cudaDeviceSynchronize();
	#endif
	TimeRec_TransportCalculation +=
		TallyMonotonicTime_CPU() - _transportCalculationStart;
}


#endif //CUDA_USED

//void GPUGetFluxResidualFromDevice_1(int _iter)
//{
//#ifdef TIME_REC
//	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
//	_time_start = TallyCurrentTime_CPU();
//#endif // TIME_REC
//	GPUFluxRes_subDom = GPUGetFluxResidualFromDevice();
//#ifdef TIME_REC
//	_time_end = TallyCurrentTime_CPU();
//	if (_iter == 0) {
//		TimeStamp9 = TimeCount;
//	}
//	TimeOverall[TimeStamp9] += _time_end - _time_start;
//	if (_iter == 0) {
//		strcpy(TimeMetrics[TimeStamp9], "Comp Flux Res");
//		TimeCount++;
//	}
//#endif // TIME_REC
//}

FP_DOUBLE CompReacRate_TotalDom(int _procID, int _iter, FP_PRECISION_FLUX *_scalarFlux_subDom, int _numLayersZ, \
	int _numCellsInDom, int *_numFSRsInCell, int _numGroup, int *_cellRTMType, FP_PRECISION *_macroXS_r, FP_PRECISION **_inpXSs_r, \
	struct RTMFSRINFO **_RTModFSRsInfo, int *_sumNumFSRsBasedGlobCellID)
{
	FP_DOUBLE _reacRate_subDom;
	FP_DOUBLE _globReacRate = 0.0;

#if (defined(CUDA_USED) && defined(M_2DMOC))
    _reacRate_subDom = GPUCalcReacRate_SubDom(NumGroups, TotalNumFSRs_subDom);
#else
    _reacRate_subDom = CalcReacRate_SubDom(_scalarFlux_subDom, _numLayersZ, _numCellsInDom, _numFSRsInCell, \
		_numGroup, _cellRTMType, _macroXS_r, _inpXSs_r, _RTModFSRsInfo, _sumNumFSRsBasedGlobCellID);
#endif

//                                           printf("\n _reacRate_subDom = %f\n", _reacRate_subDom);
#ifdef MPI_USED
	MPI_Reduce((void *)&_reacRate_subDom, (void *)&_globReacRate, 1, MPI_FP_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif // MPI_USED

#ifndef MPI_USED
	_globReacRate = _reacRate_subDom;
#endif // !MPI_USED

	return _globReacRate;
}


/**
* @ Function: CompTotalFlux_TotalDom()
*   introduction: compute the total flux of whole domian, the total flux is used to normlize the flux of fixed source problem
*
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/14, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified: 
*/
void MOCConvCheck_Flux(int _procID, int _iter, int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim, \
	int _TotalNumFSRs_subDom,int _TotalNumFSRs_perLay, int _NumGroups, FP_PRECISION_FLUX *_ScalarFluxOld, FP_PRECISION_FLUX *_ScalarFlux_MOC)
{
// #if defined(CUDA_USED) && !defined(OMP_USED)
    if (ProblemType == 0) {
	    // _scaFluxRes_subDom = GPUGetFluxResidualFromDevice();
		EigValProbConvCheck_Flux(_procID, NumProcs, MOC_checkFluxNormInf, MOC_fluxNormInf, MOC_checkFluxNorm2, \
            MOC_fluxNorm2, MOC_checkFluxRMSE, MOC_fluxRMSE, MOC_boolFluxNormInf, MOC_boolFluxNorm2, MOC_boolFluxRMSE, \
	        ConvCritFlux);
    }
	else {
		// 	_scaFluxRes_subDom = ComputeRMSE_Flux_lay_g(_TotalNumFSRs_perLay, _NumGroups, _ScalarFlux_MOC, _ScalarFluxOld, \
		// _layBoundLowLim,_layBoundUpLim,_groupBoundLowLim, _groupBoundUpLim);
		FixSouProbConvCheck_Flux(_TotalNumFSRs_perLay, _procID, NumProcs, _NumGroups, _ScalarFlux_MOC, _ScalarFluxOld, \
            _layBoundLowLim, _layBoundUpLim, _groupBoundLowLim, _groupBoundUpLim, MOC_fluxRMSE, MOC_boolFluxRMSE, ConvCritFlux);
	}
// #endif //  defined(CUDA_USED) && !defined(OMP_USED)

// #if !defined(CUDA_USED)
// 	_scaFluxRes_subDom = ComputeRMSE_Flux_lay_g(_TotalNumFSRs_perLay, _NumGroups, _ScalarFlux_MOC, _ScalarFluxOld, \
// 		_layBoundLowLim,_layBoundUpLim,_groupBoundLowLim, _groupBoundUpLim);
// 	// printf("\n _scaFluxRes_subDom=%e",_scaFluxRes_subDom);
// #endif // !defined(CUDA_USED)

// #if defined(CUDA_USED) && defined(OMP_USED)
// 	if (HardwareMap_SubDom[_procID] == 0) {
// 	_scaFluxRes_subDom = ComputeRMSE_Vector(_TotalNumFSRs_subDom * _NumGroups, _ScalarFluxOld, _ScalarFlux_MOC);
// 	}
// 	else if (HardwareMap_SubDom[_procID] == 1) {
// 		_scaFluxRes_subDom = GPUGetFluxResidualFromDevice();
// 	}
// 	else {}
// #endif // defined(CUDA_USED) && defined(OMP_USED)

// #ifdef MPI_USED
// 	MPI_Reduce((void *)&_scaFluxRes_subDom, (void *)&_global_res, 1, MPI_DataType, MPI_MAX, 0, MPI_COMM_WORLD);
// #endif // MPI_USED

// #ifndef MPI_USED
// 	_global_res = _scaFluxRes_subDom;
// #endif // !MPI_USED

// #ifdef TIME_REC
// 	_time_end = TallyCurrentTime_CPU();
// #if defined(CUDA_USED) && defined(OMP_USED) 
// 	TimeOverall[TimeID_TS_FluxRes] += _time_end - _time_start;
// #else
// 	if (_iter == 0) {
// 		TimeStamp9 = TimeCount;
// 	}
// 	TimeOverall[TimeStamp9] += _time_end - _time_start;
// 	if (_iter == 0) {
// 		strcpy(TimeMetrics[TimeStamp9], "FluxRes");
// 		TimeCount++;
// 	}
// #endif
// #endif // TIME_REC

	//MOCFluxResidual = _global_res;
	// return _global_res;
}

int ConvergenceCheck(int _procID, int _eigenValueIter, int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim,int _groupBoundUpLim)
{
	int _bool = 0, _width1 = 11;
	FP_DOUBLE _kerr=(FP_PRECISION)(0.0);
#ifdef TIME_REC
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
	#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
	#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC


//#if !defined(M_MG_CMFD) || (defined(M_MG_CMFD) && defined(MOCthenCMFD))	|| (defined(M_MG_CMFD) && defined(M_UpdateKeff))
	if (ProblemType == 0) {
		MOCGlobFissRateTemp = CompReacRate_TotalDom(ProcID, _eigenValueIter, ScalarFlux_MOC, MOCNumLayersZ, NumPinsInSubDomPerLayer, \
			NumFSRsInCell, NumGroups, CellRTMType_SubDom, MacroXS_nf, InpXSs_nf, RTModFSRsInfo_HS, SumNumFSRsBasedGlobCellID);
	}
//#endif // !defined(M_MG_CMFD) || (defined(M_MG_CMFD) && defined(MOCthenCMFD)) || (defined(M_MG_CMFD) && defined(M_UpdateKeff))

#if defined(CUDA_USED)
    if (ProblemType == 1) {
		GPUScalarFluxUpdate_FixSour1(TotalNumFSRs_perLayer, NumGroups, ScalarFlux_MOC);
	}
#endif // defined(CUDA_USED)

#ifdef TESTR
	{
		ZRES_GI* _tmpGI = zres_get_gen_info();

		if (ProblemType == 1 && _tmpGI->NormRmseConv != NULL && _tmpGI->ifOn) {
			zres_normLocalFlxAndCalRelErr(ScalarFluxOld, ScalarFlux_MOC);
		}
	}
#endif // TESTR

	MOCConvCheck_Flux(ProcID, _eigenValueIter, _layBoundLowLim, _layBoundUpLim, _groupBoundLowLim, _groupBoundUpLim, \
		TotalNumFSRs_subDom,TotalNumFSRs_perLayer, NumGroups, ScalarFluxOld, ScalarFlux_MOC);

#if !defined(M_MG_CMFD) || (defined(M_MG_CMFD) && defined(MOCthenCMFD)) || (defined(M_MG_CMFD) && defined(M_UpdateKeff))
	// if (ProblemType == 0) {
	// 	MOCKeff = CMFDKeff;
	// }
// #else
	if (ProblemType == 0 && _procID == 0) {
		if (KType == 1) {
			//if (_procID == 0) {
				MOCKeff = KeffEvaluation(MOCGlobFissRateTemp, MOCGlobFissRateOld, MOCKeffOld, TotalNumFSRs_subDom);
			//}
		}
		// else if (KType == 0) {
		// 	MOCGlobAbsRateTemp = CompReacRate_TotalDom(ProcID, _eigenValueIter, ScalarFlux_MOC, MOCNumLayersZ, NumPinsInSubDomPerLayer, \
		// 		NumFSRsInCell, NumGroups, CellRTMType_SubDom, MacroXS_a, InpXSs_a, RTModFSRsInfo_HS, SumNumFSRsBasedGlobCellID);

		// 	MOCKeff = KinfEvaluation(MOCGlobFissRateTemp, MOCGlobAbsRateTemp);
		// }
	}

#endif // defined(M_MG_CMFD)


#ifdef TESTR
	{
		ZRES_GI* _tmpGI = zres_get_gen_info();

		if (ProblemType == 1 && _tmpGI->NormRmseConv != NULL && _tmpGI->ifOn) {
			if (zres_get_norm_flux_rmse() < atof(_tmpGI->NormRmseConv)) {
				MOC_boolFluxRMSE[0] = 1.;
			}
		}
	}
#endif // TESTR


#ifdef MPI_USED
	if (ProblemType == 0) {
		MPI_Bcast(&MOCKeff, 1, MPI_DataType, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	// MPI_Bcast(&MOCFluxResidual, 1, MPI_DataType, 0, MPI_COMM_WORLD);
	// MPI_Barrier(MPI_COMM_WORLD);
#endif // MPI_USED
	if (ProblemType == 0) {
		if (_procID == 0) {
			_kerr = fabs((MOCKeff - MOCKeffOld) / MOCKeffOld);
			if (KType == 1) {
				// printf("\n [  INFO   ]  MOC_Iter :%4d    k_eff = %.10lf    k_err = %.1e", _eigenValueIter + 1, MOCKeff, _kerr);
#if defined(M_MG_CMFD) && !defined(MOCthenCMFD)
				// printf("\n              MOC Calculation   k_eff = %.10lf    k_err = %.3e", MOCKeff, _kerr);
				// fprintf(LogFile, "\n              MOC Calculation   k_eff = %.10lf    k_err = %.3e", MOCKeff, _kerr);
				printf("\n              %-*s%-*s%-*s%-*s%-*s%-*s", _width1, "MOC", _width1, "k_eff", _width1, "errEig", \
				    _width1, "Ferrlinf", _width1, "Ferrl2", _width1, "FerrRMSE");
				printf("\n              %-*s", _width1, "");

				fprintf(LogFile,"\n              %-*s%-*s%-*s%-*s%-*s%-*s", _width1, "MOC", _width1, "k_eff", _width1, "errEig", \
				    _width1, "Ferrlinf", _width1, "Ferrl2", _width1, "FerrRMSE");
				fprintf(LogFile,"\n              %-*s", _width1, "");
#else
				// printf("              ");
				// fprintf(LogFile,"              ");
#endif
				printf("%-*.7lf%-*.2e", _width1, MOCKeff,_width1, _kerr);
				if (MOC_checkFluxNormInf) {
					printf("%-*.2e", _width1, MOC_fluxNormInf[0]);
				}
				else {
					printf("%-*s", _width1, "");
				}
				if (MOC_checkFluxNorm2) {
					printf("%-*.2e", _width1, MOC_fluxNorm2[0]);
				}
				else {
					printf("%-*s", _width1, "");
				}
				if (MOC_checkFluxRMSE) {
					printf("%-*.2e", _width1, MOC_fluxRMSE[0]);
				}
				else {
					printf("%-*s", _width1, "");
				}
				fflush(stdout);
				fprintf(LogFile,"%-*.7lf%-*.2e", _width1, MOCKeff,_width1, _kerr);
				if (MOC_checkFluxNormInf) {
					fprintf(LogFile,"%-*.2e", _width1, MOC_fluxNormInf[0]);
				}
				else {
					fprintf(LogFile,"%-*s", _width1, "");
				}
				if (MOC_checkFluxNorm2) {
					fprintf(LogFile,"%-*.2e", _width1, MOC_fluxNorm2[0]);
				}
				else {
					fprintf(LogFile,"%-*s", _width1, "");
				}
				if (MOC_checkFluxRMSE) {
					fprintf(LogFile,"%-*.2e", _width1, MOC_fluxRMSE[0]);
				}
				else {
					fprintf(LogFile,"%-*s", _width1, "");
				}
				fflush(stdout);
			}

		}
		if ((fabs((MOCKeff - MOCKeffOld) / MOCKeffOld) < ConvCritK && \
		    ((MOC_checkFluxNormInf && MOC_boolFluxNormInf[0]) || (MOC_checkFluxNormInf==0)) && \
			((MOC_checkFluxNorm2 && MOC_boolFluxNorm2[0]) || (MOC_checkFluxNorm2==0)) && \
			((MOC_checkFluxRMSE && MOC_boolFluxRMSE[0]) || (MOC_checkFluxRMSE==0))) || \
			_eigenValueIter >= MaxNumOuterIter_MOC) {
			if (_procID == 0) {
				if (_eigenValueIter >= MaxNumOuterIter_MOC) {
					printf("\n [**ERROR**]  The number of outer iteration has reached the maximum(%d) without convergence", MaxNumOuterIter_MOC);
					fprintf(LogFile, "\n [**ERROR**]  The number of outer iteration has reached the maximum(%d) without convergence", MaxNumOuterIter_MOC);
					fflush(stdout);
					exit(69);
				}
				else {
					printf("\n [  RESULT ]  The Abslute Error of Eigenvalue is %d pcm", ((int)((MOCKeff - KRef)*1.e5)));
					fprintf(LogFile, "\n [  RESULT ]  The Abslute Error of Eigenvalue is %d pcm", ((int)((MOCKeff - KRef)*1.e5)));
					fflush(stdout);
				}
			}
			_bool = 1;
		}
	}
	else if (ProblemType == 1) { /* fixed source problem*/
		if (_procID == 0 && IfResonance == 0) {
			 printf("FixedSourceIter    fluxRes = %.1e", MOC_fluxRMSE[0]);
			 fflush(stdout);
			 fprintf(LogFile, "FixedSourceIter    fluxRes = %.1e", MOC_fluxRMSE[0]);
		}
		if (MOC_boolFluxRMSE[0] || _eigenValueIter >= MaxNumOuterIter_MOC) {
			_bool = 1;
//			if (OUTPUT_flux == 1) {
//				MOCGlobFlux_FixSour = CompTotalFlux_TotalDom(_procID/*, _iter*/);
//			}
		}
	}
	else {}

	if (ProblemType == 0) {
	MOCGlobFissRateOld = MOCGlobFissRateTemp;
	MOCKeffOld = MOCKeff;
	}
//	int i;
//	FP_PRECISION _relErr = 0.0;
//	for (i = 0; i < TotalNumFSRs_subDom*NumGroups; i++) {
//		_relErr = fabs((ScalarFlux_MOC[i] - ScalarFluxOld[i]) / ScalarFlux_MOC[i]);
//		printf("\n flux=%e, fluxOld=%e, _relErr=%e", ScalarFlux_MOC[i], ScalarFluxOld[i], _relErr);
//	}

// #if !defined(CUDA_USED) || (defined(CUDA_USED) && defined(OMP_USED))
// 	VectorCopy_FP_Flux(ScalarFluxOld, ScalarFlux_MOC, TotalNumFSRs_subDom*NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1));
// 	//printf("\n");
// 	//for (i = 0; i < TotalNumFSRs_subDom*NumGroups; i++) {
// 	//	printf("\n flux=%e, fluxOld=%e", ScalarFlux_MOC[i], ScalarFluxOld[i]);
// 	//}
// #endif // !defined(CUDA_USED) || (defined(CUDA_USED) && defined(OMP_USED))

#if defined(CUDA_USED)
	if (ProblemType == 0) {
        // GPUScalarFluxUpdate_TS(TotalNumFSRs_subDom, NumGroups);
	}

    else if (ProblemType == 1) {
		GPUScalarFluxUpdate_FixSour2(TotalNumFSRs_perLayer, NumGroups, ScalarFluxOld);
	}
#endif // defined(CUDA_USED)

#ifdef TIME_REC
	#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
	#endif // defined(CUDA_USED)
	_time_end = TallyCurrentTime_CPU();
	if (_eigenValueIter == 0) {
		TimeStamp_ConvCheck = TimeCount;
		strcpy(TimeMetrics[TimeStamp_ConvCheck], "MOC Conv Check");
		TimeCount++;
	}
	TimeOverall[TimeStamp_ConvCheck] += _time_end - _time_start;
#endif // TIME_REC
	return _bool;
}

void MOC_InitTransSweep(int _procID)
{
	FP_PRECISION _val1 = (FP_PRECISION)(0.0), _zero_fp = (FP_PRECISION)(M_Zero_FP);
	int _dim1;
#ifdef TIME_REC
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC

#if defined(M_3DMOCex)
	int _dim;
#endif // defined(M_3DMOCex)

	// #if !defined(M_MG_CMFD)
	// PrintCPUMemUsage(ProcID);
	// #endif // !defined(M_MG_CMFD)
// #if defined(CUDA_USED)
// 	// GPUInitScalarFlux_Dev(TotalNumFSRs_subDom*NumGroups, (FP_PRECISION_FLUX)(M_One_FP));
// 	// ScalarFluxCopy_Dev2Host(ScalarFlux_MOC, (size_t)(TotalNumFSRs_subDom*NumGroups), (char *)("ScalarFluxCopy_Dev2Host-ScalarFlux_MOC"));
// 	// ScalarFluxCopy_Dev2Host(ScalarFluxOld, (size_t)(TotalNumFSRs_subDom*NumGroups), (char *)("ScalarFluxCopy_Dev2Host-ScalarFluxOld"));

// 	_dim1 = TotalNumFSRs_perLayer * (NumScattOrder + 1) * (NumScattOrder + 1) * NumGroups;
// 	GPUInitScalarFlux_Dev(_dim1, (FP_PRECISION_FLUX)(M_One_FP));
// 	// ScalarFluxCopy_Dev2Host(ScalarFlux_MOC, (size_t)(_dim1), 0, (char *)("ScalarFluxCopy_Dev2Host-ScalarFlux_MOC"));
// 	// ScalarFluxCopy_Dev2Host(ScalarFluxOld, (size_t)(_dim1), 0, (char *)("ScalarFluxCopy_Dev2Host-ScalarFluxOld"));
// 	GPUScalarFluxUpdate_TS(TotalNumFSRs_subDom, NumGroups);
// #else
    if (isTransient < 2) {
#ifdef CUDA_USED
        warmup();
#endif
        VectorInit_FP_Flux(ScalarFlux_MOC, TotalNumFSRs_subDom * NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1),
                           (FP_PRECISION_FLUX) (M_One_FP));
        VectorInit_FP_Flux(ScalarFluxOld, TotalNumFSRs_subDom * NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1),
                           (FP_PRECISION_FLUX) (M_One_FP));

#if defined(M_MG_CMFD)
        VectorInit_FP_Flux(ScalarFluxOld_0thOrder, TotalNumFSRs_subDom * NumGroups, (FP_PRECISION_FLUX) (M_One_FP));
#endif
    }
	// VectorInit_FP_Flux(ScalarFlux_MOC, TotalNumFSRs_subDom * NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1), (FP_PRECISION_FLUX)(M_One_FP));
// #endif 

#if defined(M_Use4PI)
	_val1 = (FP_PRECISION)(ONE_OVER_FOUR_PI);
#else
	_val1 = (FP_PRECISION)(M_One_FP);
#endif // defined(M_Use4PI)

#if defined(CUDA_USED) && defined(GPU_STREAM_USED)
	GPUBoundFluxPre_StreamUsed(NumGroups * MOCNumLayersZ*NumLTs_HS[0] * 2 * NumPolarAng[0], CGBoundaryFlux, _val1);
#else
    if (isTransient < 2) {
        InitInciAngFlux(BoundaryFlux, NumTrackSet, NumLTs_HS, NumGroups, NumPolarAng, MOCNumLayersZ, _val1);
    }

    if ((ProblemType == 0) || ((ProblemType == 1) && (SouthBC == VACUUM || WestBC == VACUUM || EastBC == VACUUM || NorthBC == VACUUM))) {
        InitInciAngFlux_BC(BoundaryFlux, NumTrackSet, NumGroups, NumAzimAng, \
            NumPolarAng, MOCNumLayersZ, NumLongTracks_SubDomX, NumLongTracks_SubDomY);
    }
#endif

//#if !defined(M_UpdateRadialBoundImme)
//	InitInciAngFlux(BoundaryFlux_Temp, NumTrackSet, NumLTs_HS, NumGroups, NumPolarAng, MOCNumLayersZ, _val1);
//#endif // !defined(M_UpdateRadialBoundImme)
	
	
#if defined(M_3DMOCex)
	_dim = TotalNumFSRs_perLayer * NumGroups*NumAzimAng[0] * (NumPolarAng[0] / 2);
#if defined(CUDA_USED)
	GPU_InitAxiAngFlux(ProblemType, _dim, BottomBC, TopBC, _val1, _zero_fp);
#else
	CPU_InitAxiAngFlux(ProblemType, _dim, BottomBC, TopBC, FSRAveAngFlux_Bott, \
		FSRAveAngFlux_Top, _val1, _zero_fp);
#endif
#endif // defined(M_3DMOCex)

	if (ProblemType == 0) {
		MOCKeff = 1.0;
        if (isTransient == 2){
            MOCKeff = MOCKeffOld;
            VectorCopy_FP_Flux(ScalarFlux_MOC, PrevScalarFlux,
                               TotalNumFSRs_subDom*NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1));
        }
		MOCGlobFissRateTemp = MOC_CalcGlobalReacRate(MOC_fissRate_procs, _procID, NumProcs, \
			ScalarFlux_MOC, MOCNumLayersZ, NumPinsInSubDomPerLayer, NumFSRsInCell, \
			NumGroups, CellRTMType_SubDom, MacroXS_nf, InpXSs_nf, RTModFSRsInfo_HS, SumNumFSRsBasedGlobCellID);
        MOCGlobFissRateOld = MOCGlobFissRateTemp;
		MOCKeffOld = MOCKeff;
	}
#if !defined(CUDA_USED)
	else if (ProblemType == 1) {
		VectorCopy_FP_Flux(Source, FixedSour, TotalNumFSRs_subDom*NumGroups);
	}
#endif // !defined(CUDA_USED)

#if !defined(M_MG_CMFD) || (defined(M_MG_CMFD) && defined(MOCthenCMFD))
	if (_procID == 0) {
		if (ProblemType == 0) {
			printf("\n [  TITLE  ]  ************************************************************");
            if(!isAdjoint)
			    printf("\n [  TITLE  ]  *         Transport Sweep - Eigenvalue Problem             *");
            else
                printf("\n [  TITLE  ]  *      Transport Sweep - Eigenvalue Problem - Adjoint      *");

			if (MOCSolverType == 0) {
				printf("\n [  TITLE  ]  *             Gauss Seidel Iteration Scheme                *");
			}
			else {
				printf("\n [  TITLE  ]  *                Jacobi Iteration Scheme                   *");
			}
			printf("\n [  TITLE  ]  ************************************************************");
			fflush(stdout);
			fprintf(LogFile, "\n [  TITLE  ]  *****************************************************************");
			fprintf(LogFile, "\n [  TITLE  ]  *            Transport Sweep - Eigenvalue Problem               *");
			if (MOCSolverType == 0) {
				fprintf(LogFile, "\n [  TITLE  ]  *             Gauss Seidel Iteration Scheme                *");
			}
			else {
				fprintf(LogFile, "\n [  TITLE  ]  *                   Jacobi Iteration Scheme                     *");
			}
			fprintf(LogFile, "\n [  TITLE  ]  *****************************************************************");
			fflush(stdout);
		}
		else if (ProblemType == 1 && IfResonance == 0) {
			printf("\n [  TITLE  ]  ************************************************************");
			printf("\n [  TITLE  ]  *        Transport Sweep - Fixed Source Problem            *");
			printf("\n [  TITLE  ]  ************************************************************");
			fflush(stdout);
			fprintf(LogFile, "\n [  TITLE  ]  ************************************************************");
			fprintf(LogFile, "\n [  TITLE  ]  *        Transport Sweep - Fixed Source Problem            *");
			fprintf(LogFile, "\n [  TITLE  ]  ************************************************************");
		}
	}
#endif //!defined(M_MG_CMFD)

	//VectorCopy_FP_Flux(ScalarFluxOld, ScalarFlux_MOC, TotalNumFSRs_subDom*NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1));
	//VectorInit_FP_Flux(ScalarFlux_MOC, TotalNumFSRs_subDom * NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1), (FP_PRECISION_FLUX)(0.0));

    memcpy(ScalarFluxOld, ScalarFlux_MOC, (size_t) (sizeof (FP_PRECISION_FLUX) * TotalNumFSRs_subDom*NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1)));
    memset(ScalarFlux_MOC, 0., (size_t) (sizeof (FP_PRECISION_FLUX) * TotalNumFSRs_subDom * NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1)));

#ifdef TIME_REC
	_time_end = TallyCurrentTime_CPU();
	TimePrePart[TimeCountPrePart] = _time_end - _time_start;
	strcpy(TimeMetricsPrePart[TimeCountPrePart], "Initialize TS on CPU");
	TimeCountPrePart++;
#endif // TIME_REC
}
