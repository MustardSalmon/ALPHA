#ifndef GPUCPUINTERFACE_H
#define GPUCPUINTERFACE_H
#include "MOC_Track.h"
//extern FP_PRECISION *GC_TotalWeights;
//extern FP_PRECISION *GC_FSRsAreaInRTM;
//extern int *GC_SumNumFSRsInRTM;
//extern int GC_TotNumFSRsInRTM;
extern int *GC_SumNumLTs;

extern FP_PRECISION *CPUReacRateCollect;
extern FP_PRECISION GPUTotReacRate_subDom;
extern FP_PRECISION GPUGlobFissTemp;
extern FP_PRECISION GPUGlobFissOld,GPUKeff,GPUKeffOld;
extern FP_PRECISION *GPUFluxResRColl;
extern FP_PRECISION GPUFluxRes_subDom;
extern FP_PRECISION GPUFluxRes_Glob;

extern FP_PRECISION *GC_SinThetasTimes2;
extern FP_PRECISION *GC_OneOverSinThetas;

extern FP_PRECISION_TIME CudaMemUse[1];
/* 2D transport sweep mode: 0 = all azimuths in one sweep, positive values
 * = that many azimuths per batched sweep.  The mode is deliberately
 * independent of FSR XS clustering. */
extern int GPUAzimSweepMode;


//void GC_getTotWeights(FP_PRECISION ***_totalWeights);
void GC_getSinThetasTimes2(FP_PRECISION **_sinTheta);
//void GC_getFSRsAreaInRTM(struct RTMFSRINFO **_RTMFSRsInfo);
void GC_getSumNumLTs_Azim(int **_numLongTracks,int *_numAzimAng);
FP_PRECISION CalcGlobalFissReacFromGPUs(FP_PRECISION *_reacRateCollec,FP_PRECISION _totReacRate_GPUtoCPU, \
	int _procID, int _numProcs);

#ifdef CUDA_USED
void GPUComputeKeff(int _procID,int _numProcs, int _iter);
void GPUComputeFluxRes(int _procID,int _numProcs,int _iter);
void InitCudaDevice_1(int _CPUProcID, int _numProcs);
void CopyInitInfoFromCPUtoGPU(int _procID);
void GPUInitQuadratureSet_1(int _procID);
void GPUCopyTrackInfoHostToDevice_1(int _procID);
void SetKernelCallingPara_1();
#ifdef GPU_STREAM_USED
void GPUStreamInitialize_1();
#endif //GPU_STREAM_USED
void GPUTransportSweepInit_1(int _procID, int _numProcs, int _totalNumFSRs, int _numGroups, int _totNumLongTracks, int _numPolarAng);
void GPU_MainAllocMem_1(int _procID, int _numProcs, M_INT32 _totalNumFSRs, int _numGroups, int _totNumLongTracks, int _numPolarAng);
void GPUInitAzimSweepMode(void);
void GPUCopyBoundAngFluxHostToDevice_1(int _iter, int _firstInvoke);
void GPUTransportSweep_1(int _iter);
void GPUCopyBoundAngFluxDeviceToHost_Synch_1(int _iter);
//void GPUGetFluxResidualFromDevice_1(int _iter);
int GPUConvergenceCheck_1(int _eigenValueIter, int _procID, int _numProcs, int _iter);
#endif // CUDA_USED

FP_DOUBLE CompReacRate_TotalDom(int _procID,int _iter, FP_PRECISION_FLUX *_scalarFlux_subDom, int _numLayersZ, \
	int _numCellsInDom, int *_numFSRsInCell, int _numGroup, int *_cellRTMType, FP_PRECISION *_macroXS_r, FP_PRECISION **_inpXSs_r, \
	struct RTMFSRINFO **_RTModFSRsInfo, int *_sumNumFSRsBasedGlobCellID);
// void CompReacRate_TotalDom_Init(int _procID);
void MOCConvCheck_Flux(int _procID, int _iter, int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim, \
	int _TotalNumFSRs_subDom,int _TotalNumFSRs_perLay, int _NumGroups, FP_PRECISION_FLUX *_ScalarFluxOld, FP_PRECISION_FLUX *_ScalarFlux_MOC);
int ConvergenceCheck(int _procID, int _eigenValueIter, int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim,int _groupBoundUpLim);
FP_DOUBLE CompTotalFlux_TotalDom(int _procID/*, int _iter*/);

void GPUMOC_TransSweep_UporDown(int _dim, int _layStart, int _layAdd, int _surfLayInc, \
	int _polarBoundLowLim, int _polarBoundUpLim, int _polOffset, FP_PRECISION *_axiFlux_inp, \
	FP_PRECISION *_axiFlux_avg, int _nodeLay, int _surfLay, FP_PRECISION _JzDir, int _dir,\
	int _iter, int _fisrtInvoke, int _innerIter, int _azimBegin, int _azimEnd);
 
void GPUCMFD_GCEvaluation(int _isFirstInvoke, int _outIter);
void GPUAllocMemCMFDXSs_onCPU(int _isFirstInvoke);
void GPUCMFD_Initialization();
FP_PRECISION GPU_MGCMFDSolver(int _iter_moc, FP_PRECISION _k);
FP_PRECISION GPU_CGCMFDSolver(int _iter_moc, int _iter_mg, FP_PRECISION _k);
void GPUCGCMFD_GCEvaluation(int _isFirstInvoke);
void GPU_MOCFixSourProbSolver_lay_g(int _procID, int _iter, int _layBoundLowLim, int _layBoundUpLim, \
    int _groupBoundLowLim, int _groupBoundUpLim);
void GPU_MOCFixSourProbSolver_lay_g_sld(int _procID, int _iter, int _layBoundLowLim, int _layBoundUpLim, \
    int _groupBoundLowLim, int _groupBoundUpLim);
void CMFDConvCheck_Flux(int _procID, int _numProcs, int _isNormInf, FP_PRECISION *_normInf, int _isNorm2, \
    FP_PRECISION *_norm2, int _isRMSE, FP_PRECISION *_rmse, int *_bFNormInf,int *_bFNorm2, int *_bFRmse, \
	FP_PRECISION _tol, int *_CMFD_Diver,int _NumCMFDGroup, int _cmfd);
void EigValProbConvCheck_Flux(int _procID, int _numProcs, int _isNormInf, FP_PRECISION *_normInf, int _isNorm2, \
    FP_PRECISION *_norm2, int _isRMSE, FP_PRECISION *_rmse, int *_bFNormInf,int *_bFNorm2, int *_bFRmse, \
	FP_PRECISION _tol);
void FixSouProbConvCheck_Flux(int _numFSRs_lay, int _procID, int _numProcs, int _numGroups, FP_PRECISION *_vecNew, FP_PRECISION *_vecOld, \
    int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim, FP_PRECISION *_rmse, \
	int *_bFRmse, FP_PRECISION _tol);
void MOC_InitTransSweep(int _procID);
#endif
