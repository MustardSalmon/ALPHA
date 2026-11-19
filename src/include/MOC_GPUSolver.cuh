#ifndef GPUSOLVER_H
#define GPUSOLVER_H

#include "MOC_AllocAndFreeMem.h"
#include "MOC_CommonMacro.h"
#include "MOC_ExpEvaluation.h"
#include "MOC_FluxEvaluation.h"
#include "MOC_Geometry.h"
#include "MOC_GPUCPUInterface.h"
#include "MOC_Input.h"
#include "MOC_Macro_XSs.h"
#include "MOC_main.h"
#include "MOC_NTE_Solver.h"
#include "MOC_Output.h"
#include "MOC_Quadrature_Set.h"
#include "MOC_Track.h"
#include "MOC_VectorOperation.h"
#include "MOC_GPU_TSSolver.cuh"
#include "cublas_v2.h"

#define GPUMaxNumPolarAngles  3
#define GPUMaxPolAngTimesGroup  GPUMaxNumPolarAngles*7
#define GPUMaxNumSegsInLT  1536//1536  768
#define GPUMaxExpTempStore_group GPUMaxNumSegsInLT*3
#define GPUMaxExpTempStore_track GPUMaxNumSegsInLT*3*7
// #define GPUMaxExpTempStore_track_1 GPUMaxNumSegsInLT*3*7

/** Indexing macro for the scalar flux or 1D XS in each FSR and energy group */
#define _GPUIndFSRGroup_(tid,g) (GPUIndFSRGroup[(tid)*(*num_groups) + (g)])
/** Indexing macro for the angular fluxes for each polar angle and energy
 *  group for a given Track */
// #define _GPUBoundaryFlux(lt,pg) (_GPUBoundaryFlux[(lt)*(*GPUPolarTimesGroups)+(pg)])
// #define _GPUStartFlux(lt,pg) (_GPUStartFlux[(lt)*(*GPUPolarTimesGroups)+(pg)])
// #define _GPUMPIFlux(lt,pg) (_GPUMPIFlux[(lt)*(*GPUPolarTimesGroups)+(pg)])

// #define GPUNumThreadBlocks 256 /** The number of thread blocks */
// #define GPUNumThreadPerBlock 64 /** The number of threads per thread block */

// extern FP_PRECISION GPUGlobFissTemp,GPUGlobFissOld,GPUKeff;
extern FP_PRECISION *ExpTable;
extern FP_PRECISION *GPUMPIFlux,*GPUBoundaryFlux;
extern int GPUNumThreadBlocks; /** The number of thread blocks */
extern int GPUNumThreadPerBlock;/** The number of threads per thread block */
// extern FP_PRECISION *HostScalarFlux_output;
// extern FP_PRECISION *ReacRateCollect;
// extern FP_PRECISION GPUGlobFissTemp;
// extern FP_PRECISION GPUKeff;
// extern FP_PRECISION GPUGlobFissOld;
// extern FP_PRECISION GPUKeffOld;
// extern FP_PRECISION *CPUReacRateCollect;
extern float TSKernel_Time;

#ifdef GPU_STREAM_USED
  extern FP_PRECISION *CGBoundaryFlux, *CGLTInciAngFlux;
#endif // GPU_STREAM_USED

extern FP_PRECISION *GPUFSRAveAngFlux_Top;
extern FP_PRECISION *GPUFSRAveAngFlux_Bott;
extern FP_PRECISION_FLUX *GPUScalarFluxAzimAccum;
extern int *GPUNumFSRs_Layer,*GPUSumNumFSRs_Layer;

#ifdef TESTR
#include "condition.h"
#ifdef SAVE_ANGLE_FLUX
extern FP_PRECISION *GPUSldSpheHarmCoeff;
#endif
#endif

//extern FP_PRECISION *GPUScalarFlux, *GPUMOCSource, *GPUMacroXS_t;
//extern int GPUGridDim_source_para_Layer;


void GPUInitMacroXSs(int _procID, int _TotalNumFSRs_subDom, int _NumGroups, int _MOCNumLayersZ, \
	M_UINT64 *_MacroXS_s_NZRowStartID_Layer, M_UINT64 *_MacroXS_s_NZColStartID, size_t *_GPUPreMemXSs,\
	M_INT32 _IfResonance, M_INT32 _ProblemType, M_UINT64 _MacroXS_s_MaxNNZ_Layer, FILE *_LogFile);
void GPUCopyMacroXSsHostToDevice_Layer(int _layID, int _TotalNumFSRs_subDom, int _TotalNumFSRs_perLayer, int _NumGroups, \
	M_UINT64 *_MacroXS_s_NZStartID_Layer, M_UINT64 *_MacroXS_s_NNZ_Layer, M_INT32 _ProblemType, \
	FP_PRECISION *_MacroXS_t, FP_PRECISION *_MacroXS_a, FP_PRECISION *_MacroXS_f, FP_PRECISION *_MacroXS_nf, \
	FP_PRECISION *_Chi, FP_PRECISION *_MacroXS_s, FP_PRECISION *_FixedSour, M_INT32 _NumScattOrder,\
	M_UINT64 *_MacroXS_s_NZRowStartID_Layer, M_UINT64 *_MacroXS_s_NZColStartID, int _isCMFD, FP_PRECISION_FLUX *_ScalarFluxOld);
void GPUCopyTrackInfoHostToDevice(int _procID, M_INT32 _NumGroups, M_INT32 _TotalNumFSRs_subDom, M_INT32 _TotalNumFSRs_perLayer, \
	M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumPinsInSubDomPerLayer, M_INT32 _TotalNumFSRsInRTM, M_INT32 _NumPolarAng, \
	M_INT32 _NumLTs_HS, M_INT32 _NumLTs_2PI, M_INT32 *_LTsIndex_AzimAngID_HS, M_INT32 *_LTsIndex_LTID_HS, \
	M_INT32 *_LTsIndex_AzimAngID_2PI, M_INT32 *_LTsIndex_LTID_2PI, struct SUBDOM_TRACK_IN_LONGTRACK *_TracksInfoInLTs_2PI, \
	M_INT32 *_TrackIndexInLTs_2PI, M_INT32 _TotNumTracksInOneRTM_2PI, M_INT32 *_IndexTrackInRTM_SumAzim_2PI, \
	M_INT32 _TotalNumTracksInRTM_Total, M_INT32 *_NumSegsInTrack_2PI, M_INT32 _TotNumSegInRTM_Total, \
	struct SUBDOM_SEG_IN_TRACK_SINGLE *_SegInfoInTrack_SINGLE_2PI, M_INT32 *_SumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_SegInfoInTrack_2PI_length, M_INT32 *_SegInfoInTrack_2PI_RayWisedFSR, M_INT32 *_SegInfoInTrack_2PI_AntiClockFSR, \
	M_INT32 *_FsrIdMap_RayWisedToAntiClock, struct SUBDOM_TRACK_IN_LONGTRACK *_TracksInfoInLTs_HS, M_INT32 _NumTracks_HS, \
	struct SUBDOM_LONGTRACK_INFO *_LongTracksInfo_Total, M_INT32 *_TrackIndexInLTs_HS, M_INT32 _TotalNumTracksInRTM_HS, \
	M_INT32 *_NumSegsInTrack_HS, M_INT32 _TotNumTracksInOneRTM_HS, M_INT32 _NumAzimAng, M_INT32 *_IndexTrackInRTM_SumAzim_HS, \
	M_INT32 _TotNumSegInRTM_HS, struct SUBDOM_SEG_IN_TRACK_SINGLE *_SegInfoInTrack_SINGLE_HS, M_INT32 _MOCNumLayersZ, \
	M_INT32 *_SumNumFSRsBasedGlobCellID, M_INT32 _NumRTMGeoType, M_INT32 *_SumNumFSRsInRTM, M_INT32 *_CellRTMType_SubDom, \
	M_INT32 *_SumNumSegsInRTMTrack_HS, M_INT32 *_NumLTsSum, M_INT32 *_NumLongTracks, \
	FP_PRECISION_GEO *_OneOverFSRArea_SubDom, FP_PRECISION_GEO *_FSRVol_SubDom, M_INT32 _NorthBC, M_INT32 _SouthBC, M_INT32 _EastBC, \
	M_INT32 _WestBC, M_INT32 _BottomBC, M_INT32 _TopBC, M_INT32 *_NumFSRsInCell, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	FP_PRECISION_GEO *_EffRaySpacing, M_INT32 *_NumFSRs_Layer, M_INT32 *_SumNumFSRs_Layer, FP_PRECISION_GEO *_WidthLayersInCore, \
	M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, FP_PRECISION_GEO *_RayTracModDim, M_INT32 _NumSurfaceX, M_INT32 _NumSurfaceY, \
	M_INT32 *_NumTracksX, M_INT32 *_NumTracksY, FP_PRECISION_GEO *_RTMWidth_X, FP_PRECISION_GEO *_RTMWidth_Y, M_INT32 *_NumFSRsInRTM, \
	FP_PRECISION_GEO *_FSRsAreaInRTM_AntiClockFSR, FP_PRECISION_GEO *_FSRArea_SubDom, size_t *_GPUPreMemTrack, size_t *_GPUPreMemXSs, \
	M_INT32 _NumTracks_Total, FILE *_LogFile);

void GPUCopyTrackInfoHostToDevice_trackFile(int _procID, M_INT32 _NumGroups, M_INT32 _TotalNumFSRs_subDom, M_INT32 _TotalNumFSRs_perLayer, \
	M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumPinsInSubDomPerLayer, M_INT32 _TotalNumFSRsInRTM, M_INT32 _NumPolarAng, \
	M_INT32 _NumLTs_HS, M_INT32 _NumLTs_2PI, M_INT32 *_LTsIndex_AzimAngID_HS, M_INT32 *_LTsIndex_LTID_HS, \
	M_INT32 *_LTsIndex_AzimAngID_2PI, M_INT32 *_LTsIndex_LTID_2PI, struct SUBDOM_TRACK_IN_LONGTRACK *_TracksInfoInLTs_2PI, \
	M_INT32 *_TrackIndexInLTs_2PI, M_INT32 _TotNumTracksInOneRTM_2PI, M_INT32 *_IndexTrackInRTM_SumAzim_2PI, \
	M_INT32 _TotalNumTracksInRTM_Total, M_INT32 *_NumSegsInTrack_2PI, M_INT32 _TotNumSegInRTM_Total, \
	/*struct SUBDOM_SEG_IN_TRACK_SINGLE *_SegInfoInTrack_SINGLE_2PI,*/ M_INT32 *_SumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_SegInfoInTrack_2PI_length, /*M_INT32 *_SegInfoInTrack_2PI_RayWisedFSR,*/ M_INT32 *_SegInfoInTrack_2PI_AntiClockFSR, \
	/*M_INT32 *_FsrIdMap_RayWisedToAntiClock,*/ struct SUBDOM_TRACK_IN_LONGTRACK *_TracksInfoInLTs_HS, M_INT32 _NumTracks_HS, \
	struct SUBDOM_LONGTRACK_INFO *_LongTracksInfo_Total, /*M_INT32 *_TrackIndexInLTs_HS, M_INT32 _TotalNumTracksInRTM_HS, \
	M_INT32 *_NumSegsInTrack_HS, M_INT32 _TotNumTracksInOneRTM_HS,*/ M_INT32 _NumAzimAng, /*M_INT32 *_IndexTrackInRTM_SumAzim_HS, \
	M_INT32 _TotNumSegInRTM_HS, struct SUBDOM_SEG_IN_TRACK_SINGLE *_SegInfoInTrack_SINGLE_HS,*/ M_INT32 _MOCNumLayersZ, \
	M_INT32 *_SumNumFSRsBasedGlobCellID, M_INT32 _NumRTMGeoType, M_INT32 *_SumNumFSRsInRTM, M_INT32 *_CellRTMType_SubDom, \
	/*M_INT32 *_SumNumSegsInRTMTrack_HS,*/ M_INT32 *_NumLTsSum, M_INT32 *_NumLongTracks, \
	/*FP_PRECISION_GEO *_OneOverFSRArea_SubDom,*/ FP_PRECISION_GEO *_FSRVol_SubDom, M_INT32 _NorthBC, M_INT32 _SouthBC, M_INT32 _EastBC, \
	M_INT32 _WestBC, M_INT32 _BottomBC, M_INT32 _TopBC, M_INT32 *_NumFSRsInCell, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	FP_PRECISION_GEO *_EffRaySpacing, M_INT32 *_NumFSRs_Layer, M_INT32 *_SumNumFSRs_Layer,/**/ FP_PRECISION_GEO *_WidthLayersInCore, \
	M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, /*FP_PRECISION_GEO *_RayTracModDim,*/ M_INT32 _NumSurfaceX, M_INT32 _NumSurfaceY, \
	M_INT32 *_NumTracksX, M_INT32 *_NumTracksY, FP_PRECISION_GEO *_RTMWidth_X, FP_PRECISION_GEO *_RTMWidth_Y, M_INT32 *_NumFSRsInRTM, \
	FP_PRECISION_GEO *_FSRsAreaInRTM_AntiClockFSR, FP_PRECISION_GEO *_FSRArea_SubDom, size_t *_GPUPreMemTrack, size_t *_GPUPreMemXSs, \
	M_INT32 _NumTracks_Total, FILE *_LogFile);

void GPUInitQuadratureSet(int _procID, size_t *_GPUPreMemQuadrSet, size_t *_GPUPreMemExpTab, FILE *_LogFile, M_INT32 _NumAzimAng, \
	M_INT32 _NumPolarAng, FP_PRECISION *_SinTheta, FP_PRECISION *_CosTheta, FP_PRECISION *_GC_SinThetasTimes2, \
	FP_PRECISION *_GC_OneOverSinThetas, FP_PRECISION *_SinThetaForExp, FP_PRECISION *_AzimWeights, FP_PRECISION *_PolarWeights, \
	FP_PRECISION *_TotalWeights, FP_PRECISION *_TotalWeights_2DMOC, FP_PRECISION *_ExpTable, int _NumGroups, M_INT32 _NumScattOrder, \
	FP_PRECISION *_CoeffA_HOScatt, FP_PRECISION *_SpheHarmCoeff);

void GPU_MainAllocMem(int _procID, int _numProcs, M_INT32 _totalNumFSRs, int _numGroups, \
	int _totNumLongTracks, int _numPolarAng, size_t *_GPUPreMemScalF, size_t *_GPUPreMemRadF, \
	size_t *_GPUPreMemAxiF, size_t *_GPUPreMemMGCMFD, size_t *_GPUPreMemXSs, M_INT32 _TotalNumFSRs_perLayer, \
	M_INT32 _IfResonance, M_INT32 _ProblemType, M_UINT64 _MacroXS_s_MaxNNZ_Layer, M_INT32 _MOCNumLayersZ, \
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
	M_INT32 _OUTPUT_CoarseMesh_GroupConstant, M_INT32 _NumScattOrder, M_INT32 _GPU_TSKernelType, int _isTransient);
void GPUInitScalarFlux(M_INT32 _dim, FP_PRECISION _value);
void GPUTransportSweep(int _lay, FP_PRECISION _twoOverLayWidth, int _polarBoundLowLim, int _polarBoundUpLim,\
  FP_PRECISION *_GPUFSRAveAngFlux_inp,FP_PRECISION *_GPUFSRAveAngFlux_update, int _polOffset, \
	M_INT32 _GPU_TSKernelType, M_INT32 _NumPinsInSubDomPerLayer, M_INT32 _NumPolarAng, \
	int _azimBegin, int _azimEnd);
void GPUAddSourceToScalarFlux();
void GPUAddSourceToScalarFlux_lay_g(int _numFSRsPerLay, int _layBoundLowLim, int _layBoundUpLim, \
	int _groupBoundLowLim, int _groupBoundUpLim);

FP_PRECISION *GPUAllocMem_FP_I(size_t _dim1, char *_parameter);
M_INT32 *GPUAllocMem_INT32_I(size_t _dim1, char *_parameter);
M_UINT64 *GPUAllocMem_UINT64_I(size_t _dim1, char *_parameter);
FP_PRECISION GPUGetFluxResidualFromDevice();

void GPUScalarFluxUpdate_TS(int _TotalNumFSRs_subDom, int _NumGroups);
int GPUConvergenceCheck(int _eigenValueIter,int _procID,int _numProcs);
// FP_PRECISION CalcGlobalFissReacFromGPUs(FP_PRECISION *_reacRateCollec,FP_PRECISION *_GPUReacRateCollect, \
// 	int _procID, int _numProcs);
// FP_PRECISION CalcGlobalFluxResFromGPUs(FP_PRECISION *_GPUFluxResRColl,int _procID, int _numProcs);
void GPUCopyBoundAngFluxDeviceToHost_Synch(FP_PRECISION *_CPUMPIFlux, size_t _dim);
void SetKernelCallingPara(M_INT32 _NumLTs_HS, M_INT32 _NumGroups, M_INT32 _NumPolarAng, \
	M_INT32 _TotalNumFSRs_subDom, M_INT32 _TotalNumFSRs_perLayer, M_INT32 _MOCNumLayersZ, \
	M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, M_INT32 _NumCMFDGroup, \
	M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumPinsInSubDomPerLayer, M_INT32 _NumCoarseCMFDGroup);
//void GPUTransportSweep_1();
// void GPUTransportSweep_2();
// void GPUParameterUpdate_TS_1_1();
void GPUCopyBoundAngFluxHostToDevice(M_INT32 _MOCNumLayersZ, M_INT32 _NumLTs_HS, M_INT32 _NumPolarAng, M_INT32 _NumGroups, \
	FP_PRECISION *_BoundaryFlux, M_INT32 _NumAzimAng, M_INT32 **_GCCopy_NumDataInAngF, M_INT32 **_GCCopy_StartPInAngF);
//void GPU_BoundValueUpdate(int _procID, int _numProcs);
#ifdef GPU_STREAM_USED
  void GPUTransportSweep_GCMemcpy_Streams();
  void GPUStreamInitialize();
#endif // GPU_STREAM_USED
// void GPUCopyBoundAngFluxDeviceToDevice();
// void GPUTS_GroupDDAngFReg_Azim_AntiClockFSR();
// void GPUTS_GroupDDAngFReg_Azim_RayWisedFSR();
// void GPUTS_GroupDDAngFReg_Azim_RayWisedFSR_group();
// void GPUTS_GroupDDAngFReg_Azim_RayWisedFSR_biDirection();
// void GPUTS_GroupDDAngFReg_Azim_RayWisedFSR_group_biDrection();
// void GPUTS_GroupDDAngFReg_Azim_RayWisedFSR_group_biDrection_1();
// void GPUTS_GroupDDAngFReg_Azim_RayWisedFSR_group_biDrection_2();
//void GPUAccumAngFlux(int _lay,int _polarBoundLowLim,int _polarBoundUpLim,int _polOffset,\
//  FP_PRECISION *_aveAngFlux_update, int _GPU_TSKernelType, int _dim);
void GPUAccumAngFlux(int _lay,int _polarBoundLowLim,int _polarBoundUpLim,int _polOffset,\
	FP_PRECISION *_aveAngFlux_update, int _GPU_TSKernelType, int _dim, int _numFSRs, int _numGroups, int _numAzim, int _numPol, int _scattOrder);
void GPUAccumAngFlux_PerAzim(int _lay, int _polarBoundLowLim, int _polarBoundUpLim,
	int _polOffset, FP_PRECISION_FLUX *_aveAngFlux_update, int _dim,
	int _azimID);
void GPUAccumAngFlux_PerAzimBatch(int _lay, int _polarBoundLowLim, int _polarBoundUpLim,
	int _polOffset, FP_PRECISION_FLUX *_aveAngFlux_update, int _dim,
	int _azimBegin, int _azimEnd);
void GPUUpdateAxialSurfAngFlux(int _dim, FP_PRECISION *_aveAngFlux_inp, FP_PRECISION *_aveAngFlux_update);
void GPUUpdateAxialSurfAngFlux_1(int _dim, FP_PRECISION *_aveAngFlux_inp, FP_PRECISION *_aveAngFlux_update,\
		FP_PRECISION _twoOverLayWidth, int _polarBoundLowLim, int _polarBoundUpLim, int _polOffset);
void GPUUpdate_AxialBoundCond_1(int _dir,FP_PRECISION *_aveAngFlux_inp, FP_PRECISION *_aveAngFlux_bound, \
	M_INT32 _TopBC, M_INT32 _BottomBC, M_INT32 _dim, M_INT32 _NumPolarAng);
	void GPUManipulateAxialInciAngFlux_1(M_INT32 _dim, FP_PRECISION *_aveAngFlux_inp, \
		FP_PRECISION _twoOverLayWidth, int _polarBoundLowLim, int _polarBoundUpLim, int _polOffset);
void PrintFPArrayOnDevice(int dim, FP_PRECISION *_devArray, char *_parameter);
// void GPUFinalOutputs(M_INT32 _dim, M_INT32 _OUTPUT_flux, M_INT32 _OUTPUT_flux_mat, M_INT32 _OUTPUT_normPinPower, FP_PRECISION *_ScalarFlux_MOC);

#if defined(M_MG_CMFD)
void GPUCMFD_AllocMem(int _procID, size_t *_GPUPreMemMGCMFD, size_t *_GPUPreMemCGCMFD, M_INT32 _MOC_NumPinsInSubDom,\
	M_INT32 _NumGroups, M_INT32 _NumCMFDGroup, M_INT32 _MOCNumLayersZ, M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y,\
	M_INT32 *_CMFDGroupBound, FP_PRECISION *_CMFD_NodeVol, M_INT32 *_NumSubDomains, M_INT32 _MGCMFD_MOCParCurrInnerBound_Num,\
	M_INT32 *_MGCMFD_MOCParCurrInnerBound_Index, M_INT32 *_MGCMFD_MOCParCurrInnerBound_Dir, M_INT32 *_CMFDComm_TarProc, \
	M_INT32 _NumPolarAng, M_INT32 _MGCMFDData_MPIComm_Num, M_INT32 *_MGCMFDData_MPIComm_SumInd, M_INT32 *_MGCMFDMPI_InBoundSourCoeff_Index, \
	M_INT32 *_MGCMFDMPI_InBoundSourCoeff_Dir, FP_PRECISION *_MGCMFDMPI_InBoundSourCoeff_GeoDim, M_INT32 *_MGCMFDData_MPIComm_Index, \
	M_INT32 _CGCMFDData_MPIComm_Num, M_INT32 *_CGCMFDMPI_InBoundSourCoeff_Index, M_INT32 *_CGCMFDMPI_InBoundSourCoeff_Dir, \
	FP_PRECISION *_CGCMFDMPI_InBoundSourCoeff_GeoDim, M_INT32 *_CGCMFDData_MPIComm_Index, M_INT32 _NumCoarseCMFDGroup, \
	M_INT32 *_CGCMFDGroupBound, M_INT32 *_CGCMFDData_MPIComm_SumInd, M_INT32 _MGCMFD_LinearSolver, M_INT32 _CGCMFD_LinearSolver, \
	FILE *_LogFile, M_INT32 _GmresKrylovDim_MG, M_INT32 _GmresKrylovDim_CG, M_INT32 _OUTPUT_CMFDGroupConstant, M_INT32 _OUTPUT_CoarseMesh_GroupConstant, int _isTransient, M_INT32 _NumProcs);
  void GPUCMFD_InitMOCCurr(FP_PRECISION _partCurr2D, FP_PRECISION _netCurrZ, M_INT32 _MOCNumLayersZ, \
	  M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, M_INT32 _NumGroups);
  void GPUCMFD_CheckCollapse(int _isFirstInvoke, M_INT32 _OUTPUT_CMFDGroupConstant, M_INT32 _OUTPUT_CoarseMesh_GroupConstant, M_INT32 _MOC_NumPinsInSubDom, M_INT32 _MOCNumLayersZ, \
	  M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, M_INT32 _NumCMFDGroup, M_INT32 _CMFD_NumNoneZeroValue_Matrix, \
	  FP_PRECISION *_CMFDHetScalFlux, FP_PRECISION *_CMFDXs_t, FP_PRECISION *_CMFDXs_a, FP_PRECISION *_CMFDXs_f, FP_PRECISION *_CMFDXs_nf, FP_PRECISION *_CMFD_Chi, FP_PRECISION *_CMFDXs_s, \
	  FP_PRECISION *_CMFDNodeDiffCoeff, FP_PRECISION *_MOCtoCMFD_ParCurr_X_Pos, FP_PRECISION *_MOCtoCMFD_ParCurr_X_Neg, \
	  FP_PRECISION *_MOCtoCMFD_ParCurr_Y_Pos, FP_PRECISION *_MOCtoCMFD_ParCurr_Y_Neg, FP_PRECISION *_MGCMFDNetCurr_X, FP_PRECISION *_MGCMFDNetCurr_Y, \
	  FP_PRECISION *_CMFDSurfDiffCoeff_X, FP_PRECISION *_CMFDSurfNDF_X, FP_PRECISION *_CMFDSurfDiffCoeff_Y, FP_PRECISION *_CMFDSurfNDF_Y, \
	  M_INT32 _MGCMFD_LinearSolver, FP_PRECISION *_MatrixCSR, FP_PRECISION *_MG_ILUMatrix, M_INT32 *_MatrixCSR_RowStartInd, M_INT32 *_MatrixCSR_ColInd, \
	  FP_PRECISION *_MGCMFDMPI_InBoundSourCoeff, M_INT32 _MGCMFDData_MPIComm_Num);
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
                                FP_PRECISION **_boundaryFlux, FP_PRECISION **_CPUMPIFlux);
 void GPUCMFD_GCEvaluation_Part2(M_INT32 _ProcID, M_INT32 _NumProcs, M_INT32 _isFirstInvoke, M_INT32 _MOC_NumPinsInSubDom, \
	  M_INT32 _MOCNumLayersZ, M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, M_INT32 _NumCMFDGroup, \
	  FP_PRECISION *_MGCMFDFlux_MPIRecv, FP_PRECISION *_MGCMFDNodeD_MPIRecv, M_INT32 _MGCMFDData_MPIComm_Num, \
	  M_INT32 _SubDomLocID_2D_X, M_INT32 _SubDomLocID_2D_Y, FP_PRECISION _MGCMFD_damp, FP_PRECISION *_CMFDXs_s, \
	  M_INT32 *_CMFD_NumNoneZeroValue_Matrix,M_INT32 _MGCMFD_LinearSolver, M_INT32 _GmresPrecond_MG, M_INT32 *_MG_DiagPointer, \
	  M_INT32 _ProblemType, FP_PRECISION_TIME *_TimeRec_MGCMFD_XSCollapse, FP_PRECISION_TIME *_TimeRec_MGCMFD_MatrixConst);
  void GPUCMFD_ParaUpdate();
  // FP_PRECISION GPU_MGCMFDSolver(FP_PRECISION _k);
  void GPUUpdateFlux_CMFDtoMOC(int _MOC_NumPinsInSubDom ,int _NumCMFDGroup, \
		FP_PRECISION_FLUX *_CMFDScalarFLux, FP_PRECISION_FLUX *_MOCtoCMFD_HetScalFlux);
  // FP_PRECISION GPU_CGCMFDSolver(FP_PRECISION _k, cublasHandle_t _handle);

  void GPUGetNetCurr_Axial_1(int _lay, int _currLay, FP_PRECISION _dir, \
    int _polarBoundLowLim, int _polarBoundUpLim, FP_PRECISION_FLUX *_surfAngFlux, \
    int _polOffset);
#endif // M_MG_CMFD
//void GPUPrintf1DArray_FP(int _dim, FP_PRECISION *_GPUArray, char *_parameter, FILE *_LogFile);
void GPUCompareTwo1DArray_FP(int _dim, FP_PRECISION *_GPUArrayOrig, FP_PRECISION *_GPUArrayUp, \
  char *_parameter, FILE *_LogFile);
void GPUTallyMOCPartialCurr_InnerBound(M_INT32 _NumProcs, M_INT32 _MOCNumLayersZ, M_INT32 _NumPolarAng, M_INT32 _NumLTs_HS, \
	M_INT32 _NumGroups, M_INT32 _NumPinsInSubDomPerLayer, FP_PRECISION **_boundaryFlux, FP_PRECISION **_CPUMPIFlux);
void GPU_MGCMFDSolver_Part1(M_INT32 *_MGCMFD_Diver, M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCMFDGroup);
void GPU_MGCMFDSolver_Part2(M_INT32 _procID, M_INT32 _iter_moc, M_INT32 _iter_mg, FP_PRECISION _k, M_INT32 _ProblemType,\
	M_INT32 *_MGCMFD_FluxFlag, M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCMFDGroup, FP_PRECISION *_CMFDScalarFlux, int _isTransient, M_INT32 _NumProcs);
int GPU_MGCMFDSolver_Part3(M_INT32 _iter_moc, M_INT32 _iter_mg, FP_PRECISION _k, M_INT32 _ProcID, M_INT32 _NumProcs, \
	M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCMFDGroup, FP_PRECISION *_MGCMFDFlux_MPIRecv, M_INT32 _MGCMFDData_MPIComm_Num, \
	M_INT32 _MGCMFD_LinearSolver, M_INT32 _MaxNumIter_MGLS, FP_PRECISION _ConvCritLinearSolver_MGCMFD, M_INT32 _MGCMFD_checkFluxNormInf, \
	M_INT32 _MGCMFD_checkFluxNorm2, M_INT32 _MGCMFD_checkFluxRMSE, FP_PRECISION *_MGCMFDLS_NormInf, FP_PRECISION *_MGCMFDLS_Norm2, \
	FP_PRECISION *_MGCMFDLS_RMSE, FP_PRECISION *_MG_ILUMatrix, M_INT32 *_MG_DiagPointer, M_INT32 _GmresPrecond_MG, \
	FP_PRECISION _SOR_Factor_MG, M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, FP_PRECISION_TIME *_Time_MGCMFD_GMRES_Precond, \
	FP_PRECISION_TIME *_Time_MGCMFD_GMRES_LeastSqr, FP_PRECISION_TIME *_Time_MGCMFD_GMRES_MemCpy, FP_PRECISION_TIME *_Time_MGCMFD_GMRES_SpMV, \
	FP_PRECISION_TIME *_Time_MGCMFD_GMRES_BLAS1, float *_MGCMFD_NegFluxCount, M_INT32 *_MGCMFD_FluxFlag, FP_PRECISION_TIME *_TimeRec_MGCMFD_LinearSolver,\
	M_INT32 _CMFD_NumNoneZeroValue_Matrix, FP_PRECISION *_MG_Residual, M_INT32 _GmresKrylovDim_MG);
void GPU_MGCMFD_UpdateQFiss(M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCMFDGroup);
void GPU_CGCMFD_UpdateQFiss(M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCoarseCMFDGroup);
void GPUSetXSt_wScale_1();
void GPUComputeSourceOverSigma_3DMOCex_1(int _lay,int _polarBoundLowLim,int _polarBoundUpLim,int _polOffset, \
  FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION _twoOverLayWidth);
void GPUCompAxialAngF_FromSourceOverSigma_3DMOCex_1(int _lay,int _polarBoundLowLim,int _polarBoundUpLim,int _polOffset, \
  FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION _twoOverLayWidth);
  
  void GPUCGCMFD_GCEvaluation_Part1(M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCoarseCMFDGroup, FP_PRECISION_TIME *_TimeRec_CGCMFD_XSCollapse);
  void GPUCGCMFD_GCEvaluation_Part2(M_INT32 _ProcID, M_INT32 _NumProcs, M_INT32 _isFirstInvoke, M_INT32 _MOC_NumPinsInSubDom, \
	  M_INT32 _MOCNumLayersZ, M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, M_INT32 _NumCMFDGroup, M_INT32 _NumCoarseCMFDGroup, \
	  M_INT32 _CGCMFDData_MPIComm_Num, M_INT32 _MGCMFDData_MPIComm_Num, FP_PRECISION *_CGCMFDFlux_MPIRecv, FP_PRECISION *_CGCMFDNodeD_MPIRecv, \
	  FP_PRECISION *_MGCMFDFlux_MPIRecv, M_INT32 _SubDomLocID_2D_X, M_INT32 _SubDomLocID_2D_Y, FP_PRECISION *_CGCMFDXs_s, \
	  M_INT32 *_CGCMFD_NumNoneZeroValue_Matrix, M_INT32 _CGCMFD_LinearSolver, \
	  M_INT32 _GmresPrecond_CG, FP_PRECISION *_CG_ILUMatrix, FP_PRECISION *_CGMatrixCSR, M_INT32 *_CG_DiagPointer, M_INT32 *_CG_IW,\
	  M_INT32 _ProblemType, FP_PRECISION_TIME *_TimeRec_CGCMFD_XSCollapse, FP_PRECISION_TIME *_TimeRec_CGCMFD_MatrixConst);

  void GPU_CGCMFDSolver_Part1(M_INT32 *_CGCMFD_Diver, M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCoarseCMFDGroup);
void GPU_CGCMFDSolver_Part2(M_INT32 _procID, M_INT32 _iter_moc, M_INT32 _iter_mg, M_INT32 _iter_cg, FP_PRECISION _k,\
	M_INT32 _ProblemType, M_INT32 *_CGCMFD_FluxFlag, M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCoarseCMFDGroup, \
	FP_PRECISION *_CGCMFDScalarFlux, M_INT32 _NumProcs);
  int GPU_CGCMFDSolver_Part3(M_INT32 _iter_moc, M_INT32 _iter_mg, M_INT32 _iter_cg, FP_PRECISION _k, M_INT32 _ProcID, M_INT32 _NumProcs, \
	  M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCoarseCMFDGroup, FP_PRECISION *_CGCMFDFlux_MPIRecv, M_INT32 _CGCMFDData_MPIComm_Num, \
	  M_INT32 _CGCMFD_LinearSolver, M_INT32 _MaxNumIter_CGLS, FP_PRECISION _ConvCritLinearSolver_CGCMFD, M_INT32 _CGCMFD_checkFluxNormInf, \
		M_INT32 _CGCMFD_checkFluxNorm2, M_INT32 _CGCMFD_checkFluxRMSE, FP_PRECISION *_CGCMFDLS_NormInf, FP_PRECISION *_CGCMFDLS_Norm2, \
		FP_PRECISION *_CGCMFDLS_RMSE, FP_PRECISION _SOR_Factor_CG, M_INT32 _MOCNumPinsInSubDom_X, M_INT32 _MOCNumPinsInSubDom_Y, \
		M_INT32 _CGCMFD_NumNoneZeroValue_Matrix, M_INT32 _GmresKrylovDim_CG, FP_PRECISION *_CG_ILUMatrix, M_INT32 *_CG_DiagPointer, \
	  M_INT32 _GmresPrecond_CG, FP_PRECISION_TIME *_Time_CGCMFD_GMRES_Precond, FP_PRECISION_TIME *_Time_CGCMFD_GMRES_LeastSqr, \
	  FP_PRECISION_TIME *_Time_CGCMFD_GMRES_MemCpy, FP_PRECISION_TIME *_Time_CGCMFD_GMRES_SpMV, FP_PRECISION_TIME *_Time_CGCMFD_GMRES_BLAS1, \
	  float *_CGCMFD_NegFluxCount, M_INT32 *_CGCMFD_FluxFlag, FP_PRECISION_TIME *_TimeRec_CGCMFD_LinearSolver, FP_PRECISION *_CG_Residual);
  void GPUUpdateScalFlux_CGtoMGCMFD_1(M_INT32 _MOC_NumPinsInSubDom, M_INT32 _NumCoarseCMFDGroup);
  void cudaDeviceSynchronize_cpu();
  void GetConstantMemUsage_GPU(int _procID, int _numAzimAng, int _numPolAng, FILE *_LogFile);
  __global__ void GPUAddSourceToScalarFluxOnDevice_lay_g(int _numFSRsPerLay, int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, \
    int _groupBoundUpLim, FP_PRECISION_FLUX *_GPUScalarFlux,FP_PRECISION *_GPUSource,FP_PRECISION *_GPUMacroXS_t);
void GPUMOC_TransSweep_FixSourProb_lay_g(FP_PRECISION _twoOverLayWidth, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim, FP_PRECISION *_GPUFSRAveAngFlux_inp, \
  FP_PRECISION *_GPUFSRAveAngFlux_update, int _polOffset, int _layBoundLowLim, int _layBoundUpLim, \
	int _numFSRsPerLay, M_INT32 _NumPinsInSubDomPerLayer, M_INT32 _NumLTs_HS, M_INT32 _NumPolarAng, M_INT32 _GPU_FixSourSolver);
void GPUMOC_TransSweep_FixSourProb_lay_g_sld(FP_PRECISION _twoOverLayWidth, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim, FP_PRECISION *_GPUFSRAveAngFlux_inp, \
  FP_PRECISION *_GPUFSRAveAngFlux_update, int _polOffset, int _layBoundLowLim, int _layBoundUpLim, \
	int _numFSRsPerLay, M_INT32 _NumPinsInSubDomPerLayer, M_INT32 _NumLTs_HS, M_INT32 _NumPolarAng, M_INT32 _GPU_FixSourSolver);
void GPUScalarFluxUpdate_FixSour1(int _TotalNumFSRs_subDom, int _NumGroups, FP_PRECISION_FLUX *_scalarFlux_MOC);
void GPUScalarFluxUpdate_FixSour2(int _TotalNumFSRs_subDom, int _NumGroups, FP_PRECISION_FLUX *_scalarFluxOld);
void GPUCopyMacroXSsHostToDevice_Init(int _TotalNumFSRs_subDom, int _NumGroups, int _MOCNumLayersZ, \
	M_UINT64 *_MacroXS_s_NZRowStartID_Layer, M_UINT64 *_MacroXS_s_NZColStartID, size_t *_GPUPreMemXSs,\
	M_INT32 _IfResonance, M_INT32 _ProblemType, M_UINT64 _MacroXS_s_MaxNNZ_Layer);

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
                                         M_UINT64 *_GPUMacroXS_s_NZColStartID);


void GPUComputeFSRSource_Layer(int _layID, int _TotalNumFSRs_subDom, int _TotalNumFSRs_perLayer, int _NumGroups, M_UINT64 *_MacroXS_s_NZStartID_Layer, \
	M_UINT64 *_MacroXS_s_NNZ_Layer, M_INT32 _ProblemType, FP_PRECISION *_MacroXS_t, FP_PRECISION *_MacroXS_nf, FP_PRECISION *_Chi, \
	FP_PRECISION *_MacroXS_s, FP_PRECISION *_FixedSour, FP_PRECISION _invKeff, M_INT32 _NumScattOrder, M_INT32 _GPU_TSKernelType,\
	M_UINT64 *_MacroXS_s_NZRowStartID_Layer, M_UINT64 *_MacroXS_s_NZColStartID, FP_PRECISION_FLUX *_ScalarFluxOld, int _polarBoundLowLim,
    int _isTransient, FP_PRECISION *_fissSource, FP_PRECISION *_tranConstA, FP_PRECISION *_tranConstB, FP_PRECISION *_tranConstC);
void GPUComputeFSRSource_Layer_PerAzim(int _layID, int _TotalNumFSRs_subDom, int _TotalNumFSRs_perLayer, int _NumGroups, M_UINT64 *_MacroXS_s_NZStartID_Layer, \
	M_UINT64 *_MacroXS_s_NNZ_Layer, M_INT32 _ProblemType, FP_PRECISION *_MacroXS_t, FP_PRECISION *_MacroXS_nf, FP_PRECISION *_Chi, \
	FP_PRECISION *_MacroXS_s, FP_PRECISION *_FixedSour, FP_PRECISION _invKeff, M_INT32 _NumScattOrder, M_INT32 _GPU_TSKernelType,\
	M_UINT64 *_MacroXS_s_NZRowStartID_Layer, M_UINT64 *_MacroXS_s_NZColStartID, FP_PRECISION_FLUX *_ScalarFluxOld, int _polarBoundLowLim,\
	int _azimBegin, int _azimEnd, int _reuseAngleIndependentSource, int _isTransient, FP_PRECISION *_fissSource, FP_PRECISION *_tranConstA, FP_PRECISION *_tranConstB, FP_PRECISION *_tranConstC);
void LoadBoundaryFluxCPUtoGPU_Layer(int _layID, int _NumLongTracks_lay, int _numPolarAng, int _numGroups, FP_PRECISION_FLUX **_boundaryFlux, M_INT32 _NumProcs, FP_PRECISION **_CPUMPIFlux);
//void StoreBoundaryFluxGPUtoCPU_Layer(int _layID, int _NumLongTracks_lay, int _numPolarAng, int _numGroups);
void StoreBoundaryFluxGPUtoCPU_Layer(int _layID, int _NumLongTracks_lay, int _numPolarAng, int _numGroups, FP_PRECISION_FLUX **_boundaryFlux, M_INT32 _NumProcs, FP_PRECISION **_CPUMPIFlux);
void GPU_AllocMemScattXSs_afterResCalc();
//void GPUPrintf1DArray_I32(int _dim, M_INT32 *_GPUArray,char *_parameter, FILE *_LogFile);
//void GPUPrintf1DArray_UI64(int _dim, M_UINT64 *_GPUArray,char *_parameter, FILE *_LogFile);
void GPUInitScalarFlux_Dev(int _dim, FP_PRECISION_FLUX _val);
void GPUBoundFluxPre_StreamUsed(int _dim, FP_PRECISION *_CGBoundaryFlux, FP_PRECISION _val);
void ScalarFluxCopy_Dev2Host(FP_PRECISION_FLUX *_ScalarFlux_host, size_t _dim, int _pointer, char *_para);
void GPU_InitAxiAngFlux(M_INT32 _ProblemType, M_INT32 _dim, M_INT32 _BottomBC, M_INT32 _TopBC, \
	FP_PRECISION _val1, FP_PRECISION _zero_fp);
void CopyInpInfo_CPU2GPU(M_INT32 _GPUBlockDimInp, M_INT32 _GPUNumStreamsInp);
void OutputMacroXSs_FSRs_GPU(int _procID,int _numLayersZ, int _TotalNumFSRs_subDom,int _numCellsPerLayer,int *_numFSRsInCell,int *_sumNumFSRsBasedGlobCellID,\
	int _TotalNumFSRs_perLayer, int _NumGroups, M_UINT64 *_MacroXS_s_NZStartID_Layer, M_UINT64 *_MacroXS_s_NNZ_Layer,\
	M_INT32 _ProblemType, FP_PRECISION *_MacroXS_t, FP_PRECISION *_MacroXS_nf, FP_PRECISION *_Chi, FP_PRECISION *_MacroXS_s, \
	FP_PRECISION *_FixedSour, int _OUTPUT_MacorXSs, M_UINT64 _MacroXS_s_MaxNNZ_Layer, char *_CaseId, M_INT32 _NumScattOrder,\
	M_UINT64 *_MacroXS_s_NZRowStartID_Layer, M_UINT64 *_MacroXS_s_NZColStartID, FP_PRECISION_FLUX *_ScalarFluxOld);
void GPUCoarseMeshGroupConst_CheckCollapse(M_INT32 _OUTPUT_CoarseMesh_GroupConstant, M_INT32 _MOC_NumPinsInSubDom, \
	M_INT32 _NumCMFDGroup, FP_PRECISION *_CMFDXs_t, FP_PRECISION *_CMFDXs_a, FP_PRECISION *_CMFDXs_f, FP_PRECISION *_CMFDXs_nf, \
	FP_PRECISION *_CMFD_Chi, FP_PRECISION *_CMFDXs_s);
void GPUInitAngFlux_Dev(int _dim, FP_PRECISION_FLUX _val, int _id);
void GPUInitAzimAccumFlux(int _dim, FP_PRECISION_FLUX _value);
void GPUFinalizeAzimAccumFlux(int _dim);

extern FP_DOUBLE *GPUFSRFissRate;
extern FP_PRECISION_FLUX *GPUScalarFlux_RelErr;
FP_DOUBLE GPUCalcReacRate_SubDom(int _NumGroups, int _NumFSRs);
FP_PRECISION GPUComputeScalarFluxRelErrDot(int _NumGroups, int _NumFSRs, int _ScatterOrder);

void warmup();

#endif //GPUSOLVER_H./H
