#ifndef GPU_TSSOLVER_H
#define GPU_TSSOLVER_H

#include <cuda_runtime.h>
#include "MOC_Track.h"

extern __constant__ int GPUNumPolarAng[1]; /** Half the number of polar angles */
extern __constant__ int GPUNumMOCGroups[1]; /** The number of energy groups */
extern __constant__ int GPUNumLTs_Total[1];  // number of tracks in [0,2PI]
extern __constant__ int GPUTotNumTracksInOneRTM_2PI[1];
extern __constant__ int GPUPolarTimesGroups[1];/** The number of polar angles int [0,PI/2] times energy groups */
extern __constant__ FP_PRECISION GPUSinThetasTimes2[MAX_POLAR_ANGLES_GPU];
extern __constant__ FP_PRECISION GPUTotalWeights[MAX_POLAR_ANGLES_GPU*MAX_AZIM_ANGLES_GPU]; /** An array of the weights from the Quadrature set */

#ifdef __cplusplus
extern "C" {
#endif

extern FP_PRECISION *GPUScalarFlux, *GPUMOCSource, *GPUMacroXS_t, *GPUMacroXS_t_inv;
extern int GPUGridDim_source_para_Layer;

#ifdef __cplusplus
}
#endif


#define _GPUBoundaryFlux(lt,pg) (_GPUBoundaryFlux[(lt)*(*GPUPolarTimesGroups)+(pg)])
#define _GPUStartFlux(lt,pg) (_GPUStartFlux[(lt)*(*GPUPolarTimesGroups)+(pg)])
#define _GPUMPIFlux(lt,pg) (_GPUMPIFlux[(lt)*(*GPUPolarTimesGroups)+(pg)])

__device__ FP_PRECISION atomicAdd_(FP_PRECISION* _address, FP_PRECISION _val);

__device__ void GPUUpdateBoundaryFlux_3polar_1(int _azim, int _azimNext, int *_GPUNumLTsSum, int _ltIDNext, int _polEngID, \
	int _ltGlobID, FP_PRECISION track_flux_0, FP_PRECISION track_flux_1, FP_PRECISION track_flux_2, FP_PRECISION *_GPUBoundaryFlux, \
	FP_PRECISION *_GPUMPIFlux, FP_PRECISION _beta);
__device__ void GPUUpdateBoundaryFlux_3polar(int _azim, int _azimNext, int *_GPUNumLTsSum, int _ltIDNext, int _polEngID, \
	int _ltGlobID, FP_PRECISION track_flux_0, FP_PRECISION track_flux_1, FP_PRECISION track_flux_2, FP_PRECISION *_GPUStartFlux, \
	FP_PRECISION *_GPUMPIFlux, FP_PRECISION _beta);

__device__ void GPUTallyScalarFlux_DDExpInte_OneDir_AngFSmem(int _index, int _indexLay, int _azim, FP_PRECISION _length, FP_PRECISION _oneOverVolume, \
	FP_PRECISION *_GPUMacroXS_t, FP_PRECISION *_track_flux, FP_PRECISION *_GPUSource, FP_PRECISION_FLUX *_GPUScalarFlux, \
	FP_PRECISION *_GPULinearExpTable, int _polarBoundLowLim,int _polarBoundUpLim, int polOffset, FP_PRECISION _twoOverWidth,\
	FP_PRECISION _oneOverVolbySpacing,M_SIZET _angId,FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update);

__device__ void GPUUpdateBoundaryFlux_AngFluxOnMem_UpAngFImme(int _ltInd_lay, int _azim, int _azimNext, int *_GPUNumLTsSum, int _ltIDNext, int _polEngID, \
	int _ltGlobID, FP_PRECISION *track_flux, FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUMPIFlux, FP_PRECISION _beta,\
	int _polarBoundLowLim,int _polarBoundUpLim, int polOffset);

// __global__ void GPUTS_Group_ExpInte_OneDir_AngFReg(int _lay, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
// 	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
// 	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
// 	int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
// 	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
// 	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
//     FP_PRECISION_GEO *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
//     FP_PRECISION *_GPULinearExpTable,FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
//     FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
//     FP_PRECISION_GEO *_GPURTMWidth_Y);

// __global__ void GPUTS_Group_DD_OneDir_AngFReg(FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
// 	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
// 	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
// 	int *_GPUTrackIndexInLTs, int _lay, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
// 	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
// 	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
// 	FP_PRECISION_GEO *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
//     FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
//     FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
//     FP_PRECISION_GEO *_GPURTMWidth_Y);

// __global__ void GPUTS_Group_DDExpInte_OneDir_AngFReg(int _lay, FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
// 	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
// 	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
// 	int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
// 	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
// 	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
//     FP_PRECISION_GEO *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
//     FP_PRECISION *_GPULinearExpTable,FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
//     FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
//     FP_PRECISION_GEO *_GPURTMWidth_Y);

// __global__ void GPUTS_Group_ExpFunc_OneDir_AngFReg(FP_PRECISION_FLUX *_GPUScalarFlux, FP_PRECISION *_GPUSource, \
// 	FP_PRECISION *_GPUBoundaryFlux, FP_PRECISION *_GPUStartFlux, int *_GPULTsIndex_AzimAngID, int *_GPULTsIndex_LTID, \
// 	struct SUBDOM_LONGTRACK_INFO *_GPULongTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, \
// 	int *_GPUTrackIndexInLTs, int _lay, int _numCellsInDom, int *_GPUCellRTMType_SubDom, int *_GPUIndexTrackInRTM_SumAzim, \
// 	int *_GPUNumSegsInTrack, int *_GPUSumNumSegsInRTMTrack, int *_GPUSumNumFSRsBasedGlobCellID, int *_GPUSumNumFSRsInRTM, \
// 	FP_PRECISION *_GPUMacroXS_t, int *_GPUNumLTsSum, int *_GPUNumLongTracks, FP_PRECISION *_GPUMPIFlux, \
//     FP_PRECISION_GEO *_GPUSegInfoInTrack_2PI_length, int *_GPUSegInfoInTrack_2PI_FSRID, FP_PRECISION_GEO *_GPUOneOverFSRsAreaInRTM,\
//     FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, \
//     FP_PRECISION *_GPUMOCParCurr_Y_Neg, int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, \
// 	FP_PRECISION_GEO *_GPURTMWidth_Y);
	
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
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, int polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC);

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
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, int _polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC);

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
	FP_PRECISION *_GPUMacroXS_t_wScale);

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
	FP_PRECISION *_GPUMacroXS_t_wScale, int _azimBegin, int _azimEnd);

__global__ void GPUAccumAngFlux_ScalarFlux(int _lay, int *_GPUNumFSRs_Layer, int *_GPUSumNumFSRs_Layer, \
    int _polarBoundLowLim, int _polarBoundUpLim, FP_PRECISION_FLUX *_aveAngFlux, FP_PRECISION_FLUX *_scalarFlux, \
	int _polOffset);
__global__ void GPUUpdate_AxialSurfFlux(int _dim, FP_PRECISION *_aveAngFlux_inp, FP_PRECISION *_aveAngFlux_update);
__global__ void GPUUpdate_AxialSurfFlux_1(int _dim, FP_PRECISION *_aveAngFlux_inp, FP_PRECISION *_aveAngFlux_update,\
	FP_PRECISION _twoOverLayWidth, FP_PRECISION *_GPUFSRAngScattSour, int _polarBoundLowLim, int _polarBoundUpLim, int _polOffset);
__global__ void GPUUpdate_AxialBoundCond(FP_PRECISION _beta, int _dim, int _halfPol, \
	FP_PRECISION *_aveAngFlux_inp, FP_PRECISION *_aveAngFlux_bound);

__global__ void GPUComputeSourceOverSigma_3DMOCex(int _lay, int *_GPUNumFSRs_Layer, int *_GPUSumNumFSRs_Layer, \
    int _polarBoundLowLim, int _polarBoundUpLim, FP_PRECISION_FLUX *_axialSurfAnfFlux_inp, \
	int _polOffset, FP_PRECISION _twoOverLayWidth, FP_PRECISION *_GPUMacroXS_t_wScale, FP_PRECISION *_GPUSource);

__global__ void GPUCompAxialAngF_FromSourceOverSigma_3DMOCex(int _lay, int *_GPUNumFSRs_Layer, int *_GPUSumNumFSRs_Layer, \
    int _polarBoundLowLim, int _polarBoundUpLim, FP_PRECISION_FLUX *_axialSurfAnfFlux_inp, \
	int _polOffset, FP_PRECISION _twoOverLayWidth, FP_PRECISION *_GPUMacroXS_t_wScale, FP_PRECISION *_GPUSource);

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
    FP_PRECISION *_GPUMacroXS_t_wScale,int _groupBoundLowLim, int _groupBoundUpLim);

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
  FP_PRECISION *_GPUMacroXS_t_wScale,int _groupBoundLowLim, int _groupBoundUpLim);

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
  FP_PRECISION *_GPUMacroXS_t_wScale,int _groupBoundLowLim, int _groupBoundUpLim);

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
	int _groupBoundLowLim, int _groupBoundUpLim);

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
	int _groupBoundLowLim, int _groupBoundUpLim);






__global__ void GPUAccumAngFlux_ScalarFlux_HOScatt(int _lay, int *_GPUNumFSRs_Layer, int _polarBoundLowLim, \
	int _polarBoundUpLim, FP_PRECISION_FLUX *_aveAngFlux, FP_PRECISION_FLUX *_scalarFlux, int _polOffset, \
	FP_PRECISION *_GPUSpheHarmCoeff);

__global__ void GPUAccumAngFlux_ScalarFlux_HOScatt_opt(int _lay, int *_GPUNumFSRs_Layer, int _polarBoundLowLim, \
    int _polarBoundUpLim, FP_PRECISION_FLUX *_aveAngFlux, FP_PRECISION_FLUX *_scalarFlux, int _polOffset, \
    FP_PRECISION *_GPUSpheHarmCoeff);

__global__ void GPUAccumAngFlux_ScalarFlux_HOScatt_opt_sld(int _lay, int *_GPUNumFSRs_Layer, int _polarBoundLowLim, \
    int _polarBoundUpLim, FP_PRECISION_FLUX *_aveAngFlux, FP_PRECISION_FLUX *_scalarFlux, int _polOffset, \
    FP_PRECISION *_GPUSpheHarmCoeff, int sldOrder);

__global__ void GPUAccumAngFlux_ScalarFlux_HOScatt_opt2(int _lay, int *_GPUNumFSRs_Layer, int _polarBoundLowLim, \
    int _polarBoundUpLim, FP_PRECISION_FLUX *_aveAngFlux, FP_PRECISION_FLUX *_scalarFlux, int _polOffset, \
    FP_PRECISION *_GPUSpheHarmCoeff);

__global__ void GPUAccumAngFlux_ScalarFlux_HOScatt_opt3(int _lay, int *_GPUNumFSRs_Layer, int _polarBoundLowLim, \
	int _polarBoundUpLim, FP_PRECISION_FLUX *_aveAngFlux, FP_PRECISION_FLUX *_scalarFlux, int _polOffset, \
	FP_PRECISION *_GPUSpheHarmCoeff);

__global__ void GPUAccumAngFlux_ScalarFlux_HOScatt_PerAzim(
	int _lay, int *_GPUNumFSRs_Layer, int _polarBoundLowLim,
	int _polarBoundUpLim, FP_PRECISION_FLUX *_aveAngFlux,
	FP_PRECISION_FLUX *_scalarFlux, int _polOffset,
	FP_PRECISION *_GPUSpheHarmCoeff, int _azimBegin, int _azimEnd);



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
	M_UINT64 *_GPUMacroXS_s_NZColStartID, FP_PRECISION *_GPUSpheHarmCoeff);

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
	FP_PRECISION *_GPUMacroXS_t_wScale, FP_PRECISION *_GPUFSRAngScattSour);

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
	FP_PRECISION *_GPUMacroXS_t_wScale, FP_PRECISION *_GPUFSRAngScattSour, int _azimBegin, int _azimEnd);

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
	FP_PRECISION *_GPUMacroXS_t_wScale, FP_PRECISION *_GPUFSRAngScattSour, int, int);

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
	FP_PRECISION *_GPUMacroXS_t_wScale, FP_PRECISION *_GPUFSRAngScattSour);

__global__ void GPUComputeFSRFissSourceOnDevice(FP_PRECISION_FLUX *_GPUScalarFlux, \
	FP_PRECISION *_GPUSource, FP_PRECISION *_GPUMacroXS_nf, FP_PRECISION *_GPUMacroXS_chi, FP_PRECISION _inverseKeff);

__global__ void GPUOverWriteFissSourceWithGivenSource(FP_PRECISION *_GPUSource, FP_PRECISION *_GPUMacroXS_nf);

__global__ void GPUInitFSRHOSourceOnDevice(FP_PRECISION *_GPUFSRAngScattSour, FP_PRECISION_FLUX *_GPUScalarFlux, \
	FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID,\
    FP_PRECISION *_GPUSpheHarmCoeff, int _polarBoundLowLim);

__global__ void GPUComputeFSRScattSourceOnDevice_addFissSour(FP_PRECISION *_GPUFSRAngScattSour, FP_PRECISION_FLUX *_GPUScalarFlux, \
	FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID,\
	FP_PRECISION *_GPUSpheHarmCoeff, FP_PRECISION *_GPUFissSour, int _polarBoundLowLim, FP_PRECISION *_GPUMacroXS_t_inv);

/* Traditional full-angle high-order source kernels (types 15 and 16). */
__global__ void GPUComputeFSRScattSourceOnDevice_FsrGrpInAng(FP_PRECISION *_GPUFSRAngScattSour, FP_PRECISION_FLUX *_GPUScalarFlux, \
	FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID, \
	FP_PRECISION *_GPUSpheHarmCoeff, int _polarBoundLowLim);

__global__ void GPUComputeFSRScattSourceOnDevice_FsrGrpOutAng(FP_PRECISION *_GPUFSRAngScattSour, FP_PRECISION_FLUX *_GPUScalarFlux, \
	FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID, \
	FP_PRECISION *_GPUSpheHarmCoeff, int _polarBoundLowLim);

__global__ void GPUComputeFSRScattSourceOnDevice_FsrGrpInAng_Cluster(
	FP_PRECISION *_GPUFSRAngScattSour, FP_PRECISION_FLUX *_GPUScalarFlux,
	FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
	M_UINT64 *_GPUMacroXS_s_NZRowEndID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID,
	FP_PRECISION *_GPUSpheHarmCoeff, int _polarBoundLowLim);

__global__ void GPUComputeFSRScattSourceOnDevice_FsrGrpOutAng_Cluster(
	FP_PRECISION *_GPUFSRAngScattSour, FP_PRECISION_FLUX *_GPUScalarFlux,
	FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
	M_UINT64 *_GPUMacroXS_s_NZRowEndID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID,
	FP_PRECISION *_GPUSpheHarmCoeff, int _polarBoundLowLim);

/* LM/AP two-stage high-order scattering source kernels.
 * Stage 1 (LM): moment-resolved, angle-independent group coupling
 *   S_lm(g) = (2l+1) * sum_g1 Sigma_s,l(g1->g) * phi_lm(g1)
 * Stage 2 (AP): angular projection, shared by both LM variants
 *   q_s,g(Omega) = sum_lm (1/4pi) A_lm Y_lm(Omega) S_lm(g)
 * LM variants: gather (target-group-oriented, direct write) and
 * scatter (source-group-oriented, atomic accumulation). */
__global__ void GPUComputeFSRScatterSourceLMOnDevice_grpOut(FP_PRECISION *_GPUFSRAngScatterSourceLM, FP_PRECISION_FLUX *_GPUScalarFlux, \
	FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID);

__global__ void GPUComputeFSRScatterSourceLMOnDevice_grpIn(FP_PRECISION *_GPUFSRAngScatterSourceLM, FP_PRECISION_FLUX *_GPUScalarFlux, \
	FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID);

__global__ void GPUComputeFSRScatterSourceLMOnDevice_grpOut_Cluster(
	FP_PRECISION *_GPUFSRAngScatterSourceLM, FP_PRECISION_FLUX *_GPUScalarFlux,
	FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
	M_UINT64 *_GPUMacroXS_s_NZRowEndID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID);

__global__ void GPUComputeFSRScatterSourceLMOnDevice_grpIn_Cluster(
	FP_PRECISION *_GPUFSRAngScatterSourceLM, FP_PRECISION_FLUX *_GPUScalarFlux,
	FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
	M_UINT64 *_GPUMacroXS_s_NZRowEndID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID);

__global__ void GPUComputeFSRScatterSourceAPOnDevice(FP_PRECISION *_GPUFSRAngScattSour, FP_PRECISION *_GPUSpheHarmCoeff, \
	FP_PRECISION *_GPUFSRAngScatterSourceLM);

__global__ void GPUComputeFSRScatterSourceAPOnDeviceOpt1(FP_PRECISION *_GPUFSRAngScattSour, FP_PRECISION *_GPUSpheHarmCoeff, \
	FP_PRECISION *_GPUFSRAngScatterSourceLM);

__global__ void GPUComputeFSRScatterSourceAPOnDeviceOpt2(FP_PRECISION *_GPUFSRAngScattSour, FP_PRECISION *_GPUSpheHarmCoeff, \
	FP_PRECISION *_GPUFSRAngScatterSourceLM);

__global__ void GPUComputeFSRScatterSourceAPOnDeviceOpt3(FP_PRECISION *_GPUFSRAngScattSour, FP_PRECISION *_GPUSpheHarmCoeff, \
	FP_PRECISION *_GPUFSRAngScatterSourceLM);

__global__ void GPUComputeFSRScatterSourceAPOnDevice_PerAzim(
	FP_PRECISION *_GPUFSRAngScattSour, FP_PRECISION *_GPUSpheHarmCoeff,
	FP_PRECISION *_GPUFSRAngScatterSourceLM, int _polarBoundLowLim,
	int _azimBegin, int _azimEnd, int _apKernel);

/* Correct 2D per-azimuth variants for types 15 and 16.  The selected azimuth is
 * passed explicitly while the output remains compact: FSR x group x polar. */
__global__ void GPUComputeFSRScattSourceOnDevice_FsrGrpInAng_PerAzim(
	FP_PRECISION *_GPUFSRAngScattSour, FP_PRECISION_FLUX *_GPUScalarFlux,
	FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
	M_UINT64 *_GPUMacroXS_s_NZRowEndID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID,
	FP_PRECISION *_GPUSpheHarmCoeff, int _polarBoundLowLim, int _azimID,
	int _useCluster);

__global__ void GPUComputeFSRScattSourceOnDevice_FsrGrpOutAng_PerAzim(
	FP_PRECISION *_GPUFSRAngScattSour, FP_PRECISION_FLUX *_GPUScalarFlux,
	FP_PRECISION *_GPUMacroXS_s, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer,
	M_UINT64 *_GPUMacroXS_s_NZRowEndID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID,
	FP_PRECISION *_GPUSpheHarmCoeff, int _polarBoundLowLim, int _azimID,
	int _useCluster);

#endif // GPU_TSSOLVER_H
