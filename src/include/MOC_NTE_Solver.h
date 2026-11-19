#ifndef MOC_NTE_SOLVER_H
#define MOC_NTE_SOLVER_H
#include "MOC_Track.h"
#include "MOC_CommonMacro.h"

//#define REC_TIME_SOLVER

extern FP_PRECISION_TIME RecTime_RayWisedSolver, RecTime_RayWisedSolver_PreSource, RecTime_RayWisedSolver_PreXSt;
extern FP_PRECISION_TIME RecTime_RayWisedSolver_ZeroFlux, RecTime_RayWisedSolver_PreFSRVol, RecTime_RayWisedSolver_AccumFlux;

// void Jacobi2DMOCSolver_DiamDiffExpTab_LTParaNonAtomic(int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
// 	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
// 	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
// 	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
// 	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, FP_PRECISION *_totalWeights_2DMOC, \
// 	int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int _numOMPThreads_comp, int *_sumNumSegsInRTMTrack_2PI, \
// 	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
// 	int *_sumNumFSRsInRTM, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux);

// void Jacobi2DMOCSolver_DiamDiffExpTab_LTParaNonAtomic_OneDir(int _threadID, int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
// 	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
// 	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
// 	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
// 	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, FP_PRECISION *_totalWeights_2DMOC, \
// 	int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int _numOMPThreads_comp, int *_sumNumSegsInRTMTrack_2PI, \
// 	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
// 	int *_sumNumFSRsInRTM, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux);

void Jacobi2DMOCSolver_ExpFuncRec_LTParaNonAtomic(int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC, int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int _numOMPThreads_comp, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, \
	FP_PRECISION *_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp, \
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);

void Jacobi2DMOCSolver_ExpFuncRec_LTParaNonAtomic_BiDir(int _threadID, int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC, int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int _numOMPThreads_comp, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, \
	FP_PRECISION *_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp, \
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);

// void Jacobi2DMOCSolver_ExpTabRec_LTParaNonAtomic_BiDir_FixSourProb_1(int _threadID, int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
// 	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
// 	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
// 	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
// 	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
// 	int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int _numOMPThreads_comp, int *_sumNumSegsInRTMTrack_2PI, \
// 	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
// 	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, FP_PRECISION *_expTempStore);

// void Jacobi2DMOCSolver_ExpTabRec_LTParaNonAtomic_BiDir_FixSourProb_2(int _threadID, int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
// 	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
// 	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
// 	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
// 	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
// 	FP_PRECISION *_totalWeights_2DMOC, int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int _numOMPThreads_comp, int *_sumNumSegsInRTMTrack_2PI, \
// 	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
// 	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, \
// 	FP_PRECISION *_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp, \
// 	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);

void Jacobi2DMOCSolver_DiamDiffExpTab_LTParaNonAtomic_OneDir_FixSourProb(int _threadID, int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int _numOMPThreads_comp, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux);

// void Jacobi2DMOCSolver_ExpFuncRec_LTParaNonAtomic_BiDir_FixSourProb(int _threadID, int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
// 	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
// 	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
// 	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
// 	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
// 	int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int _numOMPThreads_comp, int *_sumNumSegsInRTMTrack_2PI, \
// 	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
// 	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_trackFlux, FP_PRECISION *_expTempStore);

#ifdef OMP_USED
void Jacobi_MOCSolver_2D_diam_LongTrack_ParaFor_Atomic(int ltHS, int _lay, int _trSet, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, FP_PRECISION *_ltInciAngFlux, \
	FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, int _polarBoundUpLim, int _numCellsInDom, \
	int *_numFSRsInCell, int *_num_LT_subDom, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	FP_PRECISION *_sinTheta, FP_PRECISION **_trackFlux_1, int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, \
	FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, int _numLongtrack_halfSpace, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, \
	int *_numSegsInTrack, struct SUBDOM_SEG_IN_TRACK_SINGLE *_segInfoInTrack, int *_sumNumSegsInRTMTrack, \
	struct RTMFSRINFO **_RTMFSRsInfo, FP_PRECISION *_macroXS_t, FP_PRECISION **_inpXSs_t, FP_PRECISION *_totalWeights, \
	int _FSRVolCorrType, int _numLTsInHalfspace, int *_numLTsSum, int _totNumTracksInOneRTM_HS, int *_indexTrackInRTM_SumAzim_HS);
void Jacobi_MOCSolver_2D_diam_LongTrack_Parallel_Atomic(int _threadID, int _lay, int _trSet, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom, int *_numFSRsInCell, int *_num_LT_subDom, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, \
	int *_cellRTMGeoType, FP_PRECISION *_sinTheta, int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, \
	FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, int _numLongtrack_halfSpace, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, \
	int *_numSegsInTrack, struct SUBDOM_SEG_IN_TRACK_SINGLE *_segInfoInTrack, int *_sumNumSegsInRTMTrack, struct RTMFSRINFO **_RTMFSRsInfo, \
	FP_PRECISION *_macroXS_t, FP_PRECISION **_inpXSs_t, FP_PRECISION *_totalWeights, int _FSRVolCorrType, int _numLTsInHalfspace, int *_numLTsSum, \
	int _totNumTracksInOneRTM_HS, int *_indexTrackInRTM_SumAzim_HS);
void Jacobi_MOCSolver_2D_diam_OMP_Parallel_Atomic(int _lay, int _trSet, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, FP_PRECISION *_ltInciAngFlux, \
	FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, int _polarBoundUpLim, int _numCellsInDom, \
	int *_numFSRsInCell, int *_num_LT_subDom, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	FP_PRECISION *_sinTheta, int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, \
	int *_sumNumFSRsBasedGlobCellID, int _numLongtrack_halfSpace, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, \
	struct SUBDOM_SEG_IN_TRACK_SINGLE *_segInfoInTrack, int *_sumNumSegsInRTMTrack, struct RTMFSRINFO **_RTMFSRsInfo, FP_PRECISION *_macroXS_t, \
	FP_PRECISION **_inpXSs_t, FP_PRECISION *_totalWeights, int _FSRVolCorrType, int _numLTsInHalfspace, int *_numLTsSum, int _totNumTracksInOneRTM_HS, \
	int *_indexTrackInRTM_SumAzim_HS);
void Reduce_ScalarFlux_OMP(FP_PRECISION_FLUX *_scalarFlux, FP_PRECISION_FLUX **_scalarFlux_omp, int _numOMPThread, int _totalNumFSRs_subDom, int _numGroups);
void Jacobi_MOCSolver_2D_diam_LongTrack_ParaFor_NonAtomic(int ltHS, int _lay, int _trSet, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom, int *_numFSRsInCell, int *_num_LT_subDom, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, \
	int *_cellRTMGeoType, FP_PRECISION *_sinTheta, int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, \
	FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, int _numLongtrack_halfSpace, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, \
	int *_numSegsInTrack, struct SUBDOM_SEG_IN_TRACK_SINGLE *_segInfoInTrack, int *_sumNumSegsInRTMTrack, struct RTMFSRINFO **_RTMFSRsInfo, \
	FP_PRECISION *_macroXS_t, FP_PRECISION **_inpXSs_t, FP_PRECISION *_totalWeights, \
	int _FSRVolCorrType, int _totNumTracksInOneRTM_HS, int *_indexTrackInRTM_SumAzim_HS);


void Jacobi_MOCSolver_2D_diam_LongTrack_Parallel_NonAtomic_1(int _threadID, int _lay, int _trSet, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom, int *_numFSRsInCell, int *_num_LT_subDom, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, \
	int *_cellRTMGeoType, FP_PRECISION *_sinTheta, FP_PRECISION **_trackFlux, int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, \
	FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, int _numLongtrack_halfSpace, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, \
	int *_numSegsInTrack, struct SUBDOM_SEG_IN_TRACK_SINGLE *_segInfoInTrack, int *_sumNumSegsInRTMTrack, struct RTMFSRINFO **_RTMFSRsInfo, \
	FP_PRECISION *_macroXS_t, FP_PRECISION **_inpXSs_t, FP_PRECISION *_totalWeights, int _FSRVolCorrType, int _numLTsInHalfspace, int *_numLTsSum, \
	int _totNumTracksInOneRTM_HS, int *_indexTrackInRTM_SumAzim_HS);

#endif // OMP_USED
void Jacobi2DMOCSolver_ExpTabRec_LTParaNonAtomic(int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC, int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int _numOMPThreads_comp, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, \
	FP_PRECISION *_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAnfFlux_update, FP_PRECISION *_SinThetaForExp,\
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);

void Jacobi2DMOCSolver_ExpTabRec_BiDir(int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC, int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, \
	FP_PRECISION *_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAnfFlux_update, FP_PRECISION *_SinThetaForExp,\
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);

void Jacobi_MOCSolver_2D_diam_LongTrack_ParaFor_NonAtomic_Package(int _threadID, int ltHS, int _lay, int _trackSet, int _groupBoundLowLim, int _groupBoundUpLim, \
	int _polarBoundLowLim, int _polarBoundUpLim, int _FSRVolCorrType);
void Jacobi_MOCSolver_2D_diam_OMP_ParaFor_NonAtomic_Package(int _setNumThreads, int _lay, int _trSet, int _groupBoundLowLim, int _groupBoundUpLim, \
	int _polarBoundLowLim, int _polarBoundUpLim, int _FSRVolCorrType);
void Jacobi2DMOCSolver_DiamDiff_LTParaNonAtomic_OneDir(int _threadID, int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, FP_PRECISION *_totalWeights_2DMOC, \
	int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int _numOMPThreads_comp, int *_sumNumSegsInRTMTrack_2PI,\
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR,FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, FP_PRECISION *_trackFlux);

void Jacobi2DMOCSolver_ExpTabRec_LTParaNonAtomic_BiDir_FixSourProb_2D(int _threadID, int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC,int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int _numOMPThreads_comp, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, \
	FP_PRECISION *_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp,\
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);

void Jacobi2DMOCSolver_ExpFuncRec_LTParaNonAtomic_BiDir_FixSourProb_2D(int _threadID, int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC, int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int _numOMPThreads_comp, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, \
	FP_PRECISION *_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp, \
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);

void ErrorCheck_Jacobi_MOCSolver_2D_expTab(int _procID, int _probType, int _threadID,int _numGroups, int _layBoundLowLim, \
    int _layBoundUpLim,  int _groupBoundLowLim, int _groupBoundUpLim, int _polarBoundLowLim, int _polarBoundUpLim, \
	int _numCellsInDom_Layer, int *_cellRTMGeoType, int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, \
	int *_sumNumFSRsBasedGlobCellID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, int _totNumTracksInOneRTM_2PI, \
	int *_indexTrackInRTM_SumAzim_2PI, int _numOMPThreads_comp, int *_sumNumSegsInRTMTrack_2PI, FP_PRECISION *_segInfoInTrack_2PI_length, \
	int *_segInfoInTrack_2PI_AntiClockFSR, int _numLongtrack_halfSpace, FP_PRECISION *_SinThetaForExp, FP_PRECISION_GEO *_LayWidth, \
	FP_PRECISION *_CosTheta, int *_LTsIndex_AzimAngID);

// void Jacobi2DMOCSolver_ExpTab_OneDir(int _threadID, int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
// 	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
// 	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
// 	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
// 	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
// 	FP_PRECISION *_totalWeights_2DMOC,int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int _numOMPThreads_comp, int *_sumNumSegsInRTMTrack_2PI, \
// 	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
// 	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, \
// 	FP_PRECISION *_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp,\
// 	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);

void CPU2D_TS_OneTrack_OnDir(int _ind, int _ltIter, int _lay, int _numGroups, int _numPolAng, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, \
    struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION *_boundaryFlux, \
	int *_trackIndexInLTs, int _numCellsInDom_Layer,int _totNumTracksInOneRTM_2PI, int *_cellRTMGeoType, int *_indexTrackInRTM_SumAzim_2PI, \
    int *_numSegsInTrack, FP_PRECISION *_segInfoInTrack_2PI_length, int *_sumNumSegsInRTMTrack_2PI, int *_segInfoInTrack_2PI_AntiClockFSR,\
	int *_sumNumFSRsBasedGlobCellID, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, int *_sumNumFSRsInRTM, int _groupBoundLowLim, \
	int _groupBoundUpLim, FP_PRECISION *_macroXS_t, FP_PRECISION *_source, int _polarBoundLowLim, int _polarBoundUpLim, FP_PRECISION *_expTable, \
	int *_numLongTracksSum, FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_SinThetaForExp, FP_PRECISION *_totalWeights_2DMOC, FP_PRECISION_FLUX *_scalarFlux,\
	FP_PRECISION_GEO *_EffRaySpacing, FP_PRECISION *_CosTheta, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update,\
	FP_PRECISION _oneOverLayWidth, int _numAzimAng, int polOffset);
void Jacobi2DMOCSolver_ExpTab_OneDir(int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC,int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux11, \
	FP_PRECISION *_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp,\
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);
void Jacobi2DMOCSolver_ExpTabRec_BiDir_OneTrack(int _ltIter, int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC,int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, \
	FP_PRECISION *_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp,\
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);
void Jacobi2DMOCSolver_OMP_ExpTabRec_BiDir(int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC,int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, \
	FP_PRECISION **_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp,\
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);

	void Jacobi2DMOCSolver_OMP_ExpTabRec_BiDir_1(int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC,int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, \
	FP_PRECISION **_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp,\
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);

	void Jacobi2DMOCSolver_ExpTabRec_BiDir_OneTrack_1(int _ltIter, int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC,int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux_1, \
	FP_PRECISION *_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp,\
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);


	void Jacobi2DMOCSolver_ExpTabRec_BiDir_11_OMP(int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC,int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION **_trackFlux, \
	FP_PRECISION **_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp,\
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);
	void Jacobi2DMOCSolver_ExpTabRec_BiDir_11(int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC,int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, \
	FP_PRECISION *_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp,\
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset, int _threadID, int _numOMPThreads_comp);



void Jacobi2DMOCSolver_ExpTabRec_BiDir_storeAngF(int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC,int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, \
	FP_PRECISION *_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp,\
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);

void Jacobi2DMOCSolver_ExpTabRec_BiDir_storeAngF_calScattSour(int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC,int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, \
	FP_PRECISION *_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp,\
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);
void Jacobi2DMOCSolver_ExpTabRec_BiDir_storeAngF_calScattSour_1(int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC,int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, \
	FP_PRECISION *_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp,\
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);
void Jacobi2DMOCSolver_ExpTabRec_BiDir_storeAngF_calScattSour_2(int _lay, int _numGroups, int _groupBoundLowLim, int _groupBoundUpLim, \
	FP_PRECISION *_ltInciAngFlux, FP_PRECISION *_boundaryFlux, FP_PRECISION *_source, int _numAzimAng, int _numPolAng, int _polarBoundLowLim, \
	int _polarBoundUpLim, int _numCellsInDom_Layer, int *_numLongTracksSum, struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_cellRTMGeoType, \
	int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, FP_PRECISION_FLUX *_scalarFlux, int *_sumNumFSRsBasedGlobCellID, \
	int _numLongtrack_2PI, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, int *_numSegsInTrack, FP_PRECISION *_macroXS_t, FP_PRECISION *_totalWeights, \
	FP_PRECISION *_totalWeights_2DMOC,int _totNumTracksInOneRTM_2PI, int *_indexTrackInRTM_SumAzim_2PI, int *_sumNumSegsInRTMTrack_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_AntiClockFSR, \
	int *_sumNumFSRsInRTM, int _numLongtrack_halfSpace, int *_num_LT_subDom, FP_PRECISION *_expTable, FP_PRECISION *_trackFlux, \
	FP_PRECISION *_expTempStore, FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAngFlux_update, FP_PRECISION *_SinThetaForExp,\
	FP_PRECISION _oneOverLayWidth, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_EffRaySpacing, int polOffset);

#endif