
#ifndef MOC_TRACK_H
#define MOC_TRACK_H

#include "MOC_Boundary.h"
#include "MOC_Cell.h"
#include "MOC_CommonMacro.h"
#include "MOC_Core.h"
#include "MOC_Pin_Cell.h"
#include "MOC_Point.h"
#include "MOC_Surface.h"

#ifdef OMP_USED
  #include <omp.h>
#endif // OMP_USED

struct SEG_IN_FSR {
    FP_PRECISION_GEO vol_corr_factor;//volume correction factor
	FP_PRECISION_GEO totalSegLength_by_spacing; // sumation of rechtangular areas in a fsr
    int num_seg;
	FP_PRECISION_GEO *seg_length;
};

/**
 * FSR information in the ray tracing module
 */
struct RTMFSRINFO {
    /** the area of this FSR */
    FP_PRECISION_GEO fsr_area;
    // struct SEG_IN_FSR *seg_in_fsr;
	struct SEG_IN_FSR **seg_in_fsr;
};

struct SUBDOM_SEG_IN_TRACK {
	FP_PRECISION_GEO length;
	int FSR_ID;
};

struct SUBDOM_SEG_IN_TRACK_SINGLE {
	FP_PRECISION length;
	int FSR_ID;
};

struct SUBDOM_LONGTRACK_INFO {
	int next_azim_angle_id;
	//int subDom_id_curr;
	int next_SubDom_id;
	int next_LongTrack_id_SubDom;
	//int sp_cell_id;
	//int sp_track_id;
	FP_PRECISION _boundBeta;
};
struct SUBDOM_TRACK_IN_LONGTRACK {
	int cell_ID;
	int track_ID;
	//int track_ID_back;
};

    extern int **NumTracks;
    extern int **NumTracksX;
    extern int **NumTracksY;
	extern FP_PRECISION_GEO **DxEff;
	extern FP_PRECISION_GEO **DyEff;
    extern FP_PRECISION_GEO **EffRaySpacing;
    extern int NumRTMGeoType;
    extern struct PIN_CELL *CellBasedRTMGeo;
    extern FP_PRECISION_GEO **EffAzimAng;
	extern int **NumLongTracks;
	extern int **NumLongTracks_SubDomX;
	extern int **NumLongTracks_SubDomY;
    extern struct RTMFSRINFO **RTModFSRsInfo_HS, **RTModFSRsInfo_2PI;
	extern int *NumFSRsInCell;
	extern M_INT32 TotalNumFSRs_globDom, TotalNumLTs_SubDom;
	extern M_INT32 TotalNumFSRs_subDom, TotalNumFSRs_perLayer;
	extern M_UINT64 TotalNumSeg;
	// extern FP_PRECISION_GEO **FSRVolume;
	extern int MaxNumTraksInRTM;
	extern int MaxNumSegsInTracks;
	extern int ****Index_MOC_TrackInRTM_HS, ****Index_MOC_TrackInRTM_Total, *TotalNumTracksInRTM_HS, *TotalNumTracksInRTM_Total;
	extern int *TotNumTracksInOneRTM_HS, *TotNumTracksInOneRTM_2PI,**IndexTrackInRTM_SumAzim_HS, **IndexTrackInRTM_SumAzim_2PI;
	extern struct POINT ***RTMStartP, ***RTMEndP;
    extern int MaxNumTracksInLTs;
    extern int *NumLTs_HS, *NumLTs_2PI/*,*SumNumLTs_Total_Azim*/;
    extern int **NumTracksInLTs_HS, **NumTracksInLTs_Total;
    extern struct SUBDOM_TRACK_IN_LONGTRACK **TracksInfoInLTs_HS, **TracksInfoInLTs_2PI;
	// int **TracksInfoInLTs_HS;
    extern struct SUBDOM_SEG_IN_TRACK **SegInfoInTrack_HS, **SegInfoInTrack_Total;
	extern struct SUBDOM_SEG_IN_TRACK_SINGLE **SegInfoInTrack_SINGLE_HS, **SegInfoInTrack_SINGLE_2PI;
	extern FP_PRECISION **SegInfoInTrack_2PI_length;
	extern int **SegInfoInTrack_2PI_RayWisedFSR, **SegInfoInTrack_2PI_AntiClockFSR, **FsrIdMap_RayWisedToAntiClock;
    extern int **NumSegsInTrack_HS, **NumSegsInTrack_2PI,**NumLTsSum;
    extern struct SUBDOM_LONGTRACK_INFO **LongTracksInfo_Total;
	extern int *NumTracks_HS, *NumTracks_Total,**TrackIndexInLTs_HS,**TrackIndexInLTs_2PI;
	extern int **SumNumSegsInRTMTrack_HS, **SumNumSegsInRTMTrack_2PI;
	extern int *TotNumSegInRTM_HS, *TotNumSegInRTM_Total;
	extern int MaxNumLTs_Azim;
	extern int *NumFSRsInRTM;
	extern int *SumNumFSRsBasedGlobCellID,*NumFSRs_Layer,*SumNumFSRs_Layer;
	extern int **NumSegsInRTM_2PI;

#if defined(OMP_USED)
    extern omp_lock_t *FSR_OMP_lock;
#endif 
	
void TrackGenetor(int _procID,int _numTrackSet, int *_numAzimAng, int *_numPolarAng, FP_PRECISION_GEO *_raySpacingIni, int _subDomID_X, int _subDomID_Y);
void SetRayTracingModule_Geometry();
FP_PRECISION_GEO GetRTModDimX(FP_PRECISION_GEO *_mod_dim);
FP_PRECISION_GEO GetRTModDimY(FP_PRECISION_GEO *_mod_dim);
FP_PRECISION_GEO GetRTModDimZ(FP_PRECISION_GEO *_mod_dim);
void CheckNumAzimAng(int _numTrackSet, int *_numAzimAng);
void CheckTrackSpacing(int _numTrackSet, FP_PRECISION_GEO *_raySpacingIni);
void GenerateTracks(int _numTrackSet, int *_numAzimAng, int *_numPolarAng, FP_PRECISION_GEO *_raySpacingIni, \
	int _numRTMGeoType, struct PIN_CELL *_pinCellRTM);
void InitializeTracks(int _numAzimAng, FP_PRECISION_GEO _raySpacingIni, FP_PRECISION_GEO *_RTMDim, int *_numTracksX, int *_numTracksY, \
    int *_numTracks, FP_PRECISION_GEO *_dxEff,FP_PRECISION_GEO *_dyEff,FP_PRECISION_GEO *_effAzimuAng,FP_PRECISION_GEO *_raySpacingEff);
void CalcStartAndEndPosOfTracks(int _numAzimAng, FP_PRECISION_GEO *_RTMDim, struct POINT **_RTMStartP,struct POINT **_RTMEndP,\
    int *_numTracks, int *_numTracksX, int *_numTracksY, FP_PRECISION_GEO *_dxEff,FP_PRECISION_GEO *_dyEff,FP_PRECISION_GEO *_effAzimuAng);
struct CELL *FindNextCell(FP_PRECISION_GEO _phi,struct POINT *_point,\
                             struct PIN_CELL *_pinCell,\
                             FP_PRECISION_GEO *_mod_dim);
FP_PRECISION_GEO cellMinSurfaceDist(struct POINT *_point,struct CELL *_cell,FP_PRECISION_GEO _phi);
FP_PRECISION_GEO surfaceMinDistance(struct POINT *_point, struct SURFACE_HALFSPACE *_surfHalf, FP_PRECISION_GEO _phi);
struct CELL *FindCell(struct POINT *_point, struct PIN_CELL *_pinCell);
void computeEndPoint(struct POINT *start,struct POINT *end,\
                                     FP_PRECISION_GEO phi, FP_PRECISION_GEO width_x,\
                                     FP_PRECISION_GEO width_y);
int planeIntersection(struct POINT *point,struct PLANE *plane,\
                      FP_PRECISION_GEO angle,struct POINT *points);
int ZCylIntersection(struct POINT *point,struct PARA_CYLINDER *zcyl,\
                     FP_PRECISION_GEO angle,struct POINT *points);
struct CELL *CellContainsPoint(struct POINT *point,struct CELL *cell);
FP_PRECISION_GEO PlaneEevaluatePoint(struct POINT *point,struct PLANE *plane);
FP_PRECISION_GEO ZCylinderEvaluatePoint(struct POINT *point,struct PARA_CYLINDER *zcyl);
FP_PRECISION_GEO distanceToPoint(struct POINT *point1,struct POINT *point2);
void AdjustCoords(struct POINT *point,FP_PRECISION_GEO phi,FP_PRECISION_GEO delta);
int IsPointInModule (struct POINT *_point, FP_PRECISION_GEO *_mod_dim);
void GetRTMFSRsInfo(struct RTMFSRINFO **_RTModFSRsInfo,int _numTrackSet, struct PIN_CELL *_pinCellRTM, \
	int *_numAzimAng, int _numRTMGeoType, int **_numTracks, int **_numSegsInTrack, \
	int ****_index_MOC_TrackInRTM, struct SUBDOM_SEG_IN_TRACK **_seginTrack,int **_sumNumSegsInRTMTrack, FP_PRECISION_GEO **_raySpacingEff, int _spaceID);
FP_PRECISION_GEO CalcFSRArea(struct CELL *_cell,FP_PRECISION_GEO _pinCell_widthX,\
                     FP_PRECISION_GEO _pinCell_widthY);
FP_PRECISION_GEO tri_area (FP_PRECISION_GEO _length,FP_PRECISION_GEO _height);
FP_PRECISION_GEO sector_area (FP_PRECISION_GEO _angle,FP_PRECISION_GEO _radius);
FP_PRECISION_GEO inters_angle(FP_PRECISION_GEO _planeCoeff[2][3]);
int LongTrack_IsEndP_TRSurf(FP_PRECISION_GEO _endPX,FP_PRECISION_GEO _endPY,FP_PRECISION_GEO _pinCellGloX,\
                        FP_PRECISION_GEO _pinCellGloY,FP_PRECISION_GEO *_mod_dim,struct CORE *_core);
int LongTrack_IsEndP_TLSurf(FP_PRECISION_GEO _endPX,FP_PRECISION_GEO _endPY,FP_PRECISION_GEO _pinCellGloX,\
                        FP_PRECISION_GEO _pinCellGloY,FP_PRECISION_GEO *_mod_dim,struct CORE *_core);
enum boundaryType longTtack_getBoundType(FP_PRECISION_GEO _endPX,FP_PRECISION_GEO _endPY,\
                                         FP_PRECISION_GEO _pinCellGloX,FP_PRECISION_GEO _pinCellGloY,\
                                         int _numOutPlane,struct PLANE *_outPlane);
int pinCell_getAsseID(FP_PRECISION_GEO _pinCellGloX,FP_PRECISION_GEO _pinCellGloY,struct CORE *_core);
int pinCell_getID(FP_PRECISION_GEO _pinCellGloX,FP_PRECISION_GEO _pinCellGloY,\
                  struct CORE *_core,int _asseID,int _lattID);
void GetNumLongTrackInSubDom(int _numTrackSet, int **_num_longTracks, int **_num_longTracksX, int **_num_longTracksY, \
	int **_numLongTracksSum,int *_numAzimAng, int _subDomID_X, int _subDomID_Y, int **_numPin_subDomain, int **_numTracksX, int **_numTracksY);
int SubDomLongTrack_IsEndP_TRSurf(FP_PRECISION_GEO _endPX, FP_PRECISION_GEO _endPY, FP_PRECISION_GEO _pinCellGloX, \
	FP_PRECISION_GEO _pinCellGloY, FP_PRECISION_GEO *_mod_dim, FP_PRECISION_GEO *_subDom_boundSurf);
int SubDomLongTrack_IsEndP_TLSurf(FP_PRECISION_GEO _endPX, FP_PRECISION_GEO _endPY, FP_PRECISION_GEO _pinCellGloX, \
	FP_PRECISION_GEO _pinCellGloY, FP_PRECISION_GEO *_mod_dim, FP_PRECISION_GEO *_subDom_boundSurf);
int longTrack_isEndP_BL(FP_PRECISION_GEO _endPX, FP_PRECISION_GEO _endPY, FP_PRECISION_GEO _pinCellGloX, \
	FP_PRECISION_GEO _pinCellGloY, FP_PRECISION_GEO *_mod_dim, struct CORE *_core);
int subDomLongTrack_isEndP_BL(FP_PRECISION_GEO _endPX, FP_PRECISION_GEO _endPY, FP_PRECISION_GEO _pinCellGloX, \
	FP_PRECISION_GEO _pinCellGloY, FP_PRECISION_GEO *_mod_dim, FP_PRECISION_GEO *_subDom_boundSurf);
int longTrack_isEndP_BR(FP_PRECISION_GEO _endPX, FP_PRECISION_GEO _endPY, FP_PRECISION_GEO _pinCellGloX, \
	FP_PRECISION_GEO _pinCellGloY, FP_PRECISION_GEO *_mod_dim, struct CORE *_core);
int subDomLongTrack_isEndP_BR(FP_PRECISION_GEO _endPX, FP_PRECISION_GEO _endPY, FP_PRECISION_GEO _pinCellGloX, \
	FP_PRECISION_GEO _pinCellGloY, FP_PRECISION_GEO *_mod_dim, FP_PRECISION_GEO *_subDom_boundSurf);
FP_PRECISION_GEO get_FSR_volume(int _volumeCorrFactor, FP_PRECISION_GEO **_FSRVolume, struct RTMFSRINFO **_RTMFSRsInfo, \
	int **_cellRTMType, int _lay, int _cell, int _fsr);
void Interface_Geometry();
void output_MOCBoundParCurr(int ProcID, FP_PRECISION_GEO ****_moc_bound_par_curr);
void GetNumTracksInLongTracks_HS(int _numTrackSet, int *_numAzimAng, int _subDomID_X, int _subDomID_Y, \
	FP_PRECISION_GEO *_RTMDim, int *_numSubDomains, int **_numPin_subDomain, struct CORE *_core, int **_numTracksX, \
	int **_numTracksY, struct POINT ***_RTMEndP,int **_numLongTracks_SubDom, int **_numTracks_in_Longtrack_SubDom,\
	int *_numTracksInHalfSpace,int **_trackIndexInLTs);
void GetNumTracksInLongTracks_Total(int _numTrackSet, int *_numAzimAng, int _subDomID_X, int _subDomID_Y, \
	FP_PRECISION_GEO *_RTMDim, int *_numSubDomains, int **_numPin_subDomain, struct CORE *_core, int **_numTracksX, \
	int **_numTracksY, struct POINT ***_RTMEndP, int **_numLongTracks_SubDom, int **_numTracks_in_Longtrack_SubDom, \
	int *_numTracks_Total, int **_trackIndexInLTs);
void LongTracks_PathLink(struct SUBDOM_LONGTRACK_INFO **_longTracksInfo,struct SUBDOM_TRACK_IN_LONGTRACK **_tracksInfoInLTs,\
    int _numTrackSet,int *_numAzimAng, int **_numLongTracks_SubDom, int **_numLongTracks_SubDomX,\
    int **_numLongTracks_SubDomY,int *_numLTsInHalfspace, int _subDomID_X, int _subDomID_Y, FP_PRECISION_GEO *_RTMDim, \
	int *_numSubDomains, int **_numPin_subDomain, struct CORE *_core, int **_numTracks, int **_numTracksX,int **_numTracksY,\
	struct POINT ***_RTMEndP/* , int *_cellRTMGeoType,int ****_index_MOC_TrackInRTM */);
int GetMaxNumTracksInLongTracks(int _numTrackSet, int *_numLongtrack_halfspace, int **_numTracks_in_Longtrack_SubDom);
int GetNumSegsInOneTrack(struct POINT *_startP, struct PIN_CELL *pinCell, \
	FP_PRECISION_GEO *_mod_dim, FP_PRECISION_GEO phi);
void GetNumSegsInTracks_RTM(int _numTrackSet, int _numRTMGeoType, int *_numAzimAng, struct PIN_CELL *_pinCellRTM, int ****_index_MOC_TrackInRTM, \
	FP_PRECISION_GEO *_RTMDim, FP_PRECISION_GEO **_effAzimuAng, int **_numTracks, int **_numSegsInTrack, \
	struct POINT ***_RTMStartP, int _spaceID);
void GetSegsInOneTrack(struct SUBDOM_SEG_IN_TRACK *_seginTrack,struct SUBDOM_SEG_IN_TRACK_SINGLE *_segInfoInTrack_SINGLE,\
    int _sumNumSegsInRTMTrack,struct POINT *_startP, struct PIN_CELL *_pinCellRTM, \
	FP_PRECISION_GEO *_RTMDim, FP_PRECISION_GEO _phi);
void GetSegsInTracks_RTM(struct SUBDOM_SEG_IN_TRACK **_segInfoInTrack, struct SUBDOM_SEG_IN_TRACK_SINGLE **_segInfoInTrack_SINGLE, \
	int _numTrackSet, int _numRTMGeoType, int *_numAzimAng, struct PIN_CELL *_pinCellRTM, int ****_index_MOC_TrackInRTM, int **_numSegsInTrack, \
	struct POINT ***_startP, FP_PRECISION_GEO *_RTMDim, FP_PRECISION_GEO **_effAzimuAng, int **_numTracks, int **_sumNumSegsInRTMTrack, int _spaceID);
int GetMaxNumSegsInTracks(int _numTrackSet, int *_totalNumTracksInRTM, int **_numSegsInTrack);
void intersecP_2Dlines(FP_PRECISION_GEO a1, FP_PRECISION_GEO b1, FP_PRECISION_GEO d1, \
	FP_PRECISION_GEO a2, FP_PRECISION_GEO b2, FP_PRECISION_GEO d2, FP_PRECISION_GEO *_intersP);
FP_PRECISION_GEO get_FSR_volume_azim(int _trackSetID, int _volumeCorrFactor, FP_PRECISION_GEO **_FSRVolume, struct RTMFSRINFO **_RTMFSRsInfo, \
	int **_cellRTMType, int _lay, int _cell, int _fsr, int _azim);
void GetTotalNumTracksInRTM(int _numTrackSet, int _numRTMGeoType, \
	int *_numAzimAng, int **_numTracks, int *_totalNumTracksInRTM, int _spaceID);
struct SUBDOM_SEG_IN_TRACK **AllocaMem_SegInfoInTrack(int _numTrackSet, int *_totalNumTracksInRTM, int **_numSegsInTrack, \
    int **_sumNumSegsInRTMTrack,int *_totNumSegInRTM,char *_parameter);
struct SUBDOM_SEG_IN_TRACK **FreeMem_SegInfoInTrack(struct SUBDOM_SEG_IN_TRACK **_pointer, int _dim1, char *_parameter);
struct SUBDOM_SEG_IN_TRACK_SINGLE **AllocaMem_SegInfoInTrack_SINGLE(int _numTrackSet, int *_totalNumTracksInRTM, int **_numSegsInTrack, \
    int **_sumNumSegsInRTMTrack,int *_totNumSegInRTM,char *_parameter);
struct SUBDOM_SEG_IN_TRACK_SINGLE **FreeMem_SegInfoInTrack_SINGLE(struct SUBDOM_SEG_IN_TRACK_SINGLE **_pointer, int _dim1, char *_parameter);
struct RTMFSRINFO ** AllocMem_RTMFSRsInfo(int _numTrackSet, struct PIN_CELL *_pinCellRTM, \
	int *_numAzimAng, int _numRTMGeoType,int **_numTracks, int **_numSegsInTrack, \
	int ****_index_MOC_TrackInRTM,struct SUBDOM_SEG_IN_TRACK **_seginTrack,int **_sumNumSegsInRTMTrack,char *_parameter, int _spaceID);
int *GetNumLTsInHalfspace(int _numTrackSet,int *_numAzimAng, int **_numLongTracks_SubDom,char *_parameter);
int *GetNumLTsTotal(int _numTrackSet, int *_numAzimAng, int **_numLongTracks_SubDom, char *_parameter);
struct SUBDOM_LONGTRACK_INFO **AllocMem_LongTracksInfo(int _numTrackSet,int *_numLTsInHalfspace, char *_parameter);
struct SUBDOM_LONGTRACK_INFO **FreeMem_LongTracksInfo(struct SUBDOM_LONGTRACK_INFO **_pointer, int _dim1, char *_parameter);
struct SUBDOM_TRACK_IN_LONGTRACK **AllocMem_TracksInfoInLTs(int _numTrackSet,int *_numTracksInHalfSpace, char *_parameter);
struct SUBDOM_TRACK_IN_LONGTRACK **FreeMem_TracksInfoInLTs(struct SUBDOM_TRACK_IN_LONGTRACK **_pointer, int _dim1, char *_parameter);
int GetMaxNumPolarAng(int _numTrackSet, int *_numPolarAng);
int *GetSumNumLTsTotal_Azim(int *_numAzimAng, int **_numLongTracks_SubDom, char *_parameter);
int LongTrack_CheckEndP_glob2D_RTSurf(int _cellIDInSubDom_2D_X, int _cellIDInSubDom_2D_Y, int _subDomOrgPinOffset_2D_X, \
	int _subDomOrgPinOffset_2D_Y, int _numCellsInCore_2D_X, int _numCellsInCore_2D_Y, int _trackID, int _numTracksY);
int LongTrack_CheckEndP_subDom2D_RTSurf(int _cellIDInSubDom_2D_X, int _cellIDInSubDom_2D_Y, int _numPinCells_subDomX, \
	int _numPinCells_subDomY, int _trackID, int _numTracksY);
int LongTrack_CheckEndP_glob2D_LTSurf(int _cellIDInSubDom_2D_X, int _cellIDInSubDom_2D_Y, int _subDomOrgPinOffset_2D_X, \
	int _subDomOrgPinOffset_2D_Y, int _numCellsInCore_2D_X, int _numCellsInCore_2D_Y, int _trackID, int _numTracksY);
int LongTrack_CheckEndP_subDom2D_LTSurf(int _cellIDInSubDom_2D_X, int _cellIDInSubDom_2D_Y, int _numPinCells_subDomX, \
	int _numPinCells_subDomY, int _trackID, int _numTracksY);
int LongTrack_CheckEndP_glob2D_LBSurf(int _cellIDInSubDom_2D_X, int _cellIDInSubDom_2D_Y, int _subDomOrgPinOffset_2D_X, \
	int _subDomOrgPinOffset_2D_Y, int _numCellsInCore_2D_X, int _numCellsInCore_2D_Y, int _trackID, int _numTracksY);
int LongTrack_CheckEndP_subDom2D_LBSurf(int _cellIDInSubDom_2D_X, int _cellIDInSubDom_2D_Y, int _numPinCells_subDomX, \
	int _numPinCells_subDomY, int _trackID, int _numTracksY);
int LongTrack_CheckEndP_glob2D_RBSurf(int _cellIDInSubDom_2D_X, int _cellIDInSubDom_2D_Y, int _subDomOrgPinOffset_2D_X, \
	int _subDomOrgPinOffset_2D_Y, int _numCellsInCore_2D_X, int _numCellsInCore_2D_Y, int _trackID, int _numTracksY);
int LongTrack_CheckEndP_subDom2D_RBSurf(int _cellIDInSubDom_2D_X, int _cellIDInSubDom_2D_Y, int _numPinCells_subDomX, \
	int _numPinCells_subDomY, int _trackID, int _numTracksY);
void LongTracks_PathLink_Modified(struct SUBDOM_LONGTRACK_INFO **_longTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK **_tracksInfoInLTs_HS, \
	struct SUBDOM_TRACK_IN_LONGTRACK **_tracksInfoInLTs_2PI, int _numTrackSet, int *_numAzimAng, int **_numLongTracks_SubDom, \
	int **_numLongTracks_SubDomX, int **_numLongTracks_SubDomY, int *_numLTsInHalfspace, int *_numSubDomains, int **_numPin_subDomain, \
	int **_numTracksX, int **_numTracksY, int _subDomLocID_2D_X, int _subDomLocID_2D_Y, int **_numTracks_in_Longtrack_SubDom, int **_trackIndexInLTs);
void GetNumFSRsInRTM(int *_numFSRsInRTM, int _numTrackSet, int _numRTMGeoType, struct PIN_CELL *_pinCellRTM);
void GetSegInfoInRTM(int _numTrackSet, int _numRTMGeoType, int *_numAzimAng, int *_totNumSegInRTM_Total, \
	FP_PRECISION **_segInfoInTrack_2PI_length, int **_segInfoInTrack_2PI_rayWisedFsrID, \
	int **_segInfoInTrack_2PI_antiClockFsrID, int **_FsrIdMap_RayWisedToAntiClock, \
	struct SUBDOM_SEG_IN_TRACK_SINGLE **_segInfoInTrack_SINGLE_Total, int **_numTracks, int ****_index_MOC_TrackInRTM, \
	int **_numSegsInTrack, int *_numFSRsInRTM, int _totalNumFSRsInRTM, int *_sumNumFSRsInRTM);
void GetFSRsAreaInRTM(FP_PRECISION_GEO *_FSRsAreaInRTM, FP_PRECISION_GEO *_oneOverFSRsAreaInRTM, \
	int _numRTMGeoType, int *_numFSRsInRTM, struct RTMFSRINFO **_RTMFSRsInfo);
void SetRayTracingModule_Width();
int ****GetIndexOfTracksInRTM(int _numTrackSet, int _numRTMGeoType, int *_numAzimAng, int **_numTrack, char *_parameter, int _spaceID);
void GetIndexOfTracksInRTM_1(int _numTrackSet, int *_numAzimAng, int **_numTrack, \
	int *_totNumTracksInOneRTM, int **_indexTrackInRTM_SumAzim, int _spaceID);
void GetIndexOfLongTracks_HS(int **_LTsIndex_AzimAngID, int **_LTsIndex_LTID, int _numTrackSet, int *_numAzimAng, int **_numLongTracksSubDom);
void GetIndexOfLongTracks_Total(int **_LTsIndex_AzimAngID, int **_LTsIndex_LTID, int _numTrackSet, int *_numAzimAng, int **_numLongTracksSubDom);
M_UINT64 GetTotalNumSegs_SubDom(int _MOC_NumPinsInSubDom, int *_CellRTMType_SubDom, int *_NumSegsInRTM_2PI);
void GetNumSegsInRTM(int _numTrackSet, int _numRTMGeoType, int *_numAzimAng, struct PIN_CELL *_pinCellRTM, \
	FP_PRECISION_GEO *_RTMDim, FP_PRECISION_GEO **_effAzimuAng, int **_numTracks, struct POINT ***_RTMStartP);
void SetFSRArea_SubDom(FP_PRECISION_GEO *_FSRArea_SubDom, FP_PRECISION_GEO *_OneOverFSRArea_SubDom, \
	FP_PRECISION_GEO *_FSRVol_SubDom, FP_PRECISION_GEO *_WidthLayersInCore, int _numLayersZ, int _numCellsInDom, \
	int *_numFSRsInCell, struct RTMFSRINFO **_RTModFSRsInfo, int *_cellRTMType, int *_sumNumFSRsBasedGlobCellID);
void SetFSRVol_SubDom(FP_PRECISION_GEO *_FSRArea_SubDom, FP_PRECISION_GEO *_FSRVol_SubDom, \
    FP_PRECISION_GEO *_WidthLayersInCore, int _numLayersZ,int _numCellsInDom, int *_numFSRsInCell,\
    int *_sumNumFSRsBasedGlobCellID);
void GetNumTracksTotal(int _NumTrackSet, int *_NumTracks_Total, int *_TotNumTracksInOneRTM_2PI, int _MOC_NumPinsInSubDom_layer);
void SetFSRsXSsID_subDomain(struct CORE *_core,int _subDomID_row, int _subDomID_column);
struct SEG_IN_FSR **FreeSegInFSR(struct SEG_IN_FSR **_seg_in_fsr, int _numTrackSet, int *_numAzimAng,int _spaceID, char *_parameter);
struct RTMFSRINFO ** FreeMem_RTMFSRsInfo(struct RTMFSRINFO **_RTModFSRsInfo, int _numRTMGeoType,\
    struct PIN_CELL *_pinCellRTM, int _numTrackSet, int *_numAzimAng, int _spaceID,char *_parameter);
void FreeTrackMem();
int ****FreeIndexOfTracksInRTM(int ****_index_MOC_TrackInRTM,int _numTrackSet, int _numRTMGeoType, \
    int *_numAzimAng, int _spaceID, char *_parameter);

void LongTracks_PathLink_Hex(struct SUBDOM_LONGTRACK_INFO **_longTracksInfo,
                             struct SUBDOM_TRACK_IN_LONGTRACK **_tracksInfoInLTs_HS,
                             struct SUBDOM_TRACK_IN_LONGTRACK **_tracksInfoInLTs_2PI,
                             int _numTrackSet,
                             int *_numAzimAng,
                             int **_numLongTracks_SubDom,
                             int **_numLongTracks_SubDomX,
                             int **_numLongTracks_SubDomY,
                             int *_numLTsInHalfspace,
                             int *_numSubDomains,
                             int **_numPin_subDomain,
                             int **_numTracksX,
                             int **_numTracksY,
                             int _subDomLocID_2D_X,
                             int _subDomLocID_2D_Y,
                             int **_numTracks_in_Longtrack_SubDom,
                             int **_trackIndexInLTs,
                             int *_nextAzimID_TrackFile,
                             int *_nextTrackID_TrackFile);

#endif // TRACK_H
