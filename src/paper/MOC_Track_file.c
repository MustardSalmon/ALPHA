/********************************************************************************/
/* @ File: MOC_Track_file.c
/*   introduction: functions for reading pre-generated tracks and
/*                constructing the downstream track/FSR indexing data
/* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*   1st modified:
/********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "MOC_AllocAndFreeMem.h"
#include "MOC_Assembly.h"
#include "MOC_Boundary.h"
#include "MOC_Cell.h"
#include "MOC_CommonMacro.h"
#include "MOC_Core.h"
#include "MOC_Geometry.h"
#include "MOC_Input.h"
#include "MOC_Lattice.h"
#include "MOC_Macro_XSs.h"
#include "MOC_NTE_Solver.h"
#include "MOC_Output.h"
#include "MOC_Pin_Cell.h"
#include "MOC_Point.h"
#include "MOC_Quadrature_Set.h"
#include "MOC_Surface.h"
#include "MOC_Track.h"
#include "MOC_Visualization.h"
#ifdef MPI_USED
    #include "mpi.h"
#endif // MPI_USED
#ifdef OMP_USED
  #include <omp.h>
#endif // OMP_USED

/* Dynamic track generation is outside the publication package.  The main
 * solver still contains the legacy call site behind InputTrackFile == 0, so
 * keep a fail-fast compatibility symbol for that unsupported mode. */
void TrackGenetor(int _procID, int _numTrackSet, int *_numAzimAng,
                  int *_numPolarAng, FP_PRECISION_GEO *_raySpacingIni,
                  int _subDomID_X, int _subDomID_Y)
{
    (void)_procID;
    (void)_numTrackSet;
    (void)_numAzimAng;
    (void)_numPolarAng;
    (void)_raySpacingIni;
    (void)_subDomID_X;
    (void)_subDomID_Y;
    fprintf(stderr, "Track generation is not included; use InputTrackFile=1.\n");
    exit(69);
}

	// number of geometry set, different parameter - geometry(mesh),set for multi-level transport acceleration technique
    /** An integer array of the number of Tracks for each azimuthal angle */
    int **NumTracks = NULL;
    /** An integer array of the number of Tracks starting on the x-axis for each azimuthal angle */
    int **NumTracksX = NULL;
    /** An integer array of the number of Tracks starting on the y-axis for each azimuthal angle */
    int **NumTracksY = NULL;
    /** A 2D ragged array of Tracks */
    //struct TRACK ***Tracks = NULL;
    /** Track spacing along x,y-axes, and perpendicular to each Track */
    FP_PRECISION_GEO **DxEff = NULL;
	FP_PRECISION_GEO **DyEff = NULL;
    FP_PRECISION_GEO **EffRaySpacing = NULL;
    /** Effective azimuthal angles with respect to positive x-axis */
    FP_PRECISION_GEO **EffAzimAng = NULL;
    /** number of ray tracing module geometry types, eg fuel _pinCell is different from CR _pinCell */
    int NumRTMGeoType;
    struct PIN_CELL *CellBasedRTMGeo = NULL; // cell based ray tracing module geometry
    struct RTMFSRINFO **RTModFSRsInfo_HS = NULL, **RTModFSRsInfo_2PI = NULL;  // need free
	// FP_PRECISION_GEO **FSRVolume = NULL;
	int **NumLongTracks = NULL;
	int **NumLongTracks_SubDomX = NULL;
	int **NumLongTracks_SubDomY = NULL;
	int *NumFSRsInCell = NULL;  //number of FSRs in cells for the subdomain
	M_INT32 TotalNumFSRs_globDom = 0;
	M_INT32 TotalNumFSRs_subDom = 0;
	M_INT32 TotalNumFSRs_perLayer = 0;
	M_UINT64 TotalNumSeg = 0;
	M_INT32 TotalNumLTs_SubDom = 0;
	FP_PRECISION_GEO ***LongestSegInFSR = NULL;
	int MaxNumTraksInRTM;  //maximum number of tracks in ray tracing module in different azimuthal angles
	int MaxNumSegsInTracks;  // maximum number of segments in tracks
	int *NumLTs_HS = NULL, *NumLTs_2PI = NULL/*, *SumNumLTs_Total_Azim = NULL*/;
	int **NumTracksInLTs_HS = NULL, **NumTracksInLTs_Total = NULL;
	int MaxNumTracksInLTs = 0;
	int ****Index_MOC_TrackInRTM_HS = NULL, ****Index_MOC_TrackInRTM_Total = NULL;
	int *TotNumTracksInOneRTM_HS, *TotNumTracksInOneRTM_2PI,**IndexTrackInRTM_SumAzim_HS, **IndexTrackInRTM_SumAzim_2PI;
	struct POINT ***RTMStartP = NULL, ***RTMEndP = NULL;
	int **NumSegsInTrack_HS = NULL, **NumSegsInTrack_2PI = NULL;
	int *TotalNumTracksInRTM_HS = NULL, *TotalNumTracksInRTM_Total = NULL;
	struct SUBDOM_TRACK_IN_LONGTRACK **TracksInfoInLTs_HS = NULL, **TracksInfoInLTs_2PI = NULL;
	// int **TracksInfoInLTs_HS = NULL;
	struct SUBDOM_SEG_IN_TRACK **SegInfoInTrack_HS = NULL, **SegInfoInTrack_Total = NULL;
	struct SUBDOM_SEG_IN_TRACK_SINGLE **SegInfoInTrack_SINGLE_HS = NULL, **SegInfoInTrack_SINGLE_2PI = NULL;
	FP_PRECISION **SegInfoInTrack_2PI_length = NULL;
	int **SegInfoInTrack_2PI_RayWisedFSR = NULL, **SegInfoInTrack_2PI_AntiClockFSR = NULL, **FsrIdMap_RayWisedToAntiClock = NULL;
	struct SUBDOM_LONGTRACK_INFO **LongTracksInfo_Total = NULL;
	int **NumLTsSum=NULL;
	int *NumTracks_HS=NULL, *NumTracks_Total = NULL;
	int **TrackIndexInLTs_HS=NULL, **TrackIndexInLTs_2PI = NULL;
    int **SumNumSegsInRTMTrack_HS=NULL, **SumNumSegsInRTMTrack_2PI = NULL;
	int *TotNumSegInRTM_HS=NULL, *TotNumSegInRTM_Total = NULL;
	int MaxNumLTs_Azim;
	int *NumFSRsInRTM = NULL;
	int *SumNumFSRsBasedGlobCellID = NULL,*NumFSRs_Layer = NULL,*SumNumFSRs_Layer = NULL;
	int **NumSegsInRTM_2PI = NULL;

#if defined(OMP_USED)
    omp_lock_t *FSR_OMP_lock=NULL;
#endif 

struct SUBDOM_LONGTRACK_INFO **AllocMem_LongTracksInfo(int _numTrackSet,int *_numLTsInHalfspace, char *_parameter)
{
	int /*i,*/_p;
	int _dataNumber = 0;
	FP_PRECISION _dataSizeMB = 0., _pointerSizeMB = 0.;
	struct SUBDOM_LONGTRACK_INFO **_longTracksInfo=NULL;
	_longTracksInfo = (struct SUBDOM_LONGTRACK_INFO **)calloc(_numTrackSet, sizeof(struct SUBDOM_LONGTRACK_INFO *));
	if (_longTracksInfo == NULL) {
		printf("\n [**ERROR**]  calloc in function \"AllocMem_LongTracksInfo\" - %s !",_parameter);
		fflush(stdout);
		exit(69);
	}
	for (_p = 0; _p < _numTrackSet; _p++) {
		_longTracksInfo[_p] = (struct SUBDOM_LONGTRACK_INFO *)calloc(_numLTsInHalfspace[_p]*2, sizeof(struct SUBDOM_LONGTRACK_INFO));
		if (_longTracksInfo[_p] == NULL) {
			printf("\n [**ERROR**]  calloc in function \"AllocMem_LongTracksInfo\" - %s[%d] !",_parameter,_p);
			fflush(stdout);
			exit(69);
		}
		_pointerSizeMB = _pointerSizeMB + (FP_PRECISION)(sizeof(_longTracksInfo[_p]));
		_dataNumber = _dataNumber + _numLTsInHalfspace[_p]*2;
	}

	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(struct SUBDOM_LONGTRACK_INFO));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter, CHAR_NONE) != 0) {
		OutputMemoryFootprint(FP_MemoryFootprint, "struct SUBDOM_LONGTRACK_INFO **", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
	}
	return _longTracksInfo;
}
struct SUBDOM_LONGTRACK_INFO **FreeMem_LongTracksInfo(struct SUBDOM_LONGTRACK_INFO **_pointer, int _dim1, char *_parameter)
{
	int _iter1;
	if (strcmp(_parameter,CHAR_NONE) != 0) {
		OutputFreeMemory(_parameter);
	}

	for (_iter1 = 0; _iter1<_dim1; _iter1++)
		free(_pointer[_iter1]);
	free(_pointer);
	_pointer = NULL;
	return _pointer;
}
/**
 * @ Function: AllocMem_TracksInfoInLTs()
 *   introduction: allocate memory for struct SUBDOM_TRACK_IN_LONGTRACK *** pointer
 * @ Parameter List: _numTrackSet - number of track set
 *                   _numLTsInHalfspace - number of the long tracks in halfspace [0,PI]
 *                   _numTracksInLTs - number of tracks in long tracks
 *                   _parameter - parameter name
 *   Return: struct SUBDOM_TRACK_IN_LONGTRACK *** object
 * @ Created     : 2018/05/15, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified: 2018/05/30, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *                 remove parameters int *_numLTsInHalfspace and int **_numTracksInLTs
 *                 add parameter int *_numTracksInHalfSpace
 *                 decrease the demension of returned parameter from 3 to 2
 */
struct SUBDOM_TRACK_IN_LONGTRACK **AllocMem_TracksInfoInLTs(int _numTrackSet,int *_numTracksInHalfSpace, char *_parameter)
{
	int _p/*,_lt*/, _dataNumber = 0;
	struct SUBDOM_TRACK_IN_LONGTRACK **_tracksInfoInLTs=NULL;
	FP_PRECISION _dataSizeMB = 0.0, _pointerSizeMB = 0.0;
	_tracksInfoInLTs = (struct SUBDOM_TRACK_IN_LONGTRACK **)calloc(_numTrackSet, sizeof(struct SUBDOM_TRACK_IN_LONGTRACK *));
	if (_tracksInfoInLTs == NULL) {
		printf("\n [**ERROR**]  calloc in function \"AllocMem_TracksInfoInLTs\" - %s !",_parameter);
		fflush(stdout);
		exit(69);
	}
	for (_p = 0; _p < _numTrackSet; _p++) {
		_tracksInfoInLTs[_p] = (struct SUBDOM_TRACK_IN_LONGTRACK *)calloc(_numTracksInHalfSpace[_p], sizeof(struct SUBDOM_TRACK_IN_LONGTRACK));
		if (_tracksInfoInLTs[_p] == NULL) {
			printf("\n [**ERROR**]  calloc in function \"AllocMem_TracksInfoInLTs\" - %s[%d] !",_parameter, _p);
			fflush(stdout);
			exit(69);
		}
		_pointerSizeMB += (FP_PRECISION)(sizeof(_tracksInfoInLTs[_p]));
		_dataNumber = _dataNumber + _numTracksInHalfSpace[_p];
	}

	_dataSizeMB = (FP_PRECISION)(_dataNumber)*M_MB_Unit*(FP_PRECISION)(sizeof(struct SUBDOM_TRACK_IN_LONGTRACK));
	_pointerSizeMB *= M_MB_Unit;
	if (strcmp(_parameter, CHAR_NONE) != 0) {
		OutputMemoryFootprint(FP_MemoryFootprint, "struct SUBDOM_TRACK_IN_LONGTRACK**", _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
	}
	return _tracksInfoInLTs;
}
struct SUBDOM_TRACK_IN_LONGTRACK **FreeMem_TracksInfoInLTs(struct SUBDOM_TRACK_IN_LONGTRACK **_pointer, int _dim1, char *_parameter)
{
	int _iter1;
	if (strcmp(_parameter,CHAR_NONE) != 0) {
		OutputFreeMemory(_parameter);
	}

	for (_iter1 = 0; _iter1<_dim1; _iter1++)
		free(_pointer[_iter1]);
	free(_pointer);
	_pointer = NULL;
	return _pointer;
}
 /**
 * @ Function: LongTracks_PathLink()
 *   introduction: get number of tracks in LongTracks for azimuthal angle [0,PI]
 * @ Parameter List: _numTrackSet - number of track set
 *                   _numAzimAng - number of the azimuthal angle in [0,2PI]
 *                   _subDomID_X - subdomain column id
 *                   _subDomID_Y - subdomain row id
 *                   _index_MOC_TrackInRTM - track index, locate a track in 1D array by [TrackSet_ID][RTMGeoType_ID][AzimAng_ID][RayID_inCurrentRTM]
 *                   _numSegsInTrack - number of segments in tracks
 *                   _startP - start position of tracks in ray tracing module
 *                   _RTMDim - ray tracing module dimension, [0]-x_dimension, [1]-y_dimension,[2]-z_dimension

 *   Return: void
 * @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified: 2018/05/15, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *                 remove memory allocate part out of the function
 *                 add formal parameter struct SUBDOM_LONGTRACK_INFO ***_longTracksInfo and struct SUBDOM_TRACK_IN_LONGTRACK ***_tracksInfoInLTs
 *   2nd modified: 2018/05/30, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *                 dicrease the dimension of parameter struct SUBDOM_TRACK_IN_LONGTRACK ***_tracksInfoInLTs from 3 to 2
 *   3rd modified: 2018/05/31, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *                 change parameter _tracksInfoInLTs type from struct SUBDOM_TRACK_IN_LONGTRACK ** to int **
 *                 add parameter int *_cellRTMGeoType,int ****_index_MOC_TrackInRTM

 */

void LongTracks_PathLink_Modified(struct SUBDOM_LONGTRACK_INFO **_longTracksInfo, struct SUBDOM_TRACK_IN_LONGTRACK **_tracksInfoInLTs_HS, \
	struct SUBDOM_TRACK_IN_LONGTRACK **_tracksInfoInLTs_2PI, int _numTrackSet, int *_numAzimAng, int **_numLongTracks_SubDom, \
	int **_numLongTracks_SubDomX, int **_numLongTracks_SubDomY, int *_numLTsInHalfspace, int *_numSubDomains, int **_numPin_subDomain, \
	int **_numTracksX, int **_numTracksY, int _subDomLocID_2D_X, int _subDomLocID_2D_Y, int **_numTracks_in_Longtrack_SubDom, int **_trackIndexInLTs)
{
	int i, j, _p, _num, _halfNumAzimAng, _trackId, _numPinCells_subDom_X = MOCNumPinsInSubDom_X, _numTracks_subDomX/*, _count*/;
	int _curr_subDomID,/* _pinSeqID_inSubDom,*/ _lt_add, ***_subDom_id_curr = NULL, _isEndP = 1, _isSubEndP = 1;
	int _azimu_ang_id, _nextLTid = 0, _next_subDom_LTid = 0, _next_subDomID = 0;
	int _numPinCells_subDomX, _numPinCells_subDomY, _cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y/*, _globPinCellID_2D*/, _pinSeqID_inSubDom_2D;
	int _next_cellIDInSubDom_2D_X, _next_cellIDInSubDom_2D_Y, _next_subDom_trackID, _next_subDomID_X, _next_subDomID_Y;
	int _lt_count, _numTracks;
	int _count11;

	_numPinCells_subDomX = MOCNumPinsInSubDom_X;
	_numPinCells_subDomY = MOCNumPinsInSubDom_Y;
	_curr_subDomID = GetSubDomSequenceID(_subDomLocID_2D_Y, _subDomLocID_2D_X, _numSubDomains);
	/** calloc for _subDom_id_curr */
	_subDom_id_curr = (int ***)calloc(_numTrackSet, sizeof(int **));
	for (_p = 0; _p < _numTrackSet; _p++) {
		_subDom_id_curr[_p] = (int **)calloc(_numAzimAng[_p], sizeof(int *));
		for (i = 0; i < _numAzimAng[_p]; i++) {
			_subDom_id_curr[_p][i] = (int *)calloc(_numLongTracks_SubDom[_p][i], sizeof(int));
		}
	}

	//path link
	for (_p = 0; _p < _numTrackSet; _p++) {
		_lt_add = 0;
		_halfNumAzimAng = _numAzimAng[_p] / 2;
		_num = 0;

		_lt_count = 0;
		//_numTracks_Total[_p] = 0;
		_count11 = 0;
		_trackIndexInLTs[_p][_lt_count] = 0;
		/** set long Tracks information for azimuthal angles in [0,2*pi] */
		for (i = 0; i<_numAzimAng[_p]; i++) {
			_numTracks_subDomX = _numPinCells_subDom_X * _numTracksX[_p][i];
			if (i<(_halfNumAzimAng / 2)) {
				for (j = 0; j<_numLongTracks_SubDom[_p][i]; j++) {
					_numTracks = 0;
					_subDom_id_curr[_p][i][j] = _curr_subDomID;
					/** calculate the first _pinCell's coordinates in LongTracks[i][j] (Core coordinate system)*/
					if (j<_numTracks_subDomX) {
						_trackId = j % _numTracksX[_p][i];
						_cellIDInSubDom_2D_X = _numPinCells_subDomX - 1 - j / _numTracksX[_p][i];
						_cellIDInSubDom_2D_Y = _numPinCells_subDomY-1;
					}
					else {
						_trackId = (j - _numTracks_subDomX) % _numTracksY[_p][i] + _numTracksX[_p][i];
						_cellIDInSubDom_2D_X = 0;
						_cellIDInSubDom_2D_Y = _numPinCells_subDomY - 1 - (j - _numTracks_subDomX) / _numTracksY[_p][i];
					}
					_pinSeqID_inSubDom_2D = _cellIDInSubDom_2D_Y * _numPinCells_subDomX + _cellIDInSubDom_2D_X;
					_tracksInfoInLTs_HS[_p][_num].cell_ID = _pinSeqID_inSubDom_2D;
					_tracksInfoInLTs_HS[_p][_num].track_ID = _trackId;
#ifdef GeoTwoPIRegion
					_tracksInfoInLTs_2PI[_p][_num].cell_ID = _pinSeqID_inSubDom_2D;
					_tracksInfoInLTs_2PI[_p][_num].track_ID = _trackId;
#endif // GeoTwoPIRegion

					_num++;
					_numTracks++;
					/** if this _pinCell is at the right bottom corner with the track end _point is on the right side,
					*  or this _pinCell is at the left top corner with the track end _point is on the top side.
					*  Then, set isEndP=1
					*/
					_isEndP = LongTrack_CheckEndP_glob2D_RTSurf(_cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y, SubDomOrgPinOffset_2D_X, \
						SubDomOrgPinOffset_2D_Y, NumCellsInCore_2D_X, NumCellsInCore_2D_Y, _trackId, _numTracksY[_p][i]);
					_isSubEndP = LongTrack_CheckEndP_subDom2D_RTSurf(_cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y, _numPinCells_subDomX, \
						_numPinCells_subDomY, _trackId, _numTracksY[_p][i]);
					/*track reaches the boundary of subdomain but does not reach the boundary of global domain*/
					if (_isSubEndP == 1 && _isEndP == 0) {
						/*track end point at the right surface of ray tracing module*/
						if (_trackId < _numTracksY[_p][i]) {
							_next_subDomID_X = _subDomLocID_2D_X + 1;
							_next_subDomID_Y = _subDomLocID_2D_Y;
							_next_cellIDInSubDom_2D_Y = _cellIDInSubDom_2D_Y;
							_next_subDom_trackID = _trackId + _numTracksX[_p][i];
							_next_subDom_LTid = _numTracksX[_p][i] * (_numPin_subDomain[0][_next_subDomID_X] - 1) + \
								_numTracksY[_p][i] * (_numPin_subDomain[1][_next_subDomID_Y] - (_next_cellIDInSubDom_2D_Y + 1)) + _next_subDom_trackID;
						}
						/*track end point at the top surface of ray tracing module*/
						else {
							_next_subDomID_X = _subDomLocID_2D_X;
							_next_subDomID_Y = _subDomLocID_2D_Y-1;
							_next_cellIDInSubDom_2D_X = _cellIDInSubDom_2D_X;
							_next_subDom_trackID = _trackId - _numTracksY[_p][i];
							_next_subDom_LTid = _numTracksX[_p][i] * (_numPin_subDomain[0][_next_subDomID_X] - (_next_cellIDInSubDom_2D_X + 1)) + \
								_next_subDom_trackID;
						}
						_next_subDomID = GetSubDomSequenceID(_next_subDomID_Y, _next_subDomID_X, _numSubDomains);
						_longTracksInfo[_p][_lt_add].next_azim_angle_id = i;
						_longTracksInfo[_p][_lt_add].next_SubDom_id = _next_subDomID;
						_longTracksInfo[_p][_lt_add].next_LongTrack_id_SubDom = _next_subDom_LTid;
					}
				    /*track reaches both the boundary of subdomain and global domain*/
					else if (_isSubEndP == 1 && _isEndP == 1) {
						_longTracksInfo[_p][_lt_add].next_SubDom_id = _curr_subDomID;
					}
					else {}
					/*successively get the track's information along LongTracks[i][j],
					 *  until the Core boundary */
					while (_isEndP==0 && _isSubEndP==0) {
						/** find next _pinCell */
						/* pin cell right surface */
						if (_trackId < _numTracksY[_p][i]) {
							_cellIDInSubDom_2D_X = _cellIDInSubDom_2D_X + 1;
							_cellIDInSubDom_2D_Y = _cellIDInSubDom_2D_Y;
							_trackId = _trackId + _numTracksX[_p][i];
						}
						else {
							_cellIDInSubDom_2D_X = _cellIDInSubDom_2D_X;
							_cellIDInSubDom_2D_Y = _cellIDInSubDom_2D_Y - 1;
							_trackId = _trackId - _numTracksY[_p][i];
						}
						_pinSeqID_inSubDom_2D = _cellIDInSubDom_2D_Y * _numPinCells_subDomX + _cellIDInSubDom_2D_X;

						_tracksInfoInLTs_HS[_p][_num].cell_ID = _pinSeqID_inSubDom_2D;
						_tracksInfoInLTs_HS[_p][_num].track_ID = _trackId;
#ifdef GeoTwoPIRegion
						_tracksInfoInLTs_2PI[_p][_num].cell_ID = _pinSeqID_inSubDom_2D;
						_tracksInfoInLTs_2PI[_p][_num].track_ID = _trackId;
#endif // GeoTwoPIRegion

						_num++;
						_numTracks++;
						/** find the end _point of the this track */
						/** check if the track end _point on the outmost surface of Core */
						_isEndP = LongTrack_CheckEndP_glob2D_RTSurf(_cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y, SubDomOrgPinOffset_2D_X, \
							SubDomOrgPinOffset_2D_Y, NumCellsInCore_2D_X, NumCellsInCore_2D_Y, _trackId, _numTracksY[_p][i]);
						_isSubEndP = LongTrack_CheckEndP_subDom2D_RTSurf(_cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y, _numPinCells_subDomX, \
							_numPinCells_subDomY, _trackId, _numTracksY[_p][i]);
						if (_isSubEndP == 1 && _isEndP == 0) {
							/*track end point at the right surface of ray tracing module*/
							if (_trackId < _numTracksY[_p][i]) {
								_next_subDomID_X = _subDomLocID_2D_X + 1;
								_next_subDomID_Y = _subDomLocID_2D_Y;
								_next_cellIDInSubDom_2D_Y = _cellIDInSubDom_2D_Y;
								_next_subDom_trackID = _trackId + _numTracksX[_p][i];
								_next_subDom_LTid = _numTracksX[_p][i] * (_numPin_subDomain[0][_next_subDomID_X] - 1) + \
									_numTracksY[_p][i] * (_numPin_subDomain[1][_next_subDomID_Y] - (_next_cellIDInSubDom_2D_Y + 1)) + _next_subDom_trackID;
							}
							/*track end point at the top surface of ray tracing module*/
							else {
								_next_subDomID_X = _subDomLocID_2D_X;
								_next_subDomID_Y = _subDomLocID_2D_Y - 1;
								_next_cellIDInSubDom_2D_X = _cellIDInSubDom_2D_X;
								_next_subDom_trackID = _trackId - _numTracksY[_p][i];
								_next_subDom_LTid = _numTracksX[_p][i] * (_numPin_subDomain[0][_next_subDomID_X] - (_next_cellIDInSubDom_2D_X + 1)) + \
									_next_subDom_trackID;
							}
							_next_subDomID = GetSubDomSequenceID(_next_subDomID_Y, _next_subDomID_X, _numSubDomains);
							_longTracksInfo[_p][_lt_add].next_azim_angle_id = i;
							_longTracksInfo[_p][_lt_add].next_SubDom_id = _next_subDomID;
							_longTracksInfo[_p][_lt_add].next_LongTrack_id_SubDom = _next_subDom_LTid;
						}
						/*track reaches both the boundary of subdomain and global domain*/
						else if (_isSubEndP == 1 && _isEndP == 1) {
							_longTracksInfo[_p][_lt_add].next_SubDom_id = _curr_subDomID;
						}
						else {}
					}
					_lt_add++;

					_numTracks_in_Longtrack_SubDom[_p][_lt_count] = _numTracks;
					//_numTracks_Total[_p] += _numTracks_in_Longtrack_SubDom[_p][_lt_count];
					_count11 += _numTracks_in_Longtrack_SubDom[_p][_lt_count];
					_trackIndexInLTs[_p][_lt_count + 1] = _count11;
					_lt_count++;
				}
			}
			else if ((i >= _halfNumAzimAng / 2) && \
				(i < _halfNumAzimAng)) {
				for (j = 0; j<_numLongTracks_SubDom[_p][i]; j++) {
					_subDom_id_curr[_p][i][j] = _curr_subDomID;
					_numTracks = 0;
					/** calculate the first _pinCell's coordinates in \
					* LongTracks[i][j] (Core coordinate system)*/
					if (j<_numTracks_subDomX) {
						_trackId = j % _numTracksX[_p][i];
						_cellIDInSubDom_2D_X = j / _numTracksX[_p][i];
						_cellIDInSubDom_2D_Y = _numPinCells_subDomY - 1;
					}
					else {
						_trackId = (j - _numTracks_subDomX) % _numTracksY[_p][i] + _numTracksX[_p][i];
						_cellIDInSubDom_2D_X = _numPinCells_subDomX - 1;
						_cellIDInSubDom_2D_Y = _numPinCells_subDomY - 1 - (j - _numTracks_subDomX) / _numTracksY[_p][i];
					}
					_pinSeqID_inSubDom_2D = _cellIDInSubDom_2D_Y * _numPinCells_subDomX + _cellIDInSubDom_2D_X;
					_tracksInfoInLTs_HS[_p][_num].cell_ID = _pinSeqID_inSubDom_2D;
					_tracksInfoInLTs_HS[_p][_num].track_ID = _trackId;
#ifdef GeoTwoPIRegion
					_tracksInfoInLTs_2PI[_p][_num].cell_ID = _pinSeqID_inSubDom_2D;
					_tracksInfoInLTs_2PI[_p][_num].track_ID = _trackId;
#endif // GeoTwoPIRegion
					_num++;
					_numTracks++;
					/** if this _pinCell is at the right bottom corner with the track end _point is on the right side,
					*  or this _pinCell is at the left top corner with the track end _point is on the top side.
					*  Then, set isEndP=0
					*/
					_isEndP = LongTrack_CheckEndP_glob2D_LTSurf(_cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y, SubDomOrgPinOffset_2D_X, \
						SubDomOrgPinOffset_2D_Y, NumCellsInCore_2D_X, NumCellsInCore_2D_Y, _trackId, _numTracksY[_p][i]);
					_isSubEndP = LongTrack_CheckEndP_subDom2D_LTSurf(_cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y, _numPinCells_subDomX, \
						_numPinCells_subDomY, _trackId, _numTracksY[_p][i]);
					/*track reaches the boundary of subdomian but does not reach the boundary of global domain*/
					if (_isSubEndP == 1 && _isEndP == 0) {
						/*track end point at the left surface of ray tracing module*/
						if (_trackId < _numTracksY[_p][i]) {
							_next_subDomID_X = _subDomLocID_2D_X - 1;
							_next_subDomID_Y = _subDomLocID_2D_Y;
							_next_cellIDInSubDom_2D_Y = _cellIDInSubDom_2D_Y;
							_next_subDom_trackID = _trackId + _numTracksX[_p][i];
							_next_subDom_LTid = _numTracksX[_p][i] * (_numPin_subDomain[0][_next_subDomID_X] - 1) + \
								_numTracksY[_p][i] * (_numPin_subDomain[1][_next_subDomID_Y] - (_next_cellIDInSubDom_2D_Y + 1)) + _next_subDom_trackID;
						}
						/*track end point at the top surface of ray tracing module*/
						else {
							_next_subDomID_X = _subDomLocID_2D_X;
							_next_subDomID_Y = _subDomLocID_2D_Y - 1;
							_next_cellIDInSubDom_2D_X = _cellIDInSubDom_2D_X;
							_next_subDom_trackID = _trackId - _numTracksY[_p][i];
							_next_subDom_LTid = _numTracksX[_p][i] * _next_cellIDInSubDom_2D_X + _next_subDom_trackID;
						}
						_next_subDomID = GetSubDomSequenceID(_next_subDomID_Y, _next_subDomID_X, _numSubDomains);
						_longTracksInfo[_p][_lt_add].next_azim_angle_id = i;
						_longTracksInfo[_p][_lt_add].next_SubDom_id = _next_subDomID;
						_longTracksInfo[_p][_lt_add].next_LongTrack_id_SubDom = _next_subDom_LTid;
					}
					/*track reaches both the boundary of subdomain and global domain*/
					else if (_isSubEndP == 1 && _isEndP == 1) {
						_longTracksInfo[_p][_lt_add].next_SubDom_id = _curr_subDomID;
					}
					else {}

					/** successively get the track's information along LongTracks[i][j], until the Core boundary */
					while (_isEndP == 0 && _isSubEndP == 0) {
						/** find next _pinCell */
						/* pin cell left surface */
						if (_trackId < _numTracksY[_p][i]) {
							_cellIDInSubDom_2D_X = _cellIDInSubDom_2D_X - 1;
							_cellIDInSubDom_2D_Y = _cellIDInSubDom_2D_Y;
							_trackId = _trackId + _numTracksX[_p][i];
						}
						else {
							_cellIDInSubDom_2D_X = _cellIDInSubDom_2D_X;
							_cellIDInSubDom_2D_Y = _cellIDInSubDom_2D_Y - 1;
							_trackId = _trackId - _numTracksY[_p][i];
						}
						_pinSeqID_inSubDom_2D = _cellIDInSubDom_2D_Y * _numPinCells_subDomX + _cellIDInSubDom_2D_X;

						_tracksInfoInLTs_HS[_p][_num].cell_ID = _pinSeqID_inSubDom_2D;
						_tracksInfoInLTs_HS[_p][_num].track_ID = _trackId;
#ifdef GeoTwoPIRegion
						_tracksInfoInLTs_2PI[_p][_num].cell_ID = _pinSeqID_inSubDom_2D;
						_tracksInfoInLTs_2PI[_p][_num].track_ID = _trackId;
#endif // GeoTwoPIRegion

						_num++;
						_numTracks++;
						/** find the end _point of the this track */
						/** check if the track end _point on the outmost surface of Core */
						_isEndP = LongTrack_CheckEndP_glob2D_LTSurf(_cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y, SubDomOrgPinOffset_2D_X, \
							SubDomOrgPinOffset_2D_Y, NumCellsInCore_2D_X, NumCellsInCore_2D_Y, _trackId, _numTracksY[_p][i]);
						_isSubEndP = LongTrack_CheckEndP_subDom2D_LTSurf(_cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y, _numPinCells_subDomX, \

							_numPinCells_subDomY, _trackId, _numTracksY[_p][i]);
						if (_isSubEndP == 1 && _isEndP == 0) {
							/*track end point at the left surface of ray tracing module*/
							if (_trackId < _numTracksY[_p][i]) {
								_next_subDomID_X = _subDomLocID_2D_X - 1;
								_next_subDomID_Y = _subDomLocID_2D_Y;
								_next_cellIDInSubDom_2D_Y = _cellIDInSubDom_2D_Y;
								_next_subDom_trackID = _trackId + _numTracksX[_p][i];
								_next_subDom_LTid = _numTracksX[_p][i] * (_numPin_subDomain[0][_next_subDomID_X] - 1) + \
									_numTracksY[_p][i] * (_numPin_subDomain[1][_next_subDomID_Y] - (_next_cellIDInSubDom_2D_Y + 1)) + _next_subDom_trackID;
							}
							/*track end point at the top surface of ray tracing module*/
							else {
								_next_subDomID_X = _subDomLocID_2D_X;
								_next_subDomID_Y = _subDomLocID_2D_Y - 1;
								_next_cellIDInSubDom_2D_X = _cellIDInSubDom_2D_X;
								_next_subDom_trackID = _trackId - _numTracksY[_p][i];
								_next_subDom_LTid = _numTracksX[_p][i] * _next_cellIDInSubDom_2D_X + _next_subDom_trackID;
							}
							_next_subDomID = GetSubDomSequenceID(_next_subDomID_Y, _next_subDomID_X, _numSubDomains);
							_longTracksInfo[_p][_lt_add].next_azim_angle_id = i;
							_longTracksInfo[_p][_lt_add].next_SubDom_id = _next_subDomID;
							_longTracksInfo[_p][_lt_add].next_LongTrack_id_SubDom = _next_subDom_LTid;
						}
						/*track reaches both the boundary of subdomain and global domain*/
						else if (_isSubEndP == 1 && _isEndP == 1) {
							_longTracksInfo[_p][_lt_add].next_SubDom_id = _curr_subDomID;
						}
						else {}
					}
					_lt_add++;

					_numTracks_in_Longtrack_SubDom[_p][_lt_count] = _numTracks;
					//_numTracks_Total[_p] += _numTracks_in_Longtrack_SubDom[_p][_lt_count];
					_count11 += _numTracks_in_Longtrack_SubDom[_p][_lt_count];
					_trackIndexInLTs[_p][_lt_count + 1] = _count11;
					_lt_count++;
				}
			}
			else if ((i >= _halfNumAzimAng) && \
				(i < 3 * _halfNumAzimAng / 2)) {
				for (j = 0; j<_numLongTracks_SubDom[_p][i]; j++) {
					_numTracks = 0;
					_subDom_id_curr[_p][i][j] = _curr_subDomID;
					/** calculate the first _pinCell's coordinates in \
					* LongTracks[i][j] (Core coordinate system)*/
					if (j<_numTracks_subDomX) {
						_trackId = j % _numTracksX[_p][i];
						_cellIDInSubDom_2D_X = j / _numTracksX[_p][i];
						_cellIDInSubDom_2D_Y = 0;
					}
					else {
						_trackId = (j - _numTracks_subDomX) % _numTracksY[_p][i] + _numTracksX[_p][i];
						_cellIDInSubDom_2D_X = _numPinCells_subDomX - 1;
						_cellIDInSubDom_2D_Y = (j - _numTracks_subDomX) / _numTracksY[_p][i];
					}
					_pinSeqID_inSubDom_2D = _cellIDInSubDom_2D_Y * _numPinCells_subDomX + _cellIDInSubDom_2D_X;
#ifdef GeoTwoPIRegion
					_tracksInfoInLTs_2PI[_p][_num].cell_ID = _pinSeqID_inSubDom_2D;
					_tracksInfoInLTs_2PI[_p][_num].track_ID = _trackId;
#endif // GeoTwoPIRegion
					_num++;
					_numTracks++;
					/** if this _pinCell is at the right bottom corner with the track end _point is on the right side,
					*  or this _pinCell is at the left top corner with the track end _point is on the top side.
					*  Then, set isEndP=0
					*/
					_isEndP = LongTrack_CheckEndP_glob2D_LBSurf(_cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y, SubDomOrgPinOffset_2D_X, \
						SubDomOrgPinOffset_2D_Y, NumCellsInCore_2D_X, NumCellsInCore_2D_Y, _trackId, _numTracksY[_p][i]);
					_isSubEndP = LongTrack_CheckEndP_subDom2D_LBSurf(_cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y, _numPinCells_subDomX, \
						_numPinCells_subDomY, _trackId, _numTracksY[_p][i]);
					/*track reaches the boundary of subdomian but does not reach the boundary of global domain*/
					if (_isSubEndP == 1 && _isEndP == 0) {
						/*track end point at the left surface of ray tracing module*/
						if (_trackId < _numTracksY[_p][i]) {
							_next_subDomID_X = _subDomLocID_2D_X - 1;
							_next_subDomID_Y = _subDomLocID_2D_Y;
							_next_cellIDInSubDom_2D_Y = _cellIDInSubDom_2D_Y;
							_next_subDom_trackID = _trackId + _numTracksX[_p][i];
							_next_subDom_LTid = _numTracksX[_p][i] * (_numPin_subDomain[0][_next_subDomID_X] - 1) + \
								_numTracksY[_p][i] * _next_cellIDInSubDom_2D_Y + _next_subDom_trackID;
						}
						/*track end point at the bottom surface of ray tracing module*/
						else {
							_next_subDomID_X = _subDomLocID_2D_X;
							_next_subDomID_Y = _subDomLocID_2D_Y + 1;
							_next_cellIDInSubDom_2D_X = _cellIDInSubDom_2D_X;
							_next_subDom_trackID = _trackId - _numTracksY[_p][i];
							_next_subDom_LTid = _numTracksX[_p][i] * _next_cellIDInSubDom_2D_X + _next_subDom_trackID;
						}
						_next_subDomID = GetSubDomSequenceID(_next_subDomID_Y, _next_subDomID_X, _numSubDomains);
						_longTracksInfo[_p][_lt_add].next_azim_angle_id = i;
						_longTracksInfo[_p][_lt_add].next_SubDom_id = _next_subDomID;
						_longTracksInfo[_p][_lt_add].next_LongTrack_id_SubDom = _next_subDom_LTid;
					}
					/*track reaches both the boundary of subdomain and global domain*/
					else if (_isSubEndP == 1 && _isEndP == 1) {
						_longTracksInfo[_p][_lt_add].next_SubDom_id = _curr_subDomID;
					}
					else {}

					/** successively get the track's information along LongTracks[i][j],
					*  until the Core boundary */
					while (_isEndP == 0 && _isSubEndP == 0) {
						/** find next _pinCell */
						/* pin cell left surface */
						if (_trackId < _numTracksY[_p][i]) {
							_cellIDInSubDom_2D_X = _cellIDInSubDom_2D_X - 1;
							_cellIDInSubDom_2D_Y = _cellIDInSubDom_2D_Y;
							_trackId = _trackId + _numTracksX[_p][i];
						}
						else {
							_cellIDInSubDom_2D_X = _cellIDInSubDom_2D_X;
							_cellIDInSubDom_2D_Y = _cellIDInSubDom_2D_Y + 1;
							_trackId = _trackId - _numTracksY[_p][i];
						}
						_pinSeqID_inSubDom_2D = _cellIDInSubDom_2D_Y * _numPinCells_subDomX + _cellIDInSubDom_2D_X;
#ifdef GeoTwoPIRegion
						_tracksInfoInLTs_2PI[_p][_num].cell_ID = _pinSeqID_inSubDom_2D;
						_tracksInfoInLTs_2PI[_p][_num].track_ID = _trackId;
#endif // GeoTwoPIRegion
						_num++;
						_numTracks++;
						/** find the end _point of the this track */
						/** check if the track end _point on the outmost surface of Core */
						_isEndP = LongTrack_CheckEndP_glob2D_LBSurf(_cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y, SubDomOrgPinOffset_2D_X, \
							SubDomOrgPinOffset_2D_Y, NumCellsInCore_2D_X, NumCellsInCore_2D_Y, _trackId, _numTracksY[_p][i]);
						_isSubEndP = LongTrack_CheckEndP_subDom2D_LBSurf(_cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y, _numPinCells_subDomX, \
							_numPinCells_subDomY, _trackId, _numTracksY[_p][i]);
						if (_isSubEndP == 1 && _isEndP == 0) {
							/*track end point at the left surface of ray tracing module*/
							if (_trackId < _numTracksY[_p][i]) {
								_next_subDomID_X = _subDomLocID_2D_X - 1;
								_next_subDomID_Y = _subDomLocID_2D_Y;
								_next_cellIDInSubDom_2D_Y = _cellIDInSubDom_2D_Y;
								_next_subDom_trackID = _trackId + _numTracksX[_p][i];
								_next_subDom_LTid = _numTracksX[_p][i] * (_numPin_subDomain[0][_next_subDomID_X] - 1) + \
									_numTracksY[_p][i] * _next_cellIDInSubDom_2D_Y + _next_subDom_trackID;
							}
							/*track end point at the bottom surface of ray tracing module*/
							else {
								_next_subDomID_X = _subDomLocID_2D_X;
								_next_subDomID_Y = _subDomLocID_2D_Y + 1;
								_next_cellIDInSubDom_2D_X = _cellIDInSubDom_2D_X;
								_next_subDom_trackID = _trackId - _numTracksY[_p][i];
								_next_subDom_LTid = _numTracksX[_p][i] * _next_cellIDInSubDom_2D_X + _next_subDom_trackID;
							}
							_next_subDomID = GetSubDomSequenceID(_next_subDomID_Y, _next_subDomID_X, _numSubDomains);
							_longTracksInfo[_p][_lt_add].next_azim_angle_id = i;
							_longTracksInfo[_p][_lt_add].next_SubDom_id = _next_subDomID;
							_longTracksInfo[_p][_lt_add].next_LongTrack_id_SubDom = _next_subDom_LTid;
						}
						/*track reaches both the boundary of subdomain and global domain*/
						else if (_isSubEndP == 1 && _isEndP == 1) {
							_longTracksInfo[_p][_lt_add].next_SubDom_id = _curr_subDomID;
						}
						else {}
					}
					_lt_add++;

					_numTracks_in_Longtrack_SubDom[_p][_lt_count] = _numTracks;
					//_numTracks_Total[_p] += _numTracks_in_Longtrack_SubDom[_p][_lt_count];
					_count11 += _numTracks_in_Longtrack_SubDom[_p][_lt_count];
					_trackIndexInLTs[_p][_lt_count + 1] = _count11;
					_lt_count++;
				}
			}
			else if ((i >= 3 * _halfNumAzimAng / 2) && \
				(i < _numAzimAng[_p])) {
				for (j = 0; j<_numLongTracks_SubDom[_p][i]; j++) {
					_subDom_id_curr[_p][i][j] = _curr_subDomID;
					_numTracks = 0;
					/** calculate the first _pinCell's coordinates in \
					* LongTracks[i][j] (Core coordinate system)*/
					if (j<_numTracks_subDomX) {
						_trackId = j % _numTracksX[_p][i];
						_cellIDInSubDom_2D_X = _numPinCells_subDomX - 1 - j / _numTracksX[_p][i];
						_cellIDInSubDom_2D_Y = 0;
					}
					else {
						_trackId = (j - _numTracks_subDomX) % _numTracksY[_p][i] + _numTracksX[_p][i];
						_cellIDInSubDom_2D_X = 0;
						_cellIDInSubDom_2D_Y = (j - _numTracks_subDomX) / _numTracksY[_p][i];
					}
					_pinSeqID_inSubDom_2D = _cellIDInSubDom_2D_Y * _numPinCells_subDomX + _cellIDInSubDom_2D_X;
#ifdef GeoTwoPIRegion
					_tracksInfoInLTs_2PI[_p][_num].cell_ID = _pinSeqID_inSubDom_2D;
					_tracksInfoInLTs_2PI[_p][_num].track_ID = _trackId;
#endif // GeoTwoPIRegion
					_num++;
					_numTracks++;
					/** if this _pinCell is at the right bottom corner with the track end _point is on the right side,
					*  or this _pinCell is at the left top corner with the track end _point is on the top side.
					*  Then, set isEndP=0
					*/
					_isEndP = LongTrack_CheckEndP_glob2D_RBSurf(_cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y, SubDomOrgPinOffset_2D_X, \
						SubDomOrgPinOffset_2D_Y, NumCellsInCore_2D_X, NumCellsInCore_2D_Y, _trackId, _numTracksY[_p][i]);
					_isSubEndP = LongTrack_CheckEndP_subDom2D_RBSurf(_cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y, _numPinCells_subDomX, \
						_numPinCells_subDomY, _trackId, _numTracksY[_p][i]);
					/*track reaches the boundary of subdomian but does not reach the boundary of global domain*/
					if (_isSubEndP == 1 && _isEndP == 0) {
						/*track end point at the right surface of ray tracing module*/
						if (_trackId < _numTracksY[_p][i]) {
							_next_subDomID_X = _subDomLocID_2D_X + 1;
							_next_subDomID_Y = _subDomLocID_2D_Y;
							_next_cellIDInSubDom_2D_Y = _cellIDInSubDom_2D_Y;
							_next_subDom_trackID = _trackId + _numTracksX[_p][i];
							_next_subDom_LTid = _numTracksX[_p][i] * (_numPin_subDomain[0][_next_subDomID_X] - 1) + \
								_numTracksY[_p][i] * _next_cellIDInSubDom_2D_Y + _next_subDom_trackID;
						}
						/*track end point at the bottom surface of ray tracing module*/
						else {
							_next_subDomID_X = _subDomLocID_2D_X;
							_next_subDomID_Y = _subDomLocID_2D_Y + 1;
							_next_cellIDInSubDom_2D_X = _cellIDInSubDom_2D_X;
							_next_subDom_trackID = _trackId - _numTracksY[_p][i];
							_next_subDom_LTid = _numTracksX[_p][i] * (_numPin_subDomain[0][_next_subDomID_X] - (_next_cellIDInSubDom_2D_X + 1)) + _next_subDom_trackID;
						}
						_next_subDomID = GetSubDomSequenceID(_next_subDomID_Y, _next_subDomID_X, _numSubDomains);
						_longTracksInfo[_p][_lt_add].next_azim_angle_id = i;
						_longTracksInfo[_p][_lt_add].next_SubDom_id = _next_subDomID;
						_longTracksInfo[_p][_lt_add].next_LongTrack_id_SubDom = _next_subDom_LTid;
					}
					/*track reaches both the boundary of subdomain and global domain*/
					else if (_isSubEndP == 1 && _isEndP == 1) {
						_longTracksInfo[_p][_lt_add].next_SubDom_id = _curr_subDomID;
					}
					else {}
					/** successively get the track's information along LongTracks[i][j],
					*  until the Core boundary */
					while (_isEndP == 0 && _isSubEndP == 0) {
						/** find next _pinCell */
						/* pin cell right surface */
						if (_trackId < _numTracksY[_p][i]) {
							_cellIDInSubDom_2D_X = _cellIDInSubDom_2D_X + 1;
							_cellIDInSubDom_2D_Y = _cellIDInSubDom_2D_Y;
							_trackId = _trackId + _numTracksX[_p][i];
						}
						else {
							_cellIDInSubDom_2D_X = _cellIDInSubDom_2D_X;
							_cellIDInSubDom_2D_Y = _cellIDInSubDom_2D_Y + 1;
							_trackId = _trackId - _numTracksY[_p][i];
						}
						_pinSeqID_inSubDom_2D = _cellIDInSubDom_2D_Y * _numPinCells_subDomX + _cellIDInSubDom_2D_X;
#ifdef GeoTwoPIRegion
						_tracksInfoInLTs_2PI[_p][_num].cell_ID = _pinSeqID_inSubDom_2D;
						_tracksInfoInLTs_2PI[_p][_num].track_ID = _trackId;
#endif // GeoTwoPIRegion
						_num++;
						_numTracks++;
						/** find the end _point of the this track */
						/** check if the track end _point on the outmost surface of Core */
						_isEndP = LongTrack_CheckEndP_glob2D_RBSurf(_cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y, SubDomOrgPinOffset_2D_X, \
							SubDomOrgPinOffset_2D_Y, NumCellsInCore_2D_X, NumCellsInCore_2D_Y, _trackId, _numTracksY[_p][i]);
						_isSubEndP = LongTrack_CheckEndP_subDom2D_RBSurf(_cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y, _numPinCells_subDomX, \
							_numPinCells_subDomY, _trackId, _numTracksY[_p][i]);
						if (_isSubEndP == 1 && _isEndP == 0) {
							/*track end point at the right surface of ray tracing module*/
							if (_trackId < _numTracksY[_p][i]) {
								_next_subDomID_X = _subDomLocID_2D_X + 1;
								_next_subDomID_Y = _subDomLocID_2D_Y;
								_next_cellIDInSubDom_2D_Y = _cellIDInSubDom_2D_Y;
								_next_subDom_trackID = _trackId + _numTracksX[_p][i];
								_next_subDom_LTid = _numTracksX[_p][i] * (_numPin_subDomain[0][_next_subDomID_X] - 1) + \
									_numTracksY[_p][i] * _next_cellIDInSubDom_2D_Y + _next_subDom_trackID;
							}
							/*track end point at the bottom surface of ray tracing module*/
							else {
								_next_subDomID_X = _subDomLocID_2D_X;
								_next_subDomID_Y = _subDomLocID_2D_Y + 1;
								_next_cellIDInSubDom_2D_X = _cellIDInSubDom_2D_X;
								_next_subDom_trackID = _trackId - _numTracksY[_p][i];
								_next_subDom_LTid = _numTracksX[_p][i] * (_numPin_subDomain[0][_next_subDomID_X] - (_next_cellIDInSubDom_2D_X + 1)) + _next_subDom_trackID;
							}
							_next_subDomID = GetSubDomSequenceID(_next_subDomID_Y, _next_subDomID_X, _numSubDomains);
							_longTracksInfo[_p][_lt_add].next_azim_angle_id = i;
							_longTracksInfo[_p][_lt_add].next_SubDom_id = _next_subDomID;
							_longTracksInfo[_p][_lt_add].next_LongTrack_id_SubDom = _next_subDom_LTid;
						}
						/*track reaches both the boundary of subdomain and global domain*/
						else if (_isSubEndP == 1 && _isEndP == 1) {
							_longTracksInfo[_p][_lt_add].next_SubDom_id = _curr_subDomID;
						}
						else {}
					}
					_lt_add++;

					_numTracks_in_Longtrack_SubDom[_p][_lt_count] = _numTracks;
					//_numTracks_Total[_p] += _numTracks_in_Longtrack_SubDom[_p][_lt_count];
					_count11 += _numTracks_in_Longtrack_SubDom[_p][_lt_count];
					_trackIndexInLTs[_p][_lt_count + 1] = _count11;
					_lt_count++;
				}
			}
			else {}
		}
		/** set the connection relationships of Tracks within a reflective tracking */
		/** set next azimuthal angle id and next track id for LongTracks[i][j] */
		_lt_add = 0;
		for (i = 0; i<_numAzimAng[_p]; i++) {

			for (j = 0; j<_numLongTracks_SubDom[_p][i]; j++) {
				_azimu_ang_id = i;
				if ((_azimu_ang_id < (_halfNumAzimAng)) && \
					(_subDom_id_curr[_p][i][j] == _longTracksInfo[_p][_lt_add].next_SubDom_id)) {
					if (j < _numLongTracks_SubDomY[_p][i]) {
						_longTracksInfo[_p][_lt_add].next_azim_angle_id = _halfNumAzimAng - 1 - _azimu_ang_id;
						_nextLTid = j + _numLongTracks_SubDomX[_p][i];
					}
					else {
						_longTracksInfo[_p][_lt_add].next_azim_angle_id = _numAzimAng[_p] - 1 - _azimu_ang_id;
						_nextLTid = j - _numLongTracks_SubDomY[_p][i];
					}
					_longTracksInfo[_p][_lt_add].next_LongTrack_id_SubDom = _nextLTid;
				}
				else if ((_azimu_ang_id >= (_halfNumAzimAng)) && \
					(_subDom_id_curr[_p][i][j] == _longTracksInfo[_p][_lt_add].next_SubDom_id)) {
					if (j < _numLongTracks_SubDomY[_p][i]) {
						_longTracksInfo[_p][_lt_add].next_azim_angle_id = _halfNumAzimAng * 3 - 1 - _azimu_ang_id;
						_nextLTid = j + _numLongTracks_SubDomX[_p][i];
					}
					else {
						_longTracksInfo[_p][_lt_add].next_azim_angle_id = _numAzimAng[_p] - 1 - _azimu_ang_id;
						_nextLTid = j - _numLongTracks_SubDomY[_p][i];
					}
					_longTracksInfo[_p][_lt_add].next_LongTrack_id_SubDom = _nextLTid;
				}
				else {}
				_lt_add++;
			}
		}

		// set boundary beta
		_lt_add = 0;
		for (i = 0; i < _numAzimAng[_p]; i++) {
			for (j = 0; j < _numLongTracks_SubDom[_p][i]; j++) {
				_longTracksInfo[_p][_lt_add]._boundBeta = RadialBoundary(i, _longTracksInfo[_p][_lt_add].next_azim_angle_id, _numAzimAng[_p]);
				_lt_add++;
			}
		}
	}
	_subDom_id_curr = FreeMem_int_IPP(_subDom_id_curr, _numTrackSet, _numAzimAng, (char *)(CHAR_NONE));

//    for (_p = 0; _p < _numTrackSet; _p++){
//        for (i=0;i<NumTracks_Total[_p];i++){
//            printf("\n_tracksInfoInLTs_2PI[%d][%d].cell_ID = %d", _p, i, _tracksInfoInLTs_2PI[_p][i].cell_ID);
//            printf("\t_tracksInfoInLTs_2PI[%d][%d].track_ID = %d", _p, i, _tracksInfoInLTs_2PI[_p][i].track_ID);
//
//        }
//    }

//    for(_p = 0; _p < _numTrackSet; _p++){
//        for (i=0; i<NumLTs_2PI[_p];i++){
//            printf("\n_longTracksInfo[%d][%d].next_SubDom_id = %d", _p, i, _longTracksInfo[_p][i].next_SubDom_id);
//            printf("  next_azim_angle_id = %d",  _longTracksInfo[_p][i].next_azim_angle_id);
//            printf("  next_LongTrack_id_SubDom = %d", _longTracksInfo[_p][i].next_LongTrack_id_SubDom);
//        }
//    }
//
//    for(_p = 0; _p < _numTrackSet; _p++){
//        for (i=0; i<NumLTs_2PI[_p]+1;i++){
//            printf("\n_trackIndexInLTs[%d][%d]=%d", _p, i, _trackIndexInLTs[_p][i]);
//        }
//    }
//    for(i=0;i<NumLTs_2PI[0];i++) printf("\n NumTracksInLTs_Total[%d]=%d", i, NumTracksInLTs_Total[0][i]);
//    exit(998);
}

int LongTrack_IsEndP_TRSurf(FP_PRECISION_GEO _endPX,FP_PRECISION_GEO _endPY,FP_PRECISION_GEO _pinCellGloX,\
                        FP_PRECISION_GEO _pinCellGloY,FP_PRECISION_GEO *_RTMDim,struct CORE *_core)
{
    int _isEndP=0;
    if ((fabs(_endPX-0.5*GetRTModDimX(_RTMDim))<ON_SURFACE &&\
        fabs(_pinCellGloX-(0.5*CoreLengthX-0.5*GetRTModDimX(_RTMDim)))<ON_SURFACE) ||\
        (fabs(_endPY-0.5*GetRTModDimY(_RTMDim))<ON_SURFACE &&\
        fabs(_pinCellGloY-(0.5*CoreLengthY-0.5*GetRTModDimY(_RTMDim)))<ON_SURFACE)) {
        _isEndP=1;
    }
    return _isEndP;
}

/**
* @ Function: LongTrack_CheckEndP_glob2D_RTSurf()
*   Introduction: check if the current pin cell is the end of a long track in global (core) 2D geometry, at the top and right surface
* @ Parameter List: _cellIDInSubDom_2D_X - pin cell id in x direction in sudomain, 2D
*                   _cellIDInSubDom_2D_Y - pin cell id in y direction in sudomain, 2D
*                   _subDomOrgPinOffset_2D_X - pin cell (at the left top cornel of subdomain) id in x direction in global domain, 2D
*                   _subDomOrgPinOffset_2D_Y - pin cell (at the left top cornel of subdomain) id in y direction in global domain, 2D
*                   _numCellsInCore_2D_X - number of cells in core in x direction, 2D
*                   _numCellsInCore_2D_Y - number of cells in core in y direction, 2D
*                   _trackID - track id in current ray tracing module
*                   _numTracksY - number of tracks in current ray tracing module in y direction, 2D
*   Return: 1 - end
*           0 - not the end
* @ Created     : 2019/01/02, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int LongTrack_CheckEndP_glob2D_RTSurf( int _cellIDInSubDom_2D_X,int _cellIDInSubDom_2D_Y, int _subDomOrgPinOffset_2D_X, \
	int _subDomOrgPinOffset_2D_Y, int _numCellsInCore_2D_X, int _numCellsInCore_2D_Y, int _trackID, int _numTracksY)
{
	int _isEndP = 0, _globCellID_2D_X, _globCellID_2D_Y;
	_globCellID_2D_X = _subDomOrgPinOffset_2D_X + _cellIDInSubDom_2D_X;
	_globCellID_2D_Y = _subDomOrgPinOffset_2D_Y + _cellIDInSubDom_2D_Y;
	if (((_trackID < _numTracksY) && (_globCellID_2D_X == (_numCellsInCore_2D_X - 1))) || \
		((_globCellID_2D_Y == 0) && (_trackID >= _numTracksY))) {
		_isEndP = 1;
	}
	return _isEndP;
}
/**
* @ Function: LongTrack_CheckEndP_subDom2D_RTSurf()
*   Introduction: check if the current pin cell is the end of a long track in subdomian 2D geometry, at the top and right surface
* @ Parameter List: _cellIDInSubDom_2D_X - pin cell id in x direction in sudomain, 2D
*                   _cellIDInSubDom_2D_Y - pin cell id in y direction in sudomain, 2D
*                   _numPinCells_subDomX - number of cells in subdomain in x direction, 2D
*                   _numPinCells_subDomY - number of cells in subdomain in Y direction, 2D
*                   _trackID - track id in current ray tracing module
*                   _numTracksY - number of tracks in current ray tracing module in y direction, 2D
*   Return: 1 - end
*           0 - not the end
* @ Created     : 2019/01/02, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int LongTrack_CheckEndP_subDom2D_RTSurf(int _cellIDInSubDom_2D_X, int _cellIDInSubDom_2D_Y, int _numPinCells_subDomX, \
	int _numPinCells_subDomY,int _trackID,int _numTracksY)
{
	int _isEndP = 0;
	if (((_trackID < _numTracksY) && (_cellIDInSubDom_2D_X == (_numPinCells_subDomX - 1))) || \
		((_cellIDInSubDom_2D_Y == 0) && (_trackID >= _numTracksY))) {
		_isEndP = 1;
	}
	return _isEndP;
}
/**
* @ Function: LongTrack_CheckEndP_glob2D_LTSurf()
*   Introduction: check if the current pin cell is the end of a long track in global (core) 2D geometry, at the top and left surface
* @ Parameter List: _cellIDInSubDom_2D_X - pin cell id in x direction in sudomain, 2D
*                   _cellIDInSubDom_2D_Y - pin cell id in y direction in sudomain, 2D
*                   _subDomOrgPinOffset_2D_X - pin cell (at the left top cornel of subdomain) id in x direction in global domain, 2D
*                   _subDomOrgPinOffset_2D_Y - pin cell (at the left top cornel of subdomain) id in y direction in global domain, 2D
*                   _numCellsInCore_2D_X - number of cells in core in x direction, 2D
*                   _numCellsInCore_2D_Y - number of cells in core in y direction, 2D
*                   _trackID - track id in current ray tracing module
*                   _numTracksY - number of tracks in current ray tracing module in y direction, 2D
*   Return: 1 - end
*           0 - not the end
* @ Created     : 2019/01/02, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/

int LongTrack_CheckEndP_glob2D_LTSurf(int _cellIDInSubDom_2D_X, int _cellIDInSubDom_2D_Y, int _subDomOrgPinOffset_2D_X, \
	int _subDomOrgPinOffset_2D_Y, int _numCellsInCore_2D_X, int _numCellsInCore_2D_Y, int _trackID, int _numTracksY)
{
	int _isEndP = 0, _globCellID_2D_X, _globCellID_2D_Y;
	_globCellID_2D_X = _subDomOrgPinOffset_2D_X + _cellIDInSubDom_2D_X;
	_globCellID_2D_Y = _subDomOrgPinOffset_2D_Y + _cellIDInSubDom_2D_Y;
	if (((_trackID < _numTracksY) && (_globCellID_2D_X == 0)) || \
		((_globCellID_2D_Y == 0) && (_trackID >= _numTracksY))) {
		_isEndP = 1;
	}
	return _isEndP;
}
/**
* @ Function: LongTrack_CheckEndP_subDom2D_LTSurf()
*   Introduction: check if the current pin cell is the end of a long track in subdomian 2D geometry, at the top and left surface
* @ Parameter List: _cellIDInSubDom_2D_X - pin cell id in x direction in sudomain, 2D
*                   _cellIDInSubDom_2D_Y - pin cell id in y direction in sudomain, 2D
*                   _numPinCells_subDomX - number of cells in subdomain in x direction, 2D
*                   _numPinCells_subDomY - number of cells in subdomain in Y direction, 2D
*                   _trackID - track id in current ray tracing module
*                   _numTracksY - number of tracks in current ray tracing module in y direction, 2D
*   Return: 1 - end
*           0 - not the end
* @ Created     : 2019/01/02, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int LongTrack_CheckEndP_subDom2D_LTSurf(int _cellIDInSubDom_2D_X, int _cellIDInSubDom_2D_Y, int _numPinCells_subDomX, \
	int _numPinCells_subDomY, int _trackID, int _numTracksY)
{
	int _isEndP = 0;
	if (((_trackID < _numTracksY) && (_cellIDInSubDom_2D_X == 0)) || \
		((_cellIDInSubDom_2D_Y == 0) && (_trackID >= _numTracksY))) {
		_isEndP = 1;
	}
	return _isEndP;
}

/**
* @ Function: LongTrack_CheckEndP_glob2D_LBSurf()
*   Introduction: check if the current pin cell is the end of a long track in global (core) 2D geometry, at the bottom and left surface
* @ Parameter List: _cellIDInSubDom_2D_X - pin cell id in x direction in sudomain, 2D
*                   _cellIDInSubDom_2D_Y - pin cell id in y direction in sudomain, 2D
*                   _subDomOrgPinOffset_2D_X - pin cell (at the left top cornel of subdomain) id in x direction in global domain, 2D
*                   _subDomOrgPinOffset_2D_Y - pin cell (at the left top cornel of subdomain) id in y direction in global domain, 2D
*                   _numCellsInCore_2D_X - number of cells in core in x direction, 2D
*                   _numCellsInCore_2D_Y - number of cells in core in y direction, 2D
*                   _trackID - track id in current ray tracing module
*                   _numTracksY - number of tracks in current ray tracing module in y direction, 2D
*   Return: 1 - end
*           0 - not the end
* @ Created     : 2019/01/02, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int LongTrack_CheckEndP_glob2D_LBSurf(int _cellIDInSubDom_2D_X, int _cellIDInSubDom_2D_Y, int _subDomOrgPinOffset_2D_X, \
	int _subDomOrgPinOffset_2D_Y, int _numCellsInCore_2D_X, int _numCellsInCore_2D_Y, int _trackID, int _numTracksY)
{
	int _isEndP = 0, _globCellID_2D_X, _globCellID_2D_Y;
	_globCellID_2D_X = _subDomOrgPinOffset_2D_X + _cellIDInSubDom_2D_X;
	_globCellID_2D_Y = _subDomOrgPinOffset_2D_Y + _cellIDInSubDom_2D_Y;
	if (((_trackID < _numTracksY) && (_globCellID_2D_X == 0)) || \
		((_globCellID_2D_Y == (_numCellsInCore_2D_Y - 1)) && (_trackID >= _numTracksY))) {
		_isEndP = 1;
	}
	return _isEndP;
}

/**
* @ Function: LongTrack_CheckEndP_subDom2D_LBSurf()
*   Introduction: check if the current pin cell is the end of a long track in subdomian 2D geometry, at the bottom and left surface
* @ Parameter List: _cellIDInSubDom_2D_X - pin cell id in x direction in sudomain, 2D
*                   _cellIDInSubDom_2D_Y - pin cell id in y direction in sudomain, 2D
*                   _numPinCells_subDomX - number of cells in subdomain in x direction, 2D
*                   _numPinCells_subDomY - number of cells in subdomain in Y direction, 2D
*                   _trackID - track id in current ray tracing module
*                   _numTracksY - number of tracks in current ray tracing module in y direction, 2D
*   Return: 1 - end
*           0 - not the end
* @ Created     : 2019/01/02, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int LongTrack_CheckEndP_subDom2D_LBSurf(int _cellIDInSubDom_2D_X, int _cellIDInSubDom_2D_Y, int _numPinCells_subDomX, \
	int _numPinCells_subDomY, int _trackID, int _numTracksY)
{
	int _isEndP = 0;
	if (((_trackID < _numTracksY) && (_cellIDInSubDom_2D_X == 0)) || \
		((_cellIDInSubDom_2D_Y == (_numPinCells_subDomY - 1)) && (_trackID >= _numTracksY))) {
		_isEndP = 1;
	}
	return _isEndP;
}

/**
* @ Function: LongTrack_CheckEndP_glob2D_RBSurf()
*   Introduction: check if the current pin cell is the end of a long track in global (core) 2D geometry, at the bottom and right surface
* @ Parameter List: _cellIDInSubDom_2D_X - pin cell id in x direction in sudomain, 2D
*                   _cellIDInSubDom_2D_Y - pin cell id in y direction in sudomain, 2D
*                   _subDomOrgPinOffset_2D_X - pin cell (at the left top cornel of subdomain) id in x direction in global domain, 2D
*                   _subDomOrgPinOffset_2D_Y - pin cell (at the left top cornel of subdomain) id in y direction in global domain, 2D
*                   _numCellsInCore_2D_X - number of cells in core in x direction, 2D
*                   _numCellsInCore_2D_Y - number of cells in core in y direction, 2D
*                   _trackID - track id in current ray tracing module
*                   _numTracksY - number of tracks in current ray tracing module in y direction, 2D
*   Return: 1 - end
*           0 - not the end
* @ Created     : 2019/01/02, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int LongTrack_CheckEndP_glob2D_RBSurf(int _cellIDInSubDom_2D_X, int _cellIDInSubDom_2D_Y, int _subDomOrgPinOffset_2D_X, \
	int _subDomOrgPinOffset_2D_Y, int _numCellsInCore_2D_X, int _numCellsInCore_2D_Y, int _trackID, int _numTracksY)
{
	int _isEndP = 0, _globCellID_2D_X, _globCellID_2D_Y;
	_globCellID_2D_X = _subDomOrgPinOffset_2D_X + _cellIDInSubDom_2D_X;
	_globCellID_2D_Y = _subDomOrgPinOffset_2D_Y + _cellIDInSubDom_2D_Y;
	if (((_trackID < _numTracksY) && (_globCellID_2D_X == (_numCellsInCore_2D_X - 1))) || \
		((_globCellID_2D_Y == (_numCellsInCore_2D_Y - 1)) && (_trackID >= _numTracksY))) {
		_isEndP = 1;
	}
	return _isEndP;
}

/**
* @ Function: LongTrack_CheckEndP_subDom2D_RBSurf()
*   Introduction: check if the current pin cell is the end of a long track in subdomian 2D geometry, at the bottom and right surface
* @ Parameter List: _cellIDInSubDom_2D_X - pin cell id in x direction in sudomain, 2D
*                   _cellIDInSubDom_2D_Y - pin cell id in y direction in sudomain, 2D
*                   _numPinCells_subDomX - number of cells in subdomain in x direction, 2D
*                   _numPinCells_subDomY - number of cells in subdomain in Y direction, 2D
*                   _trackID - track id in current ray tracing module
*                   _numTracksY - number of tracks in current ray tracing module in y direction, 2D
*   Return: 1 - end
*           0 - not the end
* @ Created     : 2019/01/02, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int LongTrack_CheckEndP_subDom2D_RBSurf(int _cellIDInSubDom_2D_X, int _cellIDInSubDom_2D_Y, int _numPinCells_subDomX, \
	int _numPinCells_subDomY, int _trackID, int _numTracksY)
{
	int _isEndP = 0;
	if (((_trackID < _numTracksY) && (_cellIDInSubDom_2D_X == (_numPinCells_subDomX - 1))) || \
		((_cellIDInSubDom_2D_Y == (_numPinCells_subDomY - 1)) && (_trackID >= _numTracksY))) {
		_isEndP = 1;
	}
	return _isEndP;
}

// int LongTrack_IsEndP_TRSurf(FP_PRECISION_GEO _endPX,FP_PRECISION_GEO _endPY,FP_PRECISION_GEO _pinCellGloX,\
//                         FP_PRECISION_GEO _pinCellGloY,FP_PRECISION_GEO *_RTMDim,struct CORE *_core)
// {
//     int _isEndP=1;
//     if ((fabs(_endPX-0.5*GetRTModDimX(_RTMDim))<ON_SURFACE &&\
//         fabs(_pinCellGloX-(0.5*CoreLengthX-0.5*GetRTModDimX(_RTMDim)))<ON_SURFACE) ||\
//         (fabs(_endPY-0.5*GetRTModDimY(_RTMDim))<ON_SURFACE &&\
//         fabs(_pinCellGloY-(0.5*CoreLengthY-0.5*GetRTModDimY(_RTMDim)))<ON_SURFACE)) {
//         _isEndP=0;
//     }
//     return _isEndP;
// }
/**
* check if the _point is the end of a long track in subdomain
* (on the top or right surface of subdomain)
*/
int SubDomLongTrack_IsEndP_TRSurf(FP_PRECISION_GEO _endPX, FP_PRECISION_GEO _endPY, FP_PRECISION_GEO _pinCellGloX, \
	FP_PRECISION_GEO _pinCellGloY, FP_PRECISION_GEO *_RTMDim, FP_PRECISION_GEO *_subDom_boundSurf)
{
	int _isEndP = 0;
	FP_PRECISION_GEO _x_right = _subDom_boundSurf[1];
	FP_PRECISION_GEO _y_top = _subDom_boundSurf[2];
	if ((fabs(_endPX - 0.5*GetRTModDimX(_RTMDim))<ON_SURFACE &&\
		fabs(_pinCellGloX - (_x_right - 0.5*GetRTModDimX(_RTMDim)))<ON_SURFACE) || \
		(fabs(_endPY - 0.5*GetRTModDimY(_RTMDim))<ON_SURFACE &&\
			fabs(_pinCellGloY - (_y_top - 0.5*GetRTModDimY(_RTMDim)))<ON_SURFACE)) {
		_isEndP = 1;
	}

	return _isEndP;
}

/**
 * check if the _point is the end of a long track
 * (on the top or left surface of Core)
 */
int LongTrack_IsEndP_TLSurf(FP_PRECISION_GEO _endPX, FP_PRECISION_GEO _endPY, FP_PRECISION_GEO _pinCellGloX, \
	FP_PRECISION_GEO _pinCellGloY, FP_PRECISION_GEO *_RTMDim, struct CORE *_core)
{
	int _isEndP =0;
	if ((fabs(_endPX + 0.5*GetRTModDimX(_RTMDim))<ON_SURFACE &&\
		fabs(_pinCellGloX - (-0.5*CoreLengthX + 0.5*GetRTModDimX(_RTMDim)))<ON_SURFACE) || \
		(fabs(_endPY - 0.5*GetRTModDimY(_RTMDim))<ON_SURFACE &&\
			fabs(_pinCellGloY - (0.5*CoreLengthY- 0.5*GetRTModDimY(_RTMDim)))<ON_SURFACE)) {
		_isEndP = 1;
	}
	return _isEndP;
}
int SubDomLongTrack_IsEndP_TLSurf(FP_PRECISION_GEO _endPX, FP_PRECISION_GEO _endPY, FP_PRECISION_GEO _pinCellGloX, \
	FP_PRECISION_GEO _pinCellGloY, FP_PRECISION_GEO *_RTMDim, FP_PRECISION_GEO *_subDom_boundSurf)
{
	int _isEndP = 0;
	FP_PRECISION_GEO _x_left = _subDom_boundSurf[0];
	FP_PRECISION_GEO _y_top = _subDom_boundSurf[2];
	if ((fabs(_endPX + 0.5*GetRTModDimX(_RTMDim))<ON_SURFACE &&\
		fabs(_pinCellGloX - (_x_left + 0.5*GetRTModDimX(_RTMDim)))<ON_SURFACE) || \
		(fabs(_endPY - 0.5*GetRTModDimY(_RTMDim))<ON_SURFACE &&\
			fabs(_pinCellGloY - (_y_top - 0.5*GetRTModDimY(_RTMDim)))<ON_SURFACE)) {
		_isEndP = 1;
	}
	return _isEndP;
}
/**
* check if the _point is the end of a long track
* (on the left or bottom surface of Core)
*/
int longTrack_isEndP_BL(FP_PRECISION_GEO _endPX, FP_PRECISION_GEO _endPY, FP_PRECISION_GEO _pinCellGloX, \
	FP_PRECISION_GEO _pinCellGloY, FP_PRECISION_GEO *_RTMDim, struct CORE *_core)
{
	int _isEndP = 0;
	if ((fabs(_endPX + 0.5*GetRTModDimX(_RTMDim))<ON_SURFACE &&\
		fabs(_pinCellGloX - (-0.5*CoreLengthX + 0.5*GetRTModDimX(_RTMDim)))<ON_SURFACE) || \
		(fabs(_endPY + 0.5*GetRTModDimY(_RTMDim))<ON_SURFACE &&\
			fabs(_pinCellGloY - (-0.5*CoreLengthY + 0.5*GetRTModDimY(_RTMDim)))<ON_SURFACE)) {
		_isEndP = 1;
	}
	return _isEndP;
}
int subDomLongTrack_isEndP_BL(FP_PRECISION_GEO _endPX, FP_PRECISION_GEO _endPY, FP_PRECISION_GEO _pinCellGloX, \
	FP_PRECISION_GEO _pinCellGloY, FP_PRECISION_GEO *_RTMDim, FP_PRECISION_GEO *_subDom_boundSurf)
{
	int _isEndP = 0;
	FP_PRECISION_GEO _x_left = _subDom_boundSurf[0];
	FP_PRECISION_GEO _y_bottom = _subDom_boundSurf[3];
	if ((fabs(_endPX + 0.5*GetRTModDimX(_RTMDim))<ON_SURFACE &&\
		fabs(_pinCellGloX - (_x_left + 0.5*GetRTModDimX(_RTMDim)))<ON_SURFACE) || \
		(fabs(_endPY + 0.5*GetRTModDimY(_RTMDim))<ON_SURFACE &&\
			fabs(_pinCellGloY - (_y_bottom + 0.5*GetRTModDimY(_RTMDim)))<ON_SURFACE)) {
		_isEndP = 1;
	}
	return _isEndP;
}
/**
* check if the _point is the end of a long track
* (on the right or bottom surface of Core)
*/
int longTrack_isEndP_BR(FP_PRECISION_GEO _endPX, FP_PRECISION_GEO _endPY, FP_PRECISION_GEO _pinCellGloX, \
	FP_PRECISION_GEO _pinCellGloY, FP_PRECISION_GEO *_RTMDim, struct CORE *_core)
{
	int _isEndP = 0;
	if ((fabs(_endPX - 0.5*GetRTModDimX(_RTMDim))<ON_SURFACE &&\
		fabs(_pinCellGloX - (0.5*CoreLengthX - 0.5*GetRTModDimX(_RTMDim)))<ON_SURFACE) || \
		(fabs(_endPY + 0.5*GetRTModDimY(_RTMDim))<ON_SURFACE &&\
			fabs(_pinCellGloY - (-0.5*CoreLengthY + 0.5*GetRTModDimY(_RTMDim)))<ON_SURFACE)) {
		_isEndP = 1;
	}
	return _isEndP;
}
/* The package always reads pre-generated tracks; keep only this cleanup path. */
void FreeTrackMem()
{
    NumLongTracks = FreeMem_int_IP(NumLongTracks, NumTrackSet, (char *)("NumLongTracks"));
    NumLTsSum = FreeMem_int_IP(NumLTsSum, NumTrackSet, (char *)("NumLTsSum"));
    NumLongTracks_SubDomX = FreeMem_int_IP(NumLongTracks_SubDomX, NumTrackSet, (char *)("NumLongTracks_SubDomX"));
    NumLongTracks_SubDomY = FreeMem_int_IP(NumLongTracks_SubDomY, NumTrackSet, (char *)("NumLongTracks_SubDomY"));
    NumLTs_HS = FreeMem_i32_I(NumLTs_HS, (char *)("NumLTs_HS"));
    NumTracksInLTs_HS = FreeMem_int_IP(NumTracksInLTs_HS, NumTrackSet, (char *)("NumTracksInLTs_HS"));
    NumTracks_HS = FreeMem_i32_I(NumTracks_HS, (char *)("NumTracks_HS"));
    LongTracksInfo_Total = FreeMem_LongTracksInfo(LongTracksInfo_Total, NumTrackSet, (char *)("LongTracksInfo_Total"));
    TracksInfoInLTs_HS = FreeMem_TracksInfoInLTs(TracksInfoInLTs_HS, NumTrackSet, (char *)("TracksInfoInLTs_HS"));
    NumFSRsInCell = FreeMem_i32_I(NumFSRsInCell, (char *)("NumFSRsInCell"));
    NumFSRs_Layer = FreeMem_i32_I(NumFSRs_Layer, (char *)("NumFSRs_Layer"));
    SumNumFSRs_Layer = FreeMem_i32_I(SumNumFSRs_Layer, (char *)("SumNumFSRs_Layer"));
    SumNumFSRsBasedGlobCellID = FreeMem_i32_I(SumNumFSRsBasedGlobCellID, (char *)("SumNumFSRsBasedGlobCellID"));
    RTMWidth_X = FreeMem_FP_I_Geo(RTMWidth_X, (char *)("RTMWidth_X"));
    RTMWidth_Y = FreeMem_FP_I_Geo(RTMWidth_Y, (char *)("RTMWidth_Y"));
    RTMArea = FreeMem_FP_I_Geo(RTMArea, (char *)("RTMArea"));
    NumTracks = FreeMem_int_IP(NumTracks, NumTrackSet, (char *)("NumTracks"));
    NumTracksX = FreeMem_int_IP(NumTracksX, NumTrackSet, (char *)("NumTracksX"));
    NumTracksY = FreeMem_int_IP(NumTracksY, NumTrackSet, (char *)("NumTracksY"));
    EffRaySpacing = FreeMem_FPGeo_IP(EffRaySpacing, NumTrackSet, (char *)("EffRaySpacing"));
    EffAzimAng = FreeMem_FPGeo_IP(EffAzimAng, NumTrackSet, (char *)("EffAzimAng"));
    TotNumTracksInOneRTM_HS = FreeMem_i32_I(TotNumTracksInOneRTM_HS, (char *)("TotNumTracksInOneRTM_HS"));
    TotalNumTracksInRTM_HS = FreeMem_i32_I(TotalNumTracksInRTM_HS, (char *)("TotalNumTracksInRTM_HS"));
    TotNumSegInRTM_HS = FreeMem_i32_I(TotNumSegInRTM_HS, (char *)("TotNumSegInRTM_HS"));
    NumFSRsInRTM = FreeMem_i32_I(NumFSRsInRTM, (char *)("NumFSRsInRTM"));
    FSRsXSID_subDom = FreeMem_i32_I(FSRsXSID_subDom, (char *)("FSRsXSID_subDom"));
}

/**
 * @ Function: GetNumLTsInHalfspace()
 *   introduction: get number of long tracks in halfspace [0,PI]
 * @ Parameter List: _numTrackSet - number of track set
 *                   _numAzimAng - number of the azimuthal angle in [0,2PI]
 *                   _numLongTracks_SubDom - number of long tracks in subdomian
 *                   _parameter - parameter name
 *   Return: void
 * @ Created     : 2018/05/15, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
int *GetNumLTsInHalfspace(int _numTrackSet,int *_numAzimAng, int **_numLongTracks_SubDom,char *_parameter)
{
	int i,_p,*_numLTsInHalfspace=NULL;
	_numLTsInHalfspace = AllocMem_i32_i32(_numTrackSet, _parameter);
	for (_p=0;_p<_numTrackSet;_p++) {
		_numLTsInHalfspace[_p]=0;
	    for (i = 0; i < _numAzimAng[_p] / 2; i++) {
	    	_numLTsInHalfspace[_p] += _numLongTracks_SubDom[_p][i];
	    }
	}
	return _numLTsInHalfspace;
}
int *GetNumLTsTotal(int _numTrackSet, int *_numAzimAng, int **_numLongTracks_SubDom, char *_parameter)
{
	int i, _p, *_numLTsTotal = AllocMem_i32_i32(_numTrackSet, _parameter);
	for (_p = 0; _p<_numTrackSet; _p++) {
		_numLTsTotal[_p] = 0;
		for (i = 0; i < _numAzimAng[_p]; i++) {
			_numLTsTotal[_p] += _numLongTracks_SubDom[_p][i];
		}
	}
	return _numLTsTotal;

}

/**
* @ Function: ()
*   introduction: get maximum number of segments in tracks
*/
int GetMaxNumSegsInTracks(int _numTrackSet, int *_totalNumTracksInRTM, int **_numSegsInTrack)
{
	int i, j, _max;
	_max = 0;
	for (j = 0; j < _numTrackSet; j++) {
		for (i = 0; i < _totalNumTracksInRTM[j]; i++) {
			if (_max < _numSegsInTrack[j][i]) {
				_max = _numSegsInTrack[j][i];
			}
		}
	}
	return _max;
}

static int MOC_GetTotalNumLongTracksForSet(int _numAzimAng, int *_numLongTracks_SubDom)
{
    int _total = 0;
    for (int _azimID = 0; _azimID < _numAzimAng; _azimID++) {
        _total += _numLongTracks_SubDom[_azimID];
    }
    return _total;
}

static int MOC_ReadHexVacuumBoundaryIDs(const char *_fileName,
                                        int _numAzimAng,
                                        int _totalNumLongTracks,
                                        unsigned char *_vacuumLTID,
                                        int *_numDuplicate,
                                        int *_numOutOfRange)
{
    FILE *_fp = fopen(_fileName, "rb");
    if (_fp == NULL) {
        printf("\n [**ERROR**]  Fail to open InpBoundCond file: %s\n", _fileName);
        fflush(stdout);
        exit(69);
    }

    int _numVacuumLT = 0;
    int _numDuplicateLocal = 0;
    int _numOutOfRangeLocal = 0;
    int _inpAngleNum = _numAzimAng / 2;

    for (int _angleIdx = 0; _angleIdx < _inpAngleNum; _angleIdx++) {
        int _azimID = 0;
        int _numInpTracks = 0;
        size_t _readCount = fread(&_azimID, sizeof(int), 1, _fp);
        _readCount += fread(&_numInpTracks, sizeof(int), 1, _fp);
        if (_readCount != 2) {
            printf("\n [**ERROR**]  Fail to read InpBoundCond header at angleIdx=%d\n", _angleIdx);
            fflush(stdout);
            fclose(_fp);
            exit(69);
        }

        for (int _trkIdx = 0; _trkIdx < _numInpTracks; _trkIdx++) {
            int _globalLTID = -1;
            _readCount = fread(&_globalLTID, sizeof(int), 1, _fp);
            if (_readCount != 1) {
                printf("\n [**ERROR**]  Fail to read InpBoundCond track ID at angleIdx=%d, trkIdx=%d\n",
                       _angleIdx, _trkIdx);
                fflush(stdout);
                fclose(_fp);
                exit(69);
            }

            if (_globalLTID < 0 || _globalLTID >= _totalNumLongTracks) {
                _numOutOfRangeLocal++;
                continue;
            }

            if (_vacuumLTID[_globalLTID]) {
                _numDuplicateLocal++;
            } else {
                _vacuumLTID[_globalLTID] = 1;
                _numVacuumLT++;
            }
        }
    }

    fclose(_fp);

    if (_numDuplicate != NULL) {
        *_numDuplicate = _numDuplicateLocal;
    }
    if (_numOutOfRange != NULL) {
        *_numOutOfRange = _numOutOfRangeLocal;
    }

    return _numVacuumLT;
}

static void MOC_MapTriLongTrackToHalfspace(int _azimID,
                                           int _trackID,
                                           int _numAzimAng,
                                           int _numLongTracks,
                                           int *_azimIDInHalf,
                                           int *_trackIDInHalf)
{
    int _halfNumAzimAng = _numAzimAng / 2;

    *_azimIDInHalf = _azimID;
    *_trackIDInHalf = _trackID;

    if (_halfNumAzimAng <= 0 || _azimID < _halfNumAzimAng) {
        return;
    }

    /* The 180-degree counterpart uses reversed track numbering. */
    *_azimIDInHalf = _azimID - _halfNumAzimAng;
    *_trackIDInHalf = _numLongTracks - 1 - _trackID;
}

static int MOC_IsTriBottomOutlet(int _azimID,
                                 int _trackID,
                                 int _nextAzimID,
                                 int _nextTrackID,
                                 int _numAzimAng,
                                 int *_numLongTracks_SubDom)
{
    int _halfNumAzimAng = _numAzimAng / 2;

    if (_numAzimAng <= 0 ||
        _halfNumAzimAng <= 0 ||
        _azimID < 0 ||
        _azimID >= _numAzimAng ||
        _nextAzimID < 0 ||
        _nextAzimID >= _numAzimAng ||
        _trackID < 0 ||
        _trackID >= _numLongTracks_SubDom[_azimID] ||
        _nextTrackID < 0 ||
        _nextTrackID >= _numLongTracks_SubDom[_nextAzimID]) {
        return 0;
    }

    /* Bottom boundary applies to outgoing tracks, not incoming tracks that also touch the edge. */
    if (_azimID < _halfNumAzimAng) {
        return 0;
    }

    return _nextAzimID == _numAzimAng - 1 - _azimID;
}

static void MOC_GetHexLongTrackMap(int _globalLTID,
                                   int _numAzimAng,
                                   int *_numLongTracks_SubDom,
                                   int *_azimID,
                                   int *_trackID)
{
    int _offset = 0;
    for (int _iazim = 0; _iazim < _numAzimAng; _iazim++) {
        int _numLT = _numLongTracks_SubDom[_iazim];
        if (_globalLTID < _offset + _numLT) {
            *_azimID = _iazim;
            *_trackID = _globalLTID - _offset;
            return;
        }
        _offset += _numLT;
    }

    *_azimID = -1;
    *_trackID = -1;
}

static int MOC_BuildAutoTriVacuumOutletMask(struct SUBDOM_LONGTRACK_INFO *_longTracksInfo,
                                            int _numAzimAng,
                                            int *_numLongTracks_SubDom,
                                            unsigned char *_autoVacuumLTID)
{
    int _numAutoVacuumLT = 0;
    int _globalLTID = 0;

    for (int _azimID = 0; _azimID < _numAzimAng; _azimID++) {
        for (int _trackID = 0; _trackID < _numLongTracks_SubDom[_azimID]; _trackID++) {
            int _nextAzimID = _longTracksInfo[_globalLTID].next_azim_angle_id;
            int _nextTrackID = _longTracksInfo[_globalLTID].next_LongTrack_id_SubDom;

            if (MOC_IsTriBottomOutlet(_azimID,
                                      _trackID,
                                      _nextAzimID,
                                      _nextTrackID,
                                      _numAzimAng,
                                      _numLongTracks_SubDom)) {
                _autoVacuumLTID[_globalLTID] = 1;
                _numAutoVacuumLT++;
            }

            _globalLTID++;
        }
    }

    return _numAutoVacuumLT;
}

static int MOC_ApplyTriBottomBoundaryBeta(struct SUBDOM_LONGTRACK_INFO *_longTracksInfo,
                                          int _numAzimAng,
                                          int *_numLongTracks_SubDom,
                                          FP_PRECISION _boundBeta)
{
    int _numApplied = 0;
    int _globalLTID = 0;

    for (int _azimID = 0; _azimID < _numAzimAng; _azimID++) {
        for (int _trackID = 0; _trackID < _numLongTracks_SubDom[_azimID]; _trackID++) {
            int _nextAzimID = _longTracksInfo[_globalLTID].next_azim_angle_id;
            int _nextTrackID = _longTracksInfo[_globalLTID].next_LongTrack_id_SubDom;

            if (MOC_IsTriBottomOutlet(_azimID,
                                      _trackID,
                                      _nextAzimID,
                                      _nextTrackID,
                                      _numAzimAng,
                                      _numLongTracks_SubDom)) {
                _longTracksInfo[_globalLTID]._boundBeta = _boundBeta;
                _numApplied++;
            }

            _globalLTID++;
        }
    }

    return _numApplied;
}

static void MOC_CheckHexTriVacuumBoundaryIDs(struct SUBDOM_LONGTRACK_INFO *_longTracksInfo,
                                             int _numAzimAng,
                                             int *_numLongTracks_SubDom,
                                             const char *_inpBoundCondFile)
{
    int _totalNumLongTracks = MOC_GetTotalNumLongTracksForSet(_numAzimAng, _numLongTracks_SubDom);
    unsigned char *_refVacuumLTID = (unsigned char *)calloc(_totalNumLongTracks, sizeof(unsigned char));
    unsigned char *_autoVacuumLTID = (unsigned char *)calloc(_totalNumLongTracks, sizeof(unsigned char));

    if (_refVacuumLTID == NULL || _autoVacuumLTID == NULL) {
        printf("\n [**ERROR**]  Fail to allocate vacuum boundary diagnostic masks.\n");
        fflush(stdout);
        exit(69);
    }

    int _numDuplicate = 0;
    int _numOutOfRange = 0;
    int _numRef = MOC_ReadHexVacuumBoundaryIDs(_inpBoundCondFile,
                                              _numAzimAng,
                                              _totalNumLongTracks,
                                              _refVacuumLTID,
                                              &_numDuplicate,
                                              &_numOutOfRange);
    int _numAuto = MOC_BuildAutoTriVacuumOutletMask(_longTracksInfo,
                                                    _numAzimAng,
                                                    _numLongTracks_SubDom,
                                                    _autoVacuumLTID);

    int _numMatched = 0;
    int _numOnlyRef = 0;
    int _numOnlyAuto = 0;
    for (int _globalLTID = 0; _globalLTID < _totalNumLongTracks; _globalLTID++) {
        if (_refVacuumLTID[_globalLTID] && _autoVacuumLTID[_globalLTID]) {
            _numMatched++;
        } else if (_refVacuumLTID[_globalLTID]) {
            _numOnlyRef++;
        } else if (_autoVacuumLTID[_globalLTID]) {
            _numOnlyAuto++;
        }
    }

    printf("\n [HexTriBCCheck] InpBoundCond file: %s", _inpBoundCondFile);
    printf("\n [HexTriBCCheck] totalLT=%d, refVacuum=%d, autoVacuum=%d, matched=%d, onlyRef=%d, onlyAuto=%d",
           _totalNumLongTracks, _numRef, _numAuto, _numMatched, _numOnlyRef, _numOnlyAuto);
    printf("\n [HexTriBCCheck] duplicateInRef=%d, outOfRangeInRef=%d", _numDuplicate, _numOutOfRange);

    int _numPrinted = 0;
    const int _maxPrint = 50;
    for (int _globalLTID = 0; _globalLTID < _totalNumLongTracks && _numPrinted < _maxPrint; _globalLTID++) {
        if (_refVacuumLTID[_globalLTID] == _autoVacuumLTID[_globalLTID]) {
            continue;
        }

        int _azimID = -1;
        int _trackID = -1;
        MOC_GetHexLongTrackMap(_globalLTID, _numAzimAng, _numLongTracks_SubDom, &_azimID, &_trackID);

        int _nextAzimID = _longTracksInfo[_globalLTID].next_azim_angle_id;
        int _nextTrackID = _longTracksInfo[_globalLTID].next_LongTrack_id_SubDom;
        int _nextAzimIDInHalf = -1;
        int _nextTrackIDInHalf = -1;
        if (_nextAzimID >= 0 &&
            _nextAzimID < _numAzimAng &&
            _nextTrackID >= 0 &&
            _nextTrackID < _numLongTracks_SubDom[_nextAzimID]) {
            MOC_MapTriLongTrackToHalfspace(_nextAzimID,
                                           _nextTrackID,
                                           _numAzimAng,
                                           _numLongTracks_SubDom[_nextAzimID],
                                           &_nextAzimIDInHalf,
                                           &_nextTrackIDInHalf);
        }

        printf("\n [HexTriBCCheck] mismatch gid=%d, ref=%d, auto=%d, azim=%d, track=%d, nextAzim=%d, nextTrack=%d, nextHalfAzim=%d, nextHalfTrack=%d",
               _globalLTID,
               (int)_refVacuumLTID[_globalLTID],
               (int)_autoVacuumLTID[_globalLTID],
               _azimID,
               _trackID,
               _nextAzimID,
               _nextTrackID,
               _nextAzimIDInHalf,
               _nextTrackIDInHalf);
        _numPrinted++;
    }

    if (_numOnlyRef == 0 && _numOnlyAuto == 0 && _numOutOfRange == 0) {
        printf("\n [HexTriBCCheck] auto vacuum outlet IDs match InpBoundCond.\n");
    } else if (_numOnlyRef + _numOnlyAuto > _maxPrint) {
        printf("\n [HexTriBCCheck] printed first %d mismatches only.\n", _maxPrint);
    }
    fflush(stdout);

    free(_refVacuumLTID);
    free(_autoVacuumLTID);
}


void GetIndexOfLongTracks_HS(int **_LTsIndex_AzimAngID, int **_LTsIndex_LTID, int _numTrackSet, int *_numAzimAng, int **_numLongTracksSubDom)
{
	int _iter1, _iter2, _iter3, _counter, _halfNumAzimAng;
	for (_iter1 = 0; _iter1 < _numTrackSet; _iter1++) {
		_counter = 0;
		_halfNumAzimAng = _numAzimAng[_iter1] / 2;
		for (_iter2 = 0; _iter2 < _halfNumAzimAng; _iter2++) {
			for (_iter3 = 0; _iter3 < _numLongTracksSubDom[_iter1][_iter2]; _iter3++) {
				_LTsIndex_AzimAngID[_iter1][_counter] = _iter2;
				_LTsIndex_LTID[_iter1][_counter] = _iter3;
				_counter++;
			}
		}
	}
}

void GetIndexOfLongTracks_Total(int **_LTsIndex_AzimAngID, int **_LTsIndex_LTID, int _numTrackSet, int *_numAzimAng, int **_numLongTracksSubDom)
{
	int _iter1, _iter2, _iter3, _counter;
	for (_iter1 = 0; _iter1 < _numTrackSet; _iter1++) {
		_counter = 0;
		for (_iter2 = 0; _iter2 < _numAzimAng[_iter1]; _iter2++) {
			for (_iter3 = 0; _iter3 < _numLongTracksSubDom[_iter1][_iter2]; _iter3++) {
				_LTsIndex_AzimAngID[_iter1][_counter] = _iter2;
				_LTsIndex_LTID[_iter1][_counter] = _iter3;
				_counter++;
			}
		}
	}
}
void SetFSRVol_SubDom(FP_PRECISION_GEO *_FSRArea_SubDom, FP_PRECISION_GEO *_FSRVol_SubDom, \
    FP_PRECISION_GEO *_WidthLayersInCore, int _numLayersZ,int _numCellsInDom, int *_numFSRsInCell,\
    int *_sumNumFSRsBasedGlobCellID)
{
	int _lay, _cell, _reg,_fsrID;
	FP_PRECISION _hight;
	FP_PRECISION_GEO _fsrVolume;
	for (_lay = 0; _lay < _numLayersZ; _lay++) {
      _hight = _WidthLayersInCore[_lay];
	  for (_cell = 0; _cell < _numCellsInDom; _cell++) {
		for (_reg = 0; _reg < _numFSRsInCell[_lay*_numCellsInDom + _cell]; _reg++) {
		  _fsrID = _sumNumFSRsBasedGlobCellID[_lay*_numCellsInDom + _cell] + _reg;
		  _fsrVolume = _FSRArea_SubDom[_fsrID];
		  _FSRVol_SubDom[_fsrID]=_fsrVolume*_hight;
		}
	  }
	}
}
void SetFSRsXSsID_subDomain(struct CORE *_core,int _subDomID_row, int _subDomID_column)
{
	int /*g, g1, */i, j, k, i1, j1/*, k1*/,_lay,_globLayID,_globFSRID;
	int _asse_type, _latt_type, _pin_inLatt, _glo_pin_id, _mat_id, _fsr_id, _CellID_inSubDom, _subDomID_set, _subDomSeqID;
	struct CELLS_IN_PINCELL *_pointer1;

    FSRsXSID_subDom = AllocMem_i32_i32(TotalNumFSRs_subDom, "FSRsXSID_subDom");

	_subDomID_set = GetSubDomSequenceID(_subDomID_row, _subDomID_column, NumSubDomains);

    if (INPUT_XSs_MATorFSR == 0){
        for (_lay = 0; _lay < MOCNumLayersZ; _lay++) {
            _globLayID = SubDomGlobLayerID[_lay];
            for (i = 0; i<(_core->num_X)*(_core->num_Y); i++) {
                _asse_type = _core->asse_in_core[i];
                for (j = 0; j<NumAssemblies; j++) {
                    if (Assemblies[j].type_id == _asse_type) {
                        break;
                    }
                }
                _latt_type = Assemblies[j].latt_in_asse[_globLayID];
                for (k = 0; k<NumLattices; k++) {
                    if (Lattices[k].type_id == _latt_type) {
                        break;
                    }
                }

                for (i1 = 0; i1<Lattices[k].num_x*Lattices[k].num_y; i1++) {
                    _pin_inLatt = Lattices[k].pinCells_in_lattice[i1];
                    for (j1 = 0; j1<NumCells; j1++) {
                        if (Cells[j1].type_id == _pin_inLatt) {
                            break;
                        }
                    }

                    _glo_pin_id = GetGlobalCellID_2D(i, i1, _globLayID);
                    _subDomSeqID = GetSubDomIDBaseGlobPinID(_glo_pin_id, NumSubDomains, NumPin_subDomain);
                    _CellID_inSubDom = GetPinSeqIDInSubDomBaseGlobPinID(_glo_pin_id, NumSubDomains, NumPin_subDomain);
                    if (_subDomSeqID == _subDomID_set) {
                        for (_pointer1 = Cells[j1].cells_in_pinCell; _pointer1 != NULL; _pointer1 = _pointer1->next) {
                            _fsr_id = _pointer1->cells->id;
                            _mat_id = _pointer1->cells->mat_id;

                            _globFSRID=SumNumFSRsBasedGlobCellID[_lay*NumPinsInSubDomPerLayer+_CellID_inSubDom]+_fsr_id;

                            FSRsXSID_subDom[_globFSRID]=_mat_id;
                        }
                    }
                }
            }
        }
    } else if (INPUT_XSs_MATorFSR == 1){
        for (_globFSRID=0; _globFSRID<TotalNumFSRs_subDom; _globFSRID++){
            FSRsXSID_subDom[_globFSRID]=_globFSRID;
        }
    }
}


void Interface_Geometry()
{
	/** get maxmum number of segments in tracks*/
	
	CoreArea2D = CoreLengthX*CoreLengthY;
}
int GetMaxNumTracksInLongTracks(int _numTrackSet, int *_numLTsInHalfspace, int **_numTracks_in_Longtrack_SubDom)
{
	int i, _lt, _max_num;
	_max_num = 0;
	for (i = 0; i < _numTrackSet; i++) {
		for (_lt = 0; _lt < _numLTsInHalfspace[i]; _lt++) {
			if (_max_num < _numTracks_in_Longtrack_SubDom[i][_lt]) {
				_max_num = _numTracks_in_Longtrack_SubDom[i][_lt];
			}
		}
	}
	return _max_num;
}
int GetMaxNumPolarAng(int _numTrackSet, int *_numPolarAng)
{
	int i/*,j*/,_max;
	_max=0;
	for (i=0;i<_numTrackSet;i++) {
	if (_max<_numPolarAng[i]) {
			_max=_numPolarAng[i];
		}
	}
	return _max;
}

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
                             int *_nextTrackID_TrackFile) {

    /// only for one pin, the capability of path link in hex mode should be developed by another poor guy
    ///                                                                                               ———— Mr. Zou


    int i, j, _p, _num, _halfNumAzimAng, _trackId, _numPinCells_subDom_X = MOCNumPinsInSubDom_X, _numTracks_subDomX/*, _count*/;
    int _curr_subDomID,/* _pinSeqID_inSubDom,*/ _lt_add, ***_subDom_id_curr = NULL, _isEndP = 1, _isSubEndP = 1;
    int _azimu_ang_id, _nextLTid = 0, _next_subDom_LTid = 0, _next_subDomID = 0;
    int _numPinCells_subDomX, _numPinCells_subDomY, _cellIDInSubDom_2D_X, _cellIDInSubDom_2D_Y/*, _globPinCellID_2D*/, _pinSeqID_inSubDom_2D;
    int _next_cellIDInSubDom_2D_X, _next_cellIDInSubDom_2D_Y, _next_subDom_trackID, _next_subDomID_X, _next_subDomID_Y;
    int _lt_count, _numTracks;
    int _count11;

    _numPinCells_subDomX = MOCNumPinsInSubDom_X;
    _numPinCells_subDomY = MOCNumPinsInSubDom_Y;
    _curr_subDomID = GetSubDomSequenceID(_subDomLocID_2D_Y, _subDomLocID_2D_X, _numSubDomains);
    /** calloc for _subDom_id_curr */

    _lt_add = 0;
    for (_p=0; _p < _numTrackSet; _p++) {
        for (i = 0; i < _numAzimAng[_p]; i++) {
//            printf("\n_numLongTracks_SubDom[%d][%d] = %d", _p, i, _numLongTracks_SubDom[_p][i]);
            for (j = 0; j < _numLongTracks_SubDom[_p][i]; j++) {
                _tracksInfoInLTs_2PI[_p][_lt_add].cell_ID = 0;
                _tracksInfoInLTs_2PI[_p][_lt_add].track_ID = j;

                if (i<_numAzimAng[_p]/2){
                    _tracksInfoInLTs_HS[_p][_lt_add].cell_ID = 0;
                    _tracksInfoInLTs_HS[_p][_lt_add].track_ID = j;
                }

                _lt_add++;
            }
        }
    }

//    for (_p = 0; _p < _numTrackSet; _p++){
//        for (i=0;i<_lt_add;i++){
//            printf("\n[hex] _tracksInfoInLTs_2PI[%d][%d].cell_ID = %d", _p, i, _tracksInfoInLTs_2PI[_p][i].cell_ID);
//            printf("\t_tracksInfoInLTs_2PI[%d][%d].track_ID = %d", _p, i, _tracksInfoInLTs_2PI[_p][i].track_ID);
//
//        }
//    }


    _lt_add = 0;
    for (_p=0; _p < _numTrackSet; _p++) {
        for (i = 0; i < _numAzimAng[_p]; i++) {
            for (j = 0; j < _numLongTracks_SubDom[_p][i]; j++) {
                _longTracksInfo[_p][_lt_add].next_SubDom_id = 0;  // only one pin
                _longTracksInfo[_p][_lt_add].next_azim_angle_id = _nextAzimID_TrackFile[_lt_add];
                _longTracksInfo[_p][_lt_add].next_LongTrack_id_SubDom = _nextTrackID_TrackFile[_lt_add];
                _lt_add++;
            }
        }
    }


    _lt_add = 0;
    for (_p=0; _p < _numTrackSet; _p++) {
        for (i = 0; i < _numAzimAng[_p]; i++) {
            for (j = 0; j < _numLongTracks_SubDom[_p][i]; j++) {
                _numTracks_in_Longtrack_SubDom[_p][_lt_add] = 1;
                _lt_add++;
            }
        }
    }

//    for(_p = 0; _p < _numTrackSet; _p++){
//        for (i=0; i<_lt_add;i++){
//            printf("\n[hex] _longTracksInfo[%d][%d].next_SubDom_id = %d", _p, i, _longTracksInfo[_p][i].next_SubDom_id);
//            printf("  next_azim_angle_id = %d",  _longTracksInfo[_p][i].next_azim_angle_id);
//            printf("  next_LongTrack_id_SubDom = %d", _longTracksInfo[_p][i].next_LongTrack_id_SubDom);
//        }
//    }


    _lt_add = 0;
    for (_p = 0; _p < _numTrackSet; _p++) {
        for (i = 0; i < _numAzimAng[_p]; i++) {
            for (j = 0; j < _numLongTracks_SubDom[_p][i]; j++) {
                _longTracksInfo[_p][_lt_add]._boundBeta = RadialBoundary(i,
                                                                         _longTracksInfo[_p][_lt_add].next_azim_angle_id,
                                                                         _numAzimAng[_p]);
                _lt_add++;
            }
        }
    }

    if (SouthBC == VACUUM || SouthBC == REFLECTIVE) {
        FP_PRECISION _triBottomBeta = (SouthBC == VACUUM) ? 0.0 : 1.0;
        int _numTriBottomOutlet = MOC_ApplyTriBottomBoundaryBeta(_longTracksInfo[0],
                                                                 _numAzimAng[0],
                                                                 _numLongTracks_SubDom[0],
                                                                 _triBottomBeta);

        if (getenv("MOC_DEBUG_TRI_BC") != NULL) {
            printf("\n [HexTriBCCheck] auto bottom outlet beta applied: count=%d, SouthBC=%d, beta=%g",
                   _numTriBottomOutlet,
                   SouthBC,
                   _triBottomBeta);
        }
    }

    if (getenv("MOC_DEBUG_TRI_BC") != NULL) {
        if (getenv("InpBoundCond") != NULL) {
            MOC_CheckHexTriVacuumBoundaryIDs(_longTracksInfo[0],
                                             _numAzimAng[0],
                                             _numLongTracks_SubDom[0],
                                             getenv("InpBoundCond"));
        } else {
            printf("\n [HexTriBCCheck] InpBoundCond is not set; skipped reference check.\n");
            fflush(stdout);
        }
    }

    _trackIndexInLTs[0][0] = 0;
    _lt_add = 1;
    for (_p = 0; _p < _numTrackSet; _p++) {
        for (i = 0; i < _numAzimAng[_p]; i++) {
//            printf("\n_numLongTracks_SubDom[%d][%d]=%d", _p, i, _numLongTracks_SubDom[_p][i]);
            for (j = 0; j < _numLongTracks_SubDom[_p][i]; j++) {
                _trackIndexInLTs[_p][_lt_add] = _lt_add;
                _lt_add++;
            }
        }
    }

//    for(i=0;i<73;i++) printf("\n _trackIndexInLTs[0][%d]=%d", i, _trackIndexInLTs[0][i]);
//
//
//    exit(998);
}
