#ifndef MOC_GEOMETRY_H
#define MOC_GEOMETRY_H
#include "MOC_Core.h"
#include "MOC_CommonMacro.h"
#include "MOC_Cell.h"
#include "MOC_Pin_Cell.h"

extern int NumCellsInCore, NumCellsInCore_2D_X, NumCellsInCore_2D_Y;  //number of cells in the whole Core for a specific layer
extern int NumPinsInSubDomPerLayer;
extern int NumLayer_Core;
extern int *CellType;  //specify the RTM type for every cell in Core
extern int *CellRTMType_SubDom;
extern int *SubDomGlobCellID;
extern int SubDomLocID_2D_Y, SubDomLocID_2D_X, MOC_subDomID_layer;
extern int MOCNumPinsInSubDom_X, MOCNumPinsInSubDom_Y, MOCNumLayersZ,MOC_NumPinsInSubDom,MOC_NumPinsInWholeDom, MOC_NumPinsInGlobDom;
extern int NumSurfaceX, NumSurfaceY, NumSurfaceZ;
extern int *SubDomGlobLayerID;
extern FP_PRECISION_GEO *WidthLayersInCore;
extern int SubDomOrgPinOffset_2D_X, SubDomOrgPinOffset_2D_Y;
extern int TotalNumFSRsInRTM;
extern FP_PRECISION_GEO *OneOverFSRsAreaInRTM_AntiClockFSR, *OneOverFSRsAreaInRTM_RayWisedFSR;
extern FP_PRECISION_GEO *FSRsAreaInRTM_AntiClockFSR, *FSRArea_SubDom, *FSRVol_SubDom, /**GC_FSRsAreaInGlob, */*OneOverFSRArea_SubDom;
extern int *SumNumFSRsInRTM;
extern FP_PRECISION_GEO *RTMWidth_X, *RTMWidth_Y, *RTMArea;

int GetNumCellsInCore_layer(struct CORE *_core, int _layerNumInCore);
void MOCInitFSRs(int _procID, int _numProcs, int _subDomID_row, int _subDomID_column, int _subDomID_layer, \
	int _numCellsInSubDom_X, int _numCellsInSubDom_Y, int _numLayersInsubDom_Z, \
	int *_numSubDomains, int **_numPin_subDomain, int _numCells, struct PIN_CELL *_cells);
int GetNumLayersInCore(struct CORE *_core);
int GetGlobalCellID_2D(int _asseIdInCore, int _cellIdInAsse, int _layerNumInCore);
int GetNumCellsInAsse_X(int _asseIdInCore, int _lattIdInAsse);
int GetNumCellsInAsse_Y(int _asseIdInCore, int _lattIdInAsse);
int GetNumCellsInDomain_X(int *_numSubDomains, int **_numPin_subDomain);
int GetNumCellsInDomain_Y(int *_numSubDomains, int **_numPin_subDomain);
int get_NumEffSubDomains_layer(int *_numSubDomains, int **_subDomainsMap);
int GetSubDomIDBaseGlobPinID_Y(int _globPinID, int *_numSubDomains, int **_numPin_subDomain);
int GetSubDomIDBaseGlobPinID_X(int _globPinID, int *_numSubDomains, int **_numPin_subDomain);
int GetGlobRowIDBaseGlobPinID(int _globPinID, int *_numSubDomains, int **_numPin_subDomain);
int GetGlobColumnIDBaseGlobPinID(int _globPinID, int *_numSubDomains, int **_numPin_subDomain);
int GetSubDomSequenceID(int _row_id, int _colume_id, int *_numSubDomains);
int GetSubDomIDBaseGlobPinID(int _globPinID, int *_numSubDomains, int **_numPin_subDomain);
int GetPinRowIDInSubDomBaseGlobPinID(int _globPinID, int *_numSubDomains, int **_numPin_subDomain);
int GetPinColumnIDInSubDomBaseGlobPinID(int _globPinID, int *_numSubDomains, int **_numPin_subDomain);
int GetPinSeqIDInSubDomBaseGlobPinID(int _globPinID, int *_numSubDomains, int **_numPin_subDomain);
int get_procID(int _subDom_rowID, int _sunDom_columnID, int **_subDomainsMap);
int get_SubDomRow_basedSubDomID(int _subDomID, int *_numSubDoms);
int get_SubDomColumn_basedSubDomID(int _subDomID, int *_numSubDoms);
int get_procID_basedSubDomID(int _subDomID, int _subDom_layerID, int *_numSubDoms, int ***_subDomainsMap);
void GetSubDomLocIDBasedProcID(int _procID, int *_numSubDoms, int ***_subDomainsMap, \
	int *_rowID, int *_colID, int *_layID);
void GetNumCellsInSubDom(int _subDomLocIDRow, int _subDomLocIDCol, int _subDomLocIDLay, int **_numPinsInSubDomain, \
	int *_MOC_numCellsInSubDom_X, int *_MOC_numCellsInSubDom_Y, int *_MOC_numLayersInsubDom_Z);
int get_subDomRow_basedProcID(int _procID, int *_numSubDoms, int ***_subDomainsMap);
int get_subDomCol_basedProcID(int _procID, int *_numSubDoms, int ***_subDomainsMap);
int get_subDomLay_basedProcID(int _procID, int *_numSubDoms, int ***_subDomainsMap);
int get_numCellsInSubDom_X_baseProcID(int _procID, int *_numSubDoms, int ***_subDomainsMap, int **_numPin_subDomain);
int get_numCellsInSubDom_Y_baseProcID(int _procID, int *_numSubDoms, int ***_subDomainsMap, int **_numPin_subDomain);
int GetNumCellsInAsse_layer(int _asseIdInCore, int _layerID);
int get_procID_1(int _subDom_rowID, int _sunDom_columnID, int _subDom_layID, int ***_subDomainsMap);
FP_PRECISION_GEO *GetWidthLayersInCore(int _numLayersInCore, struct CORE *_core);
int getNumSubLayersInCore(struct CORE *_core);
int getNumCellsInCore_SubLayer(struct CORE *_core);
//_sublayerID start from 1
int getLayerID_subDom(int ProcID, int _sublayerID);
//_sublayerID start from 1
FP_PRECISION_GEO getLayerWidth_subDom(int ProcID, int _sublayerID);
//_sublayerID start from 1
FP_PRECISION_GEO getSubLayerWidth_subDom(int ProcID, int _sublayerID);
//_layerID start from 0
int getLayerID_byLay_subDom(int ProcID, int _layerID);
int GetNumCellsInCore_X(struct CORE *_core, int _layerIDInCore);
int GetNumCellsInCore_Y(struct CORE *_core, int _layerIDInCore);
void GetGlobPinCellOffset_SubDom(int _subDomLocID_2D_Y, int _subDomLocID_2D_X, int **_numPinsInSubDomain, \
	int *_subDomOrgPinOffset_2D_X, int *_subDomOrgPinOffset_2D_Y);
int GetGlobalPinCellID_2D(int _cellIDInSubDom_2D_X, int _cellIDInSubDom_2D_Y, int _subDomOrgPinOffset_2D_X, \
	int _subDomOrgPinOffset_2D_Y, int _numCellsInCore_2D_X);
void GetSumNumFSRsInRTM(int *_SumNumFSRsInRTM, int _numRTMGeoType, int *_numFSRsInRTM);
void TransOneOverFSRVolInTRM_RayWised(FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_antiClockFSR, \
	FP_PRECISION_GEO *_OneOverFSRsAreaInRTM_rayWisedFSR, int _totalNumFSRsInRTM, int _azimID, \
	int *_fsrIdMap_rayWisedToAntiClock, int _numRTMGeoType, int *_numFSRsInRTM, int *_sumNumFSRsInRTM);
void SetCMFDNodeVolume(FP_PRECISION *_CMFDNodeVol, int _numPinsInDom, FP_PRECISION_GEO *_RTMArea, \
	int *_cellRTMType, FP_PRECISION_GEO *_WidthLayersInCore, int _NumPinsInSubDomPerLayer);
void FreeGeoMem();
#endif // GEOMETRY_H
