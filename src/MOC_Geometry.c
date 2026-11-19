/********************************************************************************/
/* @ File: MOC_Geometry.c
/*   introduction: functions related to geometry initialization, genetare fsr
/* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*   1st modified:
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "MOC_AllocAndFreeMem.h"
#include "MOC_Assembly.h"
#include "MOC_Cell.h"
#include "MOC_CommonMacro.h"
#include "MOC_Core.h"
#include "MOC_Geometry.h"
#include "MOC_Input.h"
#include "MOC_Lattice.h"
#include "MOC_Output.h"
#include "MOC_Pin_Cell.h"
#include "MOC_Surface.h"
#include "MOC_Visualization.h"
#ifdef MPI_USED
  #include <mpi.h>
#endif

int NumCellsInCore, NumCellsInCore_2D_X, NumCellsInCore_2D_Y;  //number of cells in the whole core for a specific layer
int NumPinsInSubDomPerLayer; // number of cells in subdomain-2D
int *CellType = NULL;  //specify the RTM type for every cell in core
int *CellRTMType_SubDom = NULL;  //specify the RTM type for every cell in subdomain
int *SubDomGlobCellID = NULL;
int SubDomLocID_2D_Y, SubDomLocID_2D_X, MOC_subDomID_layer;
int MOCNumPinsInSubDom_X, MOCNumPinsInSubDom_Y, MOCNumLayersZ, MOC_NumPinsInSubDom, MOC_NumPinsInWholeDom=0,MOC_NumPinsInGlobDom=0;
int NumSurfaceX, NumSurfaceY, NumSurfaceZ;
int *SubDomGlobLayerID = NULL; // global layer id in subdomain
int NumLayer_Core;
int SubDomOrgPinOffset_2D_X, SubDomOrgPinOffset_2D_Y;
FP_PRECISION_GEO *WidthLayersInCore = NULL;
int TotalNumFSRsInRTM;
FP_PRECISION_GEO *OneOverFSRsAreaInRTM_AntiClockFSR = NULL,*OneOverFSRsAreaInRTM_RayWisedFSR=NULL,*FSRArea_SubDom=NULL,*FSRVol_SubDom=NULL, *OneOverFSRArea_SubDom=NULL;
FP_PRECISION_GEO *FSRsAreaInRTM_AntiClockFSR = NULL;
FP_PRECISION_GEO *RTMWidth_X = NULL, *RTMWidth_Y = NULL, *RTMArea=NULL;
int *SumNumFSRsInRTM = NULL;

/*
 * Dynamic geometry/FSR construction is outside the publication package.
 * The supported workflow always loads pre-generated track data.
 */
void MOCInitFSRs(int _procID, int _numProcs, int _subDomIDRow, int _subDomIDColumn, int _subDomIDLayer,
    int _numCellsInSubDom_X, int _numCellsInSubDom_Y, int _numLayersInsubDom_Z,
    int *_numSubDomains, int **_numPinsInSubDomain, int _numCells, struct PIN_CELL *_cells)
{
    (void)_procID;
    (void)_numProcs;
    (void)_subDomIDRow;
    (void)_subDomIDColumn;
    (void)_subDomIDLayer;
    (void)_numCellsInSubDom_X;
    (void)_numCellsInSubDom_Y;
    (void)_numLayersInsubDom_Z;
    (void)_numSubDomains;
    (void)_numPinsInSubDomain;
    (void)_numCells;
    (void)_cells;
    fprintf(stderr, "[ERROR] Dynamic FSR construction is unavailable; use UseTrackFile=1.\n");
    exit(69);
}

/**
* @ Function: GetNumLayersInCore()
*   introduction: Get number of layers in whole Core
* @ Parameter List: _core - struct CORE object
*   Return: _numLayers - number of layers in whole Core
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int GetNumLayersInCore(struct CORE *_core)
{
	int i, _numLayers;
	_numLayers = 0;
	for (i = 0; i<NumAssemblies; i++) {
		if (_core->asse_in_core[0] == Assemblies[i].type_id) {
			// _numLayers = Assemblies[i].num_layer;
			_numLayers = Assemblies[i].num_sublayer;
			break;
		}
	}
	return _numLayers;
}


/**
* @ Function: GetWidthLayersInCore()
*   introduction: Get the width of each layers in whole Core
* @ Parameter List: _numLayersInCore - number of layers in core
*                   _core - struct CORE object
*   Return: _withdlayersInCore - 1D array, width of layers in whole Core
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
FP_PRECISION_GEO *GetWidthLayersInCore(int _numLayersInCore, struct CORE *_core)
{
	int j;
	FP_PRECISION_GEO *_withdlayersInCore=NULL;
	_withdlayersInCore = AllocMem_FPGeo_I(_numLayersInCore, (char *)("_withdlayersInCore"));
	for (j = 0; j < _numLayersInCore; j++) {
		_withdlayersInCore[j]= Assemblies[_core->asse_in_core[0]].latt_sub_width_z[j];
	}
	return _withdlayersInCore;
}

 /**
 * @ Function: GetNumCellsInAsse_layer()
 *   introduction: get number of cells in a specific layer of an assembly
 * @ Parameter List: _asseIdInCore - number of layers in core
 *                   _layerID - layer ID in assembly
 *   Return: _withdlayersInCore - 1D array, width of layers in whole Core
 * @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */

/**
* @ Function: GetNumCellsInAsse_X()
*   introduction: get number of cells in one row (x-direction) of assembly with a specific location id in core and lattice id in assembly
* @ Parameter List: _asseIdInCore - assmebly id in core
*                   _lattIdInAsse - layerID in assembly
*   Return: _numCellsInAsse_InOneRow - number of cells in one row of assembly in a specific layer
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GetNumCellsInAsse_Y()
*   introduction: get number of cells in one column (Y-direction) of assembly with a specific location id in core and lattice id in assembly
* @ Parameter List: _asseIdInCore - assmebly id in core
*                   _lattIdInAsse - layerID in assembly
*   Return: _numCellsInAsse_InOneColumn - number of cells in one column of assembly in a specific layer
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GetNumCellsInCore_X()
*   introduction: get number of cells in one row (x-direction) of core for a specific layer, based on the geometry input
* @ Parameter List: _core - struct CORE object
*                   _layerIDInCore - layerID in core
*   Return: _numCellsInCore_InOneRow - number of cells in one row of core for a specific layer
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GetNumCellsInCore_Y()
*   introduction: get number of cells in one column (y-direction) of core for a specific layer, based on the geometry input
* @ Parameter List: _core - struct CORE object
*                   _layerIDInCore - layerID in core
*   Return: _numCellsInCore_InOneColumn - number of cells in one column of core for a specific layer
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GetGlobalCellID_2D()
*   introduction: coordinate transformation, get global cell id base on layer id, assembly id in core and cell id in assembly
* @ Parameter List: _asseIdInCore - assembly ID in core
*                   _cellIdInAsse - cell ID in core
*                   _layerIDInCore - layer ID in core
*                   _core - struct CORE  object
*   Return: _globalCellId - global cell ID
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/

int GetGlobalCellID_2D(int _asseIdInCore, int _cellIdInAsse, int _layerIDInCore)
{
	int i, j;
	int _globalCellId = 0;
	for (i = 0; i<_asseIdInCore / (Core->num_X); i++) {
		for (j = 0; j<Core->num_X; j++) {
			_globalCellId = _globalCellId + GetNumCellsInAsse_layer(i*(Core->num_X) + j, _layerIDInCore);
		}
	}
	for (i = 0; i<_cellIdInAsse / GetNumCellsInAsse_X(_asseIdInCore, _layerIDInCore); i++) {
		for (j = _asseIdInCore / (Core->num_X)*(Core->num_X); \
			j<_asseIdInCore / (Core->num_X)*(Core->num_X) + (Core->num_X); j++) {
			_globalCellId = _globalCellId + GetNumCellsInAsse_X(j, _layerIDInCore);
		}
	}
	for (i = _asseIdInCore / (Core->num_X)*(Core->num_X); \
		i<_asseIdInCore / (Core->num_X)*(Core->num_X) + _asseIdInCore % (Core->num_X); i++) {
		_globalCellId = _globalCellId + GetNumCellsInAsse_X(i, _layerIDInCore);
	}
	_globalCellId = _globalCellId + _cellIdInAsse % GetNumCellsInAsse_X(_asseIdInCore, _layerIDInCore);

	return _globalCellId;
}

// get number of cells in X directin of whole computational domain (core), sumation of all subdoamins per layer
/**
* @ Function: GetNumCellsInDomain_X()
*   introduction: get number of cells in X directin of whole computational domain (core) based on the input parallel setting, 
*                 sumation of all subdoamins in a specific layer
* @ Parameter List: _numSubDomains - number of subdomains in x/y/z directions, respectively, user defined in input file
*                   _numPinsInSubDomain - number of pins in x/y/z directions of each subdomain, respectively, user defined in input file
*   Return: _counter - number of cells in X directin of whole computational domain
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GetNumCellsInDomain_Y()
*   introduction: get number of cells in y directin of whole computational domain (core), sumation of all subdoamins in a specific layer
* @ Parameter List: _numSubDomains - number of subdomains in x/y/z directions, respectively, user defined in input file
*                   _numPinsInSubDomain - number of pins in x/y/z directions of each subdomain, respectively, user defined in input file
*   Return: _counter - get number of cells in y directin of whole computational domain 
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
// get number of effective subdomains(processes) in a axial layer
/**
* @ Function: GetGlobRowIDBaseGlobPinID()
*   introduction: get global column id based on global pin id, global column is counted starting from the left bondary of pinCells 2D map
* @ Parameter List: _globPinID - global pin ID
*                   _numSubDomains - number of subdomains in x/y/z directions, respectively, user defined in input file
*                   _numPinsInSubDomain - number of pins in x/y/z directions of each subdomain, respectively, user defined in input file
*   Return: subDomain column id in parallel 2D subDomain IDs map
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
/**
* @ Function: GetGlobColumnIDBaseGlobPinID()
*   introduction: get global column id based on global pin id, global column is counted starting from the left bondary of pinCells 2D map
* @ Parameter List: _globPinID - global pin ID
*                   _numSubDomains - number of subdomains in x/y/z directions, respectively, user defined in input file
*                   _numPinsInSubDomain - number of pins in x/y/z directions of each subdomain, respectively, user defined in input file
*   Return: subDomain column id in parallel 2D subDomain IDs map
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GetPinRowIDInSubDomBaseGlobPinID()
*   introduction: get pin row id (X-direction) in subdomain (one layer) based on global pin id
* @ Parameter List: _globPinID - global pin ID in 2D
*                   _numSubDomains - number of subdomains in x/y/z directions, respectively, user defined in input file
*                   _numPinsInSubDomain - number of pins in x/y/z directions of each subdomain, respectively, user defined in input file
*   Return: pin row id in subdomain (2D)
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GetPinColumnIDInSubDomBaseGlobPinID()
*   introduction: get pin column id (Y-direction) in subdomain (one layer) based on global pin id
* @ Parameter List: _globPinID - global pin ID in 2D
*                   _numSubDomains - number of subdomains in x/y/z directions, respectively, user defined in input file
*                   _numPinsInSubDomain - number of pins in x/y/z directions of each subdomain, respectively, user defined in input file
*   Return: pin column id in subdomain (2D)
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GetPinSeqIDInSubDomBaseGlobPinID()
*   introduction: get pin sequence id in subdomain (one layer) based on global pin id
* @ Parameter List: _globPinID - global pin ID in 2D
*                   _numSubDomains - number of subdomains in x/y/z directions, respectively, user defined in input file
*                   _numPinsInSubDomain - number of pins in x/y/z directions of each subdomain, respectively, user defined in input file
*   Return: pin sequence id in subdomain (2D)
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int GetPinSeqIDInSubDomBaseGlobPinID(int _globPinID, int *_numSubDomains, int **_numPinsInSubDomain)
{
	int _pinRowID, _pinColID, _subDomColID, _pinSequenceIDInSubDom;
	_pinSequenceIDInSubDom = 0;
	_pinRowID = GetPinRowIDInSubDomBaseGlobPinID(_globPinID, _numSubDomains, _numPinsInSubDomain);
	_pinColID = GetPinColumnIDInSubDomBaseGlobPinID(_globPinID, _numSubDomains, _numPinsInSubDomain);
	_subDomColID = GetSubDomIDBaseGlobPinID_X(_globPinID, _numSubDomains, _numPinsInSubDomain);
	_pinSequenceIDInSubDom = _pinRowID*_numPinsInSubDomain[0][_subDomColID] + _pinColID;
	return _pinSequenceIDInSubDom;
}

/**
* @ Function: GetSubDomIDBaseGlobPinID_Y()
*   introduction: get subdomain column id based on global pin id
* @ Parameter List: _globPinID - global pin ID
*                   _numSubDomains - number of subdomains in x/y/z directions, respectively, user defined in input file
*                   _numPinsInSubDomain - number of pins in x/y/z directions of each subdomain, respectively, user defined in input file
*   Return: subDomain column id in parallel 2D subDomain IDs map
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GetSubDomIDBaseGlobPinID_X()
*   introduction: get subdomain column id based on global pin id
* @ Parameter List: _globPinID - global pin ID
*                   _numSubDomains - number of subdomains in x/y/z directions, respectively, user defined in input file
*                   _numPinsInSubDomain - number of pins in x/y/z directions of each subdomain, respectively, user defined in input file
*   Return: subDomain row id in parallel 2D subDomain IDs map
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GetSubDomSequenceID()
*   introduction: return the subdomain sequence id in parallel 2D subDomain IDs map based on the _rowID and _ColumnID
* @ Parameter List: _procID - current process ID
*                   _rowID - row id of subDomain in parallel 2D subDomain IDs map
*                   _ColumnID - column id of subDomain in parallel subDomain IDs map
*                   _numSubDomains - number of subdomains in x/y/z directions, respectively, user defined in input file
*   Return: subDomain sequence id in parallel 2D subDomain IDs map
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int GetSubDomSequenceID(int _rowID, int _ColumnID, int *_numSubDomains)
{
	return (_rowID*_numSubDomains[0] + _ColumnID);
}

/**
* @ Function: GetSubDomIDBaseGlobPinID()
*   introduction: get subdomain sequence id based on global pin id
* @ Parameter List: _globPinID - global pin ID
*                   _numSubDomains - number of subdomains in x/y/z directions, respectively, user defined in input file
*                   _numPinsInSubDomain - number of pins in x/y/z directions of each subdomain, respectively, user defined in input file
*   Return: subDomain sequence id in parallel 2D subDomain IDs map
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int GetSubDomIDBaseGlobPinID(int _globPinID, int *_numSubDomains, int **_numPinsInSubDomain)
{
	int _subDomID_X, _subDomID_Y, _subDomSeqID;
	_subDomSeqID = 0;
	_subDomID_X = GetSubDomIDBaseGlobPinID_X(_globPinID, _numSubDomains, _numPinsInSubDomain);
	_subDomID_Y = GetSubDomIDBaseGlobPinID_Y(_globPinID, _numSubDomains, _numPinsInSubDomain);
	_subDomSeqID = GetSubDomSequenceID(_subDomID_Y, _subDomID_X, _numSubDomains);
	return _subDomSeqID;
}

// get boundary surface of subdomain in 2D, x=x0, y=y0, left/top/right/bottom
// get process id based on subdomain row and column id 
// get process id based on subdomain row , column id and layer id 
// get subdomain row id based on subdomain sequence id 
// get subdomain column id based on subdomain sequence id 
// get process id based on subdomain sequence id 
/**
* @ Function: GetSubDomLocIDBasedProcID()
*   introduction: get the subdomain location id in parallel 3D subDomain IDs map based on the process id
* @ Parameter List: _procID - current process ID
*                   _numSubDoms - number of subdomains in x/y/z directions, respectively, user defined in input file
*                   _subDomainsMap - 3D subDomain IDs map, user defined in input file
*                   _rowID - stored row id of subDomain in parallel 3D subDomain IDs map, X-direction
*                   _colID - stored column id of subDomain in parallel subDomain IDs map, Y-direction
*                   _layID - stored layer id of subDomain in parallel subDomain IDs map, Z-direction
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void GetSubDomLocIDBasedProcID(int _procID, int *_numSubDoms, int ***_subDomainsMap, \
	int *_rowID, int *_colID,int *_layID)
{
	int _row, _col, _lay, _bool;
	_row = 0;
	_col = 0;
	_lay = 0;
	_bool = 0;
	for (_lay = 0; _lay < _numSubDoms[2]; _lay++) {
		for (_row = 0; _row < _numSubDoms[1]; _row++) {
			for (_col = 0; _col < _numSubDoms[0]; _col++) {
				if (_subDomainsMap[_lay][_row][_col] - 1 == _procID) {
					_bool = 1;					
					break;
				}
			}
			if (_bool == 1) {
				break;
			}
		}
		if (_bool == 1) {
			break;
		}
	}
	*_rowID = _row;
	*_colID = _col;
	*_layID = _lay;
}

// get  subdomain row id based on process id
// get  subdomain column id based on process id
// get subdomain layer id based on proc id
// get number of cells in subdom x direction based on process id
// get number of cells in subdom y direction based on process id
/**
* @ Function: GetNumCellsInSubDom()
*   introduction: get the number of cells in X/Y/Z directions in a specific subdomian based on the sudomain location IDs
* @ Parameter List: _subDomLocIDRow - row id of current subDomain in parallel 3D subDomain IDs map
*                   _subDomLocIDCol - column id of current subDomain in parallel 3D subDomain IDs map
*                   _subDomLocIDLay - lay id of current subDomain in parallel 3D subDomain IDs map
*                   _numPinsInSubDomain - number of pins in subdomains, 2D array
*                   _MOC_numCellsInSubDom_X - number of cells in current subdomain in X-direction
*                   _MOC_numCellsInSubDom_Y - number of cells in current subdomain in Y-direction
*                   _MOC_numCellsInSubDom_Z - number of cells in current subdomain in Z-direction
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void GetNumCellsInSubDom(int _subDomLocIDRow, int _subDomLocIDCol,int _subDomLocIDLay, int **_numPinsInSubDomain, \
	int *_MOC_numCellsInSubDom_X, int *_MOC_numCellsInSubDom_Y, int *_MOC_numLayersInsubDom_Z)
{
	*_MOC_numCellsInSubDom_X = _numPinsInSubDomain[0][_subDomLocIDCol];
	*_MOC_numCellsInSubDom_Y = _numPinsInSubDomain[1][_subDomLocIDRow];
	*_MOC_numLayersInsubDom_Z = _numPinsInSubDomain[2][_subDomLocIDLay];
// #if defined(M_2DMOC)
//     *_MOC_numLayersInsubDom_Z = 1;
// #endif // defined(M_2DMOC)
}

void GetGlobPinCellOffset_SubDom(int _subDomLocID_2D_Y, int _subDomLocID_2D_X, int **_numPinsInSubDomain, \
	int *_subDomOrgPinOffset_2D_X, int *_subDomOrgPinOffset_2D_Y)
{
	int i;
	*_subDomOrgPinOffset_2D_X = 0;
	*_subDomOrgPinOffset_2D_Y = 0;
	for (i = 0; i < _subDomLocID_2D_X; i++) {
		(*_subDomOrgPinOffset_2D_X) += _numPinsInSubDomain[0][i];
	}
	for (i = 0; i < _subDomLocID_2D_Y; i++) {
		(*_subDomOrgPinOffset_2D_Y) += _numPinsInSubDomain[1][i];
	}
}

/**
* @ Function: GetSumNumFSRsInRTM()
*   introduction: get the sumation of FSRs in ray tracing module.
* @ Parameter List: _FSRsAreaInRTM -
*                   _numRTMGeoType -
*                   _numFSRsInRTM -
*   Return: void
* @ Created     : 2019/06/12, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void GetSumNumFSRsInRTM(int *_SumNumFSRsInRTM, int _numRTMGeoType, int *_numFSRsInRTM)
{
	int i, _count;
	_count = 0;
	for (i = 0; i < _numRTMGeoType; i++) {
		_SumNumFSRsInRTM[i] = _count;
		_count += _numFSRsInRTM[i];
	}
}

/**
* @ Function: TransOneOverFSRVolInTRM_RayWised()
*   introduction: transform the original FSR related volume to a new assigned array with ray-wised ID
* @ Parameter List: _numTrackSet - number of track set
*                   _numRTMGeoType - number of ray tracing module geometry type
*                   _numAzimAng - number of the azimuthal angle in [2,2PI]
*   Return: void
* @ Created     : 2019/06/12, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void FreeGeoMem()
{
    // TODO: free memory of 'CellRTMType_SubDom' and 'WidthLayersInCore'
    if (InputTrackFile != 1){
        FreeCores(1, Core, (char *)("Core"));
//        CellRTMType_SubDom = FreeMem_i32_I(CellRTMType_SubDom,(char *)("CellRTMType_SubDom"));
//        WidthLayersInCore = FreeMem_FPGeo_I(WidthLayersInCore,(char *)("_withdlayersInCore"));
    }
    CellRTMType_SubDom = FreeMem_i32_I(CellRTMType_SubDom,(char *)("CellRTMType_SubDom"));
    WidthLayersInCore = FreeMem_FPGeo_I(WidthLayersInCore,(char *)("_withdlayersInCore"));

	FreeParaCyls(Cylinders,(char *)("Cylinders"));
	FreeCell(NumPrimitives, Primitives, (char *)("Primitives"));
	FreePinCells(NumCells, Cells, (char *)("Cells"));
	FreeLattices(NumLattices, Lattices, (char *)("Lattices"));
	FreeAssemblies(NumAssemblies, Assemblies, (char *)("Assemblies"));
//	FreeCores(1, Core, (char *)("Core"));
	
//	CellRTMType_SubDom = FreeMem_i32_I(CellRTMType_SubDom,(char *)("CellRTMType_SubDom"));
	if (Gene_VisualFile==1) {
		CellTypeID_visu = FreeMem_int_II(CellTypeID_visu, MOCNumLayersZ, (char *)("CellTypeID_visu"));
	}
	SubDomGlobLayerID = FreeMem_i32_I(SubDomGlobLayerID,(char *)("subDomGlobLayerID"));
//	WidthLayersInCore = FreeMem_FPGeo_I(WidthLayersInCore,(char *)("_withdlayersInCore"));
	SubDomGlobCellID = FreeMem_i32_I(SubDomGlobCellID, (char *)(CHAR_NONE));
}
