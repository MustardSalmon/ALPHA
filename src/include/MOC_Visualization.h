#ifndef MOC_VISUALIZATION
#define MOC_VISUALIZATION
#include "MOC_Assembly.h"
#include "MOC_Lattice.h"
#include "MOC_Pin_Cell.h"
#define MAX_POINT_ITER 100

extern FP_PRECISION ****ConstructPoints, **RadiusPerRing_RTM_Mat, **RadiusPerRing_RTM_Mesh;
extern int *NumPoints_RTM, Gene_VisualFile, NumPointsPerRing, NumPointsPerOctant, MaxNumSecs_Visu; // number of point per ring
extern int *NumRings_RTM_Mat, *NumRings_RTM_Mesh, **NumSectorsPerRing_RTM_Mat, **NumSectorsPerRing_RTM_Mesh, **RTMType_Lattice;
extern int **MatID_RTM, MaxMatID, *RTMMeshType, *NumCellsInRTM,*NumCellsInPin, *CoreAsseTypeID,**CellTypeID_visu;

void GeoInfoInit_Visu(int _numPinCell, struct PIN_CELL *_pinCell, int _numPrimitives, struct CELL *_primitives);
void GetRadiusPerRing_RTM_Mesh(int _numRTMGeoType, struct PIN_CELL *_cellBasedRTMGeo,int *_RTMMeshType, \
	FP_PRECISION **_radiusPerRing_RTM_Mesh,int *_numRings_RTM_Mesh);
void SortVectorInDecrease(int _num, FP_PRECISION *_vec);
int GetNumPointsPerSec_Visu(int _maxNumSec_visu);
int *GetNumPointsInRTM(int _objVis, int _numRTMGeoType,int *_RTMMeshType,int * _numRings_RTM_Mat,int * _numRings_RTM_Mesh,\
    int _numPointsPerRing,int **_numSectorsPerRing_RTM_Mesh,char *_parameter);
int GetMaxNumberSectorsInPrimitive(int _numPrimitives, struct CELL *_primitives);
void GenerateVTKFile_Visu(int _objVis, int _procID, int _numPointsPerRing, int _numRTMGeoType, struct PIN_CELL *_cellBasedRTMGeo,int g,\
    FP_PRECISION **_scalarFlux, FP_PRECISION **_normPinPower,int *_RTMMeshType,int *_numRings_RTM_Mat,int * _numRings_RTM_Mesh,\
    int **_numSectorsPerRing_RTM_Mesh,FP_PRECISION **_radiusPerRing_RTM_Mat,FP_PRECISION **_radiusPerRing_RTM_Mesh,\
	int *_numSubDomains,int **_numPin_subDomain,FP_PRECISION *_widthLayersInCore,int **_cellRTMType_SubDom,int _numCellsPerLayer,\
	int *_subDomGlobCellID,int **_cellTypeID_SubDom,int _numCells,struct PIN_CELL *_cells, int **_matID_RTM, int *_numCellsInRTM,\
	int _numAssemblies,struct ASSEMBLY *_assemblies, int _numLattices,struct LATTICE *_lattices,int ***_indexMOCGlobFSRID,int *_numCellsInPin);
FP_PRECISION ****GetConstructPoints(int _objVis, int _numPointsPerRing, int _numRTMGeoType, struct PIN_CELL *_cellBasedRTMGeo,\
    int *_numRings_RTM_Mat,int *_numRings_RTM_Mesh,FP_PRECISION **_radiusPerRing_RTM_Mat,FP_PRECISION **_radiusPerRing_RTM_Mesh,\
	int *_RTMMeshType,int **_numSectorsPerRing_RTM_Mesh) ;
void GenerateVTKFile_LayerMultiProc(int _procID,int _objVis, int _lay, int g, int _numRTMGeoType, int _numPointsPerOct, \
    FP_PRECISION ****_constructPoints,FP_PRECISION **_scalarFlux, FP_PRECISION **_normPinPower,int *_numSubDomains,int **_numPin_subDomain, \
	int *_numRings_RTM_Mat,int *_numRings_RTM_Mesh,FP_PRECISION *_widthLayersInCore,int **_cellRTMType_SubDom,int _numCellsPerLayer,\
	int *_numPoints_RTM,int *_RTMMeshType,int **_numSectorsPerRing_RTM_Mesh,int *_subDomGlobCellID,int **_cellTypeID_SubDom,int _numCells,\
	struct PIN_CELL *_cells, int **_matID_RTM, int *_numCellsInRTM,int _numAssemblies,struct ASSEMBLY *_assemblies, int _numLattices,\
	struct LATTICE *_lattices,int ***_indexMOCGlobFSRID,int *_numCellsInPin);
void GetGlobalOffsetBasedGlobPinID(int _globPinID, struct CORE *_core, int _layerNumInCore, FP_PRECISION *_offset,int *_numSubDomains, \
	int **_numPin_subDomain, int _numAssemblies,struct ASSEMBLY *_assemblies, int _numLattices,struct LATTICE *_lattices);

#endif
