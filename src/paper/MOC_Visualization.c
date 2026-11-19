/* Visualization state referenced by the common geometry/input interfaces.
 * VTK generation itself is outside the publication transport scope. */
#include "MOC_Visualization.h"

FP_PRECISION ****ConstructPoints = NULL;
FP_PRECISION **RadiusPerRing_RTM_Mat = NULL;
FP_PRECISION **RadiusPerRing_RTM_Mesh = NULL;
int *NumPoints_RTM = NULL;
int Gene_VisualFile = 0;
int NumPointsPerRing = 0;
int NumPointsPerOctant = 0;
int MaxNumSecs_Visu = 0;
int *NumRings_RTM_Mat = NULL;
int *NumRings_RTM_Mesh = NULL;
int **NumSectorsPerRing_RTM_Mat = NULL;
int **NumSectorsPerRing_RTM_Mesh = NULL;
int **RTMType_Lattice = NULL;
int **MatID_RTM = NULL;
int MaxMatID = 0;
int *RTMMeshType = NULL;
int *NumCellsInRTM = NULL;
int *NumCellsInPin = NULL;
int **CellTypeID_visu = NULL;

/* The transport snapshot fixes Gene_VisualFile to zero.  Keep the two
 * conditional call targets linkable without carrying the VTK implementation. */
void GeoInfoInit_Visu(int _numPinCell, struct PIN_CELL *_pinCell,
    int _numPrimitives, struct CELL *_primitives)
{
    (void)_numPinCell;
    (void)_pinCell;
    (void)_numPrimitives;
    (void)_primitives;
}

void GetRadiusPerRing_RTM_Mesh(int _numRTMGeoType,
    struct PIN_CELL *_cellBasedRTMGeo, int *_RTMMeshType,
    FP_PRECISION **_radiusPerRing_RTM_Mesh, int *_numRings_RTM_Mesh)
{
    (void)_numRTMGeoType;
    (void)_cellBasedRTMGeo;
    (void)_RTMMeshType;
    (void)_radiusPerRing_RTM_Mesh;
    (void)_numRings_RTM_Mesh;
}
