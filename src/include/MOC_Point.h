
#ifndef MOC_POINT_H
#define MOC_POINT_H
#include "MOC_CommonMacro.h"
//#define POINT_DIM 3
/**
 * @struct POINT
 * @introduction Represents a 3D point in space.
 */
    struct POINT {
        FP_PRECISION_GEO xyz[3];
    };
void SetCoords(struct POINT *_point,FP_PRECISION_GEO _x,FP_PRECISION_GEO _y,FP_PRECISION_GEO _z);
FP_PRECISION_GEO PointGetX(struct POINT *_point);
FP_PRECISION_GEO PointGetY(struct POINT *_point);
FP_PRECISION_GEO PointGetZ(struct POINT *_point);
struct POINT ***AllocMemoryOfPoints(int _numTrackSet, int *_numAzimAng, int**_numTracks, char *_parameter);
struct POINT ***FreeMemoryOfPoints(struct POINT ***_pointer, int _dim1, int *_dim2, char *_parameter);
#endif // POINT_H
