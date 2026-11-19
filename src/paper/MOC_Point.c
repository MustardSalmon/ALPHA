/********************************************************************************/
/* Publication snapshot: transport-only point helpers.                         */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "MOC_CommonMacro.h"
#include "MOC_Output.h"
#include "MOC_Point.h"

void SetCoords(struct POINT *_point, FP_PRECISION_GEO _x,
               FP_PRECISION_GEO _y, FP_PRECISION_GEO _z)
{
    _point->xyz[0] = _x;
    _point->xyz[1] = _y;
    _point->xyz[2] = _z;
}

FP_PRECISION_GEO PointGetX(struct POINT *_point)
{
    return _point->xyz[0];
}

FP_PRECISION_GEO PointGetY(struct POINT *_point)
{
    return _point->xyz[1];
}

FP_PRECISION_GEO PointGetZ(struct POINT *_point)
{
    return _point->xyz[2];
}

struct POINT ***AllocMemoryOfPoints(int _numTrackSet, int *_numAzimAng,
                                    int **_numTracks, char *_parameter)
{
    struct POINT ***_pointer = NULL;
    int _dataNumber = 0, _iter1, _iter2;
    FP_PRECISION _dataSizeMB = 0.0, _pointerSizeMB = 0.0;

    if ((_pointer = (struct POINT ***) calloc((size_t) _numTrackSet,
                                               sizeof(struct POINT **))) == NULL) {
        printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMemoryOfPoints\" - %s",
               _parameter);
        fflush(stdout);
        exit(69);
    }
    for (_iter1 = 0; _iter1 < _numTrackSet; _iter1++) {
        if ((_pointer[_iter1] = (struct POINT **) calloc((size_t) _numAzimAng[_iter1],
                                                         sizeof(struct POINT *))) == NULL) {
            printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMemoryOfPoints\" - %s[%d]",
                   _parameter, _iter1);
            fflush(stdout);
            exit(69);
        }
        if (OUTPUT_MemoryFootprint) {
            _pointerSizeMB += (FP_PRECISION) sizeof(_pointer[_iter1]);
        }
        for (_iter2 = 0; _iter2 < _numAzimAng[_iter1]; _iter2++) {
            if ((_pointer[_iter1][_iter2] = (struct POINT *) calloc(
                    (size_t) _numTracks[_iter1][_iter2], sizeof(struct POINT))) == NULL) {
                printf("\n [**ERROR**]  Memory Allocation Failed in \"AllocMemoryOfPoints\" - %s[%d][%d]",
                       _parameter, _iter1, _iter2);
                fflush(stdout);
                exit(69);
            }
            if (OUTPUT_MemoryFootprint) {
                _pointerSizeMB += (FP_PRECISION) sizeof(_pointer[_iter1][_iter2]);
                _dataNumber += _numTracks[_iter1][_iter2];
            }
        }
    }
    if (OUTPUT_MemoryFootprint) {
        _dataSizeMB = (FP_PRECISION) _dataNumber * M_MB_Unit * (FP_PRECISION) sizeof(struct POINT);
        _pointerSizeMB *= M_MB_Unit;
        if (strcmp(_parameter, CHAR_NONE) != 0) {
            OutputMemoryFootprint(FP_MemoryFootprint, "struct POINT***", _parameter,
                                  _dataNumber, _pointerSizeMB, _dataSizeMB);
        }
    }
    return _pointer;
}

struct POINT ***FreeMemoryOfPoints(struct POINT ***_pointer, int _dim1,
                                   int *_dim2, char *_parameter)
{
    int _iter1, _iter2;

    if (strcmp(_parameter, CHAR_NONE) != 0) {
        OutputFreeMemory(_parameter);
    }
    for (_iter1 = 0; _iter1 < _dim1; _iter1++) {
        for (_iter2 = 0; _iter2 < _dim2[_iter1]; _iter2++) {
            free(_pointer[_iter1][_iter2]);
        }
        free(_pointer[_iter1]);
    }
    free(_pointer);
    return NULL;
}
