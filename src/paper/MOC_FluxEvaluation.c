/********************************************************************************/
/* Minimal angular-flux initialization routines used by the publication       */
/* transport snapshot.  MPI exchange, legacy scalar-flux post-processing, and  */
/* unrelated fixed-source helpers remain outside the single-GPU 2-D scope.      */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "MOC_CommonMacro.h"
#include "MOC_FluxEvaluation.h"
#include "MOC_Input.h"
#include "MOC_Output.h"

/* These objects are part of the public MOC interface and are referenced by
 * the retained transport driver, even though MPI exchange is disabled here. */
FP_PRECISION **BoundAngFlux_send = NULL;
FP_PRECISION **BoundAngFlux_recv = NULL;
int **NextSubDomLTIDs_send = NULL;
int **NextSubDomLTIDs_recv = NULL;
int ***MPIProcID_recv = NULL;
int ***MPINeibID_recv = NULL;

int Get_NumFSRs_GlobDom(int *_numFSRs_array, int _numFSRs_subDom,
    int _procID, int _numProcs)
{
    int _numFSRs_globDom = 0;

#ifdef MPI_USED
    int _proc = 0;
    if (_procID == 0) {
        _numFSRs_array[0] = _numFSRs_subDom;
        for (_proc = 1; _proc < _numProcs; _proc++) {
            MPI_Recv(&_numFSRs_array[_proc], 1, MPI_INT, _proc, _proc,
                MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        }
    } else if (_procID != 0) {
        MPI_Send(&_numFSRs_subDom, 1, MPI_INT, 0, _procID,
            MPI_COMM_WORLD);
    }
    if (_procID == 0) {
        for (_proc = 0; _proc < _numProcs; _proc++) {
            _numFSRs_globDom += _numFSRs_array[_proc];
        }
    }
#else
    (void)_numFSRs_array;
    (void)_procID;
    (void)_numProcs;
    _numFSRs_globDom = _numFSRs_subDom;
#endif

    return _numFSRs_globDom;
}

FP_PRECISION **AllocMemInciAngFlux(int _numTrackSet,
    int *_numLTsInHalfspace, int _numGroups, int *_numPolarAng,
    int _MOCNumLayersZ, char *_parameter)
{
    int _iter1, _dataNumber = 0;
    size_t _dim;
    FP_PRECISION _dataSizeMB = 0.0;
    FP_PRECISION _pointerSizeMB = 0.0;
    FP_PRECISION **_pointer = NULL;

    _pointer = (FP_PRECISION **)calloc((size_t)_numTrackSet,
        sizeof(FP_PRECISION *));
    if (_pointer == NULL) {
        if (strcmp(_parameter, CHAR_NONE) != 0) {
            printf("\n [**ERROR**]  Memory Allocation Failed in "
                "\"AllocMemInciAngFlux\" - %s\n", _parameter);
        } else {
            printf("\n [**ERROR**]  Memory Allocation Failed in "
                "\"AllocMemInciAngFlux\" - local variable\n");
        }
        fflush(stdout);
        exit(69);
    }

    for (_iter1 = 0; _iter1 < _numTrackSet; _iter1++) {
        _dim = (size_t)_MOCNumLayersZ
            * (size_t)_numLTsInHalfspace[_iter1]
            * (size_t)(2 * _numGroups * _numPolarAng[_iter1]);
        _pointer[_iter1] = (FP_PRECISION *)calloc(_dim,
            sizeof(FP_PRECISION));
        if (_pointer[_iter1] == NULL) {
            if (strcmp(_parameter, CHAR_NONE) != 0) {
                printf("\n [**ERROR**]  Memory Allocation Failed in "
                    "\"AllocMemInciAngFlux\" - %s[%d]\n",
                    _parameter, _iter1);
            } else {
                printf("\n [**ERROR**]  Memory Allocation Failed in "
                    "\"AllocMemInciAngFlux\" - local variable [%d]\n",
                    _iter1);
            }
            fflush(stdout);
            exit(69);
        }
        _pointerSizeMB += (FP_PRECISION)sizeof(_pointer[_iter1]);
        _dataNumber += (int)_dim;
    }

    _dataSizeMB = (FP_PRECISION)_dataNumber * M_MB_Unit
        * (FP_PRECISION)sizeof(FP_PRECISION);
    _pointerSizeMB *= M_MB_Unit;
    if (strcmp(_parameter, CHAR_NONE) != 0) {
        OutputMemoryFootprint(FP_MemoryFootprint, "FP_PRECISION **",
            _parameter, _dataNumber, _pointerSizeMB, _dataSizeMB);
    }
    return _pointer;
}

void InitInciAngFlux(FP_PRECISION **_angFlux, int _numTrackSet,
    int *_numLTsInHalfspace, int _numGroups, int *_numPolarAng,
    int _MOCNumLayersZ, FP_PRECISION _value)
{
    int i;
    size_t j, _dim;

    for (i = 0; i < _numTrackSet; i++) {
        _dim = (size_t)_MOCNumLayersZ
            * (size_t)_numLTsInHalfspace[i]
            * (size_t)(2 * _numPolarAng[i]);
        for (int k = 0; k < _numGroups; k++) {
            size_t st = _dim * (size_t)k;
            size_t ed = _dim * (size_t)(k + 1);
#ifdef OMP_USED
            omp_set_num_threads(8);
#pragma omp parallel for
#endif
            for (j = st; j < ed; j++) {
                _angFlux[i][j] = _value;
            }
        }
    }
}

void InitInciAngFlux_BC(FP_PRECISION **_angFlux, int _numTrackSet,
    int _numGroups, int *_numAzimAng, int *_numPolarAng,
    int _MOCNumLayersZ, int **_NumLongTracks_SubDomX,
    int **_NumLongTracks_SubDomY)
{
    int i, _lay, _azim, _lt, _azim4, g, _pol;
    size_t _ind;
    FP_PRECISION _zero = (FP_PRECISION)M_Zero_FP;
    FP_PRECISION _one = (FP_PRECISION)M_One_FP;
    FP_PRECISION _mul_x, _mul_y;

    for (i = 0; i < _numTrackSet; i++) {
        _ind = 0;
        _azim4 = _numAzimAng[i] / 4;
        for (_lay = 0; _lay < _MOCNumLayersZ; _lay++) {
            for (_azim = 0; _azim < _numAzimAng[i]; _azim++) {
                _mul_x = _one;
                _mul_y = _one;
                if (_azim < _azim4) {
                    if (SouthBC == VACUUM) _mul_x = _zero;
                    if (WestBC == VACUUM) _mul_y = _zero;
                } else if (_azim < _azim4 * 2) {
                    if (SouthBC == VACUUM) _mul_x = _zero;
                    if (EastBC == VACUUM) _mul_y = _zero;
                } else if (_azim < _azim4 * 3) {
                    if (NorthBC == VACUUM) _mul_x = _zero;
                    if (EastBC == VACUUM) _mul_y = _zero;
                } else {
                    if (NorthBC == VACUUM) _mul_x = _zero;
                    if (WestBC == VACUUM) _mul_y = _zero;
                }

                for (_lt = 0; _lt < _NumLongTracks_SubDomX[i][_azim];
                    _lt++) {
                    for (g = 0; g < _numGroups; g++) {
                        for (_pol = 0; _pol < _numPolarAng[i]; _pol++) {
                            _angFlux[i][_ind++] *= _mul_x;
                        }
                    }
                }
                for (_lt = 0; _lt < _NumLongTracks_SubDomY[i][_azim];
                    _lt++) {
                    for (g = 0; g < _numGroups; g++) {
                        for (_pol = 0; _pol < _numPolarAng[i]; _pol++) {
                            _angFlux[i][_ind++] *= _mul_y;
                        }
                    }
                }
            }
        }
    }
}
