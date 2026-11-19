/********************************************************************************/
/* Publication snapshot: transport-only boundary helpers.                      */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>

#include "MOC_Boundary.h"
#include "MOC_CommonMacro.h"
#include "MOC_Input.h"

enum boundaryType SetBoundaryType(M_INT32 _boundType)
{
    enum boundaryType _bound = REFLECTIVE;

    switch (_boundType) {
    case 0:
        _bound = VACUUM;
        break;
    case 1:
        _bound = REFLECTIVE;
        break;
    case 2:
        _bound = PERIODIC;
        break;
    case 3:
        _bound = BOUNDARY_NONE;
        break;
    case 4:
        _bound = EXTRAPOLATION;
        break;
    case 5:
        _bound = ZERO_SURFACE_FLUX;
        break;
    default:
        printf("\n [**ERROR**]  No Such Kind of Boundary Type, %d. \n", _boundType);
        fflush(stdout);
        exit(69);
    }
    return _bound;
}

FP_PRECISION RadialBoundary(M_INT32 _angleInID, M_INT32 _angleOutID,
                            M_INT32 _numAzimAng)
{
    FP_PRECISION _beta = 1.0;
    M_INT32 _qurtNumAzim = _numAzimAng / 4;

    if (_angleInID >= 0 && _angleInID < _qurtNumAzim) {
        if (_angleOutID >= 3 * _qurtNumAzim &&
            _angleOutID < 4 * _qurtNumAzim) {
            if (NorthBC == VACUUM) {
                _beta = 0.0;
            } else if (NorthBC != REFLECTIVE) {
                printf("Fail to judge radial boundary condition from _angleInID=%d to _angleOutID=%d !\n",
                       _angleInID, _angleOutID);
            }
        } else if (_angleOutID >= _qurtNumAzim &&
                   _angleOutID < 2 * _qurtNumAzim) {
            if (EastBC == VACUUM) {
                _beta = 0.0;
            } else if (EastBC != REFLECTIVE) {
                printf("Fail to judge radial boundary condition from _angleInID=%d to _angleOutID=%d !\n",
                       _angleInID, _angleOutID);
            }
        }
    } else if (_angleInID >= _qurtNumAzim &&
               _angleInID < 2 * _qurtNumAzim) {
        if (_angleOutID >= 2 * _qurtNumAzim &&
            _angleOutID < 3 * _qurtNumAzim) {
            if (NorthBC == VACUUM) {
                _beta = 0.0;
            } else if (NorthBC != REFLECTIVE) {
                printf("Fail to judge radial boundary condition from _angleInID=%d to _angleOutID=%d !\n",
                       _angleInID, _angleOutID);
            }
        } else if (_angleOutID >= 0 && _angleOutID < _qurtNumAzim) {
            if (WestBC == VACUUM) {
                _beta = 0.0;
            } else if (WestBC != REFLECTIVE) {
                printf("Fail to judge radial boundary condition from _angleInID=%d to _angleOutID=%d !\n",
                       _angleInID, _angleOutID);
            }
        }
    } else if (_angleInID >= 2 * _qurtNumAzim &&
               _angleInID < 3 * _qurtNumAzim) {
        if (_angleOutID >= _qurtNumAzim &&
            _angleOutID < 2 * _qurtNumAzim) {
            if (SouthBC == VACUUM) {
                _beta = 0.0;
            } else if (SouthBC != REFLECTIVE) {
                printf("Fail to judge radial boundary condition from _angleInID=%d to _angleOutID=%d !\n",
                       _angleInID, _angleOutID);
            }
        } else if (_angleOutID >= 3 * _qurtNumAzim &&
                   _angleOutID < 4 * _qurtNumAzim) {
            if (WestBC == VACUUM) {
                _beta = 0.0;
            } else if (WestBC != REFLECTIVE) {
                printf("Fail to judge radial boundary condition from _angleInID=%d to _angleOutID=%d !\n",
                       _angleInID, _angleOutID);
            }
        }
    } else if (_angleInID >= 3 * _qurtNumAzim &&
               _angleInID < 4 * _qurtNumAzim) {
        if (_angleOutID >= 0 && _angleOutID < _qurtNumAzim) {
            if (SouthBC == VACUUM) {
                _beta = 0.0;
            } else if (SouthBC != REFLECTIVE) {
                printf("Fail to judge radial boundary condition from _angleInID=%d to _angleOutID=%d !\n",
                       _angleInID, _angleOutID);
            }
        } else if (_angleOutID >= 2 * _qurtNumAzim &&
                   _angleOutID < 3 * _qurtNumAzim) {
            if (EastBC == VACUUM) {
                _beta = 0.0;
            } else if (EastBC != REFLECTIVE) {
                printf("Fail to judge radial boundary condition from _angleInID=%d to _angleOutID=%d !\n",
                       _angleInID, _angleOutID);
            }
        }
    }

    return _beta;
}
