/********************************************************************************/
/* Publication snapshot: transport-only surface helpers.                        */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>

#include "MOC_CommonMacro.h"
#include "MOC_Input.h"
#include "MOC_Output.h"
#include "MOC_Surface.h"

enum surfaceType SetSurfaceType(int _surfType)
{
    enum surfaceType _surf = PLANE;

    switch (_surfType) {
    case 0: _surf = PLANE; break;
    case 1: _surf = XPLANE; break;
    case 2: _surf = YPLANE; break;
    case 3: _surf = ZPLANE; break;
    case 4: _surf = XCYLINDER; break;
    case 5: _surf = YCYLINDER; break;
    case 6: _surf = ZCYLINDER; break;
    case 7: _surf = QUADRATIC; break;
    default:
        printf("\n [**ERROR**]  No Such Kind of Surface Type, %d. \n", _surfType);
        fflush(stdout);
        exit(69);
    }
    return _surf;
}

void CalcParaCylinderCoeff(struct PARA_CYLINDER *_pointer)
{
    switch (_pointer->surface_type) {
    case XCYLINDER:
        _pointer->A = 0.; _pointer->B = 1.; _pointer->C = 1.; _pointer->D = 0.;
        _pointer->E = -2. * _pointer->center.xyz[1];
        _pointer->F = -2. * _pointer->center.xyz[2];
        _pointer->G = _pointer->center.xyz[1] * _pointer->center.xyz[1]
                    + _pointer->center.xyz[2] * _pointer->center.xyz[2]
                    - _pointer->radius * _pointer->radius;
        break;
    case YCYLINDER:
        _pointer->A = 1.; _pointer->B = 0.; _pointer->C = 1.;
        _pointer->D = -2. * _pointer->center.xyz[0]; _pointer->E = 0.;
        _pointer->F = -2. * _pointer->center.xyz[2];
        _pointer->G = _pointer->center.xyz[0] * _pointer->center.xyz[0]
                    + _pointer->center.xyz[2] * _pointer->center.xyz[2]
                    - _pointer->radius * _pointer->radius;
        break;
    case ZCYLINDER:
        _pointer->A = 1.; _pointer->B = 1.; _pointer->C = 0.;
        _pointer->D = -2. * _pointer->center.xyz[0];
        _pointer->E = -2. * _pointer->center.xyz[1]; _pointer->F = 0.;
        _pointer->G = _pointer->center.xyz[0] * _pointer->center.xyz[0]
                    + _pointer->center.xyz[1] * _pointer->center.xyz[1]
                    - _pointer->radius * _pointer->radius;
        break;
    default:
        printf("\n [**ERROR**]  No Such Kind of Surface Type, %d.\n",
               _pointer->surface_type);
        fflush(stdout);
        exit(69);
    }
}

void FreeParaCyls(struct PARA_CYLINDER *_paraCylinder, char *_parameter)
{
    OutputFreeMemory(_parameter);
    free(_paraCylinder);
}
