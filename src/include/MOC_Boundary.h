
#ifndef MOC_BOUNDARY_H
#define MOC_BOUNDARY_H
#include <stdio.h>
#include "MOC_CommonMacro.h"
/**
 * @enum boundaryType
 * @brief The types of boundary conditions supported for surfaces.
 */
enum boundaryType {
	/** A vacuum boundary condition */
	VACUUM,
	/** A reflective boundary condition */
	REFLECTIVE,
	/** A periodic boundary condition */
	PERIODIC,
	/** No boundary type (typically an interface between flat source regions) */
	BOUNDARY_NONE,
	//
	EXTRAPOLATION,
	ZERO_SURFACE_FLUX
    };

    void PrintBoundaryType(FILE *fp,M_INT32 boun_type);
	enum boundaryType SetBoundaryType(M_INT32 _bound_type);
    FP_PRECISION RadialBoundary(M_INT32 _angleInID, M_INT32 _angleOutID,M_INT32 _numAzimAng);
	M_INT32 SwitchBC(M_INT32 _moc_bc);
#endif // BOUNDARY_H