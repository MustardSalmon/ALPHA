
#ifndef MOC_SURFACE_H
#define MOC_SURFACE_H
#include "MOC_Boundary.h"
#include "MOC_Point.h"
#include "MOC_CommonMacro.h"

/**
 * @enum surfaceType
 * @introduction The types of surfaces.
 */
    enum surfaceType {
        /** A general plane 0 */
        PLANE,
        /** A plane perpendicular to the x-axis 1 */
        XPLANE,
        /** A plane perpendicular to the y-axis 2 */
        YPLANE,
        /** A plane perpendicular to the x-axis 3 */
        ZPLANE,
        /** A cylinder with axis parallel to the x-axis 4 */
        XCYLINDER,
        /** A cylinder with axis parallel to the y-axis 5 */
        YCYLINDER,
        /** A cylinder with axis parallel to the z-axis 6 */
        ZCYLINDER,
        /** A generalized quadratic surface 7 */
        QUADRATIC
    };

/**
 * @struct SURFACE (PLANE, PARA_CYLINDER)
 * @introduction Represents a general Surface in 3D.
 * @explanation The SURFACE struct is used to define the geometry
 *              for the simulation using a constructive solid geometry
 *              (CSG) formalism. Surfaces are used during ray tracing
 *              of charateristic Tracks across the geometry.
 */

 /**
 * @struct PLANE
 * @introduction Represents a general Plane.
 */
    struct PLANE {
        /** A user-defined id for each surface created */
        int id;
        /** The coefficient for the linear term in x */
        FP_PRECISION_GEO A;
        /** The coefficient for the linear term in y */
        FP_PRECISION_GEO B;
        /** The coefficient for the linear term in z */
        FP_PRECISION_GEO C;
        /** The constant offset */
        FP_PRECISION_GEO D;
        /** The type of Plane (ie, PLANE, XPLANE, YPLANE, ZPLANE) */
        enum surfaceType surface_type;
        /** The type of boundary condition to be used for this surface
         * (ie, VACUUM or REFLECTIVE) */
        enum boundaryType boundary_type;
    };

/**
 * @struct PARA_CYLINDER
 * @introduction Represents a Cylinder with axis parallel to the some-axis.
 */
    struct PARA_CYLINDER {
        /** A user-defined id for each surface created */
        int id;
        /** The type of surface (ie, XCYLINDER, YCYLINDER, ZCYLINDER) */
        enum surfaceType surface_type;
        /** The type of boundary condition to be used for this surface
         * (ie, VACUUM or REFLECTIVE) */
        enum boundaryType boundary_type;
        /** A point for the PARA_CYLINDER's center */
        struct POINT center;
        /** The PARA_CYLINDER's radius */
        FP_PRECISION_GEO radius;
        /** The coefficient of the x-squared term */
        FP_PRECISION_GEO A;
        /** The coefficient of the y-squared term */
        FP_PRECISION_GEO B;
        /** The coefficient of the z-squared term */
        FP_PRECISION_GEO C;
        /** The coefficient of the linear term in x */
        FP_PRECISION_GEO D;
        /** The coefficient of the linear term in y */
        FP_PRECISION_GEO E;
        /** The coefficient of the linear term in z */
        FP_PRECISION_GEO F;
        /** The constant offset */
        FP_PRECISION_GEO G;
    };
    void PrintSurfaceType(FILE *fp,int surf_type);
    enum surfaceType SetSurfaceType(int _surf_type);
    void CalcParaCylinderCoeff(struct PARA_CYLINDER *pointer);
    void FreePlanes(struct PLANE *_plane,char *_parameter);
    void FreeParaCyls(struct PARA_CYLINDER *_paraCylinder,char *_parameter);
#endif // Surface_H
