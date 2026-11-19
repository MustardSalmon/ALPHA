#ifndef MOC_CELL_H
#define MOC_CELL_H
#include "MOC_CommonMacro.h"
#include "MOC_Material.h"
#include "MOC_Surface.h"

#define NUM_BOUND_SURFACES 8  /** The number of bounding surfaces for a cell */

/**
 * @struct SURFACE_HALFSPACE
 * @introduction A SURFACE_HALFSPACE represents a surface with associated
 *               halfspace. A Linked List for storing the bounding surfaces of a cell
 */
    // define a union for storing the PLANE surface or PARA_CYLINDER surface.
    union Surfaces {
            struct PLANE _plane;
            struct PARA_CYLINDER _paraCylinder;
    };

    struct SURFACE_HALFSPACE {
        /** the type of union Surface */
        enum surfaceType surf_type;
        /** A  union object */
        union Surfaces bound_surf;
        /** The halfspace associated with this surface */
        FP_PRECISION halfspace;
        struct SURFACE_HALFSPACE *next;
    };

/**
 * @struct CELL
 * @introduction Represents a Cell inside of a PIN_CELL.
 */
    struct CELL {
        /** A type_id for each cell created */
        M_INT32 id;
        /** A boolean indicator to identify whether this cell belong to GPU_TSKernelType rod*/
        M_INT32 is_cr;
        /** the id of material filling this Cell */
        M_INT32 mat_id;
        /** Linked List of bounding surface IDs with surfaces and halfspaces (+/-1) */
        struct SURFACE_HALFSPACE *bound_surfaces;
        /** The number of rings sub-dividing this cell */
        M_INT32 num_rings;
        /** The number of sectors sub-dividing this cell */
        M_INT32 num_sectors;
        /** The last two parameters are used to sub-divide the refined-reflector pinCell*/
        /** The number of segments which equivalently sub-divides this cell along X-axis*/
        M_INT32 numX_mesh;
        /** The number of segments which equivalently sub-divides this cell along X-axis*/
        M_INT32 numY_mesh;
    };

    struct SURFACE_HALFSPACE *AddPlaneToList(struct SURFACE_HALFSPACE *_list, \
		struct PLANE _plane, FP_PRECISION _halfspace);
    struct SURFACE_HALFSPACE *AddCylinderToList(struct SURFACE_HALFSPACE *_list, \
		struct PARA_CYLINDER _cyl, FP_PRECISION _halfspace);
    void CopyCell(struct CELL *_ectypeCell, struct CELL *_origCell);
    void PrintBoundSurfInCell(FILE *_fp,struct SURFACE_HALFSPACE *_surf_half);
	void FreeCell(M_INT32 _n, struct CELL *_cell,char *_parameter);
	void FreeSurfHalfInCell(struct CELL *_cell);
	M_INT32 GetByteSizeStrCELL(M_INT32 _numPrimitives, struct CELL *_primitives);
#endif // CELL_H
