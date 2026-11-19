
#ifndef MOC_ASSEMBLY_H
#define MOC_ASSEMBLY_H
#include "MOC_CommonMacro.h"
#include "MOC_Point.h"
/**
 * @struct ASSEMBLY
 * @introduction Represents a repeating 3D Lattice
 *                which contains many pin_cells.
 */
    struct ASSEMBLY {
        /** Assembly type id */
        M_INT32 type_id;
        /** A user-defined id for each assembly created */
        M_INT32 usr_id;
        /** The number of layers along the z-axis */
        M_INT32 num_layer;
        /** The width of the lattices in this assembly (cm) along the z-axis */
        FP_PRECISION_GEO *latt_width_z;
		/** The number of sublayers in each layer */
		M_INT32 *num_sub_in_layer;
		/** The number of sublayers along the z-axis */
		M_INT32 num_sublayer;
		/** The width of the sublayers in the lattices which are in this assembly (cm) along the z-axis */
		FP_PRECISION_GEO *latt_sub_width_z;
        /** A container of lattices' type ids in  */
        M_INT32 *latt_in_asse;
    };
FP_PRECISION_GEO AssemblyGetLengthX(struct ASSEMBLY *_assembly);
FP_PRECISION_GEO AssemblyGetLengthY(struct ASSEMBLY *_assembly);
FP_PRECISION_GEO AssemblyGetLengthZ(struct ASSEMBLY *_assembly);
M_INT32 AssemblyGetNumPinCellsX(struct ASSEMBLY *_assembly, M_INT32 _layerID);
M_INT32 AssemblyGetNumPinCellsY(struct ASSEMBLY *_assembly, M_INT32 _layerID);
struct POINT *LatticeGetOffset(struct ASSEMBLY *_assembly, M_INT32 _numLayer);
void FreeLattWidthInAsse(struct ASSEMBLY *_assembly);
void FreeLattsInAsse(struct ASSEMBLY *_assembly);
void FreeAssemblies(M_INT32 _n, struct ASSEMBLY *_assembly, char *_parameter);
M_INT32 GetByteSizeStrASSEMBLY(M_INT32 _numAssemblies, struct ASSEMBLY *_assembly);

#endif // ASSEMBLY_H
