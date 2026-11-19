#ifndef MOC_LATTICE_H
#define MOC_LATTICE_H
#include "MOC_Point.h"
#include "MOC_Pin_Cell.h"
#include "MOC_CommonMacro.h"
/**
 * @struct LATTICE
 * @introduction Describe the 2D Lattices made up of pinCells.
 */
     struct LATTICE {
        int type_id;
        /** A user-defined id for each Lattices created */
        int usr_id;
        /** The number of pin_cells along the x-axis */
        int num_x;
        /** The number of pin_cells along the y-axis */
        int num_y;
        /** A container of pin_cells */
        int *pinCells_in_lattice;
    };

int *LatticeReadPinCellID(FILE *_fp,int _numTotal);
FP_PRECISION_GEO LatticeGetLengthX(struct LATTICE *_latt);
FP_PRECISION_GEO LatticeGetLengthY(struct LATTICE *_latt);
int LatticeGetNumPinCellsX(struct LATTICE *_latt);
int LatticeGetNumPinCellsY(struct LATTICE *_latt);
struct POINT *PinCellGetOffset(struct LATTICE *_lattice,int _pinCell_id);
void CopyLattice(struct LATTICE *_copy_latt,struct LATTICE *_latt);
void FreePinCellsInLattice(struct LATTICE *_lattice);
void FreeLattices(int _n,struct LATTICE *_lattice,char *_parameter);
int GetByteSizeStrLATTICE(int _numLattices, struct LATTICE *_Lattices);

#endif // LATTICE_H
