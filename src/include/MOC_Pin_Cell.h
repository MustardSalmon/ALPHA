
#ifndef MOC_PIN_CELL_H
#define MOC_PIN_CELL_H
#include <stdlib.h>
#include "MOC_Point.h"
#include "MOC_CommonMacro.h"

/**
 * @struct CELLS_IN_PINCELL
 * @introduction A linked list for storing the cells in a Cells.
 */
    struct CELLS_IN_PINCELL {
		/** the cell id */
		int cell_id;
        /** cell */
        struct CELL *cells;  /**  */
        struct CELLS_IN_PINCELL *next;
		
    };

/**
 * @struct PIN_CELL
 * @introduction A PIN_CELL represents an unbounded space in the 3D.
 * @explanation A PIN_CELL contains cells which are bounded subspaces in 3D which
 *          together form the PIN_CELL. Universes allow for complex, repeating
 *          (i.e. lattices) geometries to be simply represented with as few data
 *          structures as possible.
 */
    struct PIN_CELL {
        /** */
        int type_id;
        /** A user-defined id for each PIN_CELL created */
        int usr_id;
        /** ray tracing module id */
        int RTMod_id;
        /** The width of each Cells (cm) along the x-axis */
        FP_PRECISION_GEO width_x;
        /** The width of each Cells (cm) along the y-axis */
        FP_PRECISION_GEO width_y;
        /** number of cells */
        int num_cells;
        /** A collection of Cells in this PIN_CELL */
        struct CELLS_IN_PINCELL *cells_in_pinCell;
    };
struct CELLS_IN_PINCELL *AddCellToList(struct CELLS_IN_PINCELL *list,\
                                          struct CELL *_cell,int _cell_id);
void CopyPinCell(struct PIN_CELL *_copy_pinCell,struct PIN_CELL *_pin_cell);
void FreePinCells(int _n,struct PIN_CELL *_pinCell, char *_parameter);
void FreeCellsInPinCell(struct PIN_CELL *_pinCell);
int GetByteSizeStrPIN_CELL(int _numCells, struct PIN_CELL *_pinCell);

#endif // PIN_CELL_H
