/********************************************************************************/
/* Publication snapshot: transport-only pin-cell helpers.                       */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>

#include "MOC_Cell.h"
#include "MOC_CommonMacro.h"
#include "MOC_Output.h"
#include "MOC_Pin_Cell.h"

struct CELLS_IN_PINCELL *AddCellToList(struct CELLS_IN_PINCELL *_list,
                                        struct CELL *_cell, int _cellID)
{
    struct CELLS_IN_PINCELL *_newNode = (struct CELLS_IN_PINCELL *)
        calloc(1, sizeof(struct CELLS_IN_PINCELL));
    if (_newNode == NULL) {
        printf("\n [**ERROR**]  calloc failed in Function \"AddCellToList\"!\n");
        fflush(stdout);
        exit(69);
    }
    _newNode->cells = (struct CELL *) calloc(1, sizeof(struct CELL));
    if (_newNode->cells == NULL) {
        printf("\n [**ERROR**]  calloc failed in Function \"AddCellToList\" - cells!\n");
        fflush(stdout);
        exit(69);
    }
    CopyCell(_newNode->cells, _cell);
    _newNode->cell_id = _cellID;
    _newNode->next = _list;
    return _newNode;
}

void CopyPinCell(struct PIN_CELL *_ectypePinCell,
                 struct PIN_CELL *_origPinCell)
{
    struct CELLS_IN_PINCELL *_pointer;

    _ectypePinCell->usr_id = _origPinCell->usr_id;
    _ectypePinCell->type_id = _origPinCell->type_id;
    _ectypePinCell->RTMod_id = _origPinCell->RTMod_id;
    _ectypePinCell->width_x = _origPinCell->width_x;
    _ectypePinCell->width_y = _origPinCell->width_y;
    _ectypePinCell->num_cells = _origPinCell->num_cells;
    for (_pointer = _origPinCell->cells_in_pinCell; _pointer != NULL;
         _pointer = _pointer->next) {
        _ectypePinCell->cells_in_pinCell = AddCellToList(
            _ectypePinCell->cells_in_pinCell, _pointer->cells, _pointer->cell_id);
    }
}

void FreeCellsInPinCell(struct PIN_CELL *_pinCell)
{
    struct CELLS_IN_PINCELL *_p1 = _pinCell->cells_in_pinCell;

    while (_p1 != NULL) {
        _pinCell->cells_in_pinCell = _p1->next;
        FreeCell(1, _p1->cells, "_p1->cells");
        free(_p1);
        _p1 = _pinCell->cells_in_pinCell;
    }
    _pinCell->cells_in_pinCell = NULL;
}

void FreePinCells(int _n, struct PIN_CELL *_pinCell, char *_parameter)
{
    int i;

    OutputFreeMemory(_parameter);
    for (i = 0; i < _n; i++) {
        FreeCellsInPinCell(&_pinCell[i]);
    }
    free(_pinCell);
}

int GetByteSizeStrPIN_CELL(int _numCells, struct PIN_CELL *_pinCell)
{
    int _byteNum = 0;
    int i;
    struct CELLS_IN_PINCELL *_iter;

    for (i = 0; i < _numCells; i++) {
        _byteNum += sizeof(struct PIN_CELL);
        for (_iter = _pinCell[i].cells_in_pinCell; _iter != NULL; _iter = _iter->next) {
            _byteNum += sizeof(struct CELLS_IN_PINCELL);
            _byteNum += GetByteSizeStrCELL(1, _iter->cells);
        }
    }
    return _byteNum;
}
