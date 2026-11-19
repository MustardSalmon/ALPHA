/********************************************************************************/
/* Publication snapshot: transport-only cell helpers.                           */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>

#include "MOC_Cell.h"
#include "MOC_CommonMacro.h"
#include "MOC_Output.h"

struct SURFACE_HALFSPACE *AddPlaneToList(struct SURFACE_HALFSPACE *_list,
                                          struct PLANE _plane,
                                          FP_PRECISION _halfspace)
{
    struct SURFACE_HALFSPACE *_newNode = (struct SURFACE_HALFSPACE *)
        calloc(1, sizeof(struct SURFACE_HALFSPACE));
    if (_newNode == NULL) {
        printf("\n [**ERROR**]  calloc failed in Function \"AddPlaneToList\" - _newNode!\n");
        fflush(stdout);
        exit(69);
    }
    _newNode->surf_type = _plane.surface_type;
    _newNode->bound_surf._plane = _plane;
    _newNode->halfspace = _halfspace;
    _newNode->next = _list;
    return _newNode;
}

struct SURFACE_HALFSPACE *AddCylinderToList(struct SURFACE_HALFSPACE *_list,
                                             struct PARA_CYLINDER _cyl,
                                             FP_PRECISION _halfspace)
{
    struct SURFACE_HALFSPACE *_newNode = (struct SURFACE_HALFSPACE *)
        calloc(1, sizeof(struct SURFACE_HALFSPACE));
    if (_newNode == NULL) {
        printf("\n [**ERROR**]  calloc failed in Function \"AddCylinderToList\" - _newNode!\n");
        fflush(stdout);
        exit(69);
    }
    _newNode->surf_type = _cyl.surface_type;
    _newNode->bound_surf._paraCylinder = _cyl;
    _newNode->halfspace = _halfspace;
    _newNode->next = _list;
    return _newNode;
}

void CopyCell(struct CELL *_ectypeCell, struct CELL *_origCell)
{
    struct SURFACE_HALFSPACE *_pointer;

    _ectypeCell->id = _origCell->id;
    _ectypeCell->is_cr = _origCell->is_cr;
    _ectypeCell->mat_id = _origCell->mat_id;
    _ectypeCell->numX_mesh = _origCell->numX_mesh;
    _ectypeCell->numY_mesh = _origCell->numY_mesh;
    _ectypeCell->num_rings = _origCell->num_rings;
    _ectypeCell->num_sectors = _origCell->num_sectors;

    for (_pointer = _origCell->bound_surfaces; _pointer != NULL; _pointer = _pointer->next) {
        if (_pointer->surf_type == PLANE || _pointer->surf_type == XPLANE ||
            _pointer->surf_type == YPLANE || _pointer->surf_type == ZPLANE) {
            _ectypeCell->bound_surfaces = AddPlaneToList(
                _ectypeCell->bound_surfaces, _pointer->bound_surf._plane,
                _pointer->halfspace);
        } else if (_pointer->surf_type == XCYLINDER ||
                   _pointer->surf_type == YCYLINDER ||
                   _pointer->surf_type == ZCYLINDER) {
            _ectypeCell->bound_surfaces = AddCylinderToList(
                _ectypeCell->bound_surfaces, _pointer->bound_surf._paraCylinder,
                _pointer->halfspace);
        }
    }
}

void FreeSurfHalfInCell(struct CELL *_cell)
{
    struct SURFACE_HALFSPACE *_p1 = _cell->bound_surfaces;

    while (_p1 != NULL) {
        _cell->bound_surfaces = _p1->next;
        free(_p1);
        _p1 = _cell->bound_surfaces;
    }
    _cell->bound_surfaces = NULL;
}

void FreeCell(M_INT32 _n, struct CELL *_cell, char *_parameter)
{
    M_INT32 i;

    OutputFreeMemory(_parameter);
    for (i = 0; i < _n; i++) {
        FreeSurfHalfInCell(&_cell[i]);
    }
    free(_cell);
}

M_INT32 GetByteSizeStrCELL(M_INT32 _numPrimitives, struct CELL *_primitives)
{
    M_INT32 _byteNum = 0;
    M_INT32 i;
    struct SURFACE_HALFSPACE *_iter;

    for (i = 0; i < _numPrimitives; i++) {
        _byteNum += sizeof(struct CELL);
        for (_iter = _primitives[i].bound_surfaces; _iter != NULL; _iter = _iter->next) {
            _byteNum += sizeof(struct SURFACE_HALFSPACE);
        }
    }
    return _byteNum;
}
