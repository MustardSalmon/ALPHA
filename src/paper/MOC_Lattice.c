/********************************************************************************/
/* Publication snapshot: transport-only lattice helpers.                       */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>

#include "MOC_CommonMacro.h"
#include "MOC_Input.h"
#include "MOC_Lattice.h"
#include "MOC_Output.h"

int *LatticeReadPinCellID(FILE *_fp, int _numTotal)
{
    int *_pointer = (int *) calloc((size_t) _numTotal, sizeof(int));
    int i, _inpErr;

    if (_pointer == NULL) {
        printf("\n [**ERROR**]  calloc failed in Function \"LatticeReadPinCellID\" - _pointer !\n");
        fflush(stdout);
        exit(69);
    }
    for (i = 0; i < _numTotal; i++) {
        JumpCommentsInFile(_fp);
        _inpErr = fscanf(_fp, "%d", &_pointer[i]);
        (void) _inpErr;
    }
    return _pointer;
}

FP_PRECISION_GEO LatticeGetLengthX(struct LATTICE *_latt)
{
    int i, j;
    FP_PRECISION_GEO _lengthX = 0.0;

    for (i = 0; i < _latt->num_x; i++) {
        for (j = 0; j < NumCells; j++) {
            if (_latt->pinCells_in_lattice[i] == Cells[j].type_id) {
                _lengthX += Cells[j].width_x;
                break;
            }
        }
    }
    return _lengthX;
}

FP_PRECISION_GEO LatticeGetLengthY(struct LATTICE *_latt)
{
    int i, j;
    FP_PRECISION_GEO _lengthY = 0.0;

    for (i = 0; i < _latt->num_y; i++) {
        for (j = 0; j < NumCells; j++) {
            if (_latt->pinCells_in_lattice[i * _latt->num_x] == Cells[j].type_id) {
                _lengthY += Cells[j].width_y;
                break;
            }
        }
    }
    return _lengthY;
}

void FreeLattices(int _n, struct LATTICE *_lattice, char *_parameter)
{
    int i;

    OutputFreeMemory(_parameter);
    for (i = 0; i < _n; i++) {
        free(_lattice[i].pinCells_in_lattice);
        _lattice[i].pinCells_in_lattice = NULL;
    }
    free(_lattice);
}

int GetByteSizeStrLATTICE(int _numLattices, struct LATTICE *_lattices)
{
    int _byteNum = 0;
    int i;

    for (i = 0; i < _numLattices; i++) {
        _byteNum += sizeof(struct LATTICE);
        _byteNum += _lattices[i].num_x * _lattices[i].num_y * sizeof(int);
    }
    return _byteNum;
}
