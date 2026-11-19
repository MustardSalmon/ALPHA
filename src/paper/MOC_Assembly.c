/********************************************************************************/
/* Publication snapshot: transport-only assembly helpers.                       */
/*                                                                              */
/* This file intentionally contains only the assembly functions retained by    */
/* the GPU transport executable.  The original source remains outside this     */
/* snapshot's build and is kept as the reference implementation.               */
/********************************************************************************/
#include <stdlib.h>

#include "MOC_AllocAndFreeMem.h"
#include "MOC_Assembly.h"
#include "MOC_CommonMacro.h"
#include "MOC_Input.h"
#include "MOC_Lattice.h"
#include "MOC_Output.h"

FP_PRECISION_GEO AssemblyGetLengthX(struct ASSEMBLY *_assembly)
{
    M_INT32 i;
    FP_PRECISION_GEO _lengthX = 0.0;

    for (i = 0; i < NumLattices; i++) {
        if (_assembly->latt_in_asse[0] == Lattices[i].type_id) {
            _lengthX = LatticeGetLengthX(&Lattices[i]);
            break;
        }
    }
    return _lengthX;
}

FP_PRECISION_GEO AssemblyGetLengthY(struct ASSEMBLY *_assembly)
{
    M_INT32 i;
    FP_PRECISION_GEO _lengthY = 0.0;

    for (i = 0; i < NumLattices; i++) {
        if (_assembly->latt_in_asse[0] == Lattices[i].type_id) {
            _lengthY = LatticeGetLengthY(&Lattices[i]);
            break;
        }
    }
    return _lengthY;
}

void FreeAssemblies(M_INT32 _n, struct ASSEMBLY *_assembly, char *_parameter)
{
    M_INT32 i;

    OutputFreeMemory(_parameter);
    for (i = 0; i < _n; i++) {
        free(_assembly[i].latt_width_z);
        free(_assembly[i].latt_sub_width_z);
        free(_assembly[i].latt_in_asse);
        free(_assembly[i].num_sub_in_layer);
        _assembly[i].latt_width_z = NULL;
        _assembly[i].latt_sub_width_z = NULL;
        _assembly[i].latt_in_asse = NULL;
        _assembly[i].num_sub_in_layer = NULL;
    }
    free(_assembly);
}

M_INT32 GetByteSizeStrASSEMBLY(M_INT32 _numAssemblies, struct ASSEMBLY *_assembly)
{
    M_INT32 _byteNum = 0;
    M_INT32 i;

    for (i = 0; i < _numAssemblies; i++) {
        _byteNum += sizeof(struct ASSEMBLY);
        _byteNum += _assembly[i].num_layer * sizeof(M_INT32);
        _byteNum += _assembly[i].num_layer * sizeof(FP_PRECISION_GEO);
    }
    return _byteNum;
}
