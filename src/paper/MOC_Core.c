/********************************************************************************/
/* Publication snapshot: transport-only core helpers.                           */
/********************************************************************************/
#include <stdlib.h>

#include "MOC_Assembly.h"
#include "MOC_CommonMacro.h"
#include "MOC_Core.h"
#include "MOC_Input.h"
#include "MOC_Macro_XSs.h"
#include "MOC_Output.h"

FP_PRECISION_GEO CoreGetLengthX(struct CORE *_core)
{
    int i, j;
    FP_PRECISION_GEO _coreWidthX = 0.0;

    for (i = 0; i < _core->num_X; i++) {
        for (j = 0; j < NumAssemblies; j++) {
            if (_core->asse_in_core[i] == Assemblies[j].type_id) {
                _coreWidthX += AssemblyGetLengthX(&Assemblies[j]);
                break;
            }
        }
    }
    return _coreWidthX;
}

FP_PRECISION_GEO CoreGetLengthY(struct CORE *_core)
{
    int i, j;
    FP_PRECISION_GEO _coreWidthY = 0.0;

    for (i = 0; i < _core->num_Y; i++) {
        for (j = 0; j < NumAssemblies; j++) {
            if (_core->asse_in_core[i * _core->num_X] == Assemblies[j].type_id) {
                _coreWidthY += AssemblyGetLengthY(&Assemblies[j]);
                break;
            }
        }
    }
    return _coreWidthY;
}

void FreeCores(int _n, struct CORE *_core, char *_parameter)
{
    int i;

    OutputFreeMemory(_parameter);
    for (i = 0; i < _n; i++) {
        free(_core[i].asse_in_core);
        free(_core[i].crb_id);
        _core[i].asse_in_core = NULL;
        _core[i].crb_id = NULL;
    }
    free(_core);
}

int GetByteSizeStrCORE(int _numCores, struct CORE *_core)
{
    int _byteNum = 0;
    int i;

    for (i = 0; i < _numCores; i++) {
        _byteNum += sizeof(struct CORE);
        _byteNum += _core[i].num_X * _core[i].num_Y * sizeof(int);
        if (ContRodMatLayoutInAxial) {
            _byteNum += _core[i].num_X * _core[i].num_Y * sizeof(int);
        }
    }
    return _byteNum;
}
