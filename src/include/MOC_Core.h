
#ifndef MOC_CORE_H
#define MOC_CORE_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MOC_CommonMacro.h"

struct CORE {
    /** The number of assemblies along the x-axis */
    int num_X;
    /** The number of assemblies along the y-axis */
    int num_Y;
    /** A container of assemblies' id */
    int *asse_in_core;
    /** A container of CRBs' id */
    int *crb_id;
};

/** GPU_TSKernelType rod bundle information*/
struct CRB {
    /** type id */
    int type_id;
    /** tip position*/
    FP_PRECISION_GEO tip_pos;
};

int *CoreReadAsseID(FILE *fp,int _numTotal);
FP_PRECISION_GEO CoreGetLengthX(struct CORE *_core);
FP_PRECISION_GEO CoreGetLengthY(struct CORE *_core);
FP_PRECISION_GEO CoreGetLengthZ(struct CORE *_core);
int CoreGetNumPinCellsX(struct CORE *_core, int _layerNum_in_core);
int CoreGetNumPinCellsY(struct CORE *_core, int _layerNum_in_core);
int CoreGetNumLayers(struct CORE *_core);
struct POINT *AssemblyGetOffset(struct CORE *_core,int _asse_id);
void FreeCores(int _n,struct CORE *_core, char *_parameter);
int GetByteSizeStrCORE(int _numCores, struct CORE *_core);
#endif // CORE_H
