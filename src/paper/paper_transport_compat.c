#include "paper_transport_compat.h"

/* Steady-state defaults for the legacy MOC compatibility interface. */
int isTransient = 0;
int TimeID = 0;
FP_PRECISION sKeff = 0.0;
FP_PRECISION *tranSrcConstA = NULL;
FP_PRECISION *tranSrcConstB = NULL;
FP_PRECISION *tranSrcConstC = NULL;
FP_PRECISION *FissSource = NULL;
FP_PRECISION_FLUX *PrevScalarFlux = NULL;
FP_PRECISION sum(FP_PRECISION *_srcArray, int _startIndex, int _endIndex)
{
    FP_PRECISION result = 0.0;
    int index;

    for (index = _startIndex; index < _endIndex; ++index) {
        result += _srcArray[index];
    }
    return result;
}
