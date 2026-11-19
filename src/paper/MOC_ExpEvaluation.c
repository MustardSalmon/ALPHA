/********************************************************************************/
/* Publication snapshot: exponential tables used by GPU transport.             */
/********************************************************************************/
#include <math.h>

#include "MOC_AllocAndFreeMem.h"
#include "MOC_CommonMacro.h"
#include "MOC_ExpEvaluation.h"
#include "MOC_Input.h"

FP_PRECISION MaxOpticalLength = (FP_PRECISION) 1.0;
FP_PRECISION MaxOpticalLength_limit = (FP_PRECISION) 0.14;
FP_PRECISION InverseExpTableSpacing;
FP_PRECISION InverseExpTableSpacing_0;
FP_PRECISION *ExpTab_pitch = NULL;
FP_PRECISION *ExpTab_y0xk = NULL;

FP_PRECISION *ExpH16_Initialize(void)
{
    FP_PRECISION *_expInt = AllocMem_FP_I_1(1281, "_expInt");
    int i;
    for (i = 0; i <= 1280; i++) {
        _expInt[i] = (FP_PRECISION) exp(-i * (1.0 / 32.0));
    }
    return _expInt;
}

FP_PRECISION *ExpL16_Initialize(void)
{
    FP_PRECISION *_expDec = AllocMem_FP_I_1(2049, "_expDec");
    int i;
    for (i = 0; i <= 2048; i++) {
        _expDec[i] = (FP_PRECISION) exp(-i * (1.0 / 65536.0));
    }
    return _expDec;
}

FP_PRECISION *Exp10_2C1T_Initialize(void)
{
    FP_PRECISION *_expInt = AllocMem_FP_I_1(40961, "_expInt");
    int i;
    for (i = 0; i <= 40960; i++) {
        _expInt[i] = (FP_PRECISION) exp(-i * (1.0 / 1024.0));
    }
    return _expInt;
}

FP_PRECISION *Exp10_Linear_Initialize(void)
{
    FP_PRECISION *_expInt = AllocMem_FP_I_1(327681, "_expInt");
    int i;
    for (i = 0; i <= 327680; i++) {
        _expInt[i] = (FP_PRECISION) exp(-i * (1.0 / 8192.0));
    }
    return _expInt;
}

FP_PRECISION *ExpFast_LinearExpTable_Mod1(void)
{
    FP_PRECISION *_expInt;
    int i;
    FP_PRECISION delta = (FP_PRECISION) 1.0 / NumExpPoints_One;

    _expInt = AllocMem_FP_I((size_t) NumExpPoints, (char *) "_expInt");
    for (i = 0; i < NumExpPoints; i++) {
        _expInt[i] = (FP_PRECISION) (1.0 - exp(-i * delta));
    }
    return _expInt;
}
