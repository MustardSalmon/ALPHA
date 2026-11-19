#ifndef MOC_EXP_EVALUATION_H
#define MOC_EXP_EVALUATION_H
#define exp_precision 1.0e-10
#include "MOC_CommonMacro.h"

#define ExpTab_Range 20000 // for normal problm 40 will be large enough
#define NumExpPoints_One 1024 //1024 10^10, 8192 10^13, 128
#define NumExpPoints_One_Mod1 1000 // 1000 points in unit 1
#define NumExpPoints NumExpPoints_One * ExpTab_Range + 1
#define NumExpPoints_Mod1 NumExpPoints_One_Mod1*20+1
//#define NumExpPoints NumExpPoints_One*40+1

extern FP_PRECISION MaxOpticalLength;
extern FP_PRECISION MaxOpticalLength_limit;
extern FP_PRECISION InverseExpTableSpacing;
extern FP_PRECISION InverseExpTableSpacing_0;
extern FP_PRECISION *ExpTab_pitch,*ExpTab_y0xk;

FP_PRECISION *ExpH20_Initialize();
FP_PRECISION *ExpL20_Initialize();
FP_PRECISION *ExpH16_Initialize();
FP_PRECISION *ExpL16_Initialize();
extern FP_PRECISION *Exp2TH, *Exp2TL, *Exp2C1T, *ExpLinear;
extern FP_PRECISION Exp16_C1, Exp16_C2;
extern int Index_Exp;
extern FP_PRECISION Exp10_C1, Exp10_C2;
FP_PRECISION *Exp10_2C1T_Initialize();
FP_PRECISION *Exp10_Linear_Initialize();
extern FP_PRECISION *ExpTable, **ExpF0;
FP_PRECISION **ExpFast_Linear_Initialize();
FP_PRECISION *ExpFast_LinearExpTable();

FP_PRECISION *ExpFast_LinearExpTable_Mod1();
void LinearExpTable_FromLiang();

#endif