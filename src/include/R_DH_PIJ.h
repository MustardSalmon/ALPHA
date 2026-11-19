#ifndef R_DH_PIJ_H
#define R_DH_PIJ_H
#include "MOC_CommonMacro.h"

extern int I_Order, J_Order, PIJ_Lay_Num;
extern FP_PRECISION *V_Ratio, *Vol_R, *EIS, *Radi, *XS_R, **PIJ, **EIS_tio, ***EIS_tio_k;
extern FP_PRECISION *Gauss_p, *Gauss_t;


void Cal_PIJ_DH();
FP_PRECISION DeCART_PIJ(FP_PRECISION x);
FP_PRECISION DeCART_PII(FP_PRECISION x);
FP_PRECISION Gauss_Integral(FP_PRECISION(*func)(FP_PRECISION x), FP_PRECISION a, FP_PRECISION b, int n);


;;;;;;
#endif
