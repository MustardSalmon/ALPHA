#ifndef PAPER_TRANSPORT_COMPAT_H
#define PAPER_TRANSPORT_COMPAT_H

#include "MOC_CommonMacro.h"

/*
 * The publication executable is a steady-state transport solver. These
 * symbols are retained only at the compatibility boundary for legacy MOC
 * function signatures; the transient solver itself is not part of this
 * package.
 */
extern int isTransient;
extern int TimeID;
extern FP_PRECISION sKeff;
extern FP_PRECISION *tranSrcConstA;
extern FP_PRECISION *tranSrcConstB;
extern FP_PRECISION *tranSrcConstC;
extern FP_PRECISION *FissSource;
extern FP_PRECISION_FLUX *PrevScalarFlux;
FP_PRECISION sum(FP_PRECISION *_srcArray, int _startIndex, int _endIndex);

#endif /* PAPER_TRANSPORT_COMPAT_H */
