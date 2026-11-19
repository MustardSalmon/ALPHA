#ifndef MOC_EIGENVALUE_EVALU_H
#define MOC_EIGENVALUE_EVALU_H
#include "MOC_CommonMacro.h"

FP_DOUBLE KeffEvaluation(FP_DOUBLE tot_sigf_tem, FP_DOUBLE tot_sigf_old, FP_DOUBLE _keff_old, int _totNumFSRs);
FP_PRECISION KinfEvaluation(FP_PRECISION _tot_fiss, FP_PRECISION _tot_abs);

#endif // !EIGENVALUE_EVALU_H

