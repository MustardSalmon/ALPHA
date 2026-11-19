/********************************************************************************/
/* Publication snapshot: eigenvalue update used by the transport iteration.    */
/********************************************************************************/
#include "MOC_CommonMacro.h"
#include "MOC_EigenValueEvaluation.h"

FP_DOUBLE KeffEvaluation(FP_DOUBLE _tot_vsigf_tem,
                         FP_DOUBLE _tot_vsigf_old,
                         FP_DOUBLE _keff_old,
                         int _totNumFSRs)
{
    FP_DOUBLE _keff;
    (void) _totNumFSRs;
#if defined(NORMALIZE_FLUX)
    _keff = _tot_vsigf_tem * _keff_old / (FP_DOUBLE) _totNumFSRs;
#else
    _keff = _tot_vsigf_tem / _tot_vsigf_old * _keff_old;
#endif
    return _keff;
}
