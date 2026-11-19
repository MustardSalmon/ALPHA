#ifndef R_GROUP_CONDENSE_H
#define R_GROUP_CONDENSE_H

extern FP_PRECISION *Iter_flux, **Wei_xsb, **Xsint_plus_xsb, **Wei_suba, **Wei_subnf, **Wei_subt, **Wei_subs;
extern FILE *Op_Fix, *Op_XS, *Op_sldw;

void Overlap_SlowingDown_fix_SHEM(int lay, int group);
void Overlap_1G_SlowingDown_SHEM(int _procID);
void Overlap_1G_SlowingDown_SHEM_MG(int _procID);
void Overlap_SlowingDown_fix_SHEM_MG(int lay, int group);
void Unpdate_SLDW_Fixed_Src_MG(int _groupLowLim, int _groupUpLim, int current_MG);

void Cal_ResGr_Down_Scatt_Index(int _procID);
void Unpdate_SLDW_Fixed_Src_MG_Index(int _groupLowLim, int _groupUpLim);




;;;;;;;;
#endif
