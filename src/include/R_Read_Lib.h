#ifndef R_Read_Lib_H
#define R_Read_Lib_H

#include "MOC_CommonMacro.h"

	extern FP_PRECISION *up_En;
	extern FP_PRECISION *up_u;
	extern FP_PRECISION *delta_u;

	extern int *SubTemNum, *If_SubTem_Constant;
	extern FP_PRECISION **SubTemp;
	extern int ***nsubg;
	extern FP_PRECISION ****SubWei;;
	extern FP_PRECISION ****SubTot;
	extern FP_PRECISION ****SubSca;
	extern FP_PRECISION ****SubInt;
	extern FP_PRECISION ****SubAbs;
	extern FP_PRECISION ****SubFis;

	extern FP_PRECISION **xs_pot;//[16][20];
	extern FP_PRECISION **xs_lamda;//[16][20];
	extern int NORG_HLS, NOG_HLS, NOFG_HLS, NOFS_HLS, NOTG_HLS;

	extern  int NORG_SHEM, NOG_SHEM,NOFG_SHEM,NOFS_SHEM,NOTG_SHEM, NELT_SHEM; extern int NRES_SHEM, Lib_Format;
	extern 	FP_PRECISION *Energy_SHEM,*Energy_16, *u_SHEM, *fchi_47, *fchi_SHEM, *aw_SHEM;
	extern 	int *ntemp_SHEM;
	extern 	FP_PRECISION **temp_SHEM;
	extern 	int *ntab_SHEM;
	extern 	FP_PRECISION *fis_spec_SHEM;
	extern 	int *eid_SHEM;
	extern 	int *NF_trigger_SHEM;
	extern  int NSUB_1G;
	extern  FP_PRECISION *SUB_XS_1G;

	extern FP_PRECISION **xsp_SHEM_rg;
	extern FP_PRECISION **lamda_SHEM_rg;
	extern FP_PRECISION **xiss_SHEM;
	extern FP_PRECISION **xstr_SHEM_frg;
	extern FP_PRECISION **xsa_SHEM_frg;
	extern FP_PRECISION **xsnf_SHEM_frg;
	extern FP_PRECISION **xsf_SHEM_frg;
	extern FP_PRECISION **neu_SHEM_frg;

	extern int *scatt_frg_num;
	extern FP_PRECISION **scatt_frg;
	extern int **scatt_frg_pos, ***scatt_nog_pos;
	extern int **scatt_tg_num, **scatt_nog_num;// (NELT,20);
	extern FP_PRECISION ***scatt_nog;// (NELT, 20, data_num);
	extern FP_PRECISION ***scatt_tg;// (NELT, 20, data_num);
	extern int ***scatt_tg_pos;
	extern int ***scatt_p1_pos, **scatt_p1_num;	extern  FP_PRECISION ***scatt_p1;

	extern FP_PRECISION ***xstr_SHEM_tg;//(NELT, NOTG_SHEM, 20)
	extern FP_PRECISION ***xsa_SHEM_tg;//(NELT, NOTG_SHEM, 20)
	extern FP_PRECISION ***xsnf_SHEM_tg;//(NELT, NOTG_SHEM, 20)
	extern FP_PRECISION ***xsf_SHEM_tg;//(NELT, NOTG_SHEM, 20)
	extern 	int **totalsub;//(NELT,NTEMP)
	extern 	int Max_Sum_SubGr, Max_SubGr;
	extern 	int *SubgrTotNm;//(NELT,NTEMP)
	extern int **NumSubgr; extern  int *max_nsubj;
	extern int **RI_Temp_Num;//NRES_SHEM, NORG_SHEM
	extern int **RI_BCS_Num;//NRES_SHEM, NORG_SHEM
	extern FP_PRECISION ***RI_Temp;//NRES_SHEM, NORG_SHEM, TEMPN
	extern FP_PRECISION ***RI_BCS;//NRES_SHEM, NORG_SHEM, XSB_N
	extern FP_PRECISION ****RI_abs_SHEM;//NRES_SHEM, TEMPNN, NORG_SHEM, XSB_N
	extern FP_PRECISION ****RI_nf_SHEM;//NRES_SHEM, TEMPNN, NORG_SHEM, XSB_N
	extern FP_PRECISION ****RI_sca_SHEM;//NRES_SHEM, TEMPNN, NORG_SHEM, XSB_N
	extern FP_PRECISION **imatrix_H, **imatrix_O, **imatrix_U5, **imatrix_U8, **imatrix_Zr, **imatrix_O1;
	extern FP_PRECISION ***UFG_xsa, ***UFG_xsf, ***UFG_xss, ***UFG_xst;

	void Readlib_SHEM_Path();
	void Readlib_Control();
	void Readlib_UFG_Path();


#ifdef BPO
extern int n2nCnum, n3nCnum;
extern int *n2nIdD, *n3nIdD;
#endif



#endif
