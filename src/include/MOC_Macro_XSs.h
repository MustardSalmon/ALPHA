#ifndef MOC_MACRO_XS_H
#define MOC_MACRO_XS_H
#include "MOC_Core.h"
#include "MOC_CommonMacro.h"
#include <stdint.h>
//#define NFSR 56
//#define NF_FSR 24
//#define NC_FSR 24
//#define NM_FSR 8
//#define NUC_N 30
//#define NFLX 4
//#define NORG 16
    extern int Input_MATorXSs;
	extern int INPUT_XSs_MATorFSR;
	extern int IfSetXSsForEveryFSR;
	extern int ContRodMatLayoutInAxial;
	extern int NumMats;
	extern int NumContRodMats;
	extern struct CR_MAT *ContRodMats;
	extern struct MAT *CoreMats;
	extern int TransCorre;
	extern int ReCalcXSt;
	extern int ReCalcXSa;
    extern int NumGroups;
	extern int NumXSs_Mat;
	extern int *XSs_Mat_Id;
	extern int ***XSsID_FSRs;
	extern FP_PRECISION **InpXSs_t;
	extern FP_PRECISION **InpXSs_a;
	extern FP_PRECISION **InpXSs_f;
	extern FP_PRECISION **InpXSs_nf;
	extern FP_PRECISION ***InpXSs_s;
	extern FP_PRECISION ***InpXSs_p0;
	extern FP_PRECISION ***InpXSs_p1;
	extern FP_PRECISION **InpChi;
	extern FP_PRECISION **Inp_FixedSour;
    extern FP_PRECISION *MacroXS_t/*, *MacroXS_t_RayWised*/;
    extern FP_PRECISION *MacroXS_a;
    extern FP_PRECISION *MacroXS_f;
    extern FP_PRECISION *MacroXS_nf;
    extern FP_PRECISION *MacroXS_s;
	extern M_UINT64 MacroXS_s_NNZ;
	extern M_UINT64 MacroXS_s_MaxNNZ_Layer;
    extern M_UINT64 *MacroXS_s_NZRowStartID;
    extern M_UINT64 *MacroXS_s_NZRowEndID;
	extern M_UINT64 *MacroXS_s_NNZ_Layer;
	extern M_UINT64 *MacroXS_s_NZStartID_Layer;
    extern M_UINT64 *MacroXS_s_NZRowStartID_Layer;
#ifdef __cplusplus
extern "C" {
#endif
    extern M_UINT64 *MacroXS_s_NZRowEndID_Layer;
    extern M_UINT64 *MacroXS_s_NZColStartID;
    extern int IfUseExplicitScaRowEnd;
#ifdef __cplusplus
}
#endif
	extern FP_PRECISION *MacroXS_p0;
	extern FP_PRECISION *MacroXS_p1;
    extern FP_PRECISION *Chi;
	extern FP_PRECISION *FixedSour;
	extern int NumProcs, ProcID;
	extern M_INT32 *FSRsXSID_subDom;

	extern FP_PRECISION *MacroXS_t_Mat;
    extern FP_PRECISION *MacroXS_a_Mat;
    extern FP_PRECISION *MacroXS_f_Mat;
    extern FP_PRECISION *MacroXS_nf_Mat;
    extern FP_PRECISION *MacroXS_s_Mat;
	extern FP_PRECISION *Chi_Mat;

	extern int *InpXSs_s_NNZ;
	extern int **InpXSs_s_NZColIDStart;
    extern int **InpXSs_s_NZColIDEnd;

	extern int NumScattOrder;

	void InputMat_XSs(char * _filename);
	void InputMat(FILE *_ip);
	void InputXSs_forMat(FILE *_fp1);
    void InputXSs_forFSRs(FILE* _fp1, int _fileIdx);
    FP_PRECISION GetFSRXSs(int _lay, int _cell, int _reg,int _numCellsPerlay, int _numGroups,int _g, \
        FP_PRECISION *_macroXS, FP_PRECISION **_inpXS,int *_sumNumFSRsBasedGlobCellID);
		FP_PRECISION GetFSRXSs_Scatt(int _lay, int _cell, int _reg,int _numCellsPerlay, int _numGroups, \
		int _g, int _g1, FP_PRECISION *_macroXS, FP_PRECISION ***_inpXS,int *_sumNumFSRsBasedGlobCellID);
	void generateMatrix_MATLAB_InfiniteMedium();
	int GetUpScattGroupBound(FP_PRECISION **_macroXS, FP_PRECISION **_inpXS);
	void MacroXSsEvaluation_subDomain(/*int _procID,struct CORE *_core,*/  int _subDomID_row, int _subDomID_column);
	void FissionSpectrumCheck(int _num_groups,int _globFRSId, FP_PRECISION *_chi);
	void generateMatrix_MATLAB_InfiniteMedium();


	void Update_XSleak_to_XSt(FP_PRECISION **_macroXS_t_, FP_PRECISION ***XS_leak);
	void InputFinxedSource_forFSRs(char *_filename);
	void MacroXSsEvaluation_subDomain_trackFile();

void GetFluxVolWeightedXSs_Mat_1D(int _num_mat, int _NumGroups, int _TotalNumFSRs_subDom, \
    int *_FSRsXSID_subDom, FP_PRECISION_GEO *_FSRVol_SubDom, FP_PRECISION *_ScalarFlux_MOC,\
	FP_PRECISION *_sumReatRate, FP_PRECISION *_sumFlux, FP_PRECISION *_XSs_1D, FP_PRECISION *_XSs_1D_Mat);

void GetFluxVolWeightedXSs_Mat_2D(int _num_mat, int _NumGroups, int _TotalNumFSRs_subDom, \
    int *_FSRsXSID_subDom, FP_PRECISION_GEO *_FSRVol_SubDom, FP_PRECISION *_ScalarFlux_MOC,\
	FP_PRECISION *_sumReatRate, FP_PRECISION *_sumFlux, FP_PRECISION *_XSs_2D, FP_PRECISION *_XSs_2D_Mat);

void CondenseMacroXSsForMat_FromMOCSolution(M_INT32 _isInvoke, M_INT32 _IfResonance, M_INT32 _num_mat, M_INT32 _NumGroups, M_INT32 _TotalNumFSRs_subDom, \
    M_INT32 *_FSRsXSID_subDom, FP_PRECISION_GEO *_FSRVol_SubDom, FP_PRECISION *_ScalarFlux_MOC);
M_UINT64 GetNNZ_MacroXS_s();
M_UINT64 GetMaxNNZ_MacroXS_s_Lay();
#endif // MACRO_XS_H
