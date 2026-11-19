/********************************************************************************/
/* @ File: MOC_Macro_XSs.c
/*   introduction: functions related to macroscopic cross sections evaluation
/* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*   1st modified: 2017/05/16, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*                 FP_PRECISION **MacroXS_t, **MacroXS_a, **MacroXS_f, **MacroXS_nf, 
/*                 ***MacroXS_s, ***MacroXS_p0, ***MacroXS_p1, **Chi
/*                 dicrease the array dimension by 2, compress the [layer][cell][fsr] dimensions to [glob_fsr]
/*   2nd modified: 2017/05/25, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*                 dicrease the global XSs Pointers into 1 dimension 
/*                 FP_PRECISION *MacroXS_t, *MacroXS_a, *MacroXS_f, *MacroXS_nf, *Chi,dim[totNumFSRs*numGroups]
/*                 *MacroXS_s, *MacroXS_p0, *MacroXS_p1,dim[totNumFSRs*numGroups*numGroups]
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "MOC_AllocAndFreeMem.h"
#include "MOC_Assembly.h"
#include "MOC_Cell.h"
#include "MOC_CommonMacro.h"
#include "MOC_Core.h"
#include "MOC_Geometry.h"
#include "MOC_Input.h"
#include "MOC_Lattice.h"
#include "MOC_Macro_XSs.h"
#include "MOC_main.h"
#include "MOC_Output.h"
#include "MOC_Pin_Cell.h"
#include "MOC_Track.h"
#include "R_DH_Mat_XS.h"
#include "math.h"


    int Input_MATorXSs;  // boolean value, 0 or 1, 1-give materials' information of Core in MATERIAL_BLOCK
					 // 0-don't give materials' information of Core in MATERIAL_BLOCK
	int INPUT_XSs_MATorFSR;  // boolean value, 0 or 1, 0-give XSs' information of every materials
							 // 1-XSs for every FSRs
	int IfSetXSsForEveryFSR; // 0--- don't set MacroXSs for every FSRs, 1---set MacroXSs for every FSRs
	int ReCalcXSt;
	int ReCalcXSa;
	int TransCorre;
	int ContRodMatLayoutInAxial;  // boolean value, 0 or 1, 1-give the layout of GPU_TSKernelType rod materials in azial direction of core
	int NumMats;  // number of materials in the whole Core
	struct MAT *CoreMats = NULL;  // structure--the materials' information in Core
	int NumContRodMats;  // number of materials of GPU_TSKernelType rod in Core
	struct CR_MAT *ContRodMats = NULL;  // structure--the GPU_TSKernelType rod materials in Core
	int NumGroups;
	int NumXSs_Mat;
	int NumScattOrder = 0;
	int *XSs_Mat_Id = NULL;
	int ***XSsID_FSRs = NULL;
	FP_PRECISION **InpXSs_t = NULL;
	FP_PRECISION **InpXSs_a = NULL;
	FP_PRECISION **InpXSs_f = NULL;
	FP_PRECISION **InpXSs_nf = NULL;
	FP_PRECISION ***InpXSs_s = NULL;
    FP_PRECISION ***TempXSs_s = NULL; //////////////////////////
    double *TempXSs_T = NULL;  //////////////////////////
    M_INT32 *TempIndex = NULL;        //////////////////////////
	FP_PRECISION ***InpXSs_p0 = NULL;
	FP_PRECISION ***InpXSs_p1 = NULL;
	FP_PRECISION **InpChi = NULL;
	FP_PRECISION **Inp_FixedSour = NULL;
    FP_PRECISION *MacroXS_t = NULL/*, *MacroXS_t_RayWised = NULL*/; //[Number layer in subdomian][NumGroups] [NumCellsInCore] [NumFSRsInCell]
    FP_PRECISION *MacroXS_a = NULL;
    FP_PRECISION *MacroXS_f = NULL;
    FP_PRECISION *MacroXS_nf = NULL;
    FP_PRECISION *MacroXS_s = NULL;
	M_UINT64 MacroXS_s_NNZ=0;
	M_UINT64 MacroXS_s_MaxNNZ_Layer=0;
    M_UINT64 *MacroXS_s_NZRowStartID = NULL;
    M_UINT64 *MacroXS_s_NZRowEndID = NULL;
	M_UINT64 *MacroXS_s_NNZ_Layer = NULL;
	M_UINT64 *MacroXS_s_NZStartID_Layer = NULL;
    M_UINT64 *MacroXS_s_NZRowStartID_Layer = NULL;
    M_UINT64 *MacroXS_s_NZRowEndID_Layer = NULL;
    M_UINT64 *MacroXS_s_NZColStartID = NULL;
    int IfUseExplicitScaRowEnd = 0;
	FP_PRECISION *MacroXS_p0 = NULL;
	FP_PRECISION *MacroXS_p1 = NULL;
    FP_PRECISION *Chi = NULL;
	FP_PRECISION *FixedSour = NULL;
	int Peijun = 0;
	M_INT32 *FSRsXSID_subDom = NULL;

	FP_PRECISION *MacroXS_t_Mat = NULL;
    FP_PRECISION *MacroXS_a_Mat = NULL;
    FP_PRECISION *MacroXS_f_Mat = NULL;
    FP_PRECISION *MacroXS_nf_Mat = NULL;
    FP_PRECISION *MacroXS_s_Mat = NULL;
	FP_PRECISION *Chi_Mat = NULL;

	int *InpXSs_s_NNZ=NULL;
	int **InpXSs_s_NZColIDStart=NULL;
	int **InpXSs_s_NZColIDEnd = NULL;


/**
* @ Function: InputMat_XSs()
*   introduction: 
* @ Parameter List:
*   Return: 
* @ Created     : 2016/07/12, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
 * Read the external FSR-wise transport XS metadata.
 *
 * The publication package intentionally accepts only the FSR-wise CSR
 * workflow used by the reproducibility cases. Material-wise and resonance
 * input branches are outside this source snapshot.
 */
void InputMat_XSs(char *_filename)
{
    char keyword[KEYWORD_N];
    int inputErr;
    FILE *input = fopen(_filename, "r");

    if (input == NULL) {
        printf("\\n [**ERROR**] FSR-wise XS input file opens error - %s!\\n", _filename);
        fflush(stdout);
        exit(69);
    }

    IdentifyKeyword(input, keyword, "INPUT_MATorXSs");
    JumpCommentsInFile(input);
    inputErr = fscanf(input, "%d", &Input_MATorXSs);
    (void) inputErr;
    if (Input_MATorXSs != 1) {
        printf("\\n [**ERROR**] Publication package requires INPUT_MATorXSs=1.\\n");
        fflush(stdout);
        exit(69);
    }

    IdentifyKeyword(input, keyword, "INPUT_XSs_MATorFSR");
    JumpCommentsInFile(input);
    inputErr = fscanf(input, "%d", &INPUT_XSs_MATorFSR);
    (void) inputErr;
    if (INPUT_XSs_MATorFSR != 1) {
        printf("\\n [**ERROR**] Publication package requires INPUT_XSs_MATorFSR=1.\\n");
        fflush(stdout);
        exit(69);
    }

    IdentifyKeyword(input, keyword, "Num_Group");
    JumpCommentsInFile(input);
    inputErr = fscanf(input, "%d", &NumGroups);
    (void) inputErr;

    IdentifyKeyword(input, keyword, "IfSetXSsForEveryFSR");
    JumpCommentsInFile(input);
    inputErr = fscanf(input, "%d", &IfSetXSsForEveryFSR);
    (void) inputErr;
    if (IfSetXSsForEveryFSR != 1) {
        printf("\\n [**ERROR**] Publication package requires IfSetXSsForEveryFSR=1.\\n");
        fflush(stdout);
        exit(69);
    }

    IdentifyKeyword(input, keyword, "reCalcTotalXS");
    JumpCommentsInFile(input);
    inputErr = fscanf(input, "%d", &ReCalcXSt);
    (void) inputErr;

    IdentifyKeyword(input, keyword, "reCalcAbsorbXS");
    JumpCommentsInFile(input);
    inputErr = fscanf(input, "%d", &ReCalcXSa);
    (void) inputErr;

    IdentifyKeyword(input, keyword, "transportCorrection");
    JumpCommentsInFile(input);
    inputErr = fscanf(input, "%d", &TransCorre);
    (void) inputErr;

    IdentifyKeyword(input, keyword, "Num_of_XSsTypes_FSR");
    JumpCommentsInFile(input);
    inputErr = fscanf(input, "%d", &NumXSs_Mat);
    (void) inputErr;

    IdentifyKeyword(input, keyword, "Num_of_ScattOrder");
    JumpCommentsInFile(input);
    inputErr = fscanf(input, "%d", &NumScattOrder);
    (void) inputErr;

    fclose(input);
}

M_UINT64 GetNNZ_MacroXS_s()
{
	M_INT32 _globFSRID,_mat_id;
	M_UINT64 _MacroXS_s_NNZ=0;
	if ((INPUT_XSs_MATorFSR == 0 && ProblemType == 0) || (INPUT_XSs_MATorFSR == 0 && ProblemType == 2)) {
		_MacroXS_s_NNZ=0;
	    for (_globFSRID=0;_globFSRID<TotalNumFSRs_subDom;_globFSRID++) {
		    _mat_id=FSRsXSID_subDom[_globFSRID];
		    _MacroXS_s_NNZ += InpXSs_s_NNZ[_mat_id];
	    }
	}
	return (_MacroXS_s_NNZ);
}

M_UINT64 GetMaxNNZ_MacroXS_s_Lay()
{
	M_UINT64 _MacroXS_s_MaxNNZ=0;
#if defined(CUDA_USED)
	size_t _globFSRID, _mat_id, _lay, _fsr, g, _ind = 0, _count, _ord;
	if ((INPUT_XSs_MATorFSR == 0 && ProblemType == 0) || (INPUT_XSs_MATorFSR == 0 && ProblemType == 2)) {
	    MacroXS_s_NNZ_Layer = AllocMem_ui64_I((size_t)(MOCNumLayersZ), (char *)("MacroXS_s_NNZ_Layer"));
	    MacroXS_s_NZStartID_Layer = AllocMem_ui64_I((size_t)(MOCNumLayersZ), (char *)("MacroXS_s_NZStartID_Layer"));
        MacroXS_s_NZRowStartID_Layer = AllocMem_ui64_I((size_t)(TotalNumFSRs_subDom*(NumScattOrder+1)*NumGroups+MOCNumLayersZ), (char *)("MacroXS_s_NZRowStartID_Layer"));
		for (_lay = 0; _lay < MOCNumLayersZ; _lay++) {
			_count=0;
			for (_fsr=0; _fsr<TotalNumFSRs_perLayer; _fsr++) {
				_globFSRID=_lay*TotalNumFSRs_perLayer+_fsr;
				_mat_id=FSRsXSID_subDom[_globFSRID];
				_count += InpXSs_s_NNZ[_mat_id];
			}
			if (_MacroXS_s_MaxNNZ < _count) {
				_MacroXS_s_MaxNNZ = _count;
			}
			MacroXS_s_NNZ_Layer[_lay] = _count;
			if (_lay == 0) {
				MacroXS_s_NZStartID_Layer[_lay] = 0;
			}
			else {
				MacroXS_s_NZStartID_Layer[_lay] = MacroXS_s_NZStartID_Layer[_lay - 1] + MacroXS_s_NNZ_Layer[_lay - 1];
			}
		}

		for (_lay = 0; _lay < MOCNumLayersZ; _lay++) {
			_count=0;
			for (_fsr=0; _fsr<TotalNumFSRs_perLayer; _fsr++) {
				_globFSRID=_lay*TotalNumFSRs_perLayer+_fsr;
				_mat_id=FSRsXSID_subDom[_globFSRID];
				for (_ord=0; _ord<NumScattOrder+1;_ord++) {
					for (g = 0; g<NumGroups; g++) {
						_ind = _globFSRID*(NumScattOrder+1)*NumGroups+_ord*NumGroups+g+_lay;
						if (_fsr == 0 && _ord==0 && g == 0) {
							MacroXS_s_NZRowStartID_Layer[_ind] = 0;
						}
						else {
						    MacroXS_s_NZRowStartID_Layer[_ind] = _count;
					    }
 						_count += InpXSs_s_NZColIDEnd[_mat_id][_ord*NumGroups+g] - InpXSs_s_NZColIDStart[_mat_id][_ord*NumGroups+g];
					}
				}
			}
			MacroXS_s_NZRowStartID_Layer[_ind+1] = _count;
		}
	}
#endif // deifned(CUDA_USED)
	return (_MacroXS_s_MaxNNZ);
}

/**
* read macroscopic XSs(for FSRs) from input file - material BLOCK
*/
void InputXSs_forFSRs(FILE* _fp1, int _fileIdx) {
    int g, g1, i, _inpErr, _gLow=0, _gUp=0, _gLowBool=0, j, base_fsr_idx, local_fsr_idx, global_fsr_idx;
    FP_PRECISION _sigs;
    char _keyword[KEYWORD_N];
    FP_PRECISION sum_chi = 0;  // for adjoint

    JumpCommentsInFile(_fp1);
    IdentifyKeyword(_fp1, _keyword, "INPUT_MATorXSs");
    fscanf(_fp1, "%*s");

    JumpCommentsInFile(_fp1);
    IdentifyKeyword(_fp1, _keyword, "INPUT_XSs_MATorFSR");
    fscanf(_fp1, "%*s");

    JumpCommentsInFile(_fp1);
    IdentifyKeyword(_fp1, _keyword, "Num_Group");
    fscanf(_fp1, "%*s");

    JumpCommentsInFile(_fp1);
    IdentifyKeyword(_fp1, _keyword, "IfSetXSsForEveryFSR");
    fscanf(_fp1, "%*s");

    JumpCommentsInFile(_fp1);
    IdentifyKeyword(_fp1, _keyword, "reCalcTotalXS");
    fscanf(_fp1, "%*s");

    JumpCommentsInFile(_fp1);
    IdentifyKeyword(_fp1, _keyword, "reCalcAbsorbXS");
    fscanf(_fp1, "%*s");

    JumpCommentsInFile(_fp1);
    IdentifyKeyword(_fp1, _keyword, "transportCorrection");
    fscanf(_fp1, "%*s");

    int num_fsr_cur_file;
    JumpCommentsInFile(_fp1);
    IdentifyKeyword(_fp1, _keyword, "Num_of_XSsTypes_FSR");
    fscanf(_fp1, "%d", &num_fsr_cur_file);

    JumpCommentsInFile(_fp1);
    IdentifyKeyword(_fp1, _keyword, "Num_of_ScattOrder");
    fscanf(_fp1, "%*s");

//    IdentifyKeyword(_fp1, _keyword, "Num_of_XSsTypes_FSR");
//    JumpCommentsInFile(_fp1);
//    _inpErr = fscanf(_fp1, "%d", &NumXSs_Mat);
//    IdentifyKeyword(_fp1, _keyword, "Num_of_ScattOrder");
//    JumpCommentsInFile(_fp1);
//    _inpErr = fscanf(_fp1, "%d", &NumScattOrder);

    if (_fileIdx == 0) {
        if (XSs_Mat_Id == NULL)
            XSs_Mat_Id = AllocMem_i32_i32(TotalNumFSRs_perLayer, (char *)("XSs_Mat_Id"));
        if (InpXSs_t == NULL)
            InpXSs_t = AllocMem_FP_II(TotalNumFSRs_perLayer, NumGroups, (char *)("InpXSs_t"));

        //eigenvalue problem
        // alloc the memory
        if (ProblemType == 0) {
            if (InpXSs_a == NULL)
                InpXSs_a = AllocMem_FP_II(TotalNumFSRs_perLayer, NumGroups, (char *)("InpXSs_a"));
            if (InpXSs_f == NULL)
                InpXSs_f = AllocMem_FP_II(TotalNumFSRs_perLayer, NumGroups, (char *)("InpXSs_f"));
            if (InpXSs_nf == NULL)
                InpXSs_nf = AllocMem_FP_II(TotalNumFSRs_perLayer, NumGroups, (char *)("InpXSs_nf"));
            if (InpChi == NULL)
                InpChi = AllocMem_FP_II(TotalNumFSRs_perLayer, NumGroups, (char *)("InpChi"));

            if (NumGroups == 2164)
            {
                if (TempXSs_s == NULL)
                {
                    TempXSs_s = AllocMem_FP_III(NumScattOrder + 1, NumGroups, NumGroups, (char *)("TempXSs_s"));
                }

                if (TempXSs_T == NULL)
                {
                    TempXSs_T = AllocMem_FP64_I(NumGroups, (char *)("TempXS_Tot_Nuf_Chi"));
                }

                if (TempIndex == NULL)
                {
                    TempIndex = AllocMem_int_I(NumGroups * (NumScattOrder + 1), (char *)("TempIndex"));
                }
            }
            else
            {
                if (InpXSs_s == NULL)
                    InpXSs_s = AllocMem_FP_III(TotalNumFSRs_perLayer, (NumScattOrder + 1) * NumGroups, NumGroups, (char *)("InpXSs_s"));
            }         

            if (InpXSs_s_NNZ == NULL)
                InpXSs_s_NNZ = AllocMem_i32_i32(TotalNumFSRs_perLayer, (char *)("InpXSs_s_NNZ"));
            if (InpXSs_s_NZColIDStart == NULL)
                InpXSs_s_NZColIDStart = AllocMem_int_II(TotalNumFSRs_perLayer,(NumScattOrder + 1) * NumGroups, (char *)("InpXSs_s_NZColIDStart"));
            if (InpXSs_s_NZColIDEnd == NULL)
                InpXSs_s_NZColIDEnd = AllocMem_int_II(TotalNumFSRs_perLayer,(NumScattOrder + 1) * NumGroups, (char *)("InpXSs_s_NZColIDEnd"));

            if (TransCorre == 1) {
                if (InpXSs_p0 == NULL && InpXSs_p1 == NULL) {
                    InpXSs_p0 = AllocMem_FP_III(TotalNumFSRs_perLayer, NumGroups, NumGroups, (char *)("InpXSs_p0"));
                    InpXSs_p1 = AllocMem_FP_III(TotalNumFSRs_perLayer, NumGroups, NumGroups, (char *)("InpXSs_p1"));
                }
            }
            else {}
        }
        else if (ProblemType == 1) {
            if (Inp_FixedSour == NULL)
                Inp_FixedSour = AllocMem_FP_II(TotalNumFSRs_perLayer, NumGroups, (char *)("Inp_FixedSour"));
        }
        else {}
    }

    // the number of FSR must be equal in each layer for MOC-EX method
    base_fsr_idx = _fileIdx * num_fsr_cur_file;

    if (NumGroups < 10000) {

        for (local_fsr_idx = 0; local_fsr_idx < num_fsr_cur_file; local_fsr_idx++) {
            global_fsr_idx = local_fsr_idx + base_fsr_idx;

            JumpCommentsInFile(_fp1);
            _inpErr = fscanf(_fp1, "%*s%*s%*s%*s");  // Lay, Cell, localFSRid, mat id
            XSs_Mat_Id[local_fsr_idx] = global_fsr_idx;
//        printf("\n file_idx=%d, XSs_Mat_Id = %d",_fileIdx, XSs_Mat_Id[i]);
            InpXSs_s_NNZ[local_fsr_idx] = 0;
            for (g = 0; g < NumGroups; g++) {
//            JumpCommentsInFile(_fp1);
                _inpErr = fscanf(_fp1, "%*s");  // Group id
                _inpErr = fscanf(_fp1, FP_FORMAT, &InpXSs_t[local_fsr_idx][g]);

                if (ProblemType == 0) {
//                JumpCommentsInFile(_fp1);
                    _inpErr = fscanf(_fp1, FP_FORMAT, &InpXSs_a[local_fsr_idx][g]);
//                JumpCommentsInFile(_fp1);
                    _inpErr = fscanf(_fp1, FP_FORMAT, &InpXSs_f[local_fsr_idx][g]);
//                JumpCommentsInFile(_fp1);

                    if (InpXSs_f[local_fsr_idx][g] > SMALL_VALUE) {
                        if (isAdjoint) {
                            _inpErr = fscanf(_fp1, FP_FORMAT, &InpChi[local_fsr_idx][g]);
                        } else {
                            _inpErr = fscanf(_fp1, FP_FORMAT, &InpXSs_nf[local_fsr_idx][g]);
                        }
                    } else {
                        _inpErr = fscanf(_fp1, FP_FORMAT, &InpXSs_nf[local_fsr_idx][g]);
                    }

//				_gLowBool=0;
//				_gLow = 0;
//				_gUp = 0;
                    for (g1 = 0; g1 < NumGroups; g1++) {
//                    JumpCommentsInFile(_fp1);

                        if (isAdjoint) {
                            _inpErr = fscanf(_fp1, FP_FORMAT, &InpXSs_s[local_fsr_idx][g1][g]);
                        } else {
                            _inpErr = fscanf(_fp1, FP_FORMAT, &InpXSs_s[local_fsr_idx][g][g1]);
                        }

//					if (fabs(InpXSs_s[i][g][g1])>SMALL_VALUE) {
//						_gLowBool++;
//						if (_gLowBool==1) {
//							_gLow=g1;
//						}
//						_gUp=g1;
//					}

                        if (TransCorre == 1) {
                            InpXSs_p0[local_fsr_idx][g][g1] = InpXSs_s[local_fsr_idx][g][g1];
                        }
                    }

//				if (_gLowBool>0) {
//					InpXSs_s_NNZ[i] += _gUp-_gLow+1;
//				    // printf("\n i=%d, g=%2d, _gLow=%2d, _gUp=%2d",i,g,_gLow,_gUp);
//				    InpXSs_s_NZColIDStart[i][g]=_gLow;
//				    InpXSs_s_NZColIDEnd[i][g]=_gUp+1;
//				}
//				else {
//					InpXSs_s_NZColIDStart[i][g]=0;
//				    InpXSs_s_NZColIDEnd[i][g]=0;
//				}

//                JumpCommentsInFile(_fp1);

                    if (InpXSs_f[local_fsr_idx][g] > SMALL_VALUE) {
                        if (isAdjoint) {
                            _inpErr = fscanf(_fp1, FP_FORMAT, &InpXSs_nf[local_fsr_idx][g]);
                        } else {
                            _inpErr = fscanf(_fp1, FP_FORMAT, &InpChi[local_fsr_idx][g]);
                        }
                    } else {
                        _inpErr = fscanf(_fp1, FP_FORMAT, &InpChi[local_fsr_idx][g]);
                    }
//                printf("\nFSR %d, g %d  InpChi[%d][%d]=%e", global_fsr_idx, g, InpChi[local_fsr_idx][g]);

                } else if (ProblemType == 1) {
                    JumpCommentsInFile(_fp1);
                    _inpErr = fscanf(_fp1, FP_FORMAT, &Inp_FixedSour[local_fsr_idx][g]);
                } else {}
            }

            if (isAdjoint) {
                // treat the XSs_nf and chi for adjoint calculation
                sum_chi = sum(InpChi[local_fsr_idx], 0, NumGroups);

                // is fission region
                if (sum_chi > SMALL_VALUE) {
                    for (g = 0; g < NumGroups; g++) {
                        InpChi[local_fsr_idx][g] = InpChi[local_fsr_idx][g] / sum_chi;
                        InpXSs_nf[local_fsr_idx][g] = InpXSs_nf[local_fsr_idx][g] * sum_chi;
                    }
                }

            }

            /* reading high-order XSs_s */
            if (NumScattOrder > 0) {

                for (j = 1; j <= NumScattOrder; j++) {
                    for (g = 0; g < NumGroups; g++) {
//                    JumpCommentsInFile(_fp1);
                        _inpErr = fscanf(_fp1, "%*s");  // group id
//                    _gLowBool=0;
//                    _gLow = 0;
//                    _gUp = 0;
                        for (g1 = 0; g1 < NumGroups; g1++) {
//                        JumpCommentsInFile(_fp1);
                            _inpErr = fscanf(_fp1, FP_FORMAT, &InpXSs_s[local_fsr_idx][j * NumGroups + g][g1]);

//                        if (fabs(InpXSs_s[i][j*NumGroups+g][g1])>SMALL_VALUE) {
//                            _gLowBool++;
//                            if (_gLowBool==1) {
//                                _gLow=g1;
//                            }
//                            _gUp=g1;
//                        }
                        }

//                    if (_gLowBool>0) {
//                        InpXSs_s_NNZ[i] += _gUp-_gLow+1;
//                        InpXSs_s_NZColIDStart[i][j*NumGroups+g]=_gLow;
//                        InpXSs_s_NZColIDEnd[i][j*NumGroups+g]=_gUp+1;
//                    }
//                    else {
//                        InpXSs_s_NZColIDStart[i][j*NumGroups+g]=0;
//                        InpXSs_s_NZColIDEnd[i][j*NumGroups+g]=0;
//                    }
                    }
                }
            }

            // calc the CSR param of scattering matrix
            for (g = 0; g < NumGroups * (NumScattOrder + 1); g++) {
                _gLowBool = 0;
                _gLow = 0;
                _gUp = 0;

                for (g1 = 0; g1 < NumGroups; g1++) {
//                printf("%e ", InpXSs_s[i][g][g1]);
                    if (fabs(InpXSs_s[local_fsr_idx][g][g1]) > 0.0) {
                        _gLowBool++;
                        if (_gLowBool == 1) {
                            _gLow = g1;
                        }
                        _gUp = g1;
                    }
                }

                if (_gLowBool > 0) {
                    InpXSs_s_NNZ[local_fsr_idx] += _gUp - _gLow + 1;
                    InpXSs_s_NZColIDStart[local_fsr_idx][g] = _gLow;
                    InpXSs_s_NZColIDEnd[local_fsr_idx][g] = _gUp + 1;
                } else {
                    InpXSs_s_NZColIDStart[local_fsr_idx][g] = 0;
                    InpXSs_s_NZColIDEnd[local_fsr_idx][g] = 0;
                }

//            printf("\n fsr=%d, g=%d, start=%d, end=%d", local_fsr_idx, g,
//                   InpXSs_s_NZColIDStart[local_fsr_idx][g], InpXSs_s_NZColIDEnd[local_fsr_idx][g]);

            }
//        printf("\n\n");


            if (ProblemType == 0) {
                FissionSpectrumCheck(NumGroups, 0, InpChi[local_fsr_idx]);
            }
            // }
            // for (i = 0; i < NumXSs_Mat; i++) {
            // transport correction
            if (ProblemType == 0 && TransCorre == 1) {
                for (g = 0; g < NumGroups; g++) {
                    _sigs = 0.;
                    for (g1 = 0; g1 < NumGroups; g1++) {
                        JumpCommentsInFile(_fp1);
                        _inpErr = fscanf(_fp1, FP_FORMAT, &InpXSs_p1[local_fsr_idx][g][g1]);
                        _sigs += InpXSs_p1[local_fsr_idx][g][g1];
                    }
                    InpXSs_s[local_fsr_idx][g][g] = InpXSs_s[local_fsr_idx][g][g] - _sigs;
                }
            } else {}
            // total XSs recalculate
            if (ProblemType == 0 && ReCalcXSt == 1 && ReCalcXSa == 0) {
                for (g = 0; g < NumGroups; g++) {
                    _sigs = 0.;
                    for (g1 = 0; g1 < NumGroups; g1++) {
                        _sigs += InpXSs_s[local_fsr_idx][g][g1];
                    }
                    InpXSs_t[local_fsr_idx][g] = InpXSs_a[local_fsr_idx][g] + _sigs;
                    //printf("imat = %d, g = %d, t = %e, a = %e\n", i, g, InpXSs_t[i][g], InpXSs_a[i][g]);

                }
            } else if (ProblemType == 0 && ReCalcXSt == 0 && ReCalcXSa == 1) {
                for (g = 0; g < NumGroups; g++) {
                    _sigs = 0.;
                    for (g1 = 0; g1 < NumGroups; g1++) {
                        _sigs += InpXSs_s[local_fsr_idx][g][g1];
                    }
                    InpXSs_a[local_fsr_idx][g] = InpXSs_t[local_fsr_idx][g] - _sigs;
                }
            } else {}
        }
    }
    else 
    {    
        FILE *fb = NULL;
        fb = fopen("binaryXs", "rb");
        int fi, fstp, fedp, itemp;
        double buff;
        size_t _count = 0;
        int _ind;
        int od;
        int ctIndex;
    
        for (_ind = 0; _ind < num_fsr_cur_file; _ind++)
        {
            _inpErr = fread(TempXSs_T, sizeof(double), NumGroups, fb);

            for (g = 0; g < NumGroups; g++)
            {
                InpXSs_t[_ind][g] = (FP_PRECISION) TempXSs_T[g];
            }
        }
    
        for (_ind = 0; _ind < num_fsr_cur_file; _ind++)
        {
            _inpErr = fread(TempXSs_T, sizeof(double), NumGroups, fb);

            for (g = 0; g < NumGroups; g++)
            {
                InpXSs_nf[_ind][g] = (FP_PRECISION) TempXSs_T[g];
            }
        }
    
        for (_ind = 0; _ind < num_fsr_cur_file; _ind++)
        {
            _inpErr = fread(TempXSs_T, sizeof(double), NumGroups, fb);

            for (g = 0; g < NumGroups; g++)
            {
                InpChi[_ind][g] = (FP_PRECISION) TempXSs_T[g];
            }
    
            if (ProblemType == 0)  
            {
                FissionSpectrumCheck(NumGroups,0, InpChi[_ind]);
            }
        }
    
        _count = 0;
        
        if (MacroXS_s == NULL)
        {
            for (_ind = 0; _ind < num_fsr_cur_file; _ind++)
            {
                _inpErr = fread(TempIndex, sizeof (int), NumGroups * (NumScattOrder + 1), fb);
                ctIndex = 0;

                for (od = 0; od <= NumScattOrder; od++)
                {
                    for (g = 0; g < NumGroups; g++)
                    {
                        InpXSs_s_NZColIDStart[_ind][od * NumGroups + g] = TempIndex[ctIndex];
                        ctIndex++;
                    }
                }
            }
    
            for (_ind = 0; _ind < num_fsr_cur_file; _ind++)
            {
                _inpErr = fread(TempIndex, sizeof (int), NumGroups * (NumScattOrder + 1), fb);
                ctIndex = 0;

                for (od = 0; od <= NumScattOrder; od++)
                {
                    for (g = 0; g < NumGroups; g++)
                    {
                        InpXSs_s_NZColIDEnd[_ind][od * NumGroups + g] = TempIndex[ctIndex];
                        ctIndex++;
                    }
                }
            }
            
            for (_ind = 0; _ind < num_fsr_cur_file; _ind++)
            {
                InpXSs_s_NNZ[_ind] = 0;
                for (od = 0; od <= NumScattOrder; od++)
                {
                    for (g = 0; g < NumGroups; g++)
                    {
                        InpXSs_s_NNZ[_ind] += InpXSs_s_NZColIDEnd[_ind][od * NumGroups + g] - InpXSs_s_NZColIDStart[_ind][od * NumGroups + g];
                    }
                }
            }
    
            for (_ind = 0; _ind < num_fsr_cur_file; _ind++)
            {
                for (od = 0; od <= NumScattOrder; od++)
                {
                    for (g = 0; g < NumGroups; g++)
                    {
                        fstp = InpXSs_s_NZColIDStart[_ind][od * NumGroups + g];
                        fedp = InpXSs_s_NZColIDEnd[_ind][od * NumGroups + g];
                        for (g1 = fstp; g1 < fedp; g1++)
                        {
                            _inpErr = fread(&buff, sizeof(double), 1, fb);
                            _count++;
                        }
                    }
                }
            }
    
            printf("\n [  DEBUG  ]  NNZ is %ld\n", _count);
        }
        else
        {
            for (_ind = 0; _ind < num_fsr_cur_file; _ind++)
            {
                _inpErr = fread(TempIndex, sizeof (int), NumGroups * (NumScattOrder + 1), fb);
            }
    
            for (_ind = 0; _ind < num_fsr_cur_file; _ind++)
            {
                _inpErr = fread(TempIndex, sizeof (int), NumGroups * (NumScattOrder + 1), fb);
            }
    
            for (_ind = 0; _ind < num_fsr_cur_file; _ind++)
            {
                for (od = 0; od <= NumScattOrder; od++)
                {
                    for (g = 0; g < NumGroups; g++)
                    {
                        fstp = InpXSs_s_NZColIDStart[_ind][od * NumGroups + g];
                        fedp = InpXSs_s_NZColIDEnd[_ind][od * NumGroups + g];

                        _inpErr = fread(TempXSs_T, sizeof(double), fedp - fstp, fb);
                        for (g1 = fstp; g1 < fedp; g1++)
                        {
                            MacroXS_s[_count] = (FP_PRECISION) TempXSs_T[g1 - fstp];
                            _count++;
                        }
                    }
                }
            }
        }
    
        fclose(fb);
        fb = NULL;
    }
}

/* -------------------------------------------------------------------------- */
/* Binary FSR XS reader (dense or CSR band storage).
 *
 * The binary format mirrors the text FSR XS files:
 *   header   : 9 x int32 (INPUT_MATorXSs, INPUT_XSs_MATorFSR, Num_Group,
 *              IfSetXSsForEveryFSR, reCalcTotalXS, reCalcAbsorbXS,
 *              transportCorrection, Num_of_XSsTypes_FSR, Num_of_ScattOrder)
 *   per FSR  : 4 x int32 placeholder
 *   P0 rows  : int32 grp+1, float sig_t/a/f/nf, scattering row, float chi
 *   Pk rows  : int32 grp+1, scattering row
 *   scattering row (dense): numGrp x float; band = [first, last+1] of nonzeros
 *   scattering row (CSR)  : int32 colStart, int32 colEnd,
 *                           (colEnd-colStart) x float band values
 *
 * Only eigenvalue (ProblemType==0) files are supported and the reCalc /
 * transport-correction flags must be 0. InpXSs_s (dense) is intentionally
 * never allocated here. In pass 2 (_fillValues==1) the band values are written
 * directly into MacroXS_s at the layer offset, so the assembly skips its own
 * value copy for binary files.
 */
static int IsBinaryXsFile(const char* _path) {
    /* 1 = CSR binary (.csr), 2 = dense binary (.bin), 0 = text */
    size_t _len = strlen(_path);
    if (_len >= 4 && strcmp(_path + _len - 4, ".csr") == 0) return 1;
    if (_len >= 4 && strcmp(_path + _len - 4, ".bin") == 0) return 2;
    return 0;
}

void InputXSs_forFSRs_binary(FILE* _fp1, int _fileIdx, int _isCsr, int _fillValues) {
    int g, g1, j, local_fsr_idx, global_fsr_idx, base_fsr_idx, num_fsr_cur_file;
    int _hdr[9];
    size_t _readCnt;
    M_UINT64 _localCount = 0, _layerBase = 0;
    FP_PRECISION* _rowBuf = NULL;

    /* ---- header ---- */
    _readCnt = fread(_hdr, sizeof(int), 9, _fp1);
    if (_readCnt != 9) {
        printf("\n [**ERROR**]  Failed to read binary FSR XS header (fileIdx=%d)\n", _fileIdx);
        fflush(stdout);
        exit(69);
    }
    if (_hdr[0] != 1 || _hdr[1] != 1 || _hdr[3] != 1) {
        printf("\n [**ERROR**]  Binary FSR XS file must be per-FSR format (INPUT_MATorXSs=1, INPUT_XSs_MATorFSR=1, IfSetXSsForEveryFSR=1)\n");
        fflush(stdout);
        exit(69);
    }
    if (_hdr[2] != NumGroups) {
        printf("\n [**ERROR**]  Binary FSR XS group number %d != NumGroups %d\n", _hdr[2], NumGroups);
        fflush(stdout);
        exit(69);
    }
    if (_hdr[8] != NumScattOrder) {
        printf("\n [**ERROR**]  Binary FSR XS scatter order %d != NumScattOrder %d\n", _hdr[8], NumScattOrder);
        fflush(stdout);
        exit(69);
    }
    if (_hdr[4] != 0 || _hdr[5] != 0 || _hdr[6] != 0) {
        printf("\n [**ERROR**]  Binary FSR XS file with reCalcTotalXS/reCalcAbsorbXS/transportCorrection != 0 is not supported\n");
        fflush(stdout);
        exit(69);
    }
    if (ProblemType != 0) {
        printf("\n [**ERROR**]  Binary FSR XS reader only supports eigenvalue problem (ProblemType==0)\n");
        fflush(stdout);
        exit(69);
    }
    num_fsr_cur_file = _hdr[7];
    if (num_fsr_cur_file != TotalNumFSRs_perLayer) {
        printf("\n [ WARNING ]  Binary FSR XS file FSR number %d != TotalNumFSRs_perLayer %d\n",
               num_fsr_cur_file, TotalNumFSRs_perLayer);
        fflush(stdout);
    }

    /* ---- allocate per-layer input arrays (same as text path, minus dense InpXSs_s) ---- */
    if (_fileIdx == 0) {
        if (XSs_Mat_Id == NULL)
            XSs_Mat_Id = AllocMem_i32_i32(TotalNumFSRs_perLayer, (char *)("XSs_Mat_Id"));
        if (InpXSs_t == NULL)
            InpXSs_t = AllocMem_FP_II(TotalNumFSRs_perLayer, NumGroups, (char *)("InpXSs_t"));
        if (InpXSs_a == NULL)
            InpXSs_a = AllocMem_FP_II(TotalNumFSRs_perLayer, NumGroups, (char *)("InpXSs_a"));
        if (InpXSs_f == NULL)
            InpXSs_f = AllocMem_FP_II(TotalNumFSRs_perLayer, NumGroups, (char *)("InpXSs_f"));
        if (InpXSs_nf == NULL)
            InpXSs_nf = AllocMem_FP_II(TotalNumFSRs_perLayer, NumGroups, (char *)("InpXSs_nf"));
        if (InpChi == NULL)
            InpChi = AllocMem_FP_II(TotalNumFSRs_perLayer, NumGroups, (char *)("InpChi"));
        if (InpXSs_s_NNZ == NULL)
            InpXSs_s_NNZ = AllocMem_i32_i32(TotalNumFSRs_perLayer, (char *)("InpXSs_s_NNZ"));
        if (InpXSs_s_NZColIDStart == NULL)
            InpXSs_s_NZColIDStart = AllocMem_int_II(TotalNumFSRs_perLayer, (NumScattOrder + 1) * NumGroups, (char *)("InpXSs_s_NZColIDStart"));
        if (InpXSs_s_NZColIDEnd == NULL)
            InpXSs_s_NZColIDEnd = AllocMem_int_II(TotalNumFSRs_perLayer, (NumScattOrder + 1) * NumGroups, (char *)("InpXSs_s_NZColIDEnd"));
        /* NOTE: InpXSs_s (dense) is intentionally NOT allocated in the binary path */
    }

    if (!_isCsr) {
        _rowBuf = (FP_PRECISION *) malloc((size_t) NumGroups * sizeof (FP_PRECISION));
        if (_rowBuf == NULL) {
            printf("\n [**ERROR**]  Failed to allocate binary FSR XS row buffer\n");
            fflush(stdout);
            exit(69);
        }
    }

    if (_fillValues) {
        _layerBase = MacroXS_s_NZStartID_Layer[_fileIdx];
    }

    base_fsr_idx = _fileIdx * num_fsr_cur_file;

    for (local_fsr_idx = 0; local_fsr_idx < num_fsr_cur_file; local_fsr_idx++) {
        int _dummy4[4];
        global_fsr_idx = local_fsr_idx + base_fsr_idx;

        _readCnt = fread(_dummy4, sizeof(int), 4, _fp1);
        if (_readCnt != 4) {
            printf("\n [**ERROR**]  Unexpected end of binary FSR XS file (fsr=%d)\n", global_fsr_idx);
            fflush(stdout);
            exit(69);
        }
        XSs_Mat_Id[local_fsr_idx] = global_fsr_idx;
        InpXSs_s_NNZ[local_fsr_idx] = 0;

        /* P0 rows */
        for (g = 0; g < NumGroups; g++) {
            int _gid;
            int _ordIdx = g;
            int _cs = 0, _ce = 0, _len;

            fread(&_gid, sizeof(int), 1, _fp1);
            fread(&InpXSs_t[local_fsr_idx][g], sizeof (FP_PRECISION), 1, _fp1);
            fread(&InpXSs_a[local_fsr_idx][g], sizeof (FP_PRECISION), 1, _fp1);
            fread(&InpXSs_f[local_fsr_idx][g], sizeof (FP_PRECISION), 1, _fp1);
            fread(&InpXSs_nf[local_fsr_idx][g], sizeof (FP_PRECISION), 1, _fp1);

            if (_isCsr) {
                fread(&_cs, sizeof(int), 1, _fp1);
                fread(&_ce, sizeof(int), 1, _fp1);
            } else {
                int _first = -1, _last = -1;
                _readCnt = fread(_rowBuf, sizeof (FP_PRECISION), NumGroups, _fp1);
                if (_readCnt != (size_t) NumGroups) {
                    printf("\n [**ERROR**]  Unexpected end of binary FSR XS file (fsr=%d, P0 grp=%d)\n", global_fsr_idx, g);
                    fflush(stdout);
                    exit(69);
                }
                for (g1 = 0; g1 < NumGroups; g1++) {
                    if (_rowBuf[g1] != (FP_PRECISION) 0.) {
                        if (_first < 0) {
                            _first = g1;
                        }
                        _last = g1;
                    }
                }
                _cs = (_first < 0) ? 0 : _first;
                _ce = (_first < 0) ? 0 : _last + 1;
            }

            _len = _ce - _cs;
            InpXSs_s_NZColIDStart[local_fsr_idx][_ordIdx] = _cs;
            InpXSs_s_NZColIDEnd[local_fsr_idx][_ordIdx] = _ce;
            InpXSs_s_NNZ[local_fsr_idx] += _len;

            if (_fillValues && _len > 0) {
                if (_isCsr) {
                    _readCnt = fread(&MacroXS_s[_layerBase + _localCount], sizeof (FP_PRECISION), _len, _fp1);
                    if (_readCnt != (size_t) _len) {
                        printf("\n [**ERROR**]  Unexpected end of binary FSR XS file (fsr=%d, P0 grp=%d)\n", global_fsr_idx, g);
                        fflush(stdout);
                        exit(69);
                    }
                } else {
                    for (g1 = 0; g1 < _len; g1++) {
                        MacroXS_s[_layerBase + _localCount + g1] = _rowBuf[_cs + g1];
                    }
                }
                _localCount += _len;
            } else if (_len > 0) {
                fseek(_fp1, (long) (_len * sizeof (FP_PRECISION)), SEEK_CUR);
            }

            fread(&InpChi[local_fsr_idx][g], sizeof (FP_PRECISION), 1, _fp1);
        }

        /* higher order rows */
        for (j = 1; j <= NumScattOrder; j++) {
            for (g = 0; g < NumGroups; g++) {
                int _gid;
                int _ordIdx = j * NumGroups + g;
                int _cs = 0, _ce = 0, _len;

                fread(&_gid, sizeof(int), 1, _fp1);

                if (_isCsr) {
                    fread(&_cs, sizeof(int), 1, _fp1);
                    fread(&_ce, sizeof(int), 1, _fp1);
                } else {
                    int _first = -1, _last = -1;
                    _readCnt = fread(_rowBuf, sizeof (FP_PRECISION), NumGroups, _fp1);
                    if (_readCnt != (size_t) NumGroups) {
                        printf("\n [**ERROR**]  Unexpected end of binary FSR XS file (fsr=%d, P%d grp=%d)\n", global_fsr_idx, j, g);
                        fflush(stdout);
                        exit(69);
                    }
                    for (g1 = 0; g1 < NumGroups; g1++) {
                        if (_rowBuf[g1] != (FP_PRECISION) 0.) {
                            if (_first < 0) {
                                _first = g1;
                            }
                            _last = g1;
                        }
                    }
                    _cs = (_first < 0) ? 0 : _first;
                    _ce = (_first < 0) ? 0 : _last + 1;
                }

                _len = _ce - _cs;
                InpXSs_s_NZColIDStart[local_fsr_idx][_ordIdx] = _cs;
                InpXSs_s_NZColIDEnd[local_fsr_idx][_ordIdx] = _ce;
                InpXSs_s_NNZ[local_fsr_idx] += _len;

                if (_fillValues && _len > 0) {
                    if (_isCsr) {
                        _readCnt = fread(&MacroXS_s[_layerBase + _localCount], sizeof (FP_PRECISION), _len, _fp1);
                        if (_readCnt != (size_t) _len) {
                            printf("\n [**ERROR**]  Unexpected end of binary FSR XS file (fsr=%d, P%d grp=%d)\n", global_fsr_idx, j, g);
                            fflush(stdout);
                            exit(69);
                        }
                    } else {
                        for (g1 = 0; g1 < _len; g1++) {
                            MacroXS_s[_layerBase + _localCount + g1] = _rowBuf[_cs + g1];
                        }
                    }
                    _localCount += _len;
                } else if (_len > 0) {
                    fseek(_fp1, (long) (_len * sizeof (FP_PRECISION)), SEEK_CUR);
                }
            }
        }

        if (ProblemType == 0) {
            FissionSpectrumCheck(NumGroups, 0, InpChi[local_fsr_idx]);
        }
    }

    if (_rowBuf != NULL) {
        free(_rowBuf);
        _rowBuf = NULL;
    }
}

void MacroXSsEvaluation_subDomain(int _subDomID_row, int _subDomID_column)
{
	int g, g1/*, i, j*//*, k, i1*//*, j1*/, k1,_lay/*,_globLayID*/;
	int /*_asse_type,*/ /*_latt_type, _pin_inLatt*//*, _glo_pin_id,*/ _mat_id/*, _fsr_id*//*, _CellID_inSubDom*/, _subDomID_set/*, _subDomSeqID*/;
	//struct CELLS_IN_PINCELL *_pointer1;
	M_INT32 _globFSRID,_ind,_fsr, _mul,_ord;
    M_INT32 _localFSRID;
	M_INT32 _TotalNumFSRs_subDom, _TotalNumFSRs_perLayer, _NumGroups;
	size_t _dim, _dim1;
	M_UINT64 _count;
	_TotalNumFSRs_subDom = TotalNumFSRs_subDom;
	_TotalNumFSRs_perLayer = TotalNumFSRs_perLayer;
	_NumGroups = NumGroups;
	_dim = (size_t)(_TotalNumFSRs_subDom) * (size_t)(_NumGroups);
	_dim1 = (size_t)(_TotalNumFSRs_subDom) * (size_t)((NumScattOrder + 1) * _NumGroups);
	_mul = (NumScattOrder + 1) * _NumGroups;

#ifdef TIME_REC
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC
	if (ProcID == 0) {
		printf("\n [  INFO   ]  Evaluating the Macroscopic Cross Sections ...");
		fflush(stdout);
		fprintf(LogFile, "\n [  INFO   ]  Evaluating the Macroscopic Cross Sections ...");
	}
	_subDomID_set = GetSubDomSequenceID(_subDomID_row, _subDomID_column, NumSubDomains);


    /// ALLOC THE MEMORY
    printf("\n [  INFO   ]  Allocating Memory for the Macroscopic Cross Sections ...");
    fflush(stdout);
	if (IfSetXSsForEveryFSR == 1) {
		MacroXS_t = AllocMem_FP_I(_dim, (char *)("MacroXS_t"));
		if (ProblemType == 0) {
			MacroXS_a = AllocMem_FP_I(_dim, (char *)("MacroXS_a"));
			MacroXS_f = AllocMem_FP_I(_dim, (char *)("MacroXS_f"));
			MacroXS_nf = AllocMem_FP_I(_dim, (char *)("MacroXS_nf"));
			Chi = AllocMem_FP_I(_dim, (char *)("Chi"));
			if (INPUT_XSs_MATorFSR == 0) {
				MacroXS_s = AllocMem_FP_I((size_t)(MacroXS_s_NNZ), (char *)("MacroXS_s"));
			}

			MacroXS_s_NZRowStartID = AllocMem_ui64_I(_dim1 + (M_UINT64)(1), (char *)("MacroXS_s_NZRowStartID"));
			MacroXS_s_NZColStartID = AllocMem_ui64_I(_dim1, (char *)("MacroXS_s_NZColStartID"));

			// if (TransCorre == 1) {
			// 	_dim = _dim * _NumGroups;
			// 	MacroXS_p0 = AllocMem_FP_u64I(_dim, (char *)("MacroXS_p0"));
			// 	MacroXS_p1 = AllocMem_FP_u64I(_dim, (char *)("MacroXS_p1"));
			// }
		}
		else if (ProblemType == 1) {
			FixedSour = AllocMem_FP_I(_dim, (char *)("FixedSour"));
		}
		else {}
	}

	// else if (IfSetXSsForEveryFSR == 0) {
	// 	if (INPUT_XSs_MATorFSR == 0) {
	// 		//XSs'id in FSRs
	// 		XSsID_FSRs = AllocMem_double_IIP_1(MOCNumLayersZ, NumPinsInSubDomPerLayer, NumFSRsInCell, (char *)("XSsID_FSRs"));
	// 	}
	// 	else if (INPUT_XSs_MATorFSR == 1) {
	// 		printf("\n [**ERROR**]  IfSetXSsForEveryFSR should be set as 1 when INPUT_XSs_MATorFSR=1, but IfSetXSsForEveryFSR=%d", IfSetXSsForEveryFSR);
	// 		fflush(stdout);
	// 		exit(69);
	// 	}
	// 	else {}
	// }

	else {
		printf("\n [**ERROR**]  boolean value is not correct, \"IfSetXSsForEveryFSR\" should be 0 or 1! but \"IfSetXSsForEveryFSR\"=%d", IfSetXSsForEveryFSR);
		fflush(stdout);
		exit(69);
	}

    FILE *fsr_xs_file = NULL, *layer_fsr_xs_file = NULL;
    char layer_fsr_xs_file_path[100][255];
    size_t i, nnz_layer, cnt;  // cnt for NZRowStartID_layer

		if (IfSetXSsForEveryFSR == 1) {
            _count = 0;

            // if input XSs for different FSRs
            // read XSs file by file
            // one file includes the XSs for all FSR in one layer
            // to save the memory
            // assign the MacroXS array layer by layer
            if (Input_MATorXSs == 1 && INPUT_XSs_MATorFSR == 1 && IfSetXSsForEveryFSR == 1) {
                printf("\n [  INFO   ]  Parsing the FSR XS File ...");
                fflush(stdout);
                fsr_xs_file = fopen(Mat_XSlib, "r");

                // jump the blocks which have been read
                // 0 -- INPUT_MATorXSs
                // 1 -- INPUT_XSs_MATorFSR
                // 2 -- Num_Group
                // 3 -- IfSetXSsForEveryFSR
                // 4 -- reCalcTotalXS
                // 5 -- reCalcAbsorbXS
                // 6 -- transportCorrection
                // 7 -- Num_of_XSsTypes_FSR
                // 8 -- Num_of_ScattOrder
                // 9 -- FSR_XS_File_List
                for (i = 0; i < 10; i++) {
                    JumpCommentsInFile(fsr_xs_file);
                    fscanf(fsr_xs_file, "%*s%*s");
                }

                for (i = 0; i < MOCNumLayersZ; i++) {
                    JumpCommentsInFile(fsr_xs_file);
                    fscanf(fsr_xs_file, "%s", layer_fsr_xs_file_path[i]);
                }

                fclose(fsr_xs_file);
                fsr_xs_file = NULL;

                // tally MacroXS_s_NNZ and alloc memory
                MacroXS_s_NNZ_Layer = AllocMem_ui64_I((size_t) MOCNumLayersZ, (char *) "MacroXS_s_NNZ_Layer");
                MacroXS_s_NZStartID_Layer = AllocMem_ui64_I((size_t) MOCNumLayersZ, (char *)"MacroXS_s_NZStartID_Layer");
                MacroXS_s_NZRowStartID_Layer = AllocMem_ui64_I((size_t) TotalNumFSRs_subDom*(NumScattOrder+1)*NumGroups+MOCNumLayersZ, (char *)"MacroXS_s_NZRowStartID_Layer");

                MacroXS_s_NNZ = 0;
                MacroXS_s_MaxNNZ_Layer = 0;
                for (_lay = 0; _lay < MOCNumLayersZ; _lay++) {
                    int _xsFmt = 0;   /* 0=text, 1=CSR binary, 2=dense binary */
                    printf("\n [  INFO   ]  Indexing FSR Scattering XS for layer %d/%d ...", _lay + 1, MOCNumLayersZ);
                    fflush(stdout);

                    nnz_layer = 0;
                    _xsFmt = IsBinaryXsFile(layer_fsr_xs_file_path[_lay]);
                    if (_xsFmt != 0) {
                        layer_fsr_xs_file = fopen(layer_fsr_xs_file_path[_lay], "rb");
                        InputXSs_forFSRs_binary(layer_fsr_xs_file, _lay, _xsFmt == 1, 0);
                    } else {
                        layer_fsr_xs_file = fopen(layer_fsr_xs_file_path[_lay], "r");
                        InputXSs_forFSRs(layer_fsr_xs_file, _lay);
                    }
                    fclose(layer_fsr_xs_file);
                    layer_fsr_xs_file = NULL;

                    for (_localFSRID =0; _localFSRID < TotalNumFSRs_perLayer; _localFSRID++) {
                        nnz_layer += InpXSs_s_NNZ[_localFSRID];
                    }
                    if (nnz_layer > MacroXS_s_MaxNNZ_Layer) {
                        MacroXS_s_MaxNNZ_Layer = nnz_layer;
                    }
                    MacroXS_s_NNZ += nnz_layer;

                    MacroXS_s_NNZ_Layer[_lay] = nnz_layer;
                    if (_lay == 0) {
                        MacroXS_s_NZStartID_Layer[_lay] = 0;
                    } else {
                        MacroXS_s_NZStartID_Layer[_lay] = MacroXS_s_NZStartID_Layer[_lay-1] + MacroXS_s_NNZ_Layer[_lay-1];
                    }


                    cnt = 0;
                    for (_localFSRID=0; _localFSRID < TotalNumFSRs_perLayer; _localFSRID++) {
                        _globFSRID = _lay * TotalNumFSRs_perLayer + _localFSRID;
                        for (_ord =0; _ord < NumScattOrder+1; _ord++) {
                            for (g=0; g < NumGroups; g++) {
                                _ind = _globFSRID * (NumScattOrder+1) * NumGroups + _ord * NumGroups + g + _lay;

                                if (_globFSRID == 0 && _ord == 0 && g == 0) {
                                    MacroXS_s_NZRowStartID_Layer[_ind] = 0;
                                } else {
                                    MacroXS_s_NZRowStartID_Layer[_ind] = cnt;
                                }

                                cnt += InpXSs_s_NZColIDEnd[_localFSRID][_ord * NumGroups + g] - InpXSs_s_NZColIDStart[_localFSRID][_ord * NumGroups + g];
                            }
                        }
                    }
                    MacroXS_s_NZRowStartID_Layer[_ind + 1] = cnt;

                }
                MacroXS_s = AllocMem_FP_I((size_t)(MacroXS_s_NNZ), (char *)("MacroXS_s"));

            }

            for (_lay = 0; _lay < MOCNumLayersZ; _lay++) {
                int _xsFmt = 0;   /* 0=text, 1=CSR binary, 2=dense binary */
                // read FSR XS
                if (Input_MATorXSs == 1 && INPUT_XSs_MATorFSR == 1 && IfSetXSsForEveryFSR == 1) {
                    printf("\n [  INFO   ]  Assigning FSR XS for layer %d/%d ...", _lay + 1, MOCNumLayersZ);
                    fflush(stdout);

                    _xsFmt = IsBinaryXsFile(layer_fsr_xs_file_path[_lay]);
                    if (_xsFmt != 0) {
                        layer_fsr_xs_file = fopen(layer_fsr_xs_file_path[_lay], "rb");
                        InputXSs_forFSRs_binary(layer_fsr_xs_file, _lay, _xsFmt == 1, 1);
                    } else {
                        layer_fsr_xs_file = fopen(layer_fsr_xs_file_path[_lay], "r");
                        InputXSs_forFSRs(layer_fsr_xs_file, _lay);
                    }

                    fclose(layer_fsr_xs_file);
                    layer_fsr_xs_file = NULL;
                }

                for (_localFSRID = 0; _localFSRID < TotalNumFSRs_perLayer; _localFSRID++) {
                    _globFSRID = _localFSRID + _lay * TotalNumFSRs_perLayer;
                    _mat_id = FSRsXSID_subDom[_globFSRID];

                    if (Input_MATorXSs == 1 && INPUT_XSs_MATorFSR == 1 && IfSetXSsForEveryFSR == 1) {
                        k1 = _localFSRID;  // assign XS layer by layer
                    } else {
                        for (k1 = 0; k1 < NumXSs_Mat; k1++) {
                            if (XSs_Mat_Id[k1] == _mat_id) {
                                break;
                            }
                        }
                    }

                    for (g = 0; g < NumGroups; g++) {
                        _ind = _globFSRID * NumGroups + g;
//					 printf("\n _globFSRID=%d, g=%d",_globFSRID,g);
                        MacroXS_t[_ind] = InpXSs_t[k1][g];
                        if (ProblemType == 0) {
                            MacroXS_a[_ind] = InpXSs_a[k1][g];
                            MacroXS_f[_ind] = InpXSs_f[k1][g];
                            MacroXS_nf[_ind] = InpXSs_nf[k1][g];
                            Chi[_ind] = InpChi[k1][g];

                            // if (_ind==0) {
                            // 	MacroXS_s_NZRowStartID[_ind] = 0;
                            // }
                            // else {
                            // 	MacroXS_s_NZRowStartID[_ind] = _count;
                            // }
                            // MacroXS_s_NZColStartID[_ind] = InpXSs_s_NZColIDStart[k1][g];
                            // for (g1 = InpXSs_s_NZColIDStart[k1][g]; g1<InpXSs_s_NZColIDEnd[k1][g]; g1++) {
                            // 	MacroXS_s[_count] = InpXSs_s[k1][g][g1];
                            // 	_count++;
                            // }
                        }
                    }

                    for (g = 0; g < _mul; g++) {
                        _ind = _globFSRID * _mul + g;
                        if (_ind == 0) {
                            MacroXS_s_NZRowStartID[_ind] = 0;
                        } else {
                            MacroXS_s_NZRowStartID[_ind] = _count;
                        }

                        MacroXS_s_NZColStartID[_ind] = InpXSs_s_NZColIDStart[k1][g];
                        for (g1 = InpXSs_s_NZColIDStart[k1][g]; g1 < InpXSs_s_NZColIDEnd[k1][g]; g1++) {
                            /* binary reader already wrote the band values into MacroXS_s */
                            if (_xsFmt == 0) {
                                if (INPUT_XSs_MATorFSR == 0 || (INPUT_XSs_MATorFSR == 1 && NumGroups != 2164))
                                {
                                    MacroXS_s[_count] = InpXSs_s[k1][g][g1];
                                }
                            }
                            _count++;
                        }
                    }

                }  // local fsr loop
            }  // lay loop
            MacroXS_s_NZRowStartID[TotalNumFSRs_subDom * (NumScattOrder + 1) * NumGroups] = _count;
        }


	if (InpXSs_s != NULL) {
		OutputMacroXSs_FSRs(ProcID, MOCNumLayersZ, NumPinsInSubDomPerLayer, NumFSRsInCell, SumNumFSRsBasedGlobCellID, \
			NumGroups, MacroXS_t, MacroXS_a, MacroXS_f, MacroXS_nf, MacroXS_s, Chi, InpXSs_t, InpXSs_a, InpXSs_f, \
			InpXSs_nf, InpXSs_s, InpChi, MacroXS_s_NZRowStartID, MacroXS_s_NZColStartID);
	}

    int num_mats;
    if (Input_MATorXSs == 1 && INPUT_XSs_MATorFSR == 1 && IfSetXSsForEveryFSR == 1) {
        num_mats = TotalNumFSRs_perLayer;
    } else {
        num_mats = NumXSs_Mat;
    }

	XSs_Mat_Id = FreeMem_i32_I(XSs_Mat_Id, (char *)("XSs_Mat_Id"));
	InpXSs_t = FreeMem_FP_II(InpXSs_t, num_mats, (char *)("InpXSs_t"));
	//eigenvalue problem
	if (ProblemType == 0) {
		InpXSs_a = FreeMem_FP_II(InpXSs_a, num_mats, (char *)("InpXSs_a"));
		InpXSs_f = FreeMem_FP_II(InpXSs_f, num_mats, (char *)("InpXSs_f"));
		InpXSs_nf = FreeMem_FP_II(InpXSs_nf, num_mats, (char *)("InpXSs_nf"));
		InpChi = FreeMem_FP_II(InpChi, num_mats, (char *)("InpChi"));
        if (INPUT_XSs_MATorFSR == 0)
        {
            InpXSs_s = FreeMem_FP_III(InpXSs_s, num_mats, NumGroups, (char *)("InpXSs_s"));
        }
        if (INPUT_XSs_MATorFSR == 1)
        {
            if (NumGroups == 2164)
            {
                TempXSs_s = FreeMem_FP_III(TempXSs_s, NumScattOrder + 1, NumGroups, (char *)("TempXSs_s"));
                TempXSs_T = FreeMem_FP64_I(TempXSs_T, (char *)("Temp_XS_Tot_Nuf_Chi"));
                TempIndex = FreeMem_int_I(TempIndex, (char *)("TempIndex"));
            }
            else
            {
                /* binary reader never allocates the dense InpXSs_s */
                if (InpXSs_s != NULL) {
                    InpXSs_s = FreeMem_FP_III(InpXSs_s, num_mats, NumGroups, (char *)("InpXSs_s"));
                }
            }
        }
        //TempXSs_s = FreeMem_FP_III(TempXSs_s, NumScattOrder + 1, NumGroups, (char *)("TempXSs_s"));

		InpXSs_s_NZColIDStart = FreeMem_int_II(InpXSs_s_NZColIDStart, num_mats, (char *)("InpXSs_s_NZColIDStart"));
		InpXSs_s_NZColIDEnd = FreeMem_int_II(InpXSs_s_NZColIDEnd, num_mats, (char *)("InpXSs_s_NZColIDEnd"));

		if (TransCorre == 1) {
			InpXSs_p0 = FreeMem_FP_III(InpXSs_p0, num_mats, NumGroups, (char *)("InpXSs_p0"));
			InpXSs_p1 = FreeMem_FP_III(InpXSs_p1, num_mats, NumGroups, (char *)("InpXSs_p1"));
		}
		else {}
	}
	else if (ProblemType == 1) {
		Inp_FixedSour = FreeMem_FP_II(Inp_FixedSour, num_mats, (char *)("Inp_FixedSour"));
	}
#ifdef TIME_REC
	_time_end = TallyCurrentTime_CPU();
	TimePrePart[TimeCountPrePart] = _time_end - _time_start;
	strcpy(TimeMetricsPrePart[TimeCountPrePart], "Set XSs Info");
	TimeCountPrePart++;
#endif // TIME_REC
}


FP_PRECISION GetFSRXSs(int _lay, int _cell, int _reg,int _numCellsPerlay, int _numGroups,int _g, \
    FP_PRECISION *_macroXS, FP_PRECISION **_inpXS,int *_sumNumFSRsBasedGlobCellID)
{
	int i = 0;
	FP_PRECISION _XS = 0.0;
	if (Input_MATorXSs == 1 && IfSetXSsForEveryFSR == 1) {
		if (_macroXS!=NULL) {
			_XS = _macroXS[(_sumNumFSRsBasedGlobCellID[_lay*_numCellsPerlay+_cell]+_reg)*_numGroups+_g];
		}
		else {
			_XS = 0.0;
		}
	}
	else if (Input_MATorXSs == 1 && IfSetXSsForEveryFSR == 0) {
		if (_inpXS!=NULL) {
			for (i = 0; i < NumXSs_Mat; i++) {
			    if (XSs_Mat_Id[i] == XSsID_FSRs[_lay][_cell][_reg]) {
				    break;
			    }
		    }
		    _XS = _inpXS[i][_g];
		}
		else {
			_XS = 0.0;
		}

	}
	else if(IfResonance == 1 || Input_MATorXSs == 100)
	{
		if (_macroXS != NULL) {
			_XS = _macroXS[(_sumNumFSRsBasedGlobCellID[_lay*_numCellsPerlay + _cell] + _reg)*_numGroups + _g];
		}
		else {
			_XS = 0.0;
		}
	}
	return _XS;
}


// generate matlab matrix for MATLAB verification - Infinite medium problem
// get the group boundary limit with up-scattering
void FissionSpectrumCheck(int _num_groups,int _globFRSId, FP_PRECISION *_chi)
{
	FP_PRECISION _sum = 0.0;//,_res
	int g=0,g1=0,g_zore=0;
	for (g = 0; g < _num_groups; g++) {
		_sum += _chi[_globFRSId*_num_groups+g];
		if (_chi[_globFRSId*_num_groups+g] > 0.0) {
			g_zore = g;
		}
		if (_sum > 1.0) {
			_chi[_globFRSId*_num_groups+g] = _chi[_globFRSId*_num_groups+g] - (_sum - (FP_PRECISION)(1.0));
			break;
		}
	}
	if (_sum > 1.0) {
		for (g1 = g + 1; g1 < _num_groups; g1++) {
			_chi[_globFRSId*_num_groups+g1] = 0.0;
		}
	}
	else {
		_chi[_globFRSId*_num_groups+g_zore] = _chi[_globFRSId*_num_groups+g_zore] + ((FP_PRECISION)(1.0) - _sum);
	}
}

void GetFluxVolWeightedXSs_Mat_1D(int _num_mat, int _NumGroups, int _TotalNumFSRs_subDom, \
    int *_FSRsXSID_subDom, FP_PRECISION_GEO *_FSRVol_SubDom, FP_PRECISION *_ScalarFlux_MOC,\
	FP_PRECISION *_sumReatRate, FP_PRECISION *_sumFlux, FP_PRECISION *_XSs_1D, FP_PRECISION *_XSs_1D_Mat)
{
	int g, i, _mat, _ind, _ind1;
	for (i=0;i<_num_mat*_NumGroups;i++) {
		_sumReatRate[i]=0.0;
		_sumFlux[i]=0.0;
	}

	for (i=0;i<_TotalNumFSRs_subDom;i++) {
		_mat = _FSRsXSID_subDom[i];
		for (g=0; g<_NumGroups; g++) {
			_ind=_mat*_NumGroups+g;
			_ind1=i*_NumGroups+g;
			_sumFlux[_ind] += _ScalarFlux_MOC[_ind1] * _FSRVol_SubDom[i];
			_sumReatRate[_ind] += _XSs_1D[_ind1] * _ScalarFlux_MOC[_ind1] * _FSRVol_SubDom[i];
		}
	}

	for (i=0;i<_num_mat*_NumGroups;i++) {
		_XSs_1D_Mat[i]=_sumReatRate[i]/_sumFlux[i];
	}
}


void GetFluxVolWeightedXSs_Mat_2D(int _num_mat, int _NumGroups, int _TotalNumFSRs_subDom, \
    int *_FSRsXSID_subDom, FP_PRECISION_GEO *_FSRVol_SubDom, FP_PRECISION *_ScalarFlux_MOC,\
	FP_PRECISION *_sumReatRate, FP_PRECISION *_sumFlux, FP_PRECISION *_XSs_2D, FP_PRECISION *_XSs_2D_Mat)
{
	int g, i,g1, _mat, _ind, _ind1, _ind2;
	for (i=0;i<_num_mat*_NumGroups*_NumGroups;i++) {
		_sumReatRate[i]=0.0;
		_sumFlux[i]=0.0;
	}

	for (i=0;i<_TotalNumFSRs_subDom;i++) {
		_mat = _FSRsXSID_subDom[i];
		for (g=0; g<_NumGroups; g++) {
			_ind1=i*_NumGroups+g;
			for (g1=0; g1<_NumGroups; g1++) {
				_ind=_mat*_NumGroups*_NumGroups+g*_NumGroups+g1;
				_ind2=i*_NumGroups*_NumGroups+g*_NumGroups+g1;
				_sumFlux[_ind] += _ScalarFlux_MOC[_ind1] * _FSRVol_SubDom[i];
				_sumReatRate[_ind] += _XSs_2D[_ind2] * _ScalarFlux_MOC[_ind1] * _FSRVol_SubDom[i];
			}
		}
	}

	for (i=0;i<_num_mat*_NumGroups*_NumGroups;i++) {
		_XSs_2D_Mat[i]=_sumReatRate[i]/_sumFlux[i];
	}
}


void CondenseMacroXSsForMat_FromMOCSolution(M_INT32 _isInvoke, M_INT32 _IfResonance, M_INT32 _num_mat, M_INT32 _NumGroups, M_INT32 _TotalNumFSRs_subDom, \
    M_INT32 *_FSRsXSID_subDom, FP_PRECISION_GEO *_FSRVol_SubDom, FP_PRECISION *_ScalarFlux_MOC)
{
	M_INT32 g, g1,i, _ind, _ind1;
	FP_PRECISION *_sumReatRate=NULL, *_sumFlux=NULL;
	char _fileName[225];
	FILE *_op = NULL;
	M_INT32 _width=16;

	if (_isInvoke==1 && _IfResonance==1) {
	    MacroXS_t_Mat = AllocMem_FP_I_1(_num_mat*_NumGroups, (char *)("MacroXS_t_Mat"));
		MacroXS_a_Mat = AllocMem_FP_I_1(_num_mat*_NumGroups, (char *)("MacroXS_a_Mat"));
		MacroXS_f_Mat = AllocMem_FP_I_1(_num_mat*_NumGroups, (char *)("MacroXS_f_Mat"));
		MacroXS_nf_Mat = AllocMem_FP_I_1(_num_mat*_NumGroups, (char *)("MacroXS_nf_Mat"));
		Chi_Mat = AllocMem_FP_I_1(_num_mat*_NumGroups, (char *)("Chi_Mat"));
		MacroXS_s_Mat = AllocMem_FP_I_1(_num_mat*_NumGroups*_NumGroups, (char *)("MacroXS_s_Mat"));

		_sumReatRate = AllocMem_FP_I_1(_num_mat*_NumGroups, (char *)("\0"));
		_sumFlux = AllocMem_FP_I_1(_num_mat*_NumGroups, (char *)("\0"));

		GetFluxVolWeightedXSs_Mat_1D(_num_mat, _NumGroups, _TotalNumFSRs_subDom, \
            _FSRsXSID_subDom, _FSRVol_SubDom, _ScalarFlux_MOC,_sumReatRate, _sumFlux, \
			MacroXS_t, MacroXS_t_Mat);

		GetFluxVolWeightedXSs_Mat_1D(_num_mat, _NumGroups, _TotalNumFSRs_subDom, \
            _FSRsXSID_subDom, _FSRVol_SubDom, _ScalarFlux_MOC,_sumReatRate, _sumFlux, \
			MacroXS_a, MacroXS_a_Mat);

		GetFluxVolWeightedXSs_Mat_1D(_num_mat, _NumGroups, _TotalNumFSRs_subDom, \
            _FSRsXSID_subDom, _FSRVol_SubDom, _ScalarFlux_MOC,_sumReatRate, _sumFlux, \
			MacroXS_f, MacroXS_f_Mat);

		GetFluxVolWeightedXSs_Mat_1D(_num_mat, _NumGroups, _TotalNumFSRs_subDom, \
            _FSRsXSID_subDom, _FSRVol_SubDom, _ScalarFlux_MOC,_sumReatRate, _sumFlux, \
			MacroXS_nf, MacroXS_nf_Mat);

		GetFluxVolWeightedXSs_Mat_1D(_num_mat, _NumGroups, _TotalNumFSRs_subDom, \
            _FSRsXSID_subDom, _FSRVol_SubDom, _ScalarFlux_MOC,_sumReatRate, _sumFlux, \
			Chi, Chi_Mat);

		_sumReatRate = FreeMem_FP_I(_sumReatRate, (char *)("\0"));
		_sumFlux = FreeMem_FP_I(_sumFlux, (char *)("\0"));

		_sumReatRate = AllocMem_FP_I_1(_num_mat*_NumGroups*_NumGroups, (char *)("\0"));
		_sumFlux = AllocMem_FP_I_1(_num_mat*_NumGroups*_NumGroups, (char *)("\0"));

/* this fucntion should be re-organized for compressed sparce row (CSR) format storage*/
		GetFluxVolWeightedXSs_Mat_2D(_num_mat, _NumGroups, _TotalNumFSRs_subDom, \
            _FSRsXSID_subDom, _FSRVol_SubDom, _ScalarFlux_MOC,_sumReatRate, _sumFlux, \
			MacroXS_s, MacroXS_s_Mat);

		_sumReatRate = FreeMem_FP_I(_sumReatRate, (char *)("\0"));
		_sumFlux = FreeMem_FP_I(_sumFlux, (char *)("\0"));

        sprintf(_fileName, "%s-%s", CaseId, "MatMacroXSsFromResCacl.inp");
		_op = fopen(_fileName, "w");
		if (_op == NULL) {
			printf("%s file opens error!\n", _fileName);
			exit(1);
		}
		fprintf(_op, "\n!  *****************************************************************");
		fprintf(_op, "\n!  *                                                               *");
		fprintf(_op, "\n!  *                 MacroXSs for Mats From Res Calc               *");
		fprintf(_op, "\n!  *                                                               *");
		fprintf(_op, "\n!  *****************************************************************\n");

		fprintf(_op, "\n  INPUT_MATorXSs       1");
		fprintf(_op, "\n  INPUT_XSs_MATorFSR   0");
		fprintf(_op, "\n  Num_Group            %d",_NumGroups);
		fprintf(_op, "\n  IfSetXSsForEveryFSR  1");
		fprintf(_op, "\n  reCalcTotalXS        0");
		fprintf(_op, "\n  reCalcAbsorbXS       0");
		fprintf(_op, "\n  transportCorrection  0");
		fprintf(_op, "\n  Num_of_XSsTypes_Mat  %d",_num_mat);

		for (i=0;i<_num_mat;i++) {
			fprintf(_op,"\n\n!");
			fprintf(_op,"%*s%*s%*s%*s%*s%*s",7,"MatID",8,"GroupID",_width,"Sigma_t",_width,"Sigma_a",_width,"Sigma_f",_width,"Sigma_nf");
			for (g=0;g<_NumGroups;g++) {
				fprintf(_op,"  Sigma_s_g-%*d",_width-12,g);
			}
			fprintf(_op,"%*s",_width,"Chi");

			fprintf(_op,"\n%*d",8,i);
			for (g=0;g<_NumGroups;g++) {
				_ind=i*_NumGroups+g;
				if (g==0) {
					fprintf(_op,"%*d%*.8e%*.8e%*.8e%*.8e",8,g,_width,MacroXS_t_Mat[_ind],_width,MacroXS_a_Mat[_ind],_width,MacroXS_f_Mat[_ind],_width,MacroXS_nf_Mat[_ind]);
				}
				else {
					fprintf(_op,"\n%*d%*.8e%*.8e%*.8e%*.8e",8+8,g,_width,MacroXS_t_Mat[_ind],_width,MacroXS_a_Mat[_ind],_width,MacroXS_f_Mat[_ind],_width,MacroXS_nf_Mat[_ind]);
				}
				for (g1=0;g1<_NumGroups;g1++) {
					_ind1=i*_NumGroups*_NumGroups+g*_NumGroups+g1;
					fprintf(_op,"%*.8e",_width, MacroXS_s_Mat[_ind1]);
				}
				fprintf(_op,"%*.8e",_width, Chi_Mat[_ind]);
			}
		}
		fclose(_op);
		_op=NULL;

		MacroXS_t_Mat = FreeMem_FP_I(MacroXS_t_Mat, (char *)("MacroXS_t_Mat"));
		MacroXS_a_Mat = FreeMem_FP_I(MacroXS_a_Mat, (char *)("MacroXS_a_Mat"));
		MacroXS_f_Mat = FreeMem_FP_I(MacroXS_f_Mat, (char *)("MacroXS_f_Mat"));
		MacroXS_nf_Mat = FreeMem_FP_I(MacroXS_nf_Mat, (char *)("MacroXS_nf_Mat"));
		Chi_Mat = FreeMem_FP_I(Chi_Mat, (char *)("Chi_Mat"));
		MacroXS_s_Mat = FreeMem_FP_I(MacroXS_s_Mat, (char *)("MacroXS_s_Mat"));
		_sumReatRate = FreeMem_FP_I(_sumReatRate, (char *)("\0"));
		_sumFlux = FreeMem_FP_I(_sumFlux, (char *)("\0"));
	}
}
