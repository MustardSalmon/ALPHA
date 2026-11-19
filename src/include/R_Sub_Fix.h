
#ifndef R_SUB_FIX_H
#define R_SUB_FIX_H

#ifdef CUDA_USED
#include "cublas_v2.h"
#include <cuda_runtime.h>
#endif

void Cal_Fix_Src_SHEM_1MG(int lay, int nucID);
void Sub_Fix_Src_Moc_MG(int _procID);

void Sub_Fix_Src_Moc(int _procID);
void Cal_Fix_Src_SHEM_1G(int lay, int nucID);
void Cal_Fix_Src_SHEM(int lay, int nucID);

void Cal_Fix_Src_SubUFG_1G(int lay, int nucID);
void Cal_Fix_Src_SubUFG_1G_Pesudo(int lay, int matID);
void Cal_Fix_Src_SubUFG_MG_Pesudo(int lay, int matID);
void Cal_Fix_Src_SubUFG(int lay, int nucID);
void Cal_Fix_Src_SubUFG_Tone(int lay, int nucID, int itime);
void Cal_Fix_Src_SubUFG_He(int lay, int nucID, int resID);

void InterSubGrpFlx(int lay, int nucID);
void InterSubGrpFlxPesudo(int lay, int imat);
void InterSubGrpFlxPesudoMG(int lay, int imat);

void Cal_Xs_SubUFG(int lay, int nucID, int resID);
void OutResXs(int lay, int ifOutPut);
void Bonderake(int lay);

void Cal_PartMacroXs_SubUFG(int lay);
void Cal_Scatt_Matrix_Index_SubUFG(int _procID);
void Cal_Scatt_Matrix_SubUFG(int _procID);
void Cal_Chi_SubUFG(int _procID);
void Cal_Chi_SubUFG_batched(int _procID);

void Cal_Xs_SubUFG_Tone(int lay, int nucID, int resID);

void Cal_Fix_Src_SHEM_1G_FSR(int lay, int cell, int fsr, int nucID);
void Cal_Fix_Src_SHEM_FSR(int lay, int cell, int fsr, int nucID);

void Cal_Fix_Src_SHEM_FSR_Group(int lay, int cell, int fsr, int nucID, int group, double _lamda_xsp, int swcat);
void Cal_Fix_Src_SHEM_1G_FSR_GG(int lay, int cell, int fsr, int nucID);
void Cal_Fix_Src_SHEM_FSR_Group_OPT(int lay, int cell, int fsr, int nucID, int ci, int group, double _lamda_xsp, int swcat);

extern int NUC_ID_DEBUG;

#define N 32  // batchCount for Chi calculation in 2164g

#ifdef CUDA_USED
#ifdef FP64
#define cublas_gemv_batched cublasDgemvBatched
#define cublas_gemm_batched cublasDgemmBatched
#define cublas_vsum cublasDasum
#else
#define cublas_gemv_batched cublasSgemvBatched
#define cublas_gemm_batched cublasSgemmBatched
#define cublas_vsum cublasSasum
#endif
#endif

#endif // 
