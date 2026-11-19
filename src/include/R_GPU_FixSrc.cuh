#ifndef R_GPU_FIXSRC_H
#define R_GPU_FIXSRC_H

#include "MOC_CommonMacro.h"

// extern FP_PRECISION *GPU_Temperature, *GPU_System_Temperature = NULL, *GPU_Flux_Sldw = NULL, *GPU_Sys_Mic_scatt_mtx = NULL, *GPU_Nuc_Den = NULL; 
// extern int *GPU_Nuc_Num, *GPU_Nuc_Sum_byGlob_FsrID = NULL, *GPU_Sys_Nuc_Order=NULL,*GPU_Nuclide_ID=NULL,*GPU_Sys_Mic_Inscatt_Col_Start=NULL;

void GPU_Res_Fix_Prepare(M_INT32 _TotalNumFSRs_subDom, M_INT32 _TotalNumFSRs_perLayer, M_INT32 _nuc_num, M_INT32 _Sys_Temp_Num, M_INT32 _NOG_SHEM, M_INT32 _NORG_SHEM, \
	M_INT32 _FsrNuc_sum, FP_PRECISION *_Temperature, FP_PRECISION *_System_Temperature, FP_PRECISION *_Flux_Sldw, FP_PRECISION *_Sys_Mic_scatt_mtx, \
	M_INT32 *_Nuc_Num, M_INT32 *_Nuc_Sum_byGlob_FsrID, M_INT32 *_Sys_Nuc_Order, M_INT32 *_Nuclide_ID, FP_PRECISION *_Nuc_Den, M_INT32 *_Sys_Mic_Inscatt_Col_Start);
void GPUCopySldwFluxsHostToDevice_Layer(M_INT32 _TotalNumFSRs_subDom,int _layID, M_INT32 _NORG_SHEM, FP_PRECISION *_Flux_Sldw);

void GPU_Unpdate_SLDW_Fixed_Src_MG_Layer(int _layID, int _groupLowLim, int _groupUpLim, int _current_MG, M_INT32 _Sys_Temp_Num, \
	M_INT32 _NORG_SHEM, M_INT32 _NOG_SHEM, M_INT32 _NOFG_SHEM);


void update_homo_flux(int numGroups, FP_PRECISION *Qs, FP_PRECISION *Qf, FP_PRECISION *invSigmaT,
                      FP_PRECISION_FLUX *homoFlux, FP_PRECISION keff);

#endif //R_GPU_FIXSRC_H