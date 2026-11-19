#ifndef R_XS_TREAT_H
#define R_XS_TREAT_H

#include "MOC_CommonMacro.h"
#include "R_Material.h"

extern int Mat_Number, Com_Mat_Number, IF_USE_FINE_MESH, IF_PART_READ, Iter_Limit; extern int FINE_NOG, FINE_NORG, FEW_NOG, FEW_NORG, IF_GPU_ALLOC;
extern struct R_MAT *R_Material; extern  struct FLUX_INFO *Flux_Infom; extern struct RES_INFO Res_Infom; extern struct NUC_INFO Nuc_Infom;extern struct RES_INFO *Res_Data_3D_Global;extern struct RES_INFO **Res_Data_3D_Local;
extern int *MatID_FSR_SubDom, *MatID_FSR_MOC, *MatID_FSR_MOC_DH, Current_SLDW_Iter_MG, IF_SLDW_MG;;
extern FP_PRECISION *Nuc_RImax, *Mic_Self_Scat, *Temperature, *Nuc_Den, **Fix_Src_1MG, **Fix_Rm_1MG;; extern int *Nuc_Sum_byGlob_FsrID, *Res_Sum_byGlob_FsrID;
extern int *Nuc_Num, *Res_Num, *If_Res, *Nuclide_ID, FsrNuc_sum;
extern int **res_nuc_num_layer_pro, ***res_nuc_id_layer_pro, **res_cate_num_layer_pro, ***res_cate_id_layer_pro, ***res_cate_nuc_num_layer_pro, ****res_cate_nuc_id_layer_pro;

extern FP_PRECISION *System_Temperature; extern int *Eid_Sys_Temp, Sys_Temp_Num;
extern FP_PRECISION ***Sys_Mic_a, ***Sys_Mic_f, ***Sys_Mic_nf, ***Sys_Mic_t, ***Sys_Mic_s, ***Sys_Mic_s_in, ****Sys_Mic_s_mx_p1, *Sys_Mic_scatt_mtx;
extern int  *Sys_Mic_Inscatt_Col_Start, *Sys_Mic_Inscatt_Col_End;

extern FP_PRECISION **imatrix1, **imatrix2, **mic_scat_mx, *mic_self_scat, **mic_scat_mx_p1, *nuclide_xs_a, *nuclide_xs_f, *nuclide_xs_t, *nuclide_xs_nf, *nuclide_xs_s, *nuclide_xs_n2n, *nuclide_xs_n3n;
extern FP_PRECISION *Lamda_XSP, *XS_POT, **sub_mic_xs_a, **sub_mic_xs_nf, **sub_mic_wei_a, **sub_mic_wei_nf, **sub_mic_xs_s, **sub_mic_xs_t, **sub_mic_wei;
extern FP_PRECISION *Fix_Src_1G, *Fix_Rm_1G, *Fix_Src_SHEM, **Fix_Src_SHEM2, **Fix_Rm_SHEM, **Fix_Flux_1G, ***Fix_Flux_SHEM, **Fix_Rm_1GG, **Fix_Src_1GG;
extern FP_PRECISION ***Fix_Flux_SubUFG,*Fix_Src_SubUFG,***Fix_Rm_SubUFG, ***Xs_SubUFG; // for subgroup UFG
extern FP_PRECISION ****Xss_SubUFG, ****SubUFGFlx; // for background iteration
//extern FP_PRECISION ***SigmaX; // for normal background iteration
extern FP_PRECISION **tempXSs; // for temp reaction xs
extern FP_PRECISION ***Fix_Flux_SubUFG_Tone1, ***Fix_Flux_SubUFG_Tone2; // for subgroup UFG Tone
extern FP_PRECISION ***sub_mic_xs_b, *iter_a, *iter_f, *iter_x, ***msub_mac_xs_ei;
extern FP_PRECISION *Mat_Den;
extern int **Mat_IfSub;
extern FP_PRECISION ***Fix_Rm_MG, ***Fix_Flux_MG, **Mat_MinSub;
extern FP_PRECISION *Res_Mic_a, *Res_Mic_nf, *Res_Mic_s, *Res_Mic_t, *Mic_a_NORG, *Mic_nf_NORG, *Mic_s_NORG, *****Micro_Res_nf_PSO_new, *****Micro_Res_a_PSO_new, *Mic_t_NORG, /***Micro_Res_a_PSO, **Micro_Res_nf_PSO,**Micro_Res_f_PSO, **Micro_Res_s_PSO, **Micro_Res_t_PSO,*/ *Mic_nf_NORG; extern int  *Ing_Res_Mic, *Ing_Mic_NORG;
extern FP_PRECISION *Flux_Sldw, *Mac_a_SHEM, *Mac_f_SHEM, *Mac_nf_SHEM, *Mac_s_SHEM, *Mac_t_SHEM;

extern int Max_Triso_Num, Max_Triso_Laynum, Com_Mat_Number;  extern struct DH_INFO DH_Infom; extern struct R_COM *R_Composite;
extern FP_PRECISION ***PIJ_DH, *Temperature_DH, *Nuc_Den_DH, *Triso_Vol_Fra, *Triso_Radi, *Triso_Lay_Vol_Fra;
extern int *Triso_Num, *Triso_Lay_Num, *Triso_ID, *Nuclide_ID_DH, *Nuc_Num_DH, *Res_Num_DH;
extern int *Triso_Sum_byGlob_FsrID, *TLay_Sum_byGlob_TrisoID, *TNuc_Sum_byGlob_TriLayID, *TRes_Sum_byGlob_TriLayID;
extern int TrisoLay_TotalNum_subDom;
extern FP_PRECISION *Fix_Src_1G_DH, *Fix_Rm_1G_DH, *Fix_Src_SHEM_DH, **Fix_Rm_SHEM_DH, **Fix_Flux_DH,  ***Fix_Flux_SHEM_DH;
extern FP_PRECISION *Fix_Src_1G_MTX, *Fix_Rm_1G_MTX, *Fix_Src_SHEM_MTX, **Fix_Rm_SHEM_MTX;
extern FP_PRECISION *Flux_Asym, *Gamma_c, **Flux_Asym2, **Gamma_c2, *Escape_Reduced, **Escape_Reduced2, **Flux_ik, *Flux_Asym_MOC, *Gamma_c_MOC, *Escape_Reduced_MOC, *Escape_MOC, ***PIJ_MOC;
extern int Current_Fix_G, Fix_Total_G, IF_INITIAL;
extern FP_PRECISION *Res_Mic_a_DH, *Res_Mic_nf_DH, *Res_Mic_s_DH, *Res_Mic_t_DH, *Mic_a_NORG_DH, *Mic_nf_NORG_DH, *Mic_s_NORG_DH, *Mic_t_NORG_DH;
extern FP_PRECISION *Flux_Sldw_DH, *ScalarFlux_DH, *Source_DH, *Source_MTX, *MacroXS_t_DH, *MacroXS_a_DH, *MacroXS_f_DH, *MacroXS_nf_DH, *MacroXS_s_DH, *Chi_DH;
extern FP_PRECISION *MacroXS_t_MTX, *MacroXS_a_MTX, *MacroXS_f_MTX, *MacroXS_nf_MTX, *MacroXS_s_MTX, *Chi_MTX;

extern FP_PRECISION **Rm_New, **Rm_Old, **Rm_0, **PIJ_XS, *Temporary_1FP, **Temporary_ResXS;
extern FP_PRECISION Mac_a_Group;
extern FP_PRECISION *Micro_a_1G_burn, *Micro_f_1G_burn, *Micro_a_1G_burn_DH, *Micro_f_1G_burn_DH;
extern int *MacroXS_s_NZColEnddID;

extern int UFG_NUM;
extern FP_PRECISION Res_Time_Total, Lib_time, SGFSP_time, SLDW_time, SGFSP_AND_SDLW_time;
extern int *EIDD, *Sys_Nuc_Order;
extern int *MacroXS_s_NZ_DownColStart, *MacroXS_s_NZ_Down_GlobalPos, MacroXS_s_NNZ_Down_Total, *MacroXS_s_NNZ_Down_Lay; extern FP_PRECISION *MacroXS_s_NZ_DownSca_XS;

// for burnup MG xs
extern FP_PRECISION ***AbsMG, ***FisMG;

;;;

void cal_afst_SHEM(int nucID, FP_PRECISION tem);
void Cal_SysTem_Mic_XS_MC(int _procID);

void cal_aft_SHEM_Group(int nucID, int group, FP_PRECISION tem);


void cal_afst_SHEM_MC(int nucID, FP_PRECISION tem);
void cal_scatt_SHEM_MC(int nucID, FP_PRECISION tem, int group);

;;;;;;;;
#endif
