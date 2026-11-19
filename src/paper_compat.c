/*
 * Compatibility boundary for the publication transport snapshot.
 *
 * The upstream executable contains optional resonance/slow-down branches in
 * the same translation units as the direct transport solver.  The paper
 * snapshot deliberately does not ship those solvers.  These no-op symbols
 * satisfy the optional link-time interfaces; they are unreachable when the
 * input card sets If_Resonance=0, which is the only supported mode here.
 */
#include "zres_common_func.h"
#include "R_xs_treat.h"
#include "R_I_O_File.h"
#include "R_DH_Mat_XS.h"
#include "R_GPU_FixSrc.cuh"
#include "MOC_CMFD.h"

/*
 * The transport-only input path does not initialize any resonance or CMFD
 * state.  The upstream executable nevertheless keeps optional branches in
 * the same translation units, so the linker still needs their interface
 * symbols.  These definitions deliberately contain no resonance algorithm;
 * they only make the unsupported branches inert.
 */
int Mat_Number = 0, Com_Mat_Number = 0, IF_USE_FINE_MESH = 0;
int IF_PART_READ = 0, Iter_Limit = 0, FINE_NOG = 0, FINE_NORG = 0;
int FEW_NOG = 0, FEW_NORG = 0, IF_GPU_ALLOC = 0;
/*
 * The reproducibility snapshot starts directly from an external FSR
 * macroscopic-XS file.  Input_Type=100 belongs to the omitted resonance/
 * double-heterogeneity path and makes the ordinary transport initialization
 * dereference unavailable Triso_* arrays.  Keep the direct transport
 * default here; Input_MATorXSs is read from FsrIdx.inp.
 */
int Input_Type = 0, Scatt_Type = 0, Res_Type = 0, Inter_Type = 0;
int Lib_Type = 0, TransCorre_Type = 0, IF_SET_MIN = 0, SGFSP_Type = 0;
int Sldw_Type = 0, IF_READ_LIB_INDEX = 0;
int NOFG_SHEM = 0, NORG_SHEM = 0, NOG_SHEM = 0;
int Sys_Temp_Num = 0, Current_SLDW_Iter_MG = 0;
int IF_SLDW_MG = 0, FsrNuc_sum = 0;

FP_PRECISION Res_Time_Total = 0, Lib_time = 0, SGFSP_time = 0;
FP_PRECISION SLDW_time = 0, SGFSP_AND_SDLW_time = 0;

struct R_MAT *R_Material = NULL;
struct FLUX_INFO *Flux_Infom = NULL;
struct RES_INFO Res_Infom = {0};
struct NUC_INFO Nuc_Infom = {0};
struct RES_INFO *Res_Data_3D_Global = NULL;
struct RES_INFO **Res_Data_3D_Local = NULL;
struct DH_INFO DH_Infom = {0};
struct R_COM *R_Composite = NULL;

int *MatID_FSR_SubDom = NULL, *MatID_FSR_MOC = NULL;
int *MatID_FSR_MOC_DH = NULL, *Nuc_Num = NULL, *Res_Num = NULL;
int *If_Res = NULL, *Nuclide_ID = NULL, *Nuc_Sum_byGlob_FsrID = NULL;
int *Res_Sum_byGlob_FsrID = NULL, *EIDD = NULL, *Sys_Nuc_Order = NULL;
int *Sys_Mic_Inscatt_Col_Start = NULL, *Sys_Mic_Inscatt_Col_End = NULL;
int *MacroXS_s_NZColEnddID = NULL;

FP_PRECISION *Nuc_RImax = NULL, *Mic_Self_Scat = NULL;
FP_PRECISION *Temperature = NULL, *Nuc_Den = NULL;
FP_PRECISION *System_Temperature = NULL, *Flux_Sldw = NULL;
FP_PRECISION *Sys_Mic_scatt_mtx = NULL;
FP_PRECISION *MacroXS_nf_DH = NULL, *ScalarFlux_DH = NULL;
FP_PRECISION *MacroXS_nf_MTX = NULL;
FP_PRECISION *Triso_Vol_Fra = NULL, *Triso_Lay_Vol_Fra = NULL;
int *Triso_Num = NULL, *Triso_Lay_Num = NULL;
int *Triso_Sum_byGlob_FsrID = NULL, *TLay_Sum_byGlob_TrisoID = NULL;

FP_PRECISION *HomoTotCG = NULL, *HomoNufCG = NULL, *HomoChiCG = NULL;
FP_PRECISION **HomoScatterCG = NULL;

/* Optional microscopic data passed only to the omitted slow-down branch. */
FP_PRECISION ***Sys_Mic_a = NULL, ***Sys_Mic_f = NULL;
FP_PRECISION ***Sys_Mic_nf = NULL, ***Sys_Mic_t = NULL;
FP_PRECISION ***Sys_Mic_s = NULL, ***Sys_Mic_s_in = NULL;
FP_PRECISION ****Sys_Mic_s_mx_p1 = NULL;

FP_PRECISION *CMFD_TranSourCoeff_A = NULL;
FP_PRECISION *CMFD_TranSourCoeff_B = NULL;
FP_PRECISION *CMFD_TranSourCoeff_C = NULL;
FP_PRECISION *CMFD_FissSour = NULL;
FP_PRECISION *tranCMFD_TotalSour = NULL;

/* CMFD parameters are parsed for input compatibility but are disabled by
 * the supported transport case. */
int GmresKrylovDim_CG = 2, GmresPrecond_CG = 1;
int GmresKrylovDim_MG = 2, GmresPrecond_MG = 1;
int CGCMFD_LinearSolver = 3, MGCMFD_LinearSolver = 3;
int MGCMFDData_MPIComm_Num = 0, CGCMFDData_MPIComm_Num = 0;
int MGCMFD_MOCParCurrInnerBound_Num = 0;
int *CGCMFDData_MPIComm_SumInd = NULL, *CGCMFDData_MPIComm_Index = NULL;
int *MGCMFDData_MPIComm_SumInd = NULL, *MGCMFDData_MPIComm_Index = NULL;
int *CGCMFDMPI_InBoundSourCoeff_Dir = NULL;
int *CGCMFDMPI_InBoundSourCoeff_Index = NULL;
int *MGCMFDMPI_InBoundSourCoeff_Dir = NULL;
int *MGCMFDMPI_InBoundSourCoeff_Index = NULL;
int *MGCMFD_MOCParCurrInnerBound_Dir = NULL;
int *MGCMFD_MOCParCurrInnerBound_Index = NULL;
int *CMFDComm_TarProc = NULL;
FP_PRECISION *CGCMFDMPI_InBoundSourCoeff_GeoDim = NULL;
FP_PRECISION *MGCMFDMPI_InBoundSourCoeff_GeoDim = NULL;
FP_PRECISION *CMFD_NodeVol = NULL;
FP_PRECISION MGCMFD_epseig = 1.0e-5f, MGCMFD_epsl2 = 1.0e-5f;
FP_PRECISION MGCMFD_epslinf = 1.0e-5f;
FP_PRECISION CGCMFD_epseig = 1.0e-5f, CGCMFD_epsl2 = 1.0e-5f;
FP_PRECISION CGCMFD_epslinf = 1.0e-5f;

/* These three ABI shims are still referenced by retained translation units.
 * Their bodies are intentionally inert because the package does not ship the
 * resonance slow-down or double-heterogeneity data paths. */
void zres_normLocalFlxAndCalRelErr(FP_RES_FX *oldFlux,
                                   FP_RES_FX *newFlux) {
    (void) oldFlux;
    (void) newFlux;
}

void Cal_Eff_Scalar_Flux_DH(int procID) {
    (void) procID;
}

ZRES_MTC* zres_get_local_mat(void) {
    return NULL;
}
