/********************************************************************************/
/* Publication snapshot: direct external-XS GPU transport entry point.         */
/*                                                                              */
/* Supported workflow: read the transport input and external FSR CSR data,     */
/* optionally cluster FSR cross sections, then solve the eigenvalue transport   */
/* problem with the GPU Type 13--16 scattering-source paths.                    */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "MOC_AllocAndFreeMem.h"
#include "MOC_CommonMacro.h"
#include "MOC_FluxEvaluation.h"
#include "MOC_Geometry.h"
#include "MOC_GPUCPUInterface.h"
#include "MOC_Input.h"
#include "MOC_Macro_XSs.h"
#include "MOC_GPUSolver.cuh"
#include "MOC_main.h"
#include "MOC_Output.h"
#include "MOC_Quadrature_Set.h"
#include "MOC_Track.h"
#include "auxiliary.h"
#include "paper_fsr_cluster.h"

/* Defined by the compatibility boundary because the upstream post-solve
 * condensation interface still exposes this legacy count. */
extern int Mat_Number;

int main(int argc, char **argv)
{
    FP_PRECISION_TIME _time_start;
    FP_PRECISION_TIME _time_end;
    FP_PRECISION_TIME _time_tot;
    int _procID = 0;
    int _numProcs = 1;

    TimeRec_ProgramWallStart = TallyMonotonicTime_CPU();
    logo();

    if (argc < 2 || argv[1] == NULL) {
        printf("\n [**ERROR**]  Please specify a transport input file.\n");
        fflush(stdout);
        return 69;
    }

    ProcID = _procID;
    NumProcs = _numProcs;
    strcpy(IP, argv[1]);
    strcpy(IP, cleanComments(IP, "^_^", '!'));

    _time_start = TallyMonotonicTime_CPU();

    /* Input cards provide the geometry, track file, FSR map, CSR XS file and
     * runtime controls.  Resonance and slowing-down input is deliberately not
     * called by this entry point. */
    MOCInput(_procID);
    checkSysMemUsage();

#ifdef CUDA_USED
    CopyInpInfo_CPU2GPU(GPUBlockDimInp, GPUNumStreamsInp);
#endif

    MOCInitializeGeometry(_procID, _numProcs);
    if (InputTrackFile == 0) {
        SetFSRsXSsID_subDomain(Core, SubDomLocID_2D_Y, SubDomLocID_2D_X);
    }

    SetQuadratureInfo(NumTrackSet, NumAzimAng, NumPolarAng, NumScattOrder);
    MOC_Main_InitExpTable(_procID);
    MOC_Main_AllocMem(_procID);
    MPIInfo_Preparation();

#ifdef CUDA_USED
    InitCudaDevice_1(_procID, _numProcs);
    CopyInitInfoFromCPUtoGPU(_procID);
#endif

    /* External FSR macroscopic cross sections are read here. */
    MacroXSsEvaluation_subDomain(SubDomLocID_2D_Y, SubDomLocID_2D_X);

#ifdef TESTR
    paper_prepare_external_fsr_cluster(argc, argv);
#endif

#ifdef CUDA_USED
    GPU_MainAllocMem_1(_procID, _numProcs, TotalNumFSRs_subDom,
                       NumGroups, NumLTs_HS[0] * 2, NumPolarAng[0]);
#endif

    PrintCPUMemUsage(_procID);

    if (ProblemType != 0) {
        printf("\n [**ERROR**]  The reproducibility snapshot supports only ProblemType=0.\n");
        FreeMemory();
        return 69;
    }

    MOC_EigenValueProblem();

    /* Keep the standard post-solve output and memory-release path so that the
     * published executable reports the same convergence and timing records. */
    CondenseMacroXSsForMat_FromMOCSolution(
        0, IfResonance, Mat_Number, NumGroups, TotalNumFSRs_subDom,
        FSRsXSID_subDom, FSRVol_SubDom, ScalarFlux_MOC);
    MOC_Main_Output(_procID);

    _time_end = TallyMonotonicTime_CPU();
    _time_tot = _time_end - _time_start;
    CompTime_Tot = _time_tot;

    OptimalWorkLoadDistribution(_procID, _numProcs);
    MOC_FinalOutput(_time_tot, 0.0);

    FreeMemory();
    TimeRec_ProgramWallTotal = TallyMonotonicTime_CPU() - TimeRec_ProgramWallStart;
    printf("\n [  RESULT ]  Total Program Wall Time: %.6lf sec\n",
           TimeRec_ProgramWallTotal);
    fflush(stdout);
    return 0;
}
