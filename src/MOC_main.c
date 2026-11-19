/********************************************************************************/
/* @ File: MOC_main.c
/*   introduction: functions related to trasnport sweep
/* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*   1st modified:
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "MOC_AllocAndFreeMem.h"
#include "MOC_CMFD.h"
#include "MOC_CommonMacro.h"
#include "MOC_EigenValueEvaluation.h"
#include "MOC_ExpEvaluation.h"
#include "MOC_FluxEvaluation.h"
#include "MOC_Geometry.h"
#include "MOC_GPUCPUInterface.h"
#include "MOC_Input.h"
#include "MOC_Macro_XSs.h"
#include "MOC_main.h"
#include "MOC_NTE_Solver.h"
#include "MOC_Output.h"
#include "MOC_Quadrature_Set.h"
#include "MOC_ReacRateEvaluation.h"
#include "MOC_SourceEvaluation.h"
#include "MOC_Track.h"
#include "MOC_Visualization.h"
#include "MOC_VectorOperation.h"

#include "R_I_O_File.h"
#include "R_DH_Mat_XS.h"
#include "R_xs_treat.h"
#include "R_Gr_Cond.h"
#include "R_Read_Lib.h"
#include "R_Sub_Fix.h"

#include "paper_transport_compat.h"

#include "Aux_MOC.h"

#ifdef MPI_USED
    #include "mpi.h"
#endif // MPI_USED

#ifdef CUDA_USED
    #include "MOC_GPUSolver.cuh"
	#include "MOC_GPUCmfd.cuh"
#include "MOC_GPUQuery.cuh"
#include "R_GPU_FixSrc.cuh"
#endif

#if defined(OMP_ASYN) || defined(OMP_USED)
   #include <omp.h>
#endif // OMP_ASYN

#ifdef SCHED_LINUX
  #include <sched.h>
#endif // SCHED_LINUX

#ifdef TESTR
    #include "zres_common_func.h"
	#include "condition.h"
#endif

FP_PRECISION *NormPinPowerSubDom = NULL, **CollecGlobPinPower=NULL,**CollecCompTime=NULL, *NormPinPower_SumLay=NULL;
M_INT32 *CollNumPinsSubDom=NULL;
FP_PRECISION ***XS_leakage_SubDom = NULL, *XS_leakage_SubDom_ = NULL;
int ***Index_XS_Leak = NULL;
int normFluxes = 0;
int ProcID=0;
int Conv_Para = 1; // parameter used to check convergence, 1-scalar flux of FSRs; 0-NDF
int NumProcs;// , proID_set;
int MOC_bool = 0;//_layer,
FP_PRECISION **CPUMPIFlux = NULL, **BoundaryFlux = NULL, **BoundaryFlux_Temp = NULL;
FP_PRECISION *MOC_fissRate_procs = NULL, *MOC_subDomBoundOutgAngFlux = NULL, *MOC_totPower_procs = NULL;
FP_PRECISION *ExpTable_0 = NULL/*, **MOC_trackFlux = NULL, **MOC_trackFlux_1 = NULL, ***MOC_trackFlux_omp = NULL*/;
int ***Mess_TarProc = NULL, ***Mess_SendNumLTs = NULL, *MOC_numFSRs_subDom = NULL;
size_t ***Mess_SendSatrtPInAngF = NULL, ***Mess_RecvSatrtPInAngF = NULL;
// int **GCCopy_StartPInAngF = NULL, **GCCopy_NumDataInAngF = NULL;
int EigenValueIter, numUpScattGr;// g, in_iter, upScattIter
FP_PRECISION CM_glo_fiss_old = 1.0, MOCGlobAbsRateTemp,CM_glo_fiss_tem, MOC_glo_abs_tem, CM_k_eff=1.0,CM_k_effold, MOC_Residual=0.0,MOCFluxResidual = 0.0,CM_fluxResidual = 0.0, TotPower_core, globNDFErr_InfNorm = 0.0;
FP_DOUBLE MOCGlobFissRateOld = 1.0, MOCGlobFissRateTemp;
FP_DOUBLE MOCKeff, MOCKeffOld,*MOC_keff = NULL;
FP_PRECISION start_time, end_time, *MOCGSFissSour = NULL, *MOCGSScattSour = NULL, *MOC_gs_track_flux = NULL;
FP_PRECISION ****MOCtoCMFD_NDFWest_old = NULL, ****MOCtoCMFD_NDFNorth_old = NULL, ****MOCtoCMFD_NDFEast_old = NULL, ****MOCtoCMFD_NDFSouth_old = NULL;
FP_PRECISION **partialcurrent_ZPD_top, **partialcurrent_ZPD_bottom, ***netcurrent_top, ***netcurrent_bottom, *cell_volume;
FP_PRECISION Exp16_C1 = 1.0, Exp16_C2 = 1.0;
int Index_Exp = 0;
FP_PRECISION Exp10_C1 = 1.0, Exp10_C2 = 1.0;
FP_PRECISION *Exp2TH = NULL, *Exp2TL = NULL, *Exp2C1T = NULL, *ExpLinear = NULL;
FP_PRECISION *ExpTable = NULL, **ExpF0 = NULL;
FP_PRECISION **GlobPinPower = NULL, ***GlobPinPower_Slice = NULL;
FP_PRECISION Time_Total_MOC = 0.0, Time_Total_CMFD = 0.0, Time_Total_MGCMFD = 0.0, Time_Total_CGCMFD = 0.0;
FP_PRECISION Time_CMFD_to_MOC = 0.0, Time_MOC_to_CMFD = 0.0, Time_Total_NEM = 0.0;
FP_PRECISION *Time_Total_MOC_P = NULL, *Time_Total_CMFD_P = NULL, *Time_Total_MGCMFD_P = NULL, *Time_Total_CGCMFD_P = NULL;
FP_PRECISION *Time_CMFD_to_MOC_P = NULL, *Time_MOC_to_CMFD_P = NULL, *Time_Total_NEM_P = NULL;
FP_PRECISION Time_Total_MOC_A = 0.0, Time_Total_CMFD_A = 0.0, Time_Total_MGCMFD_A = 0.0, Time_Total_CGCMFD_A = 0.0;
FP_PRECISION Time_CMFD_to_MOC_A = 0.0, Time_MOC_to_CMFD_A = 0.0, Time_Total_NEM_A = 0.0;
FP_PRECISION *Time_Total_MOC_M = NULL, *Time_Total_CMFD_M = NULL, *Time_Total_MGCMFD_M = NULL, *Time_Total_CGCMFD_M = NULL;
FP_PRECISION *Time_CMFD_to_MOC_M = NULL, *Time_MOC_to_CMFD_M = NULL, *Time_Total_NEM_M = NULL;
FP_PRECISION Time_MOC_Solver = 0.0, Time_MOC_Exp = 0.0, Time_MOC_Flux = 0.0, Time_MOC_Prep = 0.0, Time_MOC_Current = 0.0, Time_MOC_Current_lt = 0.0;
FP_PRECISION Time_MOC_Solver_A = 0.0, Time_MOC_Exp_A = 0.0, Time_MOC_Flux_A = 0.0, Time_MOC_Prep_A = 0.0, Time_MOC_Current_A = 0.0, Time_MOC_Current_A_lt = 0.0;
FP_PRECISION *Time_MOC_Solver_P = NULL, *Time_MOC_Exp_P = NULL, *Time_MOC_Flux_P = NULL, *Time_MOC_Prep_P = NULL, *Time_MOC_Current_P = NULL, *Time_MOC_Current_P_lt = NULL;
FP_PRECISION *Time_MOC_Solver_M = NULL, *Time_MOC_Exp_M = NULL, *Time_MOC_Flux_M = NULL, *Time_MOC_Prep_M = NULL, *Time_MOC_Current_M = NULL, *Time_MOC_Current_M_lt = NULL;
FP_PRECISION *MOCtoCMFD_NDFWest_old_ = NULL,*MOCtoCMFD_NDFEast_old_ = NULL;
FP_PRECISION *MOCtoCMFD_NDFNorth_old_ = NULL, *MOCtoCMFD_NDFSouth_old_ = NULL;
int Time_check_counter = 0;
int **LTsIndex_AzimAngID_HS = NULL, **LTsIndex_LTID_HS = NULL;
int **LTsIndex_AzimAngID_2PI = NULL, **LTsIndex_LTID_2PI = NULL;
int **Index_SurfID_SP = NULL, **Index_SurfID_EP = NULL;
FP_PRECISION *Source = NULL/*, *Source_RayWised = NULL*/;
FP_PRECISION_FLUX  *ScalarFlux_MOC = NULL, *ScalarFluxOld = NULL, *ScalarFluxOld_0thOrder = NULL, *ScalarFlux_MOCLayer = NULL;
FP_PRECISION_FLUX  *FSRAveAngFlux_Top = NULL, *FSRAveAngFlux_Bott = NULL, *FSRAngScattSour = NULL;
FP_PRECISION_FLUX  *FSRAveAngFlux_TopTemp = NULL;
FP_PRECISION_FLUX *FSRAveAngFlux_UpAllLay = NULL, *FSRAveAngFlux_DownAllLay = NULL;
FP_PRECISION *TwoCosThetaOverHight_Sigma = NULL, *TwoCosThetaOverHight_Source = NULL;
int *TotNumFSRsPerLayer = NULL;
FP_PRECISION ***TotalWeight_overhD = NULL;
int MaxNumPolarAng;
FP_PRECISION *****QuadAngFlux_MOC=NULL;
FP_PRECISION GlobKeffTem=1.0,GlobKeffold=1.0;
int *Mess_TarProc_1 = NULL, *Mess_SendNumLTs_1 = NULL;
FP_PRECISION **CFlux = NULL, **DFlux = NULL/*, **TestFlux = NULL*/;
FP_PRECISION_TIME MPIComm_Time = 0;
char ProcessorName[ParaNameLength];
char **ProcessorNameColl=NULL;
int *CurrCPUIDColl=NULL;
int NameLen;
FP_PRECISION_FLUX  *ScalarFlux_RayWisedFSR = NULL;

FP_DOUBLE **MatSumFlux = NULL, *MatSumVol = NULL;
#ifdef OMP_USED  

#endif // OMP_USED
  FP_PRECISION_FLUX **ScalarFlux_OMP=NULL, ***MOC_trackFlux_OMP = NULL;

FP_PRECISION **LTAngFluxInci_SubDom=NULL;
FP_PRECISION **SegAngFlux = NULL; // [NumGroups * NumPolar]
FP_PRECISION **ExpTempStore = NULL,*MOC3D_oneOverSigma = NULL,*MOC3D_QtoSigma = NULL;
FP_DOUBLE MOCGlobFlux_FixSour;

#if defined(MPI_USED)
    MPI_Request MyRequest;
	MPI_Status MyStatus;
	int CurrCPU;
#endif

FP_PRECISION *MOC_fluxNormInf = NULL, *MOC_fluxNorm2 = NULL, *MOC_fluxRMSE = NULL;
int *MOC_boolFluxNormInf = NULL, *MOC_boolFluxNorm2 = NULL, *MOC_boolFluxRMSE = NULL;

FP_PRECISION_FLUX *AngularFluxOut=NULL;

/**
* @ Function: MOCInitializeGeometry()
*   introduction: prepare geometry and XSs information
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void MOCInitializeGeometry(int _procID, int _numProcs)
{
#ifdef TIME_REC
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC
	if (InputTrackFile == 0) {
        GetSubDomLocIDBasedProcID(_procID, NumSubDomains, SubDomainsMap, &SubDomLocID_2D_Y, &SubDomLocID_2D_X,
                                  &MOC_subDomID_layer);
        GetNumCellsInSubDom(SubDomLocID_2D_Y, SubDomLocID_2D_X, MOC_subDomID_layer, NumPin_subDomain, \
        &MOCNumPinsInSubDom_X, &MOCNumPinsInSubDom_Y, &MOCNumLayersZ);
        GetGlobPinCellOffset_SubDom(SubDomLocID_2D_Y, SubDomLocID_2D_X, NumPin_subDomain, \
        &SubDomOrgPinOffset_2D_X, &SubDomOrgPinOffset_2D_Y);


        NumLayer_Core = GetNumLayersInCore(Core);
        WidthLayersInCore = GetWidthLayersInCore(NumLayer_Core, Core);
        MOCInitFSRs(_procID, _numProcs, SubDomLocID_2D_Y, SubDomLocID_2D_X, MOC_subDomID_layer, \
        MOCNumPinsInSubDom_X, MOCNumPinsInSubDom_Y, MOCNumLayersZ, \
        NumSubDomains, NumPin_subDomain, NumCells, Cells);
        CoreLengthX = CoreGetLengthX(Core);
        CoreLengthY = CoreGetLengthY(Core);
        TrackGenetor(_procID, NumTrackSet, NumAzimAng, NumPolarAng, RaySpacingIni, SubDomLocID_2D_X, SubDomLocID_2D_Y);
        Interface_Geometry();
    }

	LTsIndex_AzimAngID_HS=AllocMem_int_IP(NumTrackSet,NumLTs_HS, (char *)("LTsIndex_AzimAngID_HS"));
	LTsIndex_LTID_HS=AllocMem_int_IP(NumTrackSet,NumLTs_HS,(char *)("LTsIndex_LTID_HS"));
	GetIndexOfLongTracks_HS(LTsIndex_AzimAngID_HS,LTsIndex_LTID_HS,NumTrackSet, NumAzimAng, NumLongTracks);

#ifdef GeoTwoPIRegion
    if (InputTrackFile == 0) {
        LTsIndex_AzimAngID_2PI = AllocMem_int_IP(NumTrackSet, NumLTs_2PI, (char *)("LTsIndex_AzimAngID_2PI"));
        LTsIndex_LTID_2PI = AllocMem_int_IP(NumTrackSet, NumLTs_2PI, (char *)("LTsIndex_LTID_2PI"));
        GetIndexOfLongTracks_Total(LTsIndex_AzimAngID_2PI, LTsIndex_LTID_2PI, NumTrackSet, NumAzimAng, NumLongTracks);
    }
#endif // GeoTwoPIRegion

	MaxNumSegsInTracks = GetMaxNumSegsInTracks(NumTrackSet, TotalNumTracksInRTM_Total, NumSegsInTrack_2PI);
	MaxNumTracksInLTs = GetMaxNumTracksInLongTracks(NumTrackSet, NumLTs_HS, NumTracksInLTs_HS);
	MaxNumPolarAng = GetMaxNumPolarAng(NumTrackSet,NumPolarAng);

	if (Gene_VisualFile==1) {
		// GenerateVTKFile_Visu(0, _procID, NumPointsPerRing, NumRTMGeoType, CellBasedRTMGeo,0, NULL, NULL,RTMMeshType,\
		//     NumRings_RTM_Mat,NumRings_RTM_Mesh,NumSectorsPerRing_RTM_Mesh,RadiusPerRing_RTM_Mat,RadiusPerRing_RTM_Mesh,\
	    //     NumSubDomains,NumPin_subDomain,WidthLayersInCore,CellRTMType_SubDom,NumPinsInSubDomPerLayer, SubDomGlobCellID,\
		// 	CellTypeID_visu,NumCells,Cells, MatID_RTM, NumCellsInRTM,NumAssemblies,Assemblies, NumLattices,Lattices,\
		// 	SumNumFSRsBasedGlobCellID,NumCellsInPin);
		// GenerateVTKFile_Visu(1, _procID, NumPointsPerRing, NumRTMGeoType, CellBasedRTMGeo,0, NULL, NULL,RTMMeshType,\
		//     NumRings_RTM_Mat,NumRings_RTM_Mesh,NumSectorsPerRing_RTM_Mesh,RadiusPerRing_RTM_Mat,RadiusPerRing_RTM_Mesh,\
	    //     NumSubDomains,NumPin_subDomain,WidthLayersInCore,CellRTMType_SubDom,NumPinsInSubDomPerLayer, SubDomGlobCellID,\
		// 	CellTypeID_visu,NumCells,Cells, MatID_RTM, NumCellsInRTM,NumAssemblies,Assemblies, NumLattices,Lattices,\
		// 	SumNumFSRsBasedGlobCellID,NumCellsInPin);
	}
#ifdef TIME_REC
	_time_end = TallyCurrentTime_CPU();
	TimePrePart[TimeCountPrePart] = _time_end - _time_start;
	strcpy(TimeMetricsPrePart[TimeCountPrePart], "Initialize Geometry Info");
	TimeCountPrePart++;
#endif // TIME_REC
}


/**
* @ Function: MOC_Main_AllocMem()
*   introduction: allocate memory for MOC calculation
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void MOC_Main_AllocMem(int _procID)
{
	int _width1 = 12, i, _check=0/*, j*/;
	FP_PRECISION _val1 = (FP_PRECISION)(0.0)/*, _mem*/;
#if defined(M_MG_CMFD)
	int _dimNode, _dimSurfX, _dimSurfY, _dimSurfZ, _dimGNx;
	int _nx, _ny, _nz, _nxy, _nxyz;
	char _lsSolver_MG[64];
	char _gmresPrec[16];

#if defined(M_CG_CMFD)
	char _lsSolver_CG[64];
#endif // defined(M_CG_CMFD)
#endif // defined(M_MG_CMFD)

#ifdef TIME_REC
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC

    if ((ProblemType == 0 && IfResonance == 0) || (ProblemType == 2 && IfResonance == 0)) {
		MacroXS_s_NNZ = GetNNZ_MacroXS_s();
		MacroXS_s_MaxNNZ_Layer = GetMaxNNZ_MacroXS_s_Lay();
	}

	MemPrintCount++;
	sprintf(MemParaName[MemPrintCount], "Scalar Flux and Source Info");
	MemParaCountBoundLow[MemPrintCount] = ParamCounter;

	ScalarFlux_MOC = AllocMem_FP_Flux((size_t)(TotalNumFSRs_subDom)*(size_t)(NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1)), (char *)("ScalarFlux_MOC"));
#ifdef SAVE_ANGLE_FLUX
	if (IfResonance == 1) {
		ScalarFluxOld = AllocMem_FP_Flux((size_t)(TotalNumFSRs_subDom)*(size_t)(NumGroups * (SAVE_ORDER + 1) * (SAVE_ORDER + 1)), (char *)("ScalarFluxOld"));
	}
	else {
		ScalarFluxOld = AllocMem_FP_Flux((size_t)(TotalNumFSRs_subDom)*(size_t)(NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1)), (char *)("ScalarFluxOld"));
	}
#else
	ScalarFluxOld = AllocMem_FP_Flux((size_t)(TotalNumFSRs_subDom)*(size_t)(NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1)), (char *)("ScalarFluxOld"));
#endif
#if defined(M_MG_CMFD)
	ScalarFluxOld_0thOrder = AllocMem_FP_Flux((size_t)(TotalNumFSRs_subDom)*(size_t)(NumGroups), (char *)("ScalarFluxOld_0thOrder"));
#endif // defined(M_MG_CMFD)
#if defined(CUDA_USED)
    ScalarFlux_MOCLayer = AllocMem_FP_Flux((size_t)(TotalNumFSRs_perLayer)*(size_t)(NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1)), (char *)("ScalarFlux_MOCLayer"));
#endif

#if defined(M_2DMOC)
    if (NumScattOrder > 0 ||  CPU_TSFunctionType > 10) {
		if (CPU_TSFunctionType==11) {
			_check = NumPolarAng[0]/2;
		}
		else if (CPU_TSFunctionType==12 || CPU_TSFunctionType==13) {
			_check = NumPolarAng[0];
		}
		else {}
		
		FSRAveAngFlux_Top = AllocMem_FP_Flux((size_t)(TotalNumFSRs_perLayer)*(size_t)(NumGroups)*(size_t)(NumAzimAng[0] * _check), (char *)("FSRAveAngFlux_Top"));
		if (CPU_TSFunctionType==13) {
		    FSRAngScattSour = AllocMem_FP_Flux((size_t)(TotalNumFSRs_perLayer)*(size_t)(NumGroups)*(size_t)(NumAzimAng[0] * _check), (char *)("FSRAngScattSour"));
		}
	}
#endif

	Source = AllocMem_FP_I((size_t)(TotalNumFSRs_subDom)*(size_t)(NumGroups), (char *)("Source"));

    MOC_boolFluxNormInf = AllocMem_i32_I((size_t)(1), (char *)("MOC_boolFluxNormInf"));
    MOC_boolFluxNorm2 = AllocMem_i32_I((size_t)(1), (char *)("MOC_boolFluxNorm2"));
    MOC_boolFluxRMSE = AllocMem_i32_I((size_t)(1), (char *)("MOC_boolFluxRMSE"));

	MOC_fluxNormInf = AllocMem_FP_I((size_t)(1), (char *)("MOC_fluxNormInf"));
	MOC_fluxNorm2 = AllocMem_FP_I((size_t)(1), (char *)("MOC_fluxNorm2"));
	MOC_fluxRMSE = AllocMem_FP_I((size_t)(1), (char *)("MOC_fluxRMSE"));

	if (OUTPUT_flux_mat==1 && IfResonance==0 && INPUT_XSs_MATorFSR==0) {
		MatSumFlux = AllocMem_FPDouble_II(NumXSs_Mat, NumGroups, (char *)("MatSumFlux"));
		MatSumVol = AllocMem_FPDouble_I(NumXSs_Mat, (char *)("MatSumVol"));
	}
	MemParaCountBoundUp[MemPrintCount] = ParamCounter;

	MemPrintCount++;
	sprintf(MemParaName[MemPrintCount], "Radial Ang Flux Info");
	MemParaCountBoundLow[MemPrintCount] = ParamCounter;

#if defined(MPI_USED)
    if (NumProcs > 1) {
	    CPUMPIFlux = AllocMemInciAngFlux(NumTrackSet, NumLTs_HS, NumGroups, NumPolarAng, MOCNumLayersZ, (char *)("CPUMPIFlux"));
	}
#endif // defined(MPI_USED)
	BoundaryFlux = AllocMemInciAngFlux(NumTrackSet, NumLTs_HS, NumGroups, NumPolarAng, MOCNumLayersZ, (char *)("BoundaryFlux"));
#if !defined(M_UpdateRadialBoundImme)
	BoundaryFlux_Temp = AllocMemInciAngFlux(NumTrackSet, NumLTs_HS, NumGroups, NumPolarAng, MOCNumLayersZ, (char *)("BoundaryFlux_Temp"));
#endif // !defined(M_UpdateRadialBoundImme)

	MemParaCountBoundUp[MemPrintCount] = ParamCounter;
	// SegAngFlux = AllocMem_FP_I_1(NumGroups*MaxNumPolarAng, (char *)("SegAngFlux"));
	SegAngFlux = AllocMem_FP_II(NumOMPThread,NumGroups*MaxNumPolarAng, (char *)("SegAngFlux"));
	ExpTempStore = AllocMem_FP_II(NumOMPThread,MaxNumTracksInLTs*MaxNumSegsInTracks*NumGroups*MaxNumPolarAng, (char *)("ExpTempStore"));
#if defined(M_3DMOCex)
	MOC3D_oneOverSigma = AllocMem_FP_I_1(MaxNumTracksInLTs*MaxNumSegsInTracks*NumGroups*MaxNumPolarAng, (char *)("MOC3D_oneOverSigma"));
	MOC3D_QtoSigma = AllocMem_FP_I_1(MaxNumTracksInLTs*MaxNumSegsInTracks*NumGroups*MaxNumPolarAng, (char *)("MOC3D_QtoSigma"));

	MemPrintCount++;
	sprintf(MemParaName[MemPrintCount], "Axial Ang Flux Info");
	MemParaCountBoundLow[MemPrintCount] = ParamCounter;
	FSRAveAngFlux_Top = AllocMem_FP_Flux((size_t)(TotalNumFSRs_perLayer)*(size_t)(NumGroups)*(size_t)(NumAzimAng[0] * (NumPolarAng[0] / 2)), (char *)("FSRAveAngFlux_Top"));
	FSRAveAngFlux_Bott = AllocMem_FP_Flux((size_t)(TotalNumFSRs_perLayer)*(size_t)(NumGroups)*(size_t)(NumAzimAng[0] * (NumPolarAng[0] / 2)), (char *)("FSRAveAngFlux_Bott"));
    if (/*NumScattOrder > 0 || */CPU_TSFunctionType == 13) {
	    FSRAngScattSour = AllocMem_FP_Flux((size_t)(TotalNumFSRs_perLayer)*(size_t)(NumGroups)*(size_t)(NumAzimAng[0] * (NumPolarAng[0] / 2)), (char *)("FSRAngScattSour"));
	}
	VectorInit_FP_AxialAngFlux(FSRAveAngFlux_Top, TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0] * (NumPolarAng[0] / 2), (FP_PRECISION)(0.0));
	VectorInit_FP_AxialAngFlux(FSRAveAngFlux_Bott, TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0] * (NumPolarAng[0] / 2), (FP_PRECISION)(0.0));
	TwoCosThetaOverHight_Sigma = AllocMem_FP_I_1(MOCNumLayersZ*NumPolarAng[0], (char *)("TwoCosThetaOverHight_Sigma"));
	TwoCosThetaOverHight_Source = AllocMem_FP_I_1(MOCNumLayersZ*NumPolarAng[0], (char *)("TwoCosThetaOverHight_Source"));
	
#if !defined(M_UpdateAxialBoundImme)
	FSRAveAngFlux_TopTemp = AllocMem_FP_Flux((size_t)(TotalNumFSRs_perLayer)*(size_t)(NumGroups)*(size_t)(NumAzimAng[0] * (NumPolarAng[0] / 2)), (char *)("FSRAveAngFlux_TopTemp"));
#endif // !defined(M_UpdateAxialBoundImme)
	MemParaCountBoundUp[MemPrintCount] = ParamCounter;
#endif // defined(M_3DMOCex)


#ifdef GC_COPY_AZIM
#if defined(M_Use4PI)
	_val1 = (FP_PRECISION)(ONE_OVER_FOUR_PI);
#else
	_val1 = (FP_PRECISION)(1.0);
#endif // defined(M_Use4PI)

	InitInciAngFlux(BoundaryFlux, NumTrackSet, NumLTs_HS, NumGroups, NumPolarAng, MOCNumLayersZ, _val1);

#if !defined(M_UpdateRadialBoundImme)
	InitInciAngFlux(BoundaryFlux_Temp, NumTrackSet, NumLTs_HS, NumGroups, NumPolarAng, MOCNumLayersZ, _val1);
#endif // !defined(M_UpdateRadialBoundImme)

#else
	InitInciAngFlux(BoundaryFlux, NumTrackSet, NumLTs_HS, NumGroups, NumPolarAng, MOCNumLayersZ, (FP_PRECISION)(0.0));
#if !defined(M_UpdateRadialBoundImme)
	InitInciAngFlux(BoundaryFlux_Temp, NumTrackSet, NumLTs_HS, NumGroups, NumPolarAng, MOCNumLayersZ, (FP_PRECISION)(0.0));
#endif // !defined(M_UpdateRadialBoundImme)
#endif // !GC_COPY_AZIM

	if (OUTPUT_normPinPower == 1) {
		NormPinPowerSubDom = AllocMem_FP_I_1(MOCNumLayersZ*NumPinsInSubDomPerLayer, (char *)("NormPinPowerSubDom"));
#if defined(MPI_USED)
        if (NumProcs > 1) {
			CollNumPinsSubDom = AllocMem_i32_i32(NumProcs, (char *)("CollNumPinsSubDom"));
			MOC_NumPinsInGlobDom = Get_NumPins_GlobDom(CollNumPinsSubDom, MOC_NumPinsInSubDom, ProcID, NumProcs);
			printf("\n MOC_NumPinsInGlobDom=%d",MOC_NumPinsInGlobDom);
			if (ProcID==0) {
				CollecGlobPinPower = AllocMem_FP_IP(NumProcs, CollNumPinsSubDom, (char *)("CollecGlobPinPower"));
			}
		}
#endif
		NormPinPower_SumLay = AllocMem_FP_I_1(PinPowerSumNumSet*NumPinsInSubDomPerLayer, (char *)("NormPinPower_SumLay"));
		MOC_totPower_procs = AllocMem_FP_I_1(NumProcs, (char *)("MOC_totPower_procs"));
	}
#ifdef OMP_ASYN
	CFlux = AllocMemInciAngFlux(NumTrackSet, NumLTs_HS, NumGroups, NumPolarAng, MOCNumLayersZ, (char *)("CFlux"));
	InitInciAngFlux(CFlux, NumTrackSet, NumLTs_HS, NumGroups, NumPolarAng, MOCNumLayersZ, (FP_PRECISION)(ONE_OVER_FOUR_PI));

#endif // OMP_ASYN
	// DFlux = AllocMemInciAngFlux(NumTrackSet, NumLTs_HS, NumGroups, NumPolarAng, MOCNumLayersZ, (char *)("DFlux"));
	// InitInciAngFlux(DFlux, NumTrackSet, NumLTs_HS, NumGroups, NumPolarAng, MOCNumLayersZ, (FP_PRECISION)(-1.0));

#ifdef CUDA_USED
	CPUReacRateCollect = AllocMem_FP_I_1(NumProcs, (char *)("CPUReacRateCollect"));
	GPUFluxResRColl = AllocMem_FP_I_1(NumProcs, (char *)("GPUFluxResRColl"));
#endif // CUDA_USED

#ifdef MPI_USED
	MemPrintCount++;
	sprintf(MemParaName[MemPrintCount], "MPI Comm Info");
	MemParaCountBoundLow[MemPrintCount] = ParamCounter;
	Mess_TarProc = AllocMem_int_III(MOCNumLayersZ, NumAzimAng[0], NumProcPerLayer*NumNeibSubDom, (char *)("Mess_TarProc"));
	Mess_SendNumLTs = AllocMem_int_III(MOCNumLayersZ, NumAzimAng[0], NumProcPerLayer*NumNeibSubDom, (char *)("Mess_SendNumLTs"));
	Mess_SendSatrtPInAngF = AllocMem_sizeT_III(MOCNumLayersZ, NumAzimAng[0], NumNeibSubDom, (char *)("Mess_SendSatrtPInAngF"));
	Mess_RecvSatrtPInAngF = AllocMem_sizeT_III(MOCNumLayersZ, NumAzimAng[0], NumProcPerLayer * NumNeibSubDom, (char *)("Mess_RecvSatrtPInAngF"));
	// GCCopy_StartPInAngF = AllocMem_int_II(MOCNumLayersZ, NumAzimAng[0], (char *)("GCCopy_StartPInAngF"));
	// GCCopy_NumDataInAngF = AllocMem_int_II(MOCNumLayersZ, NumAzimAng[0],(char *)("GCCopy_NumDataInAngF"));
	Mess_TarProc_1 = AllocMem_i32_i32(NumProcPerLayer*NumNeibSubDom, (char *)("Mess_TarProc_1"));
	Mess_SendNumLTs_1 = AllocMem_i32_i32(NumProcPerLayer*NumNeibSubDom, (char *)("Mess_SendNumLTs_1"));

	BoundAngFlux_send = AllocMem_FP_II(NumNeibSubDom, MaxNumLTs_Azim * NumPolarAng[0] * NumGroups, (char *)("BoundAngFlux_send"));
	BoundAngFlux_recv = AllocMem_FP_II(NumNeibSubDom, MaxNumLTs_Azim * NumPolarAng[0] * NumGroups, (char *)("BoundAngFlux_recv"));

	NextSubDomLTIDs_send = AllocMem_int_II(NumNeibSubDom, MaxNumLTs_Azim, (char *)("NextSubDomLTIDs_send"));
	NextSubDomLTIDs_recv = AllocMem_int_II(NumNeibSubDom, MaxNumLTs_Azim, (char *)("NextSubDomLTIDs_recv"));

	MPIProcID_recv = AllocMem_int_III(MOCNumLayersZ, NumAzimAng[0], NumNeibSubDom, (char *)("MPIProcID_recv"));

	MPINeibID_recv = AllocMem_int_III(MOCNumLayersZ, NumAzimAng[0], NumNeibSubDom, (char *)("MPINeibID_recv"));
	MemParaCountBoundUp[MemPrintCount] = ParamCounter;
#endif //MPI_USED

//#if !defined(CUDA_USED) || (defined(CUDA_USED) && defined(OMP_USED))
	// MOC_trackFlux = AllocMem_FP_II(NumGroups, MaxNumPolarAng, "MOC_trackFlux");
	// MOC_trackFlux_1 = AllocMem_FP_II(NumOMPThread, NumGroups*MaxNumPolarAng, "MOC_trackFlux_1");
	// MOC_trackFlux_omp = AllocMem_FP_III(NumOMPThread, NumGroups, MaxNumPolarAng, "MOC_trackFlux_omp");
	//if (_procID == 0) {
	//	printf("\n [CheckInfo]  NumGroups = %d, MAX_NumGroups = %d", NumGroups, MAX_NumGroups);
	//	printf("\n              MaxNumPolarAng = %d, MAX_NumPolAngles = %d", MaxNumPolarAng, MAX_NumPolAngles);
	//	printf("\n              MaxNumTracksInLTs*MaxNumSegsInTracks = %d * %d = %d, MAX_NumSegInLT = %d", \
	//		MaxNumTracksInLTs, MaxNumSegsInTracks, MaxNumTracksInLTs*MaxNumSegsInTracks, MAX_NumSegInLT);
	//	fflush(stdout);
	//	if ((NumGroups > MAX_NumGroups) || (MaxNumPolarAng > MAX_NumPolAngles) || ((CPU_TSFunctionType == 3) && (MaxNumTracksInLTs*MaxNumSegsInTracks > MAX_NumSegInLT))) {
	//		exit(69);
	//	}
	//}

	if (_procID == 0) {
		MOC_fissRate_procs = AllocMem_FP_I_1(NumProcs, (char *)("MOC_fissRate_procs"));
		MOC_numFSRs_subDom = AllocMem_i32_i32(NumProcs, (char *)("MOC_numFSRs_subDom"));
		MOC_keff = AllocMem_FPDouble_I(MaxNumOuterIter_MOC, (char *)("MOC_keff"));
	}

	TotalNumFSRs_globDom = Get_NumFSRs_GlobDom(MOC_numFSRs_subDom, TotalNumFSRs_subDom, _procID, NumProcs);
//#endif // !defined(CUDA_USED) || (defined(CUDA_USED) && defined(OMP_USED))

#if defined(OMP_USED) && defined(TIME_REC) && defined(TIME_REC_OMP)
	CompTime_omp = AllocMem_FP_I_1(NumOMPThread, (char *)("CompTime_omp"));
#endif // 


#if defined(M_MG_CMFD)
	MemPrintCount++;
	sprintf(MemParaName[MemPrintCount], "MG CMFD Info");
	MemParaCountBoundLow[MemPrintCount] = ParamCounter;
	_dimNode = MOC_NumPinsInSubDom * NumCMFDGroup;
	_dimSurfX = MOCNumLayersZ * (MOCNumPinsInSubDom_X + 1) * MOCNumPinsInSubDom_Y * NumCMFDGroup;
	_dimSurfY = MOCNumLayersZ * MOCNumPinsInSubDom_X * (MOCNumPinsInSubDom_Y + 1) * NumCMFDGroup;
	_dimSurfZ = (MOCNumLayersZ + 1) * MOCNumPinsInSubDom_X * MOCNumPinsInSubDom_Y * NumCMFDGroup;

	CMFD_NodeVol = AllocMem_FP_I_1(MOC_NumPinsInSubDom, (char *)("CMFD_NodeVol"));
	SetCMFDNodeVolume(CMFD_NodeVol, MOC_NumPinsInSubDom, RTMArea, \
		CellRTMType_SubDom, WidthLayersInCore, NumPinsInSubDomPerLayer);
	CMFDHetScalFlux = AllocMem_FP_I_1(_dimNode, (char *)("CMFDHetScalFlux"));
	CMFDScalarFlux = AllocMem_FP_I_1(_dimNode, (char *)("CMFDScalarFlux"));
	CMFDScalarFlux_Old = AllocMem_FP_I_1(_dimNode, (char *)("CMFDScalarFlux_Old"));

	if (MGCMFD_LinearSolver == 0 || MGCMFD_LinearSolver == 1) {
		CMFDScalarFlux_Old_LS = AllocMem_FP_I_1(_dimNode, (char *)("CMFDScalarFlux_Old_LS"));
	}

	CMFDTotalSource = AllocMem_FP_I_1(_dimNode, (char *)("CMFDTotalSource"));
	Index_1DArray = AllocMem_i32_i32(_dimNode, (char *)("Index_1DArray"));
	CMFDXs_t = AllocMem_FP_I_1(_dimNode, (char *)("CMFDXs_t"));
	if (OUTPUT_CMFDGroupConstant == 1 || OUTPUT_CoarseMesh_GroupConstant == 1) {
		CMFDXs_a = AllocMem_FP_I_1(_dimNode, (char *)("CMFDXs_a"));
		CMFDXs_f = AllocMem_FP_I_1(_dimNode, (char *)("CMFDXs_f"));
	}
	CMFDXs_nf = AllocMem_FP_I_1(_dimNode, (char *)("CMFDXs_nf"));
	CMFD_Chi = AllocMem_FP_I_1(_dimNode, (char *)("CMFD_Chi"));
	CMFDXs_s = AllocMem_FP_I_1(_dimNode * NumCMFDGroup, (char *)("CMFDXs_s"));
	CMFDNodeDiffCoeff = AllocMem_FP_I_1(_dimNode, (char *)("CMFDNodeDiffCoeff"));
	CMFDSurfDiffCoeff_X = AllocMem_FP_I_1(_dimSurfX, (char *)("CMFDSurfDiffCoeff_X"));
	CMFDSurfDiffCoeff_Y = AllocMem_FP_I_1(_dimSurfY, (char *)("CMFDSurfDiffCoeff_Y"));
	CMFDSurfDiffCoeff_Z = AllocMem_FP_I_1(_dimSurfZ, (char *)("CMFDSurfDiffCoeff_Z"));
	CMFDSurfNDF_X = AllocMem_FP_I_1(_dimSurfX, (char *)("CMFDSurfNDF_X"));
	CMFDSurfNDF_Y = AllocMem_FP_I_1(_dimSurfY, (char *)("CMFDSurfNDF_Y"));
	CMFDSurfNDF_Z = AllocMem_FP_I_1(_dimSurfZ, (char *)("CMFDSurfNDF_Z"));
#if defined(M_YQZ_CMFD_Damp)
	CMFDSurfNDF_XOld = AllocMem_FP_I_1(_dimSurfX, (char *)("CMFDSurfNDF_XOld"));
	CMFDSurfNDF_YOld = AllocMem_FP_I_1(_dimSurfY, (char *)("CMFDSurfNDF_YOld"));
	CMFDSurfNDF_ZOld = AllocMem_FP_I_1(_dimSurfZ, (char *)("CMFDSurfNDF_ZOld"));
#endif // defined(M_YQZ_CMFD_Damp)

#if defined(MPI_USED)
	MGCMFDData_MPIComm_SumInd = AllocMem_i32_i32(6, (char *)("MGCMFDData_MPIComm_SumInd"));
	MGCMFDData_MPIComm_Num = GetDataNum_CMFDCommPara(6, MOCNumPinsInSubDom_X, MOCNumPinsInSubDom_Y, MOCNumLayersZ, NumCMFDGroup, MGCMFDData_MPIComm_SumInd);
	MGCMFDComm_MPISend = AllocMem_FP_I_1(MGCMFDData_MPIComm_Num, (char *)("MGCMFDComm_MPISend"));
	MGCMFDFlux_MPIRecv = AllocMem_FP_I_1(MGCMFDData_MPIComm_Num, (char *)("MGCMFDFlux_MPIRecv"));
	MGCMFDNodeD_MPIRecv = AllocMem_FP_I_1(MGCMFDData_MPIComm_Num, (char *)("MGCMFDNodeD_MPIRecv"));
	CMFDComm_TarProc = AllocMem_i32_i32(6, (char *)("CMFDComm_TarProc"));
	MGCMFDComm_NumData = AllocMem_i32_i32(6, (char *)("MGCMFDComm_NumData"));
	CMFDComm_DirBound = AllocMem_int_III(6, 3, 2, (char *)("CMFDComm_DirBound"));
	CMFDComm_RecvDir = AllocMem_i32_i32(6, (char *)("CMFDComm_RecvDir"));
	MGCMFDData_MPIComm_Index = AllocMem_i32_i32(MGCMFDData_MPIComm_Num, (char *)("MGCMFDData_MPIComm_Index"));
	MGCMFDMPI_InBoundSourCoeff = AllocMem_FP_I_1(MGCMFDData_MPIComm_Num, (char *)("MGCMFDMPI_InBoundSourCoeff"));

	MGCMFD_MOCParCurrInnerBound_Num = GetDataNum_CMFDCommPara(6, MOCNumPinsInSubDom_X, MOCNumPinsInSubDom_Y, MOCNumLayersZ, NumGroups, MGCMFDData_MPIComm_SumInd);
	MGCMFD_MOCParCurrInnerBound_Index = AllocMem_i32_i32(MGCMFD_MOCParCurrInnerBound_Num, (char *)("MGCMFD_MOCParCurrInnerBound_Index"));
	MGCMFD_MOCParCurrInnerBound_Dir = AllocMem_i32_i32(MGCMFD_MOCParCurrInnerBound_Num, (char *)("MGCMFD_MOCParCurrInnerBound_Dir"));

	MGCMFDMPI_InBoundSourCoeff_Index = AllocMem_i32_i32(MGCMFDData_MPIComm_Num, (char *)("MGCMFDMPI_InBoundSourCoeff_Index"));
	MGCMFDMPI_InBoundSourCoeff_Dir = AllocMem_i32_i32(MGCMFDData_MPIComm_Num, (char *)("MGCMFDMPI_InBoundSourCoeff_Dir"));
	MGCMFDMPI_InBoundSourCoeff_GeoDim = AllocMem_FP_I_1(MGCMFDData_MPIComm_Num, (char *)("MGCMFDMPI_InBoundSourCoeff_GeoDim"));
#endif //defined(MPI_USED)

	MOCParCurr_X_Pos = AllocMem_FP_I_1(MOCNumLayersZ * (MOCNumPinsInSubDom_X + 1) * MOCNumPinsInSubDom_Y * NumGroups, (char *)("MOCParCurr_X_Pos"));
	MOCParCurr_X_Neg = AllocMem_FP_I_1(MOCNumLayersZ * (MOCNumPinsInSubDom_X + 1) * MOCNumPinsInSubDom_Y * NumGroups, (char *)("MOCParCurr_X_Neg"));
	MOCParCurr_Y_Pos = AllocMem_FP_I_1(MOCNumLayersZ * MOCNumPinsInSubDom_X * (MOCNumPinsInSubDom_Y + 1) * NumGroups, (char *)("MOCParCurr_Y_Pos"));
	MOCParCurr_Y_Neg = AllocMem_FP_I_1(MOCNumLayersZ * MOCNumPinsInSubDom_X * (MOCNumPinsInSubDom_Y + 1) * NumGroups, (char *)("MOCParCurr_Y_Neg"));
	MOC_NetCurr_Z = AllocMem_FP_I_1((MOCNumLayersZ + 1) * MOCNumPinsInSubDom_X * MOCNumPinsInSubDom_Y * NumGroups, (char *)("MOC_NetCurr_Z"));

	MOCtoCMFD_ParCurr_X_Pos = AllocMem_FP_I_1(_dimSurfX, (char *)("MOCtoCMFD_ParCurr_X_Pos"));
	MOCtoCMFD_ParCurr_X_Neg = AllocMem_FP_I_1(_dimSurfX, (char *)("MOCtoCMFD_ParCurr_X_Neg"));
	MOCtoCMFD_ParCurr_Y_Pos = AllocMem_FP_I_1(_dimSurfY, (char *)("MOCtoCMFD_ParCurr_Y_Pos"));
	MOCtoCMFD_ParCurr_Y_Neg = AllocMem_FP_I_1(_dimSurfY, (char *)("MOCtoCMFD_ParCurr_Y_Neg"));

	MGCMFDNetCurr_X = AllocMem_FP_I_1(_dimSurfX, (char *)("MGCMFDNetCurr_X"));
	MGCMFDNetCurr_Y = AllocMem_FP_I_1(_dimSurfY, (char *)("MGCMFDNetCurr_Y"));
	MGCMFDNetCurr_Z = AllocMem_FP_I_1(_dimSurfZ, (char *)("MGCMFDNetCurr_Z"));
#if defined(PTESC_LIB) || !defined(DIRECT_CSR)
	CMFDMatrix_A = AllocMem_FP_I_1(MOC_NumPinsInSubDom * NumCMFDGroup * MOC_NumPinsInSubDom * NumCMFDGroup, (char *)("CMFDMatrix_A"));
#endif //defined(PTESC_LIB) || !defined(DIRECT_CSR)

	MGCMFD_FluxFlag = AllocMem_i32_i32(1, (char *)("MGCMFD_FluxFlag"));
	MGCMFD_NegFluxCount = AllocMem_float_I(1, (char *)("MGCMFD_NegFluxCount"));

	MGCMFD_fissQflagl2 = AllocMem_i32_i32(1, (char *)("MGCMFD_fissQflagl2"));
	MGCMFD_fissQflaglinf = AllocMem_i32_i32(1, (char *)("MGCMFD_fissQflaglinf"));
	MGCMFD_flageig = AllocMem_i32_i32(1, (char *)("MGCMFD_flageig"));
	MGCMFD_fissQerrlinf = AllocMem_FP_I_1(NumProcs, (char *)("MGCMFD_fissQerrlinf"));
	MGCMFD_fissQrerrl2 = AllocMem_FP_I_1(NumProcs, (char *)("MGCMFD_fissQrerrl2"));
	MGCMFD_fluxErrlinf = AllocMem_FP_I_1(1, (char *)("MGCMFD_fluxErrlinf"));

	MGCMFD_TotFissQ = AllocMem_FP_I_1(MOC_NumPinsInSubDom, (char *)("MGCMFD_TotFissQ"));
	MGCMFD_TotFissQ_old = AllocMem_FP_I_1(MOC_NumPinsInSubDom, (char *)("MGCMFD_TotFissQ_old"));

	MGCMFD_fluxNormInf = AllocMem_FP_I((size_t)(1), (char *)("MGCMFD_fluxNormInf"));
	MGCMFD_fluxNorm2 = AllocMem_FP_I((size_t)(1), (char *)("MGCMFD_fluxNorm2"));
	MGCMFD_fluxRMSE = AllocMem_FP_I((size_t)(1), (char *)("MGCMFD_fluxRMSE"));
	MGCMFD_boolFluxNormInf = AllocMem_i32_I((size_t)(1), (char *)("MGCMFD_boolFluxNormInf"));
	MGCMFD_boolFluxNorm2 = AllocMem_i32_I((size_t)(1), (char *)("MGCMFD_boolFluxNorm2"));
	MGCMFD_boolFluxRMSE = AllocMem_i32_I((size_t)(1), (char *)("MGCMFD_boolFluxRMSE"));

	MGCMFD_LSFluxRes = AllocMem_FP_I((size_t)(1), (char *)("MGCMFD_LSFluxRes"));


	if (MGCMFD_LinearSolver == 3) {
		MGCMFD_CCW = AllocMem_FP_I_1(_dimNode, (char *)("MGCMFD_CCW"));
		MGCMFD_CCN = AllocMem_FP_I_1(_dimNode, (char *)("MGCMFD_CCN"));
		MGCMFD_CCE = AllocMem_FP_I_1(_dimNode, (char *)("MGCMFD_CCE"));
		MGCMFD_CCS = AllocMem_FP_I_1(_dimNode, (char *)("MGCMFD_CCS"));
		MGCMFD_CCB = AllocMem_FP_I_1(_dimNode, (char *)("MGCMFD_CCB"));
		MGCMFD_CCT = AllocMem_FP_I_1(_dimNode, (char *)("MGCMFD_CCT"));
		MGCMFD_AM = AllocMem_FP_I_1(_dimNode, (char *)("MGCMFD_AM"));

		_dimGNx = NumCMFDGroup * MOCNumPinsInSubDom_X;
		MGCMFD_Del = AllocMem_FP_I_1(_dimGNx, (char *)("MGCMFD_Del"));
		MGCMFD_Al = AllocMem_FP_I_1(_dimNode, (char *)("MGCMFD_Al"));
		MGCMFD_Au = AllocMem_FP_I_1(_dimGNx, (char *)("MGCMFD_Au"));
		MGCMFD_Delinv = AllocMem_FP_I_1(_dimNode, (char *)("MGCMFD_Delinv"));
		MGCMFD_Deliau = AllocMem_FP_I_1(_dimNode, (char *)("MGCMFD_Deliau"));
		MGCMFD_Ainvl = AllocMem_FP_I_1(_dimGNx, (char *)("MGCMFD_Ainvl"));
		MGCMFD_Ainvu = AllocMem_FP_I_1(_dimGNx, (char *)("MGCMFD_Ainvu"));
		MGCMFD_Ainvd = AllocMem_FP_I_1(_dimGNx, (char *)("MGCMFD_Ainvd"));

		MGCMFD_TotSour_g = AllocMem_FP_I_1(MOC_NumPinsInSubDom, (char *)("MGCMFD_TotSour_g"));

		_nxyz = MOC_NumPinsInSubDom;
		_nz = MOCNumLayersZ;
		_nx = MOCNumPinsInSubDom_X;
		_ny = MOCNumPinsInSubDom_Y;
		_nxy = MOCNumPinsInSubDom_X * MOCNumPinsInSubDom_Y;

		MGCMFD_flux_in = AllocMem_FP_I_1((_nxy + 2)*_nz, (char *)("MGCMFD_flux_in"));
		MGCMFD_flux_add_in = AllocMem_FP_I_1(_nxyz, (char *)("MGCMFD_flux_add_in"));
		MGCMFD_src_in = AllocMem_FP_I_1(_nxyz, (char *)("MGCMFD_src_in"));
		MGCMFD_vr0_mg = AllocMem_FP_I_1(_nxyz, (char *)("MGCMFD_vr0_mg"));
		MGCMFD_vr_mg = AllocMem_FP_I_1(_nxyz, (char *)("MGCMFD_vr_mg"));
		MGCMFD_vp_mg = AllocMem_FP_I_1(_nxyz, (char *)("MGCMFD_vp_mg"));
		MGCMFD_vv_mg = AllocMem_FP_I_1(_nxyz, (char *)("MGCMFD_vv_mg"));
		MGCMFD_vs_mg = AllocMem_FP_I_1(_nxyz, (char *)("MGCMFD_vs_mg"));
		MGCMFD_vt_mg = AllocMem_FP_I_1(_nxyz, (char *)("MGCMFD_vt_mg"));
		MGCMFD_vy_mg = AllocMem_FP_I_1((_nxy + 2)*_nz, (char *)("MGCMFD_vy_mg"));
		MGCMFD_vz_mg = AllocMem_FP_I_1((_nxy + 2)*_nz, (char *)("MGCMFD_vz_mg"));
		MGCMFD_s_mg = AllocMem_FP_I_1(_nxy + 2 * _nx, (char *)("MGCMFD_s_mg"));
		MGCMFD_b0_mg = AllocMem_FP_I_1(_nxy, (char *)("MGCMFD_b0_mg"));
		MGCMFD_s1dl_mg = AllocMem_FP_I_1(_nx, (char *)("MGCMFD_s1dl_mg"));
		MGCMFD_b01d_mg = AllocMem_FP_I_1(_nx,(char *)("MGCMFD_b01d_mg"));
		MGCMFD_y_mg = AllocMem_FP_I_1(_nx, (char *)("MGCMFD_y_mg"));
		MGCMFD_nrnx = AllocMem_i32_i32(_ny, (char *)("MGCMFD_nrnx"));
		for (i = 0; i < _ny; i++) {
			MGCMFD_nrnx[i] = _nx;
		}
	}

#if defined(M_CG_CMFD)

	_dimNode = MOC_NumPinsInSubDom * NumCoarseCMFDGroup;
	_dimSurfX = MOCNumLayersZ * (MOCNumPinsInSubDom_X + 1) * MOCNumPinsInSubDom_Y * NumCoarseCMFDGroup;
	_dimSurfY = MOCNumLayersZ * MOCNumPinsInSubDom_X * (MOCNumPinsInSubDom_Y + 1) * NumCoarseCMFDGroup;
	_dimSurfZ = (MOCNumLayersZ + 1) * MOCNumPinsInSubDom_X * MOCNumPinsInSubDom_Y * NumCoarseCMFDGroup;

	CGCMFDHetScalFlux = AllocMem_FP_I_1(_dimNode, (char *)("CGCMFDHetScalFlux"));
	CGCMFDScalarFlux = AllocMem_FP_I_1(_dimNode, (char *)("CGCMFDScalarFlux"));
	CGCMFDScalarFlux_Old = AllocMem_FP_I_1(_dimNode, (char *)("CGCMFDScalarFlux_Old"));
	if (CGCMFD_LinearSolver == 0 || CGCMFD_LinearSolver == 1) {
	CGCMFDScalarFlux_Old_LS = AllocMem_FP_I_1(_dimNode, (char *)("CGCMFDScalarFlux_Old_LS"));
	}

	CGCMFDTotalSource = AllocMem_FP_I_1(_dimNode, (char *)("CGCMFDTotalSource"));
	CGCMFDXs_t = AllocMem_FP_I_1(_dimNode, (char *)("CGCMFDXs_t"));
	CGCMFDXs_nf = AllocMem_FP_I_1(_dimNode, (char *)("CGCMFDXs_nf"));
	CGCMFD_Chi = AllocMem_FP_I_1(_dimNode, (char *)("CGCMFD_Chi"));
	CGCMFDXs_s = AllocMem_FP_I_1(_dimNode * NumCoarseCMFDGroup, (char *)("CGCMFDXs_s"));
	CGCMFDNodeDiffCoeff = AllocMem_FP_I_1(_dimNode, (char *)("CGCMFDNodeDiffCoeff"));
	CGCMFDSurfDiffCoeff_X = AllocMem_FP_I_1(_dimSurfX, (char *)("CGCMFDSurfDiffCoeff_X"));
	CGCMFDSurfDiffCoeff_Y = AllocMem_FP_I_1(_dimSurfY, (char *)("CGCMFDSurfDiffCoeff_Y"));
	CGCMFDSurfDiffCoeff_Z = AllocMem_FP_I_1(_dimSurfZ, (char *)("CGCMFDSurfDiffCoeff_Z"));
	CGCMFDSurfNDF_X = AllocMem_FP_I_1(_dimSurfX, (char *)("CGCMFDSurfNDF_X"));
	CGCMFDSurfNDF_Y = AllocMem_FP_I_1(_dimSurfY, (char *)("CGCMFDSurfNDF_Y"));
	CGCMFDSurfNDF_Z = AllocMem_FP_I_1(_dimSurfZ, (char *)("CGCMFDSurfNDF_Z"));
	CGCMFDNetCurr_X = AllocMem_FP_I_1(_dimSurfX, (char *)("CGCMFDNetCurr_X"));
	CGCMFDNetCurr_Y = AllocMem_FP_I_1(_dimSurfY, (char *)("CGCMFDNetCurr_Y"));
	CGCMFDNetCurr_Z = AllocMem_FP_I_1(_dimSurfZ, (char *)("CGCMFDNetCurr_Z"));

	CGCMFD_TotFissQ = AllocMem_FP_I_1(MOC_NumPinsInSubDom, (char *)("CGCMFD_TotFissQ"));
	CGCMFD_TotFissQ_old = AllocMem_FP_I_1(MOC_NumPinsInSubDom, (char *)("CGCMFD_TotFissQ_old"));
	
	CGCMFD_FluxFlag = AllocMem_i32_i32(1, (char *)("CGCMFD_FluxFlag"));
	CGCMFD_NegFluxCount = AllocMem_float_I(1, (char *)("CGCMFD_NegFluxCount"));
	CGCMFD_fissQflagl2 = AllocMem_i32_i32(1, (char *)("CGCMFD_fissQflagl2"));
	CGCMFD_fissQflaglinf = AllocMem_i32_i32(1, (char *)("CGCMFD_fissQflaglinf"));
	CGCMFD_flageig = AllocMem_i32_i32(1, (char *)("CGCMFD_flageig"));
	CGCMFD_fissQerrlinf = AllocMem_FP_I_1(NumProcs, (char *)("CGCMFD_fissQerrlinf"));
	CGCMFD_fissQrerrl2 = AllocMem_FP_I_1(NumProcs, (char *)("CGCMFD_fissQrerrl2"));
	CGCMFD_fluxErrlinf = AllocMem_FP_I_1(1, (char *)("CGCMFD_fluxErrlinf"));

	CGCMFD_fluxNormInf = AllocMem_FP_I((size_t)(1), (char *)("CGCMFD_fluxNormInf"));
	CGCMFD_fluxNorm2 = AllocMem_FP_I((size_t)(1), (char *)("CGCMFD_fluxNorm2"));
	CGCMFD_fluxRMSE = AllocMem_FP_I((size_t)(1), (char *)("CGCMFD_fluxRMSE"));
	CGCMFD_boolFluxNormInf = AllocMem_i32_I((size_t)(1), (char *)("CGCMFD_boolFluxNormInf"));
	CGCMFD_boolFluxNorm2 = AllocMem_i32_I((size_t)(1), (char *)("CGCMFD_boolFluxNorm2"));
	CGCMFD_boolFluxRMSE = AllocMem_i32_I((size_t)(1), (char *)("CGCMFD_boolFluxRMSE"));

	CGCMFD_LSFluxRes = AllocMem_FP_I((size_t)(1), (char *)("MGCMFD_LSFluxRes"));

#if defined(MPI_USED)
	CGCMFDData_MPIComm_SumInd = AllocMem_i32_i32(6, (char *)("CGCMFDData_MPIComm_SumInd"));
	CGCMFDData_MPIComm_Num = GetDataNum_CMFDCommPara(6, MOCNumPinsInSubDom_X, MOCNumPinsInSubDom_Y, MOCNumLayersZ, NumCoarseCMFDGroup, CGCMFDData_MPIComm_SumInd);
	CGCMFDComm_MPISend = AllocMem_FP_I_1(CGCMFDData_MPIComm_Num, (char *)("CGCMFDComm_MPISend"));
	CGCMFDFlux_MPIRecv = AllocMem_FP_I_1(CGCMFDData_MPIComm_Num, (char *)("CGCMFDFlux_MPIRecv"));
	CGCMFDNodeD_MPIRecv = AllocMem_FP_I_1(CGCMFDData_MPIComm_Num, (char *)("CGCMFDNodeD_MPIRecv"));
	CGCMFDMPI_InBoundSourCoeff = AllocMem_FP_I_1(CGCMFDData_MPIComm_Num, (char *)("CGCMFDMPI_InBoundSourCoeff"));
	CGCMFDMPI_InBoundSourCoeff_Index = AllocMem_i32_i32(CGCMFDData_MPIComm_Num, (char *)("CGCMFDMPI_InBoundSourCoeff_Index"));
	CGCMFDMPI_InBoundSourCoeff_Dir = AllocMem_i32_i32(CGCMFDData_MPIComm_Num, (char *)("CGCMFDMPI_InBoundSourCoeff_Dir"));
	CGCMFDMPI_InBoundSourCoeff_GeoDim = AllocMem_FP_I_1(CGCMFDData_MPIComm_Num, (char *)("CGCMFDMPI_InBoundSourCoeff_GeoDim"));
    CGCMFDComm_NumData = AllocMem_i32_i32(6, (char *)("CGCMFDComm_NumData"));
	CGCMFDData_MPIComm_Index = AllocMem_i32_i32(CGCMFDData_MPIComm_Num, (char *)("CGCMFDData_MPIComm_Index"));

#endif //
#endif //defined(M_CG_CMFD)
	MemParaCountBoundUp[MemPrintCount] = ParamCounter;

	if (MGCMFD_LinearSolver == 2) {
		_dimNode = MOC_NumPinsInSubDom * NumCMFDGroup;
		/* Allocate the space for the upper Hessenburg matrix*/
		MG_HessenburgMatrix = AllocMem_FP_I_1((GmresKrylovDim_MG + 1) * GmresKrylovDim_MG, (char *)("MG_HessenburgMatrix"));
		/*The Givens rotations include the sine and cosine term. */
		/*The cosine term is in column zero, and the sine term is in column one*/
		MG_GivensRotations = AllocMem_FP_I_1((GmresKrylovDim_MG + 1) * 2, (char *)("MG_GivensRotations"));
		/*The vector MG_HessenburgRHSVec the right hand side for the system that the matrix */
		/*H satisfies in order to minimize the residual over the Krylov subspace*/
		MG_HessenburgRHSVec = AllocMem_FP_I_1(GmresKrylovDim_MG + 1, (char *)("MG_HessenburgRHSVec"));
		/*Allocate the space for the orthogonal basis vectors for the Krylov subspace*/
		MG_KrylovOrthBasisVec = AllocMem_FP_I_1((GmresKrylovDim_MG + 1) * _dimNode, (char *)("MG_KrylovOrthBasisVec"));
		/*residual*/
		MG_Residual = AllocMem_FP_I_1(_dimNode, (char *)("MG_Residual"));

		MG_IW = AllocMem_i32_i32(_dimNode, (char *)("MG_DiagPointer"));

		MG_TemnpVec_1 = AllocMem_FP_I_1(_dimNode, (char *)("MG_TemnpVec_1"));
		MG_TemnpVec_2 = AllocMem_FP_I_1(GmresKrylovDim_MG + 1, (char *)("MG_TemnpVec_2"));
#if defined(M_CG_CMFD)
		if (CGCMFD_LinearSolver == 1) {
			_dimNode = MOC_NumPinsInSubDom * NumCoarseCMFDGroup;
			CG_HessenburgMatrix = AllocMem_FP_I_1((GmresKrylovDim_CG + 1) * GmresKrylovDim_CG, (char *)("CG_HessenburgMatrix"));
			CG_GivensRotations = AllocMem_FP_I_1((GmresKrylovDim_CG + 1) * 2, (char *)("CG_GivensRotations"));
			CG_HessenburgRHSVec = AllocMem_FP_I_1(GmresKrylovDim_CG + 1, (char *)("CG_HessenburgRHSVec"));
			CG_KrylovOrthBasisVec = AllocMem_FP_I_1((GmresKrylovDim_CG + 1) * _dimNode, (char *)("CG_KrylovOrthBasisVec"));
			CG_Residual = AllocMem_FP_I_1(_dimNode, (char *)("CG_Residual"));

			CG_IW = AllocMem_i32_i32(_dimNode, (char *)("CG_DiagPointer"));
			CG_TemnpVec_2 = AllocMem_FP_I_1(GmresKrylovDim_CG + 1, (char *)("CG_TemnpVec_2"));
	    }
#endif // defined(M_CG_CMFD)
	}
#endif // defined(M_MG_CMFD)

#if defined(OMP_USED)
    if ((FSR_OMP_lock = (omp_lock_t *)calloc(TotalNumFSRs_subDom, sizeof(omp_lock_t))) == NULL) {
		printf("\n [**ERROR**]  Memory Allocation Failed in \"FSR_OMP_lock\"\n");
		fflush(stdout);
		exit(69);
	}
    
	/* Loop over all FSRs to initialize OpenMP locks */
#pragma omp parallel for schedule(guided) num_threads(NumOMPThread)
    for (i=0; i < TotalNumFSRs_subDom; i++)
      omp_init_lock(&FSR_OMP_lock[i]);
#endif 

	if (_procID == 0) {
		printf("\n [  INFO   ]  %-*s%-*s%-*s%-*s", _width1, "# of Pins", _width1, "# of FSRs", _width1, "# of LTs", _width1, "# of Segs");
		printf("\n                %-*d  %-*d  %-*d  %-*llu", _width1 - 2, MOC_NumPinsInSubDom, _width1 - 2, TotalNumFSRs_subDom, \
			_width1 - 2, TotalNumLTs_SubDom, _width1 - 2, TotalNumSeg);
		printf("\n [  INFO   ]  %-*s%-*s%-*s%-*s", _width1, "# of Groups", _width1, "# of Azim", _width1, "# of Pol", _width1, "Spacing (cm)");
		printf("\n                %-*d  %-*d  %-*d  %-*.3lf", _width1 - 2, NumGroups, _width1 - 2, NumAzimAng[0], \
			_width1 - 2, NumPolarAng[0], _width1 - 2, RaySpacingIni[0]);
		printf("\n [  INFO   ]  %-*s%-*s%-*s", _width1, "MOC MaxIter", _width1, "Keff tol", _width1, "Flux Tol");
		printf("\n              %-*d%-*.1e%-*.1e", _width1, MaxNumOuterIter_MOC, _width1, ConvCritK, _width1, ConvCritFlux);

		fprintf(LogFile, "\n [  INFO   ]  %-*s%-*s%-*s%-*s", _width1, "# of Pins", _width1, "# of FSRs", _width1, "# of LTs", _width1, "# of Segs");
		fprintf(LogFile, "\n                %-*d  %-*d  %-*d  %-*llu", _width1 - 2, MOC_NumPinsInSubDom, _width1 - 2, TotalNumFSRs_subDom, \
			_width1 - 2, TotalNumLTs_SubDom, _width1 - 2, TotalNumSeg);
		fprintf(LogFile, "\n [  INFO   ]  %-*s%-*s%-*s%-*s", _width1, "# of Groups", _width1, "# of Azim", _width1, "# of Pol", _width1, "Spacing (cm)");
		fprintf(LogFile, "\n                %-*d  %-*d  %-*d  %-*.3lf", _width1 - 2, NumGroups, _width1 - 2, NumAzimAng[0], \
			_width1 - 2, NumPolarAng[0], _width1 - 2, RaySpacingIni[0]);
		fprintf(LogFile, "\n [  INFO   ]  %-*s%-*s%-*s", _width1, "MOC MaxIter", _width1, "Keff tol", _width1, "Flux Tol");
		fprintf(LogFile, "\n              %-*d%-*.1e%-*.1e", _width1, MaxNumOuterIter_MOC, _width1, ConvCritK, _width1, ConvCritFlux);
#if defined(M_MG_CMFD)
		if (MGCMFD_LinearSolver == 0) {
			sprintf(_lsSolver_MG, "JACOBI");
		}
		else if (MGCMFD_LinearSolver == 1) {
			sprintf(_lsSolver_MG, "SOR");
		}
		else if (MGCMFD_LinearSolver == 2) {
			if (GmresPrecond_MG == 1) {
				sprintf(_lsSolver_MG, "GMRES");
			}
			else {
				sprintf(_lsSolver_MG, "PGMRES");
			}
		}
		else if (MGCMFD_LinearSolver == 3) {
			sprintf(_lsSolver_MG, "ZYQ");
		}
		else {}

#if defined(M_CG_CMFD)
		if (CGCMFD_LinearSolver == 0) {
			sprintf(_lsSolver_CG, "SOR");
		}
		else if (CGCMFD_LinearSolver == 1) {
			sprintf(_lsSolver_CG, "SOR");
		}
		else if (CGCMFD_LinearSolver == 2) {
			if (GmresPrecond_CG == 1) {
				sprintf(_lsSolver_CG, "GMRES");
			}
			else {
				sprintf(_lsSolver_CG, "PGMRES");
			}
		}
		else if (CGCMFD_LinearSolver == 3) {
			sprintf(_lsSolver_CG, "ZYQ");
		}
		else {}
#endif // defined(M_CG_CMFD)

		printf("\n [  INFO   ]  %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s", _width1, "CMFD", _width1, "LS Solver", _width1, "MaxCMFDIter", \
			_width1, "MaxLSIter", _width1, "Eig tol", _width1, "QFissl2", _width1, "QFisslinf", _width1, "LS Tol");
		printf("\n              %-*s%-*s%-*d%-*d%-*.1e%-*.1e%-*.1e%-*.1e", _width1, "MG ON", _width1, _lsSolver_MG, _width1, MaxNumOuterIter_MGCMFD, \
			_width1, MaxNumIter_MGLS, _width1, MGCMFD_epseig, _width1, MGCMFD_epsl2, _width1, MGCMFD_epslinf, _width1, ConvCritLinearSolver_MGCMFD);

		fprintf(LogFile, "\n [  INFO   ]  %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s", _width1, "CMFD", _width1, "LS Solver", _width1, "MaxCMFDIter", \
			_width1, "MaxLSIter", _width1, "Eig tol", _width1, "QFissl2", _width1, "QFisslinf", _width1, "LS Tol");
		fprintf(LogFile, "\n              %-*s%-*s%-*d%-*d%-*.1e%-*.1e%-*.1e%-*.1e", _width1, "MG ON", _width1, _lsSolver_MG, _width1, MaxNumOuterIter_MGCMFD, \
			_width1, MaxNumIter_MGLS, _width1, MGCMFD_epseig, _width1, MGCMFD_epsl2, _width1, MGCMFD_epslinf, _width1, ConvCritLinearSolver_MGCMFD);

#if defined(M_CG_CMFD)
		printf("\n              %-*s%-*s%-*d%-*d%-*.1e%-*.1e%-*.1e%-*.1e", _width1, "CG ON", _width1, _lsSolver_CG, _width1, MaxNumOuterIter_CGCMFD, \
			_width1, MaxNumIter_CGLS, _width1, CGCMFD_epseig, _width1, CGCMFD_epsl2, _width1, CGCMFD_epslinf, _width1, ConvCritLinearSolver_CGCMFD);
		fprintf(LogFile, "\n              %-*s%-*s%-*d%-*d%-*.1e%-*.1e%-*.1e%-*.1e", _width1, "CG ON", _width1, _lsSolver_CG, _width1, MaxNumOuterIter_CGCMFD, \
			_width1, MaxNumIter_CGLS, _width1, CGCMFD_epseig, _width1, CGCMFD_epsl2, _width1, CGCMFD_epslinf, _width1, ConvCritLinearSolver_CGCMFD);
#endif // defined(M_CG_CMFD)

		if (MGCMFD_LinearSolver == 1) {
			printf("\n              %-*s%-*s", _width1, "", _width1, "SOR Factor");
			printf("\n              %-*s%-*.2lf", _width1, "MG CMFD", _width1, SOR_Factor_MG);
			fprintf(LogFile, "\n              %-*s%-*s", _width1, "", _width1, "SOR Factor");
			fprintf(LogFile, "\n              %-*s%-*.2lf", _width1, "MG CMFD", _width1, SOR_Factor_MG);
		}
		//#if defined(M_CG_CMFD) && defined(M_CGLS_SOR)
		//		printf("\n              %-*s%-*.2lf", _width1, "CG CMFD", _width1, SOR_Factor_CG);
		//		fprintf(LogFile, "\n              %-*s%-*.2lf", _width1, "MG CMFD", _width1, SOR_Factor_CG);
		//#endif // defined(M_CG_CMFD) && defined(M_CGLS_SOR)


		if (MGCMFD_LinearSolver == 1) {
			printf("\n              %-*s%-*s%-*s", _width1, "", _width1, "MaxKryDim", _width1, "Precond");
			Printf_GMRESPrecondition(GmresPrecond_MG, _gmresPrec);
			printf("\n              %-*s%-*d%-*s", _width1, "MG CMFD", _width1, GmresKrylovDim_MG, _width1, _gmresPrec);
			fprintf(LogFile, "\n              %-*s%-*s%-*s", _width1, "", _width1, "MaxKryDim", _width1, "Precond");
			fprintf(LogFile, "\n              %-*s%-*d%-*s", _width1, "MG CMFD", _width1, GmresKrylovDim_MG, _width1, _gmresPrec);
#if defined(M_CG_CMFD)
			if (CGCMFD_LinearSolver == 2) {
				Printf_GMRESPrecondition(GmresPrecond_CG, _gmresPrec);
				printf("\n              %-*s%-*d%-*s", _width1, "CG CMFD", _width1, GmresKrylovDim_CG, _width1, _gmresPrec);
				fprintf(LogFile, "\n              %-*s%-*d%-*s", _width1, "CG CMFD", _width1, GmresKrylovDim_CG, _width1, _gmresPrec);
		    }
#endif // defined(M_CG_CMFD)
		}

#endif // defined(M_MG_CMFD)

		/*printf memory info*/
// 		if (ProcID==0) {
// 		printf("\n [  INFO   ]  Memory Usage on CPU");
// 		printf("\n              %-*s%-*s", 30, "Parameter", 18, "Memory (MB)");
// 		fflush(stdout);
// 		fprintf(LogFile, "\n [  INFO   ]  Memory Usage on CPU");
// 		fprintf(LogFile, "\n              %-*s%-*s", 30, "Parameter", 18, "Memory (MB)");
// 		for (i = 0; i < MemPrintCount + 1; i++) {
// 			_mem = 0.0;
// 			for (j = MemParaCountBoundLow[i]; j < MemParaCountBoundUp[i]; j++) {
// 				_mem += ParaMemSize[j];
// 			}
// 			printf("\n              %-*s%-*.1lf", 30, MemParaName[i], 12, _mem);
// 			fflush(stdout);
// 			fprintf(LogFile, "\n              %-*s%-*.1lf", 30, MemParaName[i], 12, _mem);
// 		}
// #if !defined(M_MG_CMFD) || defined(CUDA_USED)
// 		printf("\n              %-*s%-*.1lf", 30, "Total Mem Usage", 12, NumTotalData);
// 		fflush(stdout);
// 		fprintf(LogFile, "\n              %-*s%-*.1lf", 30, "Total Mem Usage", 12, NumTotalData);
// #endif //!defined(M_MG_CMFD) || defined(CUDA_USED)
//         }
	}

#ifdef TIME_REC
	_time_end = TallyCurrentTime_CPU();
	TimePrePart[TimeCountPrePart] = _time_end - _time_start;
	strcpy(TimeMetricsPrePart[TimeCountPrePart], "Allocate Mem on CPU");
	TimeCountPrePart++;
#endif // TIME_REC
}

void FreeMemory()
{
	if (ProcID == 0) {
		printf("\n [  INFO   ]  Free Allocated Mem on CPU ... ");
		fflush(stdout);
		fprintf(LogFile, "\n [  INFO   ]  Free Allocated Mem on CPU ... ");
	}
	ScalarFlux_MOC = FreeMem_FPFlux_I(ScalarFlux_MOC, (char *)("ScalarFlux_MOC"));
	ScalarFluxOld = FreeMem_FPFlux_I(ScalarFluxOld, (char *)("ScalarFluxOld"));
	Source = FreeMem_FP_I(Source, (char *)("Source"));

// #ifdef OMP_USED
// 	ScalarFlux_OMP = FreeMem_FP_II(ScalarFlux_OMP,NumOMPThread, (char *)("ScalarFlux_OMP"));
// 	MOC_trackFlux_OMP = FreeMem_FP_III(MOC_trackFlux_OMP, NumOMPThread, NumGroups, (char *)("MOC_trackFlux_OMP"));
// #endif // OMP_USED

#if defined(MPI_USED)
    if (NumProcs > 1) {
		CPUMPIFlux = FreeMem_FP_II(CPUMPIFlux, NumTrackSet, (char *)("CPUMPIFlux"));
	}
#endif
	BoundaryFlux = FreeMem_FP_II(BoundaryFlux, NumTrackSet, (char *)("BoundaryFlux"));
#if !defined(M_UpdateRadialBoundImme)
	BoundaryFlux_Temp = FreeMem_FP_II(BoundaryFlux_Temp, NumTrackSet, (char *)("BoundaryFlux_Temp"));
#endif // !defined(M_UpdateRadialBoundImme)

    // SegAngFlux = FreeMem_FP_I(SegAngFlux, (char *)("SegAngFlux"));
    SegAngFlux = FreeMem_FP_II(SegAngFlux, NumOMPThread, (char *)("SegAngFlux"));
    ExpTempStore = FreeMem_FP_II(ExpTempStore, NumOMPThread, (char *)("ExpTempStore"));
#if defined(M_3DMOCex)
    MOC3D_oneOverSigma = FreeMem_FP_I(MOC3D_oneOverSigma, (char *)("MOC3D_oneOverSigma"));
    MOC3D_QtoSigma = FreeMem_FP_I(MOC3D_QtoSigma, (char *)("MOC3D_QtoSigma"));
	FSRAveAngFlux_Top = FreeMem_FPFlux_I(FSRAveAngFlux_Top, (char *)("FSRAveAngFlux_Top"));
	FSRAveAngFlux_Bott = FreeMem_FPFlux_I(FSRAveAngFlux_Bott, (char *)("FSRAveAngFlux_Bott"));
	TwoCosThetaOverHight_Sigma = FreeMem_FP_I(TwoCosThetaOverHight_Sigma, (char *)("TwoCosThetaOverHight_Sigma"));
	TwoCosThetaOverHight_Source = FreeMem_FP_I(TwoCosThetaOverHight_Source, (char *)("TwoCosThetaOverHight_Source"));
#if defined(M_3DAxialFluxSOR)
    FSRAveAngFlux_UpAllLay = FreeMem_FPFlux_I(FSRAveAngFlux_UpAllLay, (char *)("FSRAveAngFlux_UpAllLay"));
    FSRAveAngFlux_DownAllLay = FreeMem_FPFlux_I(FSRAveAngFlux_DownAllLay, (char *)("FSRAveAngFlux_DownAllLay"));
#endif // M_3DAxialFluxSOR
#if !defined(M_UpdateAxialBoundImme)
	FSRAveAngFlux_TopTemp = FreeMem_FPFlux_I(FSRAveAngFlux_TopTemp, (char *)("FSRAveAngFlux_TopTemp"));
#endif // !defined(M_UpdateAxialBoundImme)
#endif //defined(M_3DMOCex)
    if (OUTPUT_normPinPower == 1) {
		NormPinPowerSubDom = FreeMem_FP_I(NormPinPowerSubDom, (char *)("NormPinPowerSubDom"));
		NormPinPower_SumLay = FreeMem_FP_I(NormPinPower_SumLay, (char *)("NormPinPower_SumLay"));
		MOC_totPower_procs = FreeMem_FP_I(MOC_totPower_procs, (char *)("MOC_totPower_procs"));
	}

#ifdef OMP_ASYN
	CFlux = FreeMem_FP_II(CFlux, NumTrackSet, (char *)("CFlux"));
#endif // OMP_ASYN

#ifdef CUDA_USED
	CPUReacRateCollect = FreeMem_FP_I(CPUReacRateCollect, (char *)("CPUReacRateCollect"));
	GPUFluxResRColl = FreeMem_FP_I(GPUFluxResRColl, (char *)("GPUFluxResRColl"));
#endif // CUDA_USED

#ifdef MPI_USED
	Mess_TarProc = FreeMem_int_III(Mess_TarProc, MOCNumLayersZ, NumAzimAng[0], (char *)("Mess_TarProc"));
	Mess_SendNumLTs = FreeMem_int_III(Mess_SendNumLTs, MOCNumLayersZ, NumAzimAng[0],(char *)("Mess_SendNumLTs"));
	Mess_SendSatrtPInAngF = FreeMem_sizeT_III(Mess_SendSatrtPInAngF, MOCNumLayersZ, NumAzimAng[0], (char *)("Mess_SendSatrtPInAngF"));
	Mess_RecvSatrtPInAngF = FreeMem_sizeT_III(Mess_RecvSatrtPInAngF, MOCNumLayersZ, NumAzimAng[0], (char *)("Mess_RecvSatrtPInAngF"));
	// GCCopy_StartPInAngF = FreeMem_int_II(GCCopy_StartPInAngF, MOCNumLayersZ, (char *)("GCCopy_StartPInAngF"));
	// GCCopy_NumDataInAngF = FreeMem_int_II(GCCopy_NumDataInAngF, MOCNumLayersZ, (char *)("GCCopy_NumDataInAngF"));
	Mess_TarProc_1 = FreeMem_i32_I(Mess_TarProc_1, (char *)("Mess_TarProc_1"));
	Mess_SendNumLTs_1 = FreeMem_i32_I(Mess_SendNumLTs_1, (char *)("Mess_SendNumLTs_1"));
	BoundAngFlux_send = FreeMem_FP_II(BoundAngFlux_send, NumNeibSubDom, (char *)("BoundAngFlux_send"));
	BoundAngFlux_recv = FreeMem_FP_II(BoundAngFlux_recv, NumNeibSubDom, (char *)("BoundAngFlux_recv"));
	NextSubDomLTIDs_send = FreeMem_int_II(NextSubDomLTIDs_send, NumNeibSubDom, (char *)("NextSubDomLTIDs_send"));
	NextSubDomLTIDs_recv = FreeMem_int_II(NextSubDomLTIDs_recv, NumNeibSubDom, (char *)("NextSubDomLTIDs_recv"));
	MPIProcID_recv = FreeMem_int_III(MPIProcID_recv, MOCNumLayersZ, NumAzimAng[0], (char *)("MPIProcID_recv"));
	MPINeibID_recv = FreeMem_int_III(MPINeibID_recv, MOCNumLayersZ, NumAzimAng[0], (char *)("MPINeibID_recv"));
#endif //MPI_USED
    if (ProcID == 0) {
		MOC_fissRate_procs = FreeMem_FP_I(MOC_fissRate_procs, (char *)("MOC_fissRate_procs"));
		MOC_numFSRs_subDom = FreeMem_i32_I(MOC_numFSRs_subDom, (char *)("MOC_numFSRs_subDom"));
		MOC_keff = FreeMem_FPDouble_I(MOC_keff, (char *)("MOC_keff"));
	}

#if defined(OMP_USED) && defined(TIME_REC) && defined(TIME_REC_OMP)
	CompTime_omp = FreeMem_FP_I(CompTime_omp, (char *)("CompTime_omp"));
#endif //

#if defined(M_MG_CMFD)
	CMFD_NodeVol = FreeMem_FP_I(CMFD_NodeVol, (char *)("CMFD_NodeVol"));
	CMFDHetScalFlux = FreeMem_FP_I(CMFDHetScalFlux, (char *)("CMFDHetScalFlux"));
	CMFDScalarFlux = FreeMem_FP_I(CMFDScalarFlux, (char *)("CMFDScalarFlux"));
	CMFDScalarFlux_Old = FreeMem_FP_I(CMFDScalarFlux_Old, (char *)("CMFDScalarFlux_Old"));
	if (MGCMFD_LinearSolver == 0 || MGCMFD_LinearSolver == 1) {
		CMFDScalarFlux_Old_LS = FreeMem_FP_I(CMFDScalarFlux_Old_LS, (char *)("CMFDScalarFlux_Old_LS"));
	}

	CMFDTotalSource = FreeMem_FP_I(CMFDTotalSource, (char *)("CMFDTotalSource"));
	Index_1DArray = FreeMem_i32_I(Index_1DArray, (char *)("Index_1DArray"));
	CMFDXs_t = FreeMem_FP_I(CMFDXs_t, (char *)("CMFDXs_t"));
	if (OUTPUT_CMFDGroupConstant == 1 || OUTPUT_CoarseMesh_GroupConstant == 1) {
		CMFDXs_a = FreeMem_FP_I(CMFDXs_a, (char *)("CMFDXs_a"));
		CMFDXs_f = FreeMem_FP_I(CMFDXs_f, (char *)("CMFDXs_f"));
	}
	CMFDXs_nf = FreeMem_FP_I(CMFDXs_nf, (char *)("CMFDXs_nf"));
	CMFD_Chi = FreeMem_FP_I(CMFD_Chi, (char *)("CMFD_Chi"));
	CMFDXs_s = FreeMem_FP_I(CMFDXs_s, (char *)("CMFDXs_s"));
	CMFDNodeDiffCoeff = FreeMem_FP_I(CMFDNodeDiffCoeff, (char *)("CMFDNodeDiffCoeff"));
	CMFDSurfDiffCoeff_X = FreeMem_FP_I(CMFDSurfDiffCoeff_X, (char *)("CMFDSurfDiffCoeff_X"));
	CMFDSurfDiffCoeff_Y = FreeMem_FP_I(CMFDSurfDiffCoeff_Y, (char *)("CMFDSurfDiffCoeff_Y"));
	CMFDSurfDiffCoeff_Z = FreeMem_FP_I(CMFDSurfDiffCoeff_Z, (char *)("CMFDSurfDiffCoeff_Z"));
	CMFDSurfNDF_X = FreeMem_FP_I(CMFDSurfNDF_X, (char *)("CMFDSurfNDF_X"));
	CMFDSurfNDF_Y = FreeMem_FP_I(CMFDSurfNDF_Y, (char *)("CMFDSurfNDF_Y"));
	CMFDSurfNDF_Z = FreeMem_FP_I(CMFDSurfNDF_Z, (char *)("CMFDSurfNDF_Z"));
#if defined(M_YQZ_CMFD_Damp)
	CMFDSurfNDF_XOld = FreeMem_FP_I(CMFDSurfNDF_XOld, (char *)("CMFDSurfNDF_XOld"));
	CMFDSurfNDF_YOld = FreeMem_FP_I(CMFDSurfNDF_YOld, (char *)("CMFDSurfNDF_YOld"));
	CMFDSurfNDF_ZOld = FreeMem_FP_I(CMFDSurfNDF_ZOld, (char *)("CMFDSurfNDF_ZOld"));
#endif // defined(M_YQZ_CMFD_Damp)

#if defined(MPI_USED)
	MGCMFDData_MPIComm_SumInd = FreeMem_i32_I(MGCMFDData_MPIComm_SumInd, (char *)("MGCMFDData_MPIComm_SumInd"));
	MGCMFDComm_MPISend = FreeMem_FP_I(MGCMFDComm_MPISend, (char *)("MGCMFDComm_MPISend"));
	MGCMFDFlux_MPIRecv = FreeMem_FP_I(MGCMFDFlux_MPIRecv, (char *)("MGCMFDFlux_MPIRecv"));
	MGCMFDNodeD_MPIRecv = FreeMem_FP_I(MGCMFDNodeD_MPIRecv, (char *)("MGCMFDNodeD_MPIRecv"));
	CMFDComm_TarProc = FreeMem_i32_I(CMFDComm_TarProc, (char *)("CMFDComm_TarProc"));
	MGCMFDComm_NumData = FreeMem_i32_I(MGCMFDComm_NumData, (char *)("MGCMFDComm_NumData"));
	CMFDComm_DirBound = FreeMem_int_III(CMFDComm_DirBound, 6, 3, (char *)("CMFDComm_DirBound"));
	CMFDComm_RecvDir = FreeMem_i32_I(CMFDComm_RecvDir, (char *)("CMFDComm_RecvDir"));
	MGCMFDData_MPIComm_Index = FreeMem_i32_I(MGCMFDData_MPIComm_Index, (char *)("MGCMFDData_MPIComm_Index"));
	MGCMFDMPI_InBoundSourCoeff = FreeMem_FP_I(MGCMFDMPI_InBoundSourCoeff, (char *)("MGCMFDMPI_InBoundSourCoeff"));
	MGCMFD_MOCParCurrInnerBound_Index = FreeMem_i32_I(MGCMFD_MOCParCurrInnerBound_Index, (char *)("MGCMFD_MOCParCurrInnerBound_Index"));
	MGCMFD_MOCParCurrInnerBound_Dir = FreeMem_i32_I(MGCMFD_MOCParCurrInnerBound_Dir, (char *)("MGCMFD_MOCParCurrInnerBound_Dir"));
	MGCMFDMPI_InBoundSourCoeff_Index = FreeMem_i32_I(MGCMFDMPI_InBoundSourCoeff_Index, (char *)("MGCMFDMPI_InBoundSourCoeff_Index"));
	MGCMFDMPI_InBoundSourCoeff_Dir = FreeMem_i32_I(MGCMFDMPI_InBoundSourCoeff_Dir, (char *)("MGCMFDMPI_InBoundSourCoeff_Dir"));
	MGCMFDMPI_InBoundSourCoeff_GeoDim = FreeMem_FP_I(MGCMFDMPI_InBoundSourCoeff_GeoDim, (char *)("MGCMFDMPI_InBoundSourCoeff_GeoDim"));
#endif //defined(MPI_USED)
	MOCParCurr_X_Pos = FreeMem_FP_I(MOCParCurr_X_Pos, (char *)("MOCParCurr_X_Pos"));
	MOCParCurr_X_Neg = FreeMem_FP_I(MOCParCurr_X_Neg, (char *)("MOCParCurr_X_Neg"));
	MOCParCurr_Y_Pos = FreeMem_FP_I(MOCParCurr_Y_Pos, (char *)("MOCParCurr_Y_Pos"));
	MOCParCurr_Y_Neg = FreeMem_FP_I(MOCParCurr_Y_Neg, (char *)("MOCParCurr_Y_Neg"));
	MOC_NetCurr_Z = FreeMem_FP_I(MOC_NetCurr_Z, (char *)("MOC_NetCurr_Z"));
	MOCtoCMFD_ParCurr_X_Pos = FreeMem_FP_I(MOCtoCMFD_ParCurr_X_Pos, (char *)("MOCtoCMFD_ParCurr_X_Pos"));
	MOCtoCMFD_ParCurr_X_Neg = FreeMem_FP_I(MOCtoCMFD_ParCurr_X_Neg, (char *)("MOCtoCMFD_ParCurr_X_Neg"));
	MOCtoCMFD_ParCurr_Y_Pos = FreeMem_FP_I(MOCtoCMFD_ParCurr_Y_Pos, (char *)("MOCtoCMFD_ParCurr_Y_Pos"));
	MOCtoCMFD_ParCurr_Y_Neg = FreeMem_FP_I(MOCtoCMFD_ParCurr_Y_Neg, (char *)("MOCtoCMFD_ParCurr_Y_Neg"));
	MGCMFDNetCurr_X = FreeMem_FP_I(MGCMFDNetCurr_X, (char *)("MGCMFDNetCurr_X"));
	MGCMFDNetCurr_Y = FreeMem_FP_I(MGCMFDNetCurr_Y, (char *)("MGCMFDNetCurr_Y"));
	MGCMFDNetCurr_Z = FreeMem_FP_I(MGCMFDNetCurr_Z, (char *)("MGCMFDNetCurr_Z"));

	MatrixCSR = FreeMem_FP_I(MatrixCSR, (char *)("MatrixCSR"));
	if (MGCMFD_LinearSolver == 2) {
		MG_ILUMatrix = FreeMem_FP_I(MG_ILUMatrix, (char *)("MG_ILUMatrix"));
	}
	MatrixCSR_RowStartInd = FreeMem_i32_I(MatrixCSR_RowStartInd, (char *)("MatrixCSR_RowStartInd"));
	MatrixCSR_ColInd = FreeMem_i32_I(MatrixCSR_ColInd, (char *)("MatrixCSR_ColInd"));
	MG_DiagPointer = FreeMem_i32_I(MG_DiagPointer, (char *)("MG_DiagPointer"));

#if defined(PTESC_LIB) || !defined(DIRECT_CSR)
	CMFDMatrix_A = FreeMem_FP_I(CMFDMatrix_A, (char *)("CMFDMatrix_A"));
#endif //defined(PTESC_LIB) || !defined(DIRECT_CSR)

	MGCMFD_FluxFlag = FreeMem_i32_I(MGCMFD_FluxFlag, (char *)("MGCMFD_FluxFlag"));
	MGCMFD_NegFluxCount = FreeMem_float_I(MGCMFD_NegFluxCount, (char *)("MGCMFD_NegFluxCount"));
	MGCMFD_fissQflagl2 = FreeMem_i32_I(MGCMFD_fissQflagl2, (char *)("MGCMFD_fissQflagl2"));
	MGCMFD_fissQflaglinf = FreeMem_i32_I(MGCMFD_fissQflaglinf, (char *)("MGCMFD_fissQflaglinf"));
	MGCMFD_flageig = FreeMem_i32_I(MGCMFD_flageig, (char *)("MGCMFD_flageig"));
	MGCMFD_fissQerrlinf = FreeMem_FP_I(MGCMFD_fissQerrlinf, (char *)("MGCMFD_fissQerrlinf"));
	MGCMFD_fissQrerrl2 = FreeMem_FP_I(MGCMFD_fissQrerrl2, (char *)("MGCMFD_fissQrerrl2"));
	MGCMFD_fluxErrlinf = FreeMem_FP_I(MGCMFD_fluxErrlinf, (char *)("MGCMFD_fluxErrlinf"));
	MGCMFD_TotFissQ = FreeMem_FP_I(MGCMFD_TotFissQ, (char *)("MGCMFD_TotFissQ"));
	MGCMFD_TotFissQ_old = FreeMem_FP_I(MGCMFD_TotFissQ_old, (char *)("MGCMFD_TotFissQ_old"));

	if (MGCMFD_LinearSolver == 3) {
		MGCMFD_CCW = FreeMem_FP_I(MGCMFD_CCW, (char *)("MGCMFD_CCW"));
		MGCMFD_CCN = FreeMem_FP_I(MGCMFD_CCN, (char *)("MGCMFD_CCN"));
		MGCMFD_CCE = FreeMem_FP_I(MGCMFD_CCE, (char *)("MGCMFD_CCE"));
		MGCMFD_CCS = FreeMem_FP_I(MGCMFD_CCS, (char *)("MGCMFD_CCS"));
		MGCMFD_CCB = FreeMem_FP_I(MGCMFD_CCB, (char *)("MGCMFD_CCB"));
		MGCMFD_CCT = FreeMem_FP_I(MGCMFD_CCT, (char *)("MGCMFD_CCT"));
		MGCMFD_AM = FreeMem_FP_I(MGCMFD_AM, (char *)("MGCMFD_AM"));
		MGCMFD_Del = FreeMem_FP_I(MGCMFD_Del, (char *)("MGCMFD_Del"));
		MGCMFD_Al = FreeMem_FP_I(MGCMFD_Al, (char *)("MGCMFD_Al"));
		MGCMFD_Au = FreeMem_FP_I(MGCMFD_Au, (char *)("MGCMFD_Au"));
		MGCMFD_Delinv = FreeMem_FP_I(MGCMFD_Delinv, (char *)("MGCMFD_Delinv"));
		MGCMFD_Deliau = FreeMem_FP_I(MGCMFD_Deliau, (char *)("MGCMFD_Deliau"));
		MGCMFD_Ainvl = FreeMem_FP_I(MGCMFD_Ainvl, (char *)("MGCMFD_Ainvl"));
		MGCMFD_Ainvu = FreeMem_FP_I(MGCMFD_Ainvu, (char *)("MGCMFD_Ainvu"));
		MGCMFD_Ainvd = FreeMem_FP_I(MGCMFD_Ainvd, (char *)("MGCMFD_Ainvd"));
		MGCMFD_TotSour_g = FreeMem_FP_I(MGCMFD_TotSour_g, (char *)("MGCMFD_TotSour_g"));
		MGCMFD_flux_in = FreeMem_FP_I(MGCMFD_flux_in, (char *)("MGCMFD_flux_in"));
		MGCMFD_flux_add_in = FreeMem_FP_I(MGCMFD_flux_add_in, (char *)("MGCMFD_flux_add_in"));
		MGCMFD_src_in = FreeMem_FP_I(MGCMFD_src_in, (char *)("MGCMFD_src_in"));
		MGCMFD_vr0_mg = FreeMem_FP_I(MGCMFD_vr0_mg, (char *)("MGCMFD_vr0_mg"));
		MGCMFD_vr_mg = FreeMem_FP_I(MGCMFD_vr_mg, (char *)("MGCMFD_vr_mg"));
		MGCMFD_vp_mg = FreeMem_FP_I(MGCMFD_vp_mg, (char *)("MGCMFD_vp_mg"));
		MGCMFD_vv_mg = FreeMem_FP_I(MGCMFD_vv_mg, (char *)("MGCMFD_vv_mg"));
		MGCMFD_vs_mg = FreeMem_FP_I(MGCMFD_vs_mg, (char *)("MGCMFD_vs_mg"));
		MGCMFD_vt_mg = FreeMem_FP_I(MGCMFD_vt_mg, (char *)("MGCMFD_vt_mg"));
		MGCMFD_vy_mg = FreeMem_FP_I(MGCMFD_vy_mg, (char *)("MGCMFD_vy_mg"));
		MGCMFD_vz_mg = FreeMem_FP_I(MGCMFD_vz_mg, (char *)("MGCMFD_vz_mg"));
		MGCMFD_s_mg = FreeMem_FP_I(MGCMFD_s_mg, (char *)("MGCMFD_s_mg"));
		MGCMFD_b0_mg = FreeMem_FP_I(MGCMFD_b0_mg, (char *)("MGCMFD_b0_mg"));
		MGCMFD_s1dl_mg = FreeMem_FP_I(MGCMFD_s1dl_mg, (char *)("MGCMFD_s1dl_mg"));
		MGCMFD_b01d_mg = FreeMem_FP_I(MGCMFD_b01d_mg,(char *)("MGCMFD_b01d_mg"));
		MGCMFD_y_mg = FreeMem_FP_I(MGCMFD_y_mg, (char *)("MGCMFD_y_mg"));
		MGCMFD_nrnx = FreeMem_i32_I(MGCMFD_nrnx, (char *)("MGCMFD_nrnx"));
	}

#if defined(M_CG_CMFD)
	CGCMFDHetScalFlux = FreeMem_FP_I(CGCMFDHetScalFlux, (char *)("CGCMFDHetScalFlux"));
	CGCMFDScalarFlux = FreeMem_FP_I(CGCMFDScalarFlux, (char *)("CGCMFDScalarFlux"));
	CGCMFDScalarFlux_Old = FreeMem_FP_I(CGCMFDScalarFlux_Old, (char *)("CGCMFDScalarFlux_Old"));
	if (CGCMFD_LinearSolver == 0 || CGCMFD_LinearSolver == 1) {
	    CGCMFDScalarFlux_Old_LS = FreeMem_FP_I(CGCMFDScalarFlux_Old_LS, (char *)("CGCMFDScalarFlux_Old_LS"));
	}
	CGCMFDTotalSource = FreeMem_FP_I(CGCMFDTotalSource, (char *)("CGCMFDTotalSource"));
	CGCMFDXs_t = FreeMem_FP_I(CGCMFDXs_t, (char *)("CGCMFDXs_t"));
	CGCMFDXs_nf = FreeMem_FP_I(CGCMFDXs_nf, (char *)("CGCMFDXs_nf"));
	CGCMFD_Chi = FreeMem_FP_I(CGCMFD_Chi, (char *)("CGCMFD_Chi"));
	CGCMFDXs_s = FreeMem_FP_I(CGCMFDXs_s, (char *)("CGCMFDXs_s"));
	CGCMFDNodeDiffCoeff = FreeMem_FP_I(CGCMFDNodeDiffCoeff, (char *)("CGCMFDNodeDiffCoeff"));
	CGCMFDSurfDiffCoeff_X = FreeMem_FP_I(CGCMFDSurfDiffCoeff_X, (char *)("CGCMFDSurfDiffCoeff_X"));
	CGCMFDSurfDiffCoeff_Y = FreeMem_FP_I(CGCMFDSurfDiffCoeff_Y, (char *)("CGCMFDSurfDiffCoeff_Y"));
	CGCMFDSurfDiffCoeff_Z = FreeMem_FP_I(CGCMFDSurfDiffCoeff_Z, (char *)("CGCMFDSurfDiffCoeff_Z"));
	CGCMFDSurfNDF_X = FreeMem_FP_I(CGCMFDSurfNDF_X, (char *)("CGCMFDSurfNDF_X"));
	CGCMFDSurfNDF_Y = FreeMem_FP_I(CGCMFDSurfNDF_Y, (char *)("CGCMFDSurfNDF_Y"));
	CGCMFDSurfNDF_Z = FreeMem_FP_I(CGCMFDSurfNDF_Z, (char *)("CGCMFDSurfNDF_Z"));
	CGCMFDNetCurr_X = FreeMem_FP_I(CGCMFDNetCurr_X, (char *)("CGCMFDNetCurr_X"));
	CGCMFDNetCurr_Y = FreeMem_FP_I(CGCMFDNetCurr_Y, (char *)("CGCMFDNetCurr_Y"));
	CGCMFDNetCurr_Z = FreeMem_FP_I(CGCMFDNetCurr_Z, (char *)("CGCMFDNetCurr_Z"));
	CGCMFD_TotFissQ = FreeMem_FP_I(CGCMFD_TotFissQ, (char *)("CGCMFD_TotFissQ"));
	CGCMFD_TotFissQ_old = FreeMem_FP_I(CGCMFD_TotFissQ_old, (char *)("CGCMFD_TotFissQ_old"));
	CGCMFD_FluxFlag = FreeMem_i32_I(CGCMFD_FluxFlag, (char *)("CGCMFD_FluxFlag"));
	CGCMFD_NegFluxCount = FreeMem_float_I(CGCMFD_NegFluxCount, (char *)("CGCMFD_NegFluxCount"));
	CGCMFD_fissQflagl2 = FreeMem_i32_I(CGCMFD_fissQflagl2, (char *)("CGCMFD_fissQflagl2"));
	CGCMFD_fissQflaglinf = FreeMem_i32_I(CGCMFD_fissQflaglinf, (char *)("CGCMFD_fissQflaglinf"));
	CGCMFD_flageig = FreeMem_i32_I(CGCMFD_flageig, (char *)("CGCMFD_flageig"));
	CGCMFD_fissQerrlinf = FreeMem_FP_I(CGCMFD_fissQerrlinf, (char *)("CGCMFD_fissQerrlinf"));
	CGCMFD_fissQrerrl2 = FreeMem_FP_I(CGCMFD_fissQrerrl2, (char *)("CGCMFD_fissQrerrl2"));
	CGCMFD_fluxErrlinf = FreeMem_FP_I(CGCMFD_fluxErrlinf, (char *)("CGCMFD_fluxErrlinf"));
	
	CGMatrixCSR = FreeMem_FP_I(CGMatrixCSR, (char *)("CGMatrixCSR"));
	CGMatrixCSR_RowStartInd = FreeMem_i32_I(CGMatrixCSR_RowStartInd, (char *)("CGMatrixCSR_RowStartInd"));
	CGMatrixCSR_ColInd = FreeMem_i32_I(CGMatrixCSR_ColInd, (char *)("CGMatrixCSR_ColInd"));
	CG_DiagPointer = FreeMem_i32_I(CG_DiagPointer, (char *)("CG_DiagPointer"));
	if (CGCMFD_LinearSolver == 2) {
		CG_ILUMatrix = FreeMem_FP_I(CG_ILUMatrix, (char *)("CG_ILUMatrix"));
	}
#if defined(MPI_USED)
	CGCMFDData_MPIComm_SumInd = FreeMem_i32_I(CGCMFDData_MPIComm_SumInd, (char *)("CGCMFDData_MPIComm_SumInd"));
	CGCMFDComm_MPISend = FreeMem_FP_I(CGCMFDComm_MPISend, (char *)("CGCMFDComm_MPISend"));
	CGCMFDFlux_MPIRecv = FreeMem_FP_I(CGCMFDFlux_MPIRecv, (char *)("CGCMFDFlux_MPIRecv"));
	CGCMFDNodeD_MPIRecv = FreeMem_FP_I(CGCMFDNodeD_MPIRecv, (char *)("CGCMFDNodeD_MPIRecv"));
	CGCMFDMPI_InBoundSourCoeff = FreeMem_FP_I(CGCMFDMPI_InBoundSourCoeff, (char *)("CGCMFDMPI_InBoundSourCoeff"));
	CGCMFDMPI_InBoundSourCoeff_Index = FreeMem_i32_I(CGCMFDMPI_InBoundSourCoeff_Index, (char *)("CGCMFDMPI_InBoundSourCoeff_Index"));
	CGCMFDMPI_InBoundSourCoeff_Dir = FreeMem_i32_I(CGCMFDMPI_InBoundSourCoeff_Dir, (char *)("CGCMFDMPI_InBoundSourCoeff_Dir"));
	CGCMFDMPI_InBoundSourCoeff_GeoDim = FreeMem_FP_I(CGCMFDMPI_InBoundSourCoeff_GeoDim, (char *)("CGCMFDMPI_InBoundSourCoeff_GeoDim"));
    CGCMFDComm_NumData = FreeMem_i32_I(CGCMFDComm_NumData, (char *)("CGCMFDComm_NumData"));
	CGCMFDData_MPIComm_Index = FreeMem_i32_I(CGCMFDData_MPIComm_Index, (char *)("CGCMFDData_MPIComm_Index"));
#endif // defined(MPI_USED)
#endif // defined(M_CG_CMFD)
	if (MGCMFD_LinearSolver == 2) {
		MG_HessenburgMatrix = FreeMem_FP_I(MG_HessenburgMatrix, (char *)("MG_HessenburgMatrix"));
		MG_GivensRotations = FreeMem_FP_I(MG_GivensRotations, (char *)("MG_GivensRotations"));
		MG_HessenburgRHSVec = FreeMem_FP_I(MG_HessenburgRHSVec, (char *)("MG_HessenburgRHSVec"));
		MG_KrylovOrthBasisVec = FreeMem_FP_I(MG_KrylovOrthBasisVec, (char *)("MG_KrylovOrthBasisVec"));
		MG_Residual = FreeMem_FP_I(MG_Residual, (char *)("MG_Residual"));
		MG_IW = FreeMem_i32_I(MG_IW, (char *)("MG_DiagPointer"));
		MG_TemnpVec_1 = FreeMem_FP_I(MG_TemnpVec_1, (char *)("MG_TemnpVec_1"));
		MG_TemnpVec_2 = FreeMem_FP_I(MG_TemnpVec_2, (char *)("MG_TemnpVec_2"));
#if defined(M_CG_CMFD)
		if (CGCMFD_LinearSolver == 1) {
			CG_HessenburgMatrix = FreeMem_FP_I(CG_HessenburgMatrix, (char *)("CG_HessenburgMatrix"));
			CG_GivensRotations = FreeMem_FP_I(CG_GivensRotations, (char *)("CG_GivensRotations"));
			CG_HessenburgRHSVec = FreeMem_FP_I(CG_HessenburgRHSVec, (char *)("CG_HessenburgRHSVec"));
			CG_KrylovOrthBasisVec = FreeMem_FP_I(CG_KrylovOrthBasisVec, (char *)("CG_KrylovOrthBasisVec"));
			CG_Residual = FreeMem_FP_I(CG_Residual, (char *)("CG_Residual"));
			CG_IW = FreeMem_i32_I(CG_IW, (char *)("CG_DiagPointer"));
			CG_TemnpVec_2 = FreeMem_FP_I(CG_TemnpVec_2, (char *)("CG_TemnpVec_2"));
	    }
#endif // defined(M_CG_CMFD)
	}
#endif // defined(M_MG_CMFD)

	InpXSs_s_NNZ = FreeMem_i32_I(InpXSs_s_NNZ, (char *)("InpXSs_s_NNZ"));
	if (IfSetXSsForEveryFSR == 1) {
		MacroXS_t = FreeMem_FP_I(MacroXS_t, (char *)("MacroXS_t"));
		if (ProblemType == 0) {
			MacroXS_a = FreeMem_FP_I(MacroXS_a, (char *)("MacroXS_a"));
			MacroXS_f = FreeMem_FP_I(MacroXS_f, (char *)("MacroXS_f"));
			MacroXS_nf = FreeMem_FP_I(MacroXS_nf, (char *)("MacroXS_nf"));
			Chi = FreeMem_FP_I(Chi, (char *)("Chi"));
			MacroXS_s = FreeMem_FP_I(MacroXS_s, (char *)("MacroXS_s"));
			MacroXS_s_NZRowStartID = FreeMem_ui64_I(MacroXS_s_NZRowStartID, (char *)("MacroXS_s_NZRowStartID"));
			MacroXS_s_NZColStartID = FreeMem_ui64_I(MacroXS_s_NZColStartID, (char *)("MacroXS_s_NZColStartID"));

			if (TransCorre == 1) {
				MacroXS_p0 = FreeMem_FP_I(MacroXS_p0, (char *)("MacroXS_p0"));
				MacroXS_p1 = FreeMem_FP_I(MacroXS_p1, (char *)("MacroXS_p1"));
			}
		}
		else if (ProblemType == 1) {
			FixedSour = FreeMem_FP_I(FixedSour, (char *)("FixedSour"));
		}
		else {}
	}
	else if (IfSetXSsForEveryFSR == 0) {
		if (INPUT_XSs_MATorFSR == 0) {
			XSsID_FSRs = FreeMem_double_II2P(XSsID_FSRs, MOCNumLayersZ, NumPinsInSubDomPerLayer, (char *)("XSsID_FSRs"));
		}
	}

	LTsIndex_AzimAngID_HS=FreeMem_int_IP(LTsIndex_AzimAngID_HS, NumTrackSet, (char *)("LTsIndex_AzimAngID_HS"));
	LTsIndex_LTID_HS=FreeMem_int_IP(LTsIndex_LTID_HS, NumTrackSet,(char *)("LTsIndex_LTID_HS"));
#ifdef GeoTwoPIRegion
	LTsIndex_AzimAngID_2PI = FreeMem_int_IP(LTsIndex_AzimAngID_2PI, NumTrackSet, (char *)("LTsIndex_AzimAngID_2PI"));
	LTsIndex_LTID_2PI = FreeMem_int_IP(LTsIndex_LTID_2PI, NumTrackSet, (char *)("LTsIndex_LTID_2PI"));
#endif // GeoTwoPIRegion

#if defined(MPI_USED)
	if (ProcID==0) {
		ProcessorNameColl = FreeMem_char_II(ProcessorNameColl, NumProcs, ((char *)("\0")));
		CurrCPUIDColl = FreeMem_i32_I(CurrCPUIDColl, ((char *)("\0")));
	}
#endif

    FreeTrackMem();
	
	FreeGeoMem();
	
	RayTracModDim = FreeMem_FPGeo_I(RayTracModDim, (char *)("RayTracModDim"));
	NumGroupBound = FreeMem_i32_I(NumGroupBound, (char *)("NumGroupBound"));
	RaySpacingIni = FreeMem_FPGeo_I(RaySpacingIni, (char *)("RaySpacingIni"));
	NumAzimAng = FreeMem_i32_I(NumAzimAng, (char *)("NumAzimAng"));
	PolarQuadType = FreeMem_i32_I(PolarQuadType, (char *)("PolarQuadType"));
	NumPolarAng = FreeMem_i32_I(NumPolarAng, (char *)("NumPolarAng"));
	GroupBoundary = FreeMem_int_IP(GroupBoundary, NumPolAngPart, (char *)("GroupBoundary"));
	PolarAngBoundary = FreeMem_i32_I(PolarAngBoundary, (char *)("PolarAngBoundary"));

	FSRVolumeCorrectionType = FreeMem_i32_I(FSRVolumeCorrectionType, (char *)("FSRVolumeCorrectionType"));
	SubDomainsMap = FreeMem_int_III(SubDomainsMap, NumSubDomains[2], NumSubDomains[1], (char *)("SubDomainsMap"));
	NumSubDomains = FreeMem_i32_I(NumSubDomains, (char *)("NumSubDomains"));
	HardwareMap_SubDom = FreeMem_i32_I(HardwareMap_SubDom, (char *)("HardwareMap_SubDom"));
	NumPin_subDomain = FreeMem_int_IP(NumPin_subDomain, 3, (char *)("NumPin_subDomain"));
	PinPowerSumLayBound = FreeMem_i32_I(PinPowerSumLayBound, (char *)("PinPowerSumLayBound"));
	CUDADeviceID = FreeMem_i32_I(CUDADeviceID, (char *)("CUDADeviceID"));
    CMFDGroupBound = FreeMem_i32_I(CMFDGroupBound, (char *)("CMFDGroupBound"));
    CGCMFDGroupBound = FreeMem_i32_I(CGCMFDGroupBound, (char *)("CGCMFDGroupBound"));

    ExpTable = FreeMem_FP_I(ExpTable, (char *)("ExpTable"));

	FreeQuadratureInfo(NumTrackSet);

	GC_SinThetasTimes2 =FreeMem_FP_I(GC_SinThetasTimes2,(char *)("GC_SinThetasTimes2"));
	GC_OneOverSinThetas = FreeMem_FP_I(GC_OneOverSinThetas, (char *)("GC_OneOverSinThetas"));

#if defined(CUDA_USED)
	if (INPUT_XSs_MATorFSR == 0 && ProblemType == 0) {
	    MacroXS_s_NNZ_Layer = FreeMem_ui64_I(MacroXS_s_NNZ_Layer, (char *)("MacroXS_s_NNZ_Layer"));
        MacroXS_s_NZRowStartID_Layer = FreeMem_ui64_I(MacroXS_s_NZRowStartID_Layer, (char *)("MacroXS_s_NZRowStartID_Layer"));
	}
#endif // deifned(CUDA_USED)

    MOC_boolFluxNormInf = FreeMem_i32_I(MOC_boolFluxNormInf, (char *)("MOC_boolFluxNormInf"));
    MOC_boolFluxNorm2 = FreeMem_i32_I(MOC_boolFluxNorm2, (char *)("MOC_boolFluxNorm2"));
    MOC_boolFluxRMSE = FreeMem_i32_I(MOC_boolFluxRMSE, (char *)("MOC_boolFluxRMSE"));

	MOC_fluxNormInf = FreeMem_FP_I(MOC_fluxNormInf, (char *)("MOC_fluxNormInf"));
	MOC_fluxNorm2 = FreeMem_FP_I(MOC_fluxNorm2, (char *)("MOC_fluxNorm2"));
	MOC_fluxRMSE = FreeMem_FP_I(MOC_fluxRMSE, (char *)("MOC_fluxRMSE"));
    if (ProcID == 0) {
		printf("Finish");
		fflush(stdout);
		fprintf(LogFile, "Finish");
	}

	CoeffA_HOScatt = FreeMem_FP_I(CoeffA_HOScatt, (char *)("CoeffA_HOScatt"));
}

void PrintCPUMemUsage(int _procID)
{
	int i, j, _width1=30, _width2=12;
	FP_PRECISION _mem=0.0;
	/*printf memory info*/
	if (ProcID==0) {
		printf("\n [  INFO   ]  Memory Usage on host (and device) for Transport Sweep");
		printf("\n              %-*s%-*s", _width1, "Parameter", _width2, "Host (MB)");
		fflush(stdout);
		fprintf(LogFile, "\n [  INFO   ] Memory Usage on host (and device) for Transport Sweep");
		fprintf(LogFile, "\n              %-*s%-*s", _width1, "Parameter", _width2, "Host (MB)");
#if defined(CUDA_USED)
		printf("%-*s", _width2, "Device (MB)");
        fflush(stdout);
		fprintf(LogFile, "%-*s", _width2, "Device (MB)");
#endif // defined(CUDA_USED)
		for (i = 0; i < MemPrintCount + 1; i++) {
			_mem = 0.0;
			for (j = MemParaCountBoundLow[i]; j < MemParaCountBoundUp[i]; j++) {
				_mem += ParaMemSize[j];
			}
			printf("\n              %-*s%-*.1lf", _width1, MemParaName[i], _width2, _mem);
			fflush(stdout);
			fprintf(LogFile, "\n              %-*s%-*.1lf", _width1, MemParaName[i], _width2, _mem);
			PrintDeviceMemUsage(LogFile, MemParaName[i]);
		}
		printf("\n              %-*s%-*.1lf", _width1, "Total Mem Usage", _width2, NumTotalData);
		fflush(stdout);
		fprintf(LogFile, "\n              %-*s%-*.1lf", _width1, "Total Mem Usage", _width2, NumTotalData);
#if defined(CUDA_USED)
		printf("%-*.1lf", _width2, CudaMemUse[0]);
        fflush(stdout);
		fprintf(LogFile, "%-*.1lf", _width2, CudaMemUse[0]);
#endif // defined(CUDA_USED)
    }
}


/**
* @ Function: MOC_Main_InitExpTable()
*   introduction: generate exponential table
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void MOC_Main_InitExpTable(int _procID)
{
#ifdef TIME_REC
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC
	if (LinearInter_Exp > 0) {
		if (_procID == 0) {
			printf("\n [  INFO   ]  Computing the LinearInter_Exp Table ...");
			fprintf(LogFile, "\n [  INFO   ]  Computing the LinearInter_Exp Table ...");
			fflush(stdout);
		}
		if (LinearInter_Exp == 1) { 
			Index_Exp = 1;
			ExpTable = ExpFast_LinearExpTable_Mod1();
		}//openmoc exponential table e-10 & Fastest MPACT exponential table e-9
		else if (LinearInter_Exp == 2) { 
			Exp2TH = ExpH16_Initialize();
			Exp2TL = ExpL16_Initialize();
			Index_Exp = 2; 
		}//2 tables scheme e-13
		else if (LinearInter_Exp == 3) { 
			Index_Exp = 3; 
			Exp2C1T = Exp10_2C1T_Initialize();
			Exp10_C1 = (FP_PRECISION)(-9.765624224182007e-4);
			Exp10_C2 = (FP_PRECISION)(4.766043938629177e-7);
		}//2 constants & 1 table scheme e-11
		else if (LinearInter_Exp == 4) { 
			ExpLinear = Exp10_Linear_Initialize();
			Index_Exp = 4; 
		}//MPACT linear exponential table e-7
	}
	else { Index_Exp = 0; }//intrinsic exp(x) function
	
#ifdef TIME_REC
	_time_end = TallyCurrentTime_CPU();
	TimePrePart[TimeCountPrePart] = _time_end - _time_start;
	strcpy(TimeMetricsPrePart[TimeCountPrePart], "Initialize Exp Interp");
	TimeCountPrePart++;
#endif // TIME_REC
}

/**
* @ Function: MOC_Main_ParaUpdate_TS()
*   introduction: update moc parameter during transport iteration
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: PrintCurrentCPUID()
*   introduction: print the id and name of current process
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: CPU_MOCTransportSweep()
*   introduction: moc transport sweep including: flux zerolization + transport sweep
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2019/03/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: CPU_MOCFixSourProbSolver_lay_g()
*   introduction: moc transport sweep including: flux zerolization + transport sweep for fixed source problem
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2019/09/16, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: MOC_TransSweep_thread()
*   introduction: moc transport sweep function with openMP parallelization
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2019/03/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: MOC_TransSweep_Layer()
*   introduction: collection of moc transport sweep functions for each 2D layer
* @ Parameter List:
*   Return: void
* @ Created     : 2019/11/23, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: MOC_TransSweep_FixSourProb_lay_g()
*   introduction: moc transport sweep function for fixed source problem
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2019/09/16, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: MOC_TransSweep_ParaFor()
*   introduction: moc transport sweep function with openMP parallelization
*                 the ray parallelization is directly implemented by #pragma omp parallel for
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2019/03/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
#if defined(OMP_USED) && defined(OMP_SCHEDULE)
void MOC_TransSweep_ParaFor(int _procID, int _numOMPThread_comp)
{
	int i, j, _count/*,g, upScattIter, in_iter,_width_3 = 14*/;
	int _lay/*, cell, reg, FSR_ID, _XSt_location*/, _trackSet = 0, _FSRVolCorrType = 1;
	int _groupBoundLowLim, _groupBoundUpLim, _polarBoundLowLim, _polarBoundUpLim;
	// jacobi - solver
	if (MOCSolverType == 1) {
		//VectorInit_FP_Flux(ScalarFlux_MOC, TotalNumFSRs_subDom*NumGroups, (FP_PRECISION_FLUX)(0.0));
		for (_lay = 0; _lay < MOCNumLayersZ; _lay++) {
			_count = 0;
			for (i = 0; i < NumPolAngPart; i++) {
				_polarBoundLowLim = PolarAngBoundary[i];
				_polarBoundUpLim = PolarAngBoundary[i + 1];
				for (j = 0; j < NumGroupBound[i]; j++) {
					_groupBoundLowLim = GroupBoundary[i][j];
					_groupBoundUpLim = GroupBoundary[i][j + 1];
					_trackSet = _count;
					_FSRVolCorrType = FSRVolumeCorrectionType[_count];

					Jacobi_MOCSolver_2D_diam_OMP_ParaFor_NonAtomic_Package(_numOMPThread_comp, _lay, _trackSet, _groupBoundLowLim, _groupBoundUpLim, \
						_polarBoundLowLim, _polarBoundUpLim, _FSRVolCorrType);

					_count++;
				}
			}
		}
	}
}
#endif // defined(MPI_USED) && defined(OMP_SCHEDULE)


/**
* @ Function: BoundAngFluxExchange_MPIComm()
*   introduction: update the inner boundary angular flux by MPI communication
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2017/03/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: MOC_Main_Output()
*   introduction: output the final MOC solution
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2017/03/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void MOC_Main_Output(int _procID)
{
//	int lay,g;
// #if defined(CUDA_USED)
//     GPUFinalOutputs(TotalNumFSRs_subDom*NumGroups, OUTPUT_flux, OUTPUT_flux_mat, OUTPUT_normPinPower, ScalarFlux_MOC);
// #endif // defined(CUDA_USED)

	OutputMOCGeometryInfo(_procID,NumTrackSet, NumAzimAng, NumPolarAng,NumTracks,NumTracksX,NumTracksY,\
		DxEff,DyEff,EffAzimAng,EffRaySpacing,AzimWeights,TotalWeights,NumRTMGeoType, CellBasedRTMGeo,RTModFSRsInfo_2PI,\
		Index_MOC_TrackInRTM_Total, NumSegsInTrack_2PI, SegInfoInTrack_2PI_RayWisedFSR, FsrIdMap_RayWisedToAntiClock, \
		SegInfoInTrack_SINGLE_2PI, RTMStartP, RTMEndP, TotalNumFSRsInRTM, SumNumFSRsInRTM,SegInfoInTrack_2PI_length,\
		NumLTs_HS,  LTsIndex_AzimAngID_2PI, LTsIndex_LTID_2PI, LongTracksInfo_Total, TrackIndexInLTs_2PI, TracksInfoInLTs_2PI, \
		NumPinsInSubDomPerLayer,  TotNumTracksInOneRTM_2PI, CellRTMType_SubDom, IndexTrackInRTM_SumAzim_2PI, \
		SegInfoInTrack_2PI_AntiClockFSR, SumNumSegsInRTMTrack_2PI, SumNumFSRsBasedGlobCellID, FSRsAreaInRTM_AntiClockFSR);

    if (NumGroups < 100) {
        OutputMacroXSs_FSRs(ProcID, MOCNumLayersZ, NumPinsInSubDomPerLayer, NumFSRsInCell, SumNumFSRsBasedGlobCellID, \
		NumGroups, MacroXS_t, MacroXS_a, MacroXS_f, MacroXS_nf, MacroXS_s, Chi, InpXSs_t, InpXSs_a, InpXSs_f, \
		InpXSs_nf, InpXSs_s, InpChi, MacroXS_s_NZRowStartID, MacroXS_s_NZColStartID);
    } else {
        if (OUTPUT_MacorXSs) {
            FILE *_opop = NULL;
            char opopName[255];
            sprintf(opopName, "%s-%s-%d", "HomoHomo", "Proc", _procID);
            _opop = fopen(opopName, "wb");
            fwrite(&TotalNumFSRs_subDom, sizeof (M_INT32), 1, _opop);
            fwrite(&NumGroups, sizeof (M_INT32), 1, _opop);
            fwrite(MacroXS_t, sizeof(FP_PRECISION), TotalNumFSRs_subDom * NumGroups, _opop);
            fwrite(MacroXS_a, sizeof(FP_PRECISION), TotalNumFSRs_subDom * NumGroups, _opop);
            fwrite(MacroXS_f, sizeof(FP_PRECISION), TotalNumFSRs_subDom * NumGroups, _opop);
            fwrite(MacroXS_nf, sizeof(FP_PRECISION), TotalNumFSRs_subDom * NumGroups, _opop);
            fwrite(Chi, sizeof(FP_PRECISION), TotalNumFSRs_subDom * NumGroups, _opop);
            fwrite(ScalarFluxOld, sizeof(FP_PRECISION_FLUX), TotalNumFSRs_subDom * NumGroups, _opop);
            fwrite(FSRVol_SubDom, sizeof(FP_PRECISION_GEO), TotalNumFSRs_subDom, _opop);
            fclose(_opop);
        }
    }

    CalcPinPower_SubDom(MOCNumLayersZ, ScalarFluxOld, NumPinsInSubDomPerLayer, NumFSRsInCell, NumGroups, NormPinPowerSubDom, \
	    SumNumFSRsBasedGlobCellID, FSRArea_SubDom, MacroXS_f, WidthLayersInCore);
#if defined(MPI_USED)
    CollPinPower_GlobDom(ProcID, NumProcs, CollecGlobPinPower, NormPinPowerSubDom, CollNumPinsSubDom, MOC_NumPinsInSubDom);
#endif // 
	SumPinPower_Lay(PinPowerSumNumSet, PinPowerSumLayBound, NumPinsInSubDomPerLayer, NormPinPowerSubDom, NormPinPower_SumLay);
//	TotPower_core = CalcTotPower_Core(_procID, NumProcs, NumPinsInSubDomPerLayer,NormPinPowerSubDom, MOC_totPower_procs);
//#ifdef MPI_USED
//	//MPI_Bcast(&TotPower_core, 1, MPI_DataType, 0, MPI_COMM_WORLD);
//#endif // MPI_USED
//	CalcNormPinPower_subDom(/*_procID,*/ NormPinPowerSubDom, TotPower_core, NumPinsInSubDomPerLayer, NumCellsInCore);
//	// if (Gene_VisualFile==1) {
//	// 	for (g=0;g<NumGroups;g++) {
//    //         GenerateVTKFile_Visu(2, _procID, NumPointsPerRing, NumRTMGeoType, CellBasedRTMGeo,g, ScalarFlux_MOC, NULL,RTMMeshType,\
//	// 	        NumRings_RTM_Mat,NumRings_RTM_Mesh,NumSectorsPerRing_RTM_Mesh,RadiusPerRing_RTM_Mat,RadiusPerRing_RTM_Mesh,\
//	//             NumSubDomains,NumPin_subDomain,WidthLayersInCore,CellRTMType_SubDom,NumPinsInSubDomPerLayer, SubDomGlobCellID,\
//	// 		    CellTypeID_visu,NumCells,Cells, MatID_RTM, NumCellsInRTM,NumAssemblies,Assemblies, NumLattices,Lattices,\
//	// 			SumNumFSRsBasedGlobCellID,NumCellsInPin);
//	// 	}	
//	// 	GenerateVTKFile_Visu(3, _procID, NumPointsPerRing, NumRTMGeoType, CellBasedRTMGeo,0, NULL, NormPinPowerSubDom,RTMMeshType,\
//	// 	    NumRings_RTM_Mat,NumRings_RTM_Mesh,NumSectorsPerRing_RTM_Mesh,RadiusPerRing_RTM_Mat,RadiusPerRing_RTM_Mesh,\
//	//         NumSubDomains,NumPin_subDomain,WidthLayersInCore,CellRTMType_SubDom,NumPinsInSubDomPerLayer, SubDomGlobCellID,\
//	// 		CellTypeID_visu,NumCells,Cells, MatID_RTM, NumCellsInRTM,NumAssemblies,Assemblies, NumLattices,Lattices,\
//	// 		SumNumFSRsBasedGlobCellID,NumCellsInPin);
//	// }
//#ifdef MPI_USED
//	//CollecGlobPinPower = CollecPinPower(_procID, NumProcs, NormPinPowerSubDom, NumSubDomains, SubDomainsMap, NumPin_subDomain);
//#endif // MPI_USED
//	if (ProblemType == 0) {
//		if (ProblemType == 0 && (OUTPUT_flux_mat==1 || OUTPUT_flux==1)) {
//		    MOCGlobFissRateTemp = CompReacRate_TotalDom(ProcID, 0, ScalarFlux_MOC, MOCNumLayersZ, NumPinsInSubDomPerLayer, \
//			    NumFSRsInCell, NumGroups, CellRTMType_SubDom, MacroXS_nf, InpXSs_nf, RTModFSRsInfo_HS, SumNumFSRsBasedGlobCellID);
//	    }
//		OutputFSRsScalarFlux(_procID, NumPinsInSubDomPerLayer, NumFSRsInCell, ScalarFlux_MOC, MOCGlobFissRateTemp);
//		OutputFSRsScalarFlux_Mat(_procID, NumPinsInSubDomPerLayer, NumFSRsInCell, ScalarFlux_MOC, MOCGlobFissRateTemp);
//	}
//	else {
//		OutputFSRsScalarFlux(_procID, NumPinsInSubDomPerLayer, NumFSRsInCell, ScalarFlux_MOC, MOCGlobFlux_FixSour);
//	}
    if (ProblemType == 0) {
    if (ProblemType == 0 && (OUTPUT_flux_mat == 1 || OUTPUT_flux == 1)) {
        MOCGlobFissRateTemp = CompReacRate_TotalDom(ProcID, 0, ScalarFluxOld, MOCNumLayersZ, NumPinsInSubDomPerLayer, \
                NumFSRsInCell, NumGroups, CellRTMType_SubDom, MacroXS_nf, InpXSs_nf, RTModFSRsInfo_HS,
                                                    SumNumFSRsBasedGlobCellID);
    }
    OutputFSRsScalarFlux(_procID, NumPinsInSubDomPerLayer, NumFSRsInCell, ScalarFluxOld, MOCGlobFissRateTemp);
    OutputFSRsScalarFlux_Mat(_procID, NumPinsInSubDomPerLayer, NumFSRsInCell, ScalarFluxOld, MOCGlobFissRateTemp);
        if (NumScattOrder > 0) {
            OutputFSRsScalarFluxMoment(_procID, NumPinsInSubDomPerLayer, NumFSRsInCell, ScalarFluxOld, NumScattOrder);
        }
    }
    else {
        OutputFSRsScalarFlux(_procID, NumPinsInSubDomPerLayer, NumFSRsInCell, ScalarFluxOld, MOCGlobFlux_FixSour);
    }
//	//OutputFSRsScalarFlux_forComp(_procID, NumPinsInSubDomPerLayer, NumFSRsInCell, ScalarFlux_MOC);
//	if (_procID == 0) {
		// output_normPinPower(NormPinPowerSubDom, NormPinPower_SumLay);
	Output_NormPinPower(ProcID, NumProcs, 0);
#if defined(M_MG_CMFD)
    if (OUTPUT_CoarseMesh_GroupConstant == 1) {
#if defined(CUDA_USED)
		GPUCoarseMeshGroupConst_CheckCollapse(OUTPUT_CoarseMesh_GroupConstant, MOC_NumPinsInSubDom, \
			NumCMFDGroup, CMFDXs_t, CMFDXs_a, CMFDXs_f, CMFDXs_nf, CMFD_Chi, CMFDXs_s);
#endif
		Output_CoarseMesh_GroupConstant(CaseId, 0);
	
	Output_CMFD_GroupConstant(CaseId, ProcID, 0, 0, CMFDKeff);
#if defined(M_CG_CMFD)
    Output_CMFD_GroupConstant(CaseId, ProcID, 1, 0, CMFDKeff);
#endif // defined(M_CG_CMFD)
}
#endif	
}

/**
* @ Function: OptimalWorkLoadDistribution()
*   introduction: get the optimal workload assignment between CPU and GPU
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2019/03/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void OptimalWorkLoadDistribution(int _procID,int _numProcs)
{
#ifdef DynamicWorkLoad
	FP_PRECISION_TIME *_time_ts = NULL, _avgTime_cpu,_avgTime_gpu, _opt_load_cpu_fp,_opt_load_gpu_fp,_nx,_a_cpu,_b_gpu,_t_cpu,_t_gpu;
	int i,j,_proc, _load_cpu, _load_gpu,_opt_load_cpu, _opt_load_gpu, _numMpiOmp, _numMpiCuda,_quotient,_remainder;
	if (_procID==0) {
		_time_ts = (FP_PRECISION_TIME *)calloc(_numProcs, sizeof(FP_PRECISION_TIME));
		_time_ts[0] = TimeTS_DWL;
		for (_proc = 1; _proc < _numProcs; _proc++) {
			MPI_Recv(&_time_ts[_proc], 1, MPI_DataType, _proc, _proc, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}
	}
	else {
		MPI_Send(&TimeTS_DWL, 1, MPI_DataType, 0, _procID, MPI_COMM_WORLD);
	}

	if (_procID == 0) {
		for (_proc = 0; _proc < _numProcs; _proc++) {
			printf("\n _time_ts[%d]=%.3e", _proc, _time_ts[_proc]);
		}
		_numMpiOmp = 0;
		_numMpiCuda = 0;
		_avgTime_cpu = (FP_PRECISION_TIME)(0.0);
		_avgTime_gpu = (FP_PRECISION_TIME)(0.0);
		for (_proc = 0; _proc < _numProcs; _proc++) {
			if (HardwareMap_SubDom[_proc] == 0) {
				_numMpiOmp++;
				_avgTime_cpu += _time_ts[_proc];
			}
			else if (HardwareMap_SubDom[_proc] == 1) {
				_numMpiCuda++;
				_avgTime_gpu += _time_ts[_proc];
			}
			else {}
		}
		_avgTime_cpu = _avgTime_cpu / _numMpiOmp;
		_avgTime_gpu = _avgTime_gpu / _numMpiCuda;
		printf("\n _numMpiOmp=%d, _numMpiCuda=%d, _avgTime_cpu=%.3e,_avgTime_gpu=%.3e", _numMpiOmp, _numMpiCuda, _avgTime_cpu, _avgTime_gpu);

		_load_cpu = 0;
		_load_gpu = 0;
		_numMpiOmp = 0;
		_numMpiCuda = 0;
		for (_proc = 0; _proc < NumSubDomains[0]; _proc++) {
			if (HardwareMap_SubDom[_proc] == 0) {
				_load_cpu += NumPin_subDomain[0][_proc];
				_numMpiOmp++;
			}
			else if (HardwareMap_SubDom[_proc] == 1) {
				_load_gpu += NumPin_subDomain[0][_proc];
				_numMpiCuda++;
			}
			else {}
		}
		printf("\n in x direction:_numMpiOmp=%d, _numMpiCuda=%d, _load_cpu=%.2lf,_load_gpu=%.2lf", _numMpiOmp, _numMpiCuda, \
			(FP_PRECISION_TIME)(_load_cpu)/(FP_PRECISION_TIME)(_numMpiOmp), (FP_PRECISION_TIME)(_load_gpu)/(FP_PRECISION_TIME)(_numMpiCuda));
		_t_cpu = _avgTime_cpu / (FP_PRECISION_TIME)(_load_cpu)*(FP_PRECISION_TIME)(_numMpiOmp);
		_t_gpu = _avgTime_gpu / (FP_PRECISION_TIME)(_load_gpu)*(FP_PRECISION_TIME)(_numMpiCuda);
		_nx = (FP_PRECISION_TIME)(NumCellsInCore_2D_X);
		_a_cpu = (FP_PRECISION_TIME)(_numMpiOmp);
		_b_gpu = (FP_PRECISION_TIME)(_numMpiCuda);
		_opt_load_cpu_fp = (_nx*_t_gpu) / (_a_cpu*_t_gpu + _b_gpu*_t_cpu);
		_opt_load_gpu_fp = (_nx*_t_cpu) / (_a_cpu*_t_gpu + _b_gpu*_t_cpu);
		printf("\n _t_cpu=%.3e,_t_gpu=%.3e", _t_cpu, _t_gpu);
		printf("\n _a_cpu=%.3e,_b_gpu=%.3e", _a_cpu, _b_gpu);
		printf("\n _nx=%.3e", _nx);
		printf("\n _opt_load_cpu_fp=%.3lf, _opt_load_gpu_fp=%.3lf", _opt_load_cpu_fp, _opt_load_gpu_fp);

		_opt_load_gpu = (int)(_opt_load_gpu_fp) + 1;
		printf("\n ### workload for CP_iter + 1U-GPU:");
		for (i = 0; i < _numMpiCuda; i++) {
			printf("%3d", _opt_load_gpu);
		}
		
		_quotient = (NumCellsInCore_2D_X - _opt_load_gpu*_numMpiCuda) / _numMpiOmp;
		_remainder = (NumCellsInCore_2D_X - _opt_load_gpu*_numMpiCuda) % _numMpiOmp;
		printf("\n ### workload for CPU-CPU:");
		for (i = 0; i < _numMpiOmp; i++) {
			if (i < _remainder) {
				printf("%3d", _quotient+1);
			}
			else {
				printf("%3d", _quotient);
			}
		}
	}
#endif // DynamicWorkLoad
}

void MPIInfo_Preparation()
{
#ifdef MPI_USED
#ifdef TIME_REC
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
#endif // TIME_REC
	if (NumProcs > 1) {
#ifdef TIME_REC
	    _time_start = TallyCurrentTime_CPU();
#endif // TIME_REC
		UpdateBoundAngFlux_PreNeibInfo(MOCNumLayersZ, NumProcPerLayer, MOC_subDomID_layer, ProcID, Mess_TarProc, Mess_SendNumLTs, \
			NumAzimAng[0], LongTracksInfo_Total[0], NumLongTracks[0], MPIProcID_recv, MPINeibID_recv, Mess_SendSatrtPInAngF, \
			Mess_RecvSatrtPInAngF, NumPolarAng[0], NumGroups, NumLTs_HS[0], NumLTsSum[0], NULL,NULL/*GCCopy_StartPInAngF, GCCopy_NumDataInAngF*/);
#if defined(M_MG_CMFD)
		MGCMFDBoundInfoComm_PreNeibInfo(ProcID, SubDomLocID_2D_X, SubDomLocID_2D_Y, 0, MOCNumLayersZ, MOCNumPinsInSubDom_X, \
			MOCNumPinsInSubDom_Y, NumCMFDGroup, NumSubDomains, CMFDComm_TarProc, MGCMFDComm_NumData, CMFDComm_DirBound, \
			CMFDComm_RecvDir, MGCMFDData_MPIComm_Index, MGCMFDData_MPIComm_Num, MGCMFDData_MPIComm_SumInd);

		SetMOCPartCurrIndex_InnerBound(ProcID, NumProcs, MOCNumLayersZ, MOCNumPinsInSubDom_X, MOCNumPinsInSubDom_Y, \
			NumGroups, CMFDComm_TarProc, CMFDComm_DirBound, MGCMFD_MOCParCurrInnerBound_Index, MGCMFD_MOCParCurrInnerBound_Dir);
		VectorInit_int_I(MGCMFDMPI_InBoundSourCoeff_Index, MGCMFDData_MPIComm_Num, -1);
		VectorInit_int_I(MGCMFDMPI_InBoundSourCoeff_Dir, MGCMFDData_MPIComm_Num, -1);
		VectorInit_FP_I(MGCMFDMPI_InBoundSourCoeff_GeoDim, MGCMFDData_MPIComm_Num, (FP_PRECISION)(0.0));
		SetCMFDMPI_InBoundSourCoeff_Index(ProcID, NumProcs, NumCMFDGroup, MOCNumLayersZ, MOCNumPinsInSubDom_X, \
			MOCNumPinsInSubDom_Y, CMFDComm_TarProc, CMFDComm_DirBound, MGCMFDData_MPIComm_SumInd, RayTracModDim, \
			WidthLayersInCore, MGCMFDMPI_InBoundSourCoeff_Index, MGCMFDMPI_InBoundSourCoeff_Dir, MGCMFDMPI_InBoundSourCoeff_GeoDim);
#if defined(M_CG_CMFD)
        CGCMFDBoundInfoComm_PreNeibInfo(ProcID,  MOCNumLayersZ, MOCNumPinsInSubDom_X, MOCNumPinsInSubDom_Y, NumCoarseCMFDGroup, \
            CMFDComm_TarProc, CGCMFDComm_NumData, CMFDComm_DirBound, CGCMFDData_MPIComm_Index, CGCMFDData_MPIComm_Num, CGCMFDData_MPIComm_SumInd);
		VectorInit_int_I(CGCMFDMPI_InBoundSourCoeff_Index, CGCMFDData_MPIComm_Num, -1);
		VectorInit_int_I(CGCMFDMPI_InBoundSourCoeff_Dir, CGCMFDData_MPIComm_Num, -1);
		VectorInit_FP_I(CGCMFDMPI_InBoundSourCoeff_GeoDim, CGCMFDData_MPIComm_Num, (FP_PRECISION)(0.0));

		SetCMFDMPI_InBoundSourCoeff_Index(ProcID, NumProcs, NumCoarseCMFDGroup, MOCNumLayersZ, MOCNumPinsInSubDom_X, \
			MOCNumPinsInSubDom_Y, CMFDComm_TarProc, CMFDComm_DirBound, CGCMFDData_MPIComm_SumInd, RayTracModDim, \
			WidthLayersInCore, CGCMFDMPI_InBoundSourCoeff_Index, CGCMFDMPI_InBoundSourCoeff_Dir, CGCMFDMPI_InBoundSourCoeff_GeoDim);
#endif // defined(M_CG_CMFD)
#endif // defined(M_MG_CMFD)
#ifdef TIME_REC
	    _time_end = TallyCurrentTime_CPU();
		TimePrePart[TimeCountPrePart] = _time_end - _time_start;
	    strcpy(TimeMetricsPrePart[TimeCountPrePart], "Prepare Info for MPI Comm");
	    TimeCountPrePart++;
#endif // TIME_REC
	}
#endif //MPI_USED
}
/**
 * @ Function: MOC_EigenValueProblem()
 *   introduction: MOC driver function for solving the transport sweep problem
 * @ Parameter List: ProblemType = 0, transport sweep problem, = 1, fixed source problem
 *   Return: void
 * @ Created     : 2019/09/13, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */

void MOC_EigenValueProblem()
{
	ProblemType = 0;
	CheckExpInteRange(ProcID, IsCheckExpInteRange, 0, MOCNumLayersZ, 0, NumGroups);
	
	CPU_MOC_EigenValueProblem();

	GPU_MOC_EigenValueProblem();
    //Aux_GPU_MOC_EigenValueProblem();

}

void CPU_MOC_EigenValueProblem()
{
	int _iter=0, _bool = 0/*, _width1, _width2, _width3*//*, _trSet, _setNumThreads, _threadID, _curr_cpu*/, _width_time = 32;
	// int _matrixSize=10;
	FP_PRECISION _kerr = 0.0, _t1 = 0.0, _t2 = 0.0, _t1_1 = 0.0, _t2_2 = 0.0, _MOCkerr = 0.0;
	FP_PRECISION_TIME _time_start = 0., _time_end = 0./*, _time_tot*/;
	//FP_PRECISION _temp;
	int _width_cmfd = 20, _width_cmfd_1 = 11, _width1=11;

#if defined(DEFAULT_TIME)
	FP_PRECISION_TIME _default_time_start=0, _default_time_end=0;
#endif // defined(DEFAULT_TIME)

#if !defined(CUDA_USED) || (defined(CUDA_USED) && defined(OMP_USED))
#ifdef OMP_USED
	if (HardwareMap_SubDom[ProcID] == 0) {
#endif // OMP_USED
		MOC_InitTransSweep(ProcID);

    int num_moc_steps_b4_cmfd = MaxNumOuterIter_MOC;
#ifdef M_MG_CMFD
    if (NumScattOrder > 0) {
        num_moc_steps_b4_cmfd = 0;
    }
#endif

//#if !defined(M_MG_CMFD) || (defined(M_MG_CMFD) && defined(MOCthenCMFD))/* || defined(MOC_FixSourProb)*/
        if (ProcID == 0) {
			printf("\n [  INFO   ]  %-*s%-*s%-*s%-*s%-*s",_width1, "k_eff", _width1, "errEig", \
				_width1, "Ferrlinf", _width1, "Ferrl2", _width1, "FerrRMSE");
			fprintf(LogFile, "\n              %-*s%-*s%-*s%-*s%-*s",_width1, "k_eff", _width1, "errEig", \
				_width1, "Ferrlinf", _width1, "Ferrl2", _width1, "FerrRMSE");
		}

		for (_iter = 0; _iter < num_moc_steps_b4_cmfd; _iter++) {
                if (ProcID == 0) {
				printf("\n [ITER-%-4d]  ", _iter + 1);
				fprintf(LogFile, "\n [ITER-%-4d]  ", _iter + 1);
			}

			/* source evaluation */
			if (ProblemType == 0) {
				if (Input_MATorXSs == 100 || Input_Type == 100) {
					Jacobi_ComputeFSRSource_DH(ProcID, _iter, MOCKeffOld, ScalarFlux_MOC, Source, NumGroups, MOCNumLayersZ, NumPinsInSubDomPerLayer, \
						NumFSRsInCell, XS_leakage_SubDom, SumNumFSRsBasedGlobCellID, Chi, MacroXS_nf, MacroXS_s, MacroXS_t);
				}
				else /*if (Input_MATorXSs <= 1 && Input_Type < 100)*/ {
					if (MOCSolverType == 1) { /*jacobi - solver*/
						// Jacobi_ComputeFSRSource(ProcID, _iter, MOCKeffOld, ScalarFlux_MOC, Source, NumGroups, MOCNumLayersZ, \
						//     NumPinsInSubDomPerLayer, NumFSRsInCell, XS_leakage_SubDom, SumNumFSRsBasedGlobCellID, Chi, MacroXS_nf, \
						// 	MacroXS_s, MacroXS_t, MacroXS_s_NZRowStartID, MacroXS_s_NZColStartID);
                        if (CPU_TSFunctionType>=11) {
                            FissSourEval(ProcID, _iter, TotalNumFSRs_subDom, NumGroups, MOCKeffOld, /*ScalarFlux_MOC*/ScalarFluxOld, Source, Chi, MacroXS_nf);
		// Printf_Vector_FP(TotalNumFSRs_subDom*NumGroups, Source, "calc Source");

                        else {
                            Jacobi_ComputeFSRSource_OMP(TotalNumFSRs_subDom, ProcID, _iter, MOCKeffOld, /*ScalarFlux_MOC*/ScalarFluxOld, Source, NumGroups, MOCNumLayersZ, \
                                NumPinsInSubDomPerLayer, NumFSRsInCell, XS_leakage_SubDom, SumNumFSRsBasedGlobCellID, Chi, MacroXS_nf, \
                                MacroXS_s, MacroXS_t, MacroXS_s_NZRowStartID, MacroXS_s_NZColStartID);
                        }
//						 Printf_Vector_FP(TotalNumFSRs_subDom*NumGroups, Source, "Source");
//						 Printf_Vector_FP(TotalNumFSRs_subDom*NumGroups, ScalarFluxOld, "ScalarFlux_MOC");
						// Printf_Vector_FP(TotalNumFSRs_subDom*NumGroups, MacroXS_t, "MacroXS_t");
						// Printf_Vector_FP(TotalNumFSRs_subDom*NumGroups, MacroXS_nf, "MacroXS_nf");
						// Printf_Vector_FP(TotalNumFSRs_subDom*NumGroups, Chi, "Chi");
						// int ii;
						// for (ii=0;ii<TotalNumFSRs_subDom*NumGroups;ii++) {
						// 	printf("\n MacroXS_s_NZRowStartID[%3d]=%4d, MacroXS_s_NZColStartID=%4d",ii,MacroXS_s_NZRowStartID[ii],MacroXS_s_NZColStartID[ii]);
						// }
						// Printf_Vector_FP(MacroXS_s_NNZ, MacroXS_s, "MacroXS_s");
						// exit(12);
					}
				}
			}

#ifdef TIME_REC // barrier for transfer angFlux data from cpu to gpu
#ifdef MPI_USED
			MPI_Barrier(MPI_COMM_WORLD);
#endif // MPI_USED
#endif // TIME_REC

            /* transport sweep without CMFD acceleration */
			CPU_MOCTransportSweep(ProcID, _iter, NumOMPThread);

#ifdef MPI_USED
			MPI_Barrier(MPI_COMM_WORLD);
#endif // MPI_USED

			/* convergence check */
			_bool = ConvergenceCheck(ProcID, _iter,0,MOCNumLayersZ,0,NumGroups);
			if (_bool == 1) {
				break;
			}
		}

		NumIters = _iter + 1;
//#endif // !defined(M_MG_CMFD) || (defined(M_MG_CMFD) && defined(MOCthenCMFD))

#if defined(M_MG_CMFD) && !defined(MOCthenCMFD) /*&& !defined(MOC_FixSourProb)*/
		CMFD_Initialization(ProcID, NumProcs);
		Output_CMFD_GroupConstant(CaseId, ProcID, 0, 0, CMFDKeff);

#if defined(M_CMFD_DEBUG)
		Output_CMFD_GroupConstant(CaseId, ProcID, 0, 0, CMFDKeff);

#if defined(M_CG_CMFD)
		Output_CMFD_GroupConstant(CaseId, ProcID, 1, 0, CMFDKeff);
#endif // defined(M_CG_CMFD)
#endif // defined(M_CMFD_DEBUG)


		for (_iter = 0; _iter < MaxNumOuterIter_MOC; _iter++) {
			// printf("\n [ITER-%-4d]  Number of Linear Solver Iterations per M_MG_CMFD calculation", _iter+1);
			if (ProblemType == 0 ) {
				if (ProcID == 0 && isTransient < 3) {
#if defined(M_CG_CMFD)
				printf("\n [ITER-%-4d]  %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s", _iter + 1, _width_cmfd_1, "MGCMFDIter", _width_cmfd_1, "#CGCMFD", \
					_width_cmfd_1, "k_eff", _width_cmfd_1, "errEig", _width_cmfd_1, "Qerrlinf", \
					_width_cmfd_1, "Qerrl2", _width_cmfd_1, "Ferrlinf", _width_cmfd_1, "# NegFlux", _width_cmfd_1, "% NegFlux");
				fprintf(LogFile, "\n [ITER-%-4d]  %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s", _iter + 1, _width_cmfd_1, "MGCMFDIter", _width_cmfd_1, "#CGCMFD", \
					_width_cmfd_1, "k_eff", _width_cmfd_1, "errEig", _width_cmfd_1, "Qerrlinf", \
					_width_cmfd_1, "Qerrl2", _width_cmfd_1, "Ferrlinf", _width_cmfd_1, "# NegFlux", _width_cmfd_1, "% NegFlux");
#else
#if defined(M_DetailLSInfo)
					printf("\n [ITER-%-4d]  ", _iter + 1);
					fprintf(LogFile, "\n [ITER-%-4d]  ", _iter + 1);
#else
					// printf("\n [ITER-%-4d]  %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s", _iter + 1, _width_cmfd_1, "MGCMFDIter", \
					// 	_width_cmfd_1, "k_eff", _width_cmfd_1, "errEig", _width_cmfd_1, "Qerrlinf", \
					// 	_width_cmfd_1, "Qerrl2", _width_cmfd_1, "Ferrlinf", _width_cmfd_1, "# NegFlux", _width_cmfd_1, "% NegFlux");
					// fprintf(LogFile, "\n [ITER-%-4d]  %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s", _iter + 1, _width_cmfd_1, "MGCMFDIter", \
					// 	_width_cmfd_1, "k_eff", _width_cmfd_1, "errEig", _width_cmfd_1, "Qerrlinf", \
					// 	_width_cmfd_1, "Qerrl2", _width_cmfd_1, "Ferrlinf", _width_cmfd_1, "# NegFlux", _width_cmfd_1, "% NegFlux");

					printf("\n [ITER-%-4d]  %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s", _iter + 1, _width_cmfd_1, "MGCMFDIter", \
						_width_cmfd_1, "k_eff", _width_cmfd_1, "errEig", _width_cmfd_1, "Ferrlinf", _width_cmfd_1, "Ferrl2", \
						_width_cmfd_1, "FerrRMSE", _width_cmfd_1, "# NegFlux", _width_cmfd_1, "% NegFlux", _width_cmfd_1, "# LSIters", \
						_width_cmfd_1, "LSFlinf", _width_cmfd_1, "LSFl2", _width_cmfd_1, "LSFRmes");
					fprintf(LogFile, "\n [ITER-%-4d]  %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s", _iter + 1, _width_cmfd_1, "MGCMFDIter", \
						_width_cmfd_1, "k_eff", _width_cmfd_1, "errEig", _width_cmfd_1, "Ferrlinf", _width_cmfd_1, "Ferrl2", _width_cmfd_1, \
						"FerrRMSE", _width_cmfd_1, "# NegFlux", _width_cmfd_1, "% NegFlux", _width_cmfd_1, "# LSIters", \
						_width_cmfd_1, "LSFlinf", _width_cmfd_1, "LSFl2", _width_cmfd_1, "LSFRmes");

#endif // defined(M_DetailLSInfo)
#endif // defined(M_CG_CMFD)
					fflush(stdout);
				}

			    CMFD_ParaUpdate();
			}
			else {
				if (ProcID == 0) {
					printf("\n [ITER-%-4d]  ", _iter + 1);
					fprintf(LogFile, "\n [ITER-%-4d]  ", _iter + 1);
				}
			}

#if defined(DEFAULT_TIME)
			_default_time_start = TIME_RECORD;
#endif // defined(DEFAULT_TIME)
#if defined(PTESC_LIB)
			CMFDKeff = CMFD_Solver_KSP(MOCNumLayersZ*MOCNumPinsInSubDom_X*MOCNumPinsInSubDom_Y, \
				NumCMFDGroup, CMFDMatrix_A, CMFDScalarFlux, CMFDScalarFlux_Old, CMFDTotalSource, \
				Index_1DArray, CMFDKeff_Old, CMFDXs_nf, CMFD_Chi, CMFDXs_s, RTMWidth_X, RTMWidth_Y, CMFD_NodeVol);
#else
                CMFDKeff = MGCMFD_Solver(ProcID, MOC_NumPinsInSubDom, NumCMFDGroup, CMFDKeff_Old, CMFDScalarFlux, \
                CMFDScalarFlux_Old, CMFDScalarFlux_Old_LS, CMFDTotalSource, CellRTMType_SubDom, CMFDXs_nf, \
                CMFD_Chi, CMFDXs_s, MatrixCSR_RowStartInd, MatrixCSR_ColInd, MatrixCSR, MG_DiagPointer, \
                SOR_Factor_MG, ConvCritLinearSolver_MGCMFD, MaxNumIter_MGLS, MaxNumOuterIter_MGCMFD, \
                CMFD_NodeVol, MGCMFD_LinearSolver, _iter);
#endif // defined(PTESC_LIB)
#if defined(DEFAULT_TIME)
                _default_time_end = TIME_RECORD;
                TimeRec_MGCMFD_CMFDSolver += _default_time_end - _default_time_start;
#endif // defined(DEFAULT_TIME)

                /* Update MOC initial Info by M_MG_CMFD solution */
#if defined(DEFAULT_TIME)
                _default_time_start = TIME_RECORD;
#endif // defined(DEFAULT_TIME)
#if defined(M_YQZ_CMFD_Conv)
                if (MGCMFD_Conv == 1) {
#endif // defined(M_YQZ_CMFD_Conv)
                if (MGCMFD_FluxFlag[0] == 0 && MGCMFD_Diver[0] == 0) {

                    MOCKeff = CMFDKeff * MGCMFD_damp + (1.0 - MGCMFD_damp) * MOCKeff;
                    UpdateScalFlux_CMFDtoMOC(CMFDScalarFlux, /*ScalarFlux_MOC*/ScalarFluxOld, CMFDHetScalFlux,
                                             CMFDGroupBound, NumCMFDGroup, \
                        MOC_NumPinsInSubDom, NumGroups, NumFSRsInRTM, CellRTMType_SubDom, SumNumFSRsInRTM,
                                             SumNumFSRsBasedGlobCellID, \
                        (FP_PRECISION) (MGCMFD_damp), _iter, NumScattOrder, TotalNumFSRs_subDom);
//				 Printf_Vector_FP(TotalNumFSRs_subDom*NumGroups*(NumScattOrder+1)*(NumScattOrder+1), ScalarFluxOld, "A4 CMFD-->MOC flux");
//                    Printf_Vector_FP(10, ScalarFluxOld, "A4 CMFD-->MOC flux");
                } else {
                    if (MGCMFD_FluxFlag[0] == 1) {
                        printf("\n              Negative MG CMFD flux, jump without rebalance");
                    }
                    if (MGCMFD_Diver[0] == 1) {
                        printf("\n              The MGCMFD iteration divergence, jump without rebalance");
                    }
                }
#if defined(M_YQZ_CMFD_Conv)
                }
                else {
                    printf("\n              The MG CMFD iteration is not converged, jump without rebalance");
                }
#endif // defined(M_YQZ_CMFD_Conv)

#if defined(M_3DMOCex)
#if defined(M_3DAxialFluxSOR)
                UpdateAxialAngFlux_CMFDtoMOC_SOR(MOCNumLayersZ, NumPinsInSubDomPerLayer, NumGroups, \
                NumAzimAng[0], NumPolarAng[0], NumCMFDGroup, CMFDGroupBound, \
                FSRAveAngFlux_UpAllLay, CMFDScalarFlux, CMFDHetScalFlux, MOC_NumPinsInSubDom,\
                NumFSRsInRTM, CellRTMType_SubDom, SumNumFSRsBasedGlobCellID);
                UpdateAxialAngFlux_CMFDtoMOC_SOR(MOCNumLayersZ, NumPinsInSubDomPerLayer, NumGroups, \
                NumAzimAng[0], NumPolarAng[0], NumCMFDGroup, CMFDGroupBound, \
                FSRAveAngFlux_DownAllLay, CMFDScalarFlux, CMFDHetScalFlux, MOC_NumPinsInSubDom,\
                NumFSRsInRTM, CellRTMType_SubDom, SumNumFSRsBasedGlobCellID);
#endif // defined(M_3DAxialFluxSOR)
#endif // defined(M_3DMOCex)

#if defined(DEFAULT_TIME)
                _default_time_end = TIME_RECORD;
                TimeRec_MGCMFD_FluxUpdate += _default_time_end - _default_time_start;
#endif // defined(DEFAULT_TIME)

#if defined(DEFAULT_TIME)
                _default_time_start = TIME_RECORD;
#endif // defined(DEFAULT_TIME)
			/********************************************************************************/
			if (MOCSolverType == 1) { /*jacobi - solver*/
				if (ProblemType == 0) {
					if (Input_MATorXSs == 100 || Input_Type == 100) {
						Jacobi_ComputeFSRSource_DH(ProcID, _iter, MOCKeff, ScalarFlux_MOC, Source, NumGroups, MOCNumLayersZ, NumPinsInSubDomPerLayer, \
							NumFSRsInCell, XS_leakage_SubDom, SumNumFSRsBasedGlobCellID, Chi, MacroXS_nf, MacroXS_s, MacroXS_t);

					}
					else if (Input_MATorXSs <= 1 && Input_Type < 100) {
						// Jacobi_ComputeFSRSource(ProcID, _iter, MOCKeff, ScalarFlux_MOC, Source, NumGroups, MOCNumLayersZ, \
						//     NumPinsInSubDomPerLayer, NumFSRsInCell, XS_leakage_SubDom, SumNumFSRsBasedGlobCellID, Chi, \
						// 	MacroXS_nf, MacroXS_s, MacroXS_t, MacroXS_s_NZRowStartID, MacroXS_s_NZColStartID);
                            if (CPU_TSFunctionType >= 11) {
                                FissSourEval(ProcID, _iter, TotalNumFSRs_subDom, NumGroups, MOCKeffOld, ScalarFluxOld, Source, Chi, MacroXS_nf);
                            // Printf_Vector_FP(TotalNumFSRs_subDom*NumGroups, Source, "fiss sour");
                            } else {
                                Jacobi_ComputeFSRSource_OMP(TotalNumFSRs_subDom, ProcID, _iter, MOCKeffOld, /*ScalarFlux_MOC*/ScalarFluxOld, Source, \
                                                        NumGroups, MOCNumLayersZ, NumPinsInSubDomPerLayer, NumFSRsInCell, XS_leakage_SubDom,
                                                            SumNumFSRsBasedGlobCellID, \
                                                        Chi, MacroXS_nf, MacroXS_s, MacroXS_t, MacroXS_s_NZRowStartID, MacroXS_s_NZColStartID);
                            }
					}
				}
			}
			/********************************************************************************/
			/* MOC transport Sweep */
			CPU_MOCTransportSweep(ProcID, _iter, NumOMPThread);


//            fissRate_tmp = CompReacRate_TotalDom(ProcID, 0, ScalarFlux_MOC, MOCNumLayersZ, NumPinsInSubDomPerLayer,
//                                                 NumFSRsInCell, NumGroups, CellRTMType_SubDom, MacroXS_nf, InpXSs_nf,
//                                                 RTModFSRsInfo_HS, SumNumFSRsBasedGlobCellID);
//            if (_iter == 0) k = MOCKeff;
//            else k = fissRate_tmp * k_old / fissRate_old;
//            printf("\n fissRate_tmp = %e, fissRate_old = %e, k = %e, k_old = %e, err = %e", fissRate_tmp, fissRate_old, k, k_old, fabs(k - k_old)/k_old);
//            printf("\n MOCGlobFissRateTemp=%e, MOCGlobFissRateOld=%e, MOCKeff = %e, MOCKeff_Old = %e", MOCGlobFissRateTemp,MOCGlobFissRateOld, MOCKeff, MOCKeffOld);
//            printf("\n CMFDGlobFissRateTemp=%e, CMFDGlobFissRateOld=%e, MOCKeff = %e, MOCKeff_Old = %e", CMFDGlobalFissRate[0],CMFDGlobalFissRate_Old[0], MOCKeff, MOCKeffOld);
//            k_old = k;
//            fissRate_old = fissRate_tmp;

#if defined(DEFAULT_TIME)
			_default_time_end = TIME_RECORD;
			TimeRec_MOCSolution += _default_time_end - _default_time_start;
#endif // defined(DEFAULT_TIME)


			_bool = ConvergenceCheck(ProcID, _iter,0,MOCNumLayersZ,0,NumGroups);
			if (_bool == 1) {
				break;
			}
#if defined(M_YQZ_CMFD_Damp)
			MGCMFD_damp = fmax(Min_damp, MGCMFD_damp * MGCMFD_DampFactor);
#endif // defined(M_YQZ_CMFD_Damp)

			MGCMFD_GCEvaluation(ProcID, NumGroups, 0, _iter+1);

#if defined(M_CMFD_DEBUG)
			Output_CMFD_GroupConstant(CaseId, ProcID, 0, _iter + 1, CMFDKeff);
#if defined(M_CG_CMFD)
			Output_CMFD_GroupConstant(CaseId, ProcID, 1, _iter + 1, CMFDKeff);
#endif // defined(M_CG_CMFD)
#endif //#if defined(M_CMFD_DEBUG)

		}
#endif // defined(M_MG_CMFD) && !defined(MOCthenCMFD)

#ifdef OMP_USED
	} //if (HardwareMap_SubDom[ProcID] == 1) {
#endif // OMP_USED
#endif // !defined(CUDA_USED) || (defined(CUDA_USED) && defined(OMP_USED))
}

void GPU_MOC_EigenValueProblem()
{
	int _iter=0, _bool = 0, _width1 =11/*, _width2, _width3*//*, _trSet, _setNumThreads, _threadID, _curr_cpu*/, _width_time = 32;
	// int _matrixSize=10;
	FP_PRECISION _kerr = 0.0, _t1 = 0.0, _t2 = 0.0, _t1_1 = 0.0, _t2_2 = 0.0, _MOCkerr = 0.0;
	FP_PRECISION_TIME _time_start = 0., _time_end = 0./*, _time_tot*/;
	//FP_PRECISION _temp;
	//double _cudaMemUse[1], _cudaMemTot[1];
	int _width_cmfd = 20, _width_cmfd_1 = 11;
#if defined(CUDA_USED)
    FP_PRECISION_TIME _ts_gpu_TS, _te_gpu_TS;
#endif // defined(CUDA_USED)

//#if defined(CUDA_USED) && defined(M_MG_CMFD)
//    int _mocIter;
//#endif // defined(CUDA_USED) && defined(M_MG_CMFD)

#if defined(DEFAULT_TIME)
	FP_PRECISION_TIME _default_time_start=0, _default_time_end=0;
#endif // defined(DEFAULT_TIME)

#ifdef CUDA_USED
#ifdef OMP_USED
	if (HardwareMap_SubDom[ProcID] == 1) {
#endif // OMP_USED
        // PrintCPUMemUsage(ProcID);
        /* copy XSs info from CPU (external file or Res calc) */
		GPUInitMacroXSs(ProcID, TotalNumFSRs_subDom, NumGroups, MOCNumLayersZ, \
			MacroXS_s_NZRowStartID_Layer, MacroXS_s_NZColStartID, GPUPreMemXSs,\
			IfResonance, ProblemType, MacroXS_s_MaxNNZ_Layer, LogFile);

//		OutputMacroXSs_FSRs_GPU(ProcID, MOCNumLayersZ, TotalNumFSRs_subDom, NumPinsInSubDomPerLayer, NumFSRsInCell, SumNumFSRsBasedGlobCellID, \
//		    TotalNumFSRs_perLayer, NumGroups, MacroXS_s_NZStartID_Layer, MacroXS_s_NNZ_Layer, ProblemType, MacroXS_t, \
//			MacroXS_nf, Chi, MacroXS_s, FixedSour, OUTPUT_MacorXSs, MacroXS_s_MaxNNZ_Layer, CaseId, NumScattOrder,\
//	        MacroXS_s_NZRowStartID_Layer, MacroXS_s_NZColStartID, ScalarFluxOld);

		/* Initialize transport sweep*/
		GPUTransportSweepInit_1(ProcID, NumProcs, TotalNumFSRs_subDom, NumGroups,MOCNumLayersZ* NumLTs_HS[0] * 2, NumPolarAng[0]);

#ifdef M_2DMOC
    GPUCopyMacroXSsHostToDevice_Layer(0, TotalNumFSRs_subDom, TotalNumFSRs_perLayer, NumGroups, MacroXS_s_NZStartID_Layer, \
	    MacroXS_s_NNZ_Layer, ProblemType, MacroXS_t, MacroXS_a, MacroXS_f, MacroXS_nf, Chi, MacroXS_s, FixedSour, NumScattOrder,\
	    MacroXS_s_NZRowStartID_Layer, MacroXS_s_NZColStartID, 0, ScalarFluxOld);
#endif

#if defined(M_MG_CMFD) && !defined(MOCthenCMFD)
		/* Initialize M_MG_CMFD calculation */
		GPUCMFD_Initialization();
#endif // defined(M_MG_CMFD) && !defined(MOCthenCMFD)

		// _trSet = 0;
		_ts_gpu_TS = TallyCurrentTime_CPU();

// #if !defined(GPU_STREAM_USED)
// 		/*copy boundary flux from host to device */
// 		if (NumProcs > 1) {
// 			GPUCopyBoundAngFluxHostToDevice_1(_iter,0);
// 		}
// #endif // !defined(GPU_STREAM_USED)
			/* iteration */
			
		// if (ProcID==0) {
		// 	printf("\n [  TITLE  ]  ************************************************************");
   		// 	printf("\n [  TITLE  ]  *        Transport Sweep On GPU - Eigenvalue Problem       *");
       	// 	printf("\n [  TITLE  ]  *                Jacobi Iteration Scheme                   *");
		// 	printf("\n [  TITLE  ]  ************************************************************");
		// 	fflush(stdout);
		// 	fprintf(LogFile, "\n [  TITLE  ]  ************************************************************");
        // 	fprintf(LogFile, "\n [  TITLE  ]  *        Transport Sweep On GPU - Eigenvalue Problem       *");
        // 	fprintf(LogFile, "\n [  TITLE  ]  *                Jacobi Iteration Scheme                   *");
		// 	fprintf(LogFile, "\n [  TITLE  ]  ************************************************************");
		// }
#if !defined(M_MG_CMFD)
        if (ProcID == 0) {
			printf("\n              %-*s%-*s%-*s%-*s%-*s",_width1, "k_eff", _width1, "errEig", \
				_width1, "Ferrlinf", _width1, "Ferrl2", _width1, "FerrRMSE");
			fprintf(LogFile, "\n              %-*s%-*s%-*s%-*s%-*s",_width1, "k_eff", _width1, "errEig", \
				_width1, "Ferrlinf", _width1, "Ferrl2", _width1, "FerrRMSE");
		}
#endif // !defined(M_MG_CMFD)

#ifdef M_MG_CMFD
    if (NumScattOrder > 0) {
        for (_iter = 0; _iter < 500; _iter++) {
            printf("\n[MOC b4 CMFD] ITER=%d", _iter);
            GPUTransportSweep_1(_iter);
        }
    }
#endif

		for (_iter = 0; _iter < MaxNumOuterIter_MOC; _iter++) {
            if (ProcID==0) {
#if !defined(M_MG_CMFD) || (defined(M_MG_CMFD) && defined(MOCthenCMFD))
            
			printf("\n [ITER-%-4d]  ", _iter + 1);
			fprintf(LogFile, "\n [ITER-%-4d]  ", _iter + 1);
			
#endif // !defined(M_MG_CMFD) || (defined(M_MG_CMFD) && defined(MOCthenCMFD))

#if defined(M_MG_CMFD) && !defined(MOCthenCMFD)/**/
#if defined(M_CG_CMFD)
#if defined(M_DetailLSInfo)
				printf("\n [ITER-%-4d]  ", _iter + 1, _width_cmfd_1);
				printf("\n              %-*s%-*s%-*s%-*s%-*s%-*s", _iter + 1, _width_cmfd_1, "MGCMFDIter", _width_cmfd_1, "#CGCMFD", \
					_width_cmfd_1, "k_eff", _width_cmfd_1, "k_err", _width_cmfd_1, "fluxRMSE", _width_cmfd_1, "totLSIter");
#else
				printf("\n [ITER-%-4d]  %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s", _iter + 1, _width_cmfd_1, "MGCMFDIter", _width_cmfd_1, "#CGCMFD", \
					_width_cmfd_1, "k_eff", _width_cmfd_1, "errEig", _width_cmfd_1, "Qerrlinf", _width_cmfd_1, "Qerrl2", \
					_width_cmfd_1, "Ferrlinf", _width_cmfd_1, "# NegFlux", _width_cmfd_1, "% NegFlux", _width_cmfd_1, "# LSIters", \
					_width_cmfd_1, "LSFlinf", _width_cmfd_1, "LSFl2", _width_cmfd_1, "LSFRmes");
				fprintf(LogFile, "\n [ITER-%-4d]  %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s", _iter + 1, _width_cmfd_1, "MGCMFDIter", _width_cmfd_1, "#CGCMFD", \
					_width_cmfd_1, "k_eff", _width_cmfd_1, "errEig", _width_cmfd_1, "Qerrlinf", \
					_width_cmfd_1, "Qerrl2", _width_cmfd_1, "Ferrlinf", _width_cmfd_1, "# NegFlux", _width_cmfd_1, "% NegFlux", \
					_width_cmfd_1, "# LSIters", _width_cmfd_1, "LSFlinf", _width_cmfd_1, "LSFl2", _width_cmfd_1, "LSFRmes");
#endif // defined(M_DetailLSInfo)
#else
				// printf("\n [ITER-%-4d]  %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s", _iter + 1, _width_cmfd_1, "MGCMFDIter", \
				// 	_width_cmfd_1, "k_eff", _width_cmfd_1, "errEig", _width_cmfd_1, "Qerrlinf", \
				// 	_width_cmfd_1, "Qerrl2", _width_cmfd_1, "Ferrlinf", _width_cmfd_1, "# NegFlux", _width_cmfd_1, "% NegFlux");
				// fprintf(LogFile, "\n [ITER-%-4d]  %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s", _iter + 1, _width_cmfd_1, "MGCMFDIter", \
				// 	_width_cmfd_1, "k_eff", _width_cmfd_1, "errEig", _width_cmfd_1, "Qerrlinf", \
				// 	_width_cmfd_1, "Qerrl2", _width_cmfd_1, "Ferrlinf", _width_cmfd_1, "# NegFlux", _width_cmfd_1, "% NegFlux");
				
				printf("\n [ITER-%-4d]  %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s", _iter + 1, _width_cmfd_1, "MGCMFDIter", \
						_width_cmfd_1, "k_eff", _width_cmfd_1, "errEig", _width_cmfd_1, "Ferrlinf", \
						_width_cmfd_1, "Ferrl2", _width_cmfd_1, "FerrRMSE", _width_cmfd_1, "# NegFlux", _width_cmfd_1, "% NegFlux",\
						_width_cmfd_1, "# LSIters", _width_cmfd_1, "LSFlinf", _width_cmfd_1, "LSFl2", _width_cmfd_1, "LSFRmes");
				fprintf(LogFile, "\n [ITER-%-4d]  %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s", _iter + 1, _width_cmfd_1, "MGCMFDIter", \
						_width_cmfd_1, "k_eff", _width_cmfd_1, "errEig", _width_cmfd_1, "Ferrlinf", \
						_width_cmfd_1, "Ferrl2", _width_cmfd_1, "FerrRMSE", _width_cmfd_1, "# NegFlux", _width_cmfd_1, "% NegFlux",\
					_width_cmfd_1, "# LSIters", _width_cmfd_1, "LSFlinf", _width_cmfd_1, "LSFl2", _width_cmfd_1, "LSFRmes");
#endif // defined(M_CG_CMFD)
#endif //defined(M_MG_CMFD) && !defined(MOCthenCMFD)
				fflush(stdout);
            }

#if defined(M_MG_CMFD) && !defined(MOCthenCMFD)
			/* update M_MG_CMFD old values */
			//GPUCMFD_ParaUpdate();
			CMFDKeff_Old = CMFDKeff;

			/* M_MG_CMFD calculation on GPU */
#if defined(DEFAULT_TIME)
				_default_time_start = TIME_RECORD;
#endif // defined(DEFAULT_TIME)
#ifdef TIME_REC
	    _time_start = TallyCurrentTime_CPU();
#endif // TIME_REC
			CMFDKeff = GPU_MGCMFDSolver(_iter, CMFDKeff_Old);

#if defined(DEFAULT_TIME)
			_default_time_end = TIME_RECORD;
			TimeRec_MGCMFD_CMFDSolver += _default_time_end - _default_time_start;
			_default_time_start = TIME_RECORD;
#endif // defined(DEFAULT_TIME)
#ifdef TIME_REC
	    _time_end = TallyCurrentTime_CPU();
		if (_iter==0) {
			TimeStamp_CMFD=TimeCount;
			strcpy(TimeMetrics[TimeStamp_CMFD], "CMFD Calculation on GPU");
			TimeCount++;
		}
        TimeOverall[TimeStamp_CMFD] += _time_end - _time_start;
#endif // TIME_REC

#ifdef TIME_REC
	    _time_start = TallyCurrentTime_CPU();
#endif // TIME_REC
#if defined(M_YQZ_CMFD_Conv)
				//printf("\n MGCMFD_Conv=%d", MGCMFD_Conv);
				if (MGCMFD_Conv == 1) {
#endif // defined(M_YQZ_CMFD_Conv)

					if (MGCMFD_FluxFlag[0] == 0 && MGCMFD_Diver[0] == 0) {
						MOCKeff = CMFDKeff * MGCMFD_damp + (1.0 - MGCMFD_damp) * MOCKeff;

						GPUUpdateFlux_CMFDtoMOC(MOC_NumPinsInSubDom ,NumCMFDGroup, CMFDScalarFlux, CMFDHetScalFlux);
						UpdateScalFlux_CMFDtoMOC(CMFDScalarFlux, /*ScalarFlux_MOC*/ScalarFluxOld, CMFDHetScalFlux, CMFDGroupBound, NumCMFDGroup, \
						    MOC_NumPinsInSubDom, NumGroups, NumFSRsInRTM, CellRTMType_SubDom, SumNumFSRsInRTM, SumNumFSRsBasedGlobCellID, \
						    (FP_PRECISION)(MGCMFD_damp),_iter, NumScattOrder, TotalNumFSRs_subDom);
					}
					else {
						if (MGCMFD_FluxFlag[0] == 1) {
							printf("\n              Negative CMFD flux, jump without rebalance");
						}
						if (MGCMFD_Diver[0] == 1) {
							printf("\n             The MGCMFD iteration divergence, jump without rebalance");
						}
					}
#if defined(M_YQZ_CMFD_Conv)
				}
				else {
					printf("\n              The CMFD iteration is not converged, jump without rebalance");
				}
#endif // defined(M_YQZ_CMFD_Conv)
#ifdef TIME_REC
	    _time_end = TallyCurrentTime_CPU();
		if (_iter==0) {
			TimeStamp_CMFDtoMOC=TimeCount;
			strcpy(TimeMetrics[TimeStamp_CMFDtoMOC], "Flux Update CMFD->MOC on GPU");
			TimeCount++;
		}
        TimeOverall[TimeStamp_CMFDtoMOC] += _time_end - _time_start;
#endif // TIME_REC
				

#if defined(DEFAULT_TIME)
				_default_time_end = TIME_RECORD;
				TimeRec_MGCMFD_FluxUpdate += _default_time_end - _default_time_start;
#endif // defined(DEFAULT_TIME)
#endif // defined(M_MG_CMFD) && !defined(MOCthenCMFD)

#if defined(DEFAULT_TIME)
				_default_time_start = TIME_RECORD;
#endif // defined(DEFAULT_TIME)
				
#if !defined(GPU_STREAM_USED)
#ifndef OMP_ASYN
				/* source evaluation on GPU*/

				/* transport sweep on GPU */
				GPUTransportSweep_1(_iter);

#if defined(DEFAULT_TIME)
				_default_time_end = TIME_RECORD;
				TimeRec_MOCSolution += _default_time_end - _default_time_start;
#endif // defined(DEFAULT_TIME)
#ifndef CPUMPI_GPUTS_ASYNCH
#ifndef CUDA_AWARE_MPI
				if (NumProcs > 1) {
					/* copy boundary angular flux from GPU to CPU */
					// GPUCopyBoundAngFluxDeviceToHost_Synch_1(_iter);

					/* MPI communication, exchange boundary angular flux*/
#ifdef NVPROF_CPU
					nvtxRangePushA("MPI Comm");
#endif // NVPROF_CPU
#ifdef MPI_USED
					MPI_Barrier(MPI_COMM_WORLD);
#endif // MPI_USED
					BoundAngFluxExchange_MPIComm(ProcID, _iter, 0, MOCNumLayersZ);
#ifdef NVPROF_CPU
					nvtxRangePop();
#endif // NVPROF_CPU
				}
#endif // !CUDA_AWARE_MPI

// #ifdef CUDA_AWARE_MPI
// 				if (NumProcs > 1) {
// 					if (OUTPUT_CoarseCompTime) {
// 						_time_start = TIME_RECORD;
// 					}
// 					GPU_BoundValueUpdate(ProcID, NumProcs);
// 					if (OUTPUT_CoarseCompTime) {
// 						_time_end = TIME_RECORD;
// 						CompTime_MOC_UBAF_Comm += _time_end - _time_start;
// 					}
// #ifdef MPI_USED
// 					MPI_Barrier(MPI_COMM_WORLD);
// #endif // MPI_USED
// 				}
// #endif // CUDA_AWARE_MPI
#endif // !CPUMPI_GPUTS_ASYNCH

// #ifdef CPUMPI_GPUTS_ASYNCH
// 				if (NumProcs > 1) {
// 					/* MPI communication, exchange boundary angular flux*/
// #ifdef NVPROF_CPU
// 					nvtxRangePushA("MPI Comm");
// #endif // NVPROF_CPU
// 					BoundAngFluxExchange_MPIComm(ProcID, _iter, 0, MOCNumLayersZ);
// #ifdef NVPROF_CPU
// 					nvtxRangePop();
// #endif // NVPROF_CPU
// 					/* copy boundary angular flux from GPU to CPU */
// 					GPUCopyBoundAngFluxDeviceToHost_Synch_1(_iter);
// 				}
// #endif // CPUMPI_GPUTS_ASYNCH
#endif // !OMP_ASYN

// #ifdef OMP_ASYN
// 				_setNumThreads = 2;
// 				omp_set_num_threads(_setNumThreads);Output_CMFD_GroupConstant
// #pragma omp parallel private(_threadID) private(_Output_CMFD_GroupConstantstart) private(_time_end) private(_curr_cpu) private(CPUMPIFlux) private(BoundaryFlux)
// 				{
// 					_threadID = omp_get_thread_num()Output_CMFD_GroupConstant
// 					if (_threadID == 0) {
// #ifdef SCHED_LINUX
// 						_curr_cpu = sched_getcpu();
// 						printf("\n ProcID=%3d. ProcessorName=%6s, cpu is %3d, threadID=%3d", ProcID, ProcessorName, _curr_cpu, _threadID);
// 						fflush(stdout);
// #endif // SCHED_LINUX
// 						GPUTransportSweep_1(_iter);
						
// 						if (NumProcs > 1) {
// 							GPUCopyBoundAngFluxDeviceToHost_Synch();
// 						}
// 					}
// 					else {
// #ifdef SCHED_LINUX
// 						_curr_cpu = sched_getcpu();
// 						printf("\n ProcID=%3d. ProcessorName=%6s, cpu is %3d, threadID=%3d", ProcID, ProcessorName, _curr_cpu, _threadID);
// 						fflush(stdout);
// #endif // SCHED_LINUX
// 						if (NumProcs > 1) {
// #ifdef NVPROF_CPU
// 							nvtxRangePushA("MPI Comm");
// #endif // NVPROF_CPU
// 							BoundAngFluxExchange_MPIComm(ProcID, _iter, 0, MOCNumLayersZ);
// #ifdef NVPROF_CPU
// 							nvtxRangePop();
// #endif // NVPROF_CPU
// 						}
// 					}
// #pragma omp barrier
// 				}
// #endif // OMP_ASYN


				/*copy boundary flux from host to device */
				// if (NumProcs > 1) {
				// 	GPUCopyBoundAngFluxHostToDevice_1(_iter,1);
				// }



#endif // !GPU_STREAM_USED

// #ifdef GPU_STREAM_USED
// #ifndef OMP_ASYN
// 				GPUTransportSweep_GCMemcpy_Streams();
// 				/* MPI communication, exchange boundary angular flux*/
// #ifdef NVPROF_CPU
// 				nvtxRangePushA("MPI Comm");
// #endif // NVPROF_CPU
// 				BoundAngFluxExchange_MPIComm(ProcID, _iter, 0, MOCNumLayersZ);
// #ifdef NVPROF_CPU
// 				nvtxRangePop();
// #endif // NVPROF_CPU
// #endif // OMP_ASYN

// #ifdef OMP_ASYN
// 				_setNumThreads = 2;
// 				omp_set_num_threads(_setNumThreads);
// #pragma omp parallel private(_threadID) private(_time_start) private(_time_end) private(_curr_cpu) private(CPUMPIFlux) private(BoundaryFlux)
// 				{
// 					_threadID = omp_get_thread_num();
// 					if (_threadID == 0) {
// 						GPUTransportSweep_GCMemcpy_Streams();
// 					}
// 					else {
// #ifdef NVPROF_CPU
// 						nvtxRangePushA("MPI Comm");
// #endif // NVPROF_CPU
// 						BoundAngFluxExchange_MPIComm(ProcID, _iter, 0, MOCNumLayersZ);
// #ifdef NVPROF_CPU
// 						nvtxRangePop();
// #endif // NVPROF_CPU
// 					}
// #pragma omp barrier
// 				}
// #endif // OMP_ASYN

// #endif // GPU_STREAM_USED


#ifdef MPI_USED
				MPI_Barrier(MPI_COMM_WORLD);
#endif // MPI_USED
				/*convergence check*/

				// GPUCPU_TransFlux(TotalNumFSRs_subDom*NumGroups, ScalarFlux_MOC, MOC_Trans_Corre_SOR);

				_bool = ConvergenceCheck(ProcID, _iter,0,MOCNumLayersZ,0,NumGroups);

				if (_bool == 1 || _iter == MaxNumOuterIter_MOC  - 1) {
                    ScalarFluxCopy_Dev2Host(ScalarFluxOld, TotalNumFSRs_perLayer * NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1), 0, (char *)("ScalarFlux_MOC-D2H"));
                    ScalarFluxCopy_Dev2Host(ScalarFlux_MOC, TotalNumFSRs_perLayer * NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1), 0, (char *)("ScalarFlux_MOC-D2H"));
					break;
				}

#if defined(M_YQZ_CMFD_Damp)
				MGCMFD_damp = fmax(Min_damp, MGCMFD_damp * MGCMFD_DampFactor);

#endif // defined(M_YQZ_CMFD_Damp)

#if defined(M_MG_CMFD) && !defined(MOCthenCMFD)
				GPUCMFD_GCEvaluation(0, _iter + 1);

#if defined(M_CMFD_DEBUG)
                GPUAllocMemCMFDXSs_onCPU(0);
			    GPUCMFD_CheckCollapse(0, OUTPUT_CMFDGroupConstant,OUTPUT_CoarseMesh_GroupConstant, MOC_NumPinsInSubDom, MOCNumLayersZ, MOCNumPinsInSubDom_X, MOCNumPinsInSubDom_Y, \
					NumCMFDGroup, CMFD_NumNoneZeroValue_Matrix[0], CMFDHetScalFlux, CMFDXs_t, CMFDXs_a, CMFDXs_f, CMFDXs_nf, CMFD_Chi, CMFDXs_s, CMFDNodeDiffCoeff, \
					MOCtoCMFD_ParCurr_X_Pos, MOCtoCMFD_ParCurr_X_Neg, MOCtoCMFD_ParCurr_Y_Pos, MOCtoCMFD_ParCurr_Y_Neg, MGCMFDNetCurr_X, MGCMFDNetCurr_Y, \
					CMFDSurfDiffCoeff_X, CMFDSurfNDF_X, CMFDSurfDiffCoeff_Y, CMFDSurfNDF_Y, MGCMFD_LinearSolver, MatrixCSR, MG_ILUMatrix, \
					MatrixCSR_RowStartInd, MatrixCSR_ColInd, MGCMFDMPI_InBoundSourCoeff, MGCMFDData_MPIComm_Num);
			    Output_CMFD_GroupConstant(CaseId, ProcID, 0, _iter + 1, CMFDKeff);
#if defined(M_CG_CMFD)
		        Output_CMFD_GroupConstant(CaseId, ProcID, 1, _iter + 1, CMFDKeff);
#endif // defined(M_CG_CMFD)
				// Output_CMFD_GroupConstant(CaseId, ProcID, NumCMFDGroup,CMFDKeff);
#endif // defined(M_CMFD_DEBUG)
#endif // defined(M_MG_CMFD) && !defined(MOCthenCMFD)
			}

			NumIters = _iter + 1;
			// // _tEnd = TIME_RECORD;
			//t_end=clock();
			_te_gpu_TS = TIME_RECORD;
			TimeRec_TSGPU = _te_gpu_TS - _ts_gpu_TS;

			//collec_CompTime(ProcID, NumProcs);
			//GPUCPU_TransFlux();
			//MOC_Main_Output(ProcID);
			// MPI_Finalize();
			// return 0;
#ifdef OMP_USED
		} //if (HardwareMap_SubDom[ProcID] == 1) {
#endif // OMP_USED
#endif // CUDA_USED
}



/**
 * @ Function: CPUMOC_FixedSourceProblem_lay_g()
 *   introduction: MOC driver function for solving the fixed source problem
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2019/09/16, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
// void MOC_FixedSourceProblem_woResonance(int _procID)
// {
// 	/* prepare info for MPI comminication */
// 	ProblemType = 1;
// // #ifdef MPI_USED
// // 	if (NumProcs > 1) {
// // 		UpdateBoundAngFlux_PreNeibInfo(MOCNumLayersZ, NumProcPerLayer, MOC_subDomID_layer, _procID, Mess_TarProc, Mess_SendNumLTs, \
// // 			NumAzimAng[0], LongTracksInfo_Total[0], NumLongTracks[0], MPIProcID_recv, MPINeibID_recv, Mess_SendSatrtPInAngF, \
// // 			Mess_RecvSatrtPInAngF, NumPolarAng[0], NumGroups, NumLTs_HS[0], NumLTsSum[0], GCCopy_StartPInAngF, GCCopy_NumDataInAngF);
// // 	}
// // #endif //MPI_USED

// 	CPUMOC_FixedSourceProblem_lay_g(_procID, 0, MOCNumLayersZ, 0, NumGroups);
// }

/**
 * @ Function: CPU_MOCthenCMFD()
 *   introduction: MOC driver function for solving the transport sweep problem
 * @ Parameter List: 
 *   Return: void
 * @ Created     : 2019/09/13, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
/**
 * @ Function: GPU_MOCthenCMFD()
 *   introduction: the CMFD calculation is performed  after the MOC is converged
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2020/02/13, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
void GPU_MOCthenCMFD()
{
#if defined(CUDA_USED) && defined(M_MG_CMFD) && defined(MOCthenCMFD)
    int _width_cmfd_1 = 11;
	GPUCMFD_GCEvaluation(1, 0);

    GPUAllocMemCMFDXSs_onCPU(1);
	GPUCMFD_CheckCollapse(1, OUTPUT_CMFDGroupConstant,OUTPUT_CoarseMesh_GroupConstant, MOC_NumPinsInSubDom, MOCNumLayersZ, MOCNumPinsInSubDom_X, MOCNumPinsInSubDom_Y, \
		NumCMFDGroup, CMFD_NumNoneZeroValue_Matrix[0], CMFDHetScalFlux, CMFDXs_t, CMFDXs_a, CMFDXs_f, CMFDXs_nf, CMFD_Chi, CMFDXs_s, CMFDNodeDiffCoeff, \
		MOCtoCMFD_ParCurr_X_Pos, MOCtoCMFD_ParCurr_X_Neg, MOCtoCMFD_ParCurr_Y_Pos, MOCtoCMFD_ParCurr_Y_Neg, MGCMFDNetCurr_X, MGCMFDNetCurr_Y, \
		CMFDSurfDiffCoeff_X, CMFDSurfNDF_X, CMFDSurfDiffCoeff_Y, CMFDSurfNDF_Y, MGCMFD_LinearSolver, MatrixCSR, MG_ILUMatrix, \
		MatrixCSR_RowStartInd, MatrixCSR_ColInd, MGCMFDMPI_InBoundSourCoeff, MGCMFDData_MPIComm_Num);

	Output_CMFD_GroupConstant(CaseId, ProcID, 0, 0, CMFDKeff);
#if defined(M_CG_CMFD)
	Output_CMFD_GroupConstant(CaseId, ProcID, 1, 0, CMFDKeff);
#endif // defined(M_CG_CMFD)

    if (ProcID==0) {
        printf("\n [ MG CMFD ]  %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s", _width_cmfd_1, "MGCMFDIter", \
		    _width_cmfd_1, "k_eff", _width_cmfd_1, "errEig", _width_cmfd_1, "Qerrlinf", \
		    _width_cmfd_1, "Qerrl2", _width_cmfd_1, "Ferrlinf", _width_cmfd_1, "# NegFlux", _width_cmfd_1, "% NegFlux",\
			_width_cmfd_1, "# LSIters", _width_cmfd_1, "LSFlinf", _width_cmfd_1, "LSFl2", _width_cmfd_1, "LSFRmes");
	    fprintf(LogFile, "\n [ MG CMFD ]  %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s", _width_cmfd_1, "MGCMFDIter", \
		    _width_cmfd_1, "k_eff", _width_cmfd_1, "errEig", _width_cmfd_1, "Qerrlinf", \
		    _width_cmfd_1, "Qerrl2", _width_cmfd_1, "Ferrlinf", _width_cmfd_1, "# NegFlux", _width_cmfd_1, "% NegFlux",\
			_width_cmfd_1, "# LSIters", _width_cmfd_1, "LSFlinf", _width_cmfd_1, "LSFl2", _width_cmfd_1, "LSFRmes");
	}
	CMFDKeff = GPU_MGCMFDSolver(0, MOCKeff);

#endif //  defined(CUDA_USED) && defined(M_MG_CMFD) && defined(MOCthenCMFD)

}

/**
 * @ Function: MOC_FinalOutput()
 *   introduction: output the timing info
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2019/09/13, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
void MOC_FinalOutput(FP_PRECISION_TIME _time_tot, FP_PRECISION_TIME _time_omp)
{
	int _width2 = 14, _width_time = 32;
	if (ProcID == 0) {

#if defined(DEFAULT_TIME)
#if defined(M_MG_CMFD)
		TimeRec_CMFDSolution = TimeRec_MGCMFD_XSCollapse[0] + TimeRec_MGCMFD_MatrixConst[0] + \
			TimeRec_MGCMFD_CMFDSolver + TimeRec_MGCMFD_FluxUpdate;
#endif // defined(M_MG_CMFD)
#ifdef TIME_TYPE2
		TimeRec_TSPraperation /= (FP_PRECISION)(CLOCKS_PER_SEC);
		TimeRec_MOCSolution /= (FP_PRECISION)(CLOCKS_PER_SEC);
        TimeRec_SourceCalc /= (FP_PRECISION)(CLOCKS_PER_SEC);
        TimeRec_RT /= (FP_PRECISION)(CLOCKS_PER_SEC);
#if defined(M_MG_CMFD)
		TimeRec_CMFDSolution /= (FP_PRECISION)(CLOCKS_PER_SEC);

		TimeRec_MGCMFD_XSCollapse[0] /= (FP_PRECISION)(CLOCKS_PER_SEC);
		TimeRec_MGCMFD_MatrixConst[0] /= (FP_PRECISION)(CLOCKS_PER_SEC);
		TimeRec_MGCMFD_CMFDSolver /= (FP_PRECISION)(CLOCKS_PER_SEC);
		TimeRec_MGCMFD_LinearSolver[0] /= (FP_PRECISION)(CLOCKS_PER_SEC);
		TimeRec_MGCMFD_FluxUpdate /= (FP_PRECISION)(CLOCKS_PER_SEC);
    #if defined(M_TIME_GMRES)
		Time_MGCMFD_GMRES_Precond[0] /= (FP_PRECISION)(CLOCKS_PER_SEC);
		Time_MGCMFD_GMRES_LeastSqr[0] /= (FP_PRECISION)(CLOCKS_PER_SEC);
		Time_MGCMFD_GMRES_MemCpy[0] /= (FP_PRECISION)(CLOCKS_PER_SEC);
		Time_MGCMFD_GMRES_SpMV[0] /= (FP_PRECISION)(CLOCKS_PER_SEC);
		Time_MGCMFD_GMRES_BLAS1[0] /= (FP_PRECISION)(CLOCKS_PER_SEC);
	#endif // defined(M_TIME_GMRES)


#if defined(M_CG_CMFD)
		TimeRec_CGCMFD_XSCollapse[0] /= (FP_PRECISION)(CLOCKS_PER_SEC);
		TimeRec_CGCMFD_MatrixConst[0] /= (FP_PRECISION)(CLOCKS_PER_SEC);
		TimeRec_CGCMFD_CMFDSolver /= (FP_PRECISION)(CLOCKS_PER_SEC);
		TimeRec_CGCMFD_LinearSolver[0] /= (FP_PRECISION)(CLOCKS_PER_SEC);
		TimeRec_CGCMFD_FluxUpdate /= (FP_PRECISION)(CLOCKS_PER_SEC);
	#if defined(M_TIME_GMRES)
		Time_CGCMFD_GMRES_Precond[0] /= (FP_PRECISION)(CLOCKS_PER_SEC);
		Time_CGCMFD_GMRES_LeastSqr[0] /= (FP_PRECISION)(CLOCKS_PER_SEC);
		Time_CGCMFD_GMRES_MemCpy[0] /= (FP_PRECISION)(CLOCKS_PER_SEC);
		Time_CGCMFD_GMRES_SpMV[0] /= (FP_PRECISION)(CLOCKS_PER_SEC);
		Time_CGCMFD_GMRES_BLAS1[0] /= (FP_PRECISION)(CLOCKS_PER_SEC);
	#endif // defined(M_TIME_GMRES)
#endif // defined(M_CG_CMFD)
#endif
#endif // TIME_TYPE2
#endif // defined(DEFAULT_TIME)

#ifdef CUDA_USED
		printf("\n [  RESULT ]  Total Transport Calculation Time: %.6lf sec", TimeRec_TransportCalculation);
		fprintf(LogFile, "\n [  RESULT ]  Total Transport Calculation Time: %.6lf sec", TimeRec_TransportCalculation);

		if (NumScattOrder == 0 && GPU_TSKernelType < 10) {
			/* P0 uses one fused fission-plus-scattering source kernel. */
			printf("\n [  PHASE  ]  P0 Source Calculation Time: %.6lf sec", TimeRec_P0Source);
			fprintf(LogFile, "\n [  PHASE  ]  P0 Source Calculation Time: %.6lf sec", TimeRec_P0Source);
		}
		else {
			printf("\n [  PHASE  ]  Fission Source Time: %.6lf sec", TimeRec_FissionSource);
			fprintf(LogFile, "\n [  PHASE  ]  Fission Source Time: %.6lf sec", TimeRec_FissionSource);
			printf("\n [  PHASE  ]  Scattering Source Time: %.6lf sec", TimeRec_HigherOrderSource);
			fprintf(LogFile, "\n [  PHASE  ]  Scattering Source Time: %.6lf sec", TimeRec_HigherOrderSource);
			printf("\n [  PHASE  ]    LM Kernel Time: %.6lf sec", TimeRec_HigherOrderSourceLM);
			fprintf(LogFile, "\n [  PHASE  ]    LM Kernel Time: %.6lf sec", TimeRec_HigherOrderSourceLM);
			printf("\n [  PHASE  ]    AP Kernel Time: %.6lf sec", TimeRec_HigherOrderSourceAP);
			fprintf(LogFile, "\n [  PHASE  ]    AP Kernel Time: %.6lf sec", TimeRec_HigherOrderSourceAP);
			printf("\n [  PHASE  ]    LM/AP Auxiliary Time: %.6lf sec", TimeRec_HigherOrderSourceAuxiliary);
			fprintf(LogFile, "\n [  PHASE  ]    LM/AP Auxiliary Time: %.6lf sec", TimeRec_HigherOrderSourceAuxiliary);
			printf("\n [  PHASE  ]  Flux Moment Merge Time: %.6lf sec", TimeRec_FluxAccumulation);
			fprintf(LogFile, "\n [  PHASE  ]  Flux Moment Merge Time: %.6lf sec", TimeRec_FluxAccumulation);
			printf("\n [  PHASE  ]    Scalar Flux Buffer Init Time: %.6lf sec", TimeRec_FluxMomentScalarFluxInit);
			fprintf(LogFile, "\n [  PHASE  ]    Scalar Flux Buffer Init Time: %.6lf sec", TimeRec_FluxMomentScalarFluxInit);
			printf("\n [  PHASE  ]    Angular Flux Device-to-Host Copy Time: %.6lf sec", TimeRec_FluxMomentAngularFluxCopy);
			fprintf(LogFile, "\n [  PHASE  ]    Angular Flux Device-to-Host Copy Time: %.6lf sec", TimeRec_FluxMomentAngularFluxCopy);
			printf("\n [  PHASE  ]    Anisotropic Scattering Source Copy Time: %.6lf sec", TimeRec_FluxMomentAniScattSourceCopy);
			fprintf(LogFile, "\n [  PHASE  ]    Anisotropic Scattering Source Copy Time: %.6lf sec", TimeRec_FluxMomentAniScattSourceCopy);
			printf("\n [  PHASE  ]    Angular Flux Reset Time: %.6lf sec", TimeRec_FluxMomentAngularFluxReset);
			fprintf(LogFile, "\n [  PHASE  ]    Angular Flux Reset Time: %.6lf sec", TimeRec_FluxMomentAngularFluxReset);
			printf("\n [  PHASE  ]    Azimuth Accumulation Init Time: %.6lf sec", TimeRec_FluxMomentAzimAccumInit);
			fprintf(LogFile, "\n [  PHASE  ]    Azimuth Accumulation Init Time: %.6lf sec", TimeRec_FluxMomentAzimAccumInit);
			printf("\n [  PHASE  ]    Azimuth Accumulation Finalize Time: %.6lf sec", TimeRec_FluxMomentAzimAccumFinalize);
			fprintf(LogFile, "\n [  PHASE  ]    Azimuth Accumulation Finalize Time: %.6lf sec", TimeRec_FluxMomentAzimAccumFinalize);
		}

		if (TimeRec_FluxMomentKernel103Calls > 0) {
			printf("\n [ KERNEL  ]  GPUAccumAngFlux_ScalarFlux_HOScatt_opt: %.6lf sec, calls=%llu, avg=%.6lf sec",
				TimeRec_FluxMomentKernel103,
				(unsigned long long) TimeRec_FluxMomentKernel103Calls,
				TimeRec_FluxMomentKernel103 / (FP_PRECISION_TIME) TimeRec_FluxMomentKernel103Calls);
			fprintf(LogFile, "\n [ KERNEL  ]  GPUAccumAngFlux_ScalarFlux_HOScatt_opt: %.6lf sec, calls=%llu, avg=%.6lf sec",
				TimeRec_FluxMomentKernel103,
				(unsigned long long) TimeRec_FluxMomentKernel103Calls,
				TimeRec_FluxMomentKernel103 / (FP_PRECISION_TIME) TimeRec_FluxMomentKernel103Calls);
		}
		if (TimeRec_FluxMomentKernelPerAzimCalls > 0) {
			printf("\n [ KERNEL  ]  GPUAccumAngFlux_ScalarFlux_HOScatt_PerAzim: %.6lf sec, calls=%llu, avg=%.6lf sec",
				TimeRec_FluxMomentKernelPerAzim,
				(unsigned long long) TimeRec_FluxMomentKernelPerAzimCalls,
				TimeRec_FluxMomentKernelPerAzim / (FP_PRECISION_TIME) TimeRec_FluxMomentKernelPerAzimCalls);
			fprintf(LogFile, "\n [ KERNEL  ]  GPUAccumAngFlux_ScalarFlux_HOScatt_PerAzim: %.6lf sec, calls=%llu, avg=%.6lf sec",
				TimeRec_FluxMomentKernelPerAzim,
				(unsigned long long) TimeRec_FluxMomentKernelPerAzimCalls,
				TimeRec_FluxMomentKernelPerAzim / (FP_PRECISION_TIME) TimeRec_FluxMomentKernelPerAzimCalls);
		}
		if (TimeRec_HOTransportKernel1Calls > 0) {
			printf("\n [ KERNEL  ]  GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt_1: %.6lf sec, calls=%llu, avg=%.6lf sec",
				TimeRec_HOTransportKernel1,
				(unsigned long long) TimeRec_HOTransportKernel1Calls,
				TimeRec_HOTransportKernel1 / (FP_PRECISION_TIME) TimeRec_HOTransportKernel1Calls);
			fprintf(LogFile, "\n [ KERNEL  ]  GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt_1: %.6lf sec, calls=%llu, avg=%.6lf sec",
				TimeRec_HOTransportKernel1,
				(unsigned long long) TimeRec_HOTransportKernel1Calls,
				TimeRec_HOTransportKernel1 / (FP_PRECISION_TIME) TimeRec_HOTransportKernel1Calls);
		}
		if (TimeRec_HOTransportKernel11Calls > 0) {
			printf("\n [ KERNEL  ]  GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt_11: %.6lf sec, calls=%llu, avg=%.6lf sec",
				TimeRec_HOTransportKernel11,
				(unsigned long long) TimeRec_HOTransportKernel11Calls,
				TimeRec_HOTransportKernel11 / (FP_PRECISION_TIME) TimeRec_HOTransportKernel11Calls);
			fprintf(LogFile, "\n [ KERNEL  ]  GPUTS_Group_ExpInte_OneDir_AngFSmem_HOScatt_11: %.6lf sec, calls=%llu, avg=%.6lf sec",
				TimeRec_HOTransportKernel11,
				(unsigned long long) TimeRec_HOTransportKernel11Calls,
				TimeRec_HOTransportKernel11 / (FP_PRECISION_TIME) TimeRec_HOTransportKernel11Calls);
		}

		printf("\n [  PHASE  ]  Pure Transport Sweep Time: %.6lf sec", TimeRec_TransportSweep);
		fprintf(LogFile, "\n [  PHASE  ]  Pure Transport Sweep Time: %.6lf sec", TimeRec_TransportSweep);
#endif // CUDA_USED

#if !defined(OMP_USED)
		printf("\n [  RESULT ]  Total Time to Solution: %.3lf sec", _time_tot);
		fprintf(LogFile, "\n [  RESULT ]  Total Time to Solution: %.3lf sec", _time_tot);
#else
		printf("\n [  RESULT ]  Total Time to Solution: %.3lf sec", _time_omp);
		printf("\n [  RESULT ]  OMP Time with %d CPU-cores: %.3lf sec", NumOMPThread, _time_tot);
		fprintf(LogFile, "\n [  RESULT ]  Total Time to Solution: %.3lf sec", _time_omp);
		fprintf(LogFile, "\n [  RESULT ]  OMP Time with %d CPU-cores: %.3lf sec", NumOMPThread, _time_tot);
#endif

#if defined(DEFAULT_TIME)
		printf("\n [  TITLE  ]  ************************************************************");
		printf("\n [  TITLE  ]  *                     TIMING INFO                          *");
		printf("\n [  TITLE  ]  ************************************************************");
		printf("\n [  RESULT ]  %-*s%*s%*s", _width_time, "Metrices", _width2, "Time sec", _width2, "Percent %");
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "Total Execution Time", \
			_width2, _time_tot, _width2, _time_tot / _time_tot * (FP_PRECISION_TIME)(100.));
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "Prapare XSs and Track Info", \
			_width2, TimeRec_TSPraperation, _width2, TimeRec_TSPraperation / _time_tot * (FP_PRECISION_TIME)(100.));
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "Total MOC Computation Time", \
			_width2, TimeRec_MOCSolution, _width2, TimeRec_MOCSolution / _time_tot * (FP_PRECISION_TIME)(100.));
#if defined(M_MG_CMFD)
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "Total CMFD Computation Time", \
			_width2, TimeRec_CMFDSolution, _width2, TimeRec_CMFDSolution / _time_tot * (FP_PRECISION_TIME)(100.));

		printf("\n [  RESULT ]  %-*s", _width_time, "  XSs Collapse Time");
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "    Multi-group CMFD", \
			_width2, TimeRec_MGCMFD_XSCollapse[0], _width2, TimeRec_MGCMFD_XSCollapse[0] / _time_tot * (FP_PRECISION_TIME)(100.));
#if defined(M_CG_CMFD)
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "    Coarse-group CMFD", \
			_width2, TimeRec_CGCMFD_XSCollapse[0], _width2, TimeRec_CGCMFD_XSCollapse[0] / _time_tot * (FP_PRECISION_TIME)(100.));
#endif // defined(M_CG_CMFD)

		printf("\n [  RESULT ]  %-*s", _width_time, "  Matrix Construction Time");
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "    Multi-group CMFD", \
			_width2, TimeRec_MGCMFD_MatrixConst[0], _width2, TimeRec_MGCMFD_MatrixConst[0] / _time_tot * (FP_PRECISION_TIME)(100.));
#if defined(M_CG_CMFD)
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "    Coarse-group CMFD", \
			_width2, TimeRec_CGCMFD_MatrixConst[0], _width2, TimeRec_CGCMFD_MatrixConst[0] / _time_tot * (FP_PRECISION_TIME)(100.));
#endif // defined(M_CG_CMFD)

		printf("\n [  RESULT ]  %-*s", _width_time, "  Linear Solver Computation Time");
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "    Multi-group CMFD", \
			_width2, TimeRec_MGCMFD_LinearSolver[0], _width2, TimeRec_MGCMFD_LinearSolver[0] / _time_tot * (FP_PRECISION_TIME)(100.));
		
    #if defined(M_TIME_GMRES)
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      SpMV", \
		    _width2, Time_MGCMFD_GMRES_SpMV[0], _width2, Time_MGCMFD_GMRES_SpMV[0] / _time_tot * (FP_PRECISION_TIME)(100.));
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      BLAS_1", \
		    _width2, Time_MGCMFD_GMRES_BLAS1[0], _width2, Time_MGCMFD_GMRES_BLAS1[0] / _time_tot * (FP_PRECISION_TIME)(100.));
	    if (GmresPrecond_MG) {
		  #if defined(CUDA_USED) && defined(M_GMRES_IluPRE) && !defined(M_GMRES_PRE_cuSparseLib)
			printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      Memcpy_Preco", \
		    	_width2, Time_MGCMFD_GMRES_MemCpy[0], _width2, Time_MGCMFD_GMRES_MemCpy[0] / _time_tot * (FP_PRECISION_TIME)(100.));
		  #endif // defined(CUDA_USED) && defined(M_GMRES_IluPRE) && !defined(M_GMRES_PRE_cuSparseLib)
            printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      Precondition", \
		    	_width2, Time_MGCMFD_GMRES_Precond[0], _width2, Time_MGCMFD_GMRES_Precond[0] / _time_tot * (FP_PRECISION_TIME)(100.));
		}
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      Least_square", \
		    _width2, Time_MGCMFD_GMRES_LeastSqr[0], _width2, Time_MGCMFD_GMRES_LeastSqr[0] / _time_tot * (FP_PRECISION_TIME)(100.));
	#endif // defined(M_TIME_GMRES)
    	
#if defined(M_CG_CMFD)
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "    Coarse-group CMFD", \
			_width2, TimeRec_CGCMFD_LinearSolver[0], _width2, TimeRec_CGCMFD_LinearSolver[0] / _time_tot * (FP_PRECISION_TIME)(100.));
    #if defined(M_TIME_GMRES)
	    printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      SpMV", \
		    _width2, Time_CGCMFD_GMRES_SpMV[0], _width2, Time_CGCMFD_GMRES_SpMV[0] / _time_tot * (FP_PRECISION_TIME)(100.));
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      BLAS_1", \
		    _width2, Time_CGCMFD_GMRES_BLAS1[0], _width2, Time_CGCMFD_GMRES_BLAS1[0] / _time_tot * (FP_PRECISION_TIME)(100.));
	    if (GmresPrecond_CG) {
		  #if defined(CUDA_USED) && defined(M_GMRES_IluPRE) && !defined(M_GMRES_PRE_cuSparseLib)
			printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      Memcpy_Preco", \
		    	_width2, Time_CGCMFD_GMRES_MemCpy[0], _width2, Time_CGCMFD_GMRES_MemCpy[0] / _time_tot * (FP_PRECISION_TIME)(100.));
		  #endif // defined(CUDA_USED) && defined(M_GMRES_IluPRE) && !defined(M_GMRES_PRE_cuSparseLib)
            printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      Precondition", \
		    	_width2, Time_CGCMFD_GMRES_Precond[0], _width2, Time_CGCMFD_GMRES_Precond[0] / _time_tot * (FP_PRECISION_TIME)(100.));
		}
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      Least_square", \
		    _width2, Time_CGCMFD_GMRES_LeastSqr[0], _width2, Time_CGCMFD_GMRES_LeastSqr[0] / _time_tot * (FP_PRECISION_TIME)(100.));
	#endif // defined(M_TIME_GMRES)
#endif // defined(M_CG_CMFD)

		printf("\n [  RESULT ]  %-*s", _width_time, "  Flux Update Time");
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "    Multi-group CMFD", \
			_width2, TimeRec_MGCMFD_FluxUpdate, _width2, TimeRec_MGCMFD_FluxUpdate / _time_tot * (FP_PRECISION_TIME)(100.));
#if defined(M_CG_CMFD)
		printf("\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "    Coarse-group CMFD", \
			_width2, TimeRec_CGCMFD_FluxUpdate, _width2, TimeRec_CGCMFD_FluxUpdate / _time_tot * (FP_PRECISION_TIME)(100.));
#endif // defined(M_CG_CMFD)

		//#if defined(M_MG_CMFD)
		printf("\n [  RESULT ]  %-*s%*s", _width_time, "Total Number of Iterations", _width2, "MG-CMFD");
#if defined(M_CG_CMFD)
		printf("%*s", _width2, "CG-CMFD");
#endif // defined(M_CG_CMFD)
		printf("\n [  RESULT ]  %-*s%*d", _width_time, "  CMFD Iterations", _width2, TotNumMGCMFDIter);
#if defined(M_CG_CMFD)
		printf("%*d", _width2, TotNumCGCMFDIter);
#endif // defined(M_CG_CMFD)
		printf("\n [  RESULT ]  %-*s%*d", _width_time, "  Linear Solver Iterations", _width2, TotNumMGLSIter);
#if defined(M_CG_CMFD)
		printf("%*d", _width2, TotNumCGLSIter);
#endif // defined(M_CG_CMFD)


#endif // defined(M_MG_CMFD)

		fprintf(LogFile, "\n [  TITLE  ]  ************************************************************");
		fprintf(LogFile, "\n [  TITLE  ]  *                     TIMING INFO                          *");
		fprintf(LogFile, "\n [  TITLE  ]  ************************************************************");
		fprintf(LogFile, "\n [  RESULT ]  %-*s%*s%*s", _width_time, "Metrices", _width2, "Time sec", _width2, "Percent %");
		fprintf(LogFile, "\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "Total Execution Time", _width2, _time_tot, _width2, _time_tot / _time_tot * (FP_PRECISION_TIME)(100.));
		fprintf(LogFile, "\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "Prapare XSs and Track Info", _width2, TimeRec_TSPraperation, _width2, TimeRec_TSPraperation / _time_tot * (FP_PRECISION_TIME)(100.));
		fprintf(LogFile, "\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "Total MOC Computation Time", _width2, TimeRec_MOCSolution, _width2, TimeRec_MOCSolution / _time_tot * (FP_PRECISION_TIME)(100.));
#if defined(M_MG_CMFD)
		fprintf(LogFile, "\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "Total CMFD Computation Time", _width2, TimeRec_CMFDSolution, _width2, TimeRec_CMFDSolution / _time_tot * (FP_PRECISION_TIME)(100.));
		fprintf(LogFile, "\n [  RESULT ]  %-*s", _width_time, "  XSs Collapse Time");
		fprintf(LogFile, "\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "    Multi-group CMFD", \
			_width2, TimeRec_MGCMFD_XSCollapse[0], _width2, TimeRec_MGCMFD_XSCollapse[0] / _time_tot * (FP_PRECISION_TIME)(100.));
#if defined(M_CG_CMFD)
		fprintf(LogFile, "\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "    Coarse-group CMFD", \
			_width2, TimeRec_CGCMFD_XSCollapse[0], _width2, TimeRec_CGCMFD_XSCollapse[0] / _time_tot * (FP_PRECISION_TIME)(100.));
#endif // defined(M_CG_CMFD)

		fprintf(LogFile, "\n [  RESULT ]  %-*s", _width_time, "  Matrix Construction Time");
		fprintf(LogFile, "\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "    Multi-group CMFD", \
			_width2, TimeRec_MGCMFD_MatrixConst[0], _width2, TimeRec_MGCMFD_MatrixConst[0] / _time_tot * (FP_PRECISION_TIME)(100.));
#if defined(M_CG_CMFD)
		fprintf(LogFile, "\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "    Coarse-group CMFD", \
			_width2, TimeRec_CGCMFD_MatrixConst[0], _width2, TimeRec_CGCMFD_MatrixConst[0] / _time_tot * (FP_PRECISION_TIME)(100.));
#endif // defined(M_CG_CMFD)

		fprintf(LogFile, "\n [  RESULT ]  %-*s", _width_time, "  Linear Solver Computation Time");
		fprintf(LogFile, "\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "    Multi-group CMFD", \
			_width2, TimeRec_MGCMFD_LinearSolver[0], _width2, TimeRec_MGCMFD_LinearSolver[0] / _time_tot * (FP_PRECISION_TIME)(100.));
    #if defined(M_TIME_GMRES)
	    fprintf(LogFile,"\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      SpMV", \
		    _width2, Time_MGCMFD_GMRES_SpMV[0], _width2, Time_MGCMFD_GMRES_SpMV[0] / _time_tot * (FP_PRECISION_TIME)(100.));
		fprintf(LogFile,"\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      BLAS_1", \
		    _width2, Time_MGCMFD_GMRES_BLAS1[0], _width2, Time_MGCMFD_GMRES_BLAS1[0] / _time_tot * (FP_PRECISION_TIME)(100.));
	    if (GmresPrecond_MG) {
		  #if defined(CUDA_USED) && defined(M_GMRES_IluPRE) && !defined(M_GMRES_PRE_cuSparseLib)
			fprintf(LogFile,"\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      Memcpy_Preco", \
		    	_width2, Time_MGCMFD_GMRES_MemCpy[0], _width2, Time_MGCMFD_GMRES_MemCpy[0] / _time_tot * (FP_PRECISION_TIME)(100.));
		  #endif // defined(CUDA_USED) && defined(M_GMRES_IluPRE) && !defined(M_GMRES_PRE_cuSparseLib)
            fprintf(LogFile,"\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      Precondition", \
		    	_width2, Time_MGCMFD_GMRES_Precond, _width2, Time_MGCMFD_GMRES_Precond / _time_tot * (FP_PRECISION_TIME)(100.));
		}
		fprintf(LogFile,"\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      Least_square", \
		    _width2, Time_MGCMFD_GMRES_LeastSqr[0], _width2, Time_MGCMFD_GMRES_LeastSqr[0] / _time_tot * (FP_PRECISION_TIME)(100.));
	#endif // defined(M_TIME_GMRES)
#if defined(M_CG_CMFD)
		fprintf(LogFile, "\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "    Coarse-group CMFD", \
			_width2, TimeRec_CGCMFD_LinearSolver[0], _width2, TimeRec_CGCMFD_LinearSolver[0] / _time_tot * (FP_PRECISION_TIME)(100.));
    #if defined(M_TIME_GMRES)
	    fprintf(LogFile,"\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      SpMV", \
		    _width2, Time_CGCMFD_GMRES_SpMV[0], _width2, Time_CGCMFD_GMRES_SpMV[0] / _time_tot * (FP_PRECISION_TIME)(100.));
		fprintf(LogFile,"\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      BLAS_1", \
		    _width2, Time_CGCMFD_GMRES_BLAS1[0], _width2, Time_CGCMFD_GMRES_BLAS1[0] / _time_tot * (FP_PRECISION_TIME)(100.));
	    if (GmresPrecond_CG) {
		  #if defined(CUDA_USED) && defined(M_GMRES_IluPRE) && !defined(M_GMRES_PRE_cuSparseLib)
			fprintf(LogFile, "\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      Memcpy_Preco", \
		    	_width2, Time_CGCMFD_GMRES_MemCpy[0], _width2, Time_CGCMFD_GMRES_MemCpy[0] / _time_tot * (FP_PRECISION_TIME)(100.));
		  #endif // defined(CUDA_USED) && defined(M_GMRES_IluPRE) && !defined(M_GMRES_PRE_cuSparseLib)
            fprintf(LogFile, "\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      Precondition", \
		    	_width2, Time_CGCMFD_GMRES_Precond[0], _width2, Time_CGCMFD_GMRES_Precond[0] / _time_tot * (FP_PRECISION_TIME)(100.));
		}
		fprintf(LogFile, "\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "      Least_square", \
		   	_width2, Time_CGCMFD_GMRES_LeastSqr[0], _width2, Time_CGCMFD_GMRES_LeastSqr[0] / _time_tot * (FP_PRECISION_TIME)(100.));
	#endif // defined(M_TIME_GMRES)

#endif // defined(M_CG_CMFD)

		fprintf(LogFile, "\n [  RESULT ]  %-*s", _width_time, "  Flux Update Time");
		fprintf(LogFile, "\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "    Multi-group CMFD", \
			_width2, TimeRec_MGCMFD_FluxUpdate, _width2, TimeRec_MGCMFD_FluxUpdate / _time_tot * (FP_PRECISION_TIME)(100.));
#if defined(M_CG_CMFD)
		fprintf(LogFile, "\n [  RESULT ]  %-*s%*.3e%*.2lf", _width_time, "    Coarse-group CMFD", \
			_width2, TimeRec_CGCMFD_FluxUpdate, _width2, TimeRec_CGCMFD_FluxUpdate / _time_tot * (FP_PRECISION_TIME)(100.));
#endif // defined(M_CG_CMFD)

		fprintf(LogFile, "\n [  RESULT ]  %-*s%*s", _width_time, "Total Number of Iterations", _width2, "MG-CMFD");
#if defined(M_CG_CMFD)
		fprintf(LogFile, "%*s", _width2, "CG-CMFD");
#endif // defined(M_CG_CMFD)
		fprintf(LogFile, "\n [  RESULT ]  %-*s%*d", _width_time, "  CMFD Iterations", _width2, TotNumMGCMFDIter);
#if defined(M_CG_CMFD)
		fprintf(LogFile, "%*d", _width2, TotNumCGCMFDIter);
#endif // defined(M_CG_CMFD)
		fprintf(LogFile, "\n [  RESULT ]  %-*s%*d", _width_time, "  Linear Solver Iterations", _width2, TotNumMGLSIter);
#if defined(M_CG_CMFD)
		fprintf(LogFile, "%*d", _width2, TotNumCGLSIter);
#endif // defined(M_CG_CMFD)
#endif
#endif // defined(DEFAULT_TIME)

#ifdef REC_TIME_SOLVER
		printf("\n    RecTime_RayWisedSolver_PreSource = %.3lf sec", RecTime_RayWisedSolver_PreSource);
		printf("\n    RecTime_RayWisedSolver_PreXSt    = %.3lf sec", RecTime_RayWisedSolver_PreXSt);
		printf("\n    RecTime_RayWisedSolver_PreFSRVol = %.3lf sec", RecTime_RayWisedSolver_PreFSRVol);
		printf("\n    RecTime_RayWisedSolver_ZeroFlux  = %.3lf sec", RecTime_RayWisedSolver_ZeroFlux);
		printf("\n    RecTime_RayWisedSolver           = %.3lf sec", RecTime_RayWisedSolver);
		printf("\n    RecTime_RayWisedSolver_AccumFlux = %.3lf sec", RecTime_RayWisedSolver_AccumFlux);
		fprintf(LogFile, "\n [  RESULT ]  RecTime_RayWisedSolver_PreSource = %.3lf sec", RecTime_RayWisedSolver_PreSource);
		fprintf(LogFile, "\n [  RESULT ]  RecTime_RayWisedSolver_PreXSt    = %.3lf sec", RecTime_RayWisedSolver_PreXSt);
		fprintf(LogFile, "\n [  RESULT ]  RecTime_RayWisedSolver_PreFSRVol = %.3lf sec", RecTime_RayWisedSolver_PreFSRVol);
		fprintf(LogFile, "\n [  RESULT ]  RecTime_RayWisedSolver_ZeroFlux  = %.3lf sec", RecTime_RayWisedSolver_ZeroFlux);
		fprintf(LogFile, "\n [  RESULT ]  RecTime_RayWisedSolver           = %.3lf sec", RecTime_RayWisedSolver);
		fprintf(LogFile, "\n [  RESULT ]  RecTime_RayWisedSolver_AccumFlux = %.3lf sec", RecTime_RayWisedSolver_AccumFlux);
#endif //REC_TIME_SOLVER
	}
}

void CheckExpInteRange(int _procID, int _isCheckExpInteRange, int _layBoundLowLim, \
    int _layBoundUpLim,  int _groupBoundLowLim, int _groupBoundUpLim)
{
	if (_isCheckExpInteRange==1) {
	int _polarBoundLowLim, _polarBoundUpLim, _threadID,_numOMPThreads_comp,_check,_trackSet;
	_check=0;
	_threadID=0;
	_numOMPThreads_comp=1;
	_trackSet=0;

#if defined(CUDA_USED)
    if (ProblemType==0) {
		if (GPU_TSKernelType==0 || GPU_TSKernelType==2) {
			_check=1;
		}
	}
	else if (ProblemType==1) {
		if (GPU_FixSourSolver==0 || GPU_FixSourSolver==2) {
			_check=1;
		}
	}
	else {}
#else 
    if (ProblemType==0) {
		if (CPU_TSFunctionType==0) {
			_check=1;
		}
	}
	else if (ProblemType==1) {
		if (CPU_FixSourSolver==0) {
			_check=1;
		}
	}
	else {}
#endif // defined(CUDA_USED)

	if (_check==1) {
	_polarBoundLowLim=0;
	_polarBoundUpLim=NumPolarAng[0];
#if defined(M_3DMOCex)
    _polarBoundUpLim /= 2;
#endif // defined(M_3DMOCex)

	ErrorCheck_Jacobi_MOCSolver_2D_expTab(_procID,ProblemType, _threadID,NumGroups, _layBoundLowLim, \
    _layBoundUpLim,  _groupBoundLowLim, _groupBoundUpLim, _polarBoundLowLim, _polarBoundUpLim, \
	NumPinsInSubDomPerLayer, CellRTMType_SubDom, TrackIndexInLTs_2PI[_trackSet], TracksInfoInLTs_2PI[_trackSet], \
	SumNumFSRsBasedGlobCellID, NumSegsInTrack_2PI[_trackSet], MacroXS_t, TotNumTracksInOneRTM_2PI[_trackSet], \
	IndexTrackInRTM_SumAzim_2PI[_trackSet], _numOMPThreads_comp, SumNumSegsInRTMTrack_2PI[_trackSet], \
	SegInfoInTrack_2PI_length[_trackSet], SegInfoInTrack_2PI_AntiClockFSR[_trackSet], NumLTs_HS[_trackSet], \
	SinThetaForExp, WidthLayersInCore,  CosTheta[_trackSet], LTsIndex_AzimAngID_2PI[_trackSet]);
	}
	}
}

void BoundAngFluxExchange_MPIComm(int _procID,int _iter, int _layBoundLowLim, int _layBoundUpLim)
{
#ifdef MPI_USED
	int _lay_1,trSet_1;
#endif
	//int lt;
	//int i;
#ifdef TIME_REC
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC

#ifdef MPI_CLOCK
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
	_time_start = TIME_RECORD;
#endif // MPI_CLOCK
// if (ProcID==0) {
// Printf_Vector_FP(MOCNumLayersZ*NumLTs_HS[0] * 2 * NumPolarAng[0] * NumGroups, BoundaryFlux[0], "B4 Comm BoundaryFlux");	
// }


#ifdef MPI_USED
	if (NumProcs > 1/* && (NumSubDomains[0] > 1 || NumSubDomains[1] > 1)*/) {
		for (_lay_1 = _layBoundLowLim; _lay_1 < _layBoundUpLim; _lay_1++) {
			for (trSet_1 = 0; trSet_1 < NumTrackSet; trSet_1++) {
  #ifndef GPU_STREAM_USED
				UpdateBoundAngFlux_ExchAngF_Blocking(_lay_1, NumProcPerLayer, _procID, Mess_TarProc[_lay_1], Mess_SendNumLTs[_lay_1], \
					CPUMPIFlux[trSet_1], BoundaryFlux[trSet_1], NumAzimAng[trSet_1], LongTracksInfo_Total[trSet_1], \
					NumLongTracks[trSet_1], NumPolarAng[trSet_1], NumGroups, NumLTs_HS[trSet_1], NumLTsSum[trSet_1], \
					Mess_SendSatrtPInAngF[_lay_1], Mess_RecvSatrtPInAngF[_lay_1], MPIProcID_recv[_lay_1], MPINeibID_recv[_lay_1]);
  #endif // !GPU_STREAM_USED
  #ifdef GPU_STREAM_USED
				if (HardwareMap_SubDom[_procID] == 1) {
					UpdateBoundAngFlux_ExchAngF_Blocking(_lay_1, NumProcPerLayer, _procID, Mess_TarProc[_lay_1], Mess_SendNumLTs[_lay_1], \
						CGLTInciAngFlux, CGBoundaryFlux, NumAzimAng[trSet_1], LongTracksInfo_Total[trSet_1], \
						NumLongTracks[trSet_1], NumPolarAng[trSet_1], NumGroups, NumLTs_HS[trSet_1], NumLTsSum[trSet_1], \
						Mess_SendSatrtPInAngF[_lay_1], Mess_RecvSatrtPInAngF[_lay_1], MPIProcID_recv[_lay_1], MPINeibID_recv[_lay_1]);
				}
				else {
					UpdateBoundAngFlux_ExchAngF_Blocking(_lay_1, NumProcPerLayer, _procID, Mess_TarProc[_lay_1], Mess_SendNumLTs[_lay_1], \
						CPUMPIFlux[trSet_1], BoundaryFlux[trSet_1], NumAzimAng[trSet_1], LongTracksInfo_Total[trSet_1], \
						NumLongTracks[trSet_1], NumPolarAng[trSet_1], NumGroups, NumLTs_HS[trSet_1], NumLTsSum[trSet_1], \
						Mess_SendSatrtPInAngF[_lay_1], Mess_RecvSatrtPInAngF[_lay_1], MPIProcID_recv[_lay_1], MPINeibID_recv[_lay_1]);
				}
  #endif // GPU_STREAM_USED
			}
		}
//#ifndef CPUMPI_GPUTS_ASYNCH
//		MPI_Barrier(MPI_COMM_WORLD);
//#endif // !CPUMPI_GPUTS_ASYNCH
	}

// if (ProcID==0) {
// Printf_Vector_FP(MOCNumLayersZ*NumLTs_HS[0] * 2 * NumPolarAng[0] * NumGroups, BoundaryFlux[0], "A4 Comm BoundaryFlux");	
// }
#ifdef TIME_REC
	    _time_end = TallyCurrentTime_CPU();
		if (_iter==0) {
			TimeStamp_MPIComm=TimeCount;
			strcpy(TimeMetrics[TimeStamp_MPIComm], "MPI Comm (Radial F)");
			TimeCount++;
		}
        TimeOverall[TimeStamp_MPIComm] += _time_end - _time_start;
#endif // TIME_REC
#endif // MPI_UESD

#ifdef MPI_CLOCK
	_time_end = TIME_RECORD;
	MPIComm_Time+= _time_end - _time_start;
#endif // MPI_CLOCK
}
