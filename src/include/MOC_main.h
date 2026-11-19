#ifndef MOC_MAIN_H
#define MOC_MAIN_H
#include "MOC_CommonMacro.h"

extern FP_PRECISION *NormPinPowerSubDom, **CollecGlobPinPower, *NormPinPower_SumLay;
extern M_INT32 *CollNumPinsSubDom;
extern FP_PRECISION ***XS_leakage_SubDom, *XS_leakage_SubDom_;
extern int ProcID;
extern int Conv_Para;
extern int  NumProcs;
extern int normFluxes;
extern int MOC_bool;
extern FP_PRECISION **CPUMPIFlux, **BoundaryFlux, **BoundaryFlux_Temp;
extern FP_PRECISION *MOC_fissRate_procs, *MOC_subDomBoundOutgAngFlux, *MOC_totPower_procs;
extern FP_PRECISION *ExpTable_0/*, **MOC_trackFlux, **MOC_trackFlux_1, ***MOC_trackFlux_omp*/;
extern FP_DOUBLE *MOC_keff;
extern int ***Mess_TarProc, ***Mess_SendNumLTs, *MOC_numFSRs_subDom;
extern size_t ***Mess_SendSatrtPInAngF, ***Mess_RecvSatrtPInAngF;
// extern int **GCCopy_StartPInAngF, **GCCopy_NumDataInAngF;
extern int EigenValueIter, numUpScattGr;
extern FP_PRECISION MOCGlobAbsRateTemp, MOC_Residual,MOCFluxResidual, TotPower_core, globNDFErr_InfNorm;
extern FP_DOUBLE MOCKeff, MOCKeffOld;
extern FP_DOUBLE MOCGlobFissRateOld, MOCGlobFissRateTemp;
extern FP_PRECISION start_time, end_time, *MOCGSFissSour, *MOCGSScattSour, *MOC_gs_track_flux;
extern FP_PRECISION ****MOCtoCMFD_NDFWest_old, ****MOCtoCMFD_NDFNorth_old, ****MOCtoCMFD_NDFEast_old, ****MOCtoCMFD_NDFSouth_old;
extern FP_PRECISION **GlobPinPower, ***GlobPinPower_Slice;
extern FP_PRECISION *MOCtoCMFD_NDFWest_old_, *MOCtoCMFD_NDFNorth_old_, *MOCtoCMFD_NDFEast_old_, *MOCtoCMFD_NDFSouth_old_;
extern FP_PRECISION *Source/*, *Source_RayWised*/;
extern FP_PRECISION_FLUX *ScalarFlux_MOC, *ScalarFluxOld, *ScalarFluxOld_0thOrder,*ScalarFlux_MOCLayer;
extern FP_PRECISION_FLUX *FSRAveAngFlux_Top, *FSRAveAngFlux_Bott, *FSRAngScattSour;
extern FP_PRECISION_FLUX *FSRAveAngFlux_TopTemp, *FSRAveAngFlux_BottTemp;
extern FP_PRECISION_FLUX *FSRAveAngFlux_UpAllLay, *FSRAveAngFlux_DownAllLay;
extern FP_PRECISION *TwoCosThetaOverHight_Sigma, *TwoCosThetaOverHight_Source;
extern int MaxNumPolarAngs;
extern FP_PRECISION *****QuadAngFlux_MOC;
extern int *Mess_TarProc_1, *Mess_SendNumLTs_1;
extern FP_PRECISION **CFlux, **DFlux/*, **TestFlux*/;
extern FP_PRECISION_TIME MPIComm_Time;
extern char ProcessorName[ParaNameLength];
extern char **ProcessorNameColl;
extern int *CurrCPUIDColl;
extern int NameLen;
extern FP_PRECISION_FLUX *ScalarFlux_RayWisedFSR;
extern FP_PRECISION **SegAngFlux; // [NumGroups * NumPolar]
extern FP_PRECISION **ExpTempStore,*MOC3D_oneOverSigma,*MOC3D_QtoSigma; // [MaxNumSegInTrack * NumGroups * NumPolar]
extern int **LTsIndex_AzimAngID_HS, **LTsIndex_LTID_HS;
extern int **LTsIndex_AzimAngID_2PI, **LTsIndex_LTID_2PI;

extern FP_DOUBLE **MatSumFlux, *MatSumVol;

extern FP_PRECISION_FLUX **ScalarFlux_OMP, ***MOC_trackFlux_OMP;

extern FP_PRECISION **LTAngFluxInci_SubDom;
extern FP_DOUBLE MOCGlobFlux_FixSour;

extern int MaxNumPolarAng;
extern FP_PRECISION *MOC_fluxNormInf, *MOC_fluxNorm2, *MOC_fluxRMSE;
extern int *MOC_boolFluxNormInf, *MOC_boolFluxNorm2, *MOC_boolFluxRMSE;

extern FP_PRECISION_FLUX *AngularFluxOut;

void MOCInitializeGeometry(int _procID, int _numProcs);
void MOC_Main_AllocMem(int _procID);
void MOC_Main_InitExpTable(int _procID);
void MOC_Main_ParaUpdate_TS(int _porcID, int _eigenValueIter);
void BoundAngFluxExchange_MPIComm(int _procID, int _iter, int _layBoundLowLim, int _layBoundUpLim);
void MOC_Main_Output(int _procID);
void OptimalWorkLoadDistribution(int _procID, int _numProcs);
void MOC_TransSweep_thread(int _procID, int _threadID, int _numOMPThread_comp, int _iter);
void MOC_TransSweep_FixSourProb_lay_g(int _procID, int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim);
void PrintCurrentCPUID(int _procID, int _threadID);
void CPU_MOCTransportSweep(int _procID, int _iter, int _numOMPThread);
void MOC_TransSweep_ParaFor(int _procID, int _numOMPThread_comp);
void MOC_EigenValueProblem();
void CPU_MOC_EigenValueProblem();
void GPU_MOC_EigenValueProblem();
void CPUMOC_FixedSourceProblem_lay_g(int _procID, int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim);
void MOC_FixedSourceProblem_lay_g(int _procID, int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim);
// void MOC_FixedSourceProblem_woResonance(int _procID);
void CPU_MOCthenCMFD(); 
void GPU_MOCthenCMFD();
void MOC_FinalOutput(FP_PRECISION_TIME _time_tot, FP_PRECISION_TIME _time_omp);
void Printf_GMRESPrecondition(int _precond, char *_char);
void MOC_TransSweep_Layer(int _lay, int _groupBoundLowLim, int _groupBoundUpLim, int _polarBoundLowLim, \
    int _polarBoundUpLim, int _trackSet, int _threadID, int _numOMPThread_comp, FP_PRECISION _oneOverLayWidth,\
	FP_PRECISION *_axialSurfAnfFlux_inp, FP_PRECISION *_axialSurfAnfFlux_update, int polOffset);
void SetTwoCosThetaOverHight(int _MOCNumLayersZ, int _numPolAngs, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_WidthLayersInCore, \
	FP_PRECISION *_TwoCosThetaOverHight_Sigma, FP_PRECISION *_TwoCosThetaOverHight_Source);
void MOC_TransSweep_thread_UporDown(int _procID, int _threadID, int _numOMPThread_comp, int _iter, int _dim, \
	int _layStart, int _layAdd, int _surfLayInc, int _trackSet, int _groupBoundLowLim, int _groupBoundUpLim, \
	int _polarBoundLowLim, int _polarBoundUpLim, int _polOffset, FP_PRECISION *_axiFlux_inp, FP_PRECISION *_axiFlux_avg, \
	int _nodeLay, int _surfLay, FP_PRECISION _JzDir, int _dir, int _dim1, int _fisrtInvoke);
void CPU_MOCFixSourProbSolver_lay_g(int _procID, int _iter, int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim);
void MPIInfo_Preparation();
void CollecAndPrint_MPIProcInfo(int _procID, int _numProcs);
void GPUMOC_FixedSourceProblem_lay_g(int _procID, int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim);
void CheckExpInteRange(int _procID, int _isCheckExpInteRange, int _layBoundLowLim, \
    int _layBoundUpLim,  int _groupBoundLowLim, int _groupBoundUpLim);
void PrintCPUMemUsage(int _procID);
void FreeMemory();
void CPU_InitAxiAngFlux(M_INT32 _ProblemType, M_INT32 _dim, M_INT32 _BottomBC, M_INT32 _TopBC, FP_PRECISION *_FSRAveAngFlux_Bott, \
	FP_PRECISION *_FSRAveAngFlux_Top, FP_PRECISION _val1, FP_PRECISION _zero_fp);
#endif