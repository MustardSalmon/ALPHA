#ifndef MOC_OUTPUT_H
#define MOC_OUTPUT_H
#include <time.h>
#include "MOC_Core.h"
#include "MOC_CommonMacro.h"
#include "MOC_Pin_Cell.h"
#include "MOC_Track.h"
#include "paper_transport_compat.h"
//#define OUTPUT_inputInfo 0 //macro, GPU_TSKernelType if print the input information into file "Input_Check.txt"
//#define OUTPUT_XSslibInfo 0
#define IF_WRITE_MACXS 0
#define IF_WRITE_MICXS 0
#define IF_WRITE_NUM_DEN 0

#define MEM_1 5
#define MEM_2 35
#define MEM_3 40
#define MEM_4 13
#define MEM_5 16
#define MEM_6 13
#define MEM_7 15
#define MEM_8 15
#define MEM_9 27
#define MEM_0 13
#define MEM_10 15
#define NumPara 4096
//#define ParaNameLength 128
//#define NumTimeMetrics 32

//#define OUTPUT_GeoInterfaceINFO 0

//#define OUTPUT_MacorXSs 1

//#define OUTPUT_eigenvalue 1
//#define OUTPUT_normPinPower 0
//#define OUTPUT_flux 0
extern int OUTPUT_MemoryFootprint;
extern int OUTPUT_inputInfo;
extern int OUTPUT_XSslibInfo;

extern int Output_GeometryInfo;
//extern int OUTPUT_GeoInterfaceINFO;
extern int OUTPUT_MacorXSs;
extern int OUTPUT_MacorXSs_InputCheck;
extern int OUTPUT_eigenvalue;
extern int OUTPUT_normPinPower;
extern int OUTPUT_flux,OUTPUT_flux_mat;
extern int OUTPUT_CMFDGroupConstant;
extern int OUTPUT_CMFDGC_byStep;
extern int OUTPUT_CoarseMesh_GroupConstant;
extern int OUTPUT_MatrixEigenvalueProb;
extern int OUTPUT_CompTime;
extern int OUTPUT_CoarseCompTime;
extern int OUTPUT_Debug;
extern int OUTPUT_CompTime_Proc;
extern FILE *FP_MemoryFootprint;
extern FILE *CompTime_rec;
extern FILE *GeneralOutput, *Debug;
extern int ParamCounter;
extern FP_PRECISION NumTotalData;
extern char ParaName[NumPara][ParaNameLength];
extern FP_PRECISION ParaMemSize[NumPara];

extern FP_PRECISION_TIME CompTime_Tot; // total computational time
extern FP_PRECISION_TIME CompTime_CMFD_Tot; // total computational time for CMFD
extern FP_PRECISION_TIME CompTime_MOC_Tot;
extern FP_PRECISION_TIME CompTime_CMFDtoMOC; // computational time for updating moc flux using cmfd results
extern FP_PRECISION_TIME CompTime_MOC_LBF;
extern FP_PRECISION_TIME CompTime_MOC_SE;
extern FP_PRECISION_TIME CompTime_MOC_TS;
extern FP_PRECISION_TIME CompTime_MOC_In;
extern FP_PRECISION_TIME CompTime_MOC_STr;
extern FP_PRECISION_TIME CompTime_MOC_GPCIB;
extern FP_PRECISION_TIME CompTime_MOC_GPCOB;
extern FP_PRECISION_TIME CompTime_MOC_EXP;
extern FP_PRECISION_TIME CompTime_MOC_SI;
extern FP_PRECISION_TIME CompTime_MOC_BCTS;
extern FP_PRECISION_TIME CompTime_MOC_ESF;
extern FP_PRECISION_TIME CompTime_MOC_UBAF;
extern FP_PRECISION_TIME CompTime_MOC_UBAF_Pre;
extern FP_PRECISION_TIME CompTime_MOC_UBAF_Comm;
extern FP_PRECISION_TIME CompTime_MOC_EVGC;
extern FP_PRECISION_TIME CompTime_MOC_EXGC;
extern FP_PRECISION_TIME CompTime_ConvCheck;
extern FP_PRECISION_TIME CompTime_MOC_ParaUp;
extern FP_PRECISION_TIME CompTime_MOC_AddSour;
extern FP_PRECISION_TIME CompTime_CopyAngF_CpuGpu;
extern FP_PRECISION_TIME CompTime_Input, CompTime_InitGeoXSs, CompTime_ExpTab, CompTime_AllMem, CompTime_MapGtoC, CompTime_CopyXSsCtoG, CompTime_CopyTraInfoCtoG, CompTime_InitTSonGPU;

extern FP_PRECISION_TIME TimeRec_TSPraperation, TimeRec_MOCSolution, TimeRec_CMFDSolution,TimeRec_TSGPU, TimeRec_SourceCalc, TimeRec_RT;
extern FP_PRECISION_TIME TimeRec_ClusterPreparation;
extern FP_PRECISION_TIME TimeRec_FissionSource;
extern FP_PRECISION_TIME TimeRec_HigherOrderSource;
extern FP_PRECISION_TIME TimeRec_HigherOrderSourceLM;
extern FP_PRECISION_TIME TimeRec_HigherOrderSourceAP;
extern FP_PRECISION_TIME TimeRec_HigherOrderSourceAuxiliary;
extern FP_PRECISION_TIME TimeRec_TransportSweep;
extern FP_PRECISION_TIME TimeRec_FluxAccumulation;
extern FP_PRECISION_TIME TimeRec_P0Source;
extern FP_PRECISION_TIME TimeRec_TransportCalculation;
extern FP_PRECISION_TIME TimeRec_FluxMomentKernel103;
extern M_UINT64 TimeRec_FluxMomentKernel103Calls;
extern FP_PRECISION_TIME TimeRec_FluxMomentKernelPerAzim;
extern M_UINT64 TimeRec_FluxMomentKernelPerAzimCalls;
extern FP_PRECISION_TIME TimeRec_FluxMomentScalarFluxInit;
extern FP_PRECISION_TIME TimeRec_FluxMomentAngularFluxCopy;
extern FP_PRECISION_TIME TimeRec_FluxMomentAniScattSourceCopy;
extern FP_PRECISION_TIME TimeRec_FluxMomentAngularFluxReset;
extern FP_PRECISION_TIME TimeRec_FluxMomentAzimAccumInit;
extern FP_PRECISION_TIME TimeRec_FluxMomentAzimAccumFinalize;
extern FP_PRECISION_TIME TimeRec_HOTransportKernel1;
extern M_UINT64 TimeRec_HOTransportKernel1Calls;
extern FP_PRECISION_TIME TimeRec_HOTransportKernel11;
extern M_UINT64 TimeRec_HOTransportKernel11Calls;
extern FP_PRECISION_TIME TimeRec_ProgramWallStart;
extern FP_PRECISION_TIME TimeRec_ProgramWallTotal;
extern FP_PRECISION_TIME TimeRec_MGCMFD_XSCollapse[1], TimeRec_MGCMFD_MatrixConst[1], TimeRec_MGCMFD_CMFDSolver, TimeRec_MGCMFD_FluxUpdate;
extern FP_PRECISION_TIME TimeRec_CGCMFD_XSCollapse[1], TimeRec_CGCMFD_MatrixConst[1], TimeRec_CGCMFD_CMFDSolver, TimeRec_CGCMFD_FluxUpdate;
extern FP_PRECISION_TIME TimeRec_MGCMFD_LinearSolver[1], TimeRec_CGCMFD_LinearSolver[1];

extern FP_PRECISION_TIME Time_MGCMFD_GMRES_Precond[1], Time_MGCMFD_GMRES_LeastSqr[1], Time_MGCMFD_GMRES_MemCpy[1];
extern FP_PRECISION_TIME Time_MGCMFD_GMRES_SpMV[1], Time_MGCMFD_GMRES_BLAS1[1];
extern FP_PRECISION_TIME Time_CGCMFD_GMRES_Precond[1], Time_CGCMFD_GMRES_LeastSqr[1], Time_CGCMFD_GMRES_MemCpy[1];
extern FP_PRECISION_TIME Time_CGCMFD_GMRES_SpMV[1], Time_CGCMFD_GMRES_BLAS1[1];

#ifdef TIME_REC
  extern int TimeID_Input, TimeID_GeoXSs, TimeID_ExpTab, TimeID_AllocMem, TimeID_CommPre,TimeID_SetCUDADev, TimeID_InitInfoCtoG, TimeID_InitTS;
  extern int TimeID_InitTS, TimeID_TS_ParaUp, TimeID_TS_SE, TimeID_TS_CtoG, TimeID_TS, TimeID_TS_GtoC, TimeID_TS_MPIComm, TimeID_TS_ReacRate;
  extern int TimeID_TS_FluxRes, TimeID_TS_ConvCheck;
  extern FP_PRECISION_TIME *CompTime_omp;
#endif // TIME_REC

#ifdef DynamicWorkLoad
  extern FP_PRECISION_TIME TimeTS_DWL;
#endif // DynamicWorkLoad

extern int NumIters;

extern int MemParaCountBoundLow[20], MemParaCountBoundUp[20], MemPrintCount;
extern size_t GPUPreMemQuadrSet[1], GPUPreMemTrack[1],GPUPreMemExpTab[1], GPUPreMemScalF[1];
extern size_t GPUPreMemRadF[1], GPUPreMemAxiF[1], GPUPreMemXSs[1], GPUPreMemMGCMFD[1], GPUPreMemCGCMFD[1];
extern char MemParaName[20][128];

void OutputBasicTrackSetInfo(FILE *_op,int _numTrackSet, int *_numAzimAng, int *_numPolarAng,int **_numTracks,\
    int **_numTracksX,int **_numTracksY,FP_PRECISION_GEO **_dxEff,FP_PRECISION_GEO **_dyEff,FP_PRECISION_GEO **_effAzimAng,\
	FP_PRECISION_GEO **_effRaySpacing,FP_PRECISION **_azimWeights,FP_PRECISION **_totalWeights, \
	struct POINT ***_RTMStartP, struct POINT ***_RTMEndP);
void OutputFSRsCSGInfoInRTM(int _RTMod_id, FILE *_op, struct PIN_CELL *_cellBasedRTMGeo);
void OutputMOCGeometryInfo(int _procID,int _numTrackSet, int *_numAzimAng, int *_numPolarAng,int **_numTracks,\
    int **_numTracksX,int **_numTracksY,FP_PRECISION_GEO **_dxEff,FP_PRECISION_GEO **_dyEff,FP_PRECISION_GEO **_effAzimAng,\
	FP_PRECISION_GEO **_effRaySpacing,FP_PRECISION **_azimWeights,FP_PRECISION **_totalWeights,int _numRTMGeoType, \
	struct PIN_CELL *_cellBasedRTMGeo, struct RTMFSRINFO **_RTModFSRsInfo, int ****_index_MOC_TrackInRTM, int **_numSegsInTrack,\
	int **_segInfoInTrack_2PI_rayWisedFsrID, int **_fsrID_Map_orig_rayWised, struct SUBDOM_SEG_IN_TRACK_SINGLE **_segInfoInTrack_SINGLE_Total,\
	struct POINT ***_RTMStartP, struct POINT ***_RTMEndP, int _totalNumFSRsInRTM, int *_sumNumFSRsInRTM,\
	FP_PRECISION **_segInfoInTrack_2PI_length, int *_numLongtrack_halfSpace, int **_LTsIndex_AzimAngID, int **_LTsIndex_LTID, \
    struct SUBDOM_LONGTRACK_INFO **_longTracksInfo, int **_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK **_tracksInfoInLT, \
	int _numCellsInDom_Layer, int *_totNumTracksInOneRTM_2PI, int *_cellRTMGeoType, int **_indexTrackInRTM_SumAzim_2PI, \
	int **_segInfoInTrack_2PI_AntiClockFSR, int **_sumNumSegsInRTMTrack_2PI, \
	int *_sumNumFSRsBasedGlobCellID, FP_PRECISION_GEO *_FSRsAreaInRTM_AntiClockFSR);
void OutputSegInfoInRTM(FILE *_op,int _numRTMGeoType,int _numTrackSet, int *_numAzimAng, struct PIN_CELL *_cellBasedRTMGeo, \
    struct RTMFSRINFO **_RTModFSRsInfo);

void output_inputCheck(int _procID);
void output_keff(int _ifKEFF,int _outIterNum,FP_PRECISION *_k_eff, FP_PRECISION *_res,clock_t _time_begin,clock_t _time_end);
//void output_FSRsScalarFlux_fsr(FP_PRECISION ***_flux,FP_PRECISION ***_relative_flux);
//void output_FSRsScalarFlux_g(FP_PRECISION ***_flux, FP_PRECISION ***_relative_flux);
void output_normPinPower(FP_PRECISION *_norm_pin_power,FP_PRECISION *_NormPinPower_SumLay);
//void output_trackInfo_RTM_Azim(int _RTMod_id, int _azimu_ang_id, FILE *op);
void OutputMemoryFootprint(FILE * _fp_memoryFootprint, char *_data_type, char *_parameter, \
	M_UINT64 _dataNumber, FP_PRECISION _pointerSizeMB, FP_PRECISION _dataSizeMB);
void OutputFreeMemory(char *_parameter);
void output_FSRsScalarFlux_forRectangularMesh(FP_PRECISION ***_flux, FP_PRECISION ***_relative_flux, int _layerNumInCore);
void output_macroXSs_check();
void output_macroXSs_check_subDom(int _subDomID);

void output_LongTracksInfo_subDom(int _procID, FILE *_op, int *_numSubDoms, int ***_subDomMap, int **_numPinsSubDom);
void output_boundAngFlux_subDom(int _subDomID, int _azim_id, int _pol_id, int _group, FP_PRECISION ****_boundAngFlux, char *_mark);
void OutputMacroXSs_FSRs(int _procID,int _numLayersZ,int _numCellsPerLayer,int *_numFSRsInCell,int *_sumNumFSRsBasedGlobCellID,\
    int _numGroups,FP_PRECISION *_macroXS_t,FP_PRECISION *_macroXS_a,FP_PRECISION *_macroXS_f,FP_PRECISION *_macroXS_nf,\
	FP_PRECISION *_macroXS_s,FP_PRECISION *_macroXS_chi,FP_PRECISION **_inpXSs_t,FP_PRECISION **_inpXSs_a,\
	FP_PRECISION **_inpXSs_f,FP_PRECISION **_inpXSs_nf,FP_PRECISION ***_inpXSs_s,FP_PRECISION **_inpXSs_chi,\
	M_UINT64 *_MacroXS_s_NZRowStartID, M_UINT64 *_MacroXS_s_NZColStartID);
void output_FSRsScalarFlux_SubDom(int _procID, int _numCellsInDom, int **_numFSRsInCell, \
	FP_PRECISION ****_flux);
FILE *CompTime_record(int _procID);
FILE *open_debug_file(int _procID);
void output_CompTime(int _width_3, int _num_iter);
void collec_CompTime(int _procID, int _numProcs);
void output_normPinPower_Slice(FP_PRECISION ***_norm_pin_power, struct CORE *_core);
extern FP_PRECISION Time_Total_MOC, Time_Total_CMFD, Time_Total_MGCMFD, Time_Total_CGCMFD;
extern FP_PRECISION Time_CMFD_to_MOC, Time_MOC_to_CMFD, Time_Total_NEM;
extern FP_PRECISION *Time_Total_MOC_P, *Time_Total_CMFD_P, *Time_Total_MGCMFD_P, *Time_Total_CGCMFD_P;
extern FP_PRECISION *Time_CMFD_to_MOC_P, *Time_MOC_to_CMFD_P, *Time_Total_NEM_P;
extern FP_PRECISION Time_Total_MOC_A, Time_Total_CMFD_A, Time_Total_MGCMFD_A, Time_Total_CGCMFD_A;
extern FP_PRECISION Time_CMFD_to_MOC_A, Time_MOC_to_CMFD_A, Time_Total_NEM_A;
extern FP_PRECISION *Time_Total_MOC_M, *Time_Total_CMFD_M, *Time_Total_MGCMFD_M, *Time_Total_CGCMFD_M;
extern FP_PRECISION *Time_CMFD_to_MOC_M, *Time_MOC_to_CMFD_M, *Time_Total_NEM_M;
extern FP_PRECISION Time_MOC_Solver, Time_MOC_Exp, Time_MOC_Flux, Time_MOC_Prep, Time_MOC_Current, Time_MOC_Current_lt;
extern FP_PRECISION Time_MOC_Solver_A, Time_MOC_Exp_A, Time_MOC_Flux_A, Time_MOC_Prep_A, Time_MOC_Current_A, Time_MOC_Current_A_lt;
extern FP_PRECISION *Time_MOC_Solver_P, *Time_MOC_Exp_P, *Time_MOC_Flux_P, *Time_MOC_Prep_P, *Time_MOC_Current_P, *Time_MOC_Current_P_lt;
extern FP_PRECISION *Time_MOC_Solver_M, *Time_MOC_Exp_M, *Time_MOC_Flux_M, *Time_MOC_Prep_M, *Time_MOC_Current_M, *Time_MOC_Current_M_lt;
extern int Time_check_counter;
void OutputFSRsScalarFlux(int _procID, int _numCellsInDom, int *_numFSRsInCell, FP_PRECISION *_flux, FP_DOUBLE _moc_glo_fiss_tem);
void OutputFSRsScalarFlux_Mat(int _procID, int _numCellsInDom, int *_numFSRsInCell, FP_PRECISION *_flux, FP_DOUBLE _moc_glo_fiss_tem);
void OutputFSRsScalarFlux_forComp(int _procID, int _numCellsInDom, int *_numFSRsInCell, FP_PRECISION *_flux);
void output_FSRsScalarFlux_FixedSourceProblem(int _procID, FP_PRECISION ***_flux);
void collec_CompTime_1(int _procID, int _numProcs);
void SetTimeRecInfo();
void Collec_CompTime_HetePara(int _procID, int _numProcs);
void OutputRegionWisedRTMGeoInfo(FILE *_op, int _numTrackSet, int _numRTMGeoType, int *_numAzimAng, \
	int **_segInfoInTrack_2PI_rayWisedFsrID, int **_fsrID_Map_orig_rayWised, int **_numTracks, \
	int ****_index_MOC_TrackInRTM, int **_numSegsInTrack, struct SUBDOM_SEG_IN_TRACK_SINGLE **_segInfoInTrack_SINGLE_Total, \
	int _totalNumFSRsInRTM, int *_sumNumFSRsInRTM);
void Output_CMFD_GroupConstant(char *_case_id, int _procID, int _identifier, int _iter, FP_PRECISION _keff);

void OutputLongTrackInfo(int _numAzimAng, FILE *_op);
void OutputSegInfoInLongTrack(FILE *_op, int _numLongtrack_halfSpace, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, \
    struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, \
	int _numCellsInDom_Layer, int _totNumTracksInOneRTM_2PI, int *_cellRTMGeoType, int *_indexTrackInRTM_SumAzim_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, int *_sumNumSegsInRTMTrack_2PI, \
	int *_sumNumFSRsBasedGlobCellID, FP_PRECISION_GEO *_FSRsAreaInRTM_AntiClockFSR, int *_sumNumFSRsInRTM,int *_numSegsInTrack);
void OutputSegInfoInRTMTrack(FILE *_op, int _numTrackSet, int _numRTMGeoType, int *_numAzimAng, \
	int **_numTracks, int ****_index_MOC_TrackInRTM, int **_numSegsInTrack, \
	struct SUBDOM_SEG_IN_TRACK_SINGLE **_segInfoInTrack_SINGLE_Total, FP_PRECISION **_segInfoInTrack_2PI_length);
void Printf_Vector_FP(int _dim, FP_PRECISION *_vec, char *_paraName);

void OutputDetailedCompTime(int _procID, int _numProcs);
void PrintDeviceMemUsage(FILE *_fp,char *_parameter);
void Output_NormPinPower(int _procID, int _numProcs, int _timeID);

void Output_CoarseMesh_GroupConstant(char *_case_id, int _identifier);

// for high-order scattering
void OutputFSRsScalarFluxMoment(int _procID, int _numCellsInDom, int *_numFSRsInCell, FP_PRECISION *_flux, int _scatt_order);

// for high-order scattering with binary-format
void OutputFSRsScalarFluxMomentBinary(int _procID,
                                      FP_PRECISION *_flux,
                                      FP_PRECISION_GEO *_vol,
                                      int _scatt_order,
                                      int _grpNum,
                                      M_INT32 _fsrNum);

// for fsr XS output with binary-format
void OutputFSRsXSsBinary(int _procID,
                         M_INT32 _fsrNum,
                         int _grpNum,
                         int _scatt_order,
                         FP_PRECISION *_totXS,
                         FP_PRECISION *_nufXS,
                         FP_PRECISION *_chi,
                         M_UINT64 *_rowStart,
                         M_UINT64 *_colStart,
                         FP_PRECISION *_scatterXS);

// for fsr XS output with binary-format
void OutputFSRsFisXSsBinary(int _procID,
                            M_INT32 _fsrNum,
                            int _grpNum,
                            FP_PRECISION *_fXS);

#endif // OUTPUT
