#ifndef MOC_INPUT_H
#define MOC_INPUT_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MOC_CommonMacro.h"
#include "MOC_Track.h"
// #ifdef MPI_USED
//   #include <mpi.h>
// #endif
//#define IP "Input-C5G7_MOX_7g.inp"
#define CASEID_N 128    //the size of a array while is used to storage the the string of case id
//#define PM_ARR_N 7     //the size of the arrays inside the PINMESH structure
//#define A_N 10         //the size of a string array for identifying the token '/'
//#define PIN_MAT_ID 16   //the size of the last member array of structure PIN, give the max material number for a single pin
#define NAME_N 512
#define OVER_N 5120
#define KEYWORD_N 64
//#define OutputDirLength 216
//#define FILE_NAME  512

    extern FILE *LogFile;
	extern char OutputDir[CASEID_N];
    extern char CaseId[CASEID_N];
	extern M_INT32 KType;
	extern M_INT32 IfResonance, ProblemType, isAdjoint;
    extern M_INT32 NumPlanes;
	extern M_INT32 NumCylinders;
    extern struct PARA_CYLINDER *Cylinders;
	extern M_INT32 NumPrimitives;
	extern struct CELL *Primitives;
    extern M_INT32 NumCells;
    extern struct PIN_CELL *Cells;
    extern M_INT32 num_rings;
    extern M_INT32 NumLattices;
    extern struct LATTICE *Lattices;
    extern M_INT32 NumAssemblies;
    extern struct ASSEMBLY *Assemblies;
    extern struct CORE *Core;
	extern M_INT32 NumContRodBund;
	extern struct CRB *ContRodBundles;
    extern FP_PRECISION_GEO *RayTracModDim;
    extern FP_PRECISION_GEO rated_power;
    extern FP_PRECISION_GEO rated_flow;
    extern FP_PRECISION_GEO tinlet;
    extern FP_PRECISION_GEO core_power;
    extern M_INT32 WestBC;
    extern M_INT32 NorthBC;
    extern M_INT32 EastBC;
    extern M_INT32 SouthBC;
    extern M_INT32 TopBC;
    extern M_INT32 BottomBC;  // 1 bound_cond
    extern M_INT32 MaxNumOuterIter_MOC,MaxNumOuterIter_MGCMFD,MaxNumOuterIter_CGCMFD,MaxNumIter_MGLS,MaxNumIter_CGLS;
    extern FP_DOUBLE MGCMFD_DampFactor, CGCMFD_DampFactor;
    extern M_INT32 NGSweep;
    extern M_INT32 MaxNumInnerIter;  // 2 iter_lim
    extern FP_PRECISION ConvCritK, ConvCritFlux,ConvCritLinearSolver_MGCMFD,ConvCritLinearSolver_CGCMFD,SOR_Factor_MG,SOR_Factor_CG;
    //extern FP_PRECISION_GEO ConvCritK_MGCMFD, ConvCritFlux_MGCMFD,ConvCritK_CGCMFD, ConvCritFlux_CGCMFD; 
    extern M_INT32 MOCSolverType;
    extern M_INT32 solverdim;  // 4 solver
    extern FP_PRECISION_GEO *RaySpacingIni;  // 5 ray
    extern M_INT32 *PolarQuadType;
#ifdef __cplusplus
extern "C" {
#endif
    extern M_INT32 *NumAzimAng;
    extern M_INT32 *NumPolarAng, IsSetHalfPolar;  // 5 ray
#ifdef __cplusplus
}
#endif
	extern M_INT32 NumCMFDGroup, NumCoarseCMFDGroup;
	extern M_INT32 CMFDSwitch;
	extern M_INT32 XS_leak_Switch;
	extern M_INT32 *CMFDGroupBound,*CGCMFDGroupBound;
    extern M_INT32 nspace;
    extern M_INT32 nangle;
    extern M_INT32 nenergy;
    extern M_INT32 nthreads;  // 6 6parallel
    extern M_INT32 vis_edits;
    extern M_INT32 opt;  // 7 vis_edits
	extern M_INT32 LinearInter_Exp;
    extern M_INT32 validation;
    extern M_INT32 delimeter_character;  // 9 validation
    extern M_INT32 scatt_meth;  // 10 scatt_meth
    extern M_INT32 cmfd;  // 11 cmfd
    extern M_INT32 bc_ave_xy;
    extern M_INT32 bc_ave_xz;
    extern M_INT32 bc_ave_yz;  // 12 bc_average
    extern FP_PRECISION_GEO underrelax; // 13 underrelax
    extern M_INT32 nodal_1;
    extern M_INT32 nodal_2;  // 14 nodal card
    extern M_INT32 power_edit;  // 15 power_edit
    extern M_INT32 *asse_id_map;
    extern char Mat_XSlib[NAME_N];
	extern FP_PRECISION_GEO KRef;
	extern M_INT32 IfRectangularMesh;
	extern M_INT32 RectMeshNumRow_perCell; // the number of rectangular mesh in a cell per row
	extern M_INT32 RectMeshNumCol_perCell; // the number of rectangular mesh in a cell per column
	extern M_INT32 *NumSubDomains, *HardwareMap_SubDom;
	extern M_INT32 ***SubDomainsMap;
	extern M_INT32 **NumPin_subDomain;
	extern M_INT32 NumProcPerLayer;
	extern M_INT32 PinPowerCheckSwitch;
	extern M_INT32 NumLayer_in_Slice;
	extern M_INT32 MG_LUFact_Index;
	extern M_INT32 CG_LUFact_Index;
	extern M_INT32 MG_PreCD_Index;
	extern M_INT32 CG_PreCD_index;
	extern M_INT32 ML_Switch;
	extern M_INT32 WS_Switch;	
	extern M_INT32 Iteration_Number_MG_CMFD;
	extern M_INT32 Iteration_Number_CG_CMFD;
	extern M_INT32 Outer_Iteration_Maximum;
	// extern M_INT32 MOC_BoundFlux_UpdateImme;
	extern M_INT32 *FSRVolumeCorrectionType;
	extern FP_PRECISION_GEO CoreArea2D,CoreLengthX,CoreLengthY;
	extern char IP[512];
	extern M_INT32 NumTrackSet, NumPolAngPart, *NumGroupBound;
	extern M_INT32 **GroupBoundary,*PolarAngBoundary;
	extern M_INT32 GPU_TSKernelType;
    extern M_INT32 GPU_FixSourSolver;
	extern M_INT32 GPUBlockDimInp, GPUNumStreamsInp;
	extern M_INT32 NumGPUsPerNode;
	extern M_INT32 CPU_TSFunctionType;
    extern M_INT32 CPU_FixSourSolver;
	extern M_INT32 NumOMPThread;
	extern M_INT32 *CUDADeviceID;
    // extern M_INT32 IfConvertToCSR_Matrix;
    extern M_INT32 *FSRsMatIDinRTM;
    extern M_INT32 PinPowerSumNumSet, *PinPowerSumLayBound;
	extern M_INT32 InputTrackFile;
	extern char TrackFileName[NAME_N];
    extern M_INT32 IsCheckExpInteRange;

	extern M_INT32 MGCMFD_checkFluxNormInf;
	extern M_INT32 MGCMFD_checkFluxNorm2;
	extern M_INT32 MGCMFD_checkFluxRMSE;

	extern M_INT32 CGCMFD_checkFluxNormInf;
	extern M_INT32 CGCMFD_checkFluxNorm2;
	extern M_INT32 CGCMFD_checkFluxRMSE;

	extern M_INT32 MOC_checkFluxNormInf;
	extern M_INT32 MOC_checkFluxNorm2;
	extern M_INT32 MOC_checkFluxRMSE;
	extern FP_PRECISION MOC_Trans_Corre_SOR;
    void JumpCommentsInFile(FILE *fp);
    void MOCInput(M_INT32 _procID);
    void IdentifyKeyword(FILE *fp1, char *_keyword, char *p);
	void InputCMFDInfo(FILE *_fp);
    void MOC_PrintLogFile(M_INT32 _procID);
    FP_PRECISION_TIME CPU_GetCurrTime();
    FP_PRECISION_TIME CPU_GetTimeDuration(FP_PRECISION_TIME _start, FP_PRECISION_TIME _end);
    M_INT32 CountParameterInFile(char *_fileName, char *_parameter);
    void InputFromTrackFile(char *_fileName);
    void GetNumSegInTrackFromTrackFile(char *_fileName, M_INT32 *_NumSegsInTrack_2PI, M_INT32 *_NumTracks, M_INT32 _NumAzimAng);
    void GetNumTracksPerAzimFromTrackFile(char *_fileName, M_INT32 *_NumTracks, M_INT32 _NumAzimAng);
    M_INT32 GetNumTracksOctant(M_INT32 _octID, M_INT32 *_NumTracks, M_INT32 _NumAzimAng);
	void GetRTMInfoFromTrackFile(char *_fileName, M_INT32 _NumRTMGeoType, M_INT32 *_NumFSRsInRTM, M_INT32 *_SumNumFSRsInRTM, M_INT32 _TotalNumFSRsInRTM, \
		M_INT32 _TotNumTracksInOneRTM_2PI, M_INT32 *_NumTracks, M_INT32 _NumAzimAng, M_INT32 *_NumSegsInTrack_2PI, FP_PRECISION *_SegInfoInTrack_2PI_length, \
		M_INT32 *_SegInfoInTrack_2PI_AntiClockFSR, M_INT32 *_SumNumSegsInRTMTrack_2PI, M_INT32 *_IndexTrackInRTM_SumAzim_2PI);
    void GetIndexOfTracksInRTM_SumAzim(M_INT32 _numTrackSet, M_INT32 *_numAzimAng, M_INT32 **_numTrack, \
	    M_INT32 **_IndexTrackInRTM_SumAzim_2PI);
    void GetSumNumSegsInRTMTrack_2PI(M_INT32 _numTrackSet, M_INT32 *_TotalNumTracksInRTM_Total, \
        M_INT32 **_NumSegsInTrack_2PI, M_INT32 **_SumNumSegsInRTMTrack_2PI);
    void SetTrackInfoInLTs_trackFile(M_INT32 _NumTrackSet, M_INT32 *_NumAzimAng, M_INT32 **_NumTracks, \
        struct SUBDOM_TRACK_IN_LONGTRACK **_TracksInfoInLTs_2PI);

void SetLTsLinkInfo_trackFile(struct SUBDOM_LONGTRACK_INFO **_longTracksInfo, M_INT32 _numTrackSet, \
    M_INT32 *_numAzimAng, M_INT32 **_numLongTracks_SubDom, M_INT32 **_numLongTracks_SubDomX, M_INT32 **_numLongTracks_SubDomY);
void GetNumTracksInXandY_trackFile(char *_fileName, M_INT32 _numTrackSet, M_INT32 *_numAzimAng, \
    M_INT32 **_numLongTracks_SubDomX, M_INT32 **_numLongTracks_SubDomY, FP_PRECISION **_EffAzimAng, \
	FP_PRECISION **_EffRaySpacing);
M_INT32 GetNumFSRsInRTM_TrackFile(char *_fileName, M_INT32 *_NumFSRsInRTM);
M_INT32 GetTotalNumFSRs_SubDom_TrackFile(M_INT32 *_NumFSRsInRTM, M_INT32 _NumPinsInSubDom, M_INT32 *_CellRTMType_SubDom);
void SetNumFSRsInCell_trackFile(M_INT32 *_NumFSRsInCell, M_INT32 _MOC_NumPinsInSubDom, \
	M_INT32 *_NumFSRsInRTM, M_INT32 *_CellRTMType_SubDom);
void SetFSRArea_trackFile(M_INT32 _MOC_NumPinsInSubDom, M_INT32 *_NumFSRsInRTM, M_INT32 *_SumNumFSRsInRTM, \
	M_INT32 *_CellRTMType_SubDom, FP_PRECISION_GEO *_FSRsAreaInRTM, FP_PRECISION_GEO *_FSRArea_SubDom, \
	M_INT32 *_SumNumFSRsBasedGlobCellID);
void SetFSRsXSID_trackFile(M_INT32 _MOC_NumPinsInSubDom, M_INT32 *_NumFSRsInRTM, M_INT32 *_SumNumFSRsInRTM, \
	M_INT32 *_CellRTMType_SubDom, M_INT32 *_FSRsXSID_subDom, M_INT32 *_FSRsMatIDinRTM, M_INT32 *_SumNumFSRsBasedGlobCellID);
FP_PRECISION_TIME TallyCurrentTime_CPU();
void GodBless(char godName[CASEID_N]);

void logo();
#endif
