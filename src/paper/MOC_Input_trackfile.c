/********************************************************************************/
/* @ File: MOC_Input.c
/*   introduction: input from external file
/* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*   1st modified:
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "MOC_AllocAndFreeMem.h"
#include "MOC_Assembly.h"
#include "MOC_Boundary.h"
#include "MOC_Cell.h"
#include "MOC_CMFD.h"
#include "MOC_CommonMacro.h"
#include "MOC_Core.h"
#include "MOC_Geometry.h"
#include "MOC_Input.h"
#include "MOC_Lattice.h"
#include "MOC_Macro_XSs.h"
#include "MOC_main.h"
#include "MOC_Output.h"
#include "MOC_Pin_Cell.h"
#include "MOC_Quadrature_Set.h"
#include "MOC_Track.h"
#include "MOC_Visualization.h"
#include "auxiliary.h"
#ifdef MPI_USED
  #include <mpi.h>
#endif

#ifdef OMP_USED
  #include <omp.h>
#endif // OMP_USED

#include "R_I_O_File.h"

#ifdef CUDA_USED
  #include "cuda_runtime.h"
#endif

/**
* @ Function: JumpCommentsInFile()
*   introduction: jump the comments when reading values from an external file
* @ Parameter List: _fp - FILE pointer
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/

FP_PRECISION MOC_Trans_Corre_SOR = 1.0;

void JumpCommentsInFile(FILE *_fp)
{
    char _s[OVER_N],*_inpErr;
    fpos_t _fpos;
    fgetpos(_fp,&_fpos);
	_inpErr=fgets(_s,sizeof(_s),_fp);
    while((_s[0]=='!')||(_s[0]=='\n') || (_s[0] == '\r') ||(_s[0]=='\t')) {
        fgetpos(_fp,&_fpos);
		_inpErr=fgets(_s,sizeof(_s),_fp);
		if(_inpErr == NULL)
            break;
    }
    fsetpos(_fp,&_fpos);
}

/**
* @ Function: IdentifyKeyword()
*   introduction: identify the _keyword (using as locator) when reading values from an external file
* @ Parameter List: _fp - FILE pointer
*                   _keyword - required _keyword
*                   _parameter - parameter name
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void IdentifyKeyword(FILE *_fp, char *_keyword, char *_parameter)
{
    M_INT32 i,_inpErr;
    for (i=0;i<KEYWORD_N;i++)
        _keyword[i]='\0';
    JumpCommentsInFile(_fp);
    _inpErr=fscanf(_fp,"%s",_keyword);
    if (strcmp(_keyword,_parameter)!=0) {
        printf("\n [**ERROR**]  Reading \"%s\" !",_parameter);
		fflush(stdout);
		exit(69);
    }
}

/**
* @ Function: IdentifyKeyword()
*   introduction: identify the _keyword (using as locator) when reading values from an external file
* @ Parameter List: _fp - FILE pointer
*                   _keyword - required _keyword
*                   _parameter - parameter name
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
    FILE *LogFile = NULL;
	char OutputDir[CASEID_N];
    char CaseId[CASEID_N];  // parameter--caseid
	M_INT32 KType;  // boolean value, 0 or 1, 1--calculate the k-effective, 0--calculate the k-infinite, ifKEFF
	M_INT32 IfResonance, ProblemType, isAdjoint;  // boolean value, 0 or 1, 1--solve fixed source problem, 0--eigenvalue problem
	M_INT32 NumCylinders;  //number of cylinders which constitute the Primitives
	struct PARA_CYLINDER *Cylinders = NULL;  // structure--cylinders
	M_INT32 NumPrimitives;  // number of Primitives
	struct CELL *Primitives = NULL;  // structure--Primitives
    M_INT32 NumCells;  // number of types of different cells in Core
    struct PIN_CELL *Cells = NULL;  // structer--cells in Core
    M_INT32 NumLattices;  // number of types of different lattices in Core
    struct LATTICE *Lattices = NULL;  // structure--lattices in Core
    M_INT32 NumAssemblies;  // number of types of different assemblies in Core, num_assembly
    struct ASSEMBLY *Assemblies = NULL;  // structure--assemblies in Core, assembly
    struct CORE *Core = NULL;  // structure--Core geometry information, Core
    M_INT32 NumContRodBund;  // number of types of different GPU_TSKernelType rod bundle groups, num_crbg
    struct CRB *ContRodBundles = NULL;  // structure--different types of GPU_TSKernelType rod bundles in Core, crbGroups
    FP_PRECISION_GEO *RayTracModDim = NULL;  // ray tracing module dimension, mod_dim
	M_INT32 WestBC, NorthBC, EastBC, SouthBC, TopBC, BottomBC;  // 1 bound_cond
	M_INT32 MaxNumOuterIter_MOC,NGSweep, MaxNumInnerIter,MaxNumOuterIter_MGCMFD,MaxNumIter_MGLS;  // 2 iter_lim
	FP_DOUBLE MGCMFD_DampFactor, CGCMFD_DampFactor;
	M_INT32 MaxNumOuterIter_CGCMFD,MaxNumIter_CGLS;
    FP_PRECISION ConvCritK,ConvCritFlux,ConvCritLinearSolver_MGCMFD,ConvCritLinearSolver_CGCMFD, SOR_Factor_MG,SOR_Factor_CG;  // 3 conv_crit
    //FP_PRECISION_GEO ConvCritK_MGCMFD, ConvCritFlux_MGCMFD,ConvCritK_CGCMFD, ConvCritFlux_CGCMFD; 
	FP_PRECISION_GEO *RaySpacingIni=NULL;  // 5 ray
	M_INT32 *PolarQuadType=NULL, *NumAzimAng=NULL, *NumPolarAng=NULL, IsSetHalfPolar=0;  // 5 ray
	M_INT32 CMFDSwitch = 0;
	M_INT32 XS_leak_Switch = 0;
	M_INT32 PinPowerCheckSwitch = 1;
	M_INT32 NumLayer_in_Slice = 1;
	M_INT32 MG_LUFact_Index = 0;
	M_INT32 CG_LUFact_Index = 0;
	M_INT32 MG_PreCD_Index = 0;
	M_INT32 CG_PreCD_index = 0;
	M_INT32 ML_Switch = 0;
	M_INT32 WS_Switch = 0;
	M_INT32 Iteration_Number_MG_CMFD = 0;
	M_INT32 Iteration_Number_CG_CMFD = 0;
	M_INT32 Outer_Iteration_Maximum = 1;
	M_INT32 NumCMFDGroup;   // number of energy groups used in M_MG_CMFD after energy condensation
	M_INT32 NumCoarseCMFDGroup;   // number of energy groups used in CG-M_MG_CMFD	
	M_INT32 *CMFDGroupBound = NULL;
	M_INT32 *CGCMFDGroupBound = NULL;
	FP_PRECISION_GEO rated_power,rated_flow,tinlet,core_power;
    M_INT32 MOCSolverType,solverdim;  // 4 solver 0-gauss seidel method(moc solver), 1-jacobian method (moc solver)(default)
    M_INT32 nspace,nangle,nenergy,nthreads;  // 6 6parallel
    M_INT32 vis_edits,opt;  // 7 vis_edits
    // 8 liner_exp
	M_INT32 LinearInter_Exp;     //1-use interpolation to evaluate exponential, 0-direct calculate
    M_INT32 validation,delimeter_character;  // 9 validation
    M_INT32 scatt_meth;  // 10 scatt_meth
    M_INT32 cmfd;  // 11 cmfd
    M_INT32 bc_ave_xy,bc_ave_xz,bc_ave_yz;  // 12 bc_average
    FP_PRECISION_GEO underrelax; // 13 underrelax
    M_INT32 nodal_1,nodal_2;  // 14 nodal card
    M_INT32 power_edit;  // 15 power_edit
    // 16 PARAM Card
    // 17 AMPX_OUT Card . CURRENT DISABLED
    // 18 CRITBORON Card
    // 19 XENON Card
    char Mat_XSlib[NAME_N];
	FP_PRECISION_GEO KRef;  // the reference value of multiplication factor, k_ref
	M_INT32 IfRectangularMesh=0; // if the mesh is reactangular for all FSRs, 
	                       //then the scalar flux map will be generated for assemblies as the FSR map
	M_INT32 RectMeshNumRow_perCell = 0; // the number of rectangular mesh in a cell per row
	M_INT32 RectMeshNumCol_perCell = 0; // the number of rectangular mesh in a cell per column
	// Information for MPI
	M_INT32 *NumSubDomains = NULL; // number of sundomains along x,y,z directions
	M_INT32 *HardwareMap_SubDom = NULL;
	M_INT32 ***SubDomainsMap = NULL; //3D subdomains' map raleted to processes
	M_INT32 **NumPin_subDomain = NULL; // number of pins in each subdomain for 2D layout, and number of layers in axial direction
	M_INT32 NumProcPerLayer = 0; // number of subdomains per axial layer
	// M_INT32 MOC_BoundFlux_UpdateImme = 1;   // 1-update boundary angular flux in current iteration    0-for next iter
	M_INT32 *FSRVolumeCorrectionType;  // 0-no correction 1-correction
    FP_PRECISION_GEO CoreArea2D,CoreLengthX,CoreLengthY;
	char IP[512];
	// number of track set, different parameter - [NumGeometrySet] number of azimuthal angle, ray spacing
    M_INT32 NumTrackSet,NumPolAngPart, *NumGroupBound=NULL;
	M_INT32 **GroupBoundary = NULL,*PolarAngBoundary = NULL;
	M_INT32 GPU_TSKernelType=0;
	M_INT32 GPU_FixSourSolver=0;
	M_INT32 GPUBlockDimInp = 128;
	M_INT32 NumGPUsPerNode;
	M_INT32 CPU_TSFunctionType=0;
	M_INT32 CPU_FixSourSolver=0;
	M_INT32 GPUNumStreamsInp;
	M_INT32 NumOMPThread;
	M_INT32 *CUDADeviceID = NULL;
    // M_INT32 IfConvertToCSR_Matrix = 1;
	M_INT32 *FSRsMatIDinRTM = NULL;
	M_INT32 PinPowerSumNumSet, *PinPowerSumLayBound=NULL;
	M_INT32 InputTrackFile = 0;
	char TrackFileName[NAME_N];
	M_INT32 IsCheckExpInteRange=1;

	M_INT32 MGCMFD_checkFluxNormInf = 0;
	M_INT32 MGCMFD_checkFluxNorm2 = 0;
	M_INT32 MGCMFD_checkFluxRMSE = 1;

	M_INT32 CGCMFD_checkFluxNormInf = 0;
	M_INT32 CGCMFD_checkFluxNorm2 = 0;
	M_INT32 CGCMFD_checkFluxRMSE = 1;

	M_INT32 MOC_checkFluxNormInf = 0;
	M_INT32 MOC_checkFluxNorm2 = 0;
	M_INT32 MOC_checkFluxRMSE = 1;

void logo() {
    printf("\n            $$$$           $                 $$$$           $       $           $$$$    ");
    printf("\n         $$$$$$$$$$     $$$$$$            $$$$$$$$$$     $$$$$$  $$$$$$      $$$$$$$$$$ ");
    printf("\n       $$         $$   $$    $$         $$         $$   $$    $$$$    $$   $$        $$$");
    printf("\n      $$          $$  $$     $$        $$          $$  $$     $$$     $   $$          $$");
    printf("\n     $$           $$  $$    $$        $$            $  $     $$$$    $$  $$           $$");
    printf("\n     $     $$     $$  $     $$        $     $$$     $ $$     $$$     $$ $$     $$     $$");
    printf("\n    $$    $$$     $$ $$     $        $$     $$$    $$ $$     $$$     $  $     $$$     $$");
    printf("\n   $$             $$ $     $$       $$     $$$     $$ $             $$ $$             $ ");
    printf("\n   $$             $ $$     $$       $$     $$$     $ $$             $$ $$            $$ ");
    printf("\n   $             $$ $$     $        $     $$      $$ $$            $$ $$             $$ ");
    printf("\n  $$             $$ $     $$       $$            $$ $$             $$ $$             $  ");
    printf("\n  $$    $$$$    $$ $$     $$       $$           $$  $$     $$$     $  $     $$$     $$  ");
    printf("\n  $     $$$     $$ $$     $$$$$$   $          $$$   $     $$$$    $$ $$     $$$     $$  ");
    printf("\n $$     $$$     $  $$          $$ $$     $$$$$$    $$     $$$     $$ $$    $$$$     $   ");
    printf("\n $     $$$$    $$  $$          $$ $     $$         $$    $$$$     $  $     $$$     $$   ");
    printf("\n $     $ $     $$  $$$        $$  $     $          $$    $$$$    $$ $$    $$$$     $    ");
    printf("\n $$$$$$  $$$$$$     $$$$$$$$$$$   $$$$$$           $$$$$$$ $$$$$$$   $$$$$$  $$$$$$     ");
    printf("\n   $$      $$          $$$$$$       $                $$      $$       $$$     $$$       ");
}

/**
* @ Function: MOCInput()
*   introduction: initial input of the whole calculation
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified: 2018/12/09, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*                 add time recording
*/
void MOCInput(M_INT32 _procID)
{
    M_INT32 i,j,k,_bool_footprint,_dataNumber, _count, _count_1, _inpErr;
    M_INT32 *_numBoundSurf=NULL,*_numCells=NULL,*_lattsInAsse=NULL,*_asseSubLayer = NULL;
	M_INT32 *_asseWidthJudge = NULL;
    FP_PRECISION_GEO *_lattsInAsseWidthZ = NULL;
	FP_PRECISION _dataSizeMB, _pointerSizeMB;
	char _keyWord[KEYWORD_N], _MemFPFileName[256]/*, _fileName[128],_name1[128]*/;
	FILE *_ip=NULL;

	M_INT32 _counter1, _numLattsZ, j1,_boundSurfId,_surfType,_cellId,_counter2,_boundType,i1;
	FP_PRECISION _halfSpace=0.0;
	FP_PRECISION_GEO _check;

    char _prob_type[2];
    int _fpPosRec;


#ifdef TIME_REC
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC

	_bool_footprint = 0;
	_dataNumber = 0;
	_dataSizeMB = 0.;
	_pointerSizeMB = 0.;
    _ip=fopen(IP,"r");
    if(_ip==NULL) {
        printf("\n [**ERROR**]  Input file opens error with IP = \"%s\" in process %d\n", IP, _procID);
		fflush(stdout);
		exit(69);
    }
    IdentifyKeyword(_ip,_keyWord,"CASEID_BLOCK");
    IdentifyKeyword(_ip,_keyWord,"CASEID");
//    JumpCommentsInFile(_ip);
    _inpErr=fscanf(_ip,"%s",CaseId);
    GodBless(CaseId);
	if (_procID == 0) {
        printf("\n");
//        printf("\n        $$$$$         $$$       $$$$$$$$$     $$$$      $$$$         $$$$$       ");
//        printf("\n        $$$$$         $$$       $$$$$$$$$$$   $$$$      $$$$         $$$$$       ");
//        printf("\n       $$$$$$$        $$$       $$$$    $$$   $$$$      $$$$        $$$$$$$      ");
//        printf("\n       $$$ $$$        $$$       $$$$    $$$   $$$$      $$$$        $$$ $$$      ");
//        printf("\n      $$$$ $$$$       $$$       $$$$   $$$$   $$$$$$$$$$$$$$       $$$$ $$$$     ");
//        printf("\n     $$$$   $$$$      $$$       $$$$$$$$$$    $$$$$$$$$$$$$$      $$$$   $$$$    ");
//        printf("\n     $$$     $$$      $$$       $$$$$$$       $$$$      $$$$      $$$     $$$    ");
//        printf("\n    $$$$$$$$$$$$$     $$$       $$$$          $$$$      $$$$     $$$$$$$$$$$$$   ");
//        printf("\n    $$$$$$$$$$$$$     $$$       $$$$          $$$$      $$$$     $$$$$$$$$$$$$   ");
//        printf("\n   $$$$       $$$$    $$$       $$$$          $$$$      $$$$    $$$$       $$$$  ");
//        printf("\n  $$$$         $$$$   $$$$$$$$  $$$$          $$$$      $$$$   $$$$         $$$$ ");
//        printf("\n  $$$           $$$   $$$$$$$$  $$$$          $$$$      $$$$   $$$           $$$ ");

//        printf("\n  $      $                 $$        $$$$$                     $$            $$$$$        ");
//        printf("\n  $$$$$$$$$$ $$$$$$$$$$$$$$$$$$      $$$$                        $$$          $$$         ");
//        printf("\n  $$$   $$$             $$$         $$$$              $$         $$$$  $      $$$         ");
//        printf("\n  $$$  $$$              $$$        $$$$$$$$$$$$$$$$$$$$$$         $$$ $       $$$    $$   ");
//        printf("\n  $$$  $$    $          $$$        $$      $         $$$$$  $$       $ $$$$$$$$$$$$$$$$$$ ");
//        printf("\n  $$$  $     $$$$$$$$$$ $$$       $$        $$$     $$       $$$$    $        $$$         ");
//        printf("\n  $$$ $$     $$$   $$$  $$$      $$         $$$    $          $$$$  $         $$$         ");
//        printf("\n  $$$  $$    $$$   $$$  $$$     $$    $     $$$                $$  $$         $$$     $$$ ");
//        printf("\n  $$$   $$$  $$$   $$$  $$$   $$     $$$$$  $$$ $$$               $$ $$$$$$$$$$$$$$$$$$$$$");
//        printf("\n  $$$    $$  $$$   $$$  $$$          $$$    $$$   $$$             $$         $$$          ");
//        printf("\n  $$$    $$$ $$$$$$$$$  $$$         $$$     $$$     $$$          $$          $$$$         ");
//        printf("\n  $$$$$$$$$  $$$   $$$  $$$        $$$      $$$      $$$$      $$$$         $$$  $        ");
//        printf("\n  $$$  $$$   $$$        $$$       $$        $$$       $$$$    $$$$        $$$     $$      ");
//        printf("\n  $$$                   $$$     $$$         $$$       $$$$      $$       $$$       $$$    ");
//        printf("\n  $$$                   $$$    $$           $$$        $$$     $$$      $$          $$$$  ");
//        printf("\n  $$$             $$$$$$$$$  $$       $$$$$$$$$                $$$$  $$$$$$$$$$$$$$  $$$$ ");
//        printf("\n  $$$                 $$$$$              $$$$$$                $$$$   $$$$$$         $$$$ ");
//        printf("\n  $$$                 $$$                 $$$                   $$$                   $$  ");

        printf("\n");
        printf("\n [  TITLE  ]  ************************************************************");
        printf("\n [  TITLE  ]  *                SIMULATION PARAMETERS INFO                *");
        printf("\n [  TITLE  ]  ************************************************************");

#if defined(MPI_USED)
		for (i=0;i<NumProcs;i++) {
			if (i==0) {
				printf("\n [  INFO   ]  Process ID: %3d, Processor Name: %6s, CPU ID:%3d", i, ProcessorNameColl[i], CurrCPUIDColl[i]);
			}
			else {
				printf("\n              Process ID: %3d, Processor Name: %6s, CPU ID:%3d", i, ProcessorNameColl[i], CurrCPUIDColl[i]);
			}
		}
	    fflush(stdout);
#endif

		printf("\n [  INFO   ]  Input File Name:   %s", IP);
		printf("\n [  INFO   ]  CASE ID:           %s", CaseId);
		CheckDataTypeRange(ProcID, NULL);
#ifdef DOUBLE
        printf("\n [  INFO   ]  The Floating-Point Precision is DOUBLE");
#endif
#ifdef SINGLE
        printf("\n [  INFO   ]  The Floating-Point Precision is SINGLE");
#endif
		fflush(stdout);
	}
	else {}
#if defined(MPI_USED)
    MPI_Barrier(MPI_COMM_WORLD);
#endif

	// output memory footprint
	sprintf(_MemFPFileName, "%s-%s-Proc%d%s", CaseId, "MemoryFootprint", _procID, ".out");
	FP_MemoryFootprint = fopen(_MemFPFileName, "w");
	if (FP_MemoryFootprint == NULL) {
		printf(" [**ERROR**]  %s file opens error!\n", _MemFPFileName);
		fflush(stdout);
		exit(69);
	}
	ParamCounter = 0;
	MemPrintCount = 0;
	sprintf(MemParaName[MemPrintCount], "Geo and Track Info");
	MemParaCountBoundLow[MemPrintCount] = ParamCounter; /* Mem - Geometry and Track Info*/
	fprintf(FP_MemoryFootprint, "\n   *************************************************");
	fprintf(FP_MemoryFootprint, "\n   *                Memory Footprint               *");
	fprintf(FP_MemoryFootprint, "\n   *************************************************");
	fprintf(FP_MemoryFootprint, "\n\n%-*s%*s", MEM_9, "DataType", MEM_0, "DataSize(B)");
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "char", MEM_0, sizeof(char));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "M_INT32", MEM_0, sizeof(M_INT32));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "FP_PRECISION_GEO", MEM_0, sizeof(FP_PRECISION_GEO));
	fprintf(FP_MemoryFootprint, "\n\n%-*s%*zd", MEM_9, "struct MAT", MEM_0, sizeof(struct MAT));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "struct CR_MAT", MEM_0, sizeof(struct CR_MAT));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "struct CR_MAT_DIS", MEM_0, sizeof(struct CR_MAT_DIS));
	fprintf(FP_MemoryFootprint, "\n\n%-*s%*zd", MEM_9, "struct POINT", MEM_0, sizeof(struct POINT));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "enum surfaceType", MEM_0, sizeof(enum surfaceType));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "enum boundaryType", MEM_0, sizeof(enum boundaryType));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "struct PLANE", MEM_0, sizeof(struct PLANE));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "struct PARA_CYLINDER", MEM_0, sizeof(struct PARA_CYLINDER));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "struct CELL", MEM_0, sizeof(struct CELL));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, " ->struct SURFACE_HALFSPACE", MEM_0, sizeof(struct SURFACE_HALFSPACE));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "  ->union Surfaces", MEM_0, sizeof(union Surfaces));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "struct PIN_CELL", MEM_0, sizeof(struct PIN_CELL));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, " ->struct CELLS_IN_PINCELL", MEM_0, sizeof(struct CELLS_IN_PINCELL));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "struct LATTICE", MEM_0, sizeof(struct LATTICE));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "struct ASSEMBLY", MEM_0, sizeof(struct ASSEMBLY));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "struct CORE", MEM_0, sizeof(struct CORE));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "struct CRB", MEM_0, sizeof(struct CRB));
	fprintf(FP_MemoryFootprint, "\n\n%-*s%*zd", MEM_9, "struct SUBDOM_SEG_IN_TRACK", MEM_0, sizeof(struct SUBDOM_SEG_IN_TRACK));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, " ->struct SEG_IN_FSR", MEM_0, sizeof(struct SEG_IN_FSR));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "struct SUBDOM_LONGTRACK_INFO", MEM_0, sizeof(struct SUBDOM_LONGTRACK_INFO));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, " ->struct SUBDOM_TRACK_IN_LONGTRACK", MEM_0, sizeof(struct SUBDOM_TRACK_IN_LONGTRACK));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, "struct RTMFSRINFO", MEM_0, sizeof(struct RTMFSRINFO));
	fprintf(FP_MemoryFootprint, "\n%-*s%*zd", MEM_9, " ->struct SEG_IN_FSR", MEM_0, sizeof(struct SEG_IN_FSR));
	fprintf(FP_MemoryFootprint, "\n\n%*s%*s%*s%*s%*s%*s%*s%*s", \
		MEM_1, "NO.", \
		MEM_2, "Parameter", \
		MEM_3, "DataType", \
		MEM_4, "DataNumber", \
		MEM_5, "PointerSize(MB)", \
		MEM_6, "DataSize(MB)", \
		MEM_8, "TotalSize(MB)", \
		MEM_10, "MemorySum(MB)");

	IdentifyKeyword(_ip, _keyWord, "UseTrackFile");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &InputTrackFile);
	IdentifyKeyword(_ip, _keyWord, "TrackFileName");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%s", TrackFileName);

	IdentifyKeyword(_ip, _keyWord, "If_Resonance");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &IfResonance);

    IdentifyKeyword(_ip, _keyWord, "ProblemType");
//    JumpCommentsInFile(_ip);
    _fpPosRec = ftell(_ip);

    // if problem is neg, the TS mode is adjoint
    _inpErr = fscanf(_ip, "%s", &_prob_type[0]);
    if(_prob_type[0] == '-') isAdjoint = 1;

    fseek(_ip, _fpPosRec, SEEK_SET);
	_inpErr=fscanf(_ip, "%d", &ProblemType);
	ProblemType = abs(ProblemType);


    if (InputTrackFile != 1 || IfResonance != 0 || ProblemType != 0) {
        printf("\n [**ERROR**] Publication package requires UseTrackFile=1, If_Resonance=0 and ProblemType=0.\n");
        fflush(stdout);
        exit(69);
    }
#if defined(PAPER_TRANSPORT_ONLY)
    if (ProblemType == 2) {
        printf("\n [**ERROR**]  Transient ProblemType=2 is not included in the publication transport package.\n");
        fflush(stdout);
        exit(69);
    }
#endif

	//Read MATERIAL or Macro Corss Section information
	/** Material or Macro XS Block*/
	IdentifyKeyword(_ip, _keyWord, "MATERIAL_BLOCK");
//	JumpCommentsInFile(_ip);
	_inpErr=fscanf(_ip, "%s", Mat_XSlib);
#ifdef ZhangYunfei
    sprintf(Mat_XSlib, "%s_%d%s", Mat_XSlib, _procID,".txt");
#endif //ZhangYunfei
	if (_procID == 0) {
		printf("\n [  INFO   ]  Setting Macroscopic Cross Section Data ...");
		fflush(stdout);
	}
#ifndef ZhangYunfei_Visu

    strcpy(Mat_XSlib, cleanComments(Mat_XSlib, "v_v", '!'));
    if (IfResonance == 0)
	{
		InputMat_XSs(Mat_XSlib);
	}
	else
	{
	    #ifndef TESTR
		R_I_O_Res(_procID);
	    #endif
		//NumGroups = 47;
	}
	
#endif

    /**
     * Read GEOMETRY information - surfaces
     * Read Planes information
     */
    //OPITION BLOCK

    IdentifyKeyword(_ip,_keyWord,"OPTION_BLOCK");
    //Read
    // 1 bound_cond default
    WestBC=1;
    NorthBC=1;
    EastBC=1;
    SouthBC=1;
    TopBC=1;
    BottomBC=1;
    // 2 iter_lim  DEFAULT
    MaxNumOuterIter_MOC=500;
	MaxNumOuterIter_MGCMFD = 200;
    NGSweep=2;
    MaxNumInnerIter=3;
    // 3 conv_crit  DEFAULT
    ConvCritK=1.e-6;
    ConvCritFlux=1.e-6;
	//ConvCritK_MGCMFD= 0.5 * ConvCritK;
	//ConvCritFlux_MGCMFD= 0.5 * ConvCritFlux;
	//ConvCritLinearSolver_MGCMFD= 0.5 * ConvCritFlux;
	SOR_Factor_MG=1.0;
	SOR_Factor_CG=1.0;
    // 4 solver  DEFAULT
    MOCSolverType=0;
	//FSRVolumeCorrectionType = 0;
	// ray
	LinearInter_Exp = 1;  // 0-intrinsic exp(x) function, 1-linear exponential table
    // 1 bound_cond
	IdentifyKeyword(_ip, _keyWord, "BC");
//    JumpCommentsInFile(_ip);
    _inpErr=fscanf(_ip,"%d%d%d%d%d%d",&WestBC,&NorthBC,&EastBC,&SouthBC,&TopBC,&BottomBC);
    // 2 iter_lim
	// 4 solver
	IdentifyKeyword(_ip, _keyWord, "MOC_SOLVER");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &MOCSolverType);
	IdentifyKeyword(_ip, _keyWord, "MOC_ITER_LIM");
//    JumpCommentsInFile(_ip);
    _inpErr=fscanf(_ip,"%d%d%d",&MaxNumOuterIter_MOC,&NGSweep,&MaxNumInnerIter);
	// 3 conv_crit
	IdentifyKeyword(_ip, _keyWord, "MOC_CONV_CRIT");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, FP_FORMAT, &ConvCritK);
	_inpErr = fscanf(_ip, FP_FORMAT, &ConvCritFlux);

	IdentifyKeyword(_ip, _keyWord, "MOCFluxErrLInf");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &MOC_checkFluxNormInf);

	IdentifyKeyword(_ip, _keyWord, "MOCFluxErrL2");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &MOC_checkFluxNorm2);

	IdentifyKeyword(_ip, _keyWord, "MOCFluxErrRMSE");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &MOC_checkFluxRMSE);
	if (MOC_checkFluxNormInf==0 && MOC_checkFluxNorm2==0 && MOC_checkFluxRMSE==0) {
		MOC_checkFluxRMSE = 1;
	}
	if (IfResonance == 1) {
		MOC_checkFluxNormInf = 0;
		MOC_checkFluxNorm2 = 0;
		MOC_checkFluxRMSE = 1;
	}
 	IdentifyKeyword(_ip, _keyWord, "MOC_TRANS_CORRE_SOR");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, FP_FORMAT, &MOC_Trans_Corre_SOR);

	// 5 ray
	IdentifyKeyword(_ip, _keyWord, "NumPolarAnglePartition");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &NumPolAngPart);
	NumGroupBound = AllocMem_i32_i32(NumPolAngPart, (char *)("NumGroupBound"));
	IdentifyKeyword(_ip, _keyWord, "NumGroupBoundPerPolarAngPart");
	NumTrackSet = 0;
	for (i = 0; i < NumPolAngPart; i++) {
//		JumpCommentsInFile(_ip);
		_inpErr = fscanf(_ip, "%d", &NumGroupBound[i]);
		NumTrackSet += NumGroupBound[i];
	}
	RaySpacingIni = AllocMem_FPGeo_I(NumTrackSet, (char *)("RaySpacingIni"));
	NumAzimAng = AllocMem_i32_i32(NumTrackSet, (char *)("NumAzimAng"));
	PolarQuadType = AllocMem_i32_i32(NumTrackSet, (char *)("PolarQuadType"));
	NumPolarAng = AllocMem_i32_i32(NumTrackSet, (char *)("NumPolarAng"));
	GroupBoundary = AllocMem_int_IPplusI(NumPolAngPart, NumGroupBound, 1, (char *)("GroupBoundary"));
	PolarAngBoundary = AllocMem_i32_i32(NumPolAngPart + 1, (char *)("PolarAngBoundary"));
	FSRVolumeCorrectionType = AllocMem_i32_i32(NumTrackSet, (char *)("FSRVolumeCorrectionType"));
	PolarAngBoundary[0] = 0;
	_count = 0;
	for (i = 0; i < NumPolAngPart; i++) {
		GroupBoundary[i][0] = 0;
		for (j = 0; j < NumGroupBound[i]; j++) {
			RaySpacingIni[_count] = 0.05;
			NumAzimAng[_count] = 16;
			PolarQuadType[_count] = 0;
			NumPolarAng[_count] = 3;
			FSRVolumeCorrectionType[_count] = 1;
			_count++;
		}
	}
	IdentifyKeyword(_ip, _keyWord, "RAY_PARA");
//	JumpCommentsInFile(_ip);
	IsSetHalfPolar = 0;
#if defined(M_2DMOC)
    if (NumScattOrder == 0) {
		IsSetHalfPolar = 1;
	}
#endif 
	
	_count = 0;
	for (i = 0; i < NumPolAngPart; i++) {
		GroupBoundary[i][0] = 0;
		for (j = 0; j < NumGroupBound[i]; j++) {
//			JumpCommentsInFile(_ip);
			_inpErr = fscanf(_ip, FP_FORMAT_GEO, &RaySpacingIni[_count]);
			_inpErr = fscanf(_ip, "%d%d%d%d", &NumAzimAng[_count], &PolarQuadType[_count], &NumPolarAng[_count], \
				&GroupBoundary[i][j + 1]);
		    #ifdef TESTR
			NumGroups = GroupBoundary[i][j + 1];
		    #endif
			if (IsSetHalfPolar != 1) {
				NumPolarAng[_count] *= 2;
			}

			if (j == 0) {
				_inpErr = fscanf(_ip, "%d%d", &PolarAngBoundary[i + 1], &FSRVolumeCorrectionType[_count]);
			}
			else {
				_inpErr = fscanf(_ip, "%*s%d", &FSRVolumeCorrectionType[_count]);
			}
			_count++;
		}
	}

	// check self-consistency
	_count = PolarQuadType[0];
	_count_1 = NumPolarAng[0];
	for (i = 0; i < NumTrackSet; i++) {
		if (_count != PolarQuadType[i]) {
			printf("\n [**ERROR**]  The same polar quadrature type should be uesed during the whole calculation");
			printf("\n              Please check and reset!");
			fflush(stdout);
			exit(69);
		}
		if (_count_1 != NumPolarAng[i]) {
			printf("\n [**ERROR**]  The same number of polar angles should be uesed during the whole calculation");
			printf("\n              Please check and reset!");
			fflush(stdout);
			exit(69);
		}
	}
	_count = 0;
	for (i = 0; i < NumPolAngPart; i++) {
		if (PolarAngBoundary[i]<0 || PolarAngBoundary[i]>NumPolarAng[_count]) {
			printf("\n [**ERROR**]  Polar angle boundary should between [0, %d], but it's set to be %d in user input for trackSetID=%d", NumGroups, NumPolarAng[_count], _count);
			printf("\n              Please check and reset!");
			fflush(stdout);
			exit(69);
		}
		for (j = 0; j < NumGroupBound[i]; j++) {
			if (GroupBoundary[i][j + 1]<0 || GroupBoundary[i][j + 1]>NumGroups) {
				printf("\n [**ERROR**]  Group boundary should between [0, %d], but it's set to be %d in user input for trackSetID=%d", NumGroups, GroupBoundary[i][j + 1], _count);
				printf("\n              Please check and reset!");
				fflush(stdout);
				exit(69);
			}
			if (FSRVolumeCorrectionType[_count] < 0 || FSRVolumeCorrectionType[_count]>1) {
				printf("\n [**ERROR**]  FSR volume correction type should be 0-no_correction, 1-correction, but it's set to be %d in user input for trackSetID=%d", FSRVolumeCorrectionType[_count], _count);
				printf("\n              Please check and reset!");
			}
			_count++;
		}
		if (GroupBoundary[i][NumGroupBound[i]] != NumGroups) {
			printf("\n [**ERROR**]  The last group boundary should be the number of groups %d, but it's set to be %d in user input for trackSetID=%d", NumGroups, GroupBoundary[i][NumGroupBound[i]], _count);
			printf("\n              Please check and reset!");
			fflush(stdout);
			exit(69);
		}
	}
	// check solver
	if (NumPolAngPart > 1 && MOCSolverType == 0) {
		printf("\n [*WARNING*]  For the case with NumPolAngPart greater than one, MOCSolverType is automaticlly set to be 1-Jacobi Scheme");
		fflush(stdout);
		MOCSolverType = 1;
	}
	// 8 liner_exp IsCheckExpInteRange
	IdentifyKeyword(_ip, _keyWord, "EXP");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &LinearInter_Exp);
	IdentifyKeyword(_ip, _keyWord, "CheckExpInteRange");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &IsCheckExpInteRange);
	// K Reference
	IdentifyKeyword(_ip, _keyWord, "KType");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &KType);
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%*s");
	_inpErr = fscanf(_ip, FP_FORMAT_GEO, &KRef);


	InputCMFDInfo(_ip);

	NumSubDomains = AllocMem_i32_i32(3, (char *)("NumSubDomains"));
	for (i = 0; i < 3; i++) {
//		JumpCommentsInFile(_ip);
		_inpErr=fscanf(_ip, "%d", &NumSubDomains[i]);
	}
#ifndef MPI_USED
	for (i = 0; i < 3; i++) {
		if (NumSubDomains[i] != 1) {
			printf("\n [**ERROR**]  the number of subdomians in direction %d (0-x_direction, 1-y_direction, 2-z_direction) \n              should be 1, but it is %d accroding to the Parallel Card setting from input file. Please Check!",i, NumSubDomains[i]);
			fflush(stdout);
			exit(69);
		}
	}
#endif // !MPI_USED

	SubDomainsMap = AllocMem_int_III(NumSubDomains[2], NumSubDomains[1], NumSubDomains[0], (char *)("SubDomainsMap"));
	// Processes ID Map - 2D
	NumProcPerLayer = 0;
	for (i = 0; i < NumSubDomains[1]; i++) {
		for (j = 0; j < NumSubDomains[0]; j++) {
			//JumpCommentsInFile(_ip);
			//_inpErr=fscanf(_ip, "%d", &SubDomainsMap[0][i][j]);
			SubDomainsMap[0][i][j] = i * NumSubDomains[0] + j+1;
			if (SubDomainsMap[0][i][j] != 0) {
				NumProcPerLayer++;
			}
		}
	}
	// Processes ID Map - 3D
	for (k = 1; k < NumSubDomains[2]; k++) {
		for (i = 0; i < NumSubDomains[1]; i++) {
			for (j = 0; j < NumSubDomains[0]; j++) {
//				JumpCommentsInFile(_ip);
				SubDomainsMap[k][i][j] = SubDomainsMap[0][i][j] + k*NumProcPerLayer;
			}
		}
	}
	//hardware map
	HardwareMap_SubDom = AllocMem_i32_i32(NumSubDomains[1] * NumSubDomains[0], (char *)("HardwareMap_SubDom"));
	for (i = 0; i < NumSubDomains[1] * NumSubDomains[0]; i++) {
//		JumpCommentsInFile(_ip);
		_inpErr = fscanf(_ip, "%d", &HardwareMap_SubDom[i]);
	}
#ifdef MPI_USED
    if (NumProcs!=NumProcPerLayer*NumSubDomains[2]) {
		if (ProcID==0) {
			printf("\n [**ERROR**]  The required number of processes is %d based on the parallel setting, but it's %d specified by user!", NumProcPerLayer*NumSubDomains[2],NumProcs);
		    fflush(stdout);
		    exit(69);
		}
	}
#endif // MPI_USED
	// Number of cells in each process along x, y and z direction
	NumPin_subDomain = AllocMem_int_IP(3, NumSubDomains, (char *)("NumPin_subDomain"));
	NumCellsInCore_2D_X = 0;
	NumCellsInCore_2D_Y = 0;
	for (i = 0; i < 3; i++) {
		for (j = 0; j < NumSubDomains[i]; j++) {
//			JumpCommentsInFile(_ip);
			_inpErr=fscanf(_ip, "%d", &NumPin_subDomain[i][j]);
			if (i == 0) {
				NumCellsInCore_2D_X += NumPin_subDomain[i][j];
			}
			else if (i == 1) {
				NumCellsInCore_2D_Y += NumPin_subDomain[i][j];
			}
			else {}
		}
	}
	// OUTPUT_BLOCK
	IdentifyKeyword(_ip, _keyWord, "OUTPUT_BLOCK");
	IdentifyKeyword(_ip, _keyWord, "Output_inputInfo");
//	JumpCommentsInFile(_ip);
	_inpErr=fscanf(_ip, "%d", &OUTPUT_inputInfo);
	IdentifyKeyword(_ip, _keyWord, "Output_XSsLibInfo");
//	JumpCommentsInFile(_ip);
	_inpErr=fscanf(_ip, "%d", &OUTPUT_XSslibInfo);
	IdentifyKeyword(_ip, _keyWord, "Output_GeometryInfo");
//	JumpCommentsInFile(_ip);
	_inpErr=fscanf(_ip, "%d", &Output_GeometryInfo);
	IdentifyKeyword(_ip, _keyWord, "Output_MacroXSs");
//	JumpCommentsInFile(_ip);
	_inpErr=fscanf(_ip, "%d", &OUTPUT_MacorXSs);
	IdentifyKeyword(_ip, _keyWord, "Output_eigenvalue");
//	JumpCommentsInFile(_ip);
	_inpErr=fscanf(_ip, "%d", &OUTPUT_eigenvalue);//
	IdentifyKeyword(_ip, _keyWord, "Output_normPinPower");
//	JumpCommentsInFile(_ip);
	_inpErr=fscanf(_ip, "%d", &OUTPUT_normPinPower);
	IdentifyKeyword(_ip, _keyWord, "PinPowerSumNumSet");
//	JumpCommentsInFile(_ip);
	_inpErr=fscanf(_ip, "%d", &PinPowerSumNumSet);
	PinPowerSumLayBound = AllocMem_i32_i32(PinPowerSumNumSet+1, (char *)("PinPowerSumLayBound"));
	IdentifyKeyword(_ip, _keyWord, "PinPowerSumLayBound");
	for (i = 0; i < PinPowerSumNumSet+1; i++) {
//		JumpCommentsInFile(_ip);
		_inpErr=fscanf(_ip, "%d", &PinPowerSumLayBound[i]);
	}

	IdentifyKeyword(_ip, _keyWord, "Output_flux");
//	JumpCommentsInFile(_ip);
	_inpErr=fscanf(_ip, "%d", &OUTPUT_flux);
	IdentifyKeyword(_ip, _keyWord, "Output_flux_mat");
//	JumpCommentsInFile(_ip);
	_inpErr=fscanf(_ip, "%d", &OUTPUT_flux_mat);
	IdentifyKeyword(_ip, _keyWord, "Output_memoryFootprint");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &OUTPUT_MemoryFootprint);

	if (OUTPUT_MemoryFootprint == 0) {
		fclose(FP_MemoryFootprint);
		FP_MemoryFootprint = NULL;
		if (remove(_MemFPFileName) != 0) {
			printf("\n [ WARNING ]  Cannot remove file %s !", _MemFPFileName);
			fflush(stdout);
			exit(69);
		}
	}
	IdentifyKeyword(_ip, _keyWord, "OUTPUT_CMFDGroupConstant");
//	JumpCommentsInFile(_ip);
	_inpErr=fscanf(_ip, "%d", &OUTPUT_CMFDGroupConstant);
	IdentifyKeyword(_ip, _keyWord, "OUTPUT_CoarseMesh_GroupConstant");
//	JumpCommentsInFile(_ip);
	_inpErr=fscanf(_ip, "%d", &OUTPUT_CoarseMesh_GroupConstant);
	IdentifyKeyword(_ip, _keyWord, "Gene_VisualFile");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &Gene_VisualFile); 

	IdentifyKeyword(_ip, _keyWord, "NumOMPThread");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &NumOMPThread);

#if defined(OMP_USED)
    // omp_set_num_threads(NumOMPThread);
#else
    NumOMPThread=1;
#endif 

	IdentifyKeyword(_ip, _keyWord, "CPU_TSFunctionType");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &CPU_TSFunctionType);
	IdentifyKeyword(_ip, _keyWord, "CPU_FixSourSolver");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &CPU_FixSourSolver);
	if (CPU_TSFunctionType < 0 || CPU_TSFunctionType > 80) {
		printf("\n [*WARNING*]  CPU_TSFunctionType=%d From User Input", CPU_TSFunctionType);
		printf("\n              CPU_TSFunctionType Should Be 0~6, It Is Set To Be 3 (expTab_recExp) By Default");
		CPU_TSFunctionType = 3;
	}


// #if defined(OMP_USED)
// 	// if (NumOMPThread==1 && CPU_TSFunctionType==2) {
// 	// 	CPU_TSFunctionType = 0;
// 	// }
// 	// if (NumOMPThread>1) {
// 	// 	CPU_TSFunctionType = 2;
// 	// }
// #endif 
	/* GPU input info*/
	IdentifyKeyword(_ip, _keyWord, "GPUNumStreams");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &GPUNumStreamsInp);
	IdentifyKeyword(_ip, _keyWord, "GPUBlockDim");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &GPUBlockDimInp);
	IdentifyKeyword(_ip, _keyWord, "NumGPUsPerNode");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &NumGPUsPerNode);
	IdentifyKeyword(_ip, _keyWord, "GPU_TSKernelType");
//	JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &GPU_TSKernelType);
	IdentifyKeyword(_ip, _keyWord, "GPU_FixSourSolver");
//    JumpCommentsInFile(_ip);
	_inpErr = fscanf(_ip, "%d", &GPU_FixSourSolver);
    //GPU_FixSourSolver = 0;  // by zhangjinchao
	/* specify the cuda device ID*/
	CUDADeviceID = AllocMem_i32_i32(NumSubDomains[1] * NumSubDomains[0], (char *)("CUDADeviceID"));
	_count = 0;
	for (i = 0; i < NumSubDomains[1] * NumSubDomains[0]; i++) {
		CUDADeviceID[i] = -1;
		if (HardwareMap_SubDom[i] == 1) {
			CUDADeviceID[i] = _count % NumGPUsPerNode;;
			_count++;
		}
	}

    /* GPU_TSKernelType is configurable from the input card:
     *   -1 : auto select by scatter order and dimension (P0 -> 0,
     *        P>0 -> 13 for 2D / 12 for 3D)
     *  11~18 : use the input card value as-is for high-order source testing;
     *          2D additionally supports 19~21
     * GPU_TSKernelType 11 # baseline; 13/14 # LM/AP gather/scatter;
     * 15 # FSR-groupIn-azim-polar; 16 # FSR-groupOut-azim-polar;
     * 17 # FSR-groupIn;            18 # FSR-groupOut;
     * 19 # 2D fused FSR-groupOut with local S_lm reuse;
     * 20 # 2D fused FSR-groupIn with local moment reuse;
     * 21 # 2D fused warp-per-(FSR, groupOut) LM/AP;
     * 13, 14与 11 是一脉相承, 主要用于控制高阶散射源计算，输运扫描都走的同一块代码;
     * TODO: 主要13, 14主要用于测试千群高阶计算, 在少群高阶时与 11 相比, 等待测试;
     * 测试完毕, 在少群时, 13 和 14 的效果几乎一样, 但是均优于 11的效果, 因此, 因此这里在高阶计算时, 将GPU_TSKernelType设置为13
     */
    if (NumScattOrder == 0) {
//		CPU_TSFunctionType = 0;
//		GPU_TSKernelType = 0;
	}
	else {
        CPU_TSFunctionType = 13;
		int maxKernelType = 18;
		#if defined(PAPER_TRANSPORT_ONLY)
			/* The reproducibility package exposes only the reported paths. */
			maxKernelType = 16;
		#elif defined(M_2DMOC)
			maxKernelType = 21;
		#endif

		const char *envType = getenv("GPU_TSKernelTypeOverride");

		if (envType != NULL) {
			int type = atoi(envType);

			if (type < 13 || type > maxKernelType) {
				printf("\n[ERROR] GPU_TSKernelTypeOverride must be 13-%d.\n", maxKernelType);
				exit(69);
			}

			GPU_TSKernelType = type;
		}
		else if (NumScattOrder > 0 && (GPU_TSKernelType < 11 || GPU_TSKernelType > maxKernelType)) {
			#if defined(M_2DMOC)
				GPU_TSKernelType = 13;
			#else
				GPU_TSKernelType = 12;
			#endif
		}

		printf("\n Scatter Source Kernel Type %d", GPU_TSKernelType);
		fflush(stdout);
	}

    fclose(_ip);
	if (Gene_VisualFile==1) {
		GeoInfoInit_Visu(NumCells, Cells, NumPrimitives, Primitives);
	}
	MOC_PrintLogFile(_procID);

	printf("\n [  DEBUG  ]  read trk before \n");

	if (InputTrackFile == 1) {
        strcpy(TrackFileName, cleanComments(TrackFileName, "0_0", '!'));
		InputFromTrackFile(TrackFileName);
	}

	printf("\n [  DEBUG  ]  read trk after \n");

#ifdef TIME_REC
	_time_end = TallyCurrentTime_CPU();
    TimePrePart[TimeCountPrePart] = _time_end - _time_start;
	strcpy(TimeMetricsPrePart[TimeCountPrePart], "Read Input Info");
	TimeCountPrePart++;
#endif // TIME_REC
}

/**
* @ Function: InputCMFDInfo()
*   introduction: initial input of the whole calculation
* @ Parameter List: _fp - FILE pointer
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified: 2018/04/28, by SONG PEITAO
*/
void InputCMFDInfo(FILE *_fp)
{
	char _keyword[KEYWORD_N];
	M_INT32 g, _inpErr;
	//IdentifyKeyword(_fp, _keyword, "CMFD_Flag");
	//JumpCommentsInFile(_fp);
	//_inpErr=fscanf(_fp, "%d%d", &MGCMFD_Flag, &CGCMFD_Flag);
	//if (MGCMFD_Flag == 1 || MGCMFD_Flag == 0 || CGCMFD_Flag == 1 || CGCMFD_Flag == 0) {}
	//else {
	//	printf("\n [**ERROR**]  boolean value is not correct! \"CMFDSwitch\" should be 1 or 0, not %d", MGCMFD_Flag);
	//	fflush(stdout);
	//	exit(69);
	//}

	IdentifyKeyword(_fp, _keyword, "LinearSolver");
	JumpCommentsInFile(_fp);
	_inpErr = fscanf(_fp, "%d", &MGCMFD_LinearSolver);
	_inpErr = fscanf(_fp, "%d", &CGCMFD_LinearSolver);


	IdentifyKeyword(_fp, _keyword, "CMFD_ITER_LIM");
	JumpCommentsInFile(_fp);
	_inpErr = fscanf(_fp, "%d%d", &MaxNumOuterIter_MGCMFD, &MaxNumOuterIter_CGCMFD);
	IdentifyKeyword(_fp, _keyword, "LS_ITER_LIM");
	JumpCommentsInFile(_fp);
	_inpErr = fscanf(_fp, "%d%d", &MaxNumIter_MGLS, &MaxNumIter_CGLS);
	IdentifyKeyword(_fp, _keyword, "Damp_Factor");
	JumpCommentsInFile(_fp);
	_inpErr = fscanf(_fp, FP_FORMAT_D, &MGCMFD_DampFactor);
	_inpErr = fscanf(_fp, FP_FORMAT_D, &CGCMFD_DampFactor);
	IdentifyKeyword(_fp, _keyword, "CMFD_CONV_CRIT");
	IdentifyKeyword(_fp, _keyword, "epsEig");
	JumpCommentsInFile(_fp);
	_inpErr = fscanf(_fp, FP_FORMAT, &MGCMFD_epseig);
	_inpErr = fscanf(_fp, FP_FORMAT, &CGCMFD_epseig);

	IdentifyKeyword(_fp, _keyword, "epsFlux");
	JumpCommentsInFile(_fp);
	_inpErr = fscanf(_fp, FP_FORMAT, &MGCMFD_epsl2);
	_inpErr = fscanf(_fp, FP_FORMAT, &CGCMFD_epsl2);

	IdentifyKeyword(_fp, _keyword, "fluxErrLInf");
	JumpCommentsInFile(_fp);
	_inpErr = fscanf(_fp, "%d", &MGCMFD_checkFluxNormInf);
	_inpErr = fscanf(_fp, "%d", &CGCMFD_checkFluxNormInf);

	IdentifyKeyword(_fp, _keyword, "fluxErrL2");
	JumpCommentsInFile(_fp);
	_inpErr = fscanf(_fp, "%d", &MGCMFD_checkFluxNorm2);
	_inpErr = fscanf(_fp, "%d", &CGCMFD_checkFluxNorm2);

	IdentifyKeyword(_fp, _keyword, "fluxErrRMSE");
	JumpCommentsInFile(_fp);
	_inpErr = fscanf(_fp, "%d", &MGCMFD_checkFluxRMSE);
	_inpErr = fscanf(_fp, "%d", &CGCMFD_checkFluxRMSE);


	if (MGCMFD_checkFluxNormInf==0 && MGCMFD_checkFluxNorm2==0 && MGCMFD_checkFluxRMSE==0) {
		MGCMFD_checkFluxRMSE = 1;
	}

	if (CGCMFD_checkFluxNormInf==0 && CGCMFD_checkFluxNorm2==0 && CGCMFD_checkFluxRMSE==0) {
		CGCMFD_checkFluxRMSE = 1;
	}

	if (MGCMFD_LinearSolver==2) {
		MGCMFD_checkFluxNormInf=0;
		MGCMFD_checkFluxNorm2=1;
		MGCMFD_checkFluxRMSE=0;
	}

	if (CGCMFD_LinearSolver==2) {
		CGCMFD_checkFluxNormInf=0;
		CGCMFD_checkFluxNorm2=1;
		CGCMFD_checkFluxRMSE=0;
	}
	// IdentifyKeyword(_fp, _keyword, "epslinf");
	// JumpCommentsInFile(_fp);
	// _inpErr = fscanf(_fp, FP_FORMAT, &MGCMFD_epslinf);
	// _inpErr = fscanf(_fp, FP_FORMAT, &CGCMFD_epslinf);
	IdentifyKeyword(_fp, _keyword, "LS_CONV_CRIT");
	JumpCommentsInFile(_fp);
	_inpErr = fscanf(_fp, FP_FORMAT, &ConvCritLinearSolver_MGCMFD);
	_inpErr = fscanf(_fp, FP_FORMAT, &ConvCritLinearSolver_CGCMFD);
	//JumpCommentsInFile(_ip);
	//   _inpErr=fscanf(_ip,FP_FORMAT_GEO,&ConvCritK_MGCMFD);
	   //_inpErr=fscanf(_ip,FP_FORMAT_GEO,&ConvCritFlux_MGCMFD);
	   //_inpErr=fscanf(_ip,FP_FORMAT_GEO,&ConvCritK_CGCMFD);
	   //_inpErr=fscanf(_ip,FP_FORMAT_GEO,&ConvCritFlux_CGCMFD);

	IdentifyKeyword(_fp, _keyword, "SOR_FACTOR");
	JumpCommentsInFile(_fp);
	_inpErr = fscanf(_fp, FP_FORMAT, &SOR_Factor_MG);
	_inpErr = fscanf(_fp, FP_FORMAT, &SOR_Factor_CG);
	/*GMRES parameters*/
	IdentifyKeyword(_fp, _keyword, "KrylovDim");
	JumpCommentsInFile(_fp);
	_inpErr = fscanf(_fp, "%d%d", &GmresKrylovDim_MG, &GmresKrylovDim_CG);
	IdentifyKeyword(_fp, _keyword, "Precond");
	JumpCommentsInFile(_fp);
	_inpErr = fscanf(_fp, "%d%d", &GmresPrecond_MG, &GmresPrecond_CG);


	IdentifyKeyword(_fp, _keyword, "CMFD_NumGroups");
	JumpCommentsInFile(_fp);
	_inpErr=fscanf(_fp, "%d", &NumCMFDGroup);

    CMFDGroupBound = AllocMem_i32_i32(NumCMFDGroup + 1, (char *)("CMFDGroupBound"));

	IdentifyKeyword(_fp, _keyword, "CMFD_GroupBound");
	for (g = 0; g < NumCMFDGroup + 1; g++) {
		JumpCommentsInFile(_fp);
#if defined(M_MG_CMFD)
		//if (CMFDSwitch == 1) {
		_inpErr = fscanf(_fp, "%d", &CMFDGroupBound[g]);
		if (CMFDGroupBound[g] >= 0 && CMFDGroupBound[g] < NumGroups+1) {}
		else {
			printf("\n [**ERROR**]  the boundary of MG-M_MG_CMFD group is not correct, \"CMFDGroupBound\"[%d] \
should be the value between 0~%d, but it's %d", g, NumGroups, CMFDGroupBound[g]);
			fflush(stdout);
			exit(69);
		}
		if (g == NumCMFDGroup) {
			if (CMFDGroupBound[g] == NumGroups) {}
			else {
				printf("\n [**ERROR**]  the last value of \"CMFDGroupBound\" should be %d, but it's %d", NumGroups, CMFDGroupBound[g]);
				fflush(stdout);
				exit(69);
			}
		}
		else {}
		//}
#else  // !defined(M_MG_CMFD)
		//else if (CMFDSwitch == 0) {
		_inpErr = fscanf(_fp, "%*s");
		//}
		//else {}
#endif // M_MG_CMFD
	}

	IdentifyKeyword(_fp, _keyword, "CMFD_NumCoarseGroups");
	JumpCommentsInFile(_fp);
	_inpErr=fscanf(_fp, "%d", &NumCoarseCMFDGroup);

    CGCMFDGroupBound = AllocMem_i32_i32(NumCoarseCMFDGroup + 1, (char *)("CGCMFDGroupBound"));

	IdentifyKeyword(_fp, _keyword, "CMFD_CoarseGroupBound");
	for (g = 0; g < NumCoarseCMFDGroup + 1; g++) {
		JumpCommentsInFile(_fp);
#if defined(M_MG_CMFD)
		//if (CMFDSwitch == 1) {
		_inpErr = fscanf(_fp, "%d", &CGCMFDGroupBound[g]);
		if (CGCMFDGroupBound[g] >= 0 && CGCMFDGroupBound[g] < NumCMFDGroup + 1) {}
		else {
			printf("\n [**ERROR**]  the boundary of CG-M_MG_CMFD group is not correct, \"CMFDGroupBound\"[%d] \
should be the value between 0~%d, but it's %d", g, NumGroups, CMFDGroupBound[g]);
			fflush(stdout);
			exit(69);
		}
		if (g == NumCoarseCMFDGroup) {
			if (CGCMFDGroupBound[g] == NumCMFDGroup) {}
			else {
				printf("\n [**ERROR**]  the last value of \"CGCMFDGroupBound\" should be %d, \
but it's %d", NumCMFDGroup, CGCMFDGroupBound[g]);
				fflush(stdout);
				exit(69);
			}
		}
		else {}
		//}
#else  // !defined(M_MG_CMFD)
		_inpErr = fscanf(_fp, "%*s");
#endif // M_MG_CMFD
	}
}

/**
* @ Function: MOC_PrintLogFile()
*   introduction: fprintf the screen output to the log file
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2019/09/24, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void InputFromTrackFile(char *_fileName)
{
    M_INT32 _lay, _inpErr,_MOCNumPinsInWhoDom_X, _MOCNumPinsInWhoDom_Y, _NumPinsInWhoDomPerLayer=0;//, i, j;
    M_INT32 _numPinCell;
    M_INT32 *_pinCellRTMID=NULL, *_numFSRInPinCell=NULL, **_fsrMatIDInPinCell=NULL;  // PART 0 PIN_INFO
    M_INT32 *_pinCellType_WhoDom=NULL, *_pinCellType_SubDom=NULL;
    M_INT32 _rtmCount = 0, _fsrCount = 0, _trkCount = 0, _segCount = 0, _totFSRCount = 0, _cellType;
    char _keyWord[KEYWORD_N];
    FILE *_fp=NULL;
    FP_PRECISION _one=(FP_PRECISION)(1.0);
    M_INT32 i,j,k,_rtm_id,_fsr_id,_azim_id,_trk_id,_seg_id,_pin_cnt,_fsr_cnt;

    // next azimuthal angle and next ray id for each ray
    M_INT32 *_nextAzimID;
    M_INT32 *_nextTrackID;
//    M_INT32 _nextAzimID[72], _nextTrackID[72];

    int _numOfPinMaps;
    float _currAxialRegWidth;
    int _meshZ, _currLayInd, _currPinID;
    int _fpPosRec;
    int _geomType;  // hex or rect
    /* Geometry Info Set */
//    MOCNumLayersZ = 1;
    /* set the width per layer */
//    WidthLayersInCore = AllocMem_FPGeo_I(MOCNumLayersZ, "WidthLayersInCore");
//    for (_lay = 0; _lay < MOCNumLayersZ; _lay++)
//        WidthLayersInCore[_lay] = _one;

    /* reading from track.txt*/
    _fp = fopen(_fileName, "r");
    if (_fp == NULL) {
        printf("\n [**ERROR**]  Input file opens error with name = \"%s\"\n", _fileName);
        fflush(stdout);
        exit(69);
    }


//    RTMWidth_X = AllocMem_FP_I_Geo(1, (char *)("RTMWidth_X"));
//    RTMWidth_Y = AllocMem_FP_I_Geo(1, (char *)("RTMWidth_Y"));
//    RTMArea = AllocMem_FP_I_Geo(1, (char *)("RTMArea"));
//
//        RTMWidth_X[0] = 1.26;
//        RTMWidth_Y[0] = 1.26;
//        RTMArea[0] = RTMWidth_X[0] * RTMWidth_Y[0];


    IdentifyKeyword(_fp, _keyWord, "GeomType");
    _inpErr = fscanf(_fp, "%d", &_geomType);
//    if (_geomType == 1){
//        // todo: add input from hex track file
//        InputFromTrackFile_hex()
//        return;
//    }

    /* reading pin mat info*/
    IdentifyKeyword(_fp, _keyWord, "PART_0_PIN_INFO");
    IdentifyKeyword(_fp, _keyWord, "NumOfPins");
    JumpCommentsInFile(_fp);
    _inpErr = fscanf(_fp, "%d", &_numPinCell);
    _pinCellRTMID = AllocMem_i32_i32(_numPinCell, "_pinCellRTMID");
    _numFSRInPinCell = AllocMem_i32_i32(_numPinCell, "_numFSRInPinCell");
    _fsrMatIDInPinCell = (M_INT32 **)calloc(_numPinCell, sizeof(M_INT32*));
    JumpCommentsInFile(_fp);
    for(i = 0; i < _numPinCell; i++){
        _inpErr = fscanf(_fp, "%*s%d", &_pinCellRTMID[i]);
        _inpErr = fscanf(_fp, "%d", &_numFSRInPinCell[i]);
        _fsrMatIDInPinCell[i] = (M_INT32*)calloc(_numFSRInPinCell[i], sizeof(M_INT32));
        for(j = 0; j < _numFSRInPinCell[i]; j++){
            _inpErr = fscanf(_fp, "%d%*d", &_fsrMatIDInPinCell[i][j]);
            JumpCommentsInFile(_fp);
        }
    }

//    for(M_INT32 i = 0; i < _numPinCell; i++) {
//        for (M_INT32 j = 0; j < _numFSRInPinCell[i]; j++) {
//            printf("\nNumFSRInPinCell_Trelis=%d", _numFSRInPinCell[i]);
//            printf("\tPin_id=%d      FSR_id=%d     Mat_id=%d", i, j, _fsrMatIDInPinCell[i][j]);
//        }
//    }

//    exit(90);

    // reading pin layout map
    IdentifyKeyword(_fp, _keyWord, "PART_1_PIN_LAYOUT_INFO");
    JumpCommentsInFile(_fp);
    IdentifyKeyword(_fp, _keyWord, "NumOfPinMaps");
    _inpErr = fscanf(_fp, "%d", &_numOfPinMaps);
//    printf("\n_numOfPinMaps = %d", _numOfPinMaps);
//    exit(99);
    IdentifyKeyword(_fp, _keyWord, "NumOfSublayers");
    _inpErr = fscanf(_fp, "%d", &MOCNumLayersZ);
    JumpCommentsInFile(_fp);
    WidthLayersInCore = AllocMem_FPGeo_I(MOCNumLayersZ, "WidthLayersInCore");
    _currLayInd = 0;

    GetSubDomLocIDBasedProcID(ProcID, NumSubDomains, SubDomainsMap, &SubDomLocID_2D_Y, &SubDomLocID_2D_X,
                              &MOC_subDomID_layer);
    GetNumCellsInSubDom(SubDomLocID_2D_Y, SubDomLocID_2D_X, MOC_subDomID_layer, NumPin_subDomain, \
        &MOCNumPinsInSubDom_X, &MOCNumPinsInSubDom_Y, &MOCNumLayersZ);
    GetGlobPinCellOffset_SubDom(SubDomLocID_2D_Y, SubDomLocID_2D_X, NumPin_subDomain, \
        &SubDomOrgPinOffset_2D_X, &SubDomOrgPinOffset_2D_Y);

    for(k = 0; k < _numOfPinMaps; k++){
        if(k == 0){
            _inpErr = fscanf(_fp, "%*s%d%d", &_MOCNumPinsInWhoDom_X, &_MOCNumPinsInWhoDom_Y);
            _NumPinsInWhoDomPerLayer = _MOCNumPinsInWhoDom_X * _MOCNumPinsInWhoDom_Y;
            JumpCommentsInFile(_fp);
            NumPinsInSubDomPerLayer = MOCNumPinsInSubDom_X * MOCNumPinsInSubDom_Y;
            MOC_NumPinsInSubDom = MOCNumLayersZ * NumPinsInSubDomPerLayer;
            MOC_NumPinsInWholeDom = MOCNumLayersZ * _NumPinsInWhoDomPerLayer;
            _pinCellType_WhoDom = AllocMem_i32_i32(MOC_NumPinsInWholeDom, "_pinCellType_WhoDom");
            _pinCellType_SubDom = AllocMem_i32_i32(MOC_NumPinsInSubDom, "_pinCellType_WhoDom");
            CellRTMType_SubDom = AllocMem_i32_i32(MOC_NumPinsInSubDom, "CellRTMType_SubDom");
        }
        else{
            _inpErr = fscanf(_fp, "%*s%*s%*s");
        }
        _inpErr = fscanf(_fp, "%f%d", &_currAxialRegWidth, &_meshZ);
        JumpCommentsInFile(_fp);
        _fpPosRec = ftell(_fp);
        for(_lay = 0; _lay < _meshZ ; _lay++){
            fseek(_fp, _fpPosRec, SEEK_SET);
            for(i=0; i < _NumPinsInWhoDomPerLayer; i++){
                _currPinID = _NumPinsInWhoDomPerLayer * _currLayInd + i;
                _inpErr = fscanf(_fp, "%d", &_pinCellType_WhoDom[_currPinID]);
                JumpCommentsInFile(_fp);
                // CellRTMType_SubDom[_currPinID] = _pinCellRTMID[_pinCellType_SubDom[_currPinID]];
            }
            WidthLayersInCore[_currLayInd] = _currAxialRegWidth / _meshZ;
            _currLayInd++;
        }
    }
    /* SDD*/
    int _subDomSeqID=0, _pinSeqIDInSubDom=0;
    for(_lay = 0; _lay < MOCNumLayersZ ; _lay++) {
        for (i=0;i<_NumPinsInWhoDomPerLayer;i++) {
            _subDomSeqID = GetSubDomIDBaseGlobPinID(i, NumSubDomains, NumPin_subDomain);
            _pinSeqIDInSubDom = GetPinSeqIDInSubDomBaseGlobPinID(i, NumSubDomains, NumPin_subDomain);
            if (ProcID == _subDomSeqID) {
                _pinCellType_SubDom[_lay * NumPinsInSubDomPerLayer + _pinSeqIDInSubDom] = _pinCellType_WhoDom[_lay * _NumPinsInWhoDomPerLayer + i];
                CellRTMType_SubDom[_lay * NumPinsInSubDomPerLayer + _pinSeqIDInSubDom] = _pinCellRTMID[_pinCellType_WhoDom[_lay * _NumPinsInWhoDomPerLayer + i]];
                // if (ProcID == 2) {
                // 	printf("\n *****Proc=%d, _pinCellType_SubDom[%d]=%d, CellRTMType_SubDom=%d",\
			// 	ProcID, _lay * NumPinsInSubDomPerLayer + _pinSeqIDInSubDom, \
			// 	_pinCellType_SubDom[_lay * NumPinsInSubDomPerLayer + _pinSeqIDInSubDom],\
			// 	CellRTMType_SubDom[_lay * NumPinsInSubDomPerLayer + _pinSeqIDInSubDom]);
                // }
            }
            // if (ProcID == 0) {

            // 	// printf("\n i=%d, _subDomSeqID=%d, _pinSeqIDInSubDom=%d, id=%d",i,_subDomSeqID,_pinSeqIDInSubDom,_lay * NumPinsInSubDomPerLayer + _pinSeqIDInSubDom);
            // 	// printf("\n _pinCellType_SubDom=%d",_pinCellType_SubDom[_lay * NumPinsInSubDomPerLayer + _pinSeqIDInSubDom]);
            // 	// printf("\n CellRTMType_SubDom=%d",CellRTMType_SubDom[_lay * NumPinsInSubDomPerLayer + _pinSeqIDInSubDom]);
            // }
        }
    }

//    printf("\n MOCNumLayersZ=%d, MOCNumPinsInSubDom_X=%d, MOCNumPinsInSubDom_Y=%d", MOCNumLayersZ, MOCNumPinsInSubDom_X, MOCNumPinsInSubDom_Y);
//    printf("\n MOC_NumPinsInSubDom=%d, NumPinsInSubDomPerLayer=%d", MOC_NumPinsInSubDom, NumPinsInSubDomPerLayer);
//    printf("\t CellRTMType_SubDom[i=%d]=%d", i, CellRTMType_SubDom[i]);

    // reading ray parameter info
    IdentifyKeyword(_fp, _keyWord, "PART_2_RAY_PARAM_INFO");
    IdentifyKeyword(_fp, _keyWord, "NumOfAzimAng");
    _inpErr = fscanf(_fp, "%d", &NumAzimAng[0]);
    JumpCommentsInFile(_fp);
    NumTrackSet = 1;
    NumTracks = AllocMem_int_IP(NumTrackSet, NumAzimAng, "NumTracks");
    NumTracksX = AllocMem_int_IP(NumTrackSet, NumAzimAng, "NumTracksX");
    NumTracksY = AllocMem_int_IP(NumTrackSet, NumAzimAng, "NumTracksY");
    EffAzimAng = AllocMem_FPGeo_IP(NumTrackSet, NumAzimAng, "EffAzimAng");
    EffRaySpacing = AllocMem_FPGeo_IP(NumTrackSet, NumAzimAng, "EffRaySpacing");
    for (i = 0; i < NumTrackSet; i++) {
        for (j = 0; j < NumAzimAng[i]; j++) {
            _inpErr = fscanf(_fp, "%*s%d%d", &NumTracksX[i][j], &NumTracksY[i][j]);
            _inpErr = fscanf(_fp, FP_FORMAT_GEO, &EffAzimAng[i][j]);
            _inpErr = fscanf(_fp, FP_FORMAT_GEO, &EffRaySpacing[i][j]);
            NumTracks[i][j] = NumTracksX[i][j] + NumTracksY[i][j];
            if (_geomType == 1) NumTracks[i][j] *= 2;
//            printf("\n i=%d, j=%d, NumTracksX=%d, NumTracksY=%d, NumTracks=%d", i, j, NumTracksX[i][j], NumTracksY[i][j], NumTracks[i][j]);
//            printf("\n             EffAzimAng=%e, EffRaySpacing=%e", EffAzimAng[i][j], EffRaySpacing[i][j]);
        }
    }


    // if pin cell type is rectangle:                  if pin cell type is Hex:
    //
    //  ----------                                         NumTrkX
    //  |        |                                       /-------\
    //  |        | NumTrkY                              /         \ NumTrkX + NumTrkY
    //  |        |                                      \         /
    //  ----------                                       \_______/ NumTrkY
    //    NumTrkX

    // reading rtm geo info
    IdentifyKeyword(_fp, _keyWord, "PART_3_RTM_GEOM_INFO");

    IdentifyKeyword(_fp, _keyWord, "TotalNumOfRTM");
    _inpErr = fscanf(_fp, "%d", &NumRTMGeoType);

    IdentifyKeyword(_fp, _keyWord, "TotalNumOfFSR");
    _inpErr = fscanf(_fp, "%d", &TotalNumFSRsInRTM);

    TotalNumTracksInRTM_Total = AllocMem_i32_i32(NumTrackSet, "TotalNumTracksInRTM_Total");
    TotNumTracksInOneRTM_2PI = AllocMem_i32_i32(NumTrackSet, "TotNumTracksInOneRTM_2PI");
    IdentifyKeyword(_fp, _keyWord, "TotalNumOfTrk");
    _inpErr = fscanf(_fp, "%d", &TotalNumTracksInRTM_Total[0]);
    TotNumTracksInOneRTM_2PI[0] = TotalNumTracksInRTM_Total[0] / NumRTMGeoType;

    _nextAzimID = AllocMem_i32_i32(TotNumTracksInOneRTM_2PI[0], "_nextAzimID");
    _nextTrackID = AllocMem_i32_i32(TotNumTracksInOneRTM_2PI[0], "_nextTrackID");

    //printf("\n TotalNumTracksInRTM_Total[0]=%d", TotalNumTracksInRTM_Total[0]);
    //printf("\n TotNumTracksInOneRTM_2PI[0]=%d", TotNumTracksInOneRTM_2PI[0]);

    TotNumSegInRTM_Total = AllocMem_i32_i32(NumTrackSet, "TotNumSegInRTM_Total");
    IdentifyKeyword(_fp, _keyWord, "TotalNumOfSeg");
    _inpErr = fscanf(_fp, "%d", &TotNumSegInRTM_Total[0]);
    //printf("\n TotNumSegInRTM_Total[0]=%d", TotNumSegInRTM_Total[0]);

    FP_PRECISION_GEO _rtmWidthX=0., _rtmWidthY=0.;
    IdentifyKeyword(_fp, _keyWord, "RTMWidthX");
    _inpErr = fscanf(_fp, "%lf", &_rtmWidthX);
    IdentifyKeyword(_fp, _keyWord, "RTMWidthY");
    _inpErr = fscanf(_fp, "%lf", &_rtmWidthY);

    RayTracModDim = AllocMem_FPGeo_I(3, (char *)("RayTracModDim"));
    RayTracModDim[0] = _rtmWidthX;
    RayTracModDim[1] = _rtmWidthX;
    RayTracModDim[2] = 1.0;

    RTMWidth_X = AllocMem_FP_I_Geo(NumRTMGeoType, (char *)("RTMWidth_X"));
    RTMWidth_Y = AllocMem_FP_I_Geo(NumRTMGeoType, (char *)("RTMWidth_Y"));
    RTMArea = AllocMem_FP_I_Geo(NumRTMGeoType, (char *)("RTMArea"));
    for (i = 0; i < NumRTMGeoType; i++) {
        RTMWidth_X[i] = _rtmWidthX;
        RTMWidth_Y[i] = _rtmWidthY;
        RTMArea[i] = RTMWidth_X[i] * RTMWidth_Y[i];
    }

    IndexTrackInRTM_SumAzim_2PI = AllocMem_int_IP(NumTrackSet, NumAzimAng, "IndexTrackInRTM_SumAzim_2PI");
    GetIndexOfTracksInRTM_SumAzim(NumTrackSet, NumAzimAng, NumTracks, IndexTrackInRTM_SumAzim_2PI);
    //for (M_INT32 i = 0; i < NumAzimAng[0]; i++)
    //	printf("\n IndexTrackInRTM_SumAzim_2PI[%d]=%d", i, IndexTrackInRTM_SumAzim_2PI[0][i]);

    NumFSRsInRTM = AllocMem_i32_i32(NumRTMGeoType, "NumFSRsInRTM");
    FSRsAreaInRTM_AntiClockFSR = AllocMem_FP_I_Geo(TotalNumFSRsInRTM, "FSRsAreaInRTM_AntiClockFSR");
    OneOverFSRsAreaInRTM_AntiClockFSR = AllocMem_FP_I_Geo(TotalNumFSRsInRTM, "OneOverFSRsAreaInRTM_AntiClockFSR");
    FSRsMatIDinRTM = AllocMem_i32_i32(TotalNumFSRsInRTM, "FSRsMatIDinRTM");

    NumSegsInTrack_2PI = AllocMem_int_IP(NumTrackSet, TotalNumTracksInRTM_Total, "NumSegsInTrack_2PI");
    SegInfoInTrack_2PI_length = AllocMem_FP_IP(NumTrackSet, TotNumSegInRTM_Total, "SegInfoInTrack_2PI_length");
    SegInfoInTrack_2PI_AntiClockFSR = AllocMem_int_IP(NumTrackSet, TotNumSegInRTM_Total, "SegInfoInTrack_2PI_AntiClockFSR");

    for(_rtm_id = 0; _rtm_id < NumRTMGeoType; _rtm_id++){
        //M_INT32 rtm_count = 0, fsr_count = 0;
        IdentifyKeyword(_fp, _keyWord, "RTM_ID");
        _inpErr = fscanf(_fp, "%d", &_rtmCount);

        IdentifyKeyword(_fp, _keyWord, "FSRAreaInfo");
        IdentifyKeyword(_fp, _keyWord, "NumOfFSR");
        _inpErr = fscanf(_fp, "%d", &NumFSRsInRTM[_rtmCount]);
        JumpCommentsInFile(_fp);

        for(_fsr_id = 0; _fsr_id < NumFSRsInRTM[_rtmCount]; _fsr_id++){
            _inpErr = fscanf(_fp, "%*s%lf", &FSRsAreaInRTM_AntiClockFSR[_fsrCount]);
            OneOverFSRsAreaInRTM_AntiClockFSR[_fsrCount] = 1 / FSRsAreaInRTM_AntiClockFSR[_fsrCount];
            _fsrCount++;
        }

        IdentifyKeyword(_fp, _keyWord, "TrackInfo");
        JumpCommentsInFile(_fp);
        for(_azim_id = 0; _azim_id < NumAzimAng[0]; _azim_id++){
            _inpErr = fscanf(_fp, "%*s%*s");
            //printf("\nnum_trk=%d NumTracks[0][%d]=%d", num_trk, _azim_id, NumTracks[0][_azim_id]);

            for(_trk_id = 0; _trk_id < NumTracks[0][_azim_id]; _trk_id++){
                if(_geomType == 0){
                    _inpErr = fscanf(_fp, "%*s%d", &NumSegsInTrack_2PI[0][_trkCount]);
//                printf("\nNumSegsInTrack_2PI[0][%d]=%d", _trkCount, NumSegsInTrack_2PI[0][_trkCount]);
                } else {
                    _inpErr = fscanf(_fp, "%*s%d", &_nextAzimID[_trkCount]);
                    _inpErr = fscanf(_fp, "%d", &_nextTrackID[_trkCount]);
                    _inpErr = fscanf(_fp, "%d", &NumSegsInTrack_2PI[0][_trkCount]);
//                printf("\n _globalTrackID=%d\tnextAzimID=%d\tnextTrackID=%d", _trkCount, _nextAzimID[_trkCount], _nextTrackID[_trkCount]);
                }


                for(_seg_id = 0; _seg_id < NumSegsInTrack_2PI[0][_trkCount]; _seg_id++){
                    _inpErr = fscanf(_fp, "%*s");
                    _inpErr = fscanf(_fp, FP_FORMAT, &SegInfoInTrack_2PI_length[0][_segCount]);
                    _inpErr = fscanf(_fp, "%d", &SegInfoInTrack_2PI_AntiClockFSR[0][_segCount]);
                    JumpCommentsInFile(_fp);
                    _segCount++;
                }
//                printf("\nrtm_id = %d azim_id = %d trk_id = %d trk_num = %d", _rtm_id,_azim_id,_trk_id, NumTracks[0][_azim_id]);
                _trkCount++;
                JumpCommentsInFile(_fp);
            }
        }
    }

    fclose(_fp);
    _fp = NULL;

    SumNumSegsInRTMTrack_2PI = (int **) calloc(NumTrackSet, sizeof (int *));
    SumNumSegsInRTMTrack_2PI[0] = (int *) calloc(TotalNumTracksInRTM_Total[0], sizeof(int));
//    SumNumSegsInRTMTrack_2PI = AllocMem_int_IP(NumTrackSet, TotalNumTracksInRTM_Total, "SumNumSegsInRTMTrack_2PI");
    GetSumNumSegsInRTMTrack_2PI(NumTrackSet, TotalNumTracksInRTM_Total, \
        NumSegsInTrack_2PI, SumNumSegsInRTMTrack_2PI);

//    for (M_INT32 i = 0; i < TotalNumTracksInRTM_Total[0]; i++)
//    	printf("\n NumSegsInTrack_2PI[%d]=%d, SumNumSegsInRTMTrack_2PI[%d]=%d", i, NumSegsInTrack_2PI[0][i], i, SumNumSegsInRTMTrack_2PI[0][i]);
//
//    for (M_INT32 i = 0; i < TotNumSegInRTM_Total[0]; i++)
//        printf("\n SegInfoInTrack_, 2PI_length[%d]=%.16f, SegInfoInTrack_2PI_AntiClockFSR[%d]=%d", i, SegInfoInTrack_2PI_length[0][i], i, SegInfoInTrack_2PI_AntiClockFSR[0][i]);
//     exit(90);

    NumLongTracks_SubDomX = AllocMem_int_IP(NumTrackSet, NumAzimAng, "NumLongTracks_SubDomX");
    NumLongTracks_SubDomY = AllocMem_int_IP(NumTrackSet, NumAzimAng, "NumLongTracks_SubDomY");
    NumLongTracks = AllocMem_int_IP(NumTrackSet, NumAzimAng, "NumLongTracks");


    for (i = 0; i < NumTrackSet; i++) {
        for (j = 0; j < NumAzimAng[i]; j++) {
            NumLongTracks_SubDomX[i][j] = NumTracksX[i][j] * MOCNumPinsInSubDom_X;
            NumLongTracks_SubDomY[i][j] = NumTracksY[i][j] * MOCNumPinsInSubDom_Y;
            NumLongTracks[i][j] = NumLongTracks_SubDomX[i][j] + NumLongTracks_SubDomY[i][j];
            if (_geomType == 1) NumLongTracks[i][j] *= 2;
//            printf("\n i=%d, j=%d, NumLongTracks_SubDomX=%d, NumLongTracks_SubDomY=%d, NumLongTracks=%d", i, j, NumLongTracks_SubDomX[i][j], NumLongTracks_SubDomY[i][j], NumLongTracks[i][j]);
        }
    }

    /* FSRs' area Info */
    SumNumFSRsInRTM = AllocMem_i32_i32(NumRTMGeoType, "SumNumFSRsInRTM");
    GetSumNumFSRsInRTM(SumNumFSRsInRTM, NumRTMGeoType, NumFSRsInRTM);
    TotalNumFSRs_subDom = GetTotalNumFSRs_SubDom_TrackFile(NumFSRsInRTM, MOC_NumPinsInSubDom, CellRTMType_SubDom);

    TotalNumFSRs_perLayer = TotalNumFSRs_subDom / MOCNumLayersZ;

// printf("\n ProcID=%d, MOC_NumPinsInSubDom=%d, TotalNumFSRs_subDom=%d, NumFSRsInRTM[0]=%d, NumFSRsInRTM[1]=%d, NumFSRsInRTM[2]=%d",\
// ProcID , MOC_NumPinsInSubDom, TotalNumFSRs_subDom,NumFSRsInRTM[0], NumFSRsInRTM[1], NumFSRsInRTM[2]);


    NumFSRs_Layer = AllocMem_i32_i32(MOCNumLayersZ, (char *)("NumFSRs_Layer"));
    SumNumFSRs_Layer = AllocMem_i32_i32(MOCNumLayersZ, (char *)("SumNumFSRs_Layer"));
    for(_lay = 0; _lay < MOCNumLayersZ ; _lay++ ){
        NumFSRs_Layer[_lay] = TotalNumFSRs_subDom / MOCNumLayersZ;
        if(_lay == 0) {
            SumNumFSRs_Layer[_lay] = 0;
        }
        else {
            SumNumFSRs_Layer[_lay] = SumNumFSRs_Layer[_lay - 1] + NumFSRs_Layer[_lay - 1];
        }
    }

//    printf("TotalNumFSRs_subDom = %d", TotalNumFSRs_subDom);

    //FSRsMatIDinRTM = AllocMem_i32_i32(TotalNumFSRsInRTM, "FSRsMatIDinRTM");

    NumFSRsInCell = AllocMem_i32_i32(MOC_NumPinsInSubDom, "NumFSRsInCell");
    SetNumFSRsInCell_trackFile(NumFSRsInCell, MOC_NumPinsInSubDom, NumFSRsInRTM, CellRTMType_SubDom);

    NumLTs_HS = GetNumLTsInHalfspace(NumTrackSet,NumAzimAng, NumLongTracks,"NumLTs_HS");
    LongTracksInfo_Total = AllocMem_LongTracksInfo(NumTrackSet, NumLTs_HS, "LongTracksInfo_Total");
    NumTracks_Total = AllocMem_i32_i32(NumTrackSet, "NumTracks_Total");
    NumTracks_Total[0] = TotNumTracksInOneRTM_2PI[0] * MOC_NumPinsInSubDom;

    TracksInfoInLTs_2PI = AllocMem_TracksInfoInLTs(NumTrackSet, NumTracks_Total, "TracksInfoInLTs_2PI");

    NumLTsSum=AllocMem_int_IP(NumTrackSet, NumAzimAng, "NumLTsSum");
    for (i = 0; i < NumTrackSet; i++)
        for (j = 0; j < NumAzimAng[i]; j++)
            if (j == 0)
                NumLTsSum[i][j] = 0;
            else
                NumLTsSum[i][j] = NumLTsSum[i][j-1] + NumLongTracks[i][j-1];

    // SubDomLocID_2D_X = 0;
    // SubDomLocID_2D_Y = 0;
    // GetGlobPinCellOffset_SubDom(SubDomLocID_2D_Y, SubDomLocID_2D_X, NumPin_subDomain, \
	// 	&SubDomOrgPinOffset_2D_X, &SubDomOrgPinOffset_2D_Y);


    NumTracks_HS = AllocMem_i32_i32(NumTrackSet, "NumTracks_HS");
    NumTracks_HS[0] = NumTracks_Total[0] / 2;
    TracksInfoInLTs_HS = AllocMem_TracksInfoInLTs(NumTrackSet, NumTracks_HS, "TracksInfoInLTs_HS");
    NumLTs_2PI = GetNumLTsTotal(NumTrackSet, NumAzimAng, NumLongTracks, "NumLTs_2PI");
    NumTracksInLTs_Total = AllocMem_int_IP(NumTrackSet, NumLTs_2PI, "NumTracksInLTs_Total");
    TrackIndexInLTs_2PI = AllocMem_int_IPplusI(NumTrackSet, NumLTs_2PI, 1, "TrackIndexInLTs_2PI");

    if (_geomType == 0){
        LongTracks_PathLink_Modified(LongTracksInfo_Total, TracksInfoInLTs_HS, TracksInfoInLTs_2PI, \
		NumTrackSet, NumAzimAng, NumLongTracks, NumLongTracks_SubDomX, NumLongTracks_SubDomY, NumLTs_HS, \
		NumSubDomains, NumPin_subDomain, NumTracksX, NumTracksY, SubDomLocID_2D_X, SubDomLocID_2D_Y, \
		NumTracksInLTs_Total, TrackIndexInLTs_2PI);
    } else {
        LongTracks_PathLink_Hex(LongTracksInfo_Total,
                                TracksInfoInLTs_HS,
                                TracksInfoInLTs_2PI,
                                NumTrackSet,
                                NumAzimAng,
                                NumLongTracks,
                                NumLongTracks_SubDomX,
                                NumLongTracks_SubDomY,
                                NumLTs_HS,
                                NumSubDomains,
                                NumPin_subDomain,
                                NumTracksX,
                                NumTracksY,
                                SubDomLocID_2D_X,
                                SubDomLocID_2D_Y,
                                NumTracksInLTs_Total,
                                TrackIndexInLTs_2PI,
                                _nextAzimID,
                                _nextTrackID);
    }

//    printf("\n NumTracks_Total=%d", NumTracks_Total[0]);
//    printf("\n NumTracks_HS=%d", NumTracks_HS[0]);
//    for (M_INT32 i = 0; i < NumTrackSet; i++)
//        for (M_INT32 j = 0; j < NumAzimAng[i]; j++)
//            printf("\n  NumLTsSum[i=%d][j=%d]=%d",i,j, NumLTsSum[i][j]);

//    printf("\n NumLTs_2PI=%d", NumLTs_2PI[0]);
//    for (M_INT32 i = 0; i < NumTrackSet; i++)
//        for (M_INT32 j = 0; j < NumLTs_2PI[i]; j++)
//            printf("\n  NumTracksInLTs_Total[i=%d][j=%d]=%d, TrackIndexInLTs_2PI[i=%d][j=%d]=%d", i, j, NumTracksInLTs_Total[i][j], i, j, TrackIndexInLTs_2PI[i][j]);
//    exit(90);


    LTsIndex_AzimAngID_2PI = AllocMem_int_IP(NumTrackSet, NumLTs_2PI, "LTsIndex_AzimAngID_2PI");
    LTsIndex_LTID_2PI = AllocMem_int_IP(NumTrackSet, NumLTs_2PI, "LTsIndex_LTID_2PI");
    GetIndexOfLongTracks_Total(LTsIndex_AzimAngID_2PI, LTsIndex_LTID_2PI, NumTrackSet, NumAzimAng, NumLongTracks);

    SumNumFSRsBasedGlobCellID = AllocMem_i32_i32(MOCNumLayersZ*NumPinsInSubDomPerLayer, "SumNumFSRsBasedGlobCellID");
    for (i = 0; i < MOCNumLayersZ*NumPinsInSubDomPerLayer; i++)
        if (i==0)
            SumNumFSRsBasedGlobCellID[i]=0;
        else
            SumNumFSRsBasedGlobCellID[i]=SumNumFSRsBasedGlobCellID[i-1] + NumFSRsInCell[i-1];

    FSRArea_SubDom = AllocMem_FP_I_Geo(TotalNumFSRs_subDom, "FSRArea_SubDom");
    FSRVol_SubDom = AllocMem_FP_I_Geo(TotalNumFSRs_subDom, "FSRVol_SubDom");
    SetFSRArea_trackFile(MOC_NumPinsInSubDom, NumFSRsInRTM, SumNumFSRsInRTM, \
		CellRTMType_SubDom, FSRsAreaInRTM_AntiClockFSR, FSRArea_SubDom, SumNumFSRsBasedGlobCellID);

//    for(M_INT32 i = 0; i < TotalNumFSRs_subDom; i++)
//        printf("\nFSRArea_SubDom[%d] = %lf", i, FSRArea_SubDom[i]);
//    exit(90);

    SetFSRVol_SubDom(FSRArea_SubDom,FSRVol_SubDom,WidthLayersInCore,MOCNumLayersZ, NumPinsInSubDomPerLayer, \
	  NumFSRsInCell, SumNumFSRsBasedGlobCellID);

    FSRsXSID_subDom = AllocMem_i32_i32(TotalNumFSRs_subDom, "TotalNumFSRs_subDom");
    for(_pin_cnt = 0; _pin_cnt < MOC_NumPinsInSubDom; _pin_cnt++){
        _cellType = _pinCellType_SubDom[_pin_cnt];
        for(_fsr_cnt = 0; _fsr_cnt < _numFSRInPinCell[_cellType]; _fsr_cnt++){
            FSRsXSID_subDom[_totFSRCount] = _fsrMatIDInPinCell[_cellType][_fsr_cnt];
            //printf("\nFSRsXSID[%d] = %d", _totFSRCount, FSRsXSID_subDom[_totFSRCount]);
            _totFSRCount++;
        }
    }

    MaxNumSegsInTracks = GetMaxNumSegsInTracks(NumTrackSet, TotalNumTracksInRTM_Total, NumSegsInTrack_2PI);

    NumTracksInLTs_HS = AllocMem_int_IP(NumTrackSet, NumLTs_HS, "NumLongTracks");
    for (i = 0; i < NumTrackSet; i++)
        for (j = 0; j < NumLTs_HS[i]; j++)
            NumTracksInLTs_HS[i][j] = NumTracksInLTs_Total[i][j];

    MaxNumTracksInLTs = GetMaxNumTracksInLongTracks(NumTrackSet, NumLTs_HS, NumTracksInLTs_HS);
    MaxNumPolarAng = GetMaxNumPolarAng(NumTrackSet,NumPolarAng);
//exit(999);
    //for (i = 0; i < NumTrackSet; i++) {
    //	for (j = 0; j < NumAzimAng[i]; j++) {
    //		printf("\n after i=%d, j=%d, NumLongTracks_SubDomX=%d, NumLongTracks_SubDomY=%d, NumLongTracks=%d", i, j, NumLongTracks_SubDomX[i][j], NumLongTracks_SubDomY[i][j], NumLongTracks[i][j]);
    //	}
    //}

    _nextAzimID = FreeMem_i32_I(_nextAzimID, "_nextAzimID");
    _nextTrackID = FreeMem_i32_I(_nextTrackID, "_nextAzimID");
    _pinCellRTMID = FreeMem_i32_I(_pinCellRTMID, "_pinCellRTMID");
    _numFSRInPinCell = FreeMem_i32_I(_numFSRInPinCell, "_numFSRInPinCell");
    _fsrMatIDInPinCell = FreeMem_int_IP(_fsrMatIDInPinCell, _numPinCell, "_fsrMatIDInPinCell");
    _pinCellType_WhoDom = FreeMem_i32_I(_pinCellType_WhoDom, "_pinCellType_WhoDom");
    _pinCellType_SubDom = FreeMem_i32_I(_pinCellType_SubDom, "_pinCellType_SubDom");


}

FP_PRECISION_TIME TallyCurrentTime_CPU()
{
	FP_PRECISION_TIME _time = 0.;
// #if defined(CUDA_USED)
//     cudaDeviceSynchronize();
// #endif
#if defined(MPI_USED)
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    _time = TIME_RECORD;

	return(_time);
}

FP_PRECISION_TIME TallyMonotonicTime_CPU()
{
    struct timespec _timeSpec;

    if (clock_gettime(CLOCK_MONOTONIC, &_timeSpec) != 0) {
        return (FP_PRECISION_TIME) 0.0;
    }

    return (FP_PRECISION_TIME) _timeSpec.tv_sec +
        (FP_PRECISION_TIME) _timeSpec.tv_nsec * (FP_PRECISION_TIME) 1.0e-9;
}


/**
* @ Function: GodBless()
*   introduction: God Bless U, NO BUG
* @ Parameter List:       godName -> name of God, ZQ-Resonance God  LL-MOC God
*   Return: photo of god
* @ Created     : 2020/9/28, by ZouHang, NPSRC, HEU (912841017@qq.com)
*   1st modified:
*/
void GodBless(char godName[CASEID_N]){
    if(!strcmp(godName, "ZQ")){
        printf("\n");
        printf("                                                  #                                                 \n");
        printf("                                        # # # # # #   # #   # #                                     \n");
        printf("                                  # # # # # # # # # # # # # # # # # #                               \n");
        printf("                                  # # # # # # # # # # # # # # # # # # #                             \n");
        printf("                                # # # # # # # # # # # # # # # # # # # # #                           \n");
        printf("                        # # # # # # # # # # # # # # # # # # # # # # # # # #                         \n");
        printf("                      # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                     \n");
        printf("                    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                   \n");
        printf("                    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                 \n");
        printf("                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             \n");
        printf("              # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #           \n");
        printf("              # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         \n");
        printf("          # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     # # # # # # # # # #       \n");
        printf("        # # # # # # # # # # # # # # # #       # # # # # # # # # # # #       # # # # # # # # # #     \n");
        printf("      # # # # # # # # # # # # #   # #               # # # # # #   # #           # # # # # # # #     \n");
        printf("    # # # # # # # # # #   # #     # #                   #                       # # # # # # # #     \n");
        printf("  # # # # # # # # # #                                                           # # # # # # # #     \n");
        printf("  # # # # # # # # # #                                                             # # # # # # # #   \n");
        printf("# # # # # # # # # # #                                                             # # # # # # # #   \n");
        printf("# # # # # # # # # # #                                                               # # # # # # #   \n");
        printf("# # # # # # # # # #                                                                 # # # # # # #   \n");
        printf("# # # # # # # # # #                                             #                   # # # # # # #   \n");
        printf("# # # # # # # # # #         # # # # # # # # #               # # # # # #             # # # # # # #   \n");
        printf("# # # # # # # # # #       # # # # # # # #                   # # # # # # # # #         # # # # # #   \n");
        printf("# # # # # # # # #       # # #                                                 #       # # # # # #   \n");
        printf("# # # # # # # # #     # # #                                                   #         # # # # #   \n");
        printf("# # # # # # # #   # # # # # #       # # # # # #           # # # # #           # # # #   # # # # #   \n");
        printf("# # # # # # # # # # # #             # #       # # # # # #     # # # # # #         # # # # # # # #   \n");
        printf("  # # # # # # # #   # #       # # # # # # #     #     #     # # # # # # # # #     #     #   # # #   \n");
        printf("  # # # # # # #     # #   # # # # # # # # #     #     # #     # # # # #   # # #   #   # # # # # #   \n");
        printf("  # # # # # # #           # #       # #       # #     # #         # # # #         #       # # # #   \n");
        printf("  # # # # # #                                 # #       #                                 # # # #   \n");
        printf("  # # # # # #         #                       #           #                                 # # #   \n");
        printf("  # # # # # #                               #             # #                               # #   # \n");
        printf("    # # # # #                           # #                   # #                           # # # # \n");
        printf("      # # # #                           #                       #                           # # # # \n");
        printf("    # # # # #                           #     # # # # # # #     #                           # # #   \n");
        printf("    # # # # #                           # # # # # # # # # # # #                               # #   \n");
        printf("    #   # # #                             # #             # #                                   #   \n");
        printf("          #                                                                                     #   \n");
        printf("          #                                                                                     #   \n");
        printf("          #                                     #       # #                                     #   \n");
        printf("                                              # # # # # # # #                                       \n");
        printf("          #                               # # # # # #   # # # # #                                   \n");
        printf("          #                             # # #                 # # # #                               \n");
        printf("            #                         # #           # #             #                               \n");
        printf("                                                  # # #                                             \n");
        printf("          #                                                                                         \n");
        printf("           #                                                                                        \n");
        printf("                                                                                          #         \n");
        printf("            #                                                                               #       \n");
        printf("            #                                                                           # # #       \n");
        printf("              #                                                                     # # # # #       \n");
        printf("              # # #                                                               # # # # # #       \n");
        printf("              # # #               # #             # # # #           # # #   # # #   # # # # #       \n");
        printf("              # # # # #   # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         \n");
        printf("                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   # # # # # # #         \n");
        printf("                #     # # # # # # # # # # # # # #       # # # # #             # #   # # # #         \n");
        exit(666);
    }
    if(!strcmp(godName, "LL")){
        printf("\n");
        printf("                                                                                                    \n");
        printf("                                      # # # # # # # # # #                                           \n");
        printf("                                # # # # # # # # # # # # # #                                         \n");
        printf("                                # # # # # # # # # # # # # # # # #                                   \n");
        printf("                              # # # # # # # # # # # # # # # # # # #                                 \n");
        printf("                  # # # # # # # # # # # # # # # # # # # # # # # # # # #                             \n");
        printf("                  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                     \n");
        printf("                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                 \n");
        printf("              # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                 \n");
        printf("          # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #               \n");
        printf("        # # # # # # # # # # # # #   # # # # # # # # # # # # # # # # # # # # # # # # # #             \n");
        printf("        # # # # # # # # # #     #         # # # # # # # # # # # # # # # # # # # # # # # #           \n");
        printf("      # # # # # # # # # #                             # # # # # # # # # # # # # # # # # # #         \n");
        printf("      # # # # # # # # #                                               # #     # # # # # # #         \n");
        printf("      # # # # # # # #                                                   #     #   # # # # # #       \n");
        printf("      # # # # # # # #                                                     #     # # # # # # # #     \n");
        printf("      # # # # # # #                                                                 # # # # # #     \n");
        printf("      # # # # # # #                                                                 # # # # # # #   \n");
        printf("      # # # # # #                                                                   # # # # # # #   \n");
        printf("    # # # # # # #                                                                     # # # # # #   \n");
        printf("    # # # # # # #                                                                     # # # # # #   \n");
        printf("    # # # # # # #                 # #                         # # # # # #             # # # # # #   \n");
        printf("    # # # # # #           # # # # # # #                               # # #           # # # # # #   \n");
        printf("    # # # # # #         # # #                               # # # # # # # # #           # # # # #   \n");
        printf("    # # # # # #         #       # # # # #               # # # #           # # # #       # # # # #   \n");
        printf("    # # # # #         # # # #           # # # #       # #                     # # # #   # # # # #   \n");
        printf("    # # # # #     # # #                       # # # # #       # # # #         #       #   # # # #   \n");
        printf("    # # # # # # # # #       # # # # # # #     #       #     # # # # # # # #   #       # # # # # #   \n");
        printf("    # # # # # #     # #   # #     # # # # #   # #   # #     #                 #           # # # #   \n");
        printf("    # # # # #         #                     # #       #                       #           # # # #   \n");
        printf("    # # # #           #                     # #         #                                 # # # # # \n");
        printf("    # # # #                                 #           # #               #               # # # # # \n");
        printf("    # # # #             # #             # #                 # # # # #                       #   #   \n");
        printf("    #   # #                 # # # # # # #                                                   #       \n");
        printf("          #                                                                                 #       \n");
        printf("          #                                                                                 #       \n");
        printf("          #                                 # # #   # # # #                                 # #     \n");
        printf("        # #                               #       # #     # #                               # #     \n");
        printf("        # #                                                                                 #       \n");
        printf("        # #                                                                                         \n");
        printf("                                                                                                    \n");
        printf("                                                                                                    \n");
        printf("                                          # # # # # # # # #                                         \n");
        printf("                                      # # # # # # # # # # # # # #                                   \n");
        printf("                                    # # # #                 # # #                                   \n");
        printf("            #                                     #                                                 \n");
        printf("                                                  # #                                               \n");
        printf("                                                                                                    \n");
        printf("                                                                                                    \n");
        printf("                                                                                                    \n");
        printf("                                                                                                    \n");
        printf("                    #                                                           # #                 \n");
        printf("                    # # #                                             #   # # # #                   \n");
        printf("                      # # #         # #       # # # # # # # # # # # # # # # # # #                   \n");
        printf("                      # # # # # # # # # # # # # # # # # # # # # # # # # # # # #                     \n");
        printf("                             # # # # # # # # # # # # # # # # # # # # # #                            \n");
        printf("                                 # # # # # # # # # # # # # # # # # #                                \n");
        exit(666);
    }
    else{}
}

#if 0
void MOC_PrintLogFile(M_INT32 _procID)
{
	if (_procID == 0) {
		char _fileName[512] = { "" }, _name1[256] = { "" }, _name2[128] = { "" }, _name_time[64] = { "" };
		time_t _curr_time;
		struct tm *_p_time;
		M_INT32 _time_int;
#if defined(MPI_USED)
		int i;
#endif

#if defined(CUDA_USED)
		sprintf(_name1, "%s%s", _name1, "_g");
#else
		sprintf(_name1, "_c");
#endif //defined(CUDA_USED)

#if defined(FP64)
		sprintf(_name1, "%s%s", _name1, "64");
#endif //defined(FP64)

#if defined(FPMIX)
		sprintf(_name1, "%s%s", _name1, "Mix");
#endif //defined(FPMIX)

#if defined(FP32)
		sprintf(_name1, "%s%s", _name1, "32");
#endif //defined(FP32)

#if defined(M_MG_CMFD)

		if (MGCMFD_LinearSolver == 0) {
			sprintf(_name2, "%s%s", _name2, "_jacobi");
		}
		else if (MGCMFD_LinearSolver == 1) {
			sprintf(_name2, "%s%s", _name2, "_sor");
		}
		else if (MGCMFD_LinearSolver == 2) {
			if (GmresPrecond_MG == 1) {
				sprintf(_name2, "%s%s%d", _name2, "_pgmres", GmresKrylovDim_MG);
			}
			else {
				sprintf(_name2, "%s%s%d", _name2, "_gmres", GmresKrylovDim_MG);
			}
		}
		else if (MGCMFD_LinearSolver == 3) {
			sprintf(_name2, "%s%s", _name2, "_zyq");
		}
		else {
			printf("\n [**ERROR**]  MGCMFD_LinearSolver is %d, please reset!\n", MGCMFD_LinearSolver);
			fflush(stdout);
			exit(69);
		}

#if defined(M_CG_CMFD)
		sprintf(_name1, "%s%s", _name1, "_2L");
		if (CGCMFD_LinearSolver == 0) {
			sprintf(_name2, "%s%s", _name2, "+jacobi");
		}
		else if (CGCMFD_LinearSolver == 1) {
			sprintf(_name2, "%s%s", _name2, "+sor");
		}
		else if (CGCMFD_LinearSolver == 2) {
			if (GmresPrecond_CG == 1) {
				sprintf(_name2, "%s%s%d", _name2, "+pgmres", GmresKrylovDim_CG);
			}
			else {
				sprintf(_name2, "%s%s%d", _name2, "+gmres", GmresKrylovDim_CG);
			}
		}
#else
		sprintf(_name1, "%s%s", _name1, "_1L");
#endif //defined(M_CG_CMFD)
		sprintf(_name1, "%s%s", _name1, _name2);
#else 
		sprintf(_name1, "%s%s", _name1, "_moc");
#endif //  defined(M_MG_CMFD)


#if defined(CUDA_USED)
		if (GPU_TSKernelType == 0) {
			sprintf(_name1, "%s%s", _name1, "_eI");
		}
		else if (GPU_TSKernelType == 1) {
			sprintf(_name1, "%s%s", _name1, "_eF");
		}
		else if (GPU_TSKernelType == 2) {
			sprintf(_name1, "%s%s", _name1, "_ddEI");
		}
		else if (GPU_TSKernelType == 3) {
			sprintf(_name1, "%s%s", _name1, "_ddEF");
		}
		else {}
#else
		if (CPU_TSFunctionType == 0) {
			sprintf(_name1, "%s%s", _name1, "_eI");
		}
		else if (CPU_TSFunctionType == 1) {
			sprintf(_name1, "%s%s", _name1, "_ddEI");
		}
		else if (CPU_TSFunctionType == 2) {
			sprintf(_name1, "%s%s", _name1, "_eF");
		}
		else if (CPU_TSFunctionType == 3) {
			sprintf(_name1, "%s%s", _name1, "_ddEF");
		}
		else {}
#endif //defined(CUDA_USED)

        sprintf(_name1, "%s_%.2lf_a%d_p%d_k%.0e_f%.0e", _name1, (double)(RaySpacingIni[0]),NumAzimAng[0],NumPolarAng[0],ConvCritK,ConvCritFlux);

		sprintf(CaseId, "%s%s", CaseId, _name1);
		time(&_curr_time);
		_p_time = gmtime(&_curr_time);

		_time_int = 1900 + _p_time->tm_year;
		sprintf(_name_time, "%s%d", _name_time, _time_int);
		_time_int = _p_time->tm_mon + 1;
		if (_time_int < 10) {
			sprintf(_name_time, "%s%s%d", _name_time, "0", _time_int);
		}
		else {
			sprintf(_name_time, "%s%d", _name_time, _time_int);
		}

		_time_int = _p_time->tm_mday;
		if (_time_int < 10) {
			sprintf(_name_time, "%s%s%d", _name_time, "0", _time_int);
		}
		else {
			sprintf(_name_time, "%s%d", _name_time, _time_int);
		}
		_time_int = _p_time->tm_hour + 8;
		if (_time_int < 10) {
			sprintf(_name_time, "%s%s%d", _name_time, "_0", _time_int);
		}
		else {
			sprintf(_name_time, "%s%s%d", _name_time, "_", _time_int);
		}
		_time_int = _p_time->tm_min;
		if (_time_int < 10) {
			sprintf(_name_time, "%s%s%d", _name_time, "0", _time_int);
		}
		else {
			sprintf(_name_time, "%s%s%d", _name_time, "", _time_int);
		}
		_time_int = _p_time->tm_sec;
		if (_time_int < 10) {
			sprintf(_name_time, "%s%s%d", _name_time, "0", _time_int);
		}
		else {
			sprintf(_name_time, "%s%s%d", _name_time, "", _time_int);
		}
		sprintf(CaseId, "%s-%s", CaseId, _name_time);
		sprintf(_fileName, "%s%s", CaseId, "-log.out");
		LogFile = fopen(_fileName, "w");
		if (LogFile == NULL) {
			printf("\n [**ERROR**]  %s file opens error in process %d!\n", _fileName, _procID);
			fflush(stdout);
			exit(69);
		}

		fprintf(LogFile, "\n  *****************************************************************");
		fprintf(LogFile, "\n  *                                                               *");
		fprintf(LogFile, "\n  *                           Log  File                           *");
		fprintf(LogFile, "\n  *                                                               *");
		fprintf(LogFile, "\n  *****************************************************************\n");
#if defined(MPI_USED)
		for (i=0;i<NumProcs;i++) {
			if (i==0) {
				fprintf(LogFile,"\n [  INFO   ]  Process ID: %3d�?Processor Name: %6s, CPU ID:%3d", i, ProcessorNameColl[i], CurrCPUIDColl[i]);
			}
			else {
				fprintf(LogFile,"\n              Process ID: %3d�?Processor Name: %6s, CPU ID:%3d", i, ProcessorNameColl[i], CurrCPUIDColl[i]);
			}
		}
	    fflush(stdout);
#endif
		fprintf(LogFile, "\n [  INFO   ]  Input File Name:   %s", IP);
		fprintf(LogFile, "\n [  INFO   ]  CASE ID:           %s", CaseId);
#ifdef DOUBLE
		fprintf(LogFile, "\n [  INFO   ]  The Floating-Point Precision is DOUBLE");
#endif
#ifdef SINGLE
		fprintf(LogFile, "\n [  INFO   ]  The Floating-Point Precision is SINGLE");
#endif
		fprintf(LogFile, "\n [  INFO   ]  Setting Macroscopic Cross Section Data ...");
		fprintf(LogFile, "\n [  INFO   ]  Creating Geometry ...");

	}

#ifdef MPI_USED
	MPI_Bcast(CaseId, 128, MPI_CHAR, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
#endif // MPI_USED

}

#endif
#if 0
void SetFSRArea_trackFile(M_INT32 _MOC_NumPinsInSubDom, M_INT32 *_NumFSRsInRTM, M_INT32 *_SumNumFSRsInRTM, \
	M_INT32 *_CellRTMType_SubDom, FP_PRECISION_GEO *_FSRsAreaInRTM, FP_PRECISION_GEO *_FSRArea_SubDom, \
	M_INT32 *_SumNumFSRsBasedGlobCellID)
{
	M_INT32 _rtm, i, j, _fsrBase, _fsr;
	for (i = 0; i < _MOC_NumPinsInSubDom; i++) {
		_rtm = _CellRTMType_SubDom[i];
		_fsrBase = _SumNumFSRsInRTM[_rtm];
		_fsr = _SumNumFSRsBasedGlobCellID[i];
		for (j = 0; j < _NumFSRsInRTM[_rtm]; j++) {
			_FSRArea_SubDom[_fsr + j] = _FSRsAreaInRTM[_fsrBase + j];
		}
	}
}
#endif

#if 0
void SetNumFSRsInCell_trackFile(M_INT32 *_NumFSRsInCell, M_INT32 _MOC_NumPinsInSubDom, \
	M_INT32 *_NumFSRsInRTM, M_INT32 *_CellRTMType_SubDom)
{
	M_INT32 i, _rtm;
	for (i = 0; i < _MOC_NumPinsInSubDom; i++) {
		_rtm = _CellRTMType_SubDom[i];
		_NumFSRsInCell[i] = _NumFSRsInRTM[_rtm];
	}
}
#endif

