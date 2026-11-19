#ifndef MOC_COMMON_MACRO_H
#define MOC_COMMON_MACRO_H
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// #define TESTR

typedef unsigned int M_UINT32;
typedef int M_INT32;
typedef unsigned long long M_UINT64;
typedef long long M_INT64;
typedef size_t M_SIZET;

#define INT32_FORMAT "%u"
#define INT64_FORMAT "%llu"

/**********************************************************/
/**************  MACRO used on Windows OS *****************/

//  #define M_2DMOC
// #define M_3DMOCex

// #define FP64
// #define FP32

// #define MPI_USED
// #define CUDA_USED

// #define M_MG_CMFD
// #define M_CG_CMFD

//#define SCHED_LINUX

/**************  MACRO used on Windows OS *****************/
/**********************************************************/








  //#define M_ZeroNegFissSour
  //#define M_ZeroNegFlux

  // #define M_3DOpt

  //#define M_PrintDetail

 //#if defined(M_2DMOC)
   #define M_Use4PI
 //#endif // defined(M_2DMOC)

// #define M_MultiMOCIter


#define M_UpdateRadialBoundImme

#if defined(M_3DMOCex)
  #define M_UpdateAxialBoundImme
  // #define  M_LayerInnerIter
  #define M_3DInnerIter
  // #define M_3DAxialFluxSOR
  //#define M_TallyNegFlux
#endif // defined(M_3DMOCex)

  ////#define M_MGLS_JACOBI
//#define M_MGLS_SOR
//#define M_MGLS_GMRES
//#define M_CGLS_JACOBI
//#define M_CGLS_SOR
//#define M_CGLS_GMRES 


//#define OMP_USED



#if defined(M_MG_CMFD)

  #define M_CMFD_DEBUG
  //    //#define M_ZYQ_GroupOrder  // memory location is consistant for groups
  //    //#define M_ZYQ_NodeOrder // memory location is consistant for nodes

  // #define M_UpdateKeff

  #define M_CMFDMatrixIncSelfScatt // the CMFD Matrix A only include self scattering term

#define M_YQZ_CMFD_Damp // use cmfd damp


//#define M_YQZ_CMFD_Keff // fission rate for moc_keff calculation is not integrated over volume

//#define M_YQZ_CMFD_Conv // if cmfd is not converged, don't update the keff, then performe the moc sweep

  #define M_CMFDMatrixIncScatt  // the CMFD Matrix A include scattering term
  #define LARSEN_CORR

// #define YUNLIN_CORR
#endif


//#define MOCthenCMFD

// 

 #define M_VolCorr

  // #define M_TIME_GMRES

  //#define M_GMRES_JacobiPRE
   #define M_GMRES_IluPRE

  #define M_GMRES_PRE_cuSparseLib
  // #define M_GMRES_PRE_onCPU



#define CUDAErrorCheck

// #define M_MG_CMFD_FixSourProb
// #define M_CG_CMFD_FixSourProb                     // macro for coarse group cmfd

#define DEFAULT_TIME


//#define PETSc_ERR_CHECK
// #define PTESC_LIB


//#define M_DetailLSInfo


#define GROUP_WISE_ORDER
// #define PIN_WISE_ORDER

// #define NORMALIZE_FLUX


#define DIRECT_CSR



//#define OMP_ATOMIC
//#define CPUMPI_CPUTS_ASYNCH_OMP
//#define CPU_OMP_COMP
//#define CPU_OMP_COMM
//#define OMP_SCHEDULE // unrool the loop over long rays with different scheduling model (static, dynamic, guided)

//#define DynamicWorkLoad

//#define NoStream_NoAsyn_NoOMP
//#define Stream_NoAsyn_NoOMP
//#define NoStream_Asyn
//#define NoStream_OMP
//#define Stream_Asyn
//#define Stream_OMP

#ifdef Stream_NoAsyn_NoOMP
  #define GPU_STREAM_USED /* set the utilization of CUDA multi-stream */
#endif // Stream_NoAsyn_NoOMP

#ifdef NoStream_Asyn
  #define CPUMPI_GPUTS_ASYNCH  /* define the asynchronization between CPU-MPI-Communication and GPU-TSkernel-execution*/
  #define CPUMPI_CPUTS_ASYNCH_OMP
#endif // NoStream_Asyn

#ifdef NoStream_OMP
  #define OMP_ASYN   /* OpenMP used*/
#endif // NoStream_OMP

#ifdef Stream_Asyn
  #define GPU_STREAM_USED /* set the utilization of CUDA multi-stream */
  #define CPUMPI_GPUTS_ASYNCH  /* define the asynchronization between CPU-MPI-Communication and GPU-TSkernel-execution*/
  #define CPUMPI_CPUTS_ASYNCH_OMP
#endif // Stream_Asyn

#ifdef Stream_OMP
  #define GPU_STREAM_USED /* set the utilization of CUDA multi-stream */
  #define OMP_ASYN   /* OpenMP used*/
#endif // Stream_OMP

//#define CUDA_AWARE_MPI  /* define the use of CUDA-Aware-MPI, such as MVAPICH2 and OpenMPI */
//#define NVPROF_CPU
// #define TIME_REC /*set the recording of time consuming for MPI processes*/
//#define TIME_REC_OMP /*set the recording of time consuming for OMP threads*/
//#define CUDA_CLOCK /* define the execution time record by using the CUDA Event related function*/
//#define MPI_CLOCK


#define FP_PRECISION_TIME double
#define MPI_CUDA_UpAngFImme
#define GC_COPY_AZIM

//#define HYBRID_ITER

//#define OMP_TIME
/* defalut option*/
// #ifndef DOUBLE
//   #ifndef SINGLE
//     // #define DOUBLE
//     #define FP_PRECISION float//
//     #define FP_FORMAT "%f"
//   #endif
// #endif
#ifdef FP64
  #define DOUBLE  // SINGLE   DOUBLE
  #define DOUBLE_FLUX //DOUBLE_FLUX SINGLE_FLUX
  #define FP_PRECISION_TEST double /* float double */
  #define EXP_DOUBLE  // EXP_DOUBLE  EXP_SINGLE
#endif
#ifdef FPMIX
  #define DOUBLE  // SINGLE   DOUBLE
  #define SINGLE_FLUX //DOUBLE_FLUX SINGLE_FLUX
  #define FP_PRECISION_TEST double /* float double */
  #define EXP_SINGLE  // EXP_DOUBLE  EXP_SINGLE
#endif
#ifdef FP32
  #define SINGLE  // SINGLE   DOUBLE
  #define SINGLE_FLUX //DOUBLE_FLUX SINGLE_FLUX
  #define FP_PRECISION_TEST float /* float double */
  #define EXP_SINGLE  // EXP_DOUBLE  EXP_SINGLE
#endif // FP32

#ifdef EXP_SINGLE
  #define FP_PRECISION_EXP float
#endif // EXP_SINGLE

#ifdef EXP_DOUBLE
  #define FP_PRECISION_EXP double
#endif // EXP_DOUBLE

#define DOUBLE_GEO

#ifdef DOUBLE
  #define FP_PRECISION double//
  #define FP_FORMAT "%lf"
  #define FP_DOUBLE double
  #define FP_FORMAT_D "%lf"
#endif

#ifdef SINGLE
  #define FP_PRECISION float//
  #define FP_FORMAT "%f"
  #define FP_DOUBLE double
  #define FP_FORMAT_D "%lf"
#endif

#ifdef DOUBLE_GEO
  #define FP_PRECISION_GEO double//
  #define FP_FORMAT_GEO "%lf"
  #define TINY_MOVE 1.0E-8
  #define ON_SURFACE 1.0E-12
#endif

#ifdef DOUBLE_FLUX
  #define FP_PRECISION_FLUX double//
  #define FP_FORMAT_FLUX "%lf"
#endif

#ifdef SINGLE_FLUX
  #define FP_PRECISION_FLUX float//
  #define FP_FORMAT_FLUX "%f"
#endif


//#define MPI_USED
#ifdef MPI_USED
  #ifdef DOUBLE
    #define MPI_DataType MPI_DOUBLE
    #define MPI_FP_DOUBLE MPI_DOUBLE
  #endif

  #ifdef SINGLE
    #define MPI_DataType MPI_FLOAT
    #define MPI_FP_DOUBLE MPI_DOUBLE
  #endif

  #define MPI_DataType_Time MPI_DOUBLE
#endif // MPI_USED

// #ifdef MPI_USED
//     #define TIME_RECORD (FP_PRECISION_TIME)MPI_Wtime()
//     #define TIME_TYPE1 //TIME_TYPE1 - MPI_Wtime(); TIME_TYPE2 - clock()
// #else
    #define TIME_TYPE2 //TIME_TYPE1 - MPI_Wtime(); TIME_TYPE2 - clock()
    #define TIME_RECORD (FP_PRECISION)clock()
// #endif // MPI_USED

#ifdef OMP_TIME
    #define TIME_RECORD_OMP (FP_PRECISION)omp_get_wtime()
#endif // OMP_TIME


#define XS_N 60
/** Constant */
#define INFINITY_DOUBLE 1.0E8
#define INFINITY_INT 1E8
#define ON_SURFACE_THRESH 1.0E-10
#define SMALL_VALUE 1.0E-8
#define ONE_PI 3.1415926535897
#define HALF_PI 1.57079632679485
#define FOUR_PI 12.5663706143588
#define ONE_OVER_FOUR_PI 0.07957747154594767
#define FileNameLength 225
#define NumNeibSubDom 2
#define NumMessType 2
#define NUM_SURF  4
#define NUM_SURF_DIR 2
#define GeoTwoPIRegion
#define MAX_NumGroups  69
#define MAX_NumPolAngles  32
#define MAX_NumSegInLT 1586

#define M_One_FP 1.0
#define M_Zero_FP 0.0
#define M_MB_Unit 1.0/1048576.0

// #if FP_PRECISION==double
//   #define FP_FORMAT "%lf"
// #elif FP_PRECISION==float
//   #define FP_FORMAT "%f"
// #else
// #endif

#define CHAR_NONE "\0"
// #define ZhangYunfei
// #define ZhangYunfei_Visu
// #ifdef CUDA_USED
    /** The maximum number of polar angles to reserve constant memory on GPU */
    #define MAX_POLAR_ANGLES_GPU 6
    /** The maximum number of azimuthal angles to reserve constant memory on GPU */
    #define MAX_AZIM_ANGLES_GPU 128
// #endif // CUDA_USED
// #ifdef MPI_USED
// #define MPI_CHECK(call) \
// 	if (call != MPI_SUCCESS) {
// 		printf(" [**ERROR**]  Error Calling \""#call"\""); \
// 		// fflush(stdout); \
// 		exit(69);
// 	}
// #endif
#define ParaNameLength 128
#define NumTimeMetrics 32

#ifdef TIME_REC
  extern int TimeStamp_TS, TimeStamp_ConvCheck, TimeStamp_SE, TimeStamp_MPIComm,TimeStamp_CommCGCopy,TimeStamp_CommGCCopy,TimeStamp_CMFD;
  extern int TimeStamp_CMFDtoMOC,TimeStamp_MGGCCond;
  extern int TimeStamp_MOC_ParaUp, TimeStamp_GPUCopyBoundAF, TimeStamp_MOC_AStF, TimeStamp4, TimeStamp5, TimeStamp6, TimeStamp7, TimeStamp8, TimeStamp9, TimeStamp10;
  extern int TimeStamp_MOC_TS1,TimeStamp_MOC_TS2,TimeStamp_MOC_TS3,TimeStamp_MOC_TS4,TimeStamp_MOC_TS5;

  extern int TimeCount, TimeCountMOC, TimeCountMGCMFD, TimeCountCGCMFD;
  extern int TimeCountPrePart;
  extern int TimeMGStamp1,TimeMGStamp2,TimeMGStamp3,TimeMGStamp4,TimeMGStamp5,TimeMGStamp6,TimeMGStamp7,TimeMGStamp8,TimeMGStamp9,TimeMGStamp10,TimeMGStamp11,TimeMGStamp12;
  extern int TimeCGStamp1,TimeCGStamp2,TimeCGStamp3,TimeCGStamp4,TimeCGStamp5,TimeCGStamp6,TimeCGStamp7,TimeCGStamp8,TimeCGStamp9,TimeCGStamp10;
  extern int TimeStamp3;
  extern char TimeMetricsPrePart[NumTimeMetrics][ParaNameLength];
  extern char TimeMetrics[NumTimeMetrics][ParaNameLength];
  extern char TimeMetricsMOC[NumTimeMetrics][ParaNameLength];
  extern char TimeMetricsMG[NumTimeMetrics][ParaNameLength];
  extern char TimeMetricsCG[NumTimeMetrics][ParaNameLength];

  extern FP_PRECISION_TIME TimePrePart[NumTimeMetrics];
  extern FP_PRECISION_TIME TimeOverall[NumTimeMetrics];
  extern FP_PRECISION_TIME TimeMOC[NumTimeMetrics];
  extern FP_PRECISION_TIME TimeRecMG[NumTimeMetrics];
  extern FP_PRECISION_TIME TimeRecCG[NumTimeMetrics];
#endif // TIME_REC

#define ZERO_CHECK 1.e-20

#ifdef EXP_SINGLE
#define _ExpFunction_CPU(x) expf(x)
#endif // EXP_SINGLE

#ifdef EXP_DOUBLE
#define _ExpFunction_CPU(x) exp(x)
#endif // EXP_DOUBLE

FP_PRECISION_TIME TallyCurrentTime_CPU();
/* Monotonic elapsed time in seconds.  Unlike TIME_RECORD/clock(), this
 * includes time spent waiting for GPU kernels and synchronizations. */
#ifdef __cplusplus
extern "C" {
#endif
FP_PRECISION_TIME TallyMonotonicTime_CPU();
#ifdef __cplusplus
}
#endif

#if defined(FP64)
#define _CPU_FABS(x) fabs(x)
#define _CPU_CEIL(x) ceil(x)
#endif
#if defined(FP32)
#define _CPU_FABS(x) fabsf(x)
#define _CPU_CEIL(x) ceilf(x)
#endif
#endif //COMMON_MACRO_H
