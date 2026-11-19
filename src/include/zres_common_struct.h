
#ifndef ZRES_COMMON_STRUCT_H
#define ZRES_COMMON_STRUCT_H

#include "condition.h"

// for ring->fsr remap
#include "zres_remap.h"

#ifdef ALPHA_SUPPORT
    #define ZRES_SCM ZRES_SMA
    #include "MOC_Macro_XSs.h"
    #include "MOC_main.h"
    #include "MOC_Input.h"
#endif

#ifndef FP64
    #ifndef FP32
        #define FP64
    #endif
#endif

#ifdef FP64
    #define FP_RES_XS double
    #define FP_RES_FX double
    #define FP_RES_SS double
#endif

#ifdef FP32
    #define FP_RES_XS float
    #define FP_RES_FX float
    #define FP_RES_SS float
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ZRES_GPU_BLOCK_DIM 128

#ifdef CUDA_USED
    #include <cuda_runtime.h>
#ifdef CUBLAS_USED
    #include <cublas_v2.h>
    #include <cublas_api.h>
    #ifdef FP64
        #define cublas_gemv_batched cublasDgemvBatched
        #define cublas_gemm_batched cublasDgemmBatched
        #define cublas_vsum cublasDasum
    #else
        #define cublas_gemv_batched cublasSgemvBatched
        #define cublas_gemm_batched cublasSgemmBatched
        #define cublas_vsum cublasSasum
    #endif
#endif // CUBLAS_USED
    // for solve homo flux
    #include "common_cal_chi_gpu.cuh"
    #define CUDA_CHECK(call) (call)
#endif

typedef int rsint;
typedef unsigned long long rsuint;
#define ZRES_MIN_VAL 1E-10

/**
 * @brief nuclide information
 * 
 */
typedef struct NuclideInfo {  
    int existMatNum;              /* material number that existing this nuclide */
    int* existMatList;            /* list of these materials */
    int* arrayPosition;           /* which position of this nuclide */
    int* subGrpResMask;           /* subgroup remap array */
} ZRES_NUC;

/**
 * @brief nuclides collection
 * 
 */
typedef struct NuclideCollect {
    int numNuc;                   /* nuclide number in the material input */                 
    int* nucList;                 /* nuclide list */
    int** EIDD2;                  /* nuclide re direct array */
    ZRES_NUC* nucs;               /* details nuclide map to the material information */
    int maxSubGrpNum;             /* max subgroup number in all nuclides */
    FP_RES_XS* nucTemp;           /* nuclide temperature used in subgroup method */
} ZRES_NCC;

/**
 * @brief the material composition
 * 
 */
typedef struct MaterialInfo {
    int matId;                    /* material index */ 
    FP_RES_XS temperature;        /* temperature of target material */
    int tempID;                   /* temperature position in lib */
    int numNuclide;               /* nuclide num  */
    int* nucId;                   /* nuclide id list */
    FP_RES_XS* nucDen;            /* nuclide density list */
    int firstFsr;                 /* first fsr used this material */
    int sharedScaMx;             /* if shared scattering matrix between different fsr */

    int nnz;                      /* scattering matrix length for this material, related to scattering order */
    int* startRow;                /* start position of each row, shape: group number + 1, all order shared the startRow */
    int* endRow;                  /* end   position of each row */
    int* startCol;                /* start position of each col, shape: group number + 1 */
    int* endCol;                  /* end   position of each col */
    FP_RES_XS potScaXs;           /* potential scattering xs */
} ZRES_MAT;

/**
 * @brief materials collection
 * 
 */
typedef struct MaterialCollect {
    int matNum;                   /* all input materials */
    ZRES_MAT* mats;               /* material array */
    ZRES_NCC  nucCollect;         /* nuclide collection in material input */
} ZRES_MTC;

/**
 * @brief single scattering matrix
 * 
 */
typedef struct ScatterMatrix{
    int* _rowStart;               /* shape: scatter order * group */
    int* _rowEnd;                 /* shape: scatter order * group */
    FP_RES_XS** _scaXs;           /* scattering xs */
} ZRES_SM;

/**
 * @brief nuclide scattering matrix information
 * 
 */
typedef struct NuclideScatterMatrix {
    int ifExist;                  /* if used in this problem */
    ZRES_SM prob;                 /* scattering prob. matrix */
    ZRES_SM inel;                 /* in-elastic scattering matrix */
    ZRES_SM* thermal;             /* thermal scattering matrix, for multi-temperature */
} ZRES_NSM;

/**
 * @brief subgroup parameter
 * 
 */
typedef struct SubGroupParameter {
    int* subGroupNumber;          /* subgroup number in this group */

    FP_RES_XS** _sub_weight;      /* subgroup weight, first dim is group number, second dim is subgroup number, below is the same */
    FP_RES_XS** _sub_sigma_T;     /* subgroup total xs */
    FP_RES_XS** _sub_sigma_A;     /* subgroup absorption xs */
    FP_RES_XS** _sub_sigma_F;     /* subgroup fission xs */
    FP_RES_XS** _sub_sigma_S;     /* subgroup elastic scattering xs */
    
    int _ifFis;                   /* if fission */
    FP_RES_XS* _sigma_v;          /* neutron yield in this state, only has the dim of group number */

    int* _startRow;               /* start position of scattering matrix, shape: group number */
    int* _endRow;                 /* end position of scattering matrix, shape is the same as startRow */
} ZRES_SBP;

/**
 * @brief subgroup resonance library
 * 
 */
typedef struct SubGroupLibrary {
    int numGroup;                 /* group number */
    int upScaGroup;               /* up scattering start group */
    int maxlg;                    /* max legendre order */
    int maxSubGroupNum;           /* max subgroup number in library */
    int tempNum;                  /* temperature number */
    FP_RES_XS* tempList;          /* temperatures existed in resonance library */
    int numNuclideInLib;          /* nuclide number existing in resonance library */
    int* index;                   /* finding, from library position to nuclide id */
    int* EIDD;                    /* reverse finding, from nuclide id to library position */
    FP_RES_XS* potScaXs;          /* potential scttering cross section */
    
    ZRES_SBP* subGroupPara;       /* subgroup parameters in different states, size: numNuclideInLib x tempNum */
} ZRES_SUB;

/**
 * @brief resonance integral (or resonance xs) table (default: resonance xs)
 * 
 */
typedef struct ResonanceIntegralTable {
    FP_RES_XS* _eqm_sigma_T;      /* total xs table, length: grpNum x tempNum x backNum, below is the same */
    FP_RES_XS* _eqm_sigma_A;      /* absorption xs table */
    FP_RES_XS* _eqm_sigma_F;      /* fission xs table */
    FP_RES_XS* _eqm_sigma_S;      /* scattering xs table */
    
    int _ifFis;                   /* if fission */
    FP_RES_XS* _sigma_v;          /* neutron yield in this state, only has the dim of group number */
    
    int* _startRow;               /* start position of scattering matrix, shape: group number */
    int* _endRow;                 /* end position of scattering matrix, shape is the same as startRow */
} ZRES_RIT;

/**
 * @brief equivalence method library
 * 
 */
typedef struct EquivalenceMethodLibrary {
    int numGroup;                 /* group number */
    int upScaGroup;               /* up scattering start group */
    int maxlg;                    /* max legendre order */
    
    int numNuclideInLib;          /* nuclide number existing in resonance library */
    int* index;                   /* finding, from library position to nuclide id */
    int* EIDD;                    /* reverse finding, from nuclide id to library position */
    FP_RES_XS* potScaXs;          /* potential scttering cross section */

    int tempNum;                  /* temperature number */
    FP_RES_XS* tempList;          /* temperatures existed in resonance library */
    
    int backNum;                  /* background xs number */
    FP_RES_XS* backXs;            /* background xs */
    ZRES_RIT* eqmTable;           /* equivalence method resonance integral (or resonance xs) table */
} ZRES_EQM;

/**
 * @brief ufg xs set
 * @details in fact, hfg xs is also interpolated in urr, but is by smooth xs (might?)
 */
typedef struct ResonanceUFGXs {
    FP_RES_XS* _ufg_sigma_T;          /* nuclide ufg or hfg micro total      xs, dim: temp * grpnum * refineNum */
    FP_RES_XS* _ufg_sigma_S;          /* nuclide ufg or hfg micro elastic    xs, dim: temp * grpnum * refineNum */
    FP_RES_XS* _ufg_sigma_F;          /* nuclide ufg or hfg micro fission    xs, dim: temp * grpnum * refineNum */
    FP_RES_XS* _ufg_sigma_A;          /* nuclide ufg or hfg micro capture    xs, dim: temp * grpnum * refineNum */

    int _ifFis;                       /* if fission */
    FP_RES_XS* _sigma_v;              /* neutron yield in this state, only has the dim of group number */

    int* _startRow;                   /* start position of scattering matrix, shape: group number */
    int* _endRow;                     /* end position of scattering matrix, shape is the same as startRow */
    
    int _ifurr;                       /* if exist urr info */
    int _lssf;                        /* LSSF */
    int _tempNum;                     /* temperature number */
    int _backXsNum;                   /* back ground xs number*/
    FP_RES_XS* _urr_backXs;           /* back ground xs */
    int _ePointNum;                   /* urr energy point number */
    FP_RES_XS* _urr_e;                /* urr energy point */
    int* _urrBoundUFG;                /* urr boundary group */
    FP_RES_XS* _urr_sigma_T;          /* urr micro total        xs, dim:  temp * ePoint * backXsNum */
    FP_RES_XS* _urr_sigma_S;          /* urr micro elastic      xs, dim:  temp * ePoint * backXsNum */
    FP_RES_XS* _urr_sigma_F;          /* urr micro fission      xs, dim:  temp * ePoint * backXsNum */
    FP_RES_XS* _urr_sigma_A;          /* urr micro capture      xs, dim:  temp * ePoint * backXsNum */
    
    FP_RES_XS** urr_sigma_T;          /* urr micro total        xs, dim:  temp * ePoint,  backXsNum */
    FP_RES_XS** urr_sigma_S;          /* urr micro elastic      xs, dim:  temp * ePoint,  backXsNum */
    FP_RES_XS** urr_sigma_F;          /* urr micro fission      xs, dim:  temp * ePoint,  backXsNum */
    FP_RES_XS** urr_sigma_A;          /* urr micro capture      xs, dim:  temp * ePoint,  backXsNum */
} ZRES_UGX;

/**
 * @brief ufg / hfg library
 * 
 */
typedef struct UltraFineGroupLibrary {
    int numGroup;                 /* group number */
    int refineTime;               /* re-fine times */
    int upScaGroup;               /* up scattering start group */
    int maxlg;                    /* max legendre order */
    
    int numNuclideInLib;          /* nuclide number existing in resonance library */
    int* index;                   /* finding, from library position to nuclide id */
    int* EIDD;                    /* reverse finding, from nuclide id to library position */
    FP_RES_XS* potScaXs;          /* potential scttering cross section, which used to init xs */

    int tempNum;                  /* temperature number */
    FP_RES_XS* tempList;          /* temperatures existed in resonance library */
    
    FP_RES_XS* energBoundary;     /* ufg structure boundary, hyper-fine structure is calculated with refine time */
    int* reIndex;                 /* reIndex the hfg and ufg */
    double* weight;               /* atomic weight */

    ZRES_UGX* ufgXs;              /* ultra-fine / hyper-fine group xs information */
} ZRES_UFG;

/**
 * @brief resonance library
 * 
 */
typedef struct Library {
    char* libPath;                /* resonance library path */
    int libType;                  /* resonance library type, 1: eqm, 2: subgrp method */

    ZRES_SUB subGrpLib;           /* subgroup library */
    ZRES_EQM eqmLib;              /* eqm library */
    ZRES_UFG ufgLib;              /* ultra-fine / hyper-fine group library */
} ZRES_LIB;

/**
 * @brief fsr resonance xs
 * 
 */
typedef struct FsrInfo {
    int matId;                    /* material id */
    int nextId;                   /* next fsr id, if nextId == -1, as the last one */

    FP_RES_XS* _sigma_T;          /* nuclide total xs, nuclide number x group number, below are the same */
    FP_RES_XS* _sigma_A;          /* absorpotion xs */
    FP_RES_XS* _sigma_F;          /* nuclide fission xs */
    FP_RES_XS* _sigma_vF;         /* nuclide v-fission xs */
    FP_RES_XS* _sigma_S;          /* nuclide elastic scattering xs */
} ZRES_FSR;

/**
 * @brief scattering matrix used in alpha
 * 
 */
typedef struct ScaMatrixAlpha {
    rsuint nnz;                    /* cummulative non-zero element number */
    rsuint* rowStart;              /* row start position */
    rsuint* rowEnd;                /* row end   position */
    rsuint* colStart;              /* col start position */
    rsuint  maxLayer;              /* max nnz number in layers */
    rsuint* nnzLayer;              /* cummulative nnz number for each layer */
    rsuint* nnzStartLayer;         /* scattering matrix start position for each layer */
    rsuint* nnzRowStartLayer;      /* scattering matrix start position for each layer, each layer start is 0 */
    rsuint* nnzRowEndLayer;        /* scattering matrix end   position for each layer, each layer start is 0 */
    FP_RES_XS* scaXsMx;            /* scattering matrix with csr format */

    rsuint nnzCsC;                 /* cummulative non-zero element number, but by col */
    rsuint* colStartByColGrp;      /* col start position by col and grp */
    rsuint* rowStartByColGrp;      /* row start position by col and grp */
    FP_RES_XS* scaXsMxCsc;         /* scattering matrix with csc format */
} ZRES_SMA;

/**
 * @brief general information
 * 
 */
typedef struct ZresGeneralInfo {
    char* matPath;                 /* material path */
    char* libPath;                 /* resonance xs library path */
    char* outPath;                 /* output file path, such as micro xs, macro xs */
    int libType;                   /* resonance lib type */
    int resMethod;                 /* resonance method */

    int numGroup;                  /* group number */
    int numFsr;                    /* fsr number */
    double* vol;                   /* fsr area or volume */
    int scaOrder;                  /* scatter order */

    int* fsr2matMap;               /* map between fsr and material */
    FP_RES_XS* macro_T;            /* macro total xs array, used in transport */
    FP_RES_XS* macro_A;            /* macro absorption xs array, used in transport */
    FP_RES_XS* macro_F;            /* macro fission xs array, used in transport */
    FP_RES_XS* macro_Nuf;          /* macro nu-fission xs array, used in transport */
    FP_RES_XS* chi;                /* chi array, used in transport */
    ZRES_SCM* _scatterMatrix;      /* scatter matrix information, used in transport */

    FP_RES_XS* resMacro_T;         /* macro total xs array, used in resonance */
    FP_RES_XS* resMacro_A;         /* macro absorption xs array, used in resonance */
    FP_RES_XS* resMacro_F;         /* macro fission xs array, used in resonance */
    FP_RES_XS* resMacro_Nuf;       /* macro nu-fission xs array, used in resonance */
    FP_RES_XS* resChi;             /* chi array, used in resonance */
    ZRES_SCM* _resScatterMatrix;   /* scatter matrix information, used in resonance */

    FP_RES_XS** sigma_T;           /* micro total xs array, nuclide number x fsr number */
    FP_RES_XS** sigma_A;           /* micro fission xs array */
    FP_RES_XS** sigma_F;           /* micro fission xs array */
    FP_RES_XS** sigma_vF;          /* micro nu-fission xs array */
    FP_RES_XS** sigma_S;           /* micro elastic xs array */

    int ifOutNucTot;               /* if output nuclide micro total      xs, default is 0 (False) */
    int ifOutNucAbs;               /* if output nuclide micro absorption xs */
    int ifOutNucEla;               /* if output nuclide micro elastic    xs */
    int ifOutNucFis;               /* if output nuclide micro fission    xs */
    int ifOutNucNuf;               /* if output nuclide micro nu-fission xs */

    char* grpLibPath;              /* group structure info path */
    int libResGrpNum;              /* base fg lib group number */
    double* grpEnergyBound;        /* group energy strcture boundary */
    int cgLibTypes;                /* cg library type */
    int* cgLibGrpNum;              /* cg group number */
    int** cgLibGrpBound;           /* cg group index to fg */

    int ifSlowDown;                /* if use slow down solver to construct two-level scheme, default is 0 (False) */
    int resGrpNum;                 /* resonance used group number, default is the same as tspGrpNum */
    int tspGrpNum;                 /* transport used group number */
    int sldBatchWidth;              /* slow-down batch width; defaults to tspGrpNum */
    int* tspGrpEs;                 /* transport used group structure */
    FP_RES_FX* globalChiFlx;       /* homo chi calculated flux */
    int cutOffGrp;                 /* 0~cutOffGrp use IHM flx */
    int ifOn;                      /* if open slow down solver */
    int ifHfgSlowDownOn;           /* if open hfg slow down solver coupled */
    int ifSphFactor;               /* if use sph factor to correct */
    FP_RES_XS kIHM;                /* eigenvalue calculated from global homogenized */
    FP_RES_XS* espXs;              /* escape cross, calculate by Tone's Method */

    int ifRingUmMap;               /* if use ring xs subsititute um xs */
    char* ringXsPath;              /* ring xs path */
    char* reMapPath;               /* map path between ring and um */

    int ifOutPutFsrXs;             /* if output target fs macror xs */
    char* fsrListPath;             /* target fsr list file path */

    int ifFsrCluster;              /* if enable FSR clustering at runtime */
    FP_RES_XS fsrClusterSseLimit;  /* average center SSE threshold for automatic K selection */
    int fsrClusterMaxPerMat;       /* max cluster number for each material */
    int fsrClusterMaxIter;         /* max K-Means iteration number */
    FP_RES_XS fsrClusterTolerance; /* K-Means center convergence tolerance */
    unsigned int fsrClusterSeed;   /* deterministic K-Means++ seed */
    int fsrClusterAlgorithm;       /* 0: K-Means++, 1: exact one-dimensional SSE DP */
    int ifFsrClusterSldSource;     /* derived switch for cluster-based slow-down source calculation */
    int fsrClusterSldBatchMode;    /* cluster slow-down batching mode for performance comparison */

    /**
     * below variables origined from enverinment
     */

    char* HomoChiFlxPath;          /* chi_process_cal.c */
    char* SkipOtherMat;            /* chi_process_cal.c */

    char* NucXsPath;               /* master.c */
    char* NucXsByFsrPath;          /* master.c */
    char* GrpDelta;                /* master.c */
    char* GrpLength;               /* master.c */
    char* SkipMat;                 /* master.c */
    char* OneNucXsPath;            /* master.c */
    char* OneMat;                  /* master.c */
    char* OneNuc;                  /* master.c */
    char* SaveFisXs;               /* master.c */
    char* HoFlx;                   /* master.c */
    char* HoEigen;                 /* master.c */
    char* RsPh;                    /* master.c */

    char* AppHorder;               /* scatter_matrix_process_cal_mx.c */
    char* InFlowFlux;              /* scatter_matrix_process_cal_mx.c */
    char* LoFluxApp;               /* scatter_matrix_process_cal_mx.c */
    char* HoFluxPath;              /* scatter_matrix_process_cal_mx.c */
    char* OutFlowGrpUpLim;         /* scatter_matrix_process_cal_mx.c */
    char* UseAppHoFlx;             /* scatter_matrix_process_cal_mx.c */
    char* EspXsContribution;       /* scatter_matrix_process_cal_mx.c */
    char* OutFlow;                 /* scatter_matrix_process_cal_mx.c */
    char* OutFlowXsCorrect;        /* scatter_matrix_process_cal_mx.c */
    char* TotCorSt;                /* scatter_matrix_process_cal_mx.c */
    char* TotCorEd;                /* scatter_matrix_process_cal_mx.c */
    char* TotCorrect;              /* scatter_matrix_process_cal_mx.c */
    char* SetScaXsMin;             /* scatter_matrix_process_cal_mx.c */
    char* ApplyCorTerm;            /* scatter_matrix_process_cal_mx.c */
    char* CgScaMxStorage;          /* scatter_matrix_process_cal_mx.c: dense or csr */

    char* RefK;                    /* slow_down_aux.c */
    char* FissQ;                   /* slow_down_aux.c */

    char* RefFlux;                 /* slow_down_main.c */
    char* UpScaIte;                /* slow_down_main.c */
    char* FixedSolveGrp;           /* slow_down_main.c */
    char* FixedSolverGrp;          /* slow_down_main.c */
    char* SlowDownBatchWidth;      /* slow_down_main.c */
    char* SlowDownIte;             /* slow_down_main.c */
    char* MaxMOCIteInSld;          /* slow_down_main.c */
    char* NormRmseConv;            /* slow_down_main.c */
    double _rmse;                  /* calculated rmse for normRmseConv */

    char* ZresMemLogFile;          /* main.c */
    char* LibPath;                 /* main.c */
    char* LibForm;                 /* main.c */
    char* ResMethod;               /* main.c */
    char* MatFile;                 /* main.c */
    char* OutPath;                 /* main.c */
    char* NucXsInfo;               /* main.c */
    char* SlowDown;                /* main.c */
    char* ResGrpNum;               /* main.c */
    char* SphFactor;               /* main.c */
    char* cgMapLibPath;            /* main.c */
    char* IfRingMap;               /* main.c */
    char* RingMapPath;             /* main.c */
    char* RingXsPath;              /* main.c */
    char* IfOutPutFsrXs;           /* main.c */
    char* OutPutFsrXsPath;         /* main.c */
    char* FisRate;                 /* main.c */
    char* FsrCluster;              /* master.c */
    char* FsrClusterSseLimit;      /* master.c */
    char* FsrClusterMaxPerMat;     /* master.c */
    char* FsrClusterMaxIter;       /* master.c */
    char* FsrClusterTolerance;     /* master.c */
    char* FsrClusterSeed;          /* master.c */
    char* FsrClusterAlgorithm;     /* master.c */
    char* FsrClusterSldBatchMode;  /* slow_down_main.c */
} ZRES_GI;

/**
 * @brief serval array used in fission spectrum as interface
 * 
 */
typedef struct ChiInterFace {
    ZRES_MTC _mtc;                /* material collection */
    ZRES_FSR* _resFsrInfo;        /* resonance fsr information, supply v-f */
    ZRES_LIB _lib;                /* resonance library */
    int _fsrNum;                  /* fsr number */
} ZRES_CHI_ITF;

/**
 * @brief several array used in fission spectrum calculation with GPU
 * @details h* means this array is used in host (cpu)
 *          d* means this array is used in device (gpu)
 */
typedef struct ChiArrayWithGPU
{
#ifdef CUDA_USED
    int NumGrpLocal;              /* fission spectrum group number */
    int batchsize;                /* batchsize */
    int max_iter;

    FP_RES_XS* chiMat;
    FP_RES_XS* scaMat;
    FP_RES_XS* sigmaT;

    // big array, throw these to heap
    FP_RES_XS* hF;
    FP_RES_XS* hS;
    FP_RES_XS* hT;

    FP_RES_XS* dF;
    FP_RES_XS* dS;
    FP_RES_XS* dT;

    FP_RES_FX* hFlx;
    FP_RES_FX* dFlx;
    const FP_RES_FX** dFlx_array;

    FP_RES_SS* hQf_new;
    FP_RES_SS* hQf_old;
    FP_RES_SS* dQf;

    FP_RES_SS* hQs;
    FP_RES_SS* dQs;

    FP_RES_XS** F_addr;
    FP_RES_SS** S_addr;
    FP_RES_XS** T_addr;
    
    FP_RES_FX** Flx_addr;

    FP_RES_SS** Qs_addr;
    FP_RES_SS** Qf_addr;

    // ?_addr means the pointers for each matrix included in the batch
    const FP_RES_XS** dFarray;
    const FP_RES_XS** dSarray;

    FP_RES_SS** dQf_array;
    FP_RES_SS** dQs_array;

#ifdef CUBLAS_USED
    cublasHandle_t handle;
#else
    FP_RES_SS* dQf_current;
    FP_RES_XS* dkeff;
#endif

    int* FissFSRs;

    FP_RES_XS* keff;
    FP_RES_SS* sumQf_old;
    FP_RES_SS* sumQf_new;
#endif
} ZRES_CHI_GPU;

#endif // ZRES_COMMONSTRUCT_H
