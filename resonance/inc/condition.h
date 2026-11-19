
#ifndef CONDITION_H
#define CONDITION_H

#ifdef OMP_USED
    #include "omp.h"
#endif

// #ifndef TIME_RECORD
//     #define TIME_RECORD (FP_RES_XS)clock()
// #endif

// #define LIUDONG_UFG_1D_SLAB

// #define LIUDONG_UFG_1D_CYCL

// #define LIB_MAKER

// for debug
//#ifndef CUDA_USED
//    #define CUDA_USED
//#endif

// for tally time in each part
//#define Tally_Time

#define ALPHA_SUPPORT

// old 10-format subgroup library
//#define ZRES_SUB_OLD

// HFG tone test
#define ZRES_UFG_TEST
#define ZRES_UFG_REME
#define ZRES_UFG_TDB

// homo problem to get fission spectrum
// #define HOMO_CHI
//#define OUTPUT_HOMO_FLX

// re-normalized scatter prob.
//#define REN_SP

// for 5 ev
// #define NORD_TSP

// direct use fission matrix, but only support very small problem
// #define FIS_MTX

// output set
#define TESTO

// project-o
// #define PROJECT_SUPPORT

// for generate nuclide-base xs for NPIC Pb-Bi project
// #define NPIC_PBI_SUPPORT

#define STABLE_SCAMX
//#define TEST_SCAMX

// #define SCAMX_1FOR
#define SCAMX_NEW_CYC

// for high order outflow
#define HIGH_ORDER_OTF

#define OUTPUT_CG_SCAMX

#define FORSCI2

#define GPU_SLOWD

// #define SLOWD_SOURCE_1
//#define SLOWD_SOURCE_2  // failed
//#define SLOWD_SOURCE_3
//#define SLOWD_SOURCE_4  // imcomplete
// #define SLOWD_SOURCE_5  // very stable, but a little slow and memory cosuming, use with SLOWD_PLAN3
#define SLOWD_SOURCE_6 // use with SLOWD_PLAN4

// #define SLOWD_PLAN1 // construct and save scatter matrix, split, init version
// #define SLOWD_PLAN2 // construct and save scatter matrix, together
// #define SLOWD_PLAN3 // save and load cg * fg scattering matrix
#define SLOWD_PLAN4 // save and load cg * cg scattering matrix, the scatter source from other grp are calculated in CPU

#define SLOWD_1FOR

#define CHI_MAT_CYC

#define ZRES_THREAD_NUM 8

// for testing resonance elastic xs
// #define TEST_RES_ELA

// for testing only SPH Abs Else
// #define TEST_SPH_ABS

// #define FORPBEFF

// for angle scalar flux
// #define SAVE_ANGLE_FLUX
#define SAVE_ORDER 3

// #define FIX_SOLVER

// set source == nuf, used to batch calculate flux moment
// #define HO_FIXED_SOURCE

// set source == chi, flatten fission source assumption
// #define FLAT_FIS_SOURCE

// split flux used in fission source and scatter source
#define SPLIT_FIS_S_SCA_S

// 
// #define EARLY_BREAK

// calculate source with given flux and flatten fission sourcew
#define FLAT_FIS_SOURCE_CS

// test the influence from up-scatter during SI
// #define OMIT_UP_SCATT

// if use huge page
#define USE_HUGE_PAGE

#endif // CONDITION_H
