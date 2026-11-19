#ifndef R_Control_H
#define R_Control_H

#define NUC_NUM_HAS_XS 218  // for SHEM
#define IF_SAVE_RES_XS 1
#define IF_TRANS_CORREC 1
#define LIBRARY_47 "Lib_47b.lib" // for SHEM overlap
#define IF_WRITE_LIB 1
#define IF_WRITE_LIB_SHEM 1
#define IF_LIB_PATH 1
//#define FIX_SRC 1
//#define EIG_VAL 0
//#define IF_WRITE_SUBDATA 0
//#define IF_FIXED_SRC 1
#define IF_Tone 0
//#define IF_Inter 2
//#define bdite 20

#define IF_P0P1 0  // if read Pn Inform // if correct self-scatter xs
#define IF_GAMMA_DATA 0
#define URCUT 11.513
#define PAI 3.141592654
// #define NCATE 45 // resonance nuclide num

#define LINE_N 1024
#define NAME_N 512
// #define MAT_N 10

#define TINY 1E-20
#define FLUX_MAX 0.999
#define XEMIN 1E-4
#define ITER_N 30
#define MIN_ERR  0.001 
#define ITMAX  30
#define REXALATION  0.5 

#define TEMPNN 10
#define E0 10E6
#define SUBNN 5 // used for SHEM

#define NGUASS 8
#define MAX_NUCID 99999

// #define NRES_HLS 45
// #define NSUBG 8   
// #define NFLX 5
#define SIZE_LIB_N 92 // title index
#define INFCHX_N1 3   // constant
#define NF 14              // chi group number
#define SIZE_GAM_N 9       // for gamma lib read
// #define ISTART_N 2085     // 3432 * 3, Size of XTEMP() and ISTART().
// #define XSDATA_N 7540     // Maximum size of XSDATA() in any group.
#define IPNSTA_N 147      // 3431 * 3
#define PNDATA_N 1000      // Pn data max length
#define IPNSTA_PNDATA_N 47 // group number
// #define PN_N 3
#define GSTART_N 584     // for gamma lib read
#define IG2 1368         // for gamma lib read
#define NGSTAR_N 1398    // for gamma lib read
#define IG3 17167        // for gamma lib read
#define NAT_ISO_INF_N 21 // for nature nuclide inform

#define NucNumbersInWorld 3399 // Nuclide Number in world as of sep. 2021

// #define I0 1024725
// #define I1 850
// #define I2 756
// #define I3 3442
// #define I4 695
// #define I5 0
// #define I7 7540
// #define I8 49
// #define I10 938
// #define INDRES 10914
// #define INDXSD 226229
// #define INDBUR 602119
// #define INDP1M -608789
// #define NELT_HLS 295
// #define NELR_HLS 189
// #define NBUR 167
// #define NBA 0
// #define NFIS 36
// #define NP1 25
// #define INDN2N 1602
// #define INDN3N 7226
// #define N2N 118
// #define N3N 76
// #define INDCHX 1420
// #define NCHI 10
// #define NCHIX 14
// #define INDP2M 637019
// #define INDP3M 666467
// #define INDSG 168005
// #define INDRSC 79067
// #define IR3 5508
// #define NXSG 378
// #define NWSG 1589
// #define NWASGF 736
/** Constant */
#endif
