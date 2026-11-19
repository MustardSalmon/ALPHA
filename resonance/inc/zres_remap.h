
#ifndef ZRES_REMAP_H
#define ZRES_REMAP_H

/**
 * @brief ring to fsr resonance xs remap
 * current only support alpha resonance (ringed) to alpha resonance (um)
 * 
 */
typedef struct ReMapCollect {
    int numRing;                     /* ring number */
    int* ringFsrIds;                 /* fsr index in ringed geometry */
    int* ringFsrNum;                 /* 1d array, first numRing save ringFsrHomo Num, second numRing save um Fsr Num */
    int** ringFsrHomo;               /* ring list needed to average */
    int** umFsrIds;                  /* um index in normal geometry */
} ZREM_COL;

#endif // ZRES_REMAP_H
