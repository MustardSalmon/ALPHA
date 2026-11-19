//
// Created by hangZou on 2022/7/14.
//

#ifndef ALPHA_AUXILIARY_H
#define ALPHA_AUXILIARY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MOC_CommonMacro.h"

#ifdef SCHED_LINUX
#include <sys/sysinfo.h>
#include "MOC_CommonMacro.h"
#endif // SCHED_LINUX

#define MAX_TEXT_LINE 1024

void checkSysMemUsage();

void collpase0D(int* CGstructure, \
                FP_PRECISION* SourceArray, \
                FP_PRECISION* TargetArray);

void collpase1D(int* CGstructure, \
                FP_PRECISION* SourceArray, \
                FP_PRECISION* WeightArray, \
                FP_PRECISION* TargetArray);

void collpase2D(int* CGstructure, \
                FP_PRECISION* SourceArray, \
                int* FG_NZRowStartID, \
                int* FG_NZColStartID, \
                FP_PRECISION* WeightArray, \
                FP_PRECISION* TargetArray, \
                int* CG_NZRowStartID, \
                int* CG_NZColStartID);

void calScaXs(FP_PRECISION* ScaXs, \
              int fsrID, \
              int GrpNum, \
              int* RowStartID, \
              int* ColStartID, \
              FP_PRECISION** ScaMatrixDense);

char *cleanComments(char *_src_inp_name, char *_dst_inp_name, char _comment_flag);

#endif //ALPHA_AUXILIARY_H
