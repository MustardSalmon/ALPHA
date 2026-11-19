/* Minimal GPU state required by the external-XS transport path. */
#include "MOC_GPUCmfd.cuh"

FP_PRECISION *GPUTotalWeights_2DMOC = NULL;
FP_PRECISION *GPUSpheHarmCoeff = NULL;
int *GPUNumTracksX = NULL, *GPUNumTracksY = NULL;
FP_PRECISION_GEO *GPURTMWidth_X = NULL, *GPURTMWidth_Y = NULL;
FP_PRECISION_GEO *GPUWidthLayersInCore = NULL;
int *GPUNumFSRsInRTM = NULL;
FP_PRECISION_GEO *GPUFSRsAreaInRTM_AntiClockFSR = NULL;
FP_PRECISION_GEO *GPUFSRArea_SubDom = NULL;

__constant__ FP_PRECISION_GEO GPURayTracModDim[3];
__constant__ int GPUNumSurfaceX[1];
__constant__ int GPUNumSurfaceY[1];
