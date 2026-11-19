#ifndef GPU_CONSTANTS_H
#define GPU_CONSTANTS_H

#include <cuda_runtime.h>
#include "MOC_CommonMacro.h"

#ifdef EXP_SINGLE
  #define _ExpFunction_GPU(x) __expf(x)
#endif // EXP_SINGLE

#ifdef EXP_DOUBLE
  #define _ExpFunction_GPU(x) exp(x)
#endif // EXP_DOUBLE

#ifdef FP64
  #define _FABS_GPU(x) fabs(x)
#endif // FP64

#ifdef FP32
  #define _FABS_GPU(x) fabsf(x)
#endif // FP32

extern cudaError_t CUDAStatus;

extern __constant__ int GPUNumAzimAng[1]; /** The number of azimuthal angles */
extern __constant__ int GPUNumPolarAng[1]; /** Half the number of polar angles */
extern __constant__ int GPUNumMOCGroups[1]; /** The number of energy groups */
extern __constant__ int GPUNumLTs_Total[1];  // number of tracks in [0,2PI]
extern __constant__ int GPUTotNumTracksInOneRTM_2PI[1];
extern __constant__ int GPUPolarTimesGroups[1];/** The number of polar angles int [0,PI/2] times energy groups */
extern __constant__ FP_PRECISION GPUSinThetasTimes2[MAX_POLAR_ANGLES_GPU];
extern __constant__ FP_PRECISION GPUAzimWeights[MAX_AZIM_ANGLES_GPU];
extern __constant__ FP_PRECISION GPUPolarWeights[MAX_POLAR_ANGLES_GPU];
extern __constant__ FP_PRECISION GPUTotalWeights[MAX_POLAR_ANGLES_GPU*MAX_AZIM_ANGLES_GPU]; /** An array of the weights from the Quadrature set */

extern __constant__ int GPUMOCNumLayersZ[1],  GPUMOCNumPinsInSubDom_X[1], GPUMOCNumPinsInSubDom_Y[1];
extern __constant__ int GPUNumSurfLayer_X[1],  GPUNumSurfLayer_Y[1], GPUNumSurfLayer_Z[1];

extern __constant__ int GPUNumScattOrder[1];
extern __constant__ FP_PRECISION GPUCoeffA_HOScatt[6*6];
/* boundary condition */
extern __constant__ int GPUNorthBC[1],GPUSouthBC[1],GPUEastBC[1],GPUWestBC[1],GPUBottomBC[1],GPUTopBC[1];

extern __constant__ FP_PRECISION GPUSinThetaForExp[MAX_POLAR_ANGLES_GPU];

extern __constant__ FP_PRECISION GPUSinThetas[MAX_POLAR_ANGLES_GPU]; /** An array for the sines of the polar angle in the Quadrature set */
extern __constant__ FP_PRECISION GPUCosThetas[MAX_POLAR_ANGLES_GPU];
extern __constant__ FP_PRECISION GPUOneOverSinThetas[MAX_POLAR_ANGLES_GPU];

extern __constant__ FP_PRECISION_GEO GPUEffRaySpacing[MAX_AZIM_ANGLES_GPU];

extern __constant__ int GPUNumPinsInSubDomPerLayer[1];

extern __constant__ int GPUTotNumFSRs[1]; /** The number of FSRs */
extern __constant__ int GPUNumFSRsLayer[1];
extern __constant__ int GPUNumLTs_HS[1];  // number of tracks in [0,PI]
extern __constant__ int GPUTotNumTracksInOneRTM_HS[1];
//extern __constant__ FP_PRECISION GPUGlobFissRate[1];
extern __constant__ int GPUTotNumPins[1]; /** The number of pins */
extern __constant__ int GPUTotalNumFSRsInRTM[1];

extern __constant__ int GPU_angMul1_HP[1];
extern __constant__ int GPU_angMul2_HP[1];

extern __constant__ int GPU_angMul1_FP[1];
extern __constant__ int GPU_angMul2_FP[1];

#define GPU_Gmres_threadsPerBlock 128
#define GPU_CGGmres_threadsPerBlock 512

extern FP_PRECISION_TIME GPU_FISS_CALC_TIME, GPU_SCAT_CALC_TIME, GPU_XS_TRAN_TIME, GPU_TOTAL_SOURCE_CALC_TIME;

extern M_INT32 GPUBlockDim, GPUNumStreams;

void CUDA_CheckError(char *_funcName);
void CUDA_CheckError_Gmres(char *_funcName);

#endif // GPU_CONSTANTS_H
