/********************************************************************************/
/* @ File: MOC_GPUConstants.cu
/*   introduction: This file contains the defination of constants variables used 
/*                 in multiple *.cu
/* @ Created     : 2019/08/14, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*   1st modified:
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include "MOC_CommonMacro.h"

cudaError_t CUDAStatus;

__constant__ int GPUNumAzimAng[1]; /** The number of azimuthal angles */
__constant__ int GPUNumPolarAng[1]; /** Half the number of polar angles */
__constant__ int GPUNumMOCGroups[1]; /** The number of energy groups */
__constant__ int GPUNumLTs_Total[1];  // number of tracks in [0,2PI]
__constant__ int GPUTotNumTracksInOneRTM_2PI[1];
__constant__ int GPUPolarTimesGroups[1];/** The number of polar angles int [0,PI/2] times energy groups */
__constant__ FP_PRECISION GPUSinThetasTimes2[MAX_POLAR_ANGLES_GPU];
__constant__ FP_PRECISION GPUAzimWeights[MAX_AZIM_ANGLES_GPU];
__constant__ FP_PRECISION GPUPolarWeights[MAX_POLAR_ANGLES_GPU];
__constant__ FP_PRECISION GPUTotalWeights[MAX_POLAR_ANGLES_GPU*MAX_AZIM_ANGLES_GPU]; /** An array of the weights from the Quadrature set */
__constant__ int GPUNumScattOrder[1];
__constant__ FP_PRECISION GPUCoeffA_HOScatt[6*6];

__constant__ int GPUMOCNumLayersZ[1],  GPUMOCNumPinsInSubDom_X[1], GPUMOCNumPinsInSubDom_Y[1];
__constant__ int GPUNumSurfLayer_X[1],  GPUNumSurfLayer_Y[1], GPUNumSurfLayer_Z[1];

__constant__ int GPUNorthBC[1],GPUSouthBC[1],GPUEastBC[1],GPUWestBC[1],GPUBottomBC[1],GPUTopBC[1];

__constant__ FP_PRECISION GPUSinThetaForExp[MAX_POLAR_ANGLES_GPU];

__constant__ FP_PRECISION GPUSinThetas[MAX_POLAR_ANGLES_GPU]; /** An array for the sines of the polar angle in the Quadrature set */
__constant__ FP_PRECISION GPUCosThetas[MAX_POLAR_ANGLES_GPU];
__constant__ FP_PRECISION GPUOneOverSinThetas[MAX_POLAR_ANGLES_GPU];

__constant__ FP_PRECISION_GEO GPUEffRaySpacing[MAX_AZIM_ANGLES_GPU];

__constant__ int GPUNumPinsInSubDomPerLayer[1];

__constant__ int GPUTotNumFSRs[1]; /** The number of FSRs */
__constant__ int GPUNumFSRsLayer[1]; /** The number of FSRs per layer*/
__constant__ int GPUNumLTs_HS[1];  // number of tracks in [0,PI]
__constant__ int GPUTotNumTracksInOneRTM_HS[1];
//__constant__ FP_PRECISION GPUGlobFissRate[1];
__constant__ int GPUTotNumPins[1]; /** The number of pins */
__constant__ int GPUTotalNumFSRsInRTM[1];

__constant__ int GPU_angMul1_HP[1];
__constant__ int GPU_angMul2_HP[1];

__constant__ int GPU_angMul1_FP[1];
__constant__ int GPU_angMul2_FP[1];

FP_PRECISION_TIME GPU_FISS_CALC_TIME=0.0, GPU_SCAT_CALC_TIME=0.0, GPU_XS_TRAN_TIME=0.0, GPU_TOTAL_SOURCE_CALC_TIME=0.0;

M_INT32 GPUBlockDim=128, GPUNumStreams=1;

/**
* @ Function: CUDA_CheckError()
*   introduction: check the cuda invoking error
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/27, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void CUDA_CheckError(char *_funcName)
{
#ifdef CUDAErrorCheck
    cudaDeviceSynchronize();
	CUDAStatus = cudaGetLastError();
	if (CUDAStatus != cudaSuccess) {
		printf("\n [GPU-ERROR]  Lunch Failed: %s with Info: %s\n",_funcName, cudaGetErrorString(CUDAStatus));
		exit(69);
	}
#endif // CUDAErrorCheck
}

void CUDA_CheckError_Gmres(char *_funcName)
{
#ifdef CUDAErrorCheck
	cudaDeviceSynchronize();
	CUDAStatus = cudaGetLastError();
	if (CUDAStatus != cudaSuccess) {
		printf("\n [GPU-ERROR]  Lunch Failed: %s with Info: %s\n", _funcName, cudaGetErrorString(CUDAStatus));
		exit(69);
	}
#endif // CUDAErrorCheck
}