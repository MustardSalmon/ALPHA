#ifndef GPUQUERY_H
#define GPUQUERY_H
#include <assert.h>
#include "MOC_Input.h"
#include "MOC_Output.h"
#include "MOC_CommonMacro.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
//#ifdef MPI_USED
//  #include <mpi.h>
//#endif

//#define NumGPUsPerNode 2

// extern cudaDeviceProp DeviceProp;

 /* define macro for error check when calling cuda function */
#define CUDA_CALL(call) {cudaError_t err=call;if (err!=cudaSuccess) {\
    printf("\n [CudaError]  CUDA ERROR: %s (err_num=%d)\n", cudaGetErrorString(err),err);cudaDeviceReset();assert(0);}}

// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
struct sSMtoCores {
    int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
    int Cores;
};
void GPUQuery();
int ConvertSMVer2Cores(int _major, int _minor);
void MachineContainsGPU();
void CudaMpiTest(int _procID, int _numProcs);
void HelloCUDA(char _strGPUID[]);
void InitCudaDevice(int _CPUProcID,int _numProcs, char *_ProcessorName);
void WhatIsMyID();
void GPUMemUsage(int _procID, double *_cudaMemUse, double *_cudaMemTot);
void GPUQuery_1(int _CPUProcID, int _procID);
//void PrintCUDADeviceCacheConfig(cudaFuncCache _pCacheConfig);
#endif