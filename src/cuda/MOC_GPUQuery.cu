/********************************************************************************/
/* @ File: MOC_GPUQuery.c
/*   introduction: This file contains functions that query the properties of the
/*                 CUDA devices present in the system via CUDA Runtime API
/* @ Created     : 2018/05/15, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*   1st modified:
/********************************************************************************/
extern "C" {
    #include "MOC_GPUQuery.cuh"
}
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <device_launch_parameters.h>
#include "MOC_CommonMacro.h"
#include "MOC_Input.h"
//#ifdef MPI_USED
//#include <mpi.h>
//#endif
cudaError_t CUDAStatus_Query;
cudaDeviceProp DeviceProp;

extern char ProcessorName[ParaNameLength];

void PrintCUDADeviceCacheConfig(cudaFuncCache _pCacheConfig);

/**
 * @ Function: GPUQuery()
 *   introduction: query the GPU device information present in the system
 * @ Parameter List: void
 *   Return: void
 * @ Created     : 2018/05/15, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
extern "C"
/** translate cache configuration*/
void GPUQuery_1(int _procID, int _deviceID)
{
    int deviceCount,dev,driverVersion=0,runtimeVersion=0/*,memoryClock,memBusWidth,L2CacheSize*/;
    // cudaDeviceProp deviceProp;
	cudaFuncCache deviceCacheConfig;
    char msg[256];
    deviceCount = 0;
    if (_procID==0) {
    printf("\n [GPU-INFO ]  Process %d, CUDA Device Query (Runtime API) version (CUDART static linking)",_procID);
    fprintf(LogFile, "\n [GPU-INFO ]  Process %d, CUDA Device Query (Runtime API) version (CUDART static linking)",_procID);
    }
    // get the cuda device count
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        if (_procID==0) {
        printf("_procID=%d, cudaGetDeviceCount returned %d\n %s\n",_procID,  (int)error_id, cudaGetErrorString(error_id));
        fprintf(LogFile, "_procID=%d, cudaGetDeviceCount returned %d\n %s\n",_procID, (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        fprintf(LogFile, "Result = FAIL\n");
        }
        //exit(EXIT_FAILURE);
		exit(1);
    }
    // This function call returns 0 if there are no CUDA capable devices.
    if (_procID==0) {
    if (deviceCount == 0) {
        printf("\n [  INFO   ]  There are no available device(s) that support CUDA");
        fprintf(LogFile, "\n [  INFO   ]  There are no available device(s) that support CUDA");
    }
    else {
        printf("\n              Detected %d CUDA Capable device(s) in Current Node", deviceCount);
        fprintf(LogFile, "\n              Detected %d CUDA Capable device(s) in Current Node", deviceCount);
    }
    }
    dev=_deviceID;
    // for (dev = 0; dev < deviceCount; dev++) {
    cudaGetDeviceProperties(&DeviceProp, dev);
    if (_procID==0) {
        printf("\n              Device %d: \"%s\"", dev, DeviceProp.name);
        fprintf(LogFile, "\n              Device %d: \"%s\"", dev, DeviceProp.name);
        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("\n                CUDA Driver Version / Runtime Version          %d.%d / %d.%d", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        fprintf(LogFile, "\n                CUDA Driver Version / Runtime Version          %d.%d / %d.%d", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("\n                CUDA Capability Major/Minor version number:    %d.%d", DeviceProp.major, DeviceProp.minor);
        fprintf(LogFile, "\n                CUDA Capability Major/Minor version number:    %d.%d", DeviceProp.major, DeviceProp.minor);
        sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)",
            (float)DeviceProp.totalGlobalMem/1048576.0f, (unsigned long long) DeviceProp.totalGlobalMem); 
        printf("\n              %s", msg);
        fprintf(LogFile, "\n              %s", msg);
        printf("\n                (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores",
            DeviceProp.multiProcessorCount,
            ConvertSMVer2Cores(DeviceProp.major, DeviceProp.minor),
            ConvertSMVer2Cores(DeviceProp.major, DeviceProp.minor) * DeviceProp.multiProcessorCount);
            fprintf(LogFile, "\n                (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores",
            DeviceProp.multiProcessorCount,
            ConvertSMVer2Cores(DeviceProp.major, DeviceProp.minor),
            ConvertSMVer2Cores(DeviceProp.major, DeviceProp.minor) * DeviceProp.multiProcessorCount);
        printf("\n                GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)", DeviceProp.clockRate * 1e-3f, DeviceProp.clockRate * 1e-6f);
        fprintf(LogFile, "\n                GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)", DeviceProp.clockRate * 1e-3f, DeviceProp.clockRate * 1e-6f);
#if CUDART_VERSION >= 5000
        // This is supported in CUDA 5.0 (runtime API device properties)
        printf("\n                Memory Clock rate:                             %.0f Mhz", DeviceProp.memoryClockRate * 1e-3f);
        fprintf(LogFile, "\n                Memory Clock rate:                             %.0f Mhz", DeviceProp.memoryClockRate * 1e-3f);
        printf("\n                Memory Bus Width:                              %d-bit",   DeviceProp.memoryBusWidth);
        fprintf(LogFile, "\n                Memory Bus Width:                              %d-bit",   DeviceProp.memoryBusWidth);
        if (DeviceProp.l2CacheSize) {
            printf("\n                L2 Cache Size:                                 %d bytes", DeviceProp.l2CacheSize);
            fprintf(LogFile, "\n                L2 Cache Size:                                 %d bytes", DeviceProp.l2CacheSize);
        }
#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("\n                Memory Clock rate:                             %.0f Mhz", memoryClock * 1e-3f);
        fprintf(LogFile, "\n                Memory Clock rate:                             %.0f Mhz", memoryClock * 1e-3f);
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        printf("\n                Memory Bus Width:                              %d-bit", memBusWidth);
        fprintf(LogFile, "\n                Memory Bus Width:                              %d-bit", memBusWidth);
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);
        if (L2CacheSize) {
            printf("\n                L2 Cache Size:                                 %d bytes", L2CacheSize);
            fprintf(LogFile, "\n                L2 Cache Size:                                 %d bytes", L2CacheSize);
        }
#endif
        printf("\n                Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)",
            DeviceProp.maxTexture1D   , DeviceProp.maxTexture2D[0], DeviceProp.maxTexture2D[1],
            DeviceProp.maxTexture3D[0], DeviceProp.maxTexture3D[1], DeviceProp.maxTexture3D[2]);
        printf("\n                Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers",
            DeviceProp.maxTexture1DLayered[0], DeviceProp.maxTexture1DLayered[1]);
        printf("\n                Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers",
            DeviceProp.maxTexture2DLayered[0], DeviceProp.maxTexture2DLayered[1], DeviceProp.maxTexture2DLayered[2]);
        printf("\n                Total amount of constant memory:               %llu bytes", (unsigned long long)DeviceProp.totalConstMem);
        printf("\n                Total amount of shared memory per block:       %llu bytes", (unsigned long long)DeviceProp.sharedMemPerBlock);
        printf("\n                Total number of registers available per block: %d", DeviceProp.regsPerBlock);
        printf("\n                Warp size:                                     %d", DeviceProp.warpSize);
        printf("\n                Maximum number of threads per multiprocessor:  %d", DeviceProp.maxThreadsPerMultiProcessor);
        printf("\n                Maximum number of threads per block:           %d", DeviceProp.maxThreadsPerBlock);
        printf("\n                Max dimension size of a thread block (x,y,z): (%d, %d, %d)",
            DeviceProp.maxThreadsDim[0],
            DeviceProp.maxThreadsDim[1],
            DeviceProp.maxThreadsDim[2]);
        printf("\n                Max dimension size of a grid size    (x,y,z): (%d, %d, %d)",
            DeviceProp.maxGridSize[0],
            DeviceProp.maxGridSize[1],
            DeviceProp.maxGridSize[2]);
        printf("\n                Maximum memory pitch:                          %llu bytes", (unsigned long long)DeviceProp.memPitch);
        printf("\n                Texture alignment:                             %llu bytes", (unsigned long long)DeviceProp.textureAlignment);
        printf("\n                Concurrent copy and kernel execution:          %s with %d copy engine(s)", (DeviceProp.deviceOverlap ? "Yes" : "No"), DeviceProp.asyncEngineCount);
        printf("\n                Run time limit on kernels:                     %s", DeviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("\n                Integrated GPU sharing Host Memory:            %s", DeviceProp.integrated ? "Yes" : "No");
        printf("\n                Support host page-locked memory mapping:       %s", DeviceProp.canMapHostMemory ? "Yes" : "No");
        printf("\n                Alignment requirement for Surfaces:            %s", DeviceProp.surfaceAlignment ? "Yes" : "No");
        printf("\n                Device has ECC support:                        %s", DeviceProp.ECCEnabled ? "Enabled" : "Disabled");



        fprintf(LogFile, "\n                Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)",
            DeviceProp.maxTexture1D   , DeviceProp.maxTexture2D[0], DeviceProp.maxTexture2D[1],
            DeviceProp.maxTexture3D[0], DeviceProp.maxTexture3D[1], DeviceProp.maxTexture3D[2]);
        fprintf(LogFile, "\n                Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers",
            DeviceProp.maxTexture1DLayered[0], DeviceProp.maxTexture1DLayered[1]);
        fprintf(LogFile, "\n                Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers",
            DeviceProp.maxTexture2DLayered[0], DeviceProp.maxTexture2DLayered[1], DeviceProp.maxTexture2DLayered[2]);
        fprintf(LogFile, "\n                Total amount of constant memory:               %llu bytes", (unsigned long long)DeviceProp.totalConstMem);
        fprintf(LogFile, "\n                Total amount of shared memory per block:       %llu bytes", (unsigned long long)DeviceProp.sharedMemPerBlock);
        fprintf(LogFile, "\n                Total number of registers available per block: %d", DeviceProp.regsPerBlock);
        fprintf(LogFile, "\n                Warp size:                                     %d", DeviceProp.warpSize);
        fprintf(LogFile, "\n                Maximum number of threads per multiprocessor:  %d", DeviceProp.maxThreadsPerMultiProcessor);
        fprintf(LogFile, "\n                Maximum number of threads per block:           %d", DeviceProp.maxThreadsPerBlock);
        fprintf(LogFile, "\n                Max dimension size of a thread block (x,y,z): (%d, %d, %d)",
            DeviceProp.maxThreadsDim[0],
            DeviceProp.maxThreadsDim[1],
            DeviceProp.maxThreadsDim[2]);
        fprintf(LogFile, "\n                Max dimension size of a grid size    (x,y,z): (%d, %d, %d)",
            DeviceProp.maxGridSize[0],
            DeviceProp.maxGridSize[1],
            DeviceProp.maxGridSize[2]);
        fprintf(LogFile, "\n                Maximum memory pitch:                          %llu bytes", (unsigned long long)DeviceProp.memPitch);
        fprintf(LogFile, "\n                Texture alignment:                             %llu bytes", (unsigned long long)DeviceProp.textureAlignment);
        fprintf(LogFile, "\n                Concurrent copy and kernel execution:          %s with %d copy engine(s)", (DeviceProp.deviceOverlap ? "Yes" : "No"), DeviceProp.asyncEngineCount);
        fprintf(LogFile, "\n                Run time limit on kernels:                     %s", DeviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        fprintf(LogFile, "\n                Integrated GPU sharing Host Memory:            %s", DeviceProp.integrated ? "Yes" : "No");
        fprintf(LogFile, "\n                Support host page-locked memory mapping:       %s", DeviceProp.canMapHostMemory ? "Yes" : "No");
        fprintf(LogFile, "\n                Alignment requirement for Surfaces:            %s", DeviceProp.surfaceAlignment ? "Yes" : "No");
        fprintf(LogFile, "\n                Device has ECC support:                        %s", DeviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        printf("\n                CUDA Device Driver Mode (TCC or WDDM):         %s", DeviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
        fprintf(LogFile, "\n                CUDA Device Driver Mode (TCC or WDDM):         %s", DeviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
        printf("\n                Device supports Unified Addressing (UVA):      %s", DeviceProp.unifiedAddressing ? "Yes" : "No");
        fprintf(LogFile, "\n                Device supports Unified Addressing (UVA):      %s", DeviceProp.unifiedAddressing ? "Yes" : "No");
        // printf("\n                Supports Cooperative Kernel Launch:            %s", DeviceProp.cooperativeLaunch ? "Yes" : "No");
        // printf("\n                Supports MultiDevice Co-op Kernel Launch:      %s", DeviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
        printf("\n                Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d", DeviceProp.pciDomainID, DeviceProp.pciBusID, DeviceProp.pciDeviceID);
        fprintf(LogFile, "\n                Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d", DeviceProp.pciDomainID, DeviceProp.pciBusID, DeviceProp.pciDeviceID);

		cudaDeviceGetCacheConfig(&deviceCacheConfig);
		PrintCUDADeviceCacheConfig(deviceCacheConfig);
		//cudaDeviceGetLimit();
    // }
    printf("\n              GPU Query Finished, Result = PASS!");
    fprintf(LogFile, "\n              GPU Query Finished, Result = PASS!");
    }
}


/**
 * @ Function: ConvertSMVer2Cores()
 *   introduction: get the number of CUDA Cores per multiProcessor
 * @ Parameter List: _major - major compute capability
 *                   _minor - minor compute capability
 *   Return: the number of CUDA Cores of current multiProcessor
 * @ Created     : 2018/05/15, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
// extern "C"
int ConvertSMVer2Cores(int _major, int _minor)
{
    int _index;
    sSMtoCores _nGpuArchCoresPerSM[14] = {
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
        { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
        { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
        { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
        { 0x70, 64 }, // Volta Generation (SM 7.0) GV100 class
        { 0x75, 64 }, // Volta Generation (SM 7.5) turing
        { 0x86, 128 }, // Volta Generation (SM 7.5) ampere
        {   -1, -1 }
    };
    _index = 0;
    while (_nGpuArchCoresPerSM[_index].SM != -1) {
        if (_nGpuArchCoresPerSM[_index].SM == ((_major << 4) + _minor))
        {
            return _nGpuArchCoresPerSM[_index].Cores;
        }
        _index++;
    }
    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", _major, _minor, _nGpuArchCoresPerSM[_index-1].Cores);
    return _nGpuArchCoresPerSM[_index-1].Cores;
}

/**
 * @brief .
 * @return True if the node contains a GPU, false otherwise.
 */
/**
 * @ Function: MachineContainsGPU()
 *   introduction: queries a node to determine whether it contains one or more GPUs
 * @ Parameter List: void
 *   Return: void
 * @ Created     : 2018/05/15, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */

 
extern "C"
void InitCudaDevice(int _CPUProcID,int _numProcs, char *_ProcessorName)
{
    cudaDeviceProp deviceProp;
    if (_CPUProcID == 0) {
		printf("\n [GPU-INFO ]  Initializing CUDA Device ...");
        fflush(stdout);
		fprintf(LogFile, "\n [GPU-INFO ]  Initializing CUDA Device ...");        
	}

    cudaSetDevice(CUDADeviceID[_CPUProcID]/* % NumGPUsPerNode*/);
    cudaGetDeviceProperties(&deviceProp, CUDADeviceID[_CPUProcID]);
    printf("\n              Process %3d is driving device %3d (\"%s\") on node %s",_CPUProcID, CUDADeviceID[_CPUProcID], deviceProp.name, _ProcessorName);
    if (_CPUProcID==0) {
        fprintf(LogFile, "\n              Process %3d is driving device %3d (\"%s\") on node %s",_CPUProcID, CUDADeviceID[_CPUProcID], deviceProp.name, _ProcessorName);
    }
    
    // printf("\n ProcID=%d, GpuID=%d",_CPUProcID, CUDADeviceID[_CPUProcID]);
    // if (_CPUProcID==0) {
    //     GPUQuery_1(_CPUProcID);
    // }
    
#ifdef CUDAErrorCheck
	CUDAStatus_Query = cudaGetLastError();
	if (CUDAStatus_Query != cudaSuccess) {
		printf("\n [GPU-ERROR]  Lunch Failed: InitCudaDevice with Info: %s\n", cudaGetErrorString(CUDAStatus_Query));
		exit(69);
	}
#endif // CUDAErrorCheck
}

__global__ void GPUWhatIsMyID(unsigned int * const _block,
                           unsigned int * const _thread,
                           unsigned int * const _warp,
                           unsigned int * const _calcThread)
{
    /* thread id is block index * block size + thread offset in this block */
    const unsigned int _threadIdx=(blockIdx.x * blockDim.x) + threadIdx.x;
    _block[_threadIdx]=blockIdx.x;
    _thread[_threadIdx]=threadIdx.x;
    /* calculate warp using build in variable warpSize */
    _warp[_threadIdx]=threadIdx.x/warpSize;
    _calcThread[_threadIdx]=_threadIdx;
}

void GPUMemUsage(int _procID, FP_PRECISION_TIME *_cudaMemUse, FP_PRECISION_TIME *_cudaMemTot)
{
    size_t _avail, _total;
    cudaMemGetInfo(&_avail, &_total);
    _cudaMemTot[0] = (FP_PRECISION_TIME)(_total) * M_MB_Unit;
    _cudaMemUse[0] = (FP_PRECISION_TIME)(_total - _avail) * M_MB_Unit;
}

void PrintCUDADeviceCacheConfig(cudaFuncCache _pCacheConfig)
{
	if (_pCacheConfig == 0) {
		printf("\n                CUDA Device Cache Config is cudaFuncCachePreferNone");
	}
	else if (_pCacheConfig == 1) {
		printf("\n                CUDA Device Cache Config is cudaFuncCachePreferShared");
	}
	else if (_pCacheConfig == 2) {
		printf("\n                CUDA Device Cache Config is cudaFuncCachePreferL1");
	}
	else if (_pCacheConfig == 3) {
		printf("\n                CUDA Device Cache Config is cudaFuncCachePreferEqual");
	}
	else {}
}
