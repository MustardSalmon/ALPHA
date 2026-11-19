#ifndef GPUVECTOR_H
#define GPUVECTOR_H
#include <cuda_runtime.h>
#include "MOC_CommonMacro.h"
#include "cublas_v2.h"

__global__ void GPUVectorInit_FP_I(FP_PRECISION *_vec, int _dim, FP_PRECISION _value);
__global__ void GPUVectorCopy_FP_I(FP_PRECISION *_vecOld, FP_PRECISION *_vecNew, int _dim);
__global__ void GPUGetRelErrOfTwoVectors_FP_I(FP_PRECISION *_vecNew, FP_PRECISION *_vecOld, FP_PRECISION *_relErr, int _dim);
__global__ void GPUMatrixVecProduct_CSR(int _dim, int *_ia, int *_ja, \
	FP_PRECISION *_a, FP_PRECISION *_x, FP_PRECISION *_w);
__global__ void GPUMatrixVecResidual_CSR(int _dim, int *_ia, int *_ja, \
	FP_PRECISION *_a, FP_PRECISION *_x, FP_PRECISION *_b, FP_PRECISION *_res);
FP_PRECISION GPUComputeRMSE_Vector(int _dim, int _GPUGridDim, int _GPUBlockDim, \
	FP_PRECISION *_vec_x, FP_PRECISION *_vec_y, FP_PRECISION *_vecErr, cublasHandle_t _handle);
__global__ void GPU_MGMatrixVecSumSqr_CSR(int _dim, FP_PRECISION *_y,FP_PRECISION *_mat, int *_colInd,
    int *_rowInd, FP_PRECISION *_x, FP_PRECISION *_b,FP_PRECISION *_dotRes);
__global__ void GPU_VecInit_FP_1(int _dim,FP_PRECISION *_vec);
__global__ void GPU_VecNorm_Reduction(int blocks,FP_PRECISION *dev_res, FP_PRECISION *red_res,int choice);
__global__ void GPU_VecVecDot(int _dim,FP_PRECISION *_vec_a, FP_PRECISION *_vec_b, FP_PRECISION *_dot);
__global__ void GPU_VecEuclidianNorm_SumSqr(int _dim,FP_PRECISION *_vec, FP_PRECISION *_dotRes);
__global__ void GPU_LeastSquaresProb(int _m1,FP_PRECISION *_Hm,FP_PRECISION *_g,FP_PRECISION *_y,FP_PRECISION *_w, int _m);
__global__ void GPU_VecScale_Multiply(int _dim, FP_PRECISION *_scale, FP_PRECISION *_vec_y, FP_PRECISION *_vec_x);
__global__ void GPU_VecScale_Divide(int _dim, FP_PRECISION *_scale, FP_PRECISION *_vec_y, FP_PRECISION *_vec_x);
__global__ void GPU_VecUpdate_NegScale(int _dim,FP_PRECISION *_vec_y,FP_PRECISION *_scalar,FP_PRECISION *_vec_x);
__global__ void GPU_MatVec_Update(int _Ndim,int _mdim, FP_PRECISION *_x, FP_PRECISION *_A, FP_PRECISION *_y);
__global__ void GPU_VecVecMulti(int _dim,FP_PRECISION *_vec_a, FP_PRECISION *_vec_b);
__global__ void GPU_VecInverse(int _dim,FP_PRECISION *_vec);
__global__ void GPU_VecDamp_FP(int _dim, FP_PRECISION _damp, FP_PRECISION *_vec, FP_PRECISION *_vec_old);
FP_PRECISION GPU_MaxAbsVec_FP(int _dim, FP_PRECISION *_vec);
__global__ void GPUVectorInit_1DFP_Flux(FP_PRECISION_FLUX *_vector, int _dim, FP_PRECISION_FLUX _value);
__global__ void GPUVectorInit_1DFP(FP_PRECISION *_vector, int _dim, FP_PRECISION _value);


__global__ void GPUVectorInit_float_I(float *_vec, int _dim, float _value);
FP_PRECISION GPUVecReduction(FP_PRECISION *_vecDev, int _dim, cublasHandle_t _handle);
__global__ void GPUGetAbsRelErrOfTwoVectors_FP_I( FP_PRECISION *_vecNew, FP_PRECISION *_vecOld, \
	FP_PRECISION *_relErr, int _dim);
FP_PRECISION GPUGetVecNormInf(FP_PRECISION *_deviceVec, int _dim, cublasHandle_t _handle);
FP_PRECISION GPUGetVecNorm2(FP_PRECISION *_deviceVec, int _dim, cublasHandle_t _handle);
FP_PRECISION GPUGetVecRMSE(FP_PRECISION *_deviceVec, int _dim, cublasHandle_t _handle);
FP_PRECISION GPUGetVecDot(FP_PRECISION *_deviceVec1, FP_PRECISION *_deviceVec2, int _dim, cublasHandle_t _handle);
void GPUGetVecNormInfo(int _dim, int _GPUGridDim, int _GPUBlockDim, FP_PRECISION *_vec_x, FP_PRECISION *_vec_y, \
    FP_PRECISION *_vecErr, cublasHandle_t _handle, int _isNormInf, int _isNorm2, int _isRmse, FP_PRECISION *_normInf, \
    FP_PRECISION *_norm2, FP_PRECISION *_rmse);
void GPUPrintf1DArray_FP(int _dim, FP_PRECISION *_GPUArray, char *_parameter, FILE *_LogFile);
void GPUPrintf1DArray_I32(int _dim, M_INT32 *_GPUArray, char *_parameter, FILE *_LogFile);
void GPUPrintf1DArray_UI64(int _dim, M_UINT64 *_GPUArray, char *_parameter, FILE *_LogFile);
 __global__ void GPUVectorSORStore_FP_I(FP_PRECISION *_vec_new,FP_PRECISION *_vec_old, int _dim, FP_PRECISION _sor_f);
 __global__ void GPUVectorInv_FP_I(FP_PRECISION *_vec,FP_PRECISION *_vec_inv, int _dim);
#endif // GPUVECTOR_H