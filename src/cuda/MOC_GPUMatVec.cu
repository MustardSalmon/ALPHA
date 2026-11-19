/********************************************************************************/
/* @ File: MOC_GPUMatVec.cu
/*   introduction: This file contains functions that relate to the Vector operation
/*                 on GPU
/* @ Created     : 2019/08/14, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*   1st modified:
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <time.h>
#include <curand_kernel.h>
#include "cublas_v2.h"

#include "MOC_GPUMatVec.cuh"
#include "MOC_GPUConstants.cuh"

/**
* @ Function: GPUVectorInit_FP_I()
*   introduction: Iniialize Vector on GPU
* @ Parameter List:
*   Return:
* @ Created     : 2019/08/02, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GPUVectorCopy_FP_I()
*   introduction: copy Vector on GPU
* @ Parameter List:
*   Return:
* @ Created     : 2019/08/22, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GPUGetRelErrOfTwoVectors_FP_I()
*   introduction: get the relative error of two vectors on GPU
* @ Parameter List:
*   Return:
* @ Created     : 2019/08/22, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GPUComputeRMSE_Vector()
*   introduction: Computes the Root Mean Square Error of two Vectors on GPU
* @ Parameter List:
*   Return:
* @ Created     : 2019/08/22, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GPUMatrixVecProduct_CSR()
*   introduction: computes A*x for a matrix stored in sparse compressed row form.
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/25, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GPUMatrixVecResidual_CSR()
*   introduction: computes r=b-A*x for a matrix stored in sparse compressed row form.
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/25, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GPU_MGMatrixVecSumSqr_CSR()
*   introduction: computes sum(pow(r=b-A*x, 2) for a matrix stored in sparse compressed row form.
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/26, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GPU_VecInit_FP_1()
*   introduction: initialize the vector on GPU
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/26, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GPU_VecNorm_Reduction()
*   introduction: reduct the vector on GPU
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/26, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GPU_VecVecDot()
*   introduction: dot product of two vectors
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/26, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GPU_VecVecMulti()
*   introduction: multiplication of two vectors, scale _vec_a by _vec_b
* @ Parameter List:
*   Return:
* @ Created     : 2019/11/20, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GPU_VecVecMulti()
*   introduction: set the inverse of a vector
* @ Parameter List:
*   Return:
* @ Created     : 2019/11/20, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
//
/**
* @ Function: GPU_VecEuclidianNorm_SumSqr()
*   introduction: get the euclidian norm of vector vec, operating sum(x^2) on device
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/26, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GPU_LeastSquaresProb()
*   introduction: solve the serial Least Squares problem
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/26, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
 

/**
* @ Function: GPU_VecScale_Multiply()
*   introduction: scale a vector with a scalar (multiply)
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/26, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GPU_VecScale_Divide()
*   introduction: scale a vector with a scalar (devide)
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/26, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GPU_VecEuclidianNorm_SumSqr()
*   introduction: get the euclidian norm of vector vec, operating sum(x^2) on device
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/26, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
// __global__ void cuda_matvec_kernel(int dev_dim, FP_PRECISION *y, FP_PRECISION *AA, int *JA, int *IA, 
//                                    FP_PRECISION *x )
// {
//     int thread_id=threadIdx.x+blockIdx.x*blockDim.x; //global thread index
//     int i = thread_id , jj=0;
//     // if (i<dev_dim) {
//         while (i < dev_dim) {
//             y[i] = 0;
//             for( jj = IA[i] ; jj < IA[i+1]; jj ++ ){
//                y[i] += AA[jj] * x[JA[jj]];
//             }
//             i += blockDim.x * gridDim.x;
//         }
//     // }
// }

/**
* @ Function: GPU_VecUpdate_NegScale()
*   introduction: update a vector by negtive scale
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/26, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
//
/**
* @ Function: GPU_MatVec_Update()
*   introduction: Matvec+vector update-> x=A*y+x opou A non-sparse matrix
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/26, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
 * @ Function: GPUVectorInit_1DFP()
 *   introduction: initialize a 1-D array with a given value (float point)
 *                 this function is called by host and excuted on device
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2018/05/30, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
 /**
 * @ Function: GPUPrintf1DArray_FP()
 *   introduction: print the 1D array on GPU with FP_PRECISION
 * @ Parameter List:
 *   Return: void
 * @ Created     : 2020/02/10, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
 

__global__ void GPUVectorInit_1DFP_Flux(FP_PRECISION_FLUX *_vector, int _dim, FP_PRECISION_FLUX _value)
{
	int _tid = threadIdx.x + blockIdx.x * blockDim.x;
	// int _tid = threadIdx.x + blockIdx.x * GPUNumThreadPerBlock;

	while (_tid < _dim) {
		_vector[_tid] = _value;
		/* Increment the thread id */
		_tid += blockDim.x * gridDim.x;
		// _tid += GPUNumThreadPerBlock * GPUNumThreadBlocks;

	}
}

__global__ void GPUVectorInv_FP_I(FP_PRECISION *_vec,FP_PRECISION *_vec_inv, int _dim)
{
	int _tid = threadIdx.x + blockIdx.x * blockDim.x;
	/* Iterate over all elements */
	while (_tid < _dim) {
		_vec_inv[_tid] = 1.0 / _vec[_tid];
		_tid += blockDim.x * gridDim.x;
	}
}
