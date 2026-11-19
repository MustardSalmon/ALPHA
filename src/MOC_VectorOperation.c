/********************************************************************************/
/* @ File: MOC_VectorOperation.c
/*   introduction: functions related to vector operation
/* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*   1st modified:
/********************************************************************************/
#include <math.h>
#include <stdio.h>
#include "MOC_CommonMacro.h"
#include "MOC_VectorOperation.h"

#include "R_xs_treat.h"
#ifdef MPI_USED
    #include "mpi.h"
#endif // MPI_USED

void VectorInit_FP_Flux(FP_PRECISION_FLUX *_pointer, int _dim, FP_PRECISION_FLUX _value)
{
	int _iter1;
	for (_iter1 = 0; _iter1 < _dim; _iter1++) {
		_pointer[_iter1] = _value;
	}
}

void VectorCopy_FP_Flux(FP_PRECISION_FLUX *_ectypeVector, FP_PRECISION_FLUX *_origVector, int _dim)
{
	int i;
	for (i = 0; i<_dim; i++) {
		_ectypeVector[i] = _origVector[i];
	}
}

/**
* @ Function: IncrementValue_Matrix()
*   introduction: Increment a value in the matrix.
* @ Parameter List:
*   Return:
* @ Created     : 2019/08/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: ComputeRMSE_Vector()
*   introduction: Computes the Root Mean Square Error of two Vectors
* @ Parameter List:
*   Return:
* @ Created     : 2019/08/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified: 2019/09/16, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*/
/**
* @ Function: ComputeRMSE_Vector_EnergyStep()
*   introduction: Computes the Root Mean Square Error of two Vectors
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/16, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified: 
*/
FP_PRECISION ComputeRelErrPowerSum_FluxVecs_lay_g(int _numFSRs_lay, int _numGroups, FP_PRECISION *_vec_x, \
	FP_PRECISION *_vec_y, int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim)
{
	int _fsr, g,_ind, _lay;
	FP_PRECISION _err, _sum = (FP_PRECISION)(0.0);
    if (2==1){
        for (_lay=_layBoundLowLim;_lay<_layBoundUpLim;_lay++) {
            for (_fsr = 0; _fsr < _numFSRs_lay; _fsr++) {
                for (g = _groupBoundLowLim; g < _groupBoundUpLim; g++) {
                    _ind = (_lay * _numFSRs_lay + _fsr) * _numGroups + g;
                    _err = (_vec_x[_ind] - _vec_y[_ind]) / _vec_y[_ind];
                    _sum += _err * _err;
                }
            }
        }
    }
    else{
    //#ifdef OMP_USED
    //    FP_PRECISION _sums[8] = {0.};
    //#endif
		for (_fsr = 0; _fsr < _numFSRs_lay; _fsr++) {
        //#ifdef OMP_USED
        //    omp_set_num_threads(8);
        //    #pragma omp parallel for
        //#endif
			for (g = _groupBoundLowLim; g < _groupBoundUpLim; g++) {
				_ind =  _fsr * _numGroups + g;
				_err = (_vec_x[_ind] - _vec_y[_ind]) / _vec_y[_ind];
            //#ifdef OMP_USED
            //    int k = omp_get_thread_num();
            //    _sums[k] += _err * _err;
            //#else
				_sum += _err * _err;
            //#endif
			}
	    }
    //#ifdef OMP_USED
    //    _sum = 0.;
    //    for (int i = 0; i < 8; i++) {
    //        _sum += _sums[i];
    //    }
    //#endif
    }
	return (_sum);
}

void FluxVecCopy_lay_g(int _numFSRs_lay, int _numGroups, FP_PRECISION *_vec_x, \
	FP_PRECISION *_vec_y, int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim)
{
	int _fsr, g,_ind, _lay;
    if (2==1){
        for (_lay=_layBoundLowLim;_lay<_layBoundUpLim;_lay++) {
            for (_fsr = 0; _fsr < _numFSRs_lay; _fsr++) {
                for (g = _groupBoundLowLim; g < _groupBoundUpLim; g++) {
                    _ind = (_lay * _numFSRs_lay + _fsr) * _numGroups + g;
                    _vec_y[_ind] = _vec_x[_ind];
                }
            }
        }
    }
	else{
        memcpy(_vec_y, _vec_x, (size_t) (sizeof (FP_PRECISION) * _numFSRs_lay * (_groupBoundUpLim - _groupBoundLowLim)));
		//for (_fsr = 0; _fsr < _numFSRs_lay; _fsr++) {
		//	for (g = _groupBoundLowLim; g < _groupBoundUpLim; g++) {
		//		_ind = _fsr * _numGroups + g;
		//		_vec_y[_ind] = _vec_x[_ind];
		//	}
	    //}
    }
}


/**
* @ Function: Normalize_Vector()
*   introduction: normalize the Vector
* @ Parameter List:
*   Return:
* @ Created     : 2019/08/08, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: SetValue_FP_Vector()
*   introduction: set the value of a specific element in Vector
* @ Parameter List:
*   Return:
* @ Created     : 2019/08/11, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: GetRMSE_Vector()
*   introduction: Computes the Root Mean Square Error of two Vectors
* @ Parameter List:
*   Return:
* @ Created     : 2019/08/14, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: MatrixVecMulti_CSR()
*   introduction: perform the sparse matrix-vector multiplication
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/17, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: MatrixVecMulti_CSR_LUDiag()
*   introduction: perform the sparse matrix-vector multiplication
*                 the matrix is stored in LU-matrix and Diag separatly
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/17, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: MatrixVecGetResidual_CSR()
*   introduction: get the residual, r=b-Ax
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/17, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: MatrixVecGetResidual_CSR_1()
*   introduction: get the residual, r=Ax-b
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/17, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: VectorL2Norm_FP_I()
*   introduction: get the L2 norm of a vector, L2=sqrt(sum(vec*vec))
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/17, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: VectorNormalization_FP_I()
*   introduction: normlize the vector
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/17, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: VectorDotProduct_FP_I()
*   introduction: performs dot product of the given vectors "_vec1" and "_vec2"
* @ Parameter List:
*   Return:
* @ Created     : 2019/09/17, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: VectorFabsMax_FP_I()
*   introduction: get the max absolute value of a vector
* @ Parameter List:
*   Return:
* @ Created     : 2019/10/15, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: VectorScale_FP_I()
*   introduction: scale a vector with a scalar
* @ Parameter List:
*   Return:
* @ Created     : 2019/10/15, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: QRDecomposition_Matrix()
*   introduction: perform the QR decomposition
* @ Parameter List:
*   Return:
* @ Created     : 2019/10/16, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
// void QRDecomposition_Matrix(int _dim, FP_PRECISION *_matA, FP_PRECISION *_matQ, FP_PRECISION *_matR,\
//     FP_PRECISION *_sparseA, FP_PRECISION *_colA, FP_PRECISION *_colQ, int *_ia, int *_ja)
// {
//     int i, j, k;
	
// 	// 施密特正交化
// 	for (j=0; j<_dim; j++) {
// 		// store the col j into _col_A
// 		GetOneColumnOfMatrix_CSR(_dim, _matA, _colA, j, _ia, _ja);
// 		VectorCopy_FP_I(_colQ, _colA, _dim);
// 		for (k=0; k<j; k++) {

// 		}

// 	}
// }

/**
* @ Function: GetOneColumnOfMatrix_CSR()
*   introduction: get one column of a matrix with CSR format
* @ Parameter List:
*   Return:
* @ Created     : 2019/10/16, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: SpectrumRadius_JacobiMatrix()
*   introduction: get the jacobi spectrum radius of a matrix 
* @ Parameter List:
*   Return:
* @ Created     : 2019/10/15, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
FP_PRECISION CPUGetAbsRelErrInfNorm_TwoVectors_FP_I(M_UINT32 _dim, FP_PRECISION *_vecNew, FP_PRECISION *_vecOld)
{
	M_UINT32 i;
	FP_PRECISION _err=0.0, _max=0.0;
	for (i = 0; i < _dim; i++) {
#if defined(M_ZeroNegFlux)
        if(_vecOld[i]==0.0) {
	    	_err=0.0;
	    }
#else
		_err = _CPU_FABS((_vecNew[i] - _vecOld[i]) / _vecOld[i]);
#endif
		if (_max < _err) {
			_max = _err;
		}
	}
	return _max;
}
