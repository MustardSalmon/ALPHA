#ifndef MOC_VECTOR_OPE_H
#define MOC_VECTOR_OPE_H
#include "MOC_CommonMacro.h"
void VectorInit_FP_I(FP_PRECISION *_pointer, int _dim, FP_PRECISION _value);
void VectorInit_FP_II(FP_PRECISION **_pointer, int _dim1, int _dim2, FP_PRECISION _value);
void Initialization_double_IPI(FP_PRECISION ***_pointer, int _dim1, int *_dim2, int _dim3, FP_PRECISION _value);
void Initialization_double_III(FP_PRECISION ***_pointer, int _dim1, int _dim2, int _dim3, FP_PRECISION _value);
void Initialization_double_II2PI(FP_PRECISION ****_pointer, int _dim1, int _dim2, int **_dim3, int _dim4, FP_PRECISION _value);
void Initialization_double_IIIII(FP_PRECISION *****_pointer, int _dim1, int _dim2, int _dim3, int _dim4, int _dim5, FP_PRECISION _value);
void Initialization_double_IIIPI(FP_PRECISION *****_pointer, int _dim1, int _dim2, int _dim3, int *_dim4, int _dim5, FP_PRECISION _value);
void Initialization_double_IIII(FP_PRECISION ****_pointer, int _dim1, int _dim2, int _dim3, int _dim4, FP_PRECISION _value);
FP_PRECISION VectorInfNorm_FP_I(FP_PRECISION *_vec_new, FP_PRECISION *_vec_old, int _dim);
FP_PRECISION FluxL2Norm_FP_I(FP_PRECISION *_vec_new, FP_PRECISION *_vec_old, int _dim);
FP_PRECISION VectorL1Norm_double_IPI(FP_PRECISION ***_vec_new, FP_PRECISION ***_vec_old, int _dim1, int *_dim2, int _dim3);
FP_PRECISION VectorRMS_double_IPI(FP_PRECISION ***_vec_new, FP_PRECISION ***_vec_old, int _dim1, int *_dim2, int _dim3);
void VectorUpdate_double_IIII(FP_PRECISION ****_vec_new, FP_PRECISION ****_vec_old, \
	int _dim1, int _dim2, int _dim3, int _dim4);
FP_PRECISION VectorInfNorm_double_II2PI(FP_PRECISION ****_vec_new, FP_PRECISION ****_vec_old, int _dim1, int _dim2, int **_dim3, int _dim4);
void Initialization_double_IIPII(FP_PRECISION *****_pointer, int _dim1, int _dim2, int *_dim3, int _dim4, int _dim5, FP_PRECISION _value);
void Initialization_double_IIP2PIP(FP_PRECISION ******_pointer, int _dim1, int _dim2, int *_dim3, int **_dim4, int _dim5, int *_dim6, FP_PRECISION _value);
void VectorCopy_FP_II(FP_PRECISION **_ectypeVector, FP_PRECISION **_origVector,int _dim1,int _dim2);
void VectorCopy_FP_I(FP_PRECISION *_ectypeVector, FP_PRECISION *_origVector, int _dim);
void VectorCopy_FP_Flux(FP_PRECISION_FLUX *_ectypeVector, FP_PRECISION_FLUX *_origVector, int _dim);
FP_PRECISION VectorSum_FP_I(FP_PRECISION *_vector, int _dim);
FP_PRECISION VectorMax_FP_I(FP_PRECISION *_vector, int _dim);
FP_PRECISION VectorFabsMax_FP_I(FP_PRECISION *_vector, int _dim);
void VectorInit_FP_Flux(FP_PRECISION_FLUX *_pointer, int _dim, FP_PRECISION_FLUX _value);
void VectorInit_FP_AxialAngFlux(FP_PRECISION_FLUX *_pointer, size_t _dim, FP_PRECISION_FLUX _value);
void VectorInit_FP_Flux_g(int g, FP_PRECISION_FLUX *_flux, int _numFsrs, int _numGroups, FP_PRECISION_FLUX _value);
void IncrementValue_Matrix(int _eleID_row, int _eleID_col, int _numEle_perRow, \
    FP_PRECISION *_matrix_A, FP_PRECISION _value); 
FP_PRECISION ComputeRMSE_Vector(int _dim, FP_PRECISION *_vec_x, FP_PRECISION *_vec_y);
FP_PRECISION ComputeRMSE_Flux_lay_g(int _numFSRs_lay, int _numGroups, FP_PRECISION *_vec_x, \
	FP_PRECISION *_vec_y, int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim);
void Normalize_Vector(int dim, FP_PRECISION *_vec, FP_PRECISION _norm_factor);
void SetValue_FP_Vector(int _index, FP_PRECISION *_vec, FP_PRECISION _value);
FP_PRECISION GetRMSE_Vector(int _dim, FP_PRECISION *_vec_err);


void MatrixVecMulti_CSR(FP_PRECISION *_matrix, FP_PRECISION *_vector, FP_PRECISION *_solution, \
	int _numRow, int *_rowStartInd, int *_colInd);
void MatrixVecMulti_CSR_LUDiag(FP_PRECISION *_matrix_lu, FP_PRECISION *_matrix_diag, \
    FP_PRECISION *_vector, FP_PRECISION *_solution, int _numRow, int *_rowStartInd, int *_colInd);
void MatrixVecGetResidual_CSR(FP_PRECISION *_matrix, FP_PRECISION *_rhs, FP_PRECISION *_vector, \
	FP_PRECISION *_solution, int _numRow, int *_rowStartInd, int *_colInd);
void MatrixVecGetResidual_CSR_1(FP_PRECISION *_matrix, FP_PRECISION *_rhs, FP_PRECISION *_vector, \
	FP_PRECISION *_solution, int _numRow, int *_rowStartInd, int *_colInd);
FP_PRECISION VectorL2Norm_FP_I(FP_PRECISION *_vec, int _dim);
void VectorNormalization_FP_I(FP_PRECISION *_vec, int _dim, FP_PRECISION _norm);
FP_PRECISION VectorDotProduct_FP_I(FP_PRECISION *_vec1, FP_PRECISION *_vec2, int _dim);

void VectorScale_FP_I(int _dim, FP_PRECISION _sca, FP_PRECISION *_vec_x, FP_PRECISION *_vec_y);
FP_PRECISION SpectrumRadius_JacobiMatrix(FP_PRECISION *_mat_lu, FP_PRECISION *_mat_diag, \
    int *_ia, int *_ja, int _dim, FP_PRECISION *_x, FP_PRECISION *_y, FP_PRECISION _tol, int _maxIter);
void VecDamp_FP(int _dim, FP_PRECISION _damp, FP_PRECISION *_vec, FP_PRECISION *_vec_old);

FP_PRECISION Max_AbsVec_FP(int _dim, FP_PRECISION *_vec);
FP_PRECISION Max_AbsRelErr_Vec_FP(int _dim, FP_PRECISION *_vec, FP_PRECISION *_vec_old);
void VectorInit_int_I(int *_pointer, int _dim, int _value);

FP_PRECISION CPUGetAbsRelErrInfNorm_TwoVectors_FP_I(M_UINT32 _dim, FP_PRECISION *_vecNew, FP_PRECISION *_vecOld);

FP_PRECISION CPUGetRelErrPowerSum_TwoVectors_FP_I(M_UINT32 _dim, FP_PRECISION *_vecNew, FP_PRECISION *_vecOld);
FP_PRECISION ComputeRelErrPowerSum_FluxVecs_lay_g(int _numFSRs_lay, int _numGroups, FP_PRECISION *_vec_x, \
	FP_PRECISION *_vec_y, int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim);
void FluxVecCopy_lay_g(int _numFSRs_lay, int _numGroups, FP_PRECISION *_vec_x, \
	FP_PRECISION *_vec_y, int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim);
void ComputeLSFluxRes(int _dim, FP_PRECISION *_vec_x, FP_PRECISION *_vec_y, int _isNormInf, \
    int _isNorm2, int _isRmse, FP_PRECISION *_normInf, FP_PRECISION *_norm2, FP_PRECISION *_rmse);
#endif // !VECTOR_OPE_H