#ifndef MOC_ALLOCA_MEM_H
#define MOC_ALLOCA_MEM_H
#include "MOC_CommonMacro.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void CheckDataTypeRange(int _procID, FILE *_LogFile);

M_INT32 *AllocMem_i32_i32(M_INT32 _dim1, char *_parameter);
M_INT32 *AllocMem_i32_I(size_t _dim1, char *_parameter);
M_INT32 *FreeMem_i32_I(M_INT32 *_pointer, char *_parameter);

FP_PRECISION *AllocMem_FP_I(size_t _dim1, char *_parameter);
FP_PRECISION *FreeMem_FP_I(FP_PRECISION *_pointer, char *_parameter);
M_INT32 *AllocMem_int_I(int _dim1, char *_parameter);
M_INT32 *FreeMem_int_I(M_INT32 *_pointer, char *_parameter);
M_UINT64 *AllocMem_ui64_I(size_t _dim1, char *_parameter);
M_UINT64 *FreeMem_ui64_I(M_UINT64 *_pointer, char *_parameter);

int *FreeMem_i32_I(int *_pointer, char *_parameter);
int **AllocMem_int_II(int _dim1, int _dim2, char *_parameter);
int **FreeMem_int_II(int **_pointer, int _dim1, char *_parameter);
int **AllocMem_int_IP(int _dim1, int *_dim2, char *_parameter);
int **AllocMem_int_IPplusI(int _dim1, int *_dim2,int _plus, char *_parameter);
int **FreeMem_int_IP(int **_pointer, int _dim1, char *_parameter);
int ***AllocMem_int_III(int _dim1, int _dim2, int _dim3, char *_parameter);
int ***FreeMem_int_III(int ***_pointer, int _dim1, int _dim2, char *_parameter);
size_t ***AllocMem_sizeT_III(int _dim1, int _dim2, int _dim3, char *_parameter);
size_t ***FreeMem_sizeT_III(size_t ***_pointer, int _dim1, int _dim2, char *_parameter);
int ***AllocMem_double_II2P(int _dim1, int _dim2, int **_dim3, char *_parameter);
int ***FreeMem_double_II2P(int ***_pointer, int _dim1, int _dim2, char *_parameter);
int ***AllocMem_double_IIP_1(int _dim1, int _dim2, int *_dim3, char *_parameter);
FP_PRECISION *AllocMem_FP_I_1(int _dim1, char *_parameter);
FP_PRECISION_GEO *AllocMem_FPGeo_I(int _dim1, char *_parameter);
FP_PRECISION_GEO *FreeMem_FPGeo_I(FP_PRECISION_GEO *_pointer, char *_parameter);

FP_PRECISION **AllocMem_FP_II(int _dim1, int _dim2, char *_parameter);
FP_PRECISION **FreeMem_FP_II(FP_PRECISION **_pointer, int _dim1, char *_parameter);
FP_DOUBLE **AllocMem_FPDouble_II(int _dim1, int _dim2, char *_parameter);
FP_DOUBLE *AllocMem_FPDouble_I(int _dim1, char *_parameter);
FP_DOUBLE *FreeMem_FPDouble_I(FP_DOUBLE *_pointer, char *_parameter);
FP_PRECISION **AllocMem_FP_IP(int _dim1, int *_dim2, char *_parameter);
FP_PRECISION **FreeMem_FP_IP(FP_PRECISION **_pointer, int _dim1, char *_parameter);
FP_PRECISION_GEO **AllocMem_FPGeo_IP(int _dim1, int *_dim2, char *_parameter);
FP_PRECISION_GEO **FreeMem_FPGeo_IP(FP_PRECISION_GEO **_pointer, int _dim1, char *_parameter);
FP_PRECISION ***AllocMem_FP_III(int _dim1, int _dim2, int _dim3, char *_parameter);
FP_PRECISION ***FreeMem_FP_III(FP_PRECISION ***_pointer, int _dim1, int _dim2, char *_parameter);
FP_PRECISION ***AllocMem_FP_IIP(int _dim1, int _dim2, int *_dim3, char *_parameter);
FP_PRECISION ***FreeMem_FP_IIP(FP_PRECISION ***_pointer, int _dim1, int _dim2, char *_parameter);
FP_PRECISION ***allocMem_FP_IPI(int _dim1, int *_dim2, int _dim3, char *_parameter);
FP_PRECISION ***FreeMem_FP_IPI(FP_PRECISION ***_pointer, int _dim1, int *_dim2, char *_parameter);
FP_PRECISION ***AllocMem_FP_IPP(int _dim1, int *_dim2, int *_dim3, char *_parameter);
FP_PRECISION ***FreeMem_FP_IPP(FP_PRECISION ***_pointer, int _dim1, int *_dim2, char *_parameter);
int ***FreeMem_int_IPP(int ***_pointer, int _dim1, int *_dim2, char *_parameter);
FP_PRECISION ****AllocMem_FP_IIPI(int _dim1, int _dim2, int *_dim3, int _dim4, char *_parameter);
FP_PRECISION ****FreeMem_FP_IIPI(FP_PRECISION ****_pointer, int _dim1, int _dim2, int *_dim3, char *_parameter);
FP_PRECISION ****AllocMem_FP_II2PI(int _dim1, int _dim2, int **_dim3, int _dim4, char *_parameter);
FP_PRECISION ****FreeMem_FP_II2PI(FP_PRECISION ****_pointer, int _dim1, int _dim2, int **_dim3, char *_parameter);
FP_PRECISION ****AllocMem_FP_IIII(int _dim1, int _dim2, int _dim3, int _dim4, char *_parameter);
FP_PRECISION ****FreeMem_FP_IIII(FP_PRECISION ****_pointer, int _dim1, int _dim2, int _dim3, char *_parameter);
FP_PRECISION ****AllocMem_FP_IIIP(int _dim1, int _dim2, int _dim3, int **_dim4, char *_parameter);
FP_PRECISION ****FreeMem_FP_IIIP(FP_PRECISION ****_pointer, int _dim1, int _dim2, int _dim3, char *_parameter);
FP_PRECISION *****AllocMem_FP_IIIII(int _dim1, int _dim2, int _dim3, int _dim4, int _dim5, char *_parameter);
FP_PRECISION *****FreeMem_FP_IIIII(FP_PRECISION *****_pointer, int _dim1, int _dim2, int _dim3, int _dim4, char *_parameter);
FP_PRECISION *****AllocMem_FP_IIIIP(int _dim1, int _dim2, int _dim3, int _dim4, int **_dim5, char *_parameter);
FP_PRECISION *****FreeMem_FP_IIIIP(FP_PRECISION *****_pointer, int _dim1, int _dim2, int _dim3, int _dim4, char *_parameter);
FP_PRECISION *****AllocMem_FP_IIIPI(int _dim1, int _dim2, int _dim3, int *_dim4, int _dim5, char *_parameter);
FP_PRECISION *****FreeMem_FP_IIIPI(FP_PRECISION *****_pointer, int _dim1, int _dim2, int _dim3, int *_dim4, char *_parameter);
FP_PRECISION ******AllocMem_FP_IIP2PIP(int _dim1, int _dim2, int *_dim3, int **_dim4, int _dim5, int *_dim6, char *_parameter);
FP_PRECISION *****AllocMem_FP_IIPII(int _dim1, int _dim2, int *_dim3, int _dim4, int _dim5, char *_parameter);
int ***FreeMem_int_II2P(int ***_pointer, int _dim1, int _dim2, char *_parameter);
double **AllocMem_FP64_II(int _dim1, int _dim2, char *_parameter);
double *AllocMem_FP64_I(int _dim1, char *_parameter);
double *FreeMem_FP64_I(double *_pointer, char *_parameter);
double **FreeMem_FP64_II(double **_pointer, int _dim1, char *_parameter);

FP_PRECISION_FLUX *AllocMem_FP_Flux(size_t _dim1, char *_parameter);
FP_PRECISION_FLUX *FreeMem_FPFlux_I(FP_PRECISION_FLUX *_pointer, char *_parameter);

FP_PRECISION_GEO *AllocMem_FP_I_Geo(int _dim1, char *_parameter);



float *AllocMem_float_I(int _dim1, char *_parameter);
float *FreeMem_float_I(float *_pointer, char *_parameter);

FP_PRECISION **AllocMem_FP_CMFDCommPara(int _numDir, int _nx, int _ny, int _nz, int _ng, char *_parameter);
char **AllocMem_char_II(int _dim1, int _dim2, char *_parameter);
char **FreeMem_char_II(char **_pointer, int _dim1, char *_parameter);
FP_PRECISION_GEO *FreeMem_FP_I_Geo(FP_PRECISION_GEO *_pointer, char *_parameter);

FP_PRECISION *AllocMem_FP_u64I(M_UINT64 _dim1, char *_parameter);

FP_PRECISION **AllocMem_FP_SpheHarmCoeff(int _numTrSet, int *_numAzimAng, int *_numPolAng, int _scattOrder, char *_parameter);

#endif // !ALLOCA_MEM_H
