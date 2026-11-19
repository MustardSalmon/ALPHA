#ifndef MOC_QUADRATURE_H
#define MOC_QUADRATURE_H
#include "MOC_CommonMacro.h"
extern FP_PRECISION **AzimWeights;
extern FP_PRECISION **TotalWeights, **TotalWeights_2DMOC;
extern FP_PRECISION **SinTheta, **CosTheta, *OneOverSinTheta;
extern FP_PRECISION **PolarWeights;
extern FP_PRECISION *SinThetaForExp;
extern FP_PRECISION **SpheHarmCoeff;

extern FP_PRECISION *CoeffA_HOScatt;

extern FP_DOUBLE **Theta;

void SetPolarQuadrature(int _numTrackSet, int *_quadType, int *_numPolarAng, FP_PRECISION **_sinTheta, FP_PRECISION **_polarWeights,FP_PRECISION **_cosTheta);
void TYPolarQuad(FP_PRECISION *_sinTheta, FP_PRECISION *_polarWeights, int _order_theta, FP_PRECISION *_cosTheta);
void LeonardPolarQuad(FP_PRECISION *_sinTheta, FP_PRECISION *_polarWeights, int _order_theta, FP_PRECISION *_cosTheta);
void SetAzimuthalQuadrature(int _numTrackSet, int _quadType, int *_numAzimAng, FP_PRECISION_GEO **_effAzimuAng, FP_PRECISION **_azimWeights);
void GLPolarQuad(FP_PRECISION *_sinTheta, FP_PRECISION *_polarWeights, int _order_theta, FP_PRECISION *_cosTheta);
void GLPolarQuad_Calc(FP_PRECISION *_sinTheta, FP_PRECISION *_polarWeights, int _numPolarAng);
void getLegendreRoots(FP_PRECISION_GEO *_half_roots, int _numRoots);
FP_PRECISION_GEO secondLogDerivLegendre(int _numRoots, FP_PRECISION_GEO _x);
FP_PRECISION_GEO logDerivLegendre(int _numRoots, FP_PRECISION_GEO _x);
FP_PRECISION_GEO legendrePolynomial(int _numRoots, FP_PRECISION_GEO _x);
void sort_double(FP_PRECISION_GEO *_array, int _length);
void getGLWeights(FP_PRECISION_GEO *_roots, int _numRoots, FP_PRECISION *_polar_weights);
void getGLSineTheta(FP_PRECISION_GEO *_roots, int _numRoots, FP_PRECISION *_sinTheta);
void EqualWeightPolarQuad(FP_PRECISION *_sinTheta, FP_PRECISION *_polarWeights, int _order_theta);
void EqualAnglePolarQuad(FP_PRECISION *_sinTheta, FP_PRECISION *_polarWeights, int _order_theta);
void SetAzimuthalQuadrature_Weights(int _quadType,int _numAzimAng,FP_PRECISION_GEO *_effAzimuAng,FP_PRECISION *_azimWeights);
void SetTotalWeight(int _numTrackSet, int *_numAzimAng, int *_numPolarAng, FP_PRECISION **_sinTheta, \
	FP_PRECISION **_polarWeights, FP_PRECISION **_azimWeights, FP_PRECISION_GEO **_raySpacingEff, \
	FP_PRECISION **_totalWeight, FP_PRECISION **_totalWeight_2DMOC);
void SetSinThetaForExpTabel(int _numPolAng, FP_PRECISION *_sinTheta);
void SetQuadratureInfo(int _numTrackSet,int *_numAzimAng, int *_numPolarAng, int _scattOrder);
FP_PRECISION **TotalWeight_AllocMem(int _numTrackSet, int *_numAzimAng, int *_numPolarAng, char *_parameter);
void FreeQuadratureInfo(int _numTrackSet);
FP_PRECISION tildePmm(int _m, FP_PRECISION _x);
FP_PRECISION tildePmm1(int _m, FP_PRECISION _x);
FP_PRECISION tildePlm(int _l, int _m, FP_PRECISION _x);
FP_PRECISION Plm(int _l, int _m, FP_DOUBLE _x);
FP_PRECISION Ylm(int _l, int _m, FP_DOUBLE _theta, FP_DOUBLE _phi);
M_INT32 FactorialEval(M_INT32 _int);
void CoeffEval_HOScatt(M_INT32 _orderN);
void SetTheta(int _numTrackSet, int *_numPolarAng, FP_PRECISION **_sinTheta, FP_DOUBLE **_theta);
void SetSpheHarmCoeff(int _numTrSet, int *_numAzimAng, int *_numPolAng, \
    int _scattOrder, FP_DOUBLE **_effAzimAng, FP_DOUBLE **_theta, FP_PRECISION **_spheHarmCoeff);
#endif // QUADRATURE_H