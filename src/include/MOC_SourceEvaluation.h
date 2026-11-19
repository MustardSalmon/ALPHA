#ifndef MOC_SOURCE_EVALU_H
#define MOC_SOURCE_EVALU_H
#include "MOC_CommonMacro.h"


void Jacobi_ComputeFSRSource(int _procID, int _iter,FP_DOUBLE _keff, FP_PRECISION_FLUX *_scalarFlux, \
	FP_PRECISION *_source, int _numGroups, int _numLayers_SubDom, int _numCellsInDom, \
	int *_numFSRsInCell, FP_PRECISION ***_XS_leakage_SubDom,int *_sumNumFSRsBasedGlobCellID,\
	FP_PRECISION *_MacroXS_chi,FP_PRECISION *_macroXS_nf, FP_PRECISION *_macroXS_s,FP_PRECISION *_macroXS_t,\
	M_UINT64 *_MacroXS_s_NZRowStartID, int *_MacroXS_s_NZColStartID);
void Jacobi_TransFSRPara_RayWisedAndAntiClock(int _azimID, int _numGroups, int _numLayers_SubDom, int _numCellsInDom, \
	int *_numFSRsInCell, int *_sumNumFSRsBasedGlobCellID, int *_fsrIdMap_rayWisedToAntiClock, int *_cellRTMGeoType, \
	FP_PRECISION *_para, FP_PRECISION *_paraOrig, int _totalNumFSRsInRTM, int *_sumNumFSRsInRTM);
void Jacobi_ComputeFSRSource_g(int g, int _procID, int _iter, FP_DOUBLE _keff, FP_PRECISION_FLUX *_scalarFlux, \
	FP_PRECISION *_source, int _numGroups, int _numLayers_SubDom, int _numCellsInDom, \
	int *_numFSRsInCell, FP_PRECISION ***_XS_leakage_SubDom, int *_sumNumFSRsBasedGlobCellID, \
	FP_PRECISION *_MacroXS_chi, FP_PRECISION *_macroXS_nf, FP_PRECISION *_macroXS_s, FP_PRECISION *_macroXS_t,\
	M_UINT64 *_MacroXS_s_NZRowStartID, int *_MacroXS_s_NZColStartID);
void Jacobi_ComputeFSRSource_OMP(int _TotalNumFSRs_subDom, int _procID, int _iter,FP_DOUBLE _keff, \
    FP_PRECISION_FLUX *_scalarFlux, FP_PRECISION *_source, int _numGroups, int _numLayers_SubDom, \
	int _numCellsInDom, int *_numFSRsInCell, FP_PRECISION ***_XS_leakage_SubDom,int *_sumNumFSRsBasedGlobCellID,\
	FP_PRECISION *_MacroXS_chi, FP_PRECISION *_MacroXS_nf, FP_PRECISION *_MacroXS_s,FP_PRECISION *_MacroXS_t,\
	M_UINT64 *_MacroXS_s_NZRowStartID, M_UINT64 *_MacroXS_s_NZColStartID);
void Jacobi_ComputeFSRSource_FSR_g(int _fsrID, int _numGroups, FP_PRECISION _keff, FP_PRECISION *_MacroXS_t, \
    FP_PRECISION *_MacroXS_chi, FP_PRECISION *_MacroXS_nf, FP_PRECISION *_MacroXS_s, FP_PRECISION *_scalarFlux, \
	FP_PRECISION *_source, M_UINT64 *_MacroXS_s_NZRowStartID, M_UINT64 *_MacroXS_s_NZColStartID);
FP_PRECISION ScattSourEval_HOScatt(int _globFsrID, int g, int _numGroups, FP_PRECISION_FLUX *_scalarFlux, \
    M_UINT64 *_MacroXS_s_NZRowStartID, M_UINT64 *_MacroXS_s_NZColStartID, FP_PRECISION *_macroXS_s, \
	int _scattOrder, FP_PRECISION *_CoeffA_HOScatt, FP_PRECISION *_SpheHarmCoeff, int _azimID, int _polID,\
	int _numAzimAng, int _numPolAng, M_INT32 _totalNumFSRs_subDom);
FP_PRECISION ScattSourEval_HOScatt_check(int _globFsrID, int g, int _numGroups, FP_PRECISION_FLUX *_scalarFlux, \
    M_UINT64 *_MacroXS_s_NZRowStartID, int *_MacroXS_s_NZColStartID, FP_PRECISION *_macroXS_s, \
	int _scattOrder, FP_PRECISION *_CoeffA_HOScatt, FP_PRECISION *_SpheHarmCoeff, int _azimID, int _polID,\
	int _numAzimAng, int _numPolAng, M_INT32 _totalNumFSRs_subDom);
void FissSourEval(int _procID, int _iter,int _totalNumFSRs_subDom, int _numGroups,FP_DOUBLE _keff, \
    FP_PRECISION_FLUX *_scalarFlux, FP_PRECISION *_source, FP_PRECISION *_MacroXS_chi,FP_PRECISION *_macroXS_nf);
void ScattSourEval_HOScatt_1(M_INT32 _layID, M_INT32 _totalNumFSRs_perLayer, M_INT32 _totalNumFSRs_subDom, int _numGroups, \
    FP_PRECISION_FLUX *_scalarFlux, M_INT32 _numAzimAng, M_INT32 _numPolAng, M_INT32 _polarBoundLowLim, M_INT32 _polarBoundUpLim, \
    M_INT32 _polOffset, M_INT32 _scattOrder, M_UINT64 *_MacroXS_s_NZRowStartID, M_UINT64 *_MacroXS_s_NZColStartID, FP_PRECISION *_macroXS_s, \
	FP_PRECISION *_CoeffA_HOScatt, FP_PRECISION *_SpheHarmCoeff, FP_PRECISION_FLUX  *_FSRAngScattSour);

#endif // !Source_Evalu

