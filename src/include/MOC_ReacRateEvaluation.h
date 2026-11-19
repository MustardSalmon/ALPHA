#ifndef MOC_REACRATE_EVALU_H
#define MOC_REACRATE_EVALU_H
#include "MOC_CommonMacro.h"

extern M_INT32 NumFissPinsSubDom[1], NumFissPinsGlobDom[1];

//FP_PRECISION CalcFissRate_SubDom(FP_PRECISION ****_scalarFlux_subDom, int _numCellsInDom, \
//	int **_numFSRsInCell, int _num_group, int **_cellRTMType);
FP_PRECISION CalcTotPower_SubDom(FP_PRECISION **_norm_pin_power, int _numCellsInDom);
void CalcNormPinPower_subDom(FP_PRECISION **_norm_pin_power, FP_PRECISION _tot_power, int _numCellsInDom, int _numCellsInCore);
FP_PRECISION CalcTotPower_Core(int _procID, int _numProcs, int _numCellsInDom, \
	FP_PRECISION **_norm_pin_power, FP_PRECISION *_totPower_procs);
FP_PRECISION CMFD_calcFissRate_SubDom(FP_PRECISION ****_scalarFlux_subDom, \
	FP_PRECISION ****_xs_nf, int _dim_z, int _dim_y, int _dim_x, int _num_group, \
	FP_PRECISION *_hx, FP_PRECISION *_hy, FP_PRECISION *_hz);
//#ifdef MPI_USED
//FP_PRECISION ***CollecPinPower(int _procID, int _numProcs, FP_PRECISION **_norm_pinPower, \
//	int *_numSubDomains, int ***_subDomainsMap, int **_numPin_subDomain);
//#endif // MPI_USED
FP_PRECISION **CalcPinPower(FP_PRECISION ***CollecGlobPinPower, int *_numSubDomains, int **_numPin_subDomain);
FP_PRECISION ***Calc_PinPower_Slice(int num_layer_in_slice, FP_PRECISION ***CollecGlobPinPower, int *_numSubDomains, int **_numPin_subDomain);
FP_DOUBLE MOC_CalcGlobalReacRate(FP_PRECISION *_reacRate_procs, int _procID, int _numProcs, \
	FP_PRECISION_FLUX *_scalarFlux_subDom,int _numLayersZ, int _numCellsInSubDom, int *_numFSRsinCell_subDom, \
	int _numGroup, int *_cellRTMType_subDom, FP_PRECISION *_macroXS_r, FP_PRECISION **_inpXSs_r,\
	struct RTMFSRINFO **_RTModFSRsInfo,int *_sumNumFSRsBasedGlobCellID);
void CalcPinPower_SubDom(int _numLayersZ,FP_PRECISION *_scalarFlux_subDom, int _numCellsInDom, \
	int *_numFSRsInCell, int _numGroup, FP_PRECISION *_norm_pin_power,int *_sumNumFSRsBasedGlobCellID,\
	FP_PRECISION_GEO *_FSRArea_SubDom, FP_PRECISION *_macroXS_f,FP_PRECISION_GEO *_WidthLayersInCore);
void SumPinPower_Lay(int _PinPowerSumNumSet, int *_PinPowerSumLayBound, int _numCellsInDomPerLay, \
    FP_PRECISION *_NormPinPower, FP_PRECISION *_NormPinPower_SumLay);
FP_DOUBLE CalcReacRate_SubDom(FP_PRECISION_FLUX *_scalarFlux_subDom,int _numLayersZ,int _numCellsInDom, \
	int *_numFSRsInCell, int _numGroup, int *_cellRTMType, FP_PRECISION *_macroXS_r, FP_PRECISION **_inpXSs_r,\
	struct RTMFSRINFO **_RTModFSRsInfo,int *_sumNumFSRsBasedGlobCellID);
FP_DOUBLE CalcTotalFlux_SubDom(FP_PRECISION_FLUX *_scalarFlux_subDom, int _numLayersZ, int _numCellsInDom, \
	int *_numFSRsInCell, int _numGroup, int *_cellRTMType, struct RTMFSRINFO **_RTModFSRsInfo, int *_sumNumFSRsBasedGlobCellID);
void CollPinPower_GlobDom(M_INT32 _procID, M_INT32 _numProcs, FP_PRECISION **_CollecGlobPinPower, \
    FP_PRECISION *_NormPinPowerSubDom, M_INT32 *_CollNumPinsSubDom, M_INT32 _MOC_NumPinsInSubDom);
#endif
