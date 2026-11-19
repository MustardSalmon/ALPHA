#ifndef MOC_CMFD_H
#define MOC_CMFD_H
#include "MOC_CommonMacro.h"
#include "MOC_Track.h"

extern FP_PRECISION *CMFDXs_t, *CMFDXs_nf, *CMFDXs_a, *CMFDXs_f, *CMFD_Chi;
extern FP_PRECISION *CMFDXs_s;
extern FP_PRECISION *CMFDHetScalFlux, *CMFDNodeDiffCoeff, *CMFDSurfDiffCoeff_X, *CMFDSurfDiffCoeff_Y, *CMFDSurfNDF_X, *CMFDSurfNDF_Y;
extern FP_PRECISION *CMFDSurfDiffCoeff_Z, *CMFDSurfNDF_Z, *CMFDSurfNDF_XOld, *CMFDSurfNDF_YOld, *CMFDSurfNDF_ZOld;
extern FP_PRECISION *MOCParCurr_X_Pos, *MOCParCurr_X_Neg, *MOCParCurr_Y_Pos, *MOCParCurr_Y_Neg, *MOC_NetCurr_Z;
extern FP_PRECISION *MOCtoCMFD_ParCurr_X_Pos, *MOCtoCMFD_ParCurr_X_Neg, *MOCtoCMFD_ParCurr_Y_Pos, *MOCtoCMFD_ParCurr_Y_Neg;
extern FP_PRECISION *MGCMFDNetCurr_X, *MGCMFDNetCurr_Y, *MGCMFDNetCurr_Z;
extern FP_PRECISION *CMFDMatrix_A, *CMFDTotalSource, *CMFDScalarFlux,*CMFDScalarFlux_Old, *CMFDScalarFlux_Old_LS;
extern FP_PRECISION TEST1, TEST2;
extern int CheckCount_Neg, CheckCount_Pos,*Index_1DArray;
extern FP_DOUBLE  CMFDKeff, CMFDKeff_Old;

extern int CMFD_NumNoneZeroValue_Matrix[1], CMFD_NumNoneZeroValueLU_Matrix;
extern FP_PRECISION /**CRS_Matrix_A, */*MatrixCSR/*, *MatrixCSR_Diag*/;
extern int /**CSR_RowStartInd_Matrix_A, */*MatrixCSR_RowStartInd/*, *CSR_ColInd_Matrix_A*/, *MatrixCSR_ColInd;
extern int TotNumMGCMFDIter, TotNumMGLSIter, TotNumCGCMFDIter, TotNumCGLSIter;

extern FP_PRECISION *CGCMFDXs_t, *CGCMFDXs_nf, *CGCMFD_Chi;
extern FP_PRECISION *CGCMFDXs_s;
extern FP_PRECISION *CGCMFDHetScalFlux, *CGCMFDNodeDiffCoeff;
extern FP_PRECISION *CGCMFDSurfDiffCoeff_X, *CGCMFDSurfDiffCoeff_Y, *CGCMFDSurfDiffCoeff_Z;
extern FP_PRECISION *CGCMFDSurfNDF_X, *CGCMFDSurfNDF_Y,  *CGCMFDSurfNDF_Z;
extern FP_PRECISION *CGCMFDSurfNDF_XOld, *CGCMFDSurfNDF_YOld, *CGCMFDSurfNDF_ZOld;
extern FP_PRECISION *CGCMFDNetCurr_X, *CGCMFDNetCurr_Y, *CGCMFDNetCurr_Z;
extern FP_PRECISION *CGCMFDTotalSource, *CGCMFDScalarFlux, *CGCMFDScalarFlux_Old, *CGCMFDScalarFlux_Old_LS;
extern FP_PRECISION *CGMatrixCSR/*, *CGMatrixCSR_Diag*/;
// extern FP_PRECISION *MGCMFDNetCurr_X, *MGCMFDNetCurr_Y, *MGCMFDNetCurr_Z;
extern int *CGMatrixCSR_RowStartInd, *CGMatrixCSR_ColInd;
extern int CGCMFD_NumNoneZeroValue_Matrix[1], CGCMFD_NumNoneZeroValueLU_Matrix;

extern int GmresKrylovDim_MG, GmresPrecond_MG;
extern FP_PRECISION *MG_HessenburgMatrix, *MG_GivensRotations, *MG_Residual, *MG_TemnpVec_1, *MG_TemnpVec_2;
extern FP_PRECISION *MG_KrylovOrthBasisVec, *MG_HessenburgRHSVec,*MG_ILUMatrix;
extern int *MG_DiagPointer;
extern M_INT32 *MG_IW;

extern int GmresKrylovDim_CG, GmresPrecond_CG;
extern FP_PRECISION *CG_HessenburgMatrix, *CG_GivensRotations, *CG_Residual, *CG_TemnpVec_1, *CG_TemnpVec_2;
extern FP_PRECISION *CG_KrylovOrthBasisVec, *CG_HessenburgRHSVec, *CG_ILUMatrix;
extern int *CG_DiagPointer;
extern M_INT32 *CG_IW;

extern FP_PRECISION MGCMFD_SpecRadi_Jacobi;
extern FP_DOUBLE MGCMFD_damp, Min_damp;
extern int MGCMFD_Conv, CGCMFD_Conv, MGCMFD_Diver[1], CGCMFD_Diver[1];
extern FP_PRECISION *MGCMFD_CCW, *MGCMFD_CCN, *MGCMFD_CCE, *MGCMFD_CCS, *MGCMFD_CCB, *MGCMFD_CCT, *MGCMFD_AM;
extern FP_PRECISION *MGCMFD_Del, *MGCMFD_Al, *MGCMFD_Au, *MGCMFD_Delinv, *MGCMFD_Deliau, *MGCMFD_Ainvl, *MGCMFD_Ainvu, *MGCMFD_Ainvd;
extern FP_PRECISION *MGCMFD_TotFissQ, *MGCMFD_TotFissQ_old, *MGCMFD_TotSour_g;

extern FP_PRECISION *MGCMFD_flux_in, *MGCMFD_flux_add_in, *MGCMFD_src_in, *MGCMFD_vr0_mg;
extern FP_PRECISION *MGCMFD_vr_mg, *MGCMFD_vp_mg, *MGCMFD_vv_mg, *MGCMFD_vs_mg, *MGCMFD_vt_mg, *MGCMFD_vy_mg, *MGCMFD_vz_mg, *MGCMFD_s_mg;
extern FP_PRECISION *MGCMFD_b0_mg, *MGCMFD_s1dl_mg, *MGCMFD_b01d_mg, *MGCMFD_y_mg;
extern int *MGCMFD_nrnx, *MGCMFD_FluxFlag;
extern float *MGCMFD_NegFluxCount,*CGCMFD_NegFluxCount;
extern FP_PRECISION *CMFD_NodeVol;

extern int *MGCMFD_fissQflagl2, *MGCMFD_fissQflaglinf, *MGCMFD_flageig;
extern FP_PRECISION *MGCMFD_fissQerrlinf, *MGCMFD_fissQrerrl2, *MGCMFD_fluxErrlinf;
extern FP_DOUBLE MGCMFD_erreig[1];
extern FP_PRECISION MGCMFD_epseig, MGCMFD_epsl2, MGCMFD_epslinf;
extern int MGCMFD_LinearSolver, CGCMFD_LinearSolver;
extern FP_PRECISION *MGCMFD_fluxNormInf, *MGCMFD_fluxNorm2, *MGCMFD_fluxRMSE;
extern int *MGCMFD_boolFluxNormInf, *MGCMFD_boolFluxNorm2, *MGCMFD_boolFluxRMSE;

extern FP_PRECISION MGCMFDLS_NormInf[1], MGCMFDLS_Norm2[1], MGCMFDLS_RMSE[1];
extern FP_PRECISION CGCMFDLS_NormInf[1], CGCMFDLS_Norm2[1], CGCMFDLS_RMSE[1];

extern int MGCMFD_Flag, CGCMFD_Flag;

extern FP_PRECISION *MGCMFDFlux_MPIRecv, *MGCMFDNodeD_MPIRecv, *MGCMFDComm_MPISend;
extern int *CMFDComm_TarProc, *MGCMFDComm_NumData, ***CMFDComm_DirBound, *CMFDComm_RecvDir, *MGCMFDData_MPIComm_SumInd;
extern int MGCMFDData_MPIComm_Num, *MGCMFDData_MPIComm_Index, MGCMFD_MOCParCurrInnerBound_Num, *MGCMFD_MOCParCurrInnerBound_Index, *MGCMFD_MOCParCurrInnerBound_Dir;
extern FP_DOUBLE CMFDGlobalFissRate[1], CMFDGlobalFissRate_Old[1];
extern FP_PRECISION *MGCMFDMPI_InBoundSourCoeff;
extern int *MGCMFDMPI_InBoundSourCoeff_Index, *MGCMFDMPI_InBoundSourCoeff_Dir;
extern FP_PRECISION *MGCMFDMPI_InBoundSourCoeff_GeoDim;

extern int CGCMFDData_MPIComm_Num, *CGCMFDData_MPIComm_SumInd;
extern FP_PRECISION *CGCMFDFlux_MPIRecv, *CGCMFDNodeD_MPIRecv, *CGCMFDComm_MPISend;
extern FP_PRECISION *CGCMFDMPI_InBoundSourCoeff, *CGCMFDMPI_InBoundSourCoeff_GeoDim;
extern int *CGCMFDMPI_InBoundSourCoeff_Index, *CGCMFDMPI_InBoundSourCoeff_Dir;
extern FP_PRECISION *CGCMFD_TotFissQ, *CGCMFD_TotFissQ_old;
extern FP_DOUBLE CGCMFDGlobalFissRate[1], CGCMFDGlobalFissRate_Old[1];
extern int *CGCMFD_FluxFlag, *CGCMFD_fissQflagl2, *CGCMFD_fissQflaglinf, *CGCMFD_flageig;
extern FP_PRECISION *CGCMFD_fissQerrlinf, *CGCMFD_fissQrerrl2, *CGCMFD_fluxErrlinf;
extern FP_DOUBLE CGCMFD_erreig[1];
extern FP_PRECISION CGCMFD_epseig, CGCMFD_epsl2, CGCMFD_epslinf;
extern int *CGCMFDComm_NumData, *CGCMFDData_MPIComm_Index;
extern FP_PRECISION *CGCMFD_fluxNormInf, *CGCMFD_fluxNorm2, *CGCMFD_fluxRMSE;
extern int *CGCMFD_boolFluxNormInf, *CGCMFD_boolFluxNorm2, *CGCMFD_boolFluxRMSE;
extern FP_PRECISION *MGCMFD_LSFluxRes, *CGCMFD_LSFluxRes;;

FP_PRECISION_FLUX CMFD_Restriction_Flux_G(FP_PRECISION_FLUX *_MOCScalarFlux, int _groupID_MOC, \
	int _numMOCGroups, int _index_fsr, int _RTMType, FP_PRECISION_GEO *_FSRsAreaInRTM, \
	int *_numFSRsInRTM, int *_sumNumFSRsInRTM);
FP_PRECISION CMFD_Restriction_1DXSs_G(FP_PRECISION_FLUX *_MOCScalarFlux, int _groupID_MOC, \
	int _numMOCGroups, int _index_fsr, int _RTMType, FP_PRECISION_GEO *_FSRsAreaInRTM, \
	int *_numFSRsInRTM, int *_sumNumFSRsInRTM, FP_PRECISION *_1DXSs_MOC);
void CMFD_Restriction_Flux(FP_PRECISION_FLUX *_CMFDScalarFLux_MOC, FP_PRECISION_FLUX *_MOCScalarFlux, \
	int *_CMFDGroupLim, int _numCMFDGroup, int _numPinsInDom, int _numMOCGroups, \
	FP_PRECISION_GEO *_FSRsAreaInRTM, int *_numFSRsInRTM, int *_cellRTMType, int *_sumNumFSRsInRTM, \
	int *_sumNumFSRsBasedGlobCellID);
void CMFD_Restriction_1DXSs(FP_PRECISION *_1DXSs_CMFD, FP_PRECISION *_1DXSs_MOC, FP_PRECISION_FLUX *_MOCScalarFlux, \
	int *_CMFDGroupLim, int _numCMFDGroup, int _numMOCGroups, int _numPinsInDom, FP_PRECISION_GEO *_FSRsAreaInRTM, int *_numFSRsInRTM, \
	int *_cellRTMType, int *_sumNumFSRsInRTM, int *_sumNumFSRsBasedGlobCellID);
void CMFD_Restriction_VolumeWeighted(FP_PRECISION *_Para_CMFD, FP_PRECISION *_Para_MOC, \
	int *_CMFDGroupLim, int _numCMFDGroup, int _numMOCGroups, int _numPinsInDom, FP_PRECISION_GEO *_FSRsAreaInRTM, int *_numFSRsInRTM, \
	int *_cellRTMType, int *_sumNumFSRsInRTM, int *_sumNumFSRsBasedGlobCellID);
void CMFD_Restriction_FixedSource(FP_PRECISION *_FixSour_CMFD, FP_PRECISION *_FixSour_MOC, FP_PRECISION *_XSt_MOC, \
	int *_CMFDGroupLim, int _numCMFDGroup, int _numMOCGroups, int _numPinsInDom, FP_PRECISION_GEO *_FSRsAreaInRTM, int *_numFSRsInRTM, \
	int *_cellRTMType, int *_sumNumFSRsInRTM, int *_sumNumFSRsBasedGlobCellID);
void CMFD_Restriction_Chi(FP_PRECISION *_Chi_CMFD, FP_PRECISION *_Chi_MOC, FP_PRECISION *_MacroXS_nf, \
    FP_PRECISION_FLUX *_MOCScalarFlux, int *_CMFDGroupLim, int _numCMFDGroup, int _numMOCGroups, \
	int _numPinsInDom, FP_PRECISION_GEO *_FSRsAreaInRTM, int *_numFSRsInRTM, int *_cellRTMType, \
	int *_sumNumFSRsInRTM, int *_sumNumFSRsBasedGlobCellID);
void CMFD_Restriction_2DXSs(FP_PRECISION *_2DXSs_CMFD, FP_PRECISION *_2DXSs_MOC, FP_PRECISION_FLUX *_MOCScalarFlux, \
	int *_CMFDGroupLim, int _numCMFDGroup, int _numPinsInDom, int _numMOCGroups, FP_PRECISION_GEO *_FSRsAreaInRTM, \
	int *_numFSRsInRTM, int *_cellRTMType, int *_sumNumFSRsInRTM, int *_sumNumFSRsBasedGlobCellID,\
	M_UINT64 *_MacroXS_s_NZRowStartID, M_UINT64 *_MacroXS_s_NZColStartID, int _scattOrder);
void CMFD_Restriction_DiffCoeff(FP_PRECISION *_DiffCoeff_CMFD, FP_PRECISION *_TrXSs_MOC, FP_PRECISION_FLUX *_MOCScalarFlux, \
	FP_PRECISION_FLUX *_CMFDScalarFLux_MOC, int *_CMFDGroupLim, int _numCMFDGroup, int _numPinsInDom, int _numMOCGroups, \
	FP_PRECISION_GEO *_FSRsAreaInRTM, int *_numFSRsInRTM, int *_cellRTMType, int *_sumNumFSRsInRTM, int *_sumNumFSRsBasedGlobCellID);
void GetParCurr_2DMOC(FP_PRECISION *_parCurr_MOC_X_Pos, FP_PRECISION *_parCurr_MOC_X_Neg, FP_PRECISION *_parCurr_MOC_Y_Pos, \
	FP_PRECISION *_parCurr_MOC_Y_Neg, FP_PRECISION *_ang_flux, FP_PRECISION *_totalWeights, int _trackID, int _azimID, \
	int _numPolAng, int *_numTrackX, int *_numTrackY, int _numAzimuAng, int _iftrackStart, int _RTMType, int _pinID_Layer, \
	int _surfIndex_Layer_X, int _surfIndex_Layer_Y, int _numMOCGroups, FP_PRECISION_GEO *_RTMWidth_X, FP_PRECISION_GEO *_RTMWidth_Y, \
	int _numCellsInSubDom_X, int _polarBoundLowLim, int _polarBoundUpLim);
void CMFD_RestrictionCurr_GroupCond(FP_PRECISION *_curr_CMFD, FP_PRECISION *_curr_MOC, \
	int *_CMFDGroupLim, int _numCMFDGroup, int _numLayersZ, int _numMOCGroups, \
	int _numCellsInSubDom_X, int _numCellsInSubDom_Y, int _direction);
void CGCMFD_Restriction_Sumation(FP_PRECISION_FLUX *_CMFDPara_MG, FP_PRECISION_FLUX *_CMFPara_CG, \
	int *_CMFDGroupLim_CG, int _numCMFDGroup_CG, int _numCMFDGroup_MG, int _dim);
void CMFD_Restriction_SurfFlux(FP_PRECISION *_surfFlux_CMFD, FP_PRECISION *_parCurr_CMFD_Pos, FP_PRECISION *_parCurr_CMFD_Neg, \
	int _numCMFDGroup, int _numLayersZ, int _numCellsInSubDom_X, int _numCellsInSubDom_Y, int _direction);
void CMFD_RestrictionSurfDiffCoef_X(FP_PRECISION *_SurfDiffCoef_CMFD_X, FP_PRECISION *_SurfNDF_CMFD_X, FP_PRECISION *_DiffCoeff_CMFD, \
	FP_PRECISION_FLUX *_CMFDScalarFLux_MOC, FP_PRECISION *_netCurr_CMFD_X, int _numLayersZ, int _numCellsInSubDom_X, \
	int _numCellsInSubDom_Y, int *_cellRTMType, int _numCMFDGroup, FP_PRECISION_GEO *_RTMWidth_X, int _numAizmAng, int _numPolarAng, \
	FP_PRECISION *_AzimWeights, FP_PRECISION *_PolarWeights, FP_PRECISION *_CosTheta, int _SubDomLocID_2D_X, int *_NumSubDomains, \
	FP_PRECISION *_CMFDFlux_MPIRecv, FP_PRECISION *_CMFDNodeD_MPIRecv, int *_MPIComm_SumInd);
void CMFD_RestrictionSurfDiffCoef_Y(FP_PRECISION *_SurfDiffCoef_CMFD_Y, FP_PRECISION *_SurfNDF_CMFD_Y, FP_PRECISION *_DiffCoeff_CMFD, \
	FP_PRECISION_FLUX *_CMFDScalarFLux_MOC, FP_PRECISION *_netCurr_CMFD_Y, int _numLayersZ, int _numCellsInSubDom_X, \
	int _numCellsInSubDom_Y, int *_cellRTMType, int _numCMFDGroup, FP_PRECISION_GEO *_RTMWidth_Y, int _numAizmAng, int _numPolarAng, \
	FP_PRECISION *_AzimWeights, FP_PRECISION *_PolarWeights, FP_PRECISION *_CosTheta, int _SubDomLocID_2D_Y, int *_NumSubDomains, \
	FP_PRECISION *_CMFDFlux_MPIRecv, FP_PRECISION *_CMFDNodeD_MPIRecv, int *_MPIComm_SumInd);
void CMFD_RestrictionSurfDiffCoef_Z(FP_PRECISION *_SurfDiffCoef_CMFD_Z, FP_PRECISION *_SurfNDF_CMFD_Z, FP_PRECISION *_DiffCoeff_CMFD, \
	FP_PRECISION_FLUX *_CMFDScalarFLux_MOC, FP_PRECISION *_netCurr_CMFD_Z, int _numLayersZ, int _numCellsInSubDom_X, \
	int _numCellsInSubDom_Y, int *_cellRTMType, int _numCMFDGroup, FP_PRECISION_GEO *_WidthLayersInCore, int _numAizmAng, int _numPolarAng, \
	FP_PRECISION *_AzimWeights, FP_PRECISION *_PolarWeights, FP_PRECISION *_CosTheta);
FP_PRECISION CMFD_CalcSurfDiffCoef_InnerSurf(FP_PRECISION _nodeDiff, FP_PRECISION _nodeDiff_1, FP_PRECISION _area, FP_PRECISION _area_1);
FP_PRECISION CMFD_CalcSurfDiffCoef_BoundSurf(int _boundCond, FP_PRECISION _nodeDiff, FP_PRECISION _area);
FP_PRECISION CMFD_CalcSurfNDF_InnerSurf(FP_PRECISION _surfCurr, FP_PRECISION _nodeFlux, FP_PRECISION _nodeFlux_neib, FP_PRECISION _surfDiffCoef);
FP_PRECISION CMFD_CalcSurfNDF_BoundSurf(FP_PRECISION boundPos, int _boundCond, FP_PRECISION _surfCurr, FP_PRECISION _nodeFlux, FP_PRECISION _surfDiffCoef);
void MGCMFD_GCEvaluation(int _procID,int _numProcs, int _isFirstInvoke, int _outIter);
void CMFD_EvaluateFissionSource(FP_PRECISION *_cmfd_fissionSour, int _numPinsInDom, int _numCMFDGroup, \
	FP_PRECISION _oneOverKeff, FP_PRECISION_FLUX *_CMFDFlux, FP_PRECISION *_MOCtoCMFD_XS_nf, FP_PRECISION *_MOCtoCMFD_Chi,\
	FP_PRECISION *_CMFDNodeVol);
void CMFD_EvaluateScattSource(FP_PRECISION *_cmfd_scattSour, int _numPinsInDom, int _numCMFDGroup, \
	FP_PRECISION_FLUX *_CMFDFlux, FP_PRECISION *_MOCtoCMFD_XS_s);
void CMFD_EvaluateTotalSource(FP_PRECISION *_cmfd_totalSour, int _numPinsInDom, int _numCMFDGroup, \
	FP_PRECISION _oneOverKeff, FP_PRECISION *_MOCtoCMFD_XS_nf, FP_PRECISION *_MOCtoCMFD_Chi, \
	FP_PRECISION_FLUX *_CMFDFlux, FP_PRECISION *_MOCtoCMFD_XS_s, FP_PRECISION *_CMFDNodeVol);
void CMFD_EvaluateTotalSource_g(FP_PRECISION *_cmfd_totalSour, FP_PRECISION *_totFissSour, int g, \
	int _numPinsInDom, int _numCMFDGroup, FP_PRECISION _oneOverKeff, FP_PRECISION *_MOCtoCMFD_Chi, \
	FP_PRECISION_FLUX *_CMFDFlux, FP_PRECISION *_MOCtoCMFD_XS_s, FP_PRECISION *_CMFDNodeVol);
//void CMFD_ConstructMatrixA_Explicit(FP_PRECISION *_matrix_A, int _numLayerZ, int _numCellsInSubDom_X, int _numCellsInSubDom_Y, \
//	int _numCMFDGroup, FP_PRECISION *_MOCtoCMFD_XS_t, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_X, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_Y, \
//	FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_Z, FP_PRECISION *_MOCtoCMFD_SurfNDF_X, FP_PRECISION *_MOCtoCMFD_SurfNDF_Y, \
//	FP_PRECISION *_MOCtoCMFD_SurfNDF_Z, FP_PRECISION *_RTMWidth_X, FP_PRECISION *_RTMWidth_Y, FP_PRECISION_GEO *_WidthLayersInCore, \
//	int *_cellRTMType, FP_PRECISION *_MOCtoCMFD_XS_s);
FP_PRECISION GetCoefficient_CentralCell(int _surf_Z_Neg, int _surf_Z_Pos, int _surf_X_Neg, \
	int _surf_Y_Neg, int _surf_X_Pos, int _surf_Y_Pos, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_X, \
	FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_Y, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_Z, \
	FP_PRECISION *_MOCtoCMFD_SurfNDF_X, FP_PRECISION *_MOCtoCMFD_SurfNDF_Y, \
	FP_PRECISION *_MOCtoCMFD_SurfNDF_Z, FP_PRECISION _area_xy, FP_PRECISION _area_xz, \
	FP_PRECISION _area_yz, FP_PRECISION _volume, FP_PRECISION *_MOCtoCMFD_XS_t, FP_PRECISION *_MOCtoCMFD_XS_s, \
	int _globPinID, int _numCMFDGroup, int _groupID, int _numPinsInDom);
//void SetValue_Matrix_Diagonal(int _numPinsInDom, int _surf_Z_Neg, int _surf_Z_Pos, int _surf_X_Neg, \
//	int _surf_Y_Neg, int _surf_X_Pos, int _surf_Y_Pos, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_X, \
//	FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_Y, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_Z, \
//	FP_PRECISION *_MOCtoCMFD_SurfNDF_X, FP_PRECISION *_MOCtoCMFD_SurfNDF_Y, \
//	FP_PRECISION *_MOCtoCMFD_SurfNDF_Z, FP_PRECISION _area_xy, FP_PRECISION _area_xz, \
//	FP_PRECISION _area_yz, FP_PRECISION _volume, FP_PRECISION *_MOCtoCMFD_XS_t, FP_PRECISION *_MOCtoCMFD_XS_s, \
//	int _globPinID, int _numCMFDGroup, int _numEle_row, FP_PRECISION *_matrix_A, FP_PRECISION *_CRS_Matrix_Diag);
void SetValue_Matrix_NonDiagonal_Negtive(int _globPinID, int _neib_pin_ID, int _numPin_Dom, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff, \
	FP_PRECISION *_MOCtoCMFD_SurfNDF, int _surf_index, int _numCMFDGroup, int _numEle_row, FP_PRECISION *_matrix_A, FP_PRECISION _area);
void SetValue_Matrix_NonDiagonal_Positive(int _globPinID, int _neib_pin_ID, int _numPin_Dom, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff, \
	FP_PRECISION *_MOCtoCMFD_SurfNDF, int _surf_index, int _numCMFDGroup, int _numEle_row, FP_PRECISION *_matrix_A, FP_PRECISION _area);
FP_PRECISION GetCoefficient_NeibCell_Negtive(FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff, \
	FP_PRECISION *_MOCtoCMFD_SurfNDF, int _surf_index, int _numCMFDGroup, \
	int _groupID, FP_PRECISION _area);
FP_PRECISION GetCoefficient_NeibCell_Positive(FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff, \
	FP_PRECISION *_MOCtoCMFD_SurfNDF, int _surf_index, int _numCMFDGroup, \
	int _groupID, FP_PRECISION _area);
void SetValue_Matrix(int _row_ID, int _col_ID, int _numEle_row, FP_PRECISION _value, FP_PRECISION *_matrix_A);

FP_PRECISION CMFD_Solver_KSP(int _numCellsInDom_3D, int _numCMFDGroup, FP_PRECISION *_CMFD_Matrix_A, \
    FP_PRECISION *_CMFD_ScalarFlux,FP_PRECISION *_CMFD_ScalarFlux_Old, FP_PRECISION *_CMFD_TotalSource, \
	int *_indices, FP_PRECISION _k,FP_PRECISION *_MOCtoCMFD_XS_nf, FP_PRECISION *_MOCtoCMFD_Chi, \
	FP_PRECISION *_MOCtoCMFD_XS_s,FP_PRECISION *_RTMWidth_X, FP_PRECISION *_RTMWidth_Y, FP_PRECISION *_CMFDNodeVol);

FP_PRECISION CMFD_CalcReacRate_SubDom(FP_PRECISION_FLUX *_scalarFlux_subDom, int _numCellsInDom_3D, \
	int _numCMFDGroup, FP_PRECISION *_macroXS, FP_PRECISION *_CMFDNodeVol);

void CMFD_Initialization(int _procID,int _numProcs);
void CMFD_ParaUpdate();

void UpdateScalFlux_CMFDtoMOC(FP_PRECISION_FLUX *_CMFDScalarFLux, FP_PRECISION_FLUX *_MOCScalarFlux, \
	FP_PRECISION_FLUX *_MOCtoCMFD_HetScalFlux,int *_CMFDGroupLim, int _numCMFDGroup, int _numPinsInDom, \
	int _numMOCGroups, int *_numFSRsInRTM, int *_cellRTMType, int *_sumNumFSRsInRTM, \
	int *_sumNumFSRsBasedGlobCellID, FP_PRECISION _CMFD_damp, int _iter_moc, int _scattOrder, int _TotalNumFSRs_subDom);

void UpdateRadialAngFlux_ByScalarFlux(int _numLayer_Z, int _numCellsInDom_Layer, int _numLongtrack_2PI, \
    int _numMOCGroups, int _numPolAng, int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT,\
	int _numCMFDGroup, int *_CMFDGroupLim, FP_PRECISION *_boundaryFlux, FP_PRECISION_FLUX *_CMFDScalarFLux, \
	FP_PRECISION_FLUX *_MOCtoCMFD_HetScalFlux, int _numPinsInDom);

void UpdateAxialAngFlux_CMFDtoMOC(int _check, int _numZLayer, int _numCellsInDom_Layer, \
    int _numMOCGroups,int _numAzimAng, int _numPolAng, int _numCMFDGroup, int *_CMFDGroupLim, \
	FP_PRECISION *_FSRAveAngFlux, FP_PRECISION_FLUX *_CMFDScalarFLux, FP_PRECISION_FLUX *_MOCtoCMFD_HetScalFlux, \
	int _numPinsInDom, int *_numFSRsInRTM, int *_cellRTMType, int *_sumNumFSRsBasedGlobCellID);
void UpdateAxialAngFlux_CMFDtoMOC_SOR(int _numZLayer, int _numCellsInDom_Layer, \
    int _numMOCGroups,int _numAzimAng, int _numPolAng, int _numCMFDGroup, int *_CMFDGroupLim, \
	FP_PRECISION *_FSRAveAngFlux, FP_PRECISION_FLUX *_CMFDScalarFLux, FP_PRECISION_FLUX *_MOCtoCMFD_HetScalFlux, \
	int _numPinsInDom, int *_numFSRsInRTM, int *_cellRTMType, int *_sumNumFSRsBasedGlobCellID);

void CMFD_ReactionRateBalance_Check(int _numLayer_Z, int _numPinsInDom_Layer, int _numCMFDGroup, FP_PRECISION *_MOCtoCMFD_XS_t, \
	FP_PRECISION_GEO *_RTMWidth_X, FP_PRECISION_GEO *_RTMWidth_Y, FP_PRECISION_GEO *_WidthLayersInCore,FP_PRECISION *_CMFD_TotalSource, \
	FP_PRECISION *_CMFD_ScalarFlux, FP_PRECISION *_MOCtoCMFD_NetCurr_X, FP_PRECISION *_MOCtoCMFD_NetCurr_Y, FP_PRECISION *_MOCtoCMFD_NetCurr_Z, \
	int _numCellsInSubDom_X, int _numCellsInSubDom_Y, int *_cellRTMType, FILE *_op, FP_PRECISION _keff);

void SetScatterTerm_Matrix(FP_PRECISION *_matrix_A, int _globPinID, int _numCMFDGroup ,\
    FP_PRECISION _volume, FP_PRECISION *_MOCtoCMFD_XS_s, int _numEle_perRow, int _numPinsInDom);
int GetNumNoneZeroValues_Matrix(int _num, FP_PRECISION *_matrix_A);
int GetNumNoneZeroValuesLU_Matrix(int _num_row, FP_PRECISION *_matrix_A, int _NNZ);
void ConvertToCSR_Matrix(int _num_row, FP_PRECISION *_matrix_A);

FP_DOUBLE MGCMFD_Solver(int _procID, int _numCellsInDom_3D, int _numCMFDGroup, FP_DOUBLE _k, \
	FP_PRECISION *_CMFD_ScalarFlux, FP_PRECISION *_CMFD_ScalarFlux_Old, FP_PRECISION *_x_old_LinearSolver, \
	FP_PRECISION *_CMFD_TotalSource, int *_cellRTMType, FP_PRECISION *_MOCtoCMFD_XS_nf, \
	FP_PRECISION *_MOCtoCMFD_Chi, FP_PRECISION *_MOCtoCMFD_XS_s, int *_CSR_RowStartInd_Matrix_LU, \
	int *_CSR_ColInd_Matrix_LU, FP_PRECISION *_CRS_Matrix, int *_CRS_DiagPointer, FP_PRECISION _SOR_factor, \
	FP_PRECISION _tol_LS, int _maxNumIter_LS, int _maxNumIter_CMFD, FP_PRECISION *_CMFDNodeVol, int _CMFDSolver, int _iter_moc);

int LinearSolver_SOR(int _num_1D, FP_PRECISION *_x, FP_PRECISION *_x_old, FP_PRECISION *_b, \
    int *_CSR_RowStartInd_Matrix_LU, int *_CSR_ColInd_Matrix_LU, int *_CRS_DiagPointer, \
	FP_PRECISION *_CRS_Matrix, FP_PRECISION _SOR_factor, FP_PRECISION _tol,int _maxNumIter,\
	int _isNormInf, int _isNorm2, int _isRmse, FP_PRECISION *_normInf, FP_PRECISION *_norm2, FP_PRECISION *_rmse);
FP_PRECISION CMFD_ComputeLarsenEDCFactor(int _numAizmAng, int _numPolarAng, FP_PRECISION _nodeDiffCoef, \
    FP_PRECISION _width, FP_PRECISION *_AzimWeights, FP_PRECISION *_PolarWeights, FP_PRECISION *_CosTheta);

int CMFD_GetNumNoneZeroEle_MatrixA(int _numLayer_Z, int _numCellsInSubDom_X, int _numCellsInSubDom_Y, \
	int _numCMFDGroup, int _probType, FP_PRECISION *_MOCtoCMFD_XS_s);

int CMFD_GetNumNoneZeroEle_MatrixA_wScat(int _numPinsInDom_3D, int _numCMFDGroup, FP_PRECISION *_MOCtoCMFD_XS_s);
void SetValue_Matrix_NonDiagonal_Negtive_CSR(int g, int _index, int _neib_pin_ID, int _numPin_Dom, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff, \
	FP_PRECISION *_MOCtoCMFD_SurfNDF, int _surf_index, int _numCMFDGroup, int *_CSR_ColInd_Matrix_LU, FP_PRECISION *_CRS_Matrix, FP_PRECISION _area);
FP_PRECISION SetValue_Matrix_NonDiagonal_Negtive_CSR_1(int g, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff, \
	FP_PRECISION *_MOCtoCMFD_SurfNDF, int _surf_index, int _numCMFDGroup, FP_PRECISION _area);
void SetValue_Matrix_NonDiagonal_Positive_CSR(int g, int _index, int _neib_pin_ID, int _numPin_Dom, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff, \
	FP_PRECISION *_MOCtoCMFD_SurfNDF, int _surf_index, int _numCMFDGroup, int *_CSR_ColInd_Matrix_LU, FP_PRECISION *_CRS_Matrix, FP_PRECISION _area);
FP_PRECISION SetValue_Matrix_NonDiagonal_Positive_CSR_1(int g, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff, \
	FP_PRECISION *_MOCtoCMFD_SurfNDF, int _surf_index, int _numCMFDGroup, FP_PRECISION _area);
void CMFD_ConstructMatrixA_CSR(int _numLayerZ, int _numCellsInSubDom_X, int _numCellsInSubDom_Y, \
	int _numCMFDGroup, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_X, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_Y, \
	FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_Z, FP_PRECISION *_MOCtoCMFD_SurfNDF_X, FP_PRECISION *_MOCtoCMFD_SurfNDF_Y, \
	FP_PRECISION *_MOCtoCMFD_SurfNDF_Z, FP_PRECISION_GEO *_RTMWidth_X, FP_PRECISION_GEO *_RTMWidth_Y, \
	FP_PRECISION_GEO *_WidthLayersInCore, int *_cellRTMType, FP_PRECISION *_MOCtoCMFD_XS_t, \
	FP_PRECISION *_MOCtoCMFD_XS_s, FP_PRECISION *_CRS_Matrix, int *_CSR_RowStartInd_Matrix_LU);
void CMFD_ConstructMatrixA_CCNode(int _numLayerZ, int _numCellsInSubDom_X, int _numCellsInSubDom_Y, \
	int _numCMFDGroup, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_X, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_Y, \
	FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_Z, FP_PRECISION *_MOCtoCMFD_SurfNDF_X, FP_PRECISION *_MOCtoCMFD_SurfNDF_Y, \
	FP_PRECISION *_MOCtoCMFD_SurfNDF_Z, FP_PRECISION_GEO *_RTMWidth_X, FP_PRECISION_GEO *_RTMWidth_Y, \
	FP_PRECISION_GEO *_WidthLayersInCore, int *_cellRTMType, FP_PRECISION *_MOCtoCMFD_XS_t, \
	FP_PRECISION *_MOCtoCMFD_XS_s, FP_PRECISION *_CCW, FP_PRECISION *_CCN, FP_PRECISION *_CCE, \
	FP_PRECISION *_CCS, FP_PRECISION *_CCB, FP_PRECISION *_CCT, FP_PRECISION *_AM);
void CMFD_MatrixA_CSR_AllocMem(int _numRow, int _NNZ);

int SetScatterTerm_CSR(int _index, int g, int g1, int *_CSR_ColInd_Matrix_LU, FP_PRECISION *_CRS_Matrix, \
	int _globPinID, int _numCMFDGroup, FP_PRECISION _volume, FP_PRECISION *_MOCtoCMFD_XS_s, int _numPinsInDom);
int SetScatterTerm_CSR_1(int _index, int gTo, int gFrom, FP_PRECISION *_CRS_Matrix, \
	int _globPinID, int _numCMFDGroup, FP_PRECISION _volume, FP_PRECISION *_MOCtoCMFD_XS_s);
void CMFD_GetNoneZeroLocInfo_MatrixCSR(int _numLayerZ, int _numCellsInSubDom_X, int _numCellsInSubDom_Y, \
	int _numCMFDGroup, 	FP_PRECISION *_MOCtoCMFD_XS_s, int *_CSR_RowStartInd_Matrix_LU, int *_CSR_ColInd_Matrix_LU);

//void CMFD_ConstructMatrixA_DiagonalCSR(int _numLayerZ, int _numCellsInSubDom_X, int _numCellsInSubDom_Y, \
//	int _numCMFDGroup, FP_PRECISION *_MOCtoCMFD_XS_t, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_X, \
//	FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_Y, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_Z, \
//	FP_PRECISION *_MOCtoCMFD_SurfNDF_X, FP_PRECISION *_MOCtoCMFD_SurfNDF_Y, FP_PRECISION *_MOCtoCMFD_SurfNDF_Z, \
//	FP_PRECISION *_RTMWidth_X, FP_PRECISION *_RTMWidth_Y, FP_PRECISION_GEO *_WidthLayersInCore, int *_cellRTMType, FP_PRECISION *_MOCtoCMFD_XS_s,\
//	FP_PRECISION *_CRS_Matrix_Diag);
//void CMFD_ConstructMatrixA_NonDiagonalCSR(int _numLayerZ, int _numCellsInSubDom_X, int _numCellsInSubDom_Y, int _numCMFDGroup, \
//	FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_X, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_Y, FP_PRECISION *_MOCtoCMFD_SurfDiffCoeff_Z, \
//	FP_PRECISION *_MOCtoCMFD_SurfNDF_X, FP_PRECISION *_MOCtoCMFD_SurfNDF_Y, FP_PRECISION *_MOCtoCMFD_SurfNDF_Z, FP_PRECISION *_RTMWidth_X, \
//	FP_PRECISION *_RTMWidth_Y, FP_PRECISION_GEO *_WidthLayersInCore, int *_cellRTMType, FP_PRECISION *_MOCtoCMFD_XS_s, \
//	FP_PRECISION *_CRS_Matrix, int *_CSR_RowStartInd_Matrix_LU);

// void CGCMFD_Restriction_Sumation(FP_PRECISION_FLUX *_CMFDScalarFLux_MG, FP_PRECISION_FLUX *_CMFDScalarFLux_CG, \
// 	int *_CMFDGroupLim_CG, int _numCMFDGroup_CG, int _numCMFDGroup_MG, int _numPinsInDom);

void CGCMFD_Restriction_FluxWeighted(FP_PRECISION *_1DXSs_CMFD_MG, FP_PRECISION *_1DXSs_CMFD_CG, \
    FP_PRECISION_FLUX *_CMFDScalarFLux_MG, FP_PRECISION_FLUX *_CMFDScalarFLux_CG, \
	int *_CMFDGroupLim_CG, int _numCMFDGroup_CG, int _numCMFDGroup_MG, int _numPinsInDom);

void CGCMFD_Restriction_2DXSs(FP_PRECISION *_2DXSs_CMFD_MG, FP_PRECISION *_2DXSs_CMFD_CG, \
    FP_PRECISION_FLUX *_CMFDScalarFLux_MG, FP_PRECISION_FLUX *_CMFDScalarFLux_CG, \
	int *_CMFDGroupLim_CG, int _numCMFDGroup_CG, int _numCMFDGroup_MG, int _numPinsInDom);

void CGCMFD_MatrixA_CSR_AllocMem(int _numRow, int _NNZ);

void CGCMFD_GCEvaluation(int _procID,int _numProcs, int _isFirstInvoke);

FP_DOUBLE CGCMFD_Solver(int _procID, int _numCellsInDom_3D, int _numCMFDGroup, FP_DOUBLE _k,\
    FP_PRECISION *_CMFD_ScalarFlux,FP_PRECISION *_CMFD_ScalarFlux_Old,FP_PRECISION *_x_old_LinearSolver,\
	FP_PRECISION *_CMFD_TotalSource, FP_PRECISION *_MOCtoCMFD_XS_nf, FP_PRECISION *_MOCtoCMFD_Chi, \
	FP_PRECISION *_MOCtoCMFD_XS_s, int *_CSR_RowStartInd_Matrix_LU, int *_CSR_ColInd_Matrix_LU, \
	FP_PRECISION *_CRS_Matrix, int *_CRS_DiagPointer, FP_PRECISION _SOR_factor, FP_PRECISION _tol_LS, \
	int _maxNumIter_LS, int _maxNumIter_CMFD, FP_PRECISION *_CMFDNodeVol, int _iter_moc, int _iter_mg);

void UpdateScalFlux_CGtoMGCMFD(FP_PRECISION_FLUX *_CGCMFDScalarFlux, FP_PRECISION_FLUX *_CGCMFDHetScalFlux,\
    FP_PRECISION_FLUX *_CMFDScalarFLux,int *_CMFDGroupLim_CG, int _numCMFDGroup_CG, int _numCMFDGroup_MG, \
	int _numPinsInDom);

FP_PRECISION CMFD_CalcSurfNetCurrent_BoundSurf(FP_PRECISION _boundPos, int _boundCond, \
    FP_PRECISION _surfDiffCoef, FP_PRECISION _surfNDF, FP_PRECISION _nodeFlux);

FP_PRECISION CMFD_CalcSurfNetCurrent_InnerSurf(FP_PRECISION _surfDiffCoef, FP_PRECISION  _surfNDF, \
    FP_PRECISION _nodeFlux, FP_PRECISION _nodeFlux_neib);

void CMFD_GetNetCurrent_X(FP_PRECISION *_SurfDiffCoef_CMFD_X, FP_PRECISION *_SurfNDF_CMFD_X, \
    FP_PRECISION_FLUX *_CMFDScalarFLux, FP_PRECISION *_NetCurr_CMFD_X, int _numCMFDGroup, int _numLayersZ, \
	int _numCellsInSubDom_X, int _numCellsInSubDom_Y, int _SubDomLocID_2D_X, int *_NumSubDomains, \
	FP_PRECISION *_CMFDFlux_MPIRecv, int *_MPIComm_SumInd);

void CMFD_GetNetCurrent_Y(FP_PRECISION *_SurfDiffCoef_CMFD_Y, FP_PRECISION *_SurfNDF_CMFD_Y, \
    FP_PRECISION_FLUX *_CMFDScalarFLux, FP_PRECISION *_NetCurr_CMFD_Y, int _numCMFDGroup, int _numLayersZ, \
	int _numCellsInSubDom_X, int _numCellsInSubDom_Y, int _SubDomLocID_2D_Y, int *_NumSubDomains, \
	FP_PRECISION *_CMFDFlux_MPIRecv, int *_MPIComm_SumInd);

void CMFD_GetNetCurrent_Z(FP_PRECISION *_SurfDiffCoef_CMFD_Z, FP_PRECISION *_SurfNDF_CMFD_Z, \
	FP_PRECISION_FLUX *_CMFDScalarFLux, FP_PRECISION *_NetCurr_CMFD_Z, int _numCMFDGroup, int _numLayersZ, \
	int _numCellsInSubDom_X, int _numCellsInSubDom_Y);

void GMRES_Update(FP_PRECISION *_HessenburgMatrix, FP_PRECISION *_solution, \
	FP_PRECISION *_HessenburgRHSVec, FP_PRECISION *_KrylovOrthBasisVec, int _dimension, \
	int _KrylovDim, int _NumMarixRows, int _precondFlag, FP_PRECISION *_MatrixA_Diag);

int GMRES_Solver(FP_PRECISION *_HessenburgMatrix, FP_PRECISION *_GivensRotations, FP_PRECISION *_HessenburgRHSVec, \
	FP_PRECISION *_KrylovOrthBasisVec, FP_PRECISION *_Residual, int _KrylovDim, FP_PRECISION *_MatrixA, \
	FP_PRECISION *_MatrixA_Diag, FP_PRECISION *_GmresRHSVec, int _NumMarixRows, FP_PRECISION *_solution, \
	int *_MatrixRowStartInd, int *_MatrixColInd, int _numRestart, FP_PRECISION _tolerance, int _precondFlag);

int GMRES_Solver_1(FP_PRECISION *_HessenburgMatrix, FP_PRECISION *_GivensRotations, FP_PRECISION *_HessenburgRHSVec, \
	FP_PRECISION *_KrylovOrthBasisVec, FP_PRECISION *_Residual, int _KrylovDim, FP_PRECISION *_MatrixA, \
	FP_PRECISION *_MatrixA_Diag, FP_PRECISION *_GmresRHSVec, int _NumMarixRows, FP_PRECISION *_solution, \
	int *_MatrixRowStartInd, int *_MatrixColInd, int _numRestart, FP_PRECISION _tolerance, int _precondFlag);

void GMRES_Update_1(FP_PRECISION *_HessenburgMatrix, FP_PRECISION *_solution, \
	FP_PRECISION *_HessenburgRHSVec, FP_PRECISION *_KrylovOrthBasisVec, int _dimension, \
	int _KrylovDim, int _NumMarixRows, int _precondFlag, FP_PRECISION *_MatrixA_Diag);

int GMRES_Solver_2(FP_PRECISION *_HessenburgMatrix, FP_PRECISION *_GivensRotations, FP_PRECISION *_HessenburgRHSVec, \
	FP_PRECISION *_KrylovOrthBasisVec, FP_PRECISION *_Residual, int _KrylovDim, FP_PRECISION *_MatrixA, \
	FP_PRECISION *_MatrixA_Diag, FP_PRECISION *_GmresRHSVec, int _NumMarixRows, FP_PRECISION *_solution, \
	int *_MatrixRowStartInd, int *_MatrixColInd, int _numRestart, FP_PRECISION _tolerance, int _precondFlag);

void ILUFactPrecond(int _n, int _nz_num, int *_ia, int *_ja, FP_PRECISION *_a, \
	int *_ua, FP_PRECISION *_ilu, M_INT32 *_iw);
void JacobiFactPrecond(int _n, int *_ia, int *_ja, FP_PRECISION *_a, \
	int *_ua);
void JacobiFactPrecond_RHS(int _n, int *_ia, int *_ja, FP_PRECISION *_a, \
	int *_ua, FP_PRECISION *_rhs);
void RearrangeCSRMatrix(int _n, int _nz_num, int *_ia, int *_ja, FP_PRECISION *_a);
void DiagPointer_CSRMatrix(int _n, int _nz_num, int *_ia, int *_ja, int *_ua);
void AX_CSR(int _n, int _nz_num, int *_ia, int *_ja, FP_PRECISION *_a, FP_PRECISION *_x, FP_PRECISION *_w);
void ILUPrecond_CSR(int _n, int *_ia, int *_ja, FP_PRECISION *_ilu, int *_ua, FP_PRECISION *_r);
void JacobiPrecond_CSR(int _n, int *_ua, FP_PRECISION *_a, FP_PRECISION *_r);
void MultiplyGivens(FP_PRECISION c, FP_PRECISION s, int k, FP_PRECISION *g);
int PMGMRES_ILU_CSR(int _n, int _nz_num, int *_ia, int *_ja, FP_PRECISION *_a,
	FP_PRECISION *_x, FP_PRECISION *_rhs, int _maxRst, int _kryDim, FP_PRECISION _tolAbs,
	FP_PRECISION _tolRel, int *_ua, FP_PRECISION *_ilu, FP_PRECISION *_res, \
	FP_PRECISION *_g, FP_PRECISION *_h, FP_PRECISION *_v, FP_PRECISION *_givens, FP_PRECISION *_y, int _precond,\
	FP_PRECISION_TIME *_time_prec, FP_PRECISION_TIME *_time_lesSqr,FP_PRECISION_TIME *_time_SpMV, 
	FP_PRECISION_TIME *_time_blas1,FP_PRECISION *_LS_FluxRes);
int LinearSolver_Jacobi(int _num_1D, FP_PRECISION *_x, FP_PRECISION *_x_old, FP_PRECISION *_b, \
    int *_CSR_RowStartInd_Matrix_LU, int *_CSR_ColInd_Matrix_LU, int *_CRS_DiagPointer, \
	FP_PRECISION *_CRS_Matrix, FP_PRECISION _tol,int _maxNumIter,int _isNormInf, \
	int _isNorm2, int _isRmse, FP_PRECISION *_normInf, FP_PRECISION *_norm2, FP_PRECISION *_rmse);
void GetNetCurr_Axial(int _lay, int _currLay, FP_PRECISION _dir, int _numCellsInDom_perLayer, int *_sumNumFSRsBasedGlobCellID, int *_numFSRsInCell, \
	int _numGroups, int _numAzimAng, int _numPolAng, int _groupBoundLowLim, int _groupBoundUpLim, int _polarBoundLowLim, \
	int _polarBoundUpLim, FP_PRECISION *_totalWeights, FP_PRECISION_FLUX *_surfAngFlux, FP_PRECISION *_MOC_NetCurr_Z, \
	int _polOffset, FP_PRECISION *_CosTheta, FP_PRECISION_GEO *_FSRArea_SubDom);

void InitializeCurrent_beforeTS(int _MOCNumLayersZ, int _MOCNumPinsInSubDom_X, int _MOCNumPinsInSubDom_Y, \
	int _NumGroups, FP_PRECISION *_MOCParCurr_X_Pos, FP_PRECISION *_MOCParCurr_X_Neg, FP_PRECISION *_MOCParCurr_Y_Pos, \
	FP_PRECISION *_MOCParCurr_Y_Neg, FP_PRECISION *_MOC_NetCurr_Z);

void CMFD_DampSurfNDF_XYZ(FP_PRECISION _CMFD_Damp, int _numCMFDGroup, int _numLayersZ, int _numCellsInSubDom_X, int _numCellsInSubDom_Y, \
	FP_PRECISION *_SurfNDF_CMFD_X, FP_PRECISION *_SurfNDF_CMFD_Y, FP_PRECISION *_SurfNDF_CMFD_Z, \
	FP_PRECISION *_SurfNDF_CMFD_XOld, FP_PRECISION *_SurfNDF_CMFD_YOld, FP_PRECISION *_SurfNDF_CMFD_ZOld);


void IluFact(int _NumCMFDGroup, int _MOCNumLayersZ, int _MOCNumPinsInSubDom_X, int _MOCNumPinsInSubDom_Y, \
	FP_PRECISION *_Del, FP_PRECISION *_Al, FP_PRECISION *_Au, FP_PRECISION *_AM, FP_PRECISION *_CCW, \
	FP_PRECISION *_CCE, FP_PRECISION *_CCN, FP_PRECISION *_CCS, FP_PRECISION *_Delinv, FP_PRECISION *_Deliau, \
	FP_PRECISION *_Ainvl, FP_PRECISION *_Ainvu, FP_PRECISION *_Ainvd);
void Factorld_g(int g, int _irow, int _ilay, int _MOCNumPinsInSubDom_X, int _numPin_dom, int _numPin_lay, int _NumCMFDGroup, \
	FP_PRECISION *_del, FP_PRECISION *_au, FP_PRECISION *_al, FP_PRECISION *_delinv, FP_PRECISION *_deliau);
void Abild(int g, int _irow, int _ilay, int _MOCNumPinsInSubDom_X, int _numPin_dom, int _numPin_lay, int _NumCMFDGroup, \
	FP_PRECISION *_delinv, FP_PRECISION *_al, FP_PRECISION *_au, FP_PRECISION *_ainvl, FP_PRECISION *_ainvu, \
	FP_PRECISION *_ainvd);
void CMFD_EvaluateTotFissQ(FP_PRECISION *_cmfd_totalSour, int _numPinsInDom, int _numCMFDGroup, \
	FP_PRECISION *_MOCtoCMFD_XS_nf, FP_PRECISION_FLUX *_CMFDFlux, FP_PRECISION *_CMFDNodeVol);
void AxB_g(int ig, FP_PRECISION *b, FP_PRECISION *ab, int nz, int nx, int ny, int ng, FP_PRECISION *ccw, \
	FP_PRECISION *cce, FP_PRECISION *ccn, FP_PRECISION *ccs, FP_PRECISION *ccb, FP_PRECISION *cct, FP_PRECISION *am);
FP_PRECISION scalp_g(int nxyz, FP_PRECISION *x, FP_PRECISION *y);
void sol1d_mg(int ig, int irow, int k, int nz, int nx, int ny, int ng, FP_PRECISION *b, FP_PRECISION *x, \
	FP_PRECISION *y_mg, FP_PRECISION *delinv_mg, FP_PRECISION *al_mg, FP_PRECISION *deliau_mg);
void sol2d_g(int ig, int k, int nz, int nx, int ny, int ng, FP_PRECISION *b, FP_PRECISION *x, FP_PRECISION *s1dl_mg, \
	FP_PRECISION *b01d_mg, FP_PRECISION *ccn, FP_PRECISION *ccs, FP_PRECISION *y_mg, FP_PRECISION *delinv_mg, FP_PRECISION *al_mg, \
	FP_PRECISION *deliau_mg, int *nrnx);
void Minv_g(int ig, int nz, int nx, int ny, int ng, FP_PRECISION *b, FP_PRECISION *x, FP_PRECISION *s_mg, \
	FP_PRECISION *b0_mg, FP_PRECISION *ccb, FP_PRECISION *cct, FP_PRECISION *s1dl_mg, FP_PRECISION *b01d_mg, \
	FP_PRECISION *ccn, FP_PRECISION *ccs, FP_PRECISION *y_mg, FP_PRECISION *delinv_mg, FP_PRECISION *al_mg, \
	FP_PRECISION *deliau_mg, int *nrnx);

void IluGmresSolver_ZYQ(int nz, int nx, int ny, int ng, FP_PRECISION *_CMFD_TotSour_g, FP_PRECISION *_CMFD_TotFissQ, \
	FP_PRECISION _oneOverKeff, FP_PRECISION *_MOCtoCMFD_Chi, FP_PRECISION *_MOCtoCMFD_XS_s, FP_PRECISION *_CMFDNodeVol, \
	FP_PRECISION *flux_in, FP_PRECISION *flux_add_in, FP_PRECISION *flux_cmfd, FP_PRECISION *src_in, FP_PRECISION *ccw, \
	FP_PRECISION *cce, FP_PRECISION *ccn, FP_PRECISION *ccs, FP_PRECISION *ccb, FP_PRECISION *cct, FP_PRECISION *am, \
	FP_PRECISION *vr0_mg, FP_PRECISION *vr_mg, FP_PRECISION *vp_mg, FP_PRECISION *vv_mg, FP_PRECISION *vs_mg, \
	FP_PRECISION *vt_mg, FP_PRECISION *vy_mg, FP_PRECISION *vz_mg, FP_PRECISION *s_mg, FP_PRECISION *b0_mg, \
	FP_PRECISION *s1dl_mg, FP_PRECISION *b01d_mg, FP_PRECISION *y_mg, FP_PRECISION *delinv_mg, FP_PRECISION *al_mg, \
	FP_PRECISION *deliau_mg, int *nrnx, int ninmax, int *flag_flux, float *_neg_flux_count);

void  SolveLinearSystem_g(int ig, int nz, int nx, int ny, int ng, FP_PRECISION *flux_in, FP_PRECISION *flux_add_in, \
	FP_PRECISION *flux_cmfd, FP_PRECISION *src_in, FP_PRECISION *ccw, FP_PRECISION *cce, FP_PRECISION *ccn, \
	FP_PRECISION *ccs, FP_PRECISION *ccb, FP_PRECISION *cct, FP_PRECISION *am, FP_PRECISION *vr0_mg, FP_PRECISION *vr_mg, \
	FP_PRECISION *vp_mg, FP_PRECISION *vv_mg, FP_PRECISION *vs_mg, FP_PRECISION *vt_mg, FP_PRECISION *vy_mg, \
	FP_PRECISION *vz_mg, FP_PRECISION *s_mg, FP_PRECISION *b0_mg, FP_PRECISION *s1dl_mg, FP_PRECISION *b01d_mg, \
	FP_PRECISION *y_mg, FP_PRECISION *delinv_mg, FP_PRECISION *al_mg, FP_PRECISION *deliau_mg, int *nrnx, int ninmax, int *flag_flux);

FP_DOUBLE CMFD_Eigenvalue(int _procID, int _numProcs, FP_DOUBLE _keff, int _nx, int _ny, int _nz, \
    int *_flageig, FP_DOUBLE *_erreig, FP_PRECISION *_QFiss, FP_PRECISION _epseig,\
	FP_DOUBLE *_CMFDGlobalFissRate, FP_DOUBLE *_CMFDGlobalFissRate_Old, int _mg_cg);

void CMFD_ConvCheck_FissQ(int _procID, int _numProcs, int _nx, int _ny, int _nz, int *_flagl2, \
    int *_flaglinf, FP_PRECISION *_qerrlinf, FP_PRECISION *_qrerrl2, FP_PRECISION *_QFiss_old, \
	FP_PRECISION *_QFiss, FP_PRECISION _epsl2, FP_PRECISION _epslinf,FP_PRECISION _psi_max, int *_CMFD_Diver);

FP_PRECISION CMFD_UpdateQFiss(int nx, int ny, int nz, int ng, FP_PRECISION *ferrlinf, FP_PRECISION *_QFiss_old, \
	FP_PRECISION *_QFiss, FP_PRECISION *flux_cmfd,  FP_PRECISION *fluxoo_cmfd, FP_PRECISION *_xs_nf, FP_PRECISION *_CMFDNodeVol);

void SetNegCMFDFlux(int _dim, FP_PRECISION *_cmfdFlux, int *_isNegF, float *_negFCount);

void CMFD_EvaluateTotalSource_onFissQ(FP_PRECISION *_cmfd_totalSour, FP_PRECISION *_totFissSour, \
	int _numPinsInDom, int _numCMFDGroup, FP_DOUBLE _oneOverKeff, FP_PRECISION *_MOCtoCMFD_Chi, \
	FP_PRECISION_FLUX *_CMFDFlux, FP_PRECISION *_MOCtoCMFD_XS_s, FP_PRECISION *_CMFDNodeVol);

void CMFD_EvaluateFissionSource_onFissQ(FP_PRECISION *_cmfd_totalSour, FP_PRECISION *_totFissSour, \
	int _numPinsInDom, int _numCMFDGroup, FP_DOUBLE _oneOverKeff, FP_PRECISION *_MOCtoCMFD_Chi);

void MGCMFDBoundInfoComm_PreNeibInfo(int _procID, int _SubDomLocID_2D_X, int _SubDomLocID_2D_Y, int _SubDomLocID_Z, \
	int _numLayerZ, int _numCellsInSubDom_X, int _numCellsInSubDom_Y, int _numCMFDGroup, int *_NumSubDomains, \
	int *_mess_TarProc, int *_mess_NumData, int ***_mess_DirBound, int *_mess_RecvDir, int *_MPIComm_Index, \
	int _MPIComm_Num, int *_MPIComm_SumInd);

void CGCMFDBoundInfoComm_PreNeibInfo(int _procID, int _numLayerZ, int _numCellsInSubDom_X, int _numCellsInSubDom_Y, int _numCMFDGroup, \
    int *_mess_TarProc, int *_mess_NumData, int ***_mess_DirBound, int *_MPIComm_Index, int _MPIComm_Num, int *_MPIComm_SumInd);

void CMFDBoundInfoComm(int _procID, int _numProcs, int _numCMFDGroup, int _numLayerZ, int _numCellsInSubDom_X, \
	int _numCellsInSubDom_Y, int *_mess_TarProc, FP_PRECISION *_MGCMFD_Para, FP_PRECISION *_MGCMFDComm_send, \
	FP_PRECISION *_MGCMFDComm_recv, int *_mess_NumData, int ***_mess_DirBound, int *_mess_RecvDir, int *_MPIComm_SumInd);

FP_DOUBLE CMFDCompReacRate_SubDom(int _procID, int _nxyz, FP_PRECISION *_QFiss);
FP_DOUBLE CMFDCompReacRate_TotalDom(int _procID, int _nxyz, FP_PRECISION *_QFiss, int _mg_cg);
int GetDataNum_CMFDCommPara(int _numDir, int _nx, int _ny, int _nz, int _ng, int *_MPIComm_SumInd);
void SetCMFDMPI_InBoundSourCoeff(int _procID, int _numProcs, FP_PRECISION *_MPI_InBoundSourCoeff, FP_PRECISION *_SurfDiffCoef_CMFD_X, \
	FP_PRECISION *_SurfNDF_CMFD_X, FP_PRECISION *_SurfDiffCoef_CMFD_Y, FP_PRECISION *_SurfNDF_CMFD_Y, FP_PRECISION *_SurfDiffCoef_CMFD_Z, \
	FP_PRECISION *_SurfNDF_CMFD_Z, int _MPIComm_Num, int *_MPI_InBoundSourCoeff_Index, int *_MPI_InBoundSourCoeff_Dir, FP_PRECISION *_MPI_InBoundSourCoeff_GeoDim);
void SetCMFDTotSour_wInBoundInfo(int _procID, int _numProcs, int _MPIComm_Num, int *_MPIComm_Index, FP_PRECISION *_CMFD_TotalSource, \
	FP_PRECISION *_MPI_InBoundSourCoeff, FP_PRECISION *_MGCMFDComm_recv);

void GetParCurr_2DMOC_InnerBound(FP_PRECISION *_parCurr_MOC_X_Pos, FP_PRECISION *_parCurr_MOC_X_Neg, FP_PRECISION *_parCurr_MOC_Y_Pos, \
	FP_PRECISION *_parCurr_MOC_Y_Neg, FP_PRECISION *_ang_flux, FP_PRECISION *_totalWeights, int _trackID, int _azimID, \
	int _numPolAng, int *_numTrackX, int *_numTrackY, int _numAzimuAng, int _RTMType, int _pinID_Layer, \
	int _surfIndex_Layer_X, int _surfIndex_Layer_Y, int _numMOCGroups, FP_PRECISION_GEO *_RTMWidth_X, FP_PRECISION_GEO *_RTMWidth_Y, \
	int _numCellsInSubDom_X, int _polarBoundLowLim, int _polarBoundUpLim, int *_mess_TarProc);

void TallyMOCPartialCurr_InnerBound(int _procID, int _numProcs, int _numLayers_Z, int _numGroups, \
	FP_PRECISION *_boundaryFlux, int _numAzimAng, int _numPolAng, int _numCellsInDom_Layer, \
	int *_cellRTMGeoType, int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, int *_LTsIndex_AzimAngID, \
	FP_PRECISION *_totalWeights_2DMOC, int _numLongtrack_halfSpace, FP_PRECISION *_trackFlux);

void SetMOCPartCurrToZero_InnerBound(int _procID, int _numProcs, FP_PRECISION *_parCurr_MOC_X_Pos, FP_PRECISION *_parCurr_MOC_X_Neg, \
	FP_PRECISION *_parCurr_MOC_Y_Pos, FP_PRECISION *_parCurr_MOC_Y_Neg, int _MOCParCurrInnerBound_Num, int *_MOCParCurrInnerBound_Index, int *_MOCParCurrInnerBound_Dir);

void SetMOCPartCurrIndex_InnerBound(int _procID, int _numProcs, int _numLaysInSubDom_Z, int _numCellsInSubDom_X, int _numCellsInSubDom_Y, \
	int _numGroups, int *_mess_TarProc, int ***_mess_DirBound, int *_MOCParCurrInnerBound_Index, int *_MOCParCurrInnerBound_Dir);

void SetCMFDMPI_InBoundSourCoeff_Index(int _procID, int _numProcs, int _numCMFDGroup, int _numLayerZ, int _numCellsInSubDom_X, \
	int _numCellsInSubDom_Y, int *_mess_TarProc, int ***_mess_DirBound, int *_MPIComm_SumInd, FP_PRECISION_GEO *_RayTracModDim, \
	FP_PRECISION_GEO *_WidthLayersInCore, int *_MPI_InBoundSourCoeff_Index, int *_MPI_InBoundSourCoeff_Dir, FP_PRECISION *_MPI_InBoundSourCoeff_GeoDim);

#endif //MOC_CMFD_H