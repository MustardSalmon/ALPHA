#ifndef GPUCMFD_H
#define GPUCMFD_H

#if defined(CUDA_USED)
  #include <cuda_runtime.h>
#endif //defined(CUDA_USED)

#include "cublas_v2.h"
#include "cusparse.h"

#include "MOC_CommonMacro.h"
#include "MOC_Geometry.h"
#include "MOC_Input.h"

extern FP_PRECISION *GPUTotalWeights_2DMOC;
extern FP_PRECISION *GPUSpheHarmCoeff;

extern int *GPUNumTracksX,*GPUNumTracksY;
extern FP_PRECISION_GEO *GPURTMWidth_X, *GPURTMWidth_Y, *GPUWidthLayersInCore;
extern int *GPUNumFSRsInRTM;
extern FP_PRECISION_GEO *GPUFSRsAreaInRTM_AntiClockFSR, *GPUFSRArea_SubDom;

extern __constant__ FP_PRECISION_GEO GPURayTracModDim[3];
extern __constant__ int GPUNumSurfaceX[1], GPUNumSurfaceY[1];


#if defined(M_MG_CMFD)

extern FP_PRECISION *GPUCMFDXs_t, *GPUCMFDXs_a, *GPUCMFDXs_f, *GPUCMFDXs_nf, *GPUCMFD_Chi; // 1D macroscopic cross section, [NumCellsInSubDomain*NumCMFDGroups]
extern FP_PRECISION *GPUCMFDXs_s; //2D macroscopic cross section, [NumCellsInSubDomain*NumCMFDGroups*NumCMFDGroups]
extern FP_PRECISION *GPUCMFDHetScalFlux, *GPUCMFDNodeDiffCoeff, *GPUCMFDSurfDiffCoeff_X, *GPUCMFDSurfDiffCoeff_Y;
extern FP_PRECISION *GPUCMFDSurfNDF_X, *GPUCMFDSurfNDF_Y, *GPUCMFDSurfDiffCoeff_Z, *GPUCMFDSurfNDF_Z;
extern FP_PRECISION *GPUCMFDSurfNDF_XOld, *GPUCMFDSurfNDF_YOld, *GPUCMFDSurfNDF_ZOld;
extern FP_PRECISION *GPUMOCParCurr_X_Pos, *GPUMOCParCurr_X_Neg, *GPUMOCParCurr_Y_Pos, *GPUMOCParCurr_Y_Neg, *GPUMOCNetCurr_Z;
extern FP_PRECISION *GPUMOCtoCMFD_ParCurr_X_Pos, *GPUMOCtoCMFD_ParCurr_X_Neg, *GPUMOCtoCMFD_ParCurr_Y_Pos, *GPUMOCtoCMFD_ParCurr_Y_Neg;
extern FP_PRECISION *GPUMOCtoCMFD_NetCurr_X, *GPUMOCtoCMFD_NetCurr_Y, *GPUMOCtoCMFD_NetCurr_Z;
extern FP_PRECISION *GPUCMFDMatrix_A, *GPUCMFDTotalSource, *GPUCMFDScalarFlux, *GPUCMFDScalarFlux_Old, *GPUCMFDScalarFlux_Old_LS;
extern FP_PRECISION *GPUCMFDFluxRelErr, *GPUCMFDNiuFissRate;
extern FP_PRECISION *GPU_MGCMFDTotalFissQ,*GPU_CGCMFDTotalFissQ, *GPUCMFDNodeVol;
// extern FP_PRECISION *GPU_MGGMRESRes, *GPU_MGKrylovOrthBasisVec, *GPU_MGHessenburgRHSVec, *GPU_MGHessenburgMatrix;
// extern FP_PRECISION *GPU_MGNegHess;

extern FP_PRECISION *GPU_MGGMRES_r0,*GPU_MGGMRES_res,*GPU_MGGMRES_beta,*GPU_MGGMRES_u_base,*GPU_MGGMRES_g;
extern FP_PRECISION *GPU_MGGMRES_r0Z;
extern FP_PRECISION *GPU_MGGMRES_e,*GPU_MGGMRES_Hm,*GPU_MGGMRES_w,*GPU_MGGMRES_y,*GPU_MGGMRES_Wm;

extern FP_PRECISION *GPU_CGGMRES_r0,*GPU_CGGMRES_res,*GPU_CGGMRES_beta,*GPU_CGGMRES_u_base,*GPU_CGGMRES_g;
extern FP_PRECISION *GPU_CGGMRES_e,*GPU_CGGMRES_Hm,*GPU_CGGMRES_w,*GPU_CGGMRES_y,*GPU_CGGMRES_Wm;

extern FP_PRECISION *GPU_CGCMFDHetScalFlux, *GPU_CGCMFDNodeDiffCoeff;
extern FP_PRECISION *GPU_CGCMFDXs_t, *GPU_CGCMFDXs_nf, *GPU_CGCMFD_Chi,*GPU_CGCMFDXs_s;
extern FP_PRECISION *GPU_MGCMFDNetCurr_X, *GPU_MGCMFDNetCurr_Y, *GPU_MGCMFDNetCurr_Z;
extern FP_PRECISION *GPU_CGMGtoCGCMFD_NetCurr_X, *GPU_CGMGtoCGCMFD_NetCurr_Y, *GPU_CGMGtoCGCMFD_NetCurr_Z;
extern FP_PRECISION *GPU_CGCMFDSurfDiffCoeff_X, *GPU_CGCMFDSurfDiffCoeff_Y, *GPU_CGCMFDSurfDiffCoeff_Z;
extern FP_PRECISION *GPU_CGCMFDSurfNDF_X, *GPU_CGCMFDSurfNDF_Y, *GPU_CGCMFDSurfNDF_Z;
extern FP_PRECISION *GPU_CGCMFDTotalSource, *GPU_CGCMFDScalarFlux, *GPU_CGCMFDScalarFlux_Old, *GPU_CGCMFDScalarFlux_Old_LS;
extern FP_PRECISION *GPU_CGCMFDFluxRelErr, *GPU_CGCMFDNiuFissRate;

extern int *GPUIndex_1DArray;
extern FP_PRECISION GPUCMFDKeff, GPUCMFDKeff_Old;
extern int *GPU_MGCMFDGroupBound, *GPU_CGCMFDGroupBound;
extern int GPUGridDim_MOCPartCurr_X, GPUGridDim_MOCPartCurr_Y, GPUGridDim_MOCNetCurr_Z;
extern int GPUGridDim_CMFDSurf_X, GPUGridDim_CMFDSurf_Y, GPUGridDim_CMFDSurf_Z,GPUGridDim_CMFDNode,GPUGridDim_CMFDXSs;
extern int GPUGridDim_CMFDNode_Layer;
extern int GPUGridDim_CGCMFDSurf_X, GPUGridDim_CGCMFDSurf_Y, GPUGridDim_CGCMFDSurf_Z,GPUGridDim_CGCMFDNode,GPUGridDim_CGCMFDXSs;
extern int GPU_MGCMFDNumRedThreads, GPU_MGCMFDNumBlackThreads,  GPUGridDim_MGCMFDRed, GPUGridDim_MGCMFDBlack;
extern int GPU_CGCMFDNumRedThreads, GPU_CGCMFDNumBlackThreads,  GPUGridDim_CGCMFDRed, GPUGridDim_CGCMFDBlack;
extern int GPUNumRedLay_OddNx, GPUNumBlackLay_OddNx;

extern FP_PRECISION *GPU_MGCMFDMatrixCSR/*, *GPU_MGCMFDMatrixCSR_Diag*/, *GPU_MGCMFDMatrixCSR_ILUFactor;
extern FP_PRECISION *GPU_CGCMFDMatrixCSR/*, *GPU_CGCMFDMatrixCSR_Diag*/, *GPU_CGCMFDMatrixCSR_ILUFactor;
extern int *GPU_MGCMFDMatrixCSR_RowStartInd, *GPU_MGCMFDMatrixCSR_ColInd, *GPU_MGCMFDDiagPointer;
extern int *GPU_CGCMFDMatrixCSR_RowStartInd, *GPU_CGCMFDMatrixCSR_ColInd, *GPU_CGCMFDDiagPointer;

extern __constant__ int GPUNumMGCMFDGroups[1]; /** The number of MG-M_MG_CMFD groups */
extern __constant__ int GPUNumCGCMFDGroups[1]; /** The number of CG-M_MG_CMFD groups */

extern __constant__ int GPU_MGCMFDNumNoneZeroValueLU_Matrix[1], GPU_MGCMFDNumNoneZeroValue_Matrix[1], GPU_MGCMFDNumValueDiag_Matrix[1];
extern __constant__ int GPU_CGCMFDNumNoneZeroValueLU_Matrix[1], GPU_CGCMFDNumNoneZeroValue_Matrix[1], GPU_CGCMFDNumValueDiag_Matrix[1];
extern __constant__ int GPU_MGCMFDNumRedThreads_Constant[1], GPU_MGCMFDNumBlackThreads_Constant[1];
extern __constant__ int GPU_CGCMFDNumRedThreads_Constant[1], GPU_CGCMFDNumBlackThreads_Constant[1];

extern FP_PRECISION GPU_MGCMFD_Damp;

	extern cusparseHandle_t MG_cusparseHandle;
	extern cusparseMatDescr_t MG_descr_M;
	extern cusparseMatDescr_t MG_descr_L;
	extern cusparseMatDescr_t MG_descr_U;
	extern csrilu02Info_t MG_info_M;
	extern csrsv2Info_t  MG_info_L;
	extern csrsv2Info_t  MG_info_U;

	extern void *MG_pBuffer;
	extern const cusparseSolvePolicy_t MG_policy_M;
	extern const cusparseSolvePolicy_t MG_policy_L;
	extern const cusparseSolvePolicy_t MG_policy_U;
	extern const cusparseOperation_t MG_trans_L;
	extern const cusparseOperation_t MG_trans_U;

	extern cusparseHandle_t CG_cusparseHandle;
	extern cusparseMatDescr_t CG_descr_M;
	extern cusparseMatDescr_t CG_descr_L;
	extern cusparseMatDescr_t CG_descr_U;
	extern csrilu02Info_t CG_info_M;
	extern csrsv2Info_t  CG_info_L;
	extern csrsv2Info_t  CG_info_U;

	extern void *CG_pBuffer;
	extern const cusparseSolvePolicy_t CG_policy_M;
	extern const cusparseSolvePolicy_t CG_policy_L;
	extern const cusparseSolvePolicy_t CG_policy_U;
	extern const cusparseOperation_t CG_trans_L;
	extern const cusparseOperation_t CG_trans_U;

extern int GPU_MGGmres_blocksPerGrid;
extern int GPU_CGGmres_blocksPerGrid;

extern float *GPU_MGCMFDNegFluxCount, *GPU_CGCMFDNegFluxCount;

extern int *GPUMGCMFD_MOCParCurrInnerBound_Index, *GPUMGCMFD_MOCParCurrInnerBound_Dir, *GPUCMFDComm_TarProc, *GPUMGCMFDData_MPIComm_Index;
extern FP_PRECISION *GPUMGCMFD_MOCParCurrInnerBound_AngF;
extern FP_PRECISION *GPUMGCMFDFlux_MPIRecv, *GPUMGCMFDNodeD_MPIRecv;
extern int *GPUNumSubDomains, *GPUMGCMFDData_MPIComm_SumInd;
extern int *GPUMGCMFDMPI_InBoundSourCoeff_Index, *GPUMGCMFDMPI_InBoundSourCoeff_Dir;
extern FP_PRECISION *GPUMGCMFDMPI_InBoundSourCoeff_GeoDim, *GPUMGCMFDMPI_InBoundSourCoeff;

extern int *GPUCGCMFDData_MPIComm_SumInd, *GPUCGCMFDData_MPIComm_Index;
extern FP_PRECISION *GPUCGCMFDFlux_MPIRecv, *GPUCGCMFDNodeD_MPIRecv;
extern FP_PRECISION *GPUCGCMFDMPI_InBoundSourCoeff;
extern int *GPUCGCMFDMPI_InBoundSourCoeff_Index, *GPUCGCMFDMPI_InBoundSourCoeff_Dir;
extern FP_PRECISION *GPUCGCMFDMPI_InBoundSourCoeff_GeoDim;

extern M_INT32 *Dev_MGCMFDMatrixCSR_RowStartInd, *Dev_MGCMFDMatrixCSR_ColInd;
extern M_INT32 *Dev_CGCMFDMatrixCSR_RowStartInd, *Dev_CGCMFDMatrixCSR_ColInd;

//#if defined(M_MG_CMFD)
__device__ void  GPUGetParCurr_2DMOC_AngFReg(FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, \
    FP_PRECISION *_GPUMOCParCurr_Y_Pos, FP_PRECISION *_GPUMOCParCurr_Y_Neg, FP_PRECISION angFlux0, FP_PRECISION angFlux1, \
    FP_PRECISION angFlux2, int _trackID, int _azimID, int g,int *_GPUNumTracksX, int *_GPUNumTracksY, int _iftrackStart, \
    int _RTMType, int _pinID_Layer, int _surfIndex_Layer_X, int _surfIndex_Layer_Y,FP_PRECISION *_GPURTMWidth_X, \
    FP_PRECISION *_GPURTMWidth_Y);

__device__ FP_PRECISION_FLUX GPUCMFD_Restriction_Flux_G(FP_PRECISION_FLUX *_GPUScalarFlux_layer,int _groupID_MOC, \
	int _index_fsr, int _RTMType,  FP_PRECISION_GEO *_GPUFSRsAreaInRTM, int *_GPUNumFSRsInRTM, int *_GPUSumNumFSRsInRTM);

__device__ FP_PRECISION GPUCMFD_Restriction_1DXSs_G(FP_PRECISION_FLUX *_GPUScalarFlux_layer, \
	int _groupID_MOC, int _index_fsr, int _index_fsr_lay, int _RTMType, FP_PRECISION_GEO *_GPUFSRsAreaInRTM, \
	int *_GPUNumFSRsInRTM, int *_GPUSumNumFSRsInRTM, FP_PRECISION *_GPU1DXSs_MOC);

__device__ FP_PRECISION GPUCMFD_CalcSurfDiffCoef_BoundSurf(int _boundCond, FP_PRECISION _nodeDiff, FP_PRECISION _area);

__device__ FP_PRECISION GPUCMFD_CalcSurfNDF_BoundSurf(FP_PRECISION boundPos, int _boundCond, \
	FP_PRECISION _surfCurr, FP_PRECISION _nodeFlux, FP_PRECISION _surfDiffCoef);

__device__ FP_PRECISION GPUCMFD_CalcSurfDiffCoef_InnerSurf(FP_PRECISION _nodeDiff, \
	FP_PRECISION _nodeDiff_1, FP_PRECISION _area, FP_PRECISION _area_1);

__device__ FP_PRECISION GPUCMFD_CalcSurfNDF_InnerSurf(FP_PRECISION _surfCurr, FP_PRECISION _nodeFlux, \
	FP_PRECISION _nodeFlux_neib, FP_PRECISION _surfDiffCoef);

__device__ FP_PRECISION GPUCMFD_CalcSurfNetCurrent_BoundSurf(FP_PRECISION _boundPos, int _boundCond, \
    FP_PRECISION _surfDiffCoef, FP_PRECISION _surfNDF, FP_PRECISION _nodeFlux);

__device__ FP_PRECISION GPUCMFD_CalcSurfNetCurrent_InnerSurf(FP_PRECISION _surfDiffCoef, FP_PRECISION  _surfNDF, \
    FP_PRECISION _nodeFlux, FP_PRECISION _nodeFlux_neib);

__device__ FP_PRECISION GPUGetCoefficient_CentralCell(int _globPinID, int _groupID, int _surf_Z_Neg, int _surf_Z_Pos, \
	int _surf_X_Neg, int _surf_Y_Neg, int _surf_X_Pos, int _surf_Y_Pos, FP_PRECISION *_GPUCMFDSurfDiffCoeff_X, \
	FP_PRECISION *_GPUCMFDSurfDiffCoeff_Y, FP_PRECISION *_GPUCMFDSurfDiffCoeff_Z, FP_PRECISION *_GPUCMFDSurfNDF_X, \
	FP_PRECISION *_GPUCMFDSurfNDF_Y, FP_PRECISION *_GPUCMFDSurfNDF_Z, FP_PRECISION _area_xy, FP_PRECISION _area_xz, \
	FP_PRECISION _area_yz, FP_PRECISION _volume, FP_PRECISION *_GPUCMFDXs_t, FP_PRECISION *_GPUCMFDXs_s, int _GPUNumCMFDGroups);

//__device__ void GPUSetValue_Matrix_Diagonal(int _index, int _groupID, int _surf_Z_Neg, int _surf_Z_Pos, int _surf_X_Neg, \
//	int _surf_Y_Neg, int _surf_X_Pos, int _surf_Y_Pos, FP_PRECISION *_GPUCMFDSurfDiffCoeff_X, \
//	FP_PRECISION *_GPUCMFDSurfDiffCoeff_Y, FP_PRECISION *_GPUCMFDSurfDiffCoeff_Z, FP_PRECISION *_GPUCMFDSurfNDF_X, \
//	FP_PRECISION *_GPUCMFDSurfNDF_Y, FP_PRECISION *_GPUCMFDSurfNDF_Z, FP_PRECISION _area_xy, FP_PRECISION _area_xz, \
//	FP_PRECISION _area_yz, FP_PRECISION _volume, FP_PRECISION *_GPUCMFDXs_t, FP_PRECISION *_GPUCMFDXs_s, \
//	int _globPinID, FP_PRECISION *_GPUMatrixCSR_Diag, int _GPUNumCMFDGroups);

__device__ FP_PRECISION GPUSetValue_MatrixNonDiagonal_Negtive_CSR(int _groupID, int _surf_index, FP_PRECISION _area, \
	FP_PRECISION *_GPUCMFDSurfDiffCoeff_Dir, FP_PRECISION *_GPUCMFDSurfNDF_Dir, int _GPUNumCMFDGroups);

__device__ FP_PRECISION GPUSetValue_MatrixNonDiagonal_Positive_CSR(int _groupID, int _surf_index, FP_PRECISION _area, \
	FP_PRECISION *_GPUCMFDSurfDiffCoeff_Dir, FP_PRECISION *_GPUCMFDSurfNDF_Dir, int _GPUNumCMFDGroups);

__device__ int GPUSetScatterTerm_MatrixCSR(int _index, int _globPinID, int gTo, int gFrom, \
	FP_PRECISION *_GPUMatrixCSR_LU, FP_PRECISION _volume, FP_PRECISION *_GPUCMFDXs_s, int _GPUNumCMFDGroups);

__device__ int GPUSetScatterTerm_MatrixCSR_1(int _index, int _globPinID, int gTo, int gFrom, \
	FP_PRECISION *_GPUMatrixCSR, FP_PRECISION _volume, FP_PRECISION *_GPUCMFDXs_s, int _GPUNumCMFDGroups);

__global__ void GPUCMFD_Restriction_Flux(int _layID, int _dimLayer, FP_PRECISION_FLUX *_GPUCMFDHetScalFlux, \
	FP_PRECISION_FLUX *_GPUScalarFlux_layer, int *_GPUCMFDGroupBound, FP_PRECISION_GEO *_GPUFSRsAreaInRTM, \
	int *_GPUNumFSRsInRTM, int *_GPUCellRTMType_SubDom, int *_GPUSumNumFSRsInRTM, int *_GPUSumNumFSRsBasedGlobCellID);

__global__ void GPUCMFD_Restriction_1DXSs(int _layID, int _dimLayer, FP_PRECISION *_GPU1DXSs_CMFD, FP_PRECISION *_GPU1DXSs_MOC, \
	FP_PRECISION_FLUX *_GPUScalarFlux_layer, int *_GPUCMFDGroupBound, \
	FP_PRECISION_GEO *_GPUFSRsAreaInRTM, int *_GPUNumFSRsInRTM, int *_GPUCellRTMType_SubDom, \
	int *_GPUSumNumFSRsInRTM, int *_GPUSumNumFSRsBasedGlobCellID);

__global__ void GPUCMFD_Restriction_2DXSs(int _layID, int _dimLayer, FP_PRECISION *_GPU2DXSs_CMFD, FP_PRECISION *_GPU2DXSs_MOC, \
	FP_PRECISION_FLUX *_GPUScalarFlux_layer, int *_GPUCMFDGroupBound, \
	FP_PRECISION_GEO *_GPUFSRsAreaInRTM, int *_GPUNumFSRsInRTM, int *_GPUCellRTMType_SubDom, \
	int *_GPUSumNumFSRsInRTM, int *_GPUSumNumFSRsBasedGlobCellID, M_UINT64 *_GPUMacroXS_s_NZRowStartID_Layer, M_UINT64 *_GPUMacroXS_s_NZColStartID);

__global__ void GPUCMFD_Restriction_Chi(int _layID, int _dimLayer, FP_PRECISION *_GPUCMFD_Chi, FP_PRECISION *_GPUMacroXS_chi, \
	FP_PRECISION *_GPUMacroXS_nf, FP_PRECISION_FLUX *_GPUScalarFlux_layer, int *_GPUCMFDGroupBound, \
	FP_PRECISION_GEO *_GPUFSRsAreaInRTM, int *_GPUNumFSRsInRTM, int *_GPUCellRTMType_SubDom, \
	int *_GPUSumNumFSRsInRTM, int *_GPUSumNumFSRsBasedGlobCellID);

__global__ void GPUCMFD_Restriction_DiffCoeff(int _layID, int _dimLayer, FP_PRECISION *_GPUDiffCoeff_CMFD, FP_PRECISION *_GPUTrXSs_MOC, \
	FP_PRECISION_FLUX *_GPUScalarFlux_layer, FP_PRECISION_FLUX *_GPUCMFDHetScalFlux, int *_GPUCMFDGroupBound, \
	FP_PRECISION_GEO *_GPUFSRsAreaInRTM, int *_GPUNumFSRsInRTM, int *_GPUCellRTMType_SubDom, \
	int *_GPUSumNumFSRsInRTM, int *_GPUSumNumFSRsBasedGlobCellID);

__global__ void GPUCMFD_RestrictionCurr_GroupCond(int _dim, FP_PRECISION *_GPUMOCtoCMFD_ParCurr, \
	FP_PRECISION *_GPUMOCParCurr, int *_GPUCMFDGroupBound);

__global__ void GPUCMFD_RestrictionNetCurr(int _dim, FP_PRECISION *_GPUMOCtoCMFD_NetCurr, \
	FP_PRECISION *_GPUMOCtoCMFD_ParCurr_Pos, FP_PRECISION *_GPUMOCtoCMFD_ParCurr_Neg);

__global__ void GPUCMFD_RestrictionSurfDiffCoef_X(int _dim, FP_PRECISION *_GPUCMFDSurfDiffCoeff_X, \
	FP_PRECISION *_GPUCMFDSurfNDF_X, FP_PRECISION *_GPUCMFDNodeDiffCoeff, FP_PRECISION_FLUX *_GPUCMFDHetScalFlux, \
	FP_PRECISION *_GPUMOCtoCMFD_NetCurr_X, int *_GPUCellRTMType_SubDom, FP_PRECISION_GEO *_GPURTMWidth_X, int _numCMFDGroup, \
	int _SubDomLocID_2D_X, int *_NumSubDomains, FP_PRECISION *_CMFDFlux_MPIRecv, FP_PRECISION *_CMFDNodeD_MPIRecv, int *_MPIComm_SumInd);

__global__ void GPUCMFD_RestrictionSurfDiffCoef_Y(int _dim, FP_PRECISION *_GPUCMFDSurfDiffCoeff_Y, \
	FP_PRECISION *_GPUCMFDSurfNDF_Y, FP_PRECISION *_GPUCMFDNodeDiffCoeff, FP_PRECISION_FLUX *_GPUCMFDHetScalFlux, \
	FP_PRECISION *_GPUMOCtoCMFD_NetCurr_Y, int *_GPUCellRTMType_SubDom, FP_PRECISION_GEO *_GPURTMWidth_Y, int _numCMFDGroup, \
	int _SubDomLocID_2D_Y, int *_NumSubDomains, FP_PRECISION *_CMFDFlux_MPIRecv, FP_PRECISION *_CMFDNodeD_MPIRecv, int *_MPIComm_SumInd);

__global__ void GPUCMFD_RestrictionSurfDiffCoef_Z(int _dim, FP_PRECISION *_GPUCMFDSurfDiffCoeff_Z, \
	FP_PRECISION *_GPUCMFDSurfNDF_Z, FP_PRECISION *_GPUCMFDNodeDiffCoeff, FP_PRECISION_FLUX *_GPUCMFDHetScalFlux, \
	FP_PRECISION *_GPUMOCtoCMFD_NetCurr_Z, int *_GPUCellRTMType_SubDom, FP_PRECISION_GEO *_GPUWidthLayersInCore, int _numCMFDGroup);

//__global__ void GPUSetMatrix_Diagonal(int *_GPUCellRTMType_SubDom,FP_PRECISION *_GPUCMFDSurfDiffCoeff_X, \
//	FP_PRECISION *_GPUCMFDSurfDiffCoeff_Y, FP_PRECISION *_GPUCMFDSurfDiffCoeff_Z, FP_PRECISION *_GPUCMFDSurfNDF_X, \
//	FP_PRECISION *_GPUCMFDSurfNDF_Y, FP_PRECISION *_GPUCMFDSurfNDF_Z, FP_PRECISION *_GPUCMFDXs_t, FP_PRECISION *_GPUCMFDXs_s, \
//	FP_PRECISION *_GPUMatrixCSR_Diag, FP_PRECISION_GEO *_GPURTMWidth_X, FP_PRECISION_GEO *_GPURTMWidth_Y, \
//	FP_PRECISION_GEO *_GPUWidthLayersInCore, int _GPUMFDNumValueDiag_Matrix, int _GPUNumCMFDGroups);

__global__ void GPUSetMatrix_NoneDiagonal(int *_GPUCellRTMType_SubDom,FP_PRECISION *_GPUCMFDSurfDiffCoeff_X, \
	FP_PRECISION *_GPUCMFDSurfDiffCoeff_Y, FP_PRECISION *_GPUCMFDSurfDiffCoeff_Z, FP_PRECISION *_GPUCMFDSurfNDF_X, \
	FP_PRECISION *_GPUCMFDSurfNDF_Y, FP_PRECISION *_GPUCMFDSurfNDF_Z, FP_PRECISION *_GPUCMFDXs_s, \
	FP_PRECISION *_GPURTMWidth_X, FP_PRECISION *_GPURTMWidth_Y, FP_PRECISION_GEO *_GPUWidthLayersInCore, \
	int *_GPUMatrixCSR_RowStartInd_LU, FP_PRECISION *_GPUMatrixCSR_LU,int _GPUMFDNumValueDiag_Matrix,int _GPUNumCMFDGroups);

__global__ void GPUCMFD_EvaluateTotalSource(int *_GPUCellRTMType_SubDom, FP_PRECISION_GEO *_GPURTMWidth_X, \
	FP_PRECISION_GEO *_GPURTMWidth_Y, FP_PRECISION_GEO *_GPUWidthLayersInCore, FP_PRECISION *_GPUCMFDTotalSource, \
	FP_PRECISION *_GPUCMFD_Chi, FP_PRECISION *_GPUCMFDXs_nf, FP_PRECISION *_GPUCMFDXs_s, FP_PRECISION *_GPUCMFDScalarFlux, \
	FP_PRECISION _oneOverKeff, int _dim, int _GPUNumCMFDGroups);

__global__ void GPUCMFD_EvaluateTotalFissQ(FP_PRECISION *_GPUCMFDTotalFissQ, \
	FP_PRECISION *_GPUCMFDXs_nf, FP_PRECISION *_GPUCMFDScalarFlux, \
	int _dim, int _GPUNumCMFDGroups, FP_PRECISION *_GPUCMFDNodeVol);

__global__ void GPU_LS_Jacobi(FP_PRECISION *_xNew,FP_PRECISION *_xOld, FP_PRECISION *_b, \
	FP_PRECISION *_GPUMatrixCSR, int *_GPU_MGCMFDDiagPointer, \
	int *_GPUMatrixCSR_RowStartInd, int *_GPUMatrixCSR_ColInd, \
	int _CMFDNumValueDiag_Matrix);

__global__ void GPUCMFD_CalcReacRate_Vector(int _numCMFDGroup, FP_PRECISION *_GPUCMFDScalarFlux, \
	FP_PRECISION *_GPUCMFDXs_nf, FP_PRECISION_GEO *_GPURTMWidth_X, FP_PRECISION_GEO *_GPURTMWidth_Y, \
	FP_PRECISION_GEO *_GPUWidthLayersInCore, FP_PRECISION *_GPUCMFDNiuFissRate, int *_GPUCellRTMType_SubDom,int _dim);

__global__ void GPUUpdateScalFlux_CMFDtoMOC(FP_PRECISION_FLUX *_GPUCMFDScalarFlux, \
	FP_PRECISION_FLUX *_GPUCMFDHetScalFlux, FP_PRECISION_FLUX *_MOCScalarFlux, \
	int *_GPUCMFDGroupBound, int *_GPUCellRTMType_SubDom, int *_GPUNumFSRsInRTM, \
	int *_GPUSumNumFSRsBasedGlobCellID, FP_PRECISION _GPU_CMFDDamp);

__global__ void GPUUpdateRadialAngFlux_ByScalarFlux(int _dim, int *_GPUCMFDGroupBound, int *_GPUTrackIndexInLTs,\
	struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, FP_PRECISION_FLUX *_GPUCMFDScalarFlux, \
	FP_PRECISION_FLUX *_GPUCMFDHetScalFlux, FP_PRECISION_FLUX *_GPUBoundaryFlux);

__global__ void GPUUpdateAxialAngFlux_ByScalarFlux(int _dim, FP_PRECISION_FLUX *_GPUCMFDScalarFlux, \
	FP_PRECISION_FLUX *_GPUCMFDHetScalFlux, FP_PRECISION *_GPUFSRAveAngFlux_Bott, \
	int *_GPUCMFDGroupBound, int *_GPUCellRTMType_SubDom, int *_GPUNumFSRsInRTM, \
	int *_GPUSumNumFSRsBasedGlobCellID);

__global__ void GPULinearSolver_BSOR_EvenNx(int _numCMFDGroup, FP_PRECISION *_x, \
	FP_PRECISION *_b, FP_PRECISION *_GPUMatrixCSR, int *_GPU_MGCMFDDiagPointer, \
	int *_GPUMatrixCSR_RowStartInd, int *_GPUMatrixCSR_ColInd, FP_PRECISION _SOR_factor, \
	int _dim, int _offset);

void GPULinearSolver_BSOR_EvenNx_test(int _numCMFDGroup, int _MOCNumPinsInSubDom_X, \
	int _MOCNumPinsInSubDom_Y, int _dim, int _offset);
void GPULinearSolver_BSOR_OddNxOddNy_test(int _numCMFDGroup, int _dim, int _offset);
void GPULinearSolver_BSOR_OddNxEvenNy_test(int _numCMFDGroup,	int _dim, int _offset, \
	int _nThRBLay,int _MOCNumPinsInSubDom_X, int _MOCNumPinsInSubDom_Y);

__global__ void GPULinearSolver_BSOR_OddNxEvenNy(int _numCMFDGroup, FP_PRECISION *_x, \
	FP_PRECISION *_b, FP_PRECISION *_GPUMatrixCSR, int *_GPU_MGCMFDDiagPointer, \
	int *_GPUMatrixCSR_RowStartInd, int *_GPUMatrixCSR_ColInd, FP_PRECISION _SOR_factor, \
	int _dim, int _offset, int _nThRBLay);

__global__ void GPULinearSolver_BSOR_OddNxOddNy(int _numCMFDGroup, FP_PRECISION *_x, \
	FP_PRECISION *_b, FP_PRECISION *_GPUMatrixCSR, int *_GPU_MGCMFDDiagPointer, \
	int *_GPUMatrixCSR_RowStartInd, int *_GPUMatrixCSR_ColInd, FP_PRECISION _SOR_factor, \
	int _dim, int _offset);

__global__ void GPUCMFD_ConstructMatrixA_CSR(int *_GPUCellRTMType_SubDom, FP_PRECISION *_GPUCMFDSurfDiffCoeff_X, \
	FP_PRECISION *_GPUCMFDSurfDiffCoeff_Y, FP_PRECISION *_GPUCMFDSurfDiffCoeff_Z, FP_PRECISION *_GPUCMFDSurfNDF_X, \
	FP_PRECISION *_GPUCMFDSurfNDF_Y, FP_PRECISION *_GPUCMFDSurfNDF_Z, FP_PRECISION *_GPUCMFDXs_t, FP_PRECISION *_GPUCMFDXs_s, \
	FP_PRECISION_GEO *_GPURTMWidth_X, FP_PRECISION_GEO *_GPURTMWidth_Y, FP_PRECISION_GEO *_GPUWidthLayersInCore, \
	int *_GPUMatrixCSR_RowStartInd, FP_PRECISION *_GPUMatrixCSR, int _GPUMFDNumValueDiag_Matrix, int _GPUNumCMFDGroups);

FP_PRECISION GPUCMFD_CalcReacRate_SubDom(int _numCMFGroup, FP_PRECISION *_GPUCMFDScalarFlux, \
	FP_PRECISION *_GPUCMFDXs_nf, FP_PRECISION_GEO *_GPURTMWidth_X, FP_PRECISION_GEO *_GPURTMWidth_Y, \
	FP_PRECISION_GEO *_GPUWidthLayersInCore, FP_PRECISION *_GPUCMFDNiuFissRate, cublasHandle_t _handle, \
	int _dim, int *_GPUCellRTMType_SubDom, int _GPUGridDim_CMFDNode);

int GPUCMFD_GetNumNoneZeroEle_MatrixA(int _numLayer_Z, int _numCellsInSubDom_X, int _numCellsInSubDom_Y, \
	int _numCMFDGroup, FP_PRECISION *_CMFD_XS_s, int _probType);
int GPUCMFD_GetNumNoneZeroLU_MatrixA_Scat(int _numPinsInDom_3D, int _numCMFDGroup, FP_PRECISION *_MOCtoCMFD_XS_s);
void GPU_MGCMFDMatrixCSR_AllocMem(int _numDiag, int _NNZLU, M_INT32 _MGCMFD_LinearSolver);
void GPUCMFD_GetNoneZeroIndexInfo_MatrixCSR(int _numLayerZ, int _numCellsInSubDom_X, int _numCellsInSubDom_Y, \
	int _numCMFDGroup, FP_PRECISION *_MOCtoCMFD_XS_s, int *_CSR_RowStartInd_Matrix, int *_CSR_ColInd_Matrix,\
	M_INT32 _ProblemType);

int GPULinearSolver_Jacobi(FP_PRECISION *_xNew, FP_PRECISION *_xOld, FP_PRECISION *_b, \
	FP_PRECISION *_GPUMatrixCSR, int *_GPU_MGCMFDDiagPointer, int *_GPUMatrixCSR_RowStartInd, \
	int *_GPUMatrixCSR_ColInd, int _dim, FP_PRECISION *_xRelErr, int _MaxNumIter_LS, FP_PRECISION _tol, \
	cublasHandle_t _handle, int _GPUGridDim_CMFDNode, int _isNormInf, int _isNorm2, int _isRmse, \
	FP_PRECISION *_normInf, FP_PRECISION *_norm2, FP_PRECISION *_rmse);

// int GPULinearSolver_CGCMFD(int _numCMFDGroup, FP_PRECISION *_xNew, FP_PRECISION *_xOld, FP_PRECISION *_b, \
// 	FP_PRECISION *_GPUMatrixCSR_LU, FP_PRECISION *_GPUMatrixCSR_Diag, int *_GPUMatrixCSR_RowStartInd_LU, \
// 	int *_GPUMatrixCSR_ColInd_LU, int _dim, FP_PRECISION *_xRelErr, int _MaxNumIter_LS, FP_PRECISION _tol,\
// 	cublasHandle_t _handle, FP_PRECISION _SOR_factor, int _GPUNumRedThreads, int _GPUNumBlackThreads,\
// 	int _GPUGridDim_Red, int _GPUGridDim_Black, int _GPUGridDim_CMFDNode);

FP_PRECISION GPUKeffEvaluation(FP_PRECISION _tot_vsigf_tem,FP_PRECISION _tot_vsigf_old,\
	FP_PRECISION _keff_old);

void GPU_LS_BSOR(int _numCMFDGroup, FP_PRECISION *_x, FP_PRECISION *_b, \
	FP_PRECISION *_GPUMatrixCSR, int *_GPU_MGCMFDDiagPointer, int *_GPUMatrixCSR_RowStartInd, \
	int *_GPUMatrixCSR_ColInd, FP_PRECISION _SOR_factor, int _GPUNumRedThreads, int _GPUNumBlackThreads, \
	int _GPUGridDim_Red, int _GPUGridDim_Black, int _MOCNumPinsInSubDom_X, int _MOCNumPinsInSubDom_Y,\
    int _GPUNumThreadBlocks, int _GPUNumThreadPerBlock, int _GPUNumRedLay_OddNx, int _GPUNumBlackLay_OddNx);
	
// void GPULinearSolver_BSOR_CGCMFD(FP_PRECISION *_x, FP_PRECISION *_b, \
// 	FP_PRECISION *_GPUMatrixCSR_LU, FP_PRECISION *_GPUMatrixCSR_Diag, \
// 	int *_GPUMatrixCSR_RowStartInd_LU, int *_GPUMatrixCSR_ColInd_LU,\
// 	FP_PRECISION _SOR_factor, int _GPUGridDim_Red, int _GPUGridDim_Black);

__global__ void GPU_CGCMFDRestriction_Sumation(FP_PRECISION *_CMFDPara_MG, FP_PRECISION *_CMFDPara_CG, \
	int *_GPUCMFDGroupBound, int _dim);

__global__ void GPU_CGCMFDRestriction_FluxWeighted(FP_PRECISION *_CMFDPara_MG, FP_PRECISION *_CMFDPara_CG, \
    FP_PRECISION_FLUX *_CMFDScalarFLux_MG, FP_PRECISION_FLUX *_CMFDScalarFLux_CG, \
	int *_GPUCMFDGroupBound, int _dim);

__global__ void GPU_CGCMFDRestriction_2DXSs(FP_PRECISION *_2DXSs_CMFD_MG, FP_PRECISION *_2DXSs_CMFD_CG, \
    FP_PRECISION_FLUX *_CMFDScalarFLux_MG, FP_PRECISION_FLUX *_CMFDScalarFLux_CG, \
	int *_GPUCMFDGroupBound, int _dim);

__global__ void GPUUpdateScalFlux_CGtoMGCMFD(FP_PRECISION_FLUX *_CGCMFDScalarFlux, FP_PRECISION_FLUX *_CGCMFDHetScalFlux,\
    FP_PRECISION_FLUX *_CMFDScalarFLux,int *_GPUCMFDGroupBound, int _dim);

__global__ void GPUCMFD_GetNetCurrent_X(int _dim, FP_PRECISION *_GPUCMFDSurfDiffCoeff_X, \
	FP_PRECISION *_GPUCMFDSurfNDF_X, FP_PRECISION_FLUX *_CMFDScalarFLux, \
	FP_PRECISION *_GPU_MGCMFDNetCurr_X, int _SubDomLocID_2D_X, int *_GPUNumSubDomains, \
	FP_PRECISION *_CMFDFlux_MPIRecv, int *_MPIComm_SumInd);

__global__ void GPUCMFD_GetNetCurrent_Y(int _dim, FP_PRECISION *_GPUCMFDSurfDiffCoeff_Y, \
	FP_PRECISION *_GPUCMFDSurfNDF_Y, FP_PRECISION_FLUX *_CMFDScalarFLux, \
	FP_PRECISION *_GPU_MGCMFDNetCurr_Y, int _SubDomLocID_2D_Y, int *_GPUNumSubDomains, \
	FP_PRECISION *_CMFDFlux_MPIRecv, int *_MPIComm_SumInd);

__global__ void GPUCMFD_GetNetCurrent_Z(int _dim, FP_PRECISION *_GPUCMFDSurfDiffCoeff_Z, \
	FP_PRECISION *_GPUCMFDSurfNDF_Z, FP_PRECISION_FLUX *_CMFDScalarFLux, \
	FP_PRECISION *_GPU_MGCMFDNetCurr_Z);

void GPU_CGCMFDMatrixCSR_AllocMem(int _numDiag, int _NNZ, M_INT32 _CGCMFD_LinearSolver);

// int GPUGMRESSolver_cuSparse(cublasHandle_t _cublasHandle, cusparseHandle_t _cusparseHandle, \
// 	cusparseOperation_t _transA, cusparseMatDescr_t _descra, cusparseMatDescr_t _descrm, \
// 	cusparseSolveAnalysisInfo_t _info_l, cusparseSolveAnalysisInfo_t _info_u, int _dim, int _nnz, \
// 	FP_PRECISION *_csrMatA, FP_PRECISION *_csrMatA_ILU, int *_csrRowPrtA, int *_csrColIndA,\
// 	FP_PRECISION *_xNew, FP_PRECISION *_b, FP_PRECISION *_res, FP_PRECISION *_ortVec,FP_PRECISION *_g,\
// 	FP_PRECISION *_h, int _kryDim);

int GPULinearSolver_GMRES(int _dim, int _nnz,int _kryDim, int _maxRestart, FP_PRECISION *_dev_A, FP_PRECISION *_dev_A_diag, \
	FP_PRECISION *_dev_ilu, FP_PRECISION *_host_ilu,int *_dev_JA,int *_host_JA, int*_dev_IA, int*_host_IA, int *_host_diagP, \
	FP_PRECISION *_dev_x, FP_PRECISION *_dev_b, FP_PRECISION _tolAbs, int _precond, cublasHandle_t _cublasHandle, \
	cusparseHandle_t _cusparseHandle, FP_PRECISION *_dev0_r0, FP_PRECISION *_host_r0, FP_PRECISION *_dev0_res, \
	FP_PRECISION *_dev0_beta, FP_PRECISION *_dev0_u_base, FP_PRECISION *_dev0_g, FP_PRECISION *_dev0_e, FP_PRECISION *_dev0_Hm, \
	FP_PRECISION *_dev0_w, FP_PRECISION *_dev0_y, FP_PRECISION *_dev0_Wm, int _GPU_Gmres_blocksPerGrid, FP_PRECISION_TIME *_time_prec, \
	FP_PRECISION_TIME *_time_lesSqr, FP_PRECISION_TIME *_time_memcpy, FP_PRECISION_TIME *_time_SpMV, FP_PRECISION_TIME *_time_blas1, \
	FP_PRECISION *_rmse, cusparseOperation_t _trans_L, cusparseOperation_t _trans_U, cusparseMatDescr_t _descr_L, cusparseMatDescr_t _descr_U, \
	csrsv2Info_t _info_L, csrsv2Info_t _info_U, cusparseSolvePolicy_t _policy_L, cusparseSolvePolicy_t _policy_U,void *_pBuffer);

void GPU_ILUFactPrecond(int _n, int _nz_num, int *_ia, int *_ja, FP_PRECISION *_a, \
	int *_ua, FP_PRECISION *_ilu, int *_iw);
void GPUCPU_ILUPrecond_CSR(int _n, int *_ia, int *_ja, FP_PRECISION *_ilu, int *_ua, FP_PRECISION *_r);


FP_PRECISION_TIME GPU_GetCurrTime();
FP_PRECISION_TIME GPU_GetTimeDuration(FP_PRECISION_TIME _start, FP_PRECISION_TIME _end);

__global__ void GPU_DiagPointer_CSRMatrix(int _dim, int *_ia, int *_ja,int *_diagP);
__global__ void GPU_GetInvDiagTerm_CSRMatrix(int _dim, int *_ia, int *_ja, FP_PRECISION *_a, FP_PRECISION *_diag);

__device__ void  GPUGetParCurr_2DMOC_AngFSmem(FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, \
    FP_PRECISION *_GPUMOCParCurr_Y_Pos, FP_PRECISION *_GPUMOCParCurr_Y_Neg, FP_PRECISION *_track_flux, \
    int _trackID, int _azimID, int g,int *_GPUNumTracksX, int *_GPUNumTracksY, int _iftrackStart, \
    int _RTMType, int _pinID_Layer, int _surfIndex_Layer_X, int _surfIndex_Layer_Y,FP_PRECISION_GEO *_GPURTMWidth_X, \
	FP_PRECISION_GEO *_GPURTMWidth_Y,int _polarBoundLowLim,int _polarBoundUpLim, int _polOffset, FP_PRECISION *_GPUTotalWeights_2DMOC);

__global__ void GPUGetNetCurr_Axial(int _lay, int _currLay, FP_PRECISION _dir, int *_GPUSumNumFSRsBasedGlobCellID, \
	int *_GPUNumFSRsInCell, int _polarBoundLowLim, int _polarBoundUpLim, FP_PRECISION_FLUX *_surfAngFlux, \
	FP_PRECISION *_GPUMOCNetCurr_Z, int _polOffset, FP_PRECISION_GEO *_GPUFSRArea_SubDom);

__global__ void GPUCMFD_EvaluateTotalSource_onFissQ(FP_PRECISION *_GPUCMFDTotalFissQ, \
	FP_PRECISION *_GPUCMFDTotalSource, FP_PRECISION *_GPUCMFD_Chi, FP_PRECISION *_GPUCMFDXs_s, \
	FP_PRECISION *_GPUCMFDScalarFlux, FP_PRECISION _oneOverKeff, int _dim, int _GPUNumCMFDGroups, \
	FP_PRECISION *_GPUCMFDNodeVol);

__global__ void GPUCMFD_EvaluateFissionSource_onFissQ(FP_PRECISION *_GPUCMFDTotalFissQ, \
	FP_PRECISION *_GPUCMFDTotalSource, FP_PRECISION *_GPUCMFD_Chi, FP_PRECISION _oneOverKeff, \
	int _dim, int _GPUNumCMFDGroups);

// void GPU_LS_BSOR(int _numCMFDGroup, FP_PRECISION *_x, FP_PRECISION *_b, \
// 	FP_PRECISION *_GPUMatrixCSR, int *_GPU_MGCMFDDiagPointer, \
// 	int *_GPUMatrixCSR_RowStartInd, int *_GPUMatrixCSR_ColInd_LU, \
// 	FP_PRECISION _SOR_factor, int _GPUNumRedThreads, int _GPUNumBlackThreads, \
// 	int _GPUGridDim_Red, int _GPUGridDim_Black, int _MOCNumPinsInSubDom_X, int _MOCNumPinsInSubDom_Y, \
// 	int _GPUNumThreadBlocks, int _GPUNumThreadPerBlock, int _GPUNumRedLay_OddNx, int _GPUNumBlackLay_OddNx);

int GPULinearSolver_BSOR(int _numCMFDGroup, FP_PRECISION *_xNew, FP_PRECISION *_xOld, FP_PRECISION *_b, \
	FP_PRECISION *_GPUMatrixCSR, int *_GPU_MGCMFDDiagPointer, int *_GPUMatrixCSR_RowStartInd, \
	int *_GPUMatrixCSR_ColInd, int _dim, FP_PRECISION *_xRelErr, int _MaxNumIter_LS, FP_PRECISION _tol, \
	cublasHandle_t _handle, FP_PRECISION _SOR_factor, int _GPUNumRedThreads, int _GPUNumBlackThreads, \
	int _GPUGridDim_Red, int _GPUGridDim_Black, int _GPUGridDim_CMFDNode, int _MOCNumPinsInSubDom_X, \
	int _MOCNumPinsInSubDom_Y, int _GPUNumThreadBlocks, int _GPUNumThreadPerBlock, int _GPUNumRedLay_OddNx, \
	int _GPUNumBlackLay_OddNx, int _isNormInf, int _isNorm2, int _isRmse, FP_PRECISION *_normInf, \
	FP_PRECISION *_norm2, FP_PRECISION *_rmse);

// FP_PRECISION GPU_MGCMFD_Eigenvalue_ZYQ(FP_PRECISION _keff, int nx, int ny, int nz, int ng, int *flagl2, \
// 	int *flaglinf, int *flageig, FP_PRECISION *erreig, FP_PRECISION *qerrlinf, FP_PRECISION *qrerrl2, \
// 	FP_PRECISION *ferrlinf, FP_PRECISION *_QFiss_old, FP_PRECISION *_QFiss, FP_PRECISION *_GPU_QFiss, \
// 	FP_PRECISION *flux_cmfd, FP_PRECISION *fluxoo_cmfd, FP_PRECISION *_xs_nf, FP_PRECISION *_CMFDNodeVol, \
// 	FP_PRECISION epseig, FP_PRECISION epsl2, FP_PRECISION epslinf, int _GPUNumThreadBlocks, int _GPUNumThreadPerBlock);

__global__ void GPU_SetNegCMFDFlux(int _dim, FP_PRECISION *_cmfdFlux, float *_negFCount);

void GPU_SetNegCMFDFlux_MG(int _dim,int _GPUNumThreadBlocks, int _GPUNumThreadPerBlock, float *_MGCMFD_NegFluxCount, M_INT32 *_MGCMFD_FluxFlag);

__global__ void GPUSetMOCPartCurrToZero_InnerBound(FP_PRECISION *_parCurr_MOC_X_Pos, \
	FP_PRECISION *_parCurr_MOC_X_Neg, FP_PRECISION *_parCurr_MOC_Y_Pos, FP_PRECISION *_parCurr_MOC_Y_Neg, \
	int _MOCParCurrInnerBound_Num, int *_MOCParCurrInnerBound_Index, int *_MOCParCurrInnerBound_Dir);

__device__ void  GPUGetParCurr_2DMOC_InnerBound_AngFSmem(FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, \
	FP_PRECISION *_GPUMOCParCurr_Y_Pos, FP_PRECISION *_GPUMOCParCurr_Y_Neg, FP_PRECISION *_GPUBoundaryFlux, int _trackID, \
	int _azimID, int g, int *_GPUNumTracksX, int *_GPUNumTracksY, int _RTMType, int _pinID_Layer, int _surfIndex_Layer_X, \
	int _surfIndex_Layer_Y, FP_PRECISION_GEO *_GPURTMWidth_X, FP_PRECISION_GEO *_GPURTMWidth_Y, int _polarBoundLowLim, \
	int _polarBoundUpLim, FP_PRECISION *_GPUTotalWeights_2DMOC, int *_mess_TarProc, int _lt, int _polEngID);

__global__ void GPUTallyMOCPartialCurr_InnerBound_Layer(int _lay, FP_PRECISION *_GPUBoundaryFlux, int *_GPULTsIndex_AzimAngID, \
	struct SUBDOM_TRACK_IN_LONGTRACK *_GPUTracksInfoInLTs, int *_GPUTrackIndexInLTs, int _numCellsInDom, int *_GPUCellRTMType_SubDom, \
	FP_PRECISION *_GPUMOCParCurr_X_Pos, FP_PRECISION *_GPUMOCParCurr_X_Neg, FP_PRECISION *_GPUMOCParCurr_Y_Pos, FP_PRECISION *_GPUMOCParCurr_Y_Neg, \
	int *_GPUNumTracksX, int *_GPUNumTracksY, FP_PRECISION_GEO *_GPURTMWidth_X, FP_PRECISION_GEO *_GPURTMWidth_Y, int _polarBoundLowLim, int _polarBoundUpLim, \
	FP_PRECISION *_GPUTotalWeights_2DMOC, int *_mess_TarProc);

__global__ void GPUSetCMFDTotSour_wInBoundInfo(int _procID, int _numProcs, int _MPIComm_Num, int *_MPIComm_Index, \
	FP_PRECISION *_CMFD_TotalSource, FP_PRECISION *_MPI_InBoundSourCoeff, FP_PRECISION *_MGCMFDComm_recv);

__global__ void GPUSetCMFDMPI_InBoundSourCoeff(int _procID, int _numProcs, FP_PRECISION *_MPI_InBoundSourCoeff, FP_PRECISION *_SurfDiffCoef_CMFD_X, \
	FP_PRECISION *_SurfNDF_CMFD_X, FP_PRECISION *_SurfDiffCoef_CMFD_Y, FP_PRECISION *_SurfNDF_CMFD_Y, FP_PRECISION *_SurfDiffCoef_CMFD_Z, \
	FP_PRECISION *_SurfNDF_CMFD_Z, int _MPIComm_Num, int *_MPI_InBoundSourCoeff_Index, int *_MPI_InBoundSourCoeff_Dir, FP_PRECISION *_MPI_InBoundSourCoeff_GeoDim);
	
void GPU_SetNegCMFDFlux_CG(int _dim,int _GPUNumThreadBlocks, int _GPUNumThreadPerBlock, float *_CGCMFD_NegFluxCount, M_INT32 *_CGCMFD_FluxFlag);

void GPU_ILUPrecond_CSR(cusparseHandle_t _cusparseHandle, cusparseOperation_t _trans_L, cusparseOperation_t _trans_U,\
	cusparseMatDescr_t _descr_L, cusparseMatDescr_t _descr_U, csrsv2Info_t _info_L, csrsv2Info_t _info_U, \
	cusparseSolvePolicy_t _policy_L, cusparseSolvePolicy_t _policy_U, void *_pBuffer, int _dim, int _nnz, \
	FP_PRECISION *_csrMatA_ILU, int *_csrRowPrtA, int *_csrColIndA, FP_PRECISION *_vec);
#endif // M_MG_CMFD
#endif // GPUCMFD_H