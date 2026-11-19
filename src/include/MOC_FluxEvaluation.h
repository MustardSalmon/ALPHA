#ifndef MOC_FLUX_EVALU_H
#define MOC_FLUX_EVALU_H
#include "MOC_CommonMacro.h"
#include "MOC_Track.h"
#define _countof_(a) (sizeof(a)/sizeof(*(a)))

extern FP_PRECISION **BoundAngFlux_send, **BoundAngFlux_recv;
extern int **NextSubDomLTIDs_send, **NextSubDomLTIDs_recv;
extern int ***MPIProcID_recv, ***MPINeibID_recv;

#ifdef MPI_USED
void UpdateBoundAngFlux_PreNeibInfo(int _MOCNumLayersZ, int _numProcsPerLayer, int _sunDomID_layer, int _procID, int ***_mess_TarProc, int ***_mess_SendNumLTs, \
	int _numAzimAng, struct SUBDOM_LONGTRACK_INFO *_longTracks_Info, int *_numLongTracks_SubDom, int ***_procID_recv, int ***_neibID_recv, \
	size_t ***_mess_SendSatrtPInAngF, size_t ***_mess_RecvSatrtPInAngF, int _numPolarAng, int _numGroups, int _numLTsInHalfspace, int *_numLTsSum, \
	int **_GCCopy_StartPInAngF, int **_GCCopy_NumDataInAngF);
void UpdateBoundAngFlux_ExchAngF_Blocking(int _lay, int _numProcsPerLayer, int _procID, int **_mess_TarProc, int **_mess_SendNumLTs, \
	FP_PRECISION *_LTAngFluxInci_SubDom, FP_PRECISION *_boundaryFlux_SubDom, int _numAzimAng, struct SUBDOM_LONGTRACK_INFO *_longTracks_Info, \
	int *_numLongTracks_SubDom, int _numPolarAng, int _numGroups, int _numLTsInHalfspace, int *_numLTsSum, size_t **_mess_SendSatrtPInAngF, \
	size_t **_mess_RecvSatrtPInAngF, int **_procID_recv, int **_neibID_recv);
void UpdateBoundAngFlux_ExchAngF_NonBlocking(int _lay, int _numProcsPerLayer, int _procID, int **_mess_TarProc, int **_mess_SendNumLTs, \
	FP_PRECISION *_LTAngFluxInci_SubDom, FP_PRECISION *_boundaryFlux_SubDom, int _numAzimAng, struct SUBDOM_LONGTRACK_INFO *_longTracks_Info, \
	int *_numLongTracks_SubDom, int _numPolarAng, int _numGroups, int _numLTsInHalfspace, int *_numLTsSum, FP_PRECISION **_BoundAngFlux_send, \
	FP_PRECISION **_BoundAngFlux_recv, int **_nextSubDomLTIDs_send, int **_nextSubDomLTIDs_recv, int **_procID_recv, int **_neibID_recv);
#endif // MPI_USED
int Get_NumFSRs_GlobDom(int *_numFSRs_array, int _numFSRs_subDom, int _procID, int _numProcs);
FP_PRECISION InfNorm_ScalFlux(FP_PRECISION_FLUX *_scalarFlux_temp, FP_PRECISION_FLUX *_scalarFlux_old, int _totalNumFSRs_subDom, int _numGroups);
FP_PRECISION **AllocMemInciAngFlux(int _numTrackSet, int *_numLTsInHalfspace, int _numGroups, \
    int *_numPolarAng, int _MOCNumLayersZ, char *_parameter);
void InitInciAngFlux(FP_PRECISION **_angFlux, int _numTrackSet, int *_numLTsInHalfspace, \
    int _numGroups, int *_numPolarAng, int _MOCNumLayersZ, FP_PRECISION _value);
void InitInciAngFlux_BC(FP_PRECISION **_angFlux, int _numTrackSet, int _numGroups, int *_numAzimAng, \
	int *_numPolarAng, int _MOCNumLayersZ, int **_NumLongTracks_SubDomX, int **_NumLongTracks_SubDomY);
void CopyInciAngFlux(FP_PRECISION **_ectypeFlux, FP_PRECISION **_origFlux, int _numTrackSet, int *_numLTsInHalfspace, int _numGroups, int *_numPolarAng, int _MOCNumLayersZ);

int UpdateBoundAngFlux_ExchAngF_Azim(int _azim, int _startPos, int _lay, int _numProcsPerLayer, int _procID, int **_mess_TarProc, int **_mess_SendNumLTs, \
	FP_PRECISION *_LTAngFluxInci_SubDom, FP_PRECISION *_boundaryFlux_SubDom, int _numAzimAng, struct SUBDOM_LONGTRACK_INFO *_longTracks_Info, \
	int *_numLongTracks_SubDom, int _numPolarAng, int _numGroups, int _numLTsInHalfspace, int *_numLTsSum, FP_PRECISION **_BoundAngFlux_send, \
	FP_PRECISION **_BoundAngFlux_recv, int **_nextSubDomLTIDs_send, int **_nextSubDomLTIDs_recv, int **_procID_recv, int **_neibID_recv);
#if defined(CUDA_USED) && defined (MPI_USED)
void GPU_BoundValueUpdate(int _procID, int _numProcs);
#endif // defined(CUDA_USED) && defined (MPI_USED)
void Jacobi_ScalarFlux_RayWisedSumToAntiClock(int _azimID, int _numGroups, int _numLayers_SubDom, int _numCellsInDom, \
	int *_numFSRsInCell, int *_sumNumFSRsBasedGlobCellID, int *_fsrIdMap_rayWisedToAntiClock, int *_cellRTMGeoType, \
	FP_PRECISION_FLUX *_flux_rayWisedFSR, FP_PRECISION_FLUX *_flux_antiClockFSR, int _totalNumFSRsInRTM, int *_sumNumFSRsInRTM);
void NormlizeBoundAngFlux(FP_PRECISION **_angFlux,int _numTrackSet, int *_numLTsInHalfspace, \
  int _numGroups, int *_numPolarAng, int _MOCNumLayersZ, FP_PRECISION _norm_factor);
void AccumAngFlux_ScalarFlux(int _lay, int _numCellsInDom_perLayer, int *_sumNumFSRsBasedGlobCellID,int *_numFSRsInCell, \
    int _numGroups, int _numAzimAng, int _numPolAng, int _groupBoundLowLim, int _groupBoundUpLim, int _polarBoundLowLim, \
    int _polarBoundUpLim, FP_PRECISION *_totalWeights, FP_PRECISION_FLUX *_aveAngFlux, FP_PRECISION_FLUX *_scalarFlux, \
	int _polOffset);
void Update_AxialSurfFlux(int _dim,FP_PRECISION *_aveAngFlux_inp, FP_PRECISION *_aveAngFlux_update);
void Update_AxialSurfFlux_g(int _TotalNumFSRs_perLayer, int _numGroups, int _numAzimAng, int _numPolAng, \
	int _groupBoundLowLim, int _groupBoundUpLim, FP_PRECISION *_aveAngFlux_inp, FP_PRECISION *_aveAngFlux_update);
void ZerolizeNegtiveAxialSurfFlux(int _dim,FP_PRECISION *_aveAngFlux);
void RecalcAxialSurfFlux_SOR(int _dim, int _index, FP_PRECISION *_aveAngFlux_inp, \
	FP_PRECISION *_aveAngFlux_inpAllLay, FP_PRECISION _sor);
void CopyAxialSurfFlux(int _dim, int _index, FP_PRECISION *_aveAngFlux_inp, \
	FP_PRECISION *_aveAngFlux_inpAllLay);
void Update_AxialBoundCond(int _dir,int _dim, int _numPolAng, \
	FP_PRECISION *_aveAngFlux_inp, FP_PRECISION *_aveAngFlux_bound);
void Update_AxialBoundCond_g(int _dir, int _TotalNumFSRs_perLayer, int _numGroups, int _numAzimAng, int _numPolAng, \
	int _groupBoundLowLim, int _groupBoundUpLim, FP_PRECISION *_aveAngFlux_inp, FP_PRECISION *_aveAngFlux_bound);
// void Check_AxialBoundCond(int _dir,int _numCellsInDom, int *_sumNumFSRsBasedGlobCellID,int *_numFSRsInCell, \
//     int _numGroups, int _numAzimAng, int _numPolAng, int _groupBoundLowLim, int _groupBoundUpLim, int _polarBoundLowLim, \
//     int _polarBoundUpLim, FP_PRECISION *_aveAngFlux_inp, FP_PRECISION *_aveAngFlux_update);

void AddSourceToFlux(int _numCellsInDom, int *_numFSRsInCell, int *_sumNumFSRsBasedGlobCellID, \
	int _numGroups, FP_PRECISION_FLUX *_scalarFlux, FP_PRECISION *_source, FP_PRECISION *_macroXS_t,\
	int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim, int _iter_moc);
void AddSourceToFlux_g(int _g, FP_PRECISION_FLUX *_scalarFlux, FP_PRECISION *_source, int _numGroups, int _numLay, int _numCellsInDom, \
	int *_numFSRsInCell, int *_sumNumFSRsBasedGlobCellID, FP_PRECISION *_macroXS_t, FP_PRECISION **_inpXSs_t, int _iter_moc);
FP_PRECISION ExpInter(FP_PRECISION _x_point, FP_PRECISION *_expTable, int _iter_moc);
int Get_NumPins_GlobDom(int *_CollNumPinsSubDom, int _MOC_NumPinsInSubDom, int _procID, int _numProcs);
void AccumAngFlux_ScalarFlux_HOScatt(int _lay, int _numCellsInDom_perLayer, int *_sumNumFSRsBasedGlobCellID, int *_numFSRsInCell, \
	int _numGroups, int _numAzimAng, int _numPolAng, int _groupBoundLowLim, int _groupBoundUpLim, int _polarBoundLowLim, \
	int _polarBoundUpLim, FP_PRECISION *_totalWeights, FP_PRECISION_FLUX *_aveAngFlux, FP_PRECISION_FLUX *_scalarFlux, \
	int _polOffset, int _numScattOrder, FP_PRECISION *_Ylm, M_INT32 _totalNumFSRs_subDom);
void AccumAngFlux_ScalarFlux_HOScatt_1(int _lay, int _numCellsInDom_perLayer, int *_sumNumFSRsBasedGlobCellID, int *_numFSRsInCell, \
	int _numGroups, int _numAzimAng, int _numPolAng, int _groupBoundLowLim, int _groupBoundUpLim, int _polarBoundLowLim, \
	int _polarBoundUpLim, FP_PRECISION *_totalWeights, FP_PRECISION_FLUX *_aveAngFlux, FP_PRECISION_FLUX *_scalarFlux, \
	int _polOffset, int _numScattOrder, FP_PRECISION *_Ylm, M_INT32 _totalNumFSRs_subDom);
void AccuFluxByLayer(int _layID, int _TotalNumFSRs_subDom, int _TotalNumFSRs_perLayer, int _NumGroups, \
    int _NumScattOrder, FP_PRECISION_FLUX *_ScalarFlux_MOC, FP_PRECISION_FLUX *_ScalarFlux_MOCLayer);
#endif // ! FLUX_EVALU
