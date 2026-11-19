#ifndef R_DH_MAT_XS_H
#define R_DH_MAT_XS_H

void Jacobi_ComputeFSRSource_DH(int _procID, int _iter, FP_DOUBLE _keff, FP_PRECISION_FLUX *_scalarFlux, \
	FP_PRECISION *_source, int _numGroups, int _numLayers_SubDom, int _numCellsInDom, \
	int *_numFSRsInCell, FP_PRECISION ***_XS_leakage_SubDom, int *_sumNumFSRsBasedGlobCellID, \
	FP_PRECISION *_MacroXS_chi, FP_PRECISION *_macroXS_nf, FP_PRECISION *_macroXS_s, FP_PRECISION *_macroXS_t);
void Cal_Eff_Scalar_Flux_DH(int _procID);

void InputMat_XSs_DH(char * _filename);
void Output_Material_MOC_Check_DH(int _procID);
void Output_Material_MOC_Check_FSR_DH(int _procID);
void MacroXSsEvaluation_subDomain_DH(struct CORE *_core, int _subDomID_row, int _subDomID_column);
void MOC_MacroXST_Iter_DH(int lay);
void Output_Eff_Scalar_Flux_DH(int _procID);


;;;;;;;;;;
#endif
