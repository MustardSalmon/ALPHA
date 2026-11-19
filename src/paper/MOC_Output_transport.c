/********************************************************************************/
/* @ File: MOC_Output.c
/*   introduction: functions related to output
/* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*   1st modified:
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "MOC_AllocAndFreeMem.h"
#include "MOC_Assembly.h"
#include "MOC_CommonMacro.h"
#include "MOC_CMFD.h"
#include "MOC_Geometry.h"
#include "MOC_Input.h"
#include "MOC_Lattice.h"
#include "MOC_Macro_XSs.h"
#include "MOC_main.h"
#include "MOC_NTE_Solver.h"
#include "MOC_Output.h"
#include "MOC_Pin_Cell.h"
#include "MOC_Quadrature_Set.h"
#include "MOC_Track.h"
#ifdef MPI_USED
    #include "mpi.h"
#endif // MPI_USED


int OUTPUT_MatrixEigenvalueProb=0;
int OUTPUT_MemoryFootprint = 0;
int Output_GeometryInfo = 0;
int OUTPUT_inputInfo = 0;
int OUTPUT_XSslibInfo = 0;
int OUTPUT_MacorXSs = 0;
int OUTPUT_MacorXSs_InputCheck = 0;
int OUTPUT_eigenvalue = 0;
int OUTPUT_normPinPower = 0;
int OUTPUT_flux = 0, OUTPUT_flux_mat=0;
int OUTPUT_CMFDGroupConstant=0;
int OUTPUT_CMFDGC_byStep=0;
int OUTPUT_CoarseMesh_GroupConstant=0;
int OUTPUT_CompTime = 0;//If = 1, MOC computing time will highly increase.
int OUTPUT_CoarseCompTime = 0;
int OUTPUT_Debug = 0;
int OUTPUT_CompTime_Proc = 0;//If = 1, MOC computing time will highly increase.
FILE *FP_MemoryFootprint = NULL;
FILE *CompTime_rec = NULL;
FILE *GeneralOutput = NULL;
FILE *Debug = NULL;
FP_PRECISION_TIME CompTime_Tot = 0.0; // total computational time
FP_PRECISION_TIME CompTime_CMFD_Tot = 0.0; // total computational time for M_MG_CMFD
FP_PRECISION_TIME CompTime_MOC_Tot = 0.0; // total computational time for MOC
FP_PRECISION_TIME CompTime_CMFDtoMOC = 0.0; // computational time for updating moc flux using cmfd results
FP_PRECISION_TIME CompTime_MOC_LBF = 0.0; // time accumulator for moc boundary incident angular flux load procedure
FP_PRECISION_TIME CompTime_MOC_SE = 0.0; // evaluate source for FSRs
FP_PRECISION_TIME CompTime_MOC_TS = 0.0; // total time for moc transport sweep
FP_PRECISION_TIME CompTime_MOC_In = 0.0; // initialization in MOC solver
FP_PRECISION_TIME CompTime_MOC_STr = 0.0; // Segments tracking in MOC solver
FP_PRECISION_TIME CompTime_MOC_GPCIB = 0.0; // get partial current of pin boundary during TS (inner boundary)
FP_PRECISION_TIME CompTime_MOC_GPCOB = 0.0; // get partial current of pin boundary after TS (outer boundary)
FP_PRECISION_TIME CompTime_MOC_EXP = 0.0; // exp evaluation during TS
FP_PRECISION_TIME CompTime_MOC_SI = 0.0; // segment intergration and scale region-wise flux sums by angular weight
FP_PRECISION_TIME CompTime_MOC_BCTS = 0.0; // angular flux update based on BC or storage at subdomain boundary during TS
FP_PRECISION_TIME CompTime_MOC_ESF = 0.0; //evaluate scalar flux of FSRs by adding scaled region-wise flux to source
FP_PRECISION_TIME CompTime_MOC_UBAF = 0.0; // update subdomain boundary angular flux based on track connection
FP_PRECISION_TIME CompTime_MOC_UBAF_Pre = 0.0; // array preparetion time in procedure UBAF
FP_PRECISION_TIME CompTime_MOC_UBAF_Comm = 0.0; // communication tiome in procedure UBAF
FP_PRECISION_TIME CompTime_MOC_EVGC = 0.0; // evaluate group constant for M_MG_CMFD
FP_PRECISION_TIME CompTime_MOC_EXGC = 0.0; // exchange group constant for M_MG_CMFD
FP_PRECISION_TIME CompTime_ConvCheck = 0.0; // convergence check
FP_PRECISION_TIME CompTime_MOC_ParaUp = 0.0; // update MOC parameters during iteration, keff, scalar flux,
FP_PRECISION_TIME CompTime_MOC_AddSour = 0.0; // add source to scalar flux after transport sweep
FP_PRECISION_TIME CompTime_CopyAngF_CpuGpu = 0.0; // add source to scalar flux after transport sweep
FP_PRECISION_TIME CompTime_Input, CompTime_InitGeoXSs, CompTime_ExpTab, CompTime_AllMem, CompTime_MapGtoC, CompTime_CopyXSsCtoG, CompTime_CopyTraInfoCtoG, CompTime_InitTSonGPU;  // time for read input

FP_PRECISION_TIME TimeRec_TSPraperation = 0.0, TimeRec_MOCSolution = 0.0, TimeRec_CMFDSolution = 0.0, TimeRec_TSGPU = 0.0, TimeRec_SourceCalc = 0.0, TimeRec_RT = 0.0;
FP_PRECISION_TIME TimeRec_ClusterPreparation = 0.0;
FP_PRECISION_TIME TimeRec_FissionSource = 0.0;
FP_PRECISION_TIME TimeRec_HigherOrderSource = 0.0;
FP_PRECISION_TIME TimeRec_HigherOrderSourceLM = 0.0;
FP_PRECISION_TIME TimeRec_HigherOrderSourceAP = 0.0;
FP_PRECISION_TIME TimeRec_HigherOrderSourceAuxiliary = 0.0;
FP_PRECISION_TIME TimeRec_TransportSweep = 0.0;
FP_PRECISION_TIME TimeRec_FluxAccumulation = 0.0;
FP_PRECISION_TIME TimeRec_P0Source = 0.0;
FP_PRECISION_TIME TimeRec_TransportCalculation = 0.0;
FP_PRECISION_TIME TimeRec_FluxMomentKernel103 = 0.0;
M_UINT64 TimeRec_FluxMomentKernel103Calls = 0;
FP_PRECISION_TIME TimeRec_FluxMomentKernelPerAzim = 0.0;
M_UINT64 TimeRec_FluxMomentKernelPerAzimCalls = 0;
FP_PRECISION_TIME TimeRec_FluxMomentScalarFluxInit = 0.0;
FP_PRECISION_TIME TimeRec_FluxMomentAngularFluxCopy = 0.0;
FP_PRECISION_TIME TimeRec_FluxMomentAniScattSourceCopy = 0.0;
FP_PRECISION_TIME TimeRec_FluxMomentAngularFluxReset = 0.0;
FP_PRECISION_TIME TimeRec_FluxMomentAzimAccumInit = 0.0;
FP_PRECISION_TIME TimeRec_FluxMomentAzimAccumFinalize = 0.0;
FP_PRECISION_TIME TimeRec_HOTransportKernel1 = 0.0;
M_UINT64 TimeRec_HOTransportKernel1Calls = 0;
FP_PRECISION_TIME TimeRec_HOTransportKernel11 = 0.0;
M_UINT64 TimeRec_HOTransportKernel11Calls = 0;
FP_PRECISION_TIME TimeRec_ProgramWallStart = 0.0;
FP_PRECISION_TIME TimeRec_ProgramWallTotal = 0.0;
FP_PRECISION_TIME TimeRec_MGCMFD_XSCollapse[1] = { 0.0 }, TimeRec_MGCMFD_MatrixConst[1] = { 0.0 }, TimeRec_MGCMFD_CMFDSolver = 0.0, TimeRec_MGCMFD_FluxUpdate = 0.0;
FP_PRECISION_TIME TimeRec_CGCMFD_XSCollapse[1] = { 0.0 }, TimeRec_CGCMFD_MatrixConst[1] = { 0.0 }, TimeRec_CGCMFD_CMFDSolver = 0.0, TimeRec_CGCMFD_FluxUpdate = 0.0;
FP_PRECISION_TIME TimeRec_MGCMFD_LinearSolver[1] = { 0.0 }, TimeRec_CGCMFD_LinearSolver[1] = { 0.0 };

FP_PRECISION_TIME Time_MGCMFD_GMRES_Precond[1] = { 0.0 }, Time_MGCMFD_GMRES_LeastSqr[1] = { 0.0 }, Time_MGCMFD_GMRES_MemCpy[1] = { 0.0 };
FP_PRECISION_TIME Time_MGCMFD_GMRES_SpMV[1] = { 0.0 }, Time_MGCMFD_GMRES_BLAS1[1] = { 0.0 };
FP_PRECISION_TIME Time_CGCMFD_GMRES_Precond[1] = { 0.0 },Time_CGCMFD_GMRES_LeastSqr[1] = { 0.0 }, Time_CGCMFD_GMRES_MemCpy[1] = { 0.0 };
FP_PRECISION_TIME Time_CGCMFD_GMRES_SpMV[1] = { 0.0 }, Time_CGCMFD_GMRES_BLAS1[1] = { 0.0 };
int NumIters; // number of iterations

int ParamCounter = 0;
FP_PRECISION NumTotalData = 0.0;
char ParaName[NumPara][ParaNameLength] = { "\0" };
FP_PRECISION ParaMemSize[NumPara] = { 0.0 };
#ifdef TIME_REC
  int TimeCount = 0, TimeCountMOC = 0, TimeCountMGCMFD = 0, TimeCountCGCMFD = 0;
  int TimeCountPrePart = 0;
  int TimeMGStamp1,TimeMGStamp2,TimeMGStamp3,TimeMGStamp4,TimeMGStamp5,TimeMGStamp6,TimeMGStamp7,TimeMGStamp8,TimeMGStamp9,TimeMGStamp10,TimeMGStamp11,TimeMGStamp12;
  int TimeCGStamp1,TimeCGStamp2,TimeCGStamp3,TimeCGStamp4,TimeCGStamp5,TimeCGStamp6,TimeCGStamp7,TimeCGStamp8,TimeCGStamp9;
  int TimeStamp3;
  char TimeMetricsPrePart[NumTimeMetrics][ParaNameLength] = { "\0" };
  char TimeMetrics[NumTimeMetrics][ParaNameLength] = { "\0" };
  char TimeMetricsMOC[NumTimeMetrics][ParaNameLength] = { "\0" };
  char TimeMetricsMG[NumTimeMetrics][ParaNameLength] = { "\0" };
  char TimeMetricsCG[NumTimeMetrics][ParaNameLength] = { "\0" };

  FP_PRECISION_TIME TimePrePart[NumTimeMetrics] = { (FP_PRECISION_TIME)(0.0) };
  FP_PRECISION_TIME TimeOverall[NumTimeMetrics] = { (FP_PRECISION_TIME)(0.0) };
  FP_PRECISION_TIME TimeMOC[NumTimeMetrics] = { (FP_PRECISION_TIME)(0.0) };
  FP_PRECISION_TIME TimeRecMG[NumTimeMetrics] = { (FP_PRECISION_TIME)(0.0) };
  FP_PRECISION_TIME TimeRecCG[NumTimeMetrics] = { (FP_PRECISION_TIME)(0.0) };

  int TimeStamp_TS,TimeStamp_ConvCheck, TimeStamp_SE, TimeStamp_MPIComm, TimeStamp_CommCGCopy, TimeStamp_CommGCCopy, TimeStamp_CMFD;
  int TimeStamp_CMFDtoMOC,TimeStamp_MGGCCond;
  int TimeStamp_MOC_ParaUp, TimeStamp_GPUCopyBoundAF, TimeStamp_MOC_AStF, TimeStamp4, TimeStamp5, TimeStamp6, TimeStamp7, TimeStamp8, TimeStamp9, TimeStamp10;
  int TimeStamp_MOC_TS1,TimeStamp_MOC_TS2,TimeStamp_MOC_TS3,TimeStamp_MOC_TS4,TimeStamp_MOC_TS5;
  int TimeID_Input = 0, TimeID_GeoXSs = 1, TimeID_ExpTab = 2, TimeID_AllocMem = 3, TimeID_CommPre = 4, TimeID_SetCUDADev = 5, TimeID_InitInfoCtoG = 6, TimeID_InitTS = 7;
  int TimeID_TS_ParaUp = 8, TimeID_TS_SE = 9, TimeID_TS_CtoG = 10, TimeID_TS = 11, TimeID_TS_GtoC = 12, TimeID_TS_MPIComm = 13, TimeID_TS_ReacRate = 14;
  int TimeID_TS_FluxRes = 15, TimeID_TS_ConvCheck = 16;
  FP_PRECISION_TIME *CompTime_omp = NULL;
#endif // TIME_REC

#ifdef DynamicWorkLoad
  FP_PRECISION_TIME TimeTS_DWL = (FP_PRECISION_TIME)(0.0);
#endif // DynamicWorkLoad

  int MemParaCountBoundLow[20], MemParaCountBoundUp[20], MemPrintCount = 0;
  size_t GPUPreMemQuadrSet[1] = { 0 }, GPUPreMemTrack[1] = { 0 }, GPUPreMemExpTab[1] = { 0 }, GPUPreMemScalF[1] = { 0 };
  size_t GPUPreMemRadF[1] = { 0 }, GPUPreMemAxiF[1] = { 0 }, GPUPreMemXSs[1] = { 0 }, GPUPreMemMGCMFD[1] = { 0 }, GPUPreMemCGCMFD[1] = { 0 };
  char MemParaName[20][128];


/**
* Output the geometry interface parameters' information
*/

/**
 * Output normalized pin power of all pins in Core
 */
void Output_NormPinPower(int _procID, int _numProcs, int _timeID)
{
	if (OUTPUT_normPinPower==1 && _procID==0) {
		int i,j,k,i1,j1,k1,_proc=0, _check=0, _numSunDm_X, _numSunDm_Y, _loc_X=0,_loc_Y=0,_locID;
		int _width1=14, *_sumNumPinsSubDom_X=NULL, *_sumNumPinsSubDom_Y=NULL;
		char _fileName[225];
		FP_PRECISION *_NormPinPower_SumLay=NULL,*_NormPinPower_SumRad=NULL, *_NormPinPower_SumAllLay=NULL;
		_NormPinPower_SumLay = AllocMem_FP_I_1(PinPowerSumNumSet*NumCellsInCore_2D_X*NumCellsInCore_2D_Y, (char *)CHAR_NONE);
		for (i=0;i<PinPowerSumNumSet*NumCellsInCore_2D_X*NumCellsInCore_2D_Y;i++) {
			_NormPinPower_SumLay[i]=0.0;
		}
		_NormPinPower_SumRad = AllocMem_FP_I_1(MOCNumLayersZ, (char *)CHAR_NONE);
		_NormPinPower_SumAllLay = AllocMem_FP_I_1(NumCellsInCore_2D_X*NumCellsInCore_2D_Y, (char *)CHAR_NONE);
		for (i=0;i<NumCellsInCore_2D_X*NumCellsInCore_2D_Y;i++) {
			_NormPinPower_SumAllLay[i]=0.0;
		}

		FILE *_op = NULL;

		_numSunDm_X = NumSubDomains[0];
		_numSunDm_Y = NumSubDomains[1];

		_sumNumPinsSubDom_X = AllocMem_i32_i32(_numSunDm_X+1, CHAR_NONE);
		_sumNumPinsSubDom_Y = AllocMem_i32_i32(_numSunDm_Y+1, CHAR_NONE);
		for (i=0;i<_numSunDm_X;i++) {
			if (i==0) {
				_sumNumPinsSubDom_X[i]=0;
			}
			else {
				_sumNumPinsSubDom_X[i] = _sumNumPinsSubDom_X[i-1] + NumPin_subDomain[0][i-1];
			}
		}
		_sumNumPinsSubDom_X[_numSunDm_X]=NumCellsInCore_2D_X;

		for (i=0;i<_numSunDm_Y;i++) {
			if (i==0) {
				_sumNumPinsSubDom_Y[i]=0;
			}
			else {
				_sumNumPinsSubDom_Y[i] = _sumNumPinsSubDom_Y[i-1] + NumPin_subDomain[1][i-1];
			}
		}
		_sumNumPinsSubDom_Y[_numSunDm_Y]=NumCellsInCore_2D_Y;

		_check=0;
#if !defined(MPI_USED)
        _check=1;
#else
		if (_numProcs==1) {
			_check=1;
		}
#endif // !defined(MPI_USED)
        if (_check==1) {
			CollecGlobPinPower = (FP_PRECISION **)calloc(1, sizeof(FP_PRECISION *));
		    CollecGlobPinPower[0]=NormPinPowerSubDom;
		}
        if (isTransient == 2){
            sprintf(_fileName, "time-%d-%s-%s", _timeID, CaseId, "NormalizedPinPower.out");
        } else{
            sprintf(_fileName, "%s-%s", CaseId, "NormalizedPinPower.out");
        }
		_op = fopen(_fileName, "w");
		if (_op == NULL) {
			printf("\n [**ERROR**]  %s file opens error!\n", _fileName);
			fflush(stdout);
			exit(69);
		}
		// _asse_power = AllocMem_FP_II(NumLayer_Core, _core->num_X*_core->num_Y, "_asse_power");

		fprintf(_op, "\n  *************************************************************************");
		fprintf(_op, "\n  *                                                                        ");
		fprintf(_op, "\n  Part 1: Layer-integrated pin power (fission rate)");
		fprintf(_op, "\n          power(pin) = hight(lay) * sum_fsr{sum_g[xs_F(fsr,g) * phi(frs,g)] * area(fsr)}");
		fprintf(_op, "\n  *                                                                        ");
		fprintf(_op, "\n  *************************************************************************");
		fprintf(_op, "\n  # Layers: %d",MOCNumLayersZ);
		fprintf(_op, "\n  # Pins in X-dir: %d",NumCellsInCore_2D_X);
		fprintf(_op, "\n  # Pins in Y-dir: %d",NumCellsInCore_2D_Y);

		for (i=0;i<MOCNumLayersZ;i++) {
			fprintf(_op, "\n\n  ############################################################");
			fprintf(_op, "\n  Layer-integrated pin power for layer %d",i);
			fprintf(_op, "\n%*s", 6,"");
			for (k=0;k<NumCellsInCore_2D_X;k++) {
				fprintf(_op, "%*d", _width1,k+1);
			}
			for (i1=0;i1<PinPowerSumNumSet;i1++) {
				if (i>=PinPowerSumLayBound[i1] && i<PinPowerSumLayBound[i1+1]) {
					break;
				}
			}
			_NormPinPower_SumRad[i]=0.;
			for (j=0;j<NumCellsInCore_2D_Y;j++) {
				fprintf(_op, "\n%*d", 6,j+1);
				for (j1=0;j1<_numSunDm_Y;j1++) {
					if (j>=_sumNumPinsSubDom_Y[j1] && j<_sumNumPinsSubDom_Y[j1+1]) {
						_loc_Y=j-_sumNumPinsSubDom_Y[j1];
						break;
					}
				}
				for (k=0;k<NumCellsInCore_2D_X;k++) {
					for (k1=0;k1<_numSunDm_X;k1++) {
					    if (k>=_sumNumPinsSubDom_X[k1] && k<_sumNumPinsSubDom_X[k1+1]) {
							_loc_X=k-_sumNumPinsSubDom_X[k1];
					    	break;
					    }
				    }
					_proc=j1*_numSunDm_X+k1;
					_locID=i*NumPin_subDomain[0][k1]*NumPin_subDomain[1][j1]+_loc_Y*NumPin_subDomain[0][k1]+_loc_X;
					// printf("\n i=%d, j=%d, k=%d, j1=%d, k1=%d, _loc_X=%d, _loc_Y=%d, _locID=%d",i,j,k,j1,k1,_loc_X,_loc_Y,_locID);

					fprintf(_op, "%*.6e",_width1, CollecGlobPinPower[_proc][_locID]);
					_NormPinPower_SumRad[i]+=CollecGlobPinPower[_proc][_locID];
					_NormPinPower_SumLay[i1*NumCellsInCore_2D_Y*NumCellsInCore_2D_X+j*NumCellsInCore_2D_X+k]+=CollecGlobPinPower[_proc][_locID];

				}
			}
		}

		fprintf(_op, "\n  *************************************************************************");
		fprintf(_op, "\n  *                                                                        ");
		fprintf(_op, "\n  Part 2: Layer-merged pin power (fission rate)");
		fprintf(_op, "\n          merge the pin power of multi-layers defiened by user");
		fprintf(_op, "\n  *                                                                        ");
		fprintf(_op, "\n  *************************************************************************");
		fprintf(_op, "\n  # Coarse Layers: %d",PinPowerSumNumSet);
		fprintf(_op, "\n    Layer-Bounds%*s%*s",8,"Low",8,"Up");

		for (i1=0;i1<PinPowerSumNumSet;i1++) {
			fprintf(_op, "\n                %*d%*d",8,PinPowerSumLayBound[i1],8,PinPowerSumLayBound[i1+1]);
		}
		fprintf(_op, "\n  # Pins in X-dir: %d",NumCellsInCore_2D_X);
		fprintf(_op, "\n  # Pins in Y-dir: %d",NumCellsInCore_2D_Y);
		for (i1 = 0; i1 < PinPowerSumNumSet; i1++) {
			fprintf(_op, "\n\n  ############################################################");
			fprintf(_op, "\n  Normalized Pin Power for Coarse Layer %d",i1);
			fprintf(_op, "\n%*s", 6,"");
			for (j=0;j<NumCellsInCore_2D_X;j++) {
				fprintf(_op, "%*d", _width1,j+1);
			}
			for (i = 0; i < NumCellsInCore_2D_Y; i++) {
				fprintf(_op, "\n%*d", 6,i+1);
				for (j=0;j<NumCellsInCore_2D_X;j++) {
					fprintf(_op, "%*.6e",_width1, _NormPinPower_SumLay[i1*NumCellsInCore_2D_Y*NumCellsInCore_2D_X+i*NumCellsInCore_2D_X +j]);
					_NormPinPower_SumAllLay[i*NumCellsInCore_2D_X +j]+=_NormPinPower_SumLay[i1*NumCellsInCore_2D_Y*NumCellsInCore_2D_X+i*NumCellsInCore_2D_X +j];
				}
			}
		}

		fprintf(_op, "\n  *************************************************************************");
		fprintf(_op, "\n  *                                                                        ");
		fprintf(_op, "\n  Part 3: All-layer-merged pin power (fission rate)");
		fprintf(_op, "\n          merge the pin power for all layers");
		fprintf(_op, "\n  *                                                                        ");
		fprintf(_op, "\n  *************************************************************************");
		fprintf(_op, "\n\n  ############################################################");
		fprintf(_op, "\n  Normalized Pin Power Integrated Over All Layers");
		fprintf(_op, "\n%*s", 6,"");
		for (j=0;j<NumCellsInCore_2D_X;j++) {
			fprintf(_op, "%*d", _width1,j+1);
		}
		for (i = 0; i < NumCellsInCore_2D_Y; i++) {
			fprintf(_op, "\n%*d", 6,i+1);
			for (j=0;j<NumCellsInCore_2D_X;j++) {
				fprintf(_op, "%*.6e",_width1, _NormPinPower_SumAllLay[i*NumCellsInCore_2D_X +j]);
		    }
		}

		fprintf(_op, "\n  *************************************************************************");
		fprintf(_op, "\n  *                                                                        ");
		fprintf(_op, "\n  Part 4: Integrate pin power (fission rate) over all redial regions");
		fprintf(_op, "\n  *                                                                        ");
		fprintf(_op, "\n  *************************************************************************");
		fprintf(_op, "\n  %-*s%-*s", 8,"layID",_width1,"power");
		for (i=0;i<MOCNumLayersZ;i++) {
			fprintf(_op, "\n  %-*d%-*.6e", 8,i,_width1,_NormPinPower_SumRad[i]);
		}
	}
}

/**
* Output normalized pin power of all pins in Core
*/

/**
* @ Function: OutputMacroXSs_FSRs()
*   introduction: output macroscopic cross sections for every fsr in subdomain
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified: 2017/05/07, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*/
void OutputMacroXSs_FSRs(int _procID,int _numLayersZ,int _numCellsPerLayer,int *_numFSRsInCell,int *_sumNumFSRsBasedGlobCellID,\
    int _numGroups,FP_PRECISION *_macroXS_t,FP_PRECISION *_macroXS_a,FP_PRECISION *_macroXS_f,FP_PRECISION *_macroXS_nf,\
	FP_PRECISION *_macroXS_s,FP_PRECISION *_macroXS_chi,FP_PRECISION **_inpXSs_t,FP_PRECISION **_inpXSs_a,\
	FP_PRECISION **_inpXSs_f,FP_PRECISION **_inpXSs_nf,FP_PRECISION ***_inpXSs_s,FP_PRECISION **_inpXSs_chi,\
	M_UINT64 *_MacroXS_s_NZRowStartID, M_UINT64 *_MacroXS_s_NZColStartID)
{
	if (OUTPUT_MacorXSs) {
		char _fileName[225];
		FILE *_op = NULL;
		int g, g1, _lay,_cell, _fsr, _ind, _ind1, _ind2,_ord;
		M_UINT64 _id,_count;
		FP_PRECISION _xs_t, _xs_a, _xs_f, _xs_nf, _chi, _xs_s;
//		sprintf(_fileName, "%s-%s-%s%d%s", CaseId, "macroXSsFSRs", "Proc", _procID, ".out");
        sprintf(_fileName, "macroXSsFSRs.out");
        _op = fopen(_fileName, "w");
		if (_op == NULL) {
			printf("\n [**ERROR**]  %s file opens error!", _fileName);
			fflush(stdout);
			exit(69);
		}

		fprintf(_op, "\n!  *****************************************************************");
		fprintf(_op, "\n!  *                                                               *");
		fprintf(_op, "\n!  *  The Macroscopic Cross-Sections for all FSRs in Process  %d   *", _procID);
		fprintf(_op, "\n!  *                                                               *");
		fprintf(_op, "\n!  *****************************************************************");
		for (_lay = 0; _lay < _numLayersZ; _lay++) {
			for (_cell = 0; _cell < _numCellsPerLayer; _cell++) {
				for (_fsr = 0; _fsr < _numFSRsInCell[_lay*_numCellsPerLayer+_cell]; _fsr++) {
					_ind = _sumNumFSRsBasedGlobCellID[_lay*_numCellsPerLayer + _cell] + _fsr;
					fprintf(_op, "\n\n!  MacroXSs for global FSR %d",_ind);
					fprintf(_op, "\n!  Cell   FSR");
					fprintf(_op, "%-*s%-*s",
						7, "  Group", \
						17, "  XS_t");

					if (ProblemType == 0 || ProblemType == 2) {
						fprintf(_op, "%-*s%-*s%-*s  ",
							17, "  XS_a", \
							17, "  XS_f", \
							17, "  XS_nf");
						for (g = 0; g < _numGroups; g++) {
							fprintf(_op, "XSs_s_g-%2d       ", g);
						}
						fprintf(_op, "Chi");
					}
					else if (ProblemType == 1) {
						fprintf(_op, "%-*s  ",17, "  FixSour");
					}
					else {}
					//if (ProblemType == 0) {
						fprintf(_op, "\n   %4d   %3d", _cell, _fsr);
					//}
					if (ProblemType == 0 || ProblemType == 2) {
						for (g = 0; g < _numGroups; g++) {
							// _xs_t = GetFSRXSs(_lay, _cell, _fsr,_numCellsPerLayer,_numGroups, g, _macroXS_t, _inpXSs_t,_sumNumFSRsBasedGlobCellID);
							// _xs_a = GetFSRXSs(_lay, _cell, _fsr,_numCellsPerLayer,_numGroups, g, _macroXS_a, _inpXSs_a,_sumNumFSRsBasedGlobCellID);
							// _xs_f = GetFSRXSs(_lay, _cell, _fsr,_numCellsPerLayer,_numGroups, g, _macroXS_f, _inpXSs_f,_sumNumFSRsBasedGlobCellID);
							// _xs_nf = GetFSRXSs(_lay, _cell, _fsr,_numCellsPerLayer,_numGroups, g, _macroXS_nf, _inpXSs_nf,_sumNumFSRsBasedGlobCellID);
							// _chi = GetFSRXSs(_lay, _cell, _fsr,_numCellsPerLayer,_numGroups, g, _macroXS_chi, _inpXSs_chi,_sumNumFSRsBasedGlobCellID);
							_ind1 = _ind*_numGroups+g;
							_xs_t = _macroXS_t[_ind1];
							_xs_a = _macroXS_a[_ind1];
							_xs_f = _macroXS_f[_ind1];
							_xs_nf = _macroXS_nf[_ind1];
							_chi = _macroXS_chi[_ind1];
							if (g==0) {
								fprintf(_op, "  %5d  %-15.8e  %-15.8e  %-15.8e  %-15.8e  ", g, _xs_t, _xs_a, _xs_f, _xs_nf);
							}
							else {
								fprintf(_op, "\n  %18d  %-15.8e  %-15.8e  %-15.8e  %-15.8e  ", g, _xs_t, _xs_a, _xs_f, _xs_nf);
							}

							_ind2 = _ind * (NumScattOrder + 1) * _numGroups + g;
							for (g1=0; g1<_MacroXS_s_NZColStartID[_ind2]; g1++) {
								_xs_s=0.0;
								fprintf(_op, "%-17.8e", _xs_s);
							}
							for (_id=_MacroXS_s_NZRowStartID[_ind2];_id<_MacroXS_s_NZRowStartID[_ind2+1];_id++) {
								_xs_s = _macroXS_s[_id];
								fprintf(_op, "%-17.8e", _xs_s);
							}
							_count=_MacroXS_s_NZRowStartID[_ind2+1] - _MacroXS_s_NZRowStartID[_ind2];
							for (g1 = _MacroXS_s_NZColStartID[_ind2] + _count; g1<_numGroups; g1++) {
								_xs_s=0.0;
								fprintf(_op, "%-17.8e", _xs_s);
							}

							fprintf(_op, "%-.8e", _chi);
						}

						if (NumScattOrder > 0) {
							for (_ord = 1; _ord <= NumScattOrder; _ord++) {
								fprintf(_op, "\n!  %d-order Scattering XSs",_ord);
								fprintf(_op, "%-*s", 7, "\n!  Group");
								for (g = 0; g < _numGroups; g++) {
							        fprintf(_op, "  XSs_s_g-%2d     ", g);
						        }
								for (g = 0; g < _numGroups; g++) {
								    fprintf(_op, "\n%8d", g);
									_ind2 = _ind * (NumScattOrder + 1) * _numGroups + _ord * _numGroups + g;
									for (g1=0; g1<_MacroXS_s_NZColStartID[_ind2]; g1++) {
								        _xs_s=0.0;
								        fprintf(_op, "  %-15.8e", _xs_s);
							        }
							        for (_id=_MacroXS_s_NZRowStartID[_ind2];_id<_MacroXS_s_NZRowStartID[_ind2+1];_id++) {
							        	_xs_s = _macroXS_s[_id];
							        	fprintf(_op, "  %-15.8e", _xs_s);
							        }
							        _count=_MacroXS_s_NZRowStartID[_ind2+1] - _MacroXS_s_NZRowStartID[_ind2];
							        for (g1 = _MacroXS_s_NZColStartID[_ind2] + _count; g1<_numGroups; g1++) {
							        	_xs_s=0.0;
							        	fprintf(_op, "  %-15.8e", _xs_s);
							        }
								}
							}

						}
					}
					else if (ProblemType == 1) {
						for (g = 0; g < _numGroups; g++) {
							if (g == 0) {
								fprintf(_op, "  %5d  %-15.8e  %-15.8e", g, _macroXS_t[_ind*_numGroups+g], \
									FixedSour[_ind*_numGroups + g] * _macroXS_t[_ind*_numGroups + g]);
							}
							else {
								fprintf(_op, "\n  %18d  %-15.8e  %-15.8e", g, _macroXS_t[_ind*_numGroups + g], \
									FixedSour[_ind*_numGroups + g] * _macroXS_t[_ind*_numGroups + g]);
							}
						}
					}
					else {}
				}
			}
		}
        fflush(stdout);  // refresh the memory
		fclose(_op);
		_op = NULL;
	}
}

/*
 * output macro XSs from input and generate XSs for other code (dragon, moc)
 */
/**
 * Output the Ray Tracing Module Information into a file
 */
/**
* @ Function: OutputCellBasedRTMGeoInfo()
*   introduction: output the geometric information of cell based ray tracing module
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified: 2017/05/07, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*/
/**
* @ Function: OutputCellBasedRTMGeoInfo()
*   introduction: output the geometric information of cell based ray tracing module
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified: 2017/05/07, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*/
/**
* @ Function: OutputRegionWisedRTMGeoInfo()
*   introduction: output the region-wised track info in ray tracing module
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2019/06/10, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: OutputSegInfoInRTMTrack()
*   introduction: output the segments info of tracks in ray tracing module
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2019/12/25, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
/**
* @ Function: OutputBasicTrackSetInfo()
*   introduction: output the geometric information of cell based ray tracing module
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified: 2017/05/07, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   2nd modified: 2019/06/10, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*/
/**
* @ Function: OutputFSRsCSGInfoInRTM()
*   introduction: output FSRs' information in ray tracing module
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified: 2017/05/07, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*/
/**
* @ Function: OutputMOCGeometryInfo()
*   introduction: output grometry information
* @ Parameter List: _procID - current process ID
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified: 2017/05/07, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*/
void OutputMOCGeometryInfo(int _procID, int _numTrackSet, int *_numAzimAng, int *_numPolarAng, int **_numTracks, \
	int **_numTracksX, int **_numTracksY, FP_PRECISION_GEO **_dxEff, FP_PRECISION_GEO **_dyEff, FP_PRECISION_GEO **_effAzimAng, \
	FP_PRECISION_GEO **_effRaySpacing, FP_PRECISION **_azimWeights, FP_PRECISION **_totalWeights, int _numRTMGeoType, \
	struct PIN_CELL *_cellBasedRTMGeo, struct RTMFSRINFO **_RTModFSRsInfo, int ****_index_MOC_TrackInRTM, int **_numSegsInTrack, \
	int **_segInfoInTrack_2PI_rayWisedFsrID, int **_fsrID_Map_orig_rayWised, struct SUBDOM_SEG_IN_TRACK_SINGLE **_segInfoInTrack_SINGLE_Total, \
	struct POINT ***_RTMStartP, struct POINT ***_RTMEndP, int _totalNumFSRsInRTM, int *_sumNumFSRsInRTM, \
	FP_PRECISION **_segInfoInTrack_2PI_length, int *_numLongtrack_halfSpace, int **_LTsIndex_AzimAngID, int **_LTsIndex_LTID, \
	struct SUBDOM_LONGTRACK_INFO **_longTracksInfo, int **_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK **_tracksInfoInLT, \
	int _numCellsInDom_Layer, int *_totNumTracksInOneRTM_2PI, int *_cellRTMGeoType, int **_indexTrackInRTM_SumAzim_2PI, \
	int **_segInfoInTrack_2PI_AntiClockFSR, int **_sumNumSegsInRTMTrack_2PI, \
	int *_sumNumFSRsBasedGlobCellID, FP_PRECISION_GEO *_FSRsAreaInRTM_AntiClockFSR)
{

	if (Output_GeometryInfo == 1) {
		//int _rtm,_azim;
		int _part1 = 1, _part2 = 1, _part3 = 1, _part4 = 1, _part5 = 1, _part6 = 1, _part7 = 0, _trackSet;
		char _fileName[225];
		FILE *_op = NULL;
		sprintf(_fileName, "%s-%s-%s%d%s", CaseId, "GeometryInformation", "Proc", _procID, ".out");
		_op = fopen(_fileName, "w");
		if (_op == NULL) {
			printf(" %s file opens error!\n", _fileName);
			exit(1);
		}
		//if (InputTrackFile == 0) {
			// Part 1
			if (_part1 == 1) {
				fprintf(_op, "\n  *****************************************************************");
				fprintf(_op, "\n  *    OutputSegInfoInLongTrack                                                           *");
				fprintf(_op, "\n  *              2D MOC Geometry Information                      *");
				fprintf(_op, "\n  *          PART 1: Basic Track Set information                  *");
				fprintf(_op, "\n  *                                                               *");
				fprintf(_op, "\n  *****************************************************************");
				OutputBasicTrackSetInfo(_op, _numTrackSet, _numAzimAng, _numPolarAng, _numTracks, _numTracksX, _numTracksY, \
					_dxEff, _dyEff, _effAzimAng, _effRaySpacing, _azimWeights, _totalWeights, _RTMStartP, _RTMEndP);
			}
		// 	// part 2
		// 	if (_part2 == 1) {
		// 		fprintf(_op, "\n\n  *****************************************************************");
		// 		fprintf(_op, "\n  *                                                               *");
		// 		fprintf(_op, "\n  *              2D MOC Geometry Information                      *");
		// 		fprintf(_op, "\n  *          PART 2: Basic Ray Tracing Module Info                *");
		// 		fprintf(_op, "\n  *                                                               *");
		// 		fprintf(_op, "\n  *****************************************************************");
		// 		OutputCellBasedRTMGeoInfo(_op, _numRTMGeoType, _cellBasedRTMGeo, _RTModFSRsInfo);
		// 	}


		// 	// Part 3
		// 	if (_part3 == 1) {
		// 		fprintf(_op, "\n  *****************************************************************");
		// 		fprintf(_op, "\n  *                                                               *");
		// 		fprintf(_op, "\n  *              2D MOC Geometry Information                      *");
		// 		fprintf(_op, "\n  *       PART 3: Segments Info In Ray Tracing Module             *");
		// 		fprintf(_op, "\n  *                                                               *");
		// 		fprintf(_op, "\n  *****************************************************************");
		// 		OutputSegInfoInRTM(_op, _numRTMGeoType, _numTrackSet, _numAzimAng, _cellBasedRTMGeo, _RTModFSRsInfo);
		// 	}

		// 	// part 3
		// 		//fprintf(_op, "\n  *****************************************************************");
		// 		//fprintf(_op, "\n  *                                                               *");
		// 		//fprintf(_op, "\n  *              2D MOC Geometry Information                      *");
		// 		//fprintf(_op, "\n  *          PART 3: Track Info in Ray Tracing Module             *");
		// 		//fprintf(_op, "\n  *                                                               *");
		// 		//fprintf(_op, "\n  *****************************************************************\n");
		// 		// for (_rtm = 0; _rtm < NumRTMGeoType; _rtm++) {
		// 		// 	for (_azim = 0; _azim < NumAzimAng; _azim++) {
		// 		// 		output_trackInfo_RTM_Azim(_rtm, _azim, _op);
		// 		// 	}
		// 		// }

		// 	// part 4
		// 	if (_part4 == 1) {
		// 		fprintf(_op, "\n  *****************************************************************");
		// 		fprintf(_op, "\n  *                                                               *");
		// 		fprintf(_op, "\n  *              2D MOC Geometry Information                      *");
		// 		fprintf(_op, "\n  *       PART 4: Segments Info in Ray Tracing Module             *");
		// 		fprintf(_op, "\n  *                                                               *");
		// 		fprintf(_op, "\n  *****************************************************************\n");
		// 		OutputSegInfoInRTMTrack(_op, _numTrackSet, _numRTMGeoType, _numAzimAng, _numTracks, \
		// 			_index_MOC_TrackInRTM, _numSegsInTrack, _segInfoInTrack_SINGLE_Total, \
		// 			_segInfoInTrack_2PI_length);
		// 	}

		// 	// part 5
		// 	if (_part5 == 1) {
		// 		fprintf(_op, "\n  *****************************************************************");
		// 		fprintf(_op, "\n  *                                                               *");
		// 		fprintf(_op, "\n  *              2D MOC Geometry Information                      *");
		// 		fprintf(_op, "\n  *       PART 5: LongTracks Info in Whole Geometry               *");
		// 		fprintf(_op, "\n  *                                                               *");
		// 		fprintf(_op, "\n  *****************************************************************\n");
		// 		OutputLongTrackInfo(_numAzimAng[0], _op);
		// 	}
		// //}

		if (_part6 == 1) {
			fprintf(_op, "\n  *****************************************************************");
			fprintf(_op, "\n  *                                                               *");
			fprintf(_op, "\n  *              2D MOC Geometry Information                      *");
			fprintf(_op, "\n  *    PART 6: Segments Infomation in Long Tracks for Subdomain   *");
			fprintf(_op, "\n  *                                                               *");
			fprintf(_op, "\n  *****************************************************************\n");
			_trackSet = 0;
			OutputSegInfoInLongTrack(_op, _numLongtrack_halfSpace[_trackSet], _LTsIndex_AzimAngID[_trackSet], \
				_LTsIndex_LTID[_trackSet], _longTracksInfo[_trackSet], _trackIndexInLTs[_trackSet], _tracksInfoInLT[_trackSet], \
				_numCellsInDom_Layer, _totNumTracksInOneRTM_2PI[_trackSet], _cellRTMGeoType, _indexTrackInRTM_SumAzim_2PI[_trackSet], \
				_segInfoInTrack_2PI_length[_trackSet], _segInfoInTrack_2PI_AntiClockFSR[_trackSet], _sumNumSegsInRTMTrack_2PI[_trackSet], \
				_sumNumFSRsBasedGlobCellID, _FSRsAreaInRTM_AntiClockFSR, _sumNumFSRsInRTM, _numSegsInTrack[_trackSet]);
		}

		// //if (InputTrackFile == 0) {
		// 	/* part 7 */
		// 	if (_part7 == 1) {
		// 		fprintf(_op, "\n  *****************************************************************");
		// 		fprintf(_op, "\n  *                                                               *");
		// 		fprintf(_op, "\n  *              2D MOC Geometry Information                      *");
		// 		fprintf(_op, "\n  *    PART 7: Region-wised Track Info in Ray Tracing Module      *");
		// 		fprintf(_op, "\n  *                                                               *");
		// 		fprintf(_op, "\n  *****************************************************************\n");
		// 		OutputRegionWisedRTMGeoInfo(_op, _numTrackSet, _numRTMGeoType, _numAzimAng, \
		// 			_segInfoInTrack_2PI_rayWisedFsrID, _fsrID_Map_orig_rayWised, _numTracks, \
		// 			_index_MOC_TrackInRTM, _numSegsInTrack, _segInfoInTrack_SINGLE_Total, _totalNumFSRsInRTM, _sumNumFSRsInRTM);
		// 	}
		// //}

		fclose(_op);
		_op = NULL;
	}
}


/**
* SongPeitao 20180425
* Output the Geometry Information into a file
*/
// output incoming and outgoing angular flux of subdomian longtrcks for a specific zimuthal angle
/**
 * output the LongTracks Information into a file
 */
/**
 * SongPeitao 20180425
 * Write the track Information into a file
 */
/**
 * @ Function: OutputTrackInfo_RTM()
 *   introduction: output the track information of cell based ray tracing module
 * @ Parameter List: _procID - current process ID
 *   Return: void
 * @ Created     : 2018/04/25, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified: 2019/06/10, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 */
/**
* @ Function: OutputMemoryFootprint()
*   introduction: output the memory allocated for a pointer with specific type
* @ Parameter List: _FilePointerMF - FILEE *pointer for output
*                   _dataType - data type
*                   _parameter - parameter name
*                   _dataNumber - data number in this memeory, e.g., number of elements in an array
*                   _pointerSizeMB - memory comsumed by storing the pointer (address)
*                   _dataSizeMB - how many bites that the data allocated in this memory
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void OutputMemoryFootprint(FILE * _FilePointerMF,char *_dataType, char *_parameter,\
	M_UINT64 _dataNumber, FP_PRECISION _pointerSizeMB, FP_PRECISION _dataSizeMB)
{

	strcpy(ParaName[ParamCounter], _parameter);
	ParaMemSize[ParamCounter]= _pointerSizeMB + _dataSizeMB;
	NumTotalData = NumTotalData + _pointerSizeMB + _dataSizeMB;
	if (OUTPUT_MemoryFootprint) {
		fprintf(_FilePointerMF, "\n%*d%*s%*s%*llu%*.4lf%*.4lf%*.4lf%*.4lf", \
			MEM_1, ParamCounter, \
			MEM_2, _parameter, \
			MEM_3, _dataType, \
			MEM_4, _dataNumber, \
			MEM_5, _pointerSizeMB, \
			MEM_6, _dataSizeMB, \
			MEM_8, _dataSizeMB + _pointerSizeMB, \
			MEM_10, NumTotalData
		);
	}
	ParamCounter++;
}

void OutputFreeMemory(char *_parameter)
{
	if (OUTPUT_MemoryFootprint) {
		int i;
		for (i = 0; i < NumPara; i++) {
			if (strcmp(ParaName[i], _parameter) == 0) {
				NumTotalData = NumTotalData - ParaMemSize[i];
				fprintf(FP_MemoryFootprint, "\n     Free  Parameter NO.%3d  %-*s ,  Memory Change (MB) %*.4lf    %*.4lf", \
					i + 1,\
					MEM_2, _parameter, \
					MEM_6, -1*ParaMemSize[i], \
					MEM_10+26, (FP_PRECISION)(NumTotalData) / 1024. / 1024.
				);
				break;
			}
		}

	}
}

/* CMFD and legacy timing-report exporters are outside the paper transport
 * executable. */



void OutputFSRsScalarFlux_Mat(int _procID, int _numCellsInDom, int *_numFSRsInCell, FP_PRECISION *_flux, FP_DOUBLE _moc_glo_fiss_tem)
{
	if (OUTPUT_flux_mat==1) {
		int cell, reg, g, _lay, _fsrID, _width1=8, _width2=16, i, _mat, _ind;
		FILE *_op = NULL;
		char _fileName[225];
		FP_DOUBLE _aver_vfiss_rate, _vol;
		_aver_vfiss_rate = 1 / _moc_glo_fiss_tem; // normlize the flux to make sure the total fission rate is the volume of the core

		if (IfResonance==0 && INPUT_XSs_MATorFSR==0) {
			for (i=0;i<NumXSs_Mat;i++) {
				for (g = 0; g < NumGroups; g++) {
					MatSumFlux[i][g]=0.0;
				}
				MatSumVol[i]=0.0;
			}
	    }

		for (_lay = 0; _lay < MOCNumLayersZ; _lay++) {
			for (cell = 0; cell < _numCellsInDom; cell++) {
				for (reg = 0; reg < _numFSRsInCell[_lay*_numCellsInDom+cell]; reg++) {
					_fsrID = SumNumFSRsBasedGlobCellID[_lay*_numCellsInDom+cell]+reg;
				    _vol=FSRVol_SubDom[_fsrID];
					_mat=FSRsXSID_subDom[_fsrID];
					// if (_lay==0 && _mat==3) {
					// 	printf("\n _lay=%d, cell=%d, reg=%d, _fsrID=%d, FSRVol_SubDom=%e, _vol=%e, MatSumVol=%e",_lay, cell, reg, _fsrID, FSRVol_SubDom[_fsrID],_vol,MatSumVol[_mat]);
					// }
					MatSumVol[_mat]+=_vol;

					for (g = 0; g < NumGroups; g++) {
						_ind=_fsrID*NumGroups + g;
						MatSumFlux[_mat][g]+=(FP_DOUBLE)(_flux[_ind])*_vol;
					}
				}
			}
		}


			sprintf(_fileName, "%s-%s-%s%d%s", CaseId, "MatAvgScalarFlux", "ProcID", _procID, ".out");
			_op = fopen(_fileName, "w");
		    if (_op == NULL) {
			    printf("\n [**ERROR**]  %s file opens error!\n", _fileName);
			    fflush(stdout);
			    exit(69);
		    }
			fprintf(_op, "\n!  **********************************************************************");
			fprintf(_op, "\n!  *                                                                    *");
		    fprintf(_op, "\n!  *             The Avg. Scalar Flux for Materials in Core             *");
		    fprintf(_op, "\n!  *                                                                    *");
		    fprintf(_op, "\n!  **********************************************************************");
			fprintf(_op, "\n the total nu-fission rate is: %.6e", _moc_glo_fiss_tem);
			for (g = 0; g < NumGroups; g++) {
				fprintf(_op, "\n\n!  ################################################################");
			    fprintf(_op, "\n!  group %d, mat avg. scalar flux",g);
				fprintf(_op, "\n %*s%*s%*s%*s%*s", _width1, "mat", _width2, "sum(phi*vol)", _width2, "sum(vol)", _width2, "avg. phi", _width2, "norm phi");
				for (i=0;i<NumXSs_Mat;i++) {
					if (MatSumFlux[i][g]>0.0) {
						fprintf(_op, "\n %*d%*.6e%*.6e%*.6e%*.6e", _width1, i, _width2, MatSumFlux[i][g], _width2, MatSumVol[i], \
					        _width2, MatSumFlux[i][g]/MatSumVol[i], _width2, MatSumFlux[i][g]/MatSumVol[i]*_aver_vfiss_rate);
					}
				    else {
						fprintf(_op, "\n %*d%*s%*s%*s%*s", _width1, i, _width2, "", _width2, "", \
					        _width2,"", _width2, "");
					}
			    }
			}
		fclose(_op);
		_op = NULL;
	}
}

void OutputFSRsScalarFlux(int _procID, int _numCellsInDom, int *_numFSRsInCell, FP_PRECISION *_flux, FP_DOUBLE _moc_glo_fiss_tem)
{
	if (OUTPUT_flux==1) {
		int cell, reg, g, _lay, _fsrID, _width1=8, _width2=16, _ind;
		FILE *_op = NULL;
		char _fileName[225];
		FP_DOUBLE _aver_vfiss_rate;
		_aver_vfiss_rate = 1. / _moc_glo_fiss_tem; // normlize the flux to make sure the total fission rate is the volume of the core

		sprintf(_fileName, "%s-%s-%s%d%s", CaseId, "FSRsScalarFlux", "ProcID", _procID, ".out");
		_op = fopen(_fileName, "w");
		if (_op == NULL) {
			printf("\n [**ERROR**]  %s file opens error!\n", _fileName);
			fflush(stdout);
			exit(69);
		}
		fprintf(_op, "\n!  **********************************************************************");
		fprintf(_op, "\n!  *                                                                    *");
		fprintf(_op, "\n!  *              The Scalar Flux all FSRs and all Groups               *");
		fprintf(_op, "\n!  *                                                                    *");
		fprintf(_op, "\n!  **********************************************************************");
		for (_lay = 0; _lay < MOCNumLayersZ; _lay++) {
			fprintf(_op, "\n\n!  ################################################################");
			fprintf(_op, "\n!  Layer %d, Scalar Flus Maps", _lay);

			for (cell = 0; cell < _numCellsInDom; cell++) {
				for (reg = 0; reg < _numFSRsInCell[_lay*_numCellsInDom+cell]; reg++) {
					_fsrID = SumNumFSRsBasedGlobCellID[_lay*_numCellsInDom+cell]+reg;
					fprintf(_op, "\n!%*s%*s%*s%*s%*s%*s",_width1,"Layer", _width1, "Cell", _width1, "FSR", _width1, "Group", _width2, "Flux", _width2, "Normlize");
					fprintf(_op, "\n %*d%*d%*d", _width1, _lay, _width1, cell, _width1, reg);
					for (g = 0; g < NumGroups; g++) {
						_ind=_fsrID*NumGroups + g;
						if (g == 0) {
							fprintf(_op, "%*d%*.8e%*.8e\n", _width1, g, _width2, _flux[_ind], _width2, (FP_DOUBLE)(_flux[_ind]) * _aver_vfiss_rate);
						}
						else {
							fprintf(_op, " %*d%*.8e%*.8e\n", _width1*4, g, _width2, _flux[_ind], _width2, (FP_DOUBLE)(_flux[_ind]) * _aver_vfiss_rate);
						}
					}
				}
				//fprintf(_op, "\n!**************************************************************\n");
			}
		}
		fclose(_op);
		_op = NULL;
	}
}



void OutputFSRsScalarFluxMoment(int _procID, int _numCellsInDom, int *_numFSRsInCell, FP_PRECISION *_flux, int _scatt_order) {
	if (OUTPUT_flux==1) {
		int cell, reg, g, _lay, _fsrID, _width1=8, _width2=16, _ind, _order, _l, _count;
		FILE *_op = NULL;
		char _fileName[225];
		FP_DOUBLE _aver_vfiss_rate=1.;

//		sprintf(_fileName, "%s-%s-%s%d%s", CaseId, "FSRsScalarFluxMoment", "ProcID", _procID, ".out");
        sprintf(_fileName, "FSRsScalarFluxMoment.out");
		_op = fopen(_fileName, "w");
		if (_op == NULL) {
			printf("\n [**ERROR**]  %s file opens error!\n", _fileName);
			fflush(stdout);
			exit(69);
		}
		fprintf(_op, "\n!  **********************************************************************");
		fprintf(_op, "\n!  *                                                                    *");
		fprintf(_op, "\n!  *          The Scalar Flux Moment all FSRs and all Groups            *");
		fprintf(_op, "\n!  *                                                                    *");
		fprintf(_op, "\n!  **********************************************************************");
		fprintf(_op, "\n");
		fprintf(_op, "\n!  PN_ORDER  %d    NUM_FSRS  %d    NUM_GROUPS  %d", _scatt_order, TotalNumFSRs_subDom, NumGroups);

		// scattering order loop and level loop
		_count = 0;
		for (_order = 0; _order <= _scatt_order; _order++) {
			for (_l = -_order; _l <= _order; _l++) {

				fprintf(_op, "\n! Order= %d    Level= %d", _order, _l);

				// fsr loop and group loop
				for (_lay = 0; _lay < MOCNumLayersZ; _lay++) {
					fprintf(_op, "\n\n!  ################################################################");
					fprintf(_op, "\n!  Layer %d, Scalar Flux Maps", _lay);


					for (cell = 0; cell < _numCellsInDom; cell++) {
						for (reg = 0; reg < _numFSRsInCell[_lay*_numCellsInDom+cell]; reg++) {
							_fsrID = SumNumFSRsBasedGlobCellID[_lay*_numCellsInDom+cell]+reg;
							fprintf(_op, "\n!%*s%*s%*s%*s%*s%*s",_width1,"Layer", _width1, "Cell", _width1, "FSR", _width1, "Group", _width2, "Flux", _width2, "Normlize");
							fprintf(_op, "\n %*d%*d%*d", _width1, _lay, _width1, cell, _width1, reg);
							for (g = 0; g < NumGroups; g++) {
								_ind=  _count * TotalNumFSRs_subDom * NumGroups + _fsrID*NumGroups + g;
								if (g == 0) {
									fprintf(_op, "%*d%*.8e%*.8e\n", _width1, g, _width2, _flux[_ind], _width2, (FP_DOUBLE)(_flux[_ind]) * _aver_vfiss_rate);
								}
								else {
									fprintf(_op, " %*d%*.8e%*.8e\n", _width1*4, g, _width2, _flux[_ind], _width2, (FP_DOUBLE)(_flux[_ind]) * _aver_vfiss_rate);
								}
							}
						}
						fprintf(_op, "\n!**************************************************************\n");
					}
				}

				_count++;

			}
		}

		fclose(_op);
		_op = NULL;
	}
}




void PrintDeviceMemUsage(FILE *_fp,char *_parameter)
{
#if defined(CUDA_USED)
    int _caseID=0, _width=12;
	if (strcmp((char *)("Geo and Track Info"),_parameter)==0) {
		printf("%-*.1lf", _width, (FP_PRECISION_TIME)(GPUPreMemTrack[0]) * M_MB_Unit);
        fflush(stdout);
		fprintf(_fp, "%-*.1lf", _width, (FP_PRECISION_TIME)(GPUPreMemTrack[0]) * M_MB_Unit);
	}
	else if (strcmp((char *)("Quadrature Info"),_parameter)==0) {
		printf("%-*.1lf", _width, (FP_PRECISION_TIME)(GPUPreMemQuadrSet[0]) * M_MB_Unit);
        fflush(stdout);
		fprintf(_fp, "%-*.1lf", _width, (FP_PRECISION_TIME)(GPUPreMemQuadrSet[0]) * M_MB_Unit);
	}
	else if (strcmp((char *)("Exp Tabel"),_parameter)==0) {
		printf("%-*.1lf", _width, (FP_PRECISION_TIME)(GPUPreMemExpTab[0]) * M_MB_Unit);
        fflush(stdout);
		fprintf(_fp, "%-*.1lf", _width, (FP_PRECISION_TIME)(GPUPreMemExpTab[0]) * M_MB_Unit);
	}
	else if (strcmp((char *)("Scalar Flux and Source Info"),_parameter)==0) {
		printf("%-*.1lf", _width, (FP_PRECISION_TIME)(GPUPreMemScalF[0]) * M_MB_Unit);
        fflush(stdout);
		fprintf(_fp, "%-*.1lf", _width, (FP_PRECISION_TIME)(GPUPreMemScalF[0]) * M_MB_Unit);
	}
	else if (strcmp((char *)("Radial Ang Flux Info"),_parameter)==0) {
		printf("%-*.1lf", _width, (FP_PRECISION_TIME)(GPUPreMemRadF[0]) * M_MB_Unit);
        fflush(stdout);
		fprintf(_fp, "%-*.1lf", _width, (FP_PRECISION_TIME)(GPUPreMemRadF[0]) * M_MB_Unit);
	}
	else if (strcmp((char *)("Axial Ang Flux Info"),_parameter)==0) {
		printf("%-*.1lf", _width, (FP_PRECISION_TIME)(GPUPreMemAxiF[0]) * M_MB_Unit);
        fflush(stdout);
		fprintf(_fp, "%-*.1lf", _width, (FP_PRECISION_TIME)(GPUPreMemAxiF[0]) * M_MB_Unit);
	}
	else if (strcmp((char *)("XSs Info"),_parameter)==0) {
		printf("%-*.1lf", _width, (FP_PRECISION_TIME)(GPUPreMemXSs[0]) * M_MB_Unit);
        fflush(stdout);
		fprintf(_fp, "%-*.1lf", _width, (FP_PRECISION_TIME)(GPUPreMemXSs[0]) * M_MB_Unit);
	}
	else if (strcmp((char *)("MG CMFD Info"),_parameter)==0) {
		printf("%-*.1lf", _width, (FP_PRECISION_TIME)(GPUPreMemMGCMFD[0]) * M_MB_Unit);
        fflush(stdout);
		fprintf(_fp, "%-*.1lf", _width, (FP_PRECISION_TIME)(GPUPreMemMGCMFD[0]) * M_MB_Unit);
	}
	else {
		printf("%-*s", _width, "----");
        fflush(stdout);
		fprintf(_fp, "%-*s", _width, "----");
	}
#endif //defined(CUDA_USED)
}

