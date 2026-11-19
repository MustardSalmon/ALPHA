/* Restored shared functions required by the supported Type 13--16 path. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "MOC_AllocAndFreeMem.h"
#include "MOC_Assembly.h"
#include "MOC_Boundary.h"
#include "MOC_Cell.h"
#include "MOC_CMFD.h"
#include "MOC_CommonMacro.h"
#include "MOC_Core.h"
#include "MOC_EigenValueEvaluation.h"
#include "MOC_ExpEvaluation.h"
#include "MOC_FluxEvaluation.h"
#include "MOC_Geometry.h"
#include "MOC_GPUCPUInterface.h"
#include "MOC_Input.h"
#include "MOC_Lattice.h"
#include "MOC_Macro_XSs.h"
#include "MOC_main.h"
#include "MOC_Output.h"
#include "MOC_Pin_Cell.h"
#include "MOC_Quadrature_Set.h"
#include "MOC_ReacRateEvaluation.h"
#include "MOC_Track.h"
#include "MOC_VectorOperation.h"
#include "MOC_Visualization.h"
#include "paper_transport_compat.h"
#ifdef CUDA_USED
#include "MOC_GPUQuery.cuh"
#include "MOC_GPUSolver.cuh"
#include "MOC_GPU_TSSolver.cuh"
#include "cuda_runtime.h"
#endif
#ifdef MPI_USED
#include <mpi.h>
#endif

#ifdef CUDA_USED
void GPUMOC_TransSweep_UporDown(int _dim, int _layStart, int _layAdd, int _surfLayInc, \
	int _polarBoundLowLim, int _polarBoundUpLim, int _polOffset, FP_PRECISION *_axiFlux_inp, \
	FP_PRECISION *_axiFlux_avg, int _nodeLay, int _surfLay, FP_PRECISION _JzDir, int _dir,\
	int _iter, int _fisrtInvoke, int _innerIter, int _azimBegin, int _azimEnd)
{
    int _lay, _dim1, _dim2;
	FP_PRECISION _twoOverLayWidth, keff;

	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
	FP_PRECISION_TIME _phase_start = 0., _phase_end = 0.;
	FP_PRECISION_TIME _subphase_start = 0., _subphase_end = 0.;
#ifdef TIME_REC
	if (_iter==0 && _fisrtInvoke==1) {
		TimeStamp_SE = TimeCountMOC;
		strcpy(TimeMetricsMOC[TimeStamp_SE], "TS Source Evaluation");
		TimeCountMOC++;
#if defined(M_3DMOCex) && defined(M_MG_CMFD)
        TimeStamp_MOC_TS1=TimeCountMOC;
		strcpy(TimeMetricsMOC[TimeStamp_MOC_TS1], "TS Get AxiNetCur");
		TimeCountMOC++;
#endif // defined(M_3DMOCex) && defined(M_MG_CMFD)

        TimeStamp_CommCGCopy=TimeCountMOC;
		strcpy(TimeMetricsMOC[TimeStamp_CommCGCopy], "Radial F Load C->G");
		TimeCountMOC++;

        TimeStamp_MOC_TS2=TimeCountMOC;
        strcpy(TimeMetricsMOC[TimeStamp_MOC_TS2], "TS Sweep on Layer");
		TimeCountMOC++;

		TimeStamp_CommGCCopy=TimeCountMOC;
		strcpy(TimeMetricsMOC[TimeStamp_CommGCCopy], "Radial F Store G->C");
		TimeCountMOC++;


#if defined(M_3DMOCex)
        TimeStamp_MOC_TS3=TimeCountMOC;
        strcpy(TimeMetricsMOC[TimeStamp_MOC_TS3], "TS Accu Flux");
		TimeCountMOC++;
		TimeStamp_MOC_TS4=TimeCountMOC;
        strcpy(TimeMetricsMOC[TimeStamp_MOC_TS4], "TS Update AxSurf AngF");
		TimeCountMOC++;
		TimeStamp_MOC_TS5=TimeCountMOC;
        strcpy(TimeMetricsMOC[TimeStamp_MOC_TS5], "TS Update AxBound AngF");
		TimeCountMOC++;
#endif // defined(M_3DMOCex)
	}
#endif // TIME_REC


	for (_lay = _layStart; _lay >= 0 && _lay < MOCNumLayersZ; _lay = _lay + _layAdd) {
		_twoOverLayWidth = 2.0 / WidthLayersInCore[_lay];

#ifdef TIME_REC
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC
		/* evaluate source for current layer */
        _time_start = TallyCurrentTime_CPU();
        cudaDeviceSynchronize();
        if (_innerIter == 0) {
            if (isTransient < 2) keff = MOCKeff;
            else keff = sKeff;
            // printf("\n _lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);
            // GPUComputeFSRSource_Layer(_lay, TotalNumFSRs_subDom, TotalNumFSRs_perLayer, NumGroups,
            //                           MacroXS_s_NZStartID_Layer,
            //                           MacroXS_s_NNZ_Layer, ProblemType, MacroXS_t, MacroXS_nf, Chi, MacroXS_s,
            //                           FixedSour,
            //                           (FP_PRECISION) (1.0) / keff, NumScattOrder, GPU_TSKernelType,
            //                           MacroXS_s_NZRowStartID_Layer,
            //                           MacroXS_s_NZColStartID, ScalarFluxOld, _polarBoundLowLim,
            //                           isTransient, FissSource, tranSrcConstA, tranSrcConstB, tranSrcConstC);
        }
		// TEST_DEBUG_NEED
		if (_azimBegin >= 0) {
			GPUComputeFSRSource_Layer_PerAzim(_lay, TotalNumFSRs_subDom, TotalNumFSRs_perLayer, NumGroups,
										  MacroXS_s_NZStartID_Layer,
										  MacroXS_s_NNZ_Layer, ProblemType, MacroXS_t, MacroXS_nf, Chi, MacroXS_s,
										  FixedSour,
										  (FP_PRECISION) (1.0) / keff, NumScattOrder, GPU_TSKernelType,
						  MacroXS_s_NZRowStartID_Layer,
						  MacroXS_s_NZColStartID, ScalarFluxOld, _polarBoundLowLim,
						  _azimBegin, _azimEnd, _azimBegin == 0 ? 0 : 1,
										  isTransient, FissSource, tranSrcConstA, tranSrcConstB, tranSrcConstC);
		}
		else {
            GPUComputeFSRSource_Layer(_lay, TotalNumFSRs_subDom, TotalNumFSRs_perLayer, NumGroups,
                                      MacroXS_s_NZStartID_Layer,
                                      MacroXS_s_NNZ_Layer, ProblemType, MacroXS_t, MacroXS_nf, Chi, MacroXS_s,
                                      FixedSour,
                                      (FP_PRECISION) (1.0) / keff, NumScattOrder, GPU_TSKernelType,
                                      MacroXS_s_NZRowStartID_Layer,
                                      MacroXS_s_NZColStartID, ScalarFluxOld, _polarBoundLowLim,
									  isTransient, FissSource, tranSrcConstA, tranSrcConstB, tranSrcConstC);
		}
        cudaDeviceSynchronize();
        _time_end = TallyCurrentTime_CPU();
        TimeRec_SourceCalc += _time_end - _time_start;
#ifdef TIME_REC
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_end = TallyCurrentTime_CPU();
	TimeMOC[TimeStamp_SE] += _time_end - _time_start;
#endif // TIME_REC

#if defined(M_3DMOCex) && defined(M_MG_CMFD)
#ifdef TIME_REC
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC
		GPUGetNetCurr_Axial_1(_lay, _lay + _surfLayInc, _JzDir, _polarBoundLowLim, \
		    _polarBoundUpLim, _axiFlux_inp, _polOffset);
#ifdef TIME_REC
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_end = TallyCurrentTime_CPU();
	TimeMOC[TimeStamp_MOC_TS1] += _time_end - _time_start;
#endif // TIME_REC
#endif // defined(M_3DMOCex) && defined(M_MG_CMFD)

#if defined(M_3DOpt) && defined(M_3DMOCex)
        GPUComputeSourceOverSigma_3DMOCex_1(_lay,_polarBoundLowLim,_polarBoundUpLim,_polOffset, \
            _axiFlux_inp, _twoOverLayWidth);
#endif // defined(M_3DOpt) && defined(M_3DMOCex)

#ifdef TIME_REC
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC

#ifdef M_3DMOCex
    LoadBoundaryFluxCPUtoGPU_Layer(_lay, NumLTs_HS[0] * 2, NumPolarAng[0], NumGroups, BoundaryFlux, NumProcs, CPUMPIFlux);
#endif

#ifdef TIME_REC
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_end = TallyCurrentTime_CPU();
	TimeMOC[TimeStamp_CommCGCopy] += _time_end - _time_start;
#endif // TIME_REC

#ifdef TIME_REC
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC

#if defined(M_2DMOC)
    if (NumScattOrder == 0 && GPU_TSKernelType < 10) {
	    _dim2 = TotalNumFSRs_perLayer * (NumScattOrder + 1) * (NumScattOrder + 1) * NumGroups;
	    GPUInitScalarFlux_Dev(_dim2, 0.0);
	}
#endif

#if defined(M_3DMOCex)
	if (GPU_TSKernelType == 12) {
#ifdef TIME_REC
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC
// printf("\n lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);
		GPUManipulateAxialInciAngFlux_1(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0], _axiFlux_inp, \
		    _twoOverLayWidth, _polarBoundLowLim, _polarBoundUpLim, _polOffset);

#ifdef TIME_REC
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_end = TallyCurrentTime_CPU();
	TimeMOC[TimeStamp_SE] += _time_end - _time_start;
#endif // TIME_REC
	}
#endif //defined(M_3DMOCex)

	        cudaDeviceSynchronize();
	        _time_start = TallyCurrentTime_CPU();
	GPUTransportSweep(_lay, _twoOverLayWidth, _polarBoundLowLim,_polarBoundUpLim,\
			_axiFlux_inp, _axiFlux_avg, _polOffset, GPU_TSKernelType, NumPinsInSubDomPerLayer, NumPolarAng[0], _azimBegin, _azimEnd);

#ifdef TIME_REC
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_end = TallyCurrentTime_CPU();
	TimeMOC[TimeStamp_MOC_TS2] += _time_end - _time_start;
#endif // TIME_REC

#ifdef TIME_REC
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC

#ifdef M_3DMOCex
	StoreBoundaryFluxGPUtoCPU_Layer(_lay, NumLTs_HS[0] * 2, NumPolarAng[0], NumGroups, BoundaryFlux, NumProcs, CPUMPIFlux);
#endif

#ifdef TIME_REC
#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
#endif // defined(CUDA_USED)
	_time_end = TallyCurrentTime_CPU();
	TimeMOC[TimeStamp_CommGCCopy] += _time_end - _time_start;
#endif // TIME_REC

#if defined(M_3DOpt) && defined(M_3DMOCex)
        GPUCompAxialAngF_FromSourceOverSigma_3DMOCex_1(_lay,_polarBoundLowLim,_polarBoundUpLim,_polOffset, \
            _axiFlux_inp, _twoOverLayWidth);
#endif // defined(M_3DOpt) && defined(M_3DMOCex)

#if defined(M_2DMOC)
	    if (NumScattOrder > 0 || GPU_TSKernelType >= 10) {
		cudaDeviceSynchronize();
		_phase_start = TallyMonotonicTime_CPU();
        _dim1 = TotalNumFSRs_perLayer * NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1);

        // todo: GPUAccumAngFlux need to be optimized
	    if (_azimBegin >= 0) {
			GPUAccumAngFlux_PerAzimBatch(_lay, _polarBoundLowLim, _polarBoundUpLim,
				_polOffset, _axiFlux_avg, _dim1, _azimBegin, _azimEnd);
		}
		else {
			GPUAccumAngFlux(_lay, _polarBoundLowLim, _polarBoundUpLim, _polOffset,\
				_axiFlux_avg, GPU_TSKernelType, _dim1, TotalNumFSRs_perLayer, NumGroups, NumAzimAng[0], NumPolarAng[0], NumScattOrder);
		}

//	    ScalarFluxCopy_Dev2Host(ScalarFlux_MOCLayer, (size_t)(_dim1), 0, (char *)("ScalarFlux_MOCLayer-D2H"));
		// printf("\n _lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);
		// printf("\n &&&&& ScalarFlux_MOC[ _lay*_dim1=%d]=%e", _lay*_dim1,ScalarFlux_MOC[ _lay*_dim1]);
//		AccuFluxByLayer(_lay, TotalNumFSRs_subDom, TotalNumFSRs_perLayer, NumGroups, \
//            NumScattOrder, ScalarFlux_MOC, ScalarFlux_MOCLayer);
		{
			const int _azimBuffer = GPUAzimSweepMode > 0 && GPUAzimSweepMode < NumAzimAng[0]
				? GPUAzimSweepMode : NumAzimAng[0];
			_dim1 = TotalNumFSRs_perLayer * NumGroups * _azimBuffer * NumPolarAng[0];
		}
		_subphase_start = TallyMonotonicTime_CPU();
		GPUInitAngFlux_Dev(_dim1, 0.0, 0);
		cudaDeviceSynchronize();
		_subphase_end = TallyMonotonicTime_CPU();
		TimeRec_FluxMomentAngularFluxReset += _subphase_end - _subphase_start;
		_phase_end = TallyMonotonicTime_CPU();
		TimeRec_FluxAccumulation += _phase_end - _phase_start;
	}
	else {
		_dim1 = TotalNumFSRs_perLayer * NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1);
//		ScalarFluxCopy_Dev2Host(ScalarFlux_MOC, (size_t)(_dim1), 0, (char *)("ScalarFlux_MOC-D2H"));
	}
#endif // defined(M_2DMOC)

#if defined(M_3DMOCex)
#ifdef TIME_REC
	#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
	#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC

    _dim1 = TotalNumFSRs_perLayer * NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1);
    GPUAccumAngFlux(_lay, _polarBoundLowLim, _polarBoundUpLim, _polOffset,\
        _axiFlux_avg, GPU_TSKernelType, _dim1, TotalNumFSRs_perLayer, NumGroups, NumAzimAng[0], NumPolarAng[0], NumScattOrder);
    ScalarFluxCopy_Dev2Host(ScalarFlux_MOCLayer, (size_t)(_dim1), 0, (char *)("ScalarFlux_MOCLayer-D2H"));
	// printf("\n _lay=%d, _polarBoundLowLim=%d, _polarBoundUpLim=%d",_lay,_polarBoundLowLim,_polarBoundUpLim);
	// printf("\n &&&&& ScalarFlux_MOC[ _lay*_dim1=%d]=%e", _lay*_dim1,ScalarFlux_MOC[ _lay*_dim1]);
	AccuFluxByLayer(_lay, TotalNumFSRs_subDom, TotalNumFSRs_perLayer, NumGroups, \
            NumScattOrder, ScalarFlux_MOC, ScalarFlux_MOCLayer);
	// int ii;
	// for (ii=0;ii<TotalNumFSRs_subDom* NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1);ii++) {
	// 	printf("\n ScalarFlux_MOC[%d]=%e",ii,ScalarFlux_MOC[ii]);
	// }

#ifdef TIME_REC
	#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
	#endif // defined(CUDA_USED)_
	_time_end = TallyCurrentTime_CPU();
	TimeMOC[TimeStamp_MOC_TS3] += _time_end - _time_start;
#endif // TIME_REC

#ifdef TIME_REC
	#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
	#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC

		if (GPU_TSKernelType == 12) {
			GPUUpdateAxialSurfAngFlux_1(TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0], _axiFlux_inp, _axiFlux_avg, \
			    _twoOverLayWidth, _polarBoundLowLim, _polarBoundUpLim, _polOffset);
		}
		else {
			GPUUpdateAxialSurfAngFlux(_dim, _axiFlux_inp, _axiFlux_avg);
		}
			

#ifdef TIME_REC
	#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
	#endif // defined(CUDA_USED)
	_time_end = TallyCurrentTime_CPU();
	TimeMOC[TimeStamp_MOC_TS4] += _time_end - _time_start;
#endif // TIME_REC

#endif // defined(M_3DMOCex)
	}

#if defined(M_3DMOCex)
#if defined(M_MG_CMFD)
#ifdef TIME_REC
	#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
	#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC
	GPUGetNetCurr_Axial_1(_nodeLay, _surfLay, _JzDir, _polarBoundLowLim, \
		_polarBoundUpLim, _axiFlux_inp, _polOffset);
#ifdef TIME_REC
	#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
	#endif // defined(CUDA_USED)
	_time_end = TallyCurrentTime_CPU();
	TimeMOC[TimeStamp_MOC_TS1] += _time_end - _time_start;
#endif // TIME_REC
#endif // defined(M_MG_CMFD)

#ifdef TIME_REC
	#if defined(CUDA_USED) 	
	cudaDeviceSynchronize_cpu(); 
	#endif // defined(CUDA_USED)
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC

	GPUUpdate_AxialBoundCond_1(_dir, _axiFlux_inp, _axiFlux_avg, TopBC, BottomBC, TotalNumFSRs_perLayer*NumGroups*NumAzimAng[0], NumPolarAng[0]);

#ifdef TIME_REC
	#if defined(CUDA_USED) 	 
	cudaDeviceSynchronize_cpu(); 
	#endif // defined(CUDA_USED)
	_time_end = TallyCurrentTime_CPU();
	TimeMOC[TimeStamp_MOC_TS5] += _time_end - _time_start;
#endif // TIME_REC
#endif // defined(M_3DMOCex)
}
void EigValProbConvCheck_Flux(int _procID, int _numProcs, int _isNormInf, FP_PRECISION *_normInf, int _isNorm2, \
    FP_PRECISION *_norm2, int _isRMSE, FP_PRECISION *_rmse, int *_bFNormInf,int *_bFNorm2, int *_bFRmse, \
	FP_PRECISION _tol)
{
	FP_PRECISION _dot=0.0, _dotColl = 0.0, _normInfProc[1]= {0.0};
	M_UINT32 _dim, _dim1, _dim2;
	FP_PRECISION_FLUX *_scalarFluxOld = NULL;
	_dim = (M_UINT32)(TotalNumFSRs_globDom) * (M_UINT32)(NumGroups); // * (NumScattOrder + 1) * (NumScattOrder + 1)
	_dim1 = (M_UINT32)(TotalNumFSRs_subDom) * (M_UINT32)(NumGroups);
	_dim2 = (M_UINT32)(TotalNumFSRs_subDom) * (M_UINT32)(NumGroups * (NumScattOrder + 1) * (NumScattOrder + 1));

	_bFNormInf[0] = 0;
	_bFNorm2[0] = 0;
	_bFRmse[0] = 0;

// #if defined(CUDA_USED)
// 	_dot = GPUMOCConvCheck_Pre(_isNormInf, _normInfProc, _isNorm2, _isRMSE, TotalNumFSRs_subDom * NumGroups);
// #else

#if defined(M_MG_CMFD)
    _scalarFluxOld = ScalarFluxOld_0thOrder;
#else
    _scalarFluxOld = ScalarFluxOld;
#endif // defined(M_MG_CMFD)

	// FP_PRECISION _fisRateErrMax = 0.;

	// for (int ifsr = 0; ifsr < TotalNumFSRs_subDom; ifsr++) {
	// 	FP_PRECISION _pre = 0.;
	// 	FP_PRECISION _lat = 0.;
	// 	for (int igrp = 0; igrp < NumGroups; igrp++) {
	// 		size_t idx = ifsr * NumGroups + igrp;
	// 		_pre += _scalarFluxOld[idx] * MacroXS_f[idx];
	// 		_lat += ScalarFlux_MOC[idx] * MacroXS_f[idx];
	// 	}
	// 	if (_pre == 0.) {
	// 		continue;
	// 	}
	// 	FP_PRECISION _localErr = (FP_PRECISION) fabs(_lat / _pre - 1);
	// 	if (_localErr >= _fisRateErrMax) {
	// 		_fisRateErrMax = _localErr;
	// 	}
	// }

	// printf(" \n [  ERROR  ]  Local Err:%13.5e\n", _fisRateErrMax);

    if (_isNormInf) {
		_normInfProc[0] = CPUGetAbsRelErrInfNorm_TwoVectors_FP_I(_dim1, ScalarFlux_MOC, _scalarFluxOld);
	}
	if (_isNorm2 || _isRMSE) {
#if (defined(CUDA_USED) && defined(M_2DMOC))
        _dot = GPUComputeScalarFluxRelErrDot(NumGroups, TotalNumFSRs_subDom, NumScattOrder);
#else
		_dot = CPUGetRelErrPowerSum_TwoVectors_FP_I(_dim1, ScalarFlux_MOC, _scalarFluxOld);
#endif
	}

#if defined(M_MG_CMFD)
	VectorCopy_FP_I(ScalarFluxOld_0thOrder, ScalarFlux_MOC, _dim1);
#endif // defined(M_MG_CMFD)

#if defined(M_3DMOCex) || !defined(CUDA_USED)
    //todo: swap the pointer
	VectorCopy_FP_I(ScalarFluxOld, ScalarFlux_MOC, _dim2);
    //todo: memset or other efficiency method
	VectorInit_FP_Flux(ScalarFlux_MOC, _dim2, (FP_PRECISION_FLUX)(0.0));
#endif

// #endif // defined(CUDA_USED)
	
	if (_isNormInf) {
#ifdef MPI_USED
    if (NumProcs > 1) {
		MPI_Reduce((void *)_normInfProc, (void *)_normInf,1, MPI_DataType, MPI_MAX, 0, MPI_COMM_WORLD);
	}
	else {
		_normInf[0] = _normInfProc[0];
	}
#else
    _normInf[0] = _normInfProc[0];
#endif // MPI_USED
	}

    if (_isNorm2 || _isRMSE) {
#ifdef MPI_USED
    if (NumProcs > 1) {
		MPI_Reduce((void *)&_dot, (void *)&_dotColl, 1, MPI_DataType, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	else {
		_dotColl = _dot;
	}
#else
    _dotColl = _dot;
#endif // MPI_USED
	}

    if (_procID == 0) {
		if (_isNormInf) {
			if (_normInf[0] <= _tol) {
				_bFNormInf[0] = 1;
			}
	    }

	    if (_isNorm2) {
		    _norm2[0] = sqrt(_dotColl);
			if (_norm2[0] <= _tol) {
				_bFNorm2[0] = 1;
			}
	    }

		if (_isRMSE) {
		    _rmse[0] = sqrt(_dotColl/((FP_PRECISION)(_dim)));
			if (_rmse[0] <= _tol) {
				_bFRmse[0] = 1;
			}
	    }
	}

#ifdef MPI_USED
	if (_numProcs > 1) {
		MPI_Bcast(_bFNormInf, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(_bFNorm2, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(_bFRmse, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	}
#endif // MPI_USED
}

void FixSouProbConvCheck_Flux(int _numFSRs_lay, int _procID, int _numProcs, int _numGroups, FP_PRECISION *_vecNew, FP_PRECISION *_vecOld, \
    int _layBoundLowLim, int _layBoundUpLim, int _groupBoundLowLim, int _groupBoundUpLim, FP_PRECISION *_rmse, \
	int *_bFRmse, FP_PRECISION _tol)
{
	FP_PRECISION _dim = 0.0, _dimColl = 0.0;
	_bFRmse[0] = 0;
	FP_PRECISION _dot=0.0, _dotColl = 0.0;

	_dim = (FP_PRECISION)(_numFSRs_lay *(_layBoundUpLim-_layBoundLowLim) * (_groupBoundUpLim - _groupBoundLowLim));

	_dot = ComputeRelErrPowerSum_FluxVecs_lay_g(_numFSRs_lay, _numGroups, _vecNew, _vecOld, _layBoundLowLim, \
	    _layBoundUpLim, _groupBoundLowLim, _groupBoundUpLim);
	// _dot = GPUComputeScalarFluxRelErrDot(NumGroups, TotalNumFSRs_subDom, NumScattOrder);
    //_dot = _dim * 1e4;
    FluxVecCopy_lay_g(_numFSRs_lay, _numGroups, _vecNew, _vecOld, _layBoundLowLim, \
	    _layBoundUpLim, _groupBoundLowLim, _groupBoundUpLim);
#ifdef MPI_USED
    if (NumProcs > 1) {
		MPI_Reduce((void *)&_dot, (void *)&_dotColl, 1, MPI_DataType, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce((void *)&_dim, (void *)&_dimColl, 1, MPI_DataType, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	else {
		_dotColl = _dot;
		_dimColl = _dim;
	}
#else
    _dotColl = _dot;
    _dimColl = _dim;
#endif // MPI_USED

    if (_procID == 0) {
        _rmse[0] = sqrt(_dotColl/_dimColl);
	    if (_rmse[0] <= _tol) {
	    	_bFRmse[0] = 1;
	    }
	}
#ifdef MPI_USED
	if (_numProcs > 1) {
		MPI_Bcast(_bFRmse, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
	}
#endif // MPI_USED
}
#endif

void SectorizeCell(struct PLANE *_planes, struct CELL *_cell)
{
	int k1;
	FP_PRECISION_GEO _azimAngle;
	FP_PRECISION_GEO _deltaAzim;// = 2.*ONE_PI/(_cell->num_sectors);
    /* Create each of the bounding planes for the sector Cells */
	if (_cell->num_sectors > 1) {
		_deltaAzim = 2.*ONE_PI / (_cell->num_sectors);
		for (k1 = 0; k1<_cell->num_sectors; k1++) {
			/* Figure out the angle for this plane */
			_azimAngle = k1*_deltaAzim;
			/* Instantiate the plane */
			_planes[k1].id = k1;
			_planes[k1].A = cos(_azimAngle);
			_planes[k1].B = sin(_azimAngle);
			_planes[k1].surface_type = PLANE;
			_planes[k1].boundary_type = BOUNDARY_NONE;
		}
	}
}

/**
* @ Function: MeshCellInOneDirection()
*   introduction: Subdivides the Cell into rectangular clones in one (X or Y) direction, 
                  generate the planes for constituting the rectangular subCells
* @ Parameter List: _planes - a pointer for storing the planes information of the rectangular subcells in one direction
*                   _numMesh - number of planes needed to be generated, = number of subcells in a specific direction + 1
*                   _stepLength - width of the subcells in this direction, equaly separated
*                   _directionID - direction identification, 0 - X-direction, non 0 - y-direction
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void MeshCellInOneDirection(struct PLANE *_planes, int _numMesh, FP_PRECISION_GEO _stepLength, int _directionID)
{
	int k1;
	for (k1 = 0; k1<_numMesh; k1++) {
		_planes[k1].id = k1;
		_planes[k1].C = 0.;
		_planes[k1].D = -1.*(FP_PRECISION_GEO)k1*_stepLength / (FP_PRECISION_GEO)(_numMesh - 1) + _stepLength / 2.;
		if (_directionID == 0) {
			_planes[k1].A = 1.;
		    _planes[k1].B = 0.;
			_planes[k1].surface_type = XPLANE;
		}
		else {
			_planes[k1].A = 0.;
		    _planes[k1].B = 1.;
			_planes[k1].surface_type = YPLANE;
		}
		_planes[k1].boundary_type = BOUNDARY_NONE;
	}
}

int GetNumCellsInAsse_layer(int _asseIdInCore, int _layerID)
{
	int i, j, _numCellsInAsse;
	_numCellsInAsse = 0;
	for (i = 0; i<NumAssemblies; i++) {
		if (Core->asse_in_core[_asseIdInCore] == Assemblies[i].type_id) {
			for (j = 0; j<NumLattices; j++) {
				if (Assemblies[i].latt_in_asse[_layerID] == Lattices[j].type_id) {
					_numCellsInAsse = Lattices[j].num_x*Lattices[j].num_y;
					break;
				}
			}
			break;
		}
	}
	return _numCellsInAsse;
}
int GetNumCellsInCore_layer(struct CORE *_core, int _layerIDInCore)
{
	int i, _numCellsInCore;
	_numCellsInCore = 0;
	for (i = 0; i<(_core->num_X)*(_core->num_Y); i++) {
		_numCellsInCore = _numCellsInCore + GetNumCellsInAsse_layer(i, _layerIDInCore);
	}
	return _numCellsInCore;
}
int GetNumCellsInAsse_X(int _asseIdInCore, int _lattIdInAsse)
{

	int i, j, _numCellsInAsse_InOneRow;
	_numCellsInAsse_InOneRow = 0;
	for (i = 0; i<NumAssemblies; i++) {
		if (Core->asse_in_core[_asseIdInCore] == Assemblies[i].type_id) {
			for (j = 0; j<NumLattices; j++) {
				if (Assemblies[i].latt_in_asse[_lattIdInAsse] == Lattices[j].type_id) {
					_numCellsInAsse_InOneRow = Lattices[j].num_x;
					break;
				}
			}
			break;
		}

	}
	return _numCellsInAsse_InOneRow;
}
int GetNumCellsInCore_X(struct CORE *_core, int _layerIDInCore)
{
	int i, _numCellsInCore_InOneRow;
	_numCellsInCore_InOneRow = 0;
	for (i = 0; i<(_core->num_X); i++) {
		_numCellsInCore_InOneRow += GetNumCellsInAsse_X(i, _layerIDInCore);
	}
	return _numCellsInCore_InOneRow;
}
/**
* @ Function: GetNumCellsInCore_Y()
*   introduction: get number of cells in one column (y-direction) of core for a specific layer, based on the geometry input
* @ Parameter List: _core - struct CORE object
*                   _layerIDInCore - layerID in core
*   Return: _numCellsInCore_InOneColumn - number of cells in one column of core for a specific layer
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int GetNumCellsInCore_Y(struct CORE *_core, int _layerIDInCore)
{
	int i, _numCellsInCore_InOneColumn;
	_numCellsInCore_InOneColumn = 0;
	for (i = 0; i<(_core->num_Y); i++) {
		_numCellsInCore_InOneColumn += GetNumCellsInAsse_Y(i, _layerIDInCore);
	}
	return _numCellsInCore_InOneColumn;
}

int GetNumCellsInDomain_X(int *_numSubDomains, int **_numPinsInSubDomain)
{
	int j, _counter;
	_counter = 0;
	for (j = 0; j < _numSubDomains[0]; j++) {
		_counter += _numPinsInSubDomain[0][j];
	}
	return _counter;
}
/**
* @ Function: GetNumCellsInDomain_Y()
*   introduction: get number of cells in y directin of whole computational domain (core), sumation of all subdoamins in a specific layer
* @ Parameter List: _numSubDomains - number of subdomains in x/y/z directions, respectively, user defined in input file
*                   _numPinsInSubDomain - number of pins in x/y/z directions of each subdomain, respectively, user defined in input file
*   Return: _counter - get number of cells in y directin of whole computational domain 
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int GetNumCellsInDomain_Y(int *_numSubDomains, int **_numPinsInSubDomain)
{
	int i, _counter;
	_counter = 0;
	for (i = 0; i < _numSubDomains[1]; i++) {
		_counter += _numPinsInSubDomain[1][i];
	}
	return _counter;
}
int GetPinRowIDInSubDomBaseGlobPinID(int _globPinID, int *_numSubDomains, int **_numPinsInSubDomain)
{
	int _numCellstoSubDomTop, _row_1, i, _row;
	_row = GetGlobRowIDBaseGlobPinID(_globPinID, _numSubDomains, _numPinsInSubDomain);
	_row_1 = _row;
	_numCellstoSubDomTop = 0;
	for (i = 0; i < _numSubDomains[1]; i++) {
		_row_1 -= _numPinsInSubDomain[1][i];
		if (_row_1 < 0) {
			break;
		}
		_numCellstoSubDomTop += _numPinsInSubDomain[1][i];
	}
	_row = _row - _numCellstoSubDomTop;
	return _row;
}
/**
* @ Function: GetPinColumnIDInSubDomBaseGlobPinID()
*   introduction: get pin column id (Y-direction) in subdomain (one layer) based on global pin id
* @ Parameter List: _globPinID - global pin ID in 2D
*                   _numSubDomains - number of subdomains in x/y/z directions, respectively, user defined in input file
*                   _numPinsInSubDomain - number of pins in x/y/z directions of each subdomain, respectively, user defined in input file
*   Return: pin column id in subdomain (2D)
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int GetPinColumnIDInSubDomBaseGlobPinID(int _globPinID, int *_numSubDomains, int **_numPinsInSubDomain)
{
	int _numCellstoSubDomLeft, _column_1, i, _column;
	_column = GetGlobColumnIDBaseGlobPinID(_globPinID, _numSubDomains, _numPinsInSubDomain);
	_column_1 = _column;
	_numCellstoSubDomLeft = 0;
	for (i = 0; i < _numSubDomains[0]; i++) {
		_column_1 -= _numPinsInSubDomain[0][i];
		if (_column_1 < 0) {
			break;
		}
		_numCellstoSubDomLeft += _numPinsInSubDomain[0][i];
	}
	_column = _column - _numCellstoSubDomLeft;
	return (_column);
}
int GetSubDomIDBaseGlobPinID_Y(int _globPinID, int *_numSubDomains, int **_numPinsInSubDomain)
{
	int _subDomID_Y, _row, i;
	_row = GetGlobRowIDBaseGlobPinID(_globPinID, _numSubDomains, _numPinsInSubDomain);
	for (i = 0; i < _numSubDomains[1]; i++) {
		_row -= _numPinsInSubDomain[1][i];
		if (_row < 0) {
			break;
		}
	}
	_subDomID_Y = i;
	return _subDomID_Y;
}

/**
* @ Function: GetSubDomIDBaseGlobPinID_X()
*   introduction: get subdomain column id based on global pin id
* @ Parameter List: _globPinID - global pin ID
*                   _numSubDomains - number of subdomains in x/y/z directions, respectively, user defined in input file
*                   _numPinsInSubDomain - number of pins in x/y/z directions of each subdomain, respectively, user defined in input file
*   Return: subDomain row id in parallel 2D subDomain IDs map
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
int GetSubDomIDBaseGlobPinID_X(int _globPinID, int *_numSubDomains, int **_numPinsInSubDomain)
{
	int _subDomID_X, _column, i;
	_column = GetGlobColumnIDBaseGlobPinID(_globPinID, _numSubDomains, _numPinsInSubDomain);
	for (i = 0; i < _numSubDomains[0]; i++) {
		_column -= _numPinsInSubDomain[0][i];
		if (_column < 0) {
			break;
		}
	}
	_subDomID_X = i;
	return _subDomID_X;
}

void MOC_PrintLogFile(M_INT32 _procID)
{
	if (_procID == 0) {
		char _fileName[512] = { "" }, _name1[256] = { "" }, _name2[128] = { "" }, _name_time[64] = { "" };
		time_t _curr_time;
		struct tm *_p_time;
		M_INT32 _time_int;
#if defined(MPI_USED)
		int i;
#endif

#if defined(CUDA_USED)
		sprintf(_name1, "%s%s", _name1, "_g");
#else
		sprintf(_name1, "_c");
#endif //defined(CUDA_USED)

#if defined(FP64)
		sprintf(_name1, "%s%s", _name1, "64");
#endif //defined(FP64)

#if defined(FPMIX)
		sprintf(_name1, "%s%s", _name1, "Mix");
#endif //defined(FPMIX)

#if defined(FP32)
		sprintf(_name1, "%s%s", _name1, "32");
#endif //defined(FP32)

#if defined(M_MG_CMFD)

		if (MGCMFD_LinearSolver == 0) {
			sprintf(_name2, "%s%s", _name2, "_jacobi");
		}
		else if (MGCMFD_LinearSolver == 1) {
			sprintf(_name2, "%s%s", _name2, "_sor");
		}
		else if (MGCMFD_LinearSolver == 2) {
			if (GmresPrecond_MG == 1) {
				sprintf(_name2, "%s%s%d", _name2, "_pgmres", GmresKrylovDim_MG);
			}
			else {
				sprintf(_name2, "%s%s%d", _name2, "_gmres", GmresKrylovDim_MG);
			}
		}
		else if (MGCMFD_LinearSolver == 3) {
			sprintf(_name2, "%s%s", _name2, "_zyq");
		}
		else {
			printf("\n [**ERROR**]  MGCMFD_LinearSolver is %d, please reset!\n", MGCMFD_LinearSolver);
			fflush(stdout);
			exit(69);
		}

#if defined(M_CG_CMFD)
		sprintf(_name1, "%s%s", _name1, "_2L");
		if (CGCMFD_LinearSolver == 0) {
			sprintf(_name2, "%s%s", _name2, "+jacobi");
		}
		else if (CGCMFD_LinearSolver == 1) {
			sprintf(_name2, "%s%s", _name2, "+sor");
		}
		else if (CGCMFD_LinearSolver == 2) {
			if (GmresPrecond_CG == 1) {
				sprintf(_name2, "%s%s%d", _name2, "+pgmres", GmresKrylovDim_CG);
			}
			else {
				sprintf(_name2, "%s%s%d", _name2, "+gmres", GmresKrylovDim_CG);
			}
		}
#else
		sprintf(_name1, "%s%s", _name1, "_1L");
#endif //defined(M_CG_CMFD)
		sprintf(_name1, "%s%s", _name1, _name2);
#else 
		sprintf(_name1, "%s%s", _name1, "_moc");
#endif //  defined(M_MG_CMFD)


#if defined(CUDA_USED)
		if (GPU_TSKernelType == 0) {
			sprintf(_name1, "%s%s", _name1, "_eI");
		}
		else if (GPU_TSKernelType == 1) {
			sprintf(_name1, "%s%s", _name1, "_eF");
		}
		else if (GPU_TSKernelType == 2) {
			sprintf(_name1, "%s%s", _name1, "_ddEI");
		}
		else if (GPU_TSKernelType == 3) {
			sprintf(_name1, "%s%s", _name1, "_ddEF");
		}
		else {}
#else
		if (CPU_TSFunctionType == 0) {
			sprintf(_name1, "%s%s", _name1, "_eI");
		}
		else if (CPU_TSFunctionType == 1) {
			sprintf(_name1, "%s%s", _name1, "_ddEI");
		}
		else if (CPU_TSFunctionType == 2) {
			sprintf(_name1, "%s%s", _name1, "_eF");
		}
		else if (CPU_TSFunctionType == 3) {
			sprintf(_name1, "%s%s", _name1, "_ddEF");
		}
		else {}
#endif //defined(CUDA_USED)

        sprintf(_name1, "%s_%.2lf_a%d_p%d_k%.0e_f%.0e", _name1, (double)(RaySpacingIni[0]),NumAzimAng[0],NumPolarAng[0],ConvCritK,ConvCritFlux);

		sprintf(CaseId, "%s%s", CaseId, _name1);
		time(&_curr_time);
		_p_time = gmtime(&_curr_time);

		_time_int = 1900 + _p_time->tm_year;
		sprintf(_name_time, "%s%d", _name_time, _time_int);
		_time_int = _p_time->tm_mon + 1;
		if (_time_int < 10) {
			sprintf(_name_time, "%s%s%d", _name_time, "0", _time_int);
		}
		else {
			sprintf(_name_time, "%s%d", _name_time, _time_int);
		}

		_time_int = _p_time->tm_mday;
		if (_time_int < 10) {
			sprintf(_name_time, "%s%s%d", _name_time, "0", _time_int);
		}
		else {
			sprintf(_name_time, "%s%d", _name_time, _time_int);
		}
		_time_int = _p_time->tm_hour + 8;
		if (_time_int < 10) {
			sprintf(_name_time, "%s%s%d", _name_time, "_0", _time_int);
		}
		else {
			sprintf(_name_time, "%s%s%d", _name_time, "_", _time_int);
		}
		_time_int = _p_time->tm_min;
		if (_time_int < 10) {
			sprintf(_name_time, "%s%s%d", _name_time, "0", _time_int);
		}
		else {
			sprintf(_name_time, "%s%s%d", _name_time, "", _time_int);
		}
		_time_int = _p_time->tm_sec;
		if (_time_int < 10) {
			sprintf(_name_time, "%s%s%d", _name_time, "0", _time_int);
		}
		else {
			sprintf(_name_time, "%s%s%d", _name_time, "", _time_int);
		}
		sprintf(CaseId, "%s-%s", CaseId, _name_time);
		sprintf(_fileName, "%s%s", CaseId, "-log.out");
		LogFile = fopen(_fileName, "w");
		if (LogFile == NULL) {
			printf("\n [**ERROR**]  %s file opens error in process %d!\n", _fileName, _procID);
			fflush(stdout);
			exit(69);
		}

		fprintf(LogFile, "\n  *****************************************************************");
		fprintf(LogFile, "\n  *                                                               *");
		fprintf(LogFile, "\n  *                           Log  File                           *");
		fprintf(LogFile, "\n  *                                                               *");
		fprintf(LogFile, "\n  *****************************************************************\n");
#if defined(MPI_USED)
		for (i=0;i<NumProcs;i++) {
			if (i==0) {
				fprintf(LogFile,"\n [  INFO   ]  Process ID: %3d�?Processor Name: %6s, CPU ID:%3d", i, ProcessorNameColl[i], CurrCPUIDColl[i]);
			}
			else {
				fprintf(LogFile,"\n              Process ID: %3d�?Processor Name: %6s, CPU ID:%3d", i, ProcessorNameColl[i], CurrCPUIDColl[i]);
			}
		}
	    fflush(stdout);
#endif
		fprintf(LogFile, "\n [  INFO   ]  Input File Name:   %s", IP);
		fprintf(LogFile, "\n [  INFO   ]  CASE ID:           %s", CaseId);
#ifdef DOUBLE
		fprintf(LogFile, "\n [  INFO   ]  The Floating-Point Precision is DOUBLE");
#endif
#ifdef SINGLE
		fprintf(LogFile, "\n [  INFO   ]  The Floating-Point Precision is SINGLE");
#endif
		fprintf(LogFile, "\n [  INFO   ]  Setting Macroscopic Cross Section Data ...");
		fprintf(LogFile, "\n [  INFO   ]  Creating Geometry ...");

	}

#ifdef MPI_USED
	MPI_Bcast(CaseId, 128, MPI_CHAR, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
#endif // MPI_USED

}

void GetIndexOfTracksInRTM_SumAzim(M_INT32 _numTrackSet, M_INT32 *_numAzimAng, M_INT32 **_numTrack, \
	M_INT32 **_IndexTrackInRTM_SumAzim_2PI)
{
	M_INT32 i, j, _azimDim;
	_azimDim = 0;
	for (i = 0; i < _numTrackSet; i++) {
		_azimDim = _numAzimAng[i];
		for (j = 0; j < _azimDim; j++) {
			if (j == 0) {
				_IndexTrackInRTM_SumAzim_2PI[i][j] = 0;
			}
			else {
				_IndexTrackInRTM_SumAzim_2PI[i][j] = _IndexTrackInRTM_SumAzim_2PI[i][j - 1] + _numTrack[i][j - 1];
			}
		}
	}
}

void GetSumNumSegsInRTMTrack_2PI(M_INT32 _numTrackSet, M_INT32 *_TotalNumTracksInRTM_Total, \
    M_INT32 **_NumSegsInTrack_2PI, M_INT32 **_SumNumSegsInRTMTrack_2PI)
{
	M_INT32 i, j, _totTr;
	for (i = 0; i < _numTrackSet; i++) {
		_totTr=_TotalNumTracksInRTM_Total[0];
		for (j=0;j<_totTr;j++) {
			if (j == 0) {
				_SumNumSegsInRTMTrack_2PI[i][j]=0;
			}
			else {
				_SumNumSegsInRTMTrack_2PI[i][j]=_SumNumSegsInRTMTrack_2PI[i][j-1] + _NumSegsInTrack_2PI[i][j-1];
			}
		}
	}
}


M_INT32 GetTotalNumFSRs_SubDom_TrackFile(M_INT32 *_NumFSRsInRTM, M_INT32 _NumPinsInSubDom,
    M_INT32 *_CellRTMType_SubDom)
{
    M_INT32 i, total = 0;
    for (i = 0; i < _NumPinsInSubDom; ++i) {
        total += _NumFSRsInRTM[_CellRTMType_SubDom[i]];
    }
    return total;
}

void SetNumFSRsInCell_trackFile(M_INT32 *_NumFSRsInCell, M_INT32 _MOC_NumPinsInSubDom,
    M_INT32 *_NumFSRsInRTM, M_INT32 *_CellRTMType_SubDom)
{
    M_INT32 i;
    for (i = 0; i < _MOC_NumPinsInSubDom; ++i) {
        _NumFSRsInCell[i] = _NumFSRsInRTM[_CellRTMType_SubDom[i]];
    }
}

void SetFSRArea_trackFile(M_INT32 _MOC_NumPinsInSubDom, M_INT32 *_NumFSRsInRTM,
    M_INT32 *_SumNumFSRsInRTM, M_INT32 *_CellRTMType_SubDom,
    FP_PRECISION_GEO *_FSRsAreaInRTM, FP_PRECISION_GEO *_FSRArea_SubDom,
    M_INT32 *_SumNumFSRsBasedGlobCellID)
{
    M_INT32 i, j;
    for (i = 0; i < _MOC_NumPinsInSubDom; ++i) {
        M_INT32 rtm = _CellRTMType_SubDom[i];
        M_INT32 fsrBase = _SumNumFSRsInRTM[rtm];
        M_INT32 fsr = _SumNumFSRsBasedGlobCellID[i];
        for (j = 0; j < _NumFSRsInRTM[rtm]; ++j) {
            _FSRArea_SubDom[fsr + j] = _FSRsAreaInRTM[fsrBase + j];
        }
    }
}

int GetNumCellsInAsse_Y(int _asseIdInCore, int _lattIdInAsse)
{
    int i, j, numCells = 0;
    for (i = 0; i < NumAssemblies; ++i) {
        if (Core->asse_in_core[_asseIdInCore] == Assemblies[i].type_id) {
            for (j = 0; j < NumLattices; ++j) {
                if (Assemblies[i].latt_in_asse[_lattIdInAsse] == Lattices[j].type_id) {
                    numCells = Lattices[j].num_y;
                    break;
                }
            }
            break;
        }
    }
    return numCells;
}

int GetGlobRowIDBaseGlobPinID(int _globPinID, int *_numSubDomains,
    int **_numPinsInSubDomain)
{
    int numPinsCoreX = GetNumCellsInDomain_X(_numSubDomains, _numPinsInSubDomain);
    return _globPinID / numPinsCoreX;
}

int GetGlobColumnIDBaseGlobPinID(int _globPinID, int *_numSubDomains,
    int **_numPinsInSubDomain)
{
    int numPinsCoreX = GetNumCellsInDomain_X(_numSubDomains, _numPinsInSubDomain);
    return _globPinID % numPinsCoreX;
}

void OutputBasicTrackSetInfo(FILE *_op,int _numTrackSet, int *_numAzimAng, int *_numPolarAng,int **_numTracks,\
    int **_numTracksX,int **_numTracksY,FP_PRECISION_GEO **_dxEff,FP_PRECISION_GEO **_dyEff,FP_PRECISION_GEO **_effAzimAng,\
	FP_PRECISION_GEO **_effRaySpacing,FP_PRECISION **_azimWeights,FP_PRECISION **_totalWeights,\
	struct POINT ***_RTMStartP, struct POINT ***_RTMEndP)
{
	// Write the information of basec track set
	if (Output_GeometryInfo==1) {
		int i, j,k/*, _mat, _cell*/;
		int _width1 = 9, _width2 = 15;
		FP_PRECISION _dx, _dy;
		struct CELLS_IN_PINCELL *_iter = NULL;
		_dx = 0.0;
		_dy = 0.0;
		fprintf(_op, "\n  The Number of Track Set is %d",_numTrackSet);
		for (i=0;i<_numTrackSet;i++) {
            fprintf(_op, "\n  ++++++++++++++++++++++++++++++++++++++++++++++++");
		    fprintf(_op, "\n  track Information for track Set %d",i);
			fprintf(_op, "\n%7s%13s%13s%9s%9s%12s%12s%12s%12s%12s   %-27s", \
			    "aziID","aziAng(rad)", "aziAng(deg)", "numTraX", \
			    "numTraY", "numTraTot", "TraSpaX", "TraSpaY", \
			    "effRaySpa", "effAziWei","totWei(polar=0~max)");
			for (j=0;j<_numAzimAng[i];j++) {
				if (InputTrackFile == 0) {
					_dx = _dxEff[i][j];
					_dy = _dyEff[i][j];
				}

				fprintf(_op, "\n%7d%13.3e%13.3e%9d%9d%12d%12.3e%12.3e%12.3e%12.3e", \
					j, \
					_effAzimAng[i][j], \
					_effAzimAng[i][j] / ONE_PI*180., \
					_numTracksX[i][j], \
					_numTracksY[i][j], \
					_numTracks[i][j], \
					_dx, \
					_dy, \
					_effRaySpacing[i][j], \
					0.);
					// _azimWeights[i][j]);
				// for (k = 0; k < _numPolarAng[i]; k++) {
				//     fprintf(_op, "   %-10.3e", _totalWeights[i][j*_numPolarAng[i]+k]);
			    // }
			}

			if (0 && InputTrackFile == 0) {
				fprintf(_op, "\n  ++++++++++++++++++++++++++++++++++++++++++++++++");
				fprintf(_op, "\n  start and end points of tracks for track Set %d", i);
				for (j = 0; j < _numAzimAng[i]; j++) {
					fprintf(_op, "\n%*s%*s%*s%*s%*s%*s", _width1, "aziID", _width1, "trackID", \
						_width2, "startP_X", _width2, "startP_Y", _width2, "endP_X", _width2, "endP_Y");
					for (k = 0; k < _numTracks[i][j]; k++) {
						fprintf(_op, "\n%*d%*d%*.6e%*.6e%*.6e%*.6e", _width1, j, _width1, k, \
							_width2, _RTMStartP[i][j][k].xyz[0], _width2, _RTMStartP[i][j][k].xyz[1], \
							_width2, _RTMEndP[i][j][k].xyz[0], _width2, _RTMEndP[i][j][k].xyz[1]);
					}
				}
			}
		}
	}
}

void OutputSegInfoInLongTrack(FILE *_op, int _numLongtrack_halfSpace, int *_LTsIndex_AzimAngID, int *_LTsIndex_LTID, \
    struct SUBDOM_LONGTRACK_INFO *_longTracksInfo, int *_trackIndexInLTs, struct SUBDOM_TRACK_IN_LONGTRACK *_tracksInfoInLT, \
	int _numCellsInDom_Layer, int _totNumTracksInOneRTM_2PI, int *_cellRTMGeoType, int *_indexTrackInRTM_SumAzim_2PI, \
	FP_PRECISION *_segInfoInTrack_2PI_length, int *_segInfoInTrack_2PI_AntiClockFSR, int *_sumNumSegsInRTMTrack_2PI, \
	int *_sumNumFSRsBasedGlobCellID, FP_PRECISION_GEO *_FSRsAreaInRTM_AntiClockFSR, int *_sumNumFSRsInRTM,int *_numSegsInTrack)
{
	if (Output_GeometryInfo) {
		int _ltIter,_azim,_lt,_azimNext,_ltIDNext,_track,_cell,_ray,_globCellID,_lay,_trackIndex,_reg,_seg,_fsrID, _segCount;
		int _width_int=12, _width_int1=8, _width_fp=16;
		FP_PRECISION _fsrArea,_length;

        fprintf(_op,"\n  *****************************************************************");
        fprintf(_op,"\n  *                   Segment Info In LongTrack                   *");
        fprintf(_op,"\n  *****************************************************************\n");

		_lay=0;
		//printf("\n _numLongtrack_halfSpace=%d", _numLongtrack_halfSpace);
		//exit(90);
		for (_ltIter=0;_ltIter<_numLongtrack_halfSpace*2;_ltIter++) {
			_azim = _LTsIndex_AzimAngID[_ltIter];
			_lt = _LTsIndex_LTID[_ltIter];
			_azimNext = _longTracksInfo[_ltIter].next_azim_angle_id;
			_ltIDNext = _longTracksInfo[_ltIter].next_LongTrack_id_SubDom;
			fprintf(_op, "\n  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
			fprintf(_op, "\n  _ltID=%d, _azimID=%d, _azimNext=%d, _ltNext=%d", _lt, _azim, _azimNext, _ltIDNext);

			_segCount=0;
			fprintf(_op, "\n  %*s%*s%*s%*s", _width_int1, "_segID", _width_int1, "_fsrID", \
			    _width_fp, "_length", _width_fp, "_fsrArea");
			for (_track = _trackIndexInLTs[_ltIter]; _track < _trackIndexInLTs[_ltIter + 1]; _track++) {
				_cell = _tracksInfoInLT[_track].cell_ID;
				_ray = _tracksInfoInLT[_track].track_ID;
				_globCellID = _lay * _numCellsInDom_Layer + _cell;
				_trackIndex = _totNumTracksInOneRTM_2PI * _cellRTMGeoType[_globCellID] + _indexTrackInRTM_SumAzim_2PI[_azim] + _ray;
				for (_seg = 0; _seg < _numSegsInTrack[_trackIndex]; _seg++) {
					_length = _segInfoInTrack_2PI_length[_sumNumSegsInRTMTrack_2PI[_trackIndex] + _seg];
					_reg = _segInfoInTrack_2PI_AntiClockFSR[_sumNumSegsInRTMTrack_2PI[_trackIndex] + _seg];
					_fsrID = _sumNumFSRsBasedGlobCellID[_globCellID] + _reg;
					//_fsrArea = _FSRsAreaInRTM_AntiClockFSR[_sumNumFSRsInRTM[_cellRTMGeoType[_globCellID]] + _reg];
					_fsrArea = FSRArea_SubDom[_fsrID];
					fprintf(_op, "\n  %*d%*d%*.6e%*.6e", _width_int1, _segCount, _width_int1, _fsrID, \
			            _width_fp, _length, _width_fp, _fsrArea);
					_segCount++;
				}
			}
		}
	}
}

