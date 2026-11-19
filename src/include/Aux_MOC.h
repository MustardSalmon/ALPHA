
void Aux_GPU_MOC_EigenValueProblem();
void Aux_GPUTransportSweep_1(int _iter);
void Aux_GPUMOC_TransSweep_UporDown(int _dim, int _layStart, int _layAdd, int _surfLayInc, \
	int _polarBoundLowLim, int _polarBoundUpLim, int _polOffset, FP_PRECISION *_axiFlux_inp, \
	FP_PRECISION *_axiFlux_avg, int _nodeLay, int _surfLay, FP_PRECISION _JzDir, int _dir,\
	int _iter, int _fisrtInvoke, int _innerIter);
void jimoxiuzheng(double buck_inp);