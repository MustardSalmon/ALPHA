/********************************************************************************/
/* @ File: MOC_Quadrature_Set.c
/*   introduction: functions related to quadrature operation
/* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
/*   1st modified:
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "MOC_AllocAndFreeMem.h"
#include "MOC_CommonMacro.h"
#include "MOC_GPUCPUInterface.h"
#include "MOC_Input.h"
#include "MOC_Output.h"
#include "MOC_Quadrature_Set.h"
#include "MOC_ExpEvaluation.h"
#include "MOC_Macro_XSs.h"

/** An array of the azimuthal angle quadrature weights */
    FP_PRECISION **AzimWeights = NULL, **PolarWeights = NULL;
	FP_PRECISION **TotalWeights = NULL, **TotalWeights_2DMOC = NULL, **SinTheta = NULL, **CosTheta = NULL, *OneOverSinTheta = NULL;
	FP_PRECISION *SinThetaForExp = NULL;
	FP_PRECISION **SpheHarmCoeff = NULL;
	FP_PRECISION *CoeffA_HOScatt = NULL;
	FP_DOUBLE **Theta = NULL;

void SetQuadratureInfo(int _numTrackSet,int *_numAzimAng, int *_numPolarAng, int _scattOrder)
{
	/** set quadrature info */
	//int i;
#ifdef TIME_REC
	FP_PRECISION_TIME _time_start = 0., _time_end = 0.;
	_time_start = TallyCurrentTime_CPU();
#endif // TIME_REC
	
	SinTheta = AllocMem_FP_IP(_numTrackSet, _numPolarAng, (char *)("SinTheta"));
	OneOverSinTheta = AllocMem_FP_I_1(_numPolarAng[0], (char *)("OneOverSinTheta"));
	CosTheta = AllocMem_FP_IP(_numTrackSet, _numPolarAng, (char *)("CosTheta"));
	Theta = AllocMem_FPGeo_IP(_numTrackSet, _numPolarAng, (char *)("Theta"));
	PolarWeights = AllocMem_FP_IP(_numTrackSet, _numPolarAng, (char *)("PolarWeights"));
	AzimWeights = AllocMem_FP_IP(_numTrackSet, _numAzimAng, (char *)("AzimWeights"));
	SetPolarQuadrature(_numTrackSet, PolarQuadType, NumPolarAng, SinTheta, PolarWeights,CosTheta);

// int i;
// for (i=0;i<NumPolarAng[0];i++) {
// 	printf("\n PolarWeights[%d]=%.10lf, SinTheta=%.10lf, CosTheta=%.10lf",i, PolarWeights[0][i],SinTheta[0][i],CosTheta[0][i]);
// }
// GLPolarQuad_Calc(SinTheta[0], PolarWeights[0], NumPolarAng[0]);
// // int i;
// for (i=0;i<NumPolarAng[0];i++) {
// 	printf("\n PolarWeights[%d]=%.10lf, SinTheta=%.10lf",i, PolarWeights[0][i],SinTheta[0][i]);
// }
// exit(12);
	SetAzimuthalQuadrature(_numTrackSet, 0,NumAzimAng, EffAzimAng, AzimWeights);
	TotalWeights = TotalWeight_AllocMem(_numTrackSet, NumAzimAng, NumPolarAng, (char *)("TotalWeights"));
	TotalWeights_2DMOC = TotalWeight_AllocMem(_numTrackSet, NumAzimAng, NumPolarAng, (char *)("TotalWeights_2DMOC"));
	SetTotalWeight(_numTrackSet,_numAzimAng,NumPolarAng,SinTheta,PolarWeights,AzimWeights,EffRaySpacing, TotalWeights, TotalWeights_2DMOC);
	SetSinThetaForExpTabel(NumPolarAng[0], SinTheta[0]);
	GC_getSinThetasTimes2(SinTheta);

	SetTheta(_numTrackSet, NumPolarAng, SinTheta, Theta);
	CoeffEval_HOScatt(_scattOrder);
	SpheHarmCoeff = AllocMem_FP_SpheHarmCoeff(_numTrackSet, _numAzimAng, _numPolarAng, _scattOrder, (char *)("SpheHarmCoeff"));
	SetSpheHarmCoeff(_numTrackSet, _numAzimAng, _numPolarAng, _scattOrder, EffAzimAng, Theta, SpheHarmCoeff);
#ifdef TIME_REC
	_time_end = TallyCurrentTime_CPU();
	TimePrePart[TimeCountPrePart] = _time_end - _time_start;
	strcpy(TimeMetricsPrePart[TimeCountPrePart], "Set Quadrature Info");
	TimeCountPrePart++;
#endif // TIME_REC
}

void FreeQuadratureInfo(int _numTrackSet)
{
	SinTheta = FreeMem_FP_IP(SinTheta, _numTrackSet, (char *)("SinTheta"));
	OneOverSinTheta = FreeMem_FP_I(OneOverSinTheta, (char *)("OneOverSinTheta"));
	CosTheta = FreeMem_FP_IP(CosTheta, _numTrackSet, (char *)("CosTheta"));
	PolarWeights = FreeMem_FP_IP(PolarWeights, _numTrackSet, (char *)("PolarWeights"));
	AzimWeights = FreeMem_FP_IP(AzimWeights, _numTrackSet, (char *)("AzimWeights"));
	TotalWeights = FreeMem_FP_IP(TotalWeights, _numTrackSet, (char *)("TotalWeights"));
	TotalWeights_2DMOC = FreeMem_FP_IP(TotalWeights_2DMOC, _numTrackSet, (char *)("TotalWeights_2DMOC"));
	SinThetaForExp = FreeMem_FP_I(SinThetaForExp, "SinThetaForExp");
}
 /**
 * @ Function: SetPolarQuadrature()
 *   introduction: generate polar quadrature set
 * @ Parameter List: _numTrackSet - number of track set
 *                   _quadType - quadrature type, 0-TYPolarQuad, 1-LeonardPolarQuad, 
 *                               2-GLPolarQuad, 3-EqualWeightPolarQuad, 4-EqualAnglePolarQuad
 *                   _numPolarAng - number of polar angles
 *   Return: void
 * @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
void SetPolarQuadrature(int _numTrackSet, int *_quadType, int *_numPolarAng, FP_PRECISION **_sinTheta, \
    FP_PRECISION **_polarWeights,FP_PRECISION **_cosTheta)
{
	int i, _np = 0;
	//check
// #if defined(M_2DMOC)
	for (i=0;i<_numTrackSet;i++) {
		if (IsSetHalfPolar == 1) {
			_np = _numPolarAng[i];
		}
		else {
			_np = _numPolarAng[i]/2;
		}
		
        if (_quadType[i] == 0 || _quadType[i] == 1 || _quadType[i] == 2 || _quadType[i] == 3 || _quadType[i] == 4) {}
	    else {
	    	printf("\n [ WARNING ]  Unable to set the polar quadrature with type_id %d (0~4) for TrackSetID=%d", _quadType[i],i);
	    	_quadType[i] = 0;
	    	printf("\n [ WARNING ]  The type of polar quadrature is automatically set to 0");
	    	fflush(stdout);
	    }

	    if (_quadType[i] == 0) {
		   if (_np == 1 || _np == 2 || _np == 3) {}
	    	else {
		    	printf("\n [**ERROR**]  Unable to set the number of polar angles to %d for TYPolarQuad in [0, PI/2] (1, 2 or 3 angles, for TrackSetID=%d)", _np,i);
			    _np = 3;
			    printf("\n [ WARNING ]  The number of polar angles in [0, PI/2] is automatically set to 3 for TYPolarQuad");
			    fflush(stdout);
		    }
	    }
	    else if (_quadType[i] == 1) {
	    	if (_np == 2 || _np == 3) {}
	    	else {
		    	printf("\n [**ERROR**]  Unable to set the number of polar angles to %d for LeonardPolarQuad in [0,PI/2] (2 or 3 angles), for TrackSetID=%d", _np,i);
		    	_np = 3;
		    	printf("\n [ WARNING ]  The number of polar angles in [0, PI/2] is automatically set to 3 for LeonardPolarQuad");
		    }
	    }
	    else if (_quadType[i] == 2) {
	    	if ((_np > 0 && _np <= 10) || _np==12 || _np==14 || _np==16 || _np==32) {}
		    else {
			    printf("\n [**ERROR**]  Unable to set the number of polar angles to %d for GLPolarQuad in [0,PI/2] (1~10, 12, 14, 16, 32), for TrackSetID=%d", _np,i);
			    _np = 4;
			    printf("\n [ WARNING ]  The number of polar angles in [0, PI/2] is automatically set to 4 for GLPolarQuad");
			    fflush(stdout);
		    }
	    }
    	else if (_quadType[i] == 3) {
	    	if (_np > 0) {}
		    else {
		    	printf("\n [**ERROR**]  Unable to set the number of polar angles to %d for EqualWeightPolarQuad in [0,PI/2] (positive interger), for TrackSetID=%d", _np,i);
			    _np = 8;
		    	printf("\n [ WARNING ]  The number of polar angles in [0, PI/2] is automatically set to 8 for EqualWeightPolarQuad");
			    fflush(stdout);
		    }
	    }
	    else if (_quadType[i] == 4) {
		    if (_np > 0) {}
	    	else {
		    	printf("\n [**ERROR**]  Unable to set the number of polar angles to %d for EqualAnglePolarQuad in [0,PI/2] (positive interger), for TrackSetID=%d", _np,i);
			    _np = 8;
			    printf("\n [ WARNING ]  The number of polar angles in [0, PI/2] is automatically set to 8 for EqualAnglePolarQuad");
			    fflush(stdout);
		    }
	    }
	    else {}
	}
// #endif // defined(M_2DMOC)
	
	//set quatrature
	for (i=0;i<_numTrackSet;i++) {
    	if (_quadType[i] == 0) {
		    TYPolarQuad(_sinTheta[i], _polarWeights[i], _numPolarAng[i], _cosTheta[i]);
	    }
	    else if (_quadType[i] == 1) {
	    	LeonardPolarQuad(_sinTheta[i], _polarWeights[i], _numPolarAng[i], _cosTheta[i]);
    	}
	    else if (_quadType[i] == 2) {
	    	GLPolarQuad(_sinTheta[i], _polarWeights[i], _numPolarAng[i], _cosTheta[i]);
    	}
    	else if (_quadType[i] == 3) {
		    EqualWeightPolarQuad(_sinTheta[i], _polarWeights[i], _numPolarAng[i]);
    	}
	    else if (_quadType[i] == 4) {
	    	EqualAnglePolarQuad(_sinTheta[i], _polarWeights[i], _numPolarAng[i]);
	    }
    	else {}
	}
}

//This routine uses the tabulated values for the Tabuchi-Yamamoto polar angle quadrature, including the sine thetas and weights.
//the number of polar angles (1, 2 or 3)
// theta is the polar direction measured from z-axis
void TYPolarQuad(FP_PRECISION *_sinTheta, FP_PRECISION *_polarWeights, \
    int _numPolarAng, FP_PRECISION *_cosTheta)
{
	int i, _polAng=0;
	FP_PRECISION _scale=(FP_PRECISION)(1.0);

    if (IsSetHalfPolar == 1) {
	    _polAng=_numPolarAng;
	    _scale=(FP_PRECISION)(1.0);
	}
	else {
        _polAng=_numPolarAng/2;
	    _scale=(FP_PRECISION)(0.5);
	}

	if (_polAng == 1) {
		_sinTheta[0] = (FP_PRECISION)(0.798184);
		_polarWeights[0] = 1.0 * _scale;
	}													

	else if (_polAng == 2) {
		_sinTheta[0] = (FP_PRECISION)(0.363900);
		_sinTheta[1] = (FP_PRECISION)(0.899900);
		_polarWeights[0] = 0.212854 * _scale;
		_polarWeights[1] = 0.787146 * _scale;
	}

	else if (_polAng == 3) {
		_sinTheta[0] = (FP_PRECISION)(0.166648);
		_sinTheta[1] = (FP_PRECISION)(0.537707);
		_sinTheta[2] = (FP_PRECISION)(0.932954);
		_polarWeights[0] = 0.046233 * _scale;
		_polarWeights[1] = 0.283619 * _scale;
		_polarWeights[2] = 0.670148 * _scale;
	}
	else {}
// printf("\n _polAng=%d, _numPolarAng=%d",_polAng,_numPolarAng);
// #if defined(M_3DMOCex)
    for (i=_polAng;i<_numPolarAng;i++) {
		_sinTheta[i] = _sinTheta[_numPolarAng-1-i];
		_polarWeights[i] = _polarWeights[_numPolarAng-1-i];
	}
// #endif // defined(M_3DMOCex)
    for (i=0;i<_numPolarAng;i++) {
		_cosTheta[i] = sqrt((FP_PRECISION)(1.0) - pow(_sinTheta[i], (FP_PRECISION)(2.0)));
	}
}
//This routine uses the tabulated values for the Leonard polar angle quadrature, including the sine thetas and weights.
//the number of polar angles (2 or 3)
// optimal angle
void LeonardPolarQuad(FP_PRECISION *_sinTheta, FP_PRECISION *_polarWeights, int _numPolarAng, FP_PRECISION *_cosTheta)
{
	int i, _polAng=0;
	FP_PRECISION _scale = (FP_PRECISION)(1.0);

    if (IsSetHalfPolar == 1) {
	    _polAng=_numPolarAng;
	    _scale=(FP_PRECISION)(1.0);
	}
	else {
        _polAng=_numPolarAng/2;
	    _scale=(FP_PRECISION)(0.5);
	}

	if (_polAng == 2) {
		_sinTheta[0] = (FP_PRECISION)(0.273658);
		_sinTheta[1] = (FP_PRECISION)(0.865714);
		_polarWeights[0] = 0.139473 * _scale;
		_polarWeights[1] = 0.860527 * _scale;
	}

	else if (_polAng == 3) {
		_sinTheta[0] = (FP_PRECISION)(0.099812);
		_sinTheta[1] = (FP_PRECISION)(0.395534);
		_sinTheta[2] = (FP_PRECISION)(0.891439);
		_polarWeights[0] = 0.017620 * _scale;
		_polarWeights[1] = 0.188561 * _scale;
		_polarWeights[2] = 0.793819 * _scale;
	}
	else {}
// #if defined(M_3DMOCex)
	for (i = _polAng; i < _numPolarAng; i++) {
		_sinTheta[i] = _sinTheta[_numPolarAng - 1 - i];
		_polarWeights[i] = _polarWeights[_numPolarAng - 1 - i];
	}
// #endif // defined(M_3DMOCex)
	for (i = 0; i < _numPolarAng; i++) {
		_cosTheta[i] = sqrt((FP_PRECISION)(1.0) - pow(_sinTheta[i], (FP_PRECISION)(2.0)));
	}
}

//This routine uses the tabulated values for the Gauss-Legendre polar angle quadrature, including the sine thetas and weights.
void GLPolarQuad(FP_PRECISION *_sinTheta, FP_PRECISION *_polarWeights, int _numPolarAng, FP_PRECISION *_cosTheta)
{
	int i, _polAng=0;
	FP_PRECISION _scale=(FP_PRECISION)(1.0);

    if (IsSetHalfPolar == 1) {
	    _polAng=_numPolarAng;
	    _scale=(FP_PRECISION)(1.0);
	}
	else {
        _polAng=_numPolarAng/2;
	    _scale=(FP_PRECISION)(0.5);
	}

	if (_polAng == 1) {
		_sinTheta[0] = (FP_PRECISION)(-0.577350269189626);
		_polarWeights[0] = 1.0;
	}
	else if (_polAng == 2) {
		_sinTheta[0] = (FP_PRECISION)(-0.861136311594053);
		_polarWeights[0] = 0.347854845137454;
		_sinTheta[1] = (FP_PRECISION)(-0.339981043584856);
		_polarWeights[1] = 0.652145154862546;
	}
	else if (_polAng == 3) {
		_sinTheta[0] = (FP_PRECISION)(-0.932469514203152);
		_polarWeights[0] = 0.17132449237917;
		_sinTheta[1] = (FP_PRECISION)(-0.661209386466265);
		_polarWeights[1] = 0.360761573048139;
		_sinTheta[2] = (FP_PRECISION)(-0.238619186083197);
		_polarWeights[2] = 0.467913934572691;
	}
	else if (_polAng == 4) {
		_sinTheta[0] = (FP_PRECISION)(-0.960289856497536);
		_polarWeights[0] = 0.101228536290376;
		_sinTheta[1] = (FP_PRECISION)(-0.796666477413627);
		_polarWeights[1] = 0.222381034453374;
		_sinTheta[2] = (FP_PRECISION)(-0.525532409916329);
		_polarWeights[2] = 0.313706645877887;
		_sinTheta[3] = (FP_PRECISION)(-0.18343464249565);
		_polarWeights[3] = 0.362683783378362;
	}
	else if (_polAng == 5) {
		_sinTheta[0] = (FP_PRECISION)(-0.973906528517172);
		_polarWeights[0] = 0.066671344308688;
		_sinTheta[1] = (FP_PRECISION)(-0.865063366688985);
		_polarWeights[1] = 0.149451349150581;
		_sinTheta[2] = (FP_PRECISION)(-0.679409568299024);
		_polarWeights[2] = 0.219086362515982;
		_sinTheta[3] = (FP_PRECISION)(-0.433395394129247);
		_polarWeights[3] = 0.269266719309996;
		_sinTheta[4] = (FP_PRECISION)(-0.148874338981631);
		_polarWeights[4] = 0.295524224714753;
	}
	else if (_polAng == 6) {
		_sinTheta[0] = (FP_PRECISION)(-0.981560634246719);
		_polarWeights[0] = 0.047175336386512;
		_sinTheta[1] = (FP_PRECISION)(-0.904117256370475);
		_polarWeights[1] = 0.106939325995318;
		_sinTheta[2] = (FP_PRECISION)(-0.769902674194305);
		_polarWeights[2] = 0.160078328543346;
		_sinTheta[3] = (FP_PRECISION)(-0.587317954286617);
		_polarWeights[3] = 0.203167426723066;
		_sinTheta[4] = (FP_PRECISION)(-0.36783149899818);
		_polarWeights[4] = 0.233492536538355;
		_sinTheta[5] = (FP_PRECISION)(-0.125233408511469);
		_polarWeights[5] = 0.249147045813403;
	}
	else if (_polAng == 7) {
		_sinTheta[0] = (FP_PRECISION)(-0.986283808696812);
		_polarWeights[0] = 0.035119460331752;
		_sinTheta[1] = (FP_PRECISION)(-0.928434883663573);
		_polarWeights[1] = 0.08015808715976;
		_sinTheta[2] = (FP_PRECISION)(-0.827201315069765);
		_polarWeights[2] = 0.121518570687903;
		_sinTheta[3] = (FP_PRECISION)(-0.687292904811685);
		_polarWeights[3] = 0.157203167158193;
		_sinTheta[4] = (FP_PRECISION)(-0.515248636358154);
		_polarWeights[4] = 0.185538397477938;
		_sinTheta[5] = (FP_PRECISION)(-0.31911236892789);
		_polarWeights[5] = 0.205198463721296;
		_sinTheta[6] = (FP_PRECISION)(-0.108054948707344);
		_polarWeights[6] = 0.215263853463158;
	}
	else if (_polAng == 8) {
		_sinTheta[0] = (FP_PRECISION)(-0.98940093499165);
		_polarWeights[0] = 0.027152459411754;
		_sinTheta[1] = (FP_PRECISION)(-0.944575023073233);
		_polarWeights[1] = 0.062253523938648;
		_sinTheta[2] = (FP_PRECISION)(-0.865631202387832);
		_polarWeights[2] = 0.095158511682493;
		_sinTheta[3] = (FP_PRECISION)(-0.755404408355003);
		_polarWeights[3] = 0.124628971255534;
		_sinTheta[4] = (FP_PRECISION)(-0.617876244402644);
		_polarWeights[4] = 0.149595988816577;
		_sinTheta[5] = (FP_PRECISION)(-0.458016777657227);
		_polarWeights[5] = 0.169156519395003;
		_sinTheta[6] = (FP_PRECISION)(-0.281603550779259);
		_polarWeights[6] = 0.182603415044924;
		_sinTheta[7] = (FP_PRECISION)(-0.095012509837637);
		_polarWeights[7] = 0.189450610455068;
	}
	else if (_polAng == 9) {
		_sinTheta[0] = (FP_PRECISION)(-0.99156516842093);
		_polarWeights[0] = 0.021616013526483;
		_sinTheta[1] = (FP_PRECISION)(-0.955823949571397);
		_polarWeights[1] = 0.04971454889497;
		_sinTheta[2] = (FP_PRECISION)(-0.892602466497556);
		_polarWeights[2] = 0.076425730254889;
		_sinTheta[3] = (FP_PRECISION)(-0.803704958972523);
		_polarWeights[3] = 0.100942044106287;
		_sinTheta[4] = (FP_PRECISION)(-0.691687043060353);
		_polarWeights[4] = 0.122555206711478;
		_sinTheta[5] = (FP_PRECISION)(-0.559770831073947);
		_polarWeights[5] = 0.140642914670651;
		_sinTheta[6] = (FP_PRECISION)(-0.411751161462843);
		_polarWeights[6] = 0.154684675126265;
		_sinTheta[7] = (FP_PRECISION)(-0.251886225691505);
		_polarWeights[7] = 0.164276483745833;
		_sinTheta[8] = (FP_PRECISION)(-0.084775013041735);
		_polarWeights[8] = 0.169142382963143;
	}
	else if (_polAng == 10) {
		_sinTheta[0] = (FP_PRECISION)(-0.993128599185095);
		_polarWeights[0] = 0.017614007139152;
		_sinTheta[1] = (FP_PRECISION)(-0.963971927277914);
		_polarWeights[1] = 0.040601429800387;
		_sinTheta[2] = (FP_PRECISION)(-0.912234428251326);
		_polarWeights[2] = 0.062672048334109;
		_sinTheta[3] = (FP_PRECISION)(-0.839116971822219);
		_polarWeights[3] = 0.083276741576705;
		_sinTheta[4] = (FP_PRECISION)(-0.746331906460151);
		_polarWeights[4] = 0.10193011981724;
		_sinTheta[5] = (FP_PRECISION)(-0.636053680726515);
		_polarWeights[5] = 0.118194531961518;
		_sinTheta[6] = (FP_PRECISION)(-0.510867001950827);
		_polarWeights[6] = 0.131688638449177;
		_sinTheta[7] = (FP_PRECISION)(-0.37370608871542);
		_polarWeights[7] = 0.142096109318382;
		_sinTheta[8] = (FP_PRECISION)(-0.227785851141645);
		_polarWeights[8] = 0.149172986472604;
		_sinTheta[9] = (FP_PRECISION)(-0.076526521133497);
		_polarWeights[9] = 0.152753387130726;
	}
	else if (_polAng == 12) {
		_sinTheta[0] = (FP_PRECISION)(-0.995187219997021);
		_polarWeights[0] = 0.012341229799987;
		_sinTheta[1] = (FP_PRECISION)(-0.974728555971309);
		_polarWeights[1] = 0.028531388628934;
		_sinTheta[2] = (FP_PRECISION)(-0.938274552002733);
		_polarWeights[2] = 0.04427743881742;
		_sinTheta[3] = (FP_PRECISION)(-0.886415527004401);
		_polarWeights[3] = 0.059298584915437;
		_sinTheta[4] = (FP_PRECISION)(-0.820001985973903);
		_polarWeights[4] = 0.07334648141108;
		_sinTheta[5] = (FP_PRECISION)(-0.740124191578554);
		_polarWeights[5] = 0.086190161531953;
		_sinTheta[6] = (FP_PRECISION)(-0.648093651936975);
		_polarWeights[6] = 0.097618652104114;
		_sinTheta[7] = (FP_PRECISION)(-0.545421471388839);
		_polarWeights[7] = 0.107444270115966;
		_sinTheta[8] = (FP_PRECISION)(-0.433793507626045);
		_polarWeights[8] = 0.115505668053726;
		_sinTheta[9] = (FP_PRECISION)(-0.315042679696163);
		_polarWeights[9] = 0.121670472927803;
		_sinTheta[10] = (FP_PRECISION)(-0.191118867473616);
		_polarWeights[10] = 0.125837456346828;
		_sinTheta[11] = (FP_PRECISION)(-0.064056892862605);
		_polarWeights[11] = 0.127938195346752;
	}
	else if (_polAng == 14) {
		_sinTheta[0] = (FP_PRECISION)(-0.996442497573954);
		_polarWeights[0] = 0.009124282593094;
		_sinTheta[1] = (FP_PRECISION)(-0.981303165370873);
		_polarWeights[1] = 0.021132112592771;
		_sinTheta[2] = (FP_PRECISION)(-0.954259280628938);
		_polarWeights[2] = 0.032901427782304;
		_sinTheta[3] = (FP_PRECISION)(-0.915633026392132);
		_polarWeights[3] = 0.044272934759004;
		_sinTheta[4] = (FP_PRECISION)(-0.865892522574395);
		_polarWeights[4] = 0.055107345675717;
		_sinTheta[5] = (FP_PRECISION)(-0.805641370917179);
		_polarWeights[5] = 0.065272923966999;
		_sinTheta[6] = (FP_PRECISION)(-0.735610878013632);
		_polarWeights[6] = 0.074646214234569;
		_sinTheta[7] = (FP_PRECISION)(-0.656651094038865);
		_polarWeights[7] = 0.083113417228901;
		_sinTheta[8] = (FP_PRECISION)(-0.569720471811402);
		_polarWeights[8] = 0.090571744393033;
		_sinTheta[9] = (FP_PRECISION)(-0.475874224955118);
		_polarWeights[9] = 0.09693065799793;
		_sinTheta[10] = (FP_PRECISION)(-0.376251516089079);
		_polarWeights[10] = 0.102112967578061;
		_sinTheta[11] = (FP_PRECISION)(-0.272061627635178);
		_polarWeights[11] = 0.106055765922846;
		_sinTheta[12] = (FP_PRECISION)(-0.16456928213338);
		_polarWeights[12] = 0.108711192258294;
		_sinTheta[13] = (FP_PRECISION)(-0.055079289884034);
		_polarWeights[13] = 0.110047013016475;
	}
	else if (_polAng == 16) {
		_sinTheta[0] = (FP_PRECISION)(-0.997263861849482);
		_polarWeights[0] = 0.00701861000947;
		_sinTheta[1] = (FP_PRECISION)(-0.985611511545268);
		_polarWeights[1] = 0.016274394730906;
		_sinTheta[2] = (FP_PRECISION)(-0.964762255587506);
		_polarWeights[2] = 0.025392065309262;
		_sinTheta[3] = (FP_PRECISION)(-0.93490607593774);
		_polarWeights[3] = 0.034273862913021;
		_sinTheta[4] = (FP_PRECISION)(-0.896321155766052);
		_polarWeights[4] = 0.042835898022227;
		_sinTheta[5] = (FP_PRECISION)(-0.84936761373257);
		_polarWeights[5] = 0.050998059262376;
		_sinTheta[6] = (FP_PRECISION)(-0.794483795967942);
		_polarWeights[6] = 0.058684093478536;
		_sinTheta[7] = (FP_PRECISION)(-0.73218211874029);
		_polarWeights[7] = 0.065822222776362;
		_sinTheta[8] = (FP_PRECISION)(-0.663044266930215);
		_polarWeights[8] = 0.072345794108849;
		_sinTheta[9] = (FP_PRECISION)(-0.587715757240762);
		_polarWeights[9] = 0.07819389578707;
		_sinTheta[10] = (FP_PRECISION)(-0.506899908932229);
		_polarWeights[10] = 0.083311924226947;
		_sinTheta[11] = (FP_PRECISION)(-0.421351276130635);
		_polarWeights[11] = 0.087652093004404;
		_sinTheta[12] = (FP_PRECISION)(-0.331868602282128);
		_polarWeights[12] = 0.091173878695764;
		_sinTheta[13] = (FP_PRECISION)(-0.239287362252137);
		_polarWeights[13] = 0.093844399080805;
		_sinTheta[14] = (FP_PRECISION)(-0.144471961582796);
		_polarWeights[14] = 0.095638720079275;
		_sinTheta[15] = (FP_PRECISION)(-0.048307665687738);
		_polarWeights[15] = 0.096540088514728;
	}
	else if (_polAng == 32) {
		_sinTheta[0] = (FP_PRECISION)(-0.999305041735772);
		_polarWeights[0] = 0.001783280721696;
		_sinTheta[1] = (FP_PRECISION)(-0.996340116771955);
		_polarWeights[1] = 0.004147033260562;
		_sinTheta[2] = (FP_PRECISION)(-0.991013371476744);
		_polarWeights[2] = 0.006504457968978;
		_sinTheta[3] = (FP_PRECISION)(-0.983336253884626);
		_polarWeights[3] = 0.008846759826364;
		_sinTheta[4] = (FP_PRECISION)(-0.973326827789911);
		_polarWeights[4] = 0.011168139460131;
		_sinTheta[5] = (FP_PRECISION)(-0.961008799652054);
		_polarWeights[5] = 0.013463047896719;
		_sinTheta[6] = (FP_PRECISION)(-0.946411374858403);
		_polarWeights[6] = 0.015726030476025;
		_sinTheta[7] = (FP_PRECISION)(-0.92956917213194);
		_polarWeights[7] = 0.017951715775697;
		_sinTheta[8] = (FP_PRECISION)(-0.910522137078503);
		_polarWeights[8] = 0.02013482315353;
		_sinTheta[9] = (FP_PRECISION)(-0.889315445995114);
		_polarWeights[9] = 0.022270173808383;
		_sinTheta[10] = (FP_PRECISION)(-0.865999398154093);
		_polarWeights[10] = 0.024352702568711;
		_sinTheta[11] = (FP_PRECISION)(-0.84062929625258);
		_polarWeights[11] = 0.026377469715055;
		_sinTheta[12] = (FP_PRECISION)(-0.813265315122798);
		_polarWeights[12] = 0.028339672614259;
		_sinTheta[13] = (FP_PRECISION)(-0.783972358943341);
		_polarWeights[13] = 0.030234657072402;
		_sinTheta[14] = (FP_PRECISION)(-0.752819907260532);
		_polarWeights[14] = 0.032057928354852;
		_sinTheta[15] = (FP_PRECISION)(-0.719881850171611);
		_polarWeights[15] = 0.033805161837142;
		_sinTheta[16] = (FP_PRECISION)(-0.685236313054233);
		_polarWeights[16] = 0.035472213256882;
		_sinTheta[17] = (FP_PRECISION)(-0.648965471254657);
		_polarWeights[17] = 0.03705512854024;
		_sinTheta[18] = (FP_PRECISION)(-0.611155355172393);
		_polarWeights[18] = 0.038550153178616;
		_sinTheta[19] = (FP_PRECISION)(-0.571895646202634);
		_polarWeights[19] = 0.03995374113272;
		_sinTheta[20] = (FP_PRECISION)(-0.531279464019894);
		_polarWeights[20] = 0.041262563242624;
		_sinTheta[21] = (FP_PRECISION)(-0.489403145707053);
		_polarWeights[21] = 0.042473515123654;
		_sinTheta[22] = (FP_PRECISION)(-0.446366017253464);
		_polarWeights[22] = 0.043583724529323;
		_sinTheta[23] = (FP_PRECISION)(-0.402270157963992);
		_polarWeights[23] = 0.044590558163757;
		_sinTheta[24] = (FP_PRECISION)(-0.357220158337668);
		_polarWeights[24] = 0.045491627927418;
		_sinTheta[25] = (FP_PRECISION)(-0.311322871990211);
		_polarWeights[25] = 0.046284796581314;
		_sinTheta[26] = (FP_PRECISION)(-0.264687162208767);
		_polarWeights[26] = 0.04696818281621;
		_sinTheta[27] = (FP_PRECISION)(-0.217423643740007);
		_polarWeights[27] = 0.04754016571483;
		_sinTheta[28] = (FP_PRECISION)(-0.169644420423993);
		_polarWeights[28] = 0.047999388596458;
		_sinTheta[29] = (FP_PRECISION)(-0.121462819296121);
		_polarWeights[29] = 0.048344762234803;
		_sinTheta[30] = (FP_PRECISION)(-0.072993121787799);
		_polarWeights[30] = 0.048575467441503;
		_sinTheta[31] = (FP_PRECISION)(-0.024350292663424);
		_polarWeights[31] = 0.04869095700914;
	}
	else {}
// #if defined(M_3DMOCex)
    for (i=0;i<_polAng;i++) {
		_polarWeights[i]*=_scale;
	}

    for (i=_polAng;i<_numPolarAng;i++) {
		_sinTheta[i] = _sinTheta[_numPolarAng-1-i];
		_polarWeights[i] = _polarWeights[_numPolarAng-1-i];
	}
// #endif // defined(M_3DMOCex)

	for (i = 0; i < _numPolarAng; i++) {
		_sinTheta[i] = (FP_PRECISION)(sqrt(1.0 - _sinTheta[i] * _sinTheta[i]));
	}
	for (i=0;i<_numPolarAng;i++) {
		_cosTheta[i] = sqrt((FP_PRECISION)(1.0) - pow(_sinTheta[i], (FP_PRECISION)(2.0)));
	}
}
//This routine uses the tabulated values for the Gauss-Legendre polar angle quadrature, including the sine thetas and weights.
/**
* @brief    finds the roots of Legendre polynomial of order n
* @detail   guesses for positive roots are set at logarithmic intervals.
*           Positive roots are found simultaneously using an
*           Alberth-Householder-n method. Each guess is successively nudged
*           towards a true root. Only the positive roots are calculated
* @param    n the order of the polynomial
* @return   a list of the roots of the polynomial
*/
/**
* @brief    the first logarithmic derivative of a Legendre polynomial
* @param    _numRoots the order of the polynomial
* @param    x point at which to evaluate the logarithmic derivative
* @return   the value of the logarithmic derivative at x
*/
/**
* @brief    the second logarithmic derivative of a Legendre polynomial
* @param    _numRoots the order of the polynomial
* @param    _x point at which to evaluate the logarithmic derivative
* @return   the value of the logarithmic derivative at _x
*/
/**
* @brief    the Legendre polynomial of degree n evaluated at x
* @param    n an integer >=0: the order of the polynomial
* @param    x in (-1,1), the point at which to evaluate the polynomial
* @return   the value of the Legendre polynomial of degree n at x
*/
 /**
 * @ Function: SetAzimuthalQuadrature()
 *   introduction: generate azimuthal quadrature set
 * @ Parameter List: _numTrackSet - number of track set
 *                   _quadType - quadrature type, 0-modified equal weights, 1-equal weights, 
 *                   _numAzimAng - number of azimuthal angles
 *                   _effAzimuAng - effective azimuthal angle
 *   Return: void
 * @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
void SetAzimuthalQuadrature_Weights(int _quadType,int _numAzimAng,FP_PRECISION_GEO *_effAzimuAng,FP_PRECISION *_azimWeights)
{
	int i;
	FP_PRECISION_GEO _x1,_x2,_oneOverTwoPI,_oneOver;
	_oneOverTwoPI=0.5/ONE_PI;
	if (_quadType==0) {
        for (i=0;i<_numAzimAng;i++) {
            if (i<_numAzimAng-1)
                _x1=0.5*(_effAzimuAng[i+1]-_effAzimuAng[i]);
            else
                _x1=2.*ONE_PI-_effAzimuAng[i];
            if (i>=1)
                _x2=0.5*(_effAzimuAng[i]-_effAzimuAng[i-1]);
            else
                _x2=_effAzimuAng[i];
			_azimWeights[i]=(_x1+_x2)*_oneOverTwoPI;//

        }
    }
    /**  equal weight*/
    else if (_quadType==1) {
		_oneOver=1./((FP_PRECISION_GEO)_numAzimAng);
        for (i=0;i<_numAzimAng;i++) {
			_azimWeights[i]=_oneOver;
        }
    }
    else {}
}

/**
* @ Function: TotalWeight_AllocMem()
*   introduction: allocate the memory for the array of total weights
* @ Parameter List: 
*   Return: void
* @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
 /**
 * @ Function: SetTotalWeight()
 *   introduction: generate azimuthal quadrature set
 * @ Parameter List: _numTrackSet - number of track set
 *                   _numAzimAng - number of azimuthal angles in [0,2PI]
 *                   _numPolarAng - number of polar angles in [0,PI/2]
 *                   _sinTheta - sin(polar angle)
 *                   _polarWeights - polar weights
 *                   _azimWeights - azimuthal weights
 *                   _raySpacingEff - effective ray spacing, cm
 *   Return: void
 * @ Created     : 2017/01/01, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
 *   1st modified:
 */
void SetTotalWeight(int _numTrackSet,int *_numAzimAng,int *_numPolarAng,FP_PRECISION **_sinTheta,\
    FP_PRECISION **_polarWeights,FP_PRECISION **_azimWeights,FP_PRECISION_GEO **_raySpacingEff,\
	FP_PRECISION **_totalWeight, FP_PRECISION **_totalWeight_2DMOC)
{
	int i, j, k, _id = 0;
	FP_PRECISION _temp = (FP_PRECISION)(1.0);
#if defined(M_PrintDetail)
	printf("\n [PrintDetail] quadrature information");
#endif // defined(M_PrintDetail)

#if defined(M_Use4PI)
	_temp = (FP_PRECISION)(FOUR_PI);
#else
	_temp = (FP_PRECISION)(1.0);
#endif //defined(M_Use4PI)
	// printf("\n _temp=%e",_temp);
	for (k=0;k<_numTrackSet;k++) {
	    for (i = 0; i < _numAzimAng[k]; i++) {
		    for (j = 0; j < _numPolarAng[k]; j++) {
				_totalWeight_2DMOC[k][i*_numPolarAng[k] + j] = (FP_PRECISION)(_temp * _sinTheta[k][j] * _polarWeights[k][j] * _azimWeights[k][i] * _raySpacingEff[k][i]);
				if (IsSetHalfPolar == 1) {
					_totalWeight[k][i*_numPolarAng[k] + j] = _totalWeight_2DMOC[k][i*_numPolarAng[k] + j];
				}
				else {
					_totalWeight[k][i*_numPolarAng[k] + j] = (FP_PRECISION)(_temp * _polarWeights[k][j] * _azimWeights[k][i]);
				}
// printf("\n _totalWeight_2DMOC[%d][%d]=%e",k,i*_numPolarAng[k] + j,_totalWeight_2DMOC[k][i*_numPolarAng[k] + j]);
// printf("\n _temp=%e, _sinTheta=%e, _polarWeights=%e, _azimWeights=%e, _raySpacingEff=%e",\
//     _temp, _sinTheta[k][j],_polarWeights[k][j],_azimWeights[k][i], _raySpacingEff[k][i]);
			}
		}
	}
}

FP_PRECISION Plm(int _l, int _m, FP_DOUBLE _x)
{
    FP_DOUBLE _tmp1 = 1.0, _tmp2 = 1.0, _factor;
	int i;
    if (_m < 0 || _m > _l || fabs(_x) > 1.0) {
		printf("\n [**ERROR**]  Bad arguments in Plm, m=%d, l=%d, x=%e", _m, _l, _x);
        return 0.0;
    }
    for (i = 1; i <= (_l - _m); i++) {
        _tmp1 *= (FP_DOUBLE)(i);
    }
    for (i = 1; i <= (_l + _m); i++) {
        _tmp2 *= (FP_DOUBLE)(i);
    }
    // _factor = sqrt((4.0 * ONE_PI * _tmp2) / ((FP_DOUBLE)(2 * _l + 1) * _tmp1));
    _factor = sqrt(_tmp2 / ((FP_DOUBLE)(2 * _l + 1) * _tmp1));
	_factor *= tildePlm(_l, _m, _x);
    return (FP_PRECISION)(_factor);
}

FP_PRECISION tildePlm(int _l, int _m, FP_PRECISION _x)
{
    FP_DOUBLE _factor1, _factor2, _val = 0.0;
    if (_m < 0 || _m > _l || fabs(_x) > 1.0) {
		printf("\n [**ERROR**]  Bad arguments in tildePlm, m=%d, l=%d, x=%e", _m, _l, _x);
        return 0.0;
    }
    _factor1 = sqrt((FP_DOUBLE)(4 * _l * _l - 1) / (FP_DOUBLE)(_l * _l - _m * _m));
    _factor2 = sqrt((FP_DOUBLE)((_l - 1) * (_l - 1) - _m * _m) / (FP_DOUBLE)(4 * (_l - 1) * (_l - 1) - 1));
    if (_l == _m) {
		_val = tildePmm(_m, _x);
	} 
	else if (_l == _m + 1) {
		_val = tildePmm1(_m, _x);
	}
	else {
		_val = _factor1 * (_x * tildePlm(_l-1, _m, _x) - _factor2 * tildePlm(_l-2, _m, _x));
	}
    return (FP_PRECISION)(_val);
}

FP_PRECISION tildePmm(int _m, FP_PRECISION _x) {
    FP_DOUBLE _factor, _tmp, _tmp2;
	int i;
	// if (_m % 2 == 0) {
	// 	_factor = 1.0;
	// }
	// else {
	// 	_factor = -1.0;
	// }
    _tmp = 1.0 - _x * _x;
    _tmp2 = 1.0;
    for (i = 1; i <= _m * 2 - 1; i += 2) {
        _tmp2 *= _tmp * (FP_DOUBLE)(i) / (FP_DOUBLE)(i + 1);
    }
    // _factor *= sqrt((FP_DOUBLE)(2 * _m + 1) * _tmp2 / (4.0 * ONE_PI));
    _factor = sqrt((FP_DOUBLE)(2 * _m + 1) * _tmp2);
    return (FP_PRECISION)(_factor);
}

FP_PRECISION tildePmm1(int _m, FP_PRECISION _x) {
	FP_DOUBLE _val;
	_val = _x * sqrt((FP_DOUBLE)(2 * _m + 3)) * tildePmm(_m, _x);
    return (FP_PRECISION)(_val);
}

M_INT32 FactorialEval(M_INT32 _int)
{
	M_INT32 _fact=1, i;
	if (_int == 0) {
		_fact = 1;
	}
	else {
		_fact = 1;
	    for (i = 1; i < _int + 1; i++) {
			_fact *= i;
	    }	
	}
	return (_fact);
}

void SetAzimuthalQuadrature(int _numTrackSet, int _quadType,int *_numAzimAng,FP_PRECISION_GEO **_effAzimuAng, FP_PRECISION **_azimWeights)
{
    int i;
    if (_quadType == 0 || _quadType == 1) {}
	else {
	    printf("\n [ WARNING ]  Unable to set the azimuthal quadrature with type_id %d (0~1) ", _quadType);
	    _quadType = 0;
	    printf("\n [ WARNING ]  The type of polar quadrature is automatically set to 0");
	    fflush(stdout);
	}
    /** Compute azimuthal angle quadrature weights*/
	for (i=0;i<_numTrackSet;i++) {
        SetAzimuthalQuadrature_Weights(_quadType,_numAzimAng[i],_effAzimuAng[i],_azimWeights[i]);
	}
}

void SetSinThetaForExpTabel(int _numPolAng, FP_PRECISION *_sinTheta)
{
	int i;
	FP_PRECISION _one=(FP_PRECISION)(1.0);
	SinThetaForExp = AllocMem_FP_I_1(_numPolAng, "SinThetaForExp");
	for (i = 0; i < _numPolAng; i++) {
		SinThetaForExp[i] = NumExpPoints_One/*NumExpPoints_One_Mod1*/ / _sinTheta[i];
		OneOverSinTheta[i] = _one/_sinTheta[i];
	}
}

void SetTheta(int _numTrackSet, int *_numPolarAng, FP_PRECISION **_sinTheta, FP_DOUBLE **_theta)
{
	int i, j, _polAng=0, _sum;
	for (i=0;i<_numTrackSet;i++) {
		_polAng = _numPolarAng[i]/2;
		_sum = _numPolarAng[i] - 1;
		for (j=0;j<_polAng;j++) {
			_theta[i][j] = asin(_sinTheta[i][j]);
			_theta[i][_sum-j] = ONE_PI - _theta[i][j];
		}
	}
}

FP_PRECISION **TotalWeight_AllocMem(int _numTrackSet, int *_numAzimAng, int *_numPolarAng, char *_parameter)
{
	int i, *_sum = NULL;
	FP_PRECISION **_totalWeight = NULL;
	/* count number */
	_sum = AllocMem_i32_i32(_numTrackSet, (char *)("\0"));
	for (i = 0; i < _numTrackSet; i++) {
		_sum[i] = _numAzimAng[i] * _numPolarAng[i];
	}
	
	_totalWeight = AllocMem_FP_IP(_numTrackSet, _sum, _parameter);
	_sum = FreeMem_i32_I(_sum, (char *)("\0"));
	return _totalWeight;
}

FP_PRECISION Ylm(int _l, int _m, FP_DOUBLE _theta, FP_DOUBLE _phi)
{
    FP_PRECISION _val = 0.0, _mul = 0.0;
	int _m1 = 0;
	if (_m < 0) {
		_m1 = abs(_m);
		_mul = sin((FP_DOUBLE)_m1 * _phi);
	}
	else {
		_m1 = _m;
		_mul = cos((FP_DOUBLE)_m1 * _phi);
	}

	_val = Plm(_l, _m1, cos(_theta)) * _mul;
    return _val;
}

//sequence
/**
* @brief    calculates the weights to be used in Gauss-Legendre Quadrature
* @param    roots a vector containing the roots of the Legendre polynomial
* @param    _numRoots the order of the Legendre Polynomial
* @return   a vector of weights matched by index to the vector of roots
*/
/**
* @brief    calculates the sine theta to be used in Gauss-Legendre Quadrature
*/
/**
* @brief Routine to initialize the equal weight polar quadrature.
* @details This routine generates the sine thetas and weights.
*/
void EqualWeightPolarQuad(FP_PRECISION *_sinTheta, FP_PRECISION *_polarWeights, int _numPolarAng)
{
	FP_PRECISION_GEO cos_theta_a, cos_theta_b, y1, y2;
	FP_PRECISION_GEO *_thetas = NULL;
	int i;
	// check
	
	if ((_thetas = (FP_PRECISION_GEO *)calloc(_numPolarAng, sizeof(FP_PRECISION_GEO))) == NULL) {
		printf("\n  Calloc failed in EqualWeightPolarQuad - _thetas !");
		exit(58);
	}

	/* Generate the sin thetas and weights using equations 420-422 of the
	* DOE Nucl. Eng. Handbook "Lattice Physics Computations" */
	cos_theta_a = 1.;
	for (i = 0; i < _numPolarAng; i++) {
		cos_theta_b = cos_theta_a - (1. / _numPolarAng);
		_thetas[i] = acos(0.5 * (cos_theta_a + cos_theta_b));
		_sinTheta[i] = (FP_PRECISION)(sin(_thetas[i]));
		cos_theta_a = cos_theta_b;
	}
	//set weights
	for (i = 0; i < _numPolarAng; i++) {

		if (i < _numPolarAng - 1)
			y1 = 0.5 * (cos(_thetas[i]) - cos(_thetas[i + 1]));
		else
			y1 = cos(_thetas[i]);

		if (i >= 1)
			y2 = 0.5 * (cos(_thetas[i - 1]) - cos(_thetas[i]));
		else
			y2 = 1.0 - cos(_thetas[i]);
		_polarWeights[i] = (y1 + y2) / 2.0;
	}
	free(_thetas);
	_thetas = NULL;
}

/**
* @brief Routine to initialize the equal angle polar quadrature.
* @details This routine generates the sine thetas and weights.
*/
void EqualAnglePolarQuad(FP_PRECISION *_sinTheta, FP_PRECISION *_polarWeights, int _numPolarAng)
{
	FP_PRECISION_GEO *_thetas = NULL;
	FP_PRECISION_GEO cos_theta_a, cos_theta_b, theta_a, theta_b, delta_theta, y1, y2;
	int i;
	// check
	_thetas = (FP_PRECISION_GEO *)calloc(_numPolarAng, sizeof(FP_PRECISION_GEO));
	if (_thetas == NULL) {
		printf("\n  Calloc failed in EqualAnglePolarQuad - _thetas !");
		exit(58);
	}
	delta_theta = ONE_PI / (_numPolarAng*2);
	/* Generate the sin thetas and weights using equations 420-422 of the
	* DOE Nucl. Eng. Handbook "Lattice Physics Computations" */
	theta_a = 0.;
	for (i = 0; i < _numPolarAng; i++) {
		theta_b = theta_a + delta_theta;
		cos_theta_a = cos(theta_a);
		cos_theta_b = cos(theta_b);
		_thetas[i] = acos((0.5 * (cos_theta_a + cos_theta_b)));
		_sinTheta[i] = (FP_PRECISION)(sin(_thetas[i]));
		theta_a = theta_b;
	}
	// set weights
	for (i = 0; i < _numPolarAng; i++) {
		if (i < _numPolarAng - 1)
			y1 = 0.5 * (cos(_thetas[i]) - cos(_thetas[i + 1]));
		else
			y1 = cos(_thetas[i]);

		if (i >= 1)
			y2 = 0.5 * (cos(_thetas[i - 1]) - cos(_thetas[i]));
		else
			y2 = 1.0 - cos(_thetas[i]);

		_polarWeights[i] = (y1 + y2) / 2.0;
	}
	free(_thetas);
	_thetas = NULL;
}

// calculate \tilde{P}_m^m
// calculate \tilde{P}_{(m+1)}^m
// calculate \tilde{P}_m^l
// associated Legendre
// calculate the spherical harmonics for theta and phi
/**
* @ Function: CoeffEval_HOScatt()
*   introduction: compute the coefficient a in high-order expension of angular flux
* @ Created     : 2020/12/03, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void CoeffEval_HOScatt(M_INT32 _orderN)
{
	M_INT32 i, l, m, _abs;
	FP_PRECISION _kronecker = 0.0, _mul = 0.0;
	CoeffA_HOScatt = AllocMem_FP_I((size_t)((_orderN + 1) * (_orderN + 1)), (char *)("CoeffA_HOScatt"));

	i=0;
	for (l=0;l<=_orderN;l++) {
		for (m=-l;m<=l;m++) {
			if (m==0) {
				_kronecker = 1.0;
			}
			else {
				_kronecker = 0.0;
			}
			_mul = 2.0 / (1.0 + _kronecker);
			_abs = abs(m);
			CoeffA_HOScatt[i] = (FP_PRECISION)(FactorialEval(l - _abs)) / (FP_PRECISION)(FactorialEval(l + _abs))  * _mul;
			i++;
		}
	}
}

/**
* @ Function: FactorialEval()
*   introduction: compute the factorial of a positive integer
* @ Created     : 2020/12/03, by SONG PEITAO, NPSRC, HEU (songpeitao2015@163.com)
*   1st modified:
*/
void SetSpheHarmCoeff(int _numTrSet, int *_numAzimAng, int *_numPolAng, \
    int _scattOrder, FP_DOUBLE **_effAzimAng, FP_DOUBLE **_theta, FP_PRECISION **_spheHarmCoeff)
{
	int i,j,k, m, l, _count;
	FP_DOUBLE _azim, _pol;
	for (i=0; i<_numTrSet; i++) {
		_count = 0;
		for (l=0; l<=_scattOrder; l++) {
			for (m=-l;m<=l;m++) {
				for (j=0;j<_numAzimAng[i];j++) {
					_azim = _effAzimAng[i][j];
					for (k=0;k<_numPolAng[i];k++) {
						_pol = _theta[i][k];
						_spheHarmCoeff[i][_count] = Ylm(l, m, _pol, _azim);
					    // printf("\n _spheHarmCoeff[j=%2d, phi=%.2lf][k=%2d, pol=%.2lf][m=%2d][_count=%3d]=%e,",j, _azim, k, _pol,m,_count, _spheHarmCoeff[i][_count]);
					    _count++;
					}
				}
			}
		}
	}
}
