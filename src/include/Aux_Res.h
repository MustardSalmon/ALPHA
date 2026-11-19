#ifndef AUX_RES_H
#define AUX_RES_H

extern int NELT_SubUFG,NOUFG;

extern int *eid_SubUFG,*EIDD;

extern int *SubTemNumUFG;         //Number of temperature for Nuclide
extern int ***nsubgUFG;           //Number of Nuclide, TEMPNN, 2164
extern FP_PRECISION **SubTempUFG;
extern FP_PRECISION ****SubWeiUFG;//Number of Nuclide, TEMPNN, 2164, SUBNN
extern FP_PRECISION ****SubTotUFG;//Number of Nuclide, TEMPNN, 2164, SUBNN
extern FP_PRECISION ****SubScaUFG;//Number of Nuclide, TEMPNN, 2164, SUBNN
extern FP_PRECISION ****SubIntUFG;//Number of Nuclide, TEMPNN, 2164, SUBNN
extern FP_PRECISION ****SubAbsUFG;//Number of Nuclide, TEMPNN, 2164, SUBNN
extern FP_PRECISION ****SubNufUFG;//Number of Nuclide, TEMPNN, 2164, SUBNN

extern FP_PRECISION *xs_pot_MC23;//Number of Nuclide

extern int NucFissCount; // Number of fission nuclide
extern int *NucIfChi; //Number of Nuclide
extern FP_PRECISION **NucChi; //Number of Fission Nuclide, 2164
extern int **NucSt; // Number of Nuclide, 2164
extern int **NucEd; // Number of Nuclide, 2164
extern int *tempSt;
extern int *tempEd;
extern FP_PRECISION **NucInE; //Number of Nuclide, level In-elastic Xs  

void Readlib_UFG();
void ReadMatTitle();
void ReadMatTitle2();
void ReadScaMatrix(int nucID, FP_PRECISION ***micScaMat);
int getGlobalIndex(int lay, int nucID);

void setLocalHomoPara(int NumGroups, int NumGroupsCG, int *GroupBound, \
    int _scaOrderMax, \
    FP_PRECISION_FLUX *_flux, 
    FP_PRECISION *_MacroTotXs, FP_PRECISION *_MacroNufXs, FP_PRECISION *_MacroScaXs, FP_PRECISION *_MacroChi, \
    M_UINT64 *_MacroXS_s_NZRowStartID, M_UINT64 *_MacroXS_s_NZColStartID, M_INT32 _totalNumFSRs_subDom, \
    double *_Vol, FILE *homoprofile);
void resetLocalHomoPara();

void CalHomoPara();
void CalHomoFlux(int _scaOrderlm, FP_PRECISION *_HomoFlux);
void CalHomoHORate(int _scaOrderlm, FP_PRECISION *_Rate, FP_PRECISION *_MacroXs);
void CalHomoHOScaRate(int _scaOrder, int _scaOrderlm, FP_PRECISION *_HOScaRate);
void CalHomoChi(FP_PRECISION *_Chi);

void freeMem();


extern FP_PRECISION *HomoFlux;
extern FP_PRECISION *HomoTot, *HomoNuf, *HomoChi;
extern FP_PRECISION *HomoScatterTemp, **HomoScatter;

extern FP_PRECISION *HomoFluxCG;
extern FP_PRECISION *HomoTotCG, *HomoNufCG, *HomoChiCG;
extern FP_PRECISION *HomoScatterTempCG, **HomoScatterCG;


#endif