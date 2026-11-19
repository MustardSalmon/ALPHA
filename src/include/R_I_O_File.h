#ifndef R_I_O_File_H
#define R_I_O_File_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MOC_Core.h"

#define R_NAME_N 512
#define R_KEYWORD_N 128
	
extern char IP_MAT[256];

extern int Input_Type, Scatt_Type, Res_Type, Inter_Type, Lib_Type, TransCorre_Type, IF_SET_MIN, SGFSP_Type, Sldw_Type, IF_READ_LIB_INDEX;
extern int IF_OUTP_LOG_FILE; extern  int IF_OUTP_FIX; extern int IF_OUTP_FIX_FLUX; extern  int IF_OUTP_XS; extern int IF_REB_CHECK; extern int IF_OUTP_RES_INFO;
extern int IF_OUTP_SCATT_INDEX; extern int IF_OUTP_RES_XS, IF_OUTP_MATERIAL_XS;
extern FILE *ip_r, *op_r, *ipg_r; extern char Keyword[R_KEYWORD_N], Lib_Path[R_KEYWORD_N];

extern int IF_Inter, bdite;

int R_g_s_pos(FILE *fp);
void R_identify_keyword(int _procID, FILE *fp1, char *keyword, char *p);

void Output_Material_Check(int _procID);
void R_I_O_and_Lib(int _procID);
void Output_Material_Check_FSR(int _procID);
void R_I_O_File_DH_new(int _procID);
void Output_Material_Check_DH(int _procID);
void reb_check_DH(int _procID);
void Material_Evaluation_subDomain(struct CORE *_core, int _subDomID_row, int _subDomID_column);
void Material_Evaluation_subDomain_trackFile();
void R_I_O_Res(int _procID);


;;;;
#endif
