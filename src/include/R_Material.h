
#ifndef R_MATERIAL_H
#define R_MATERIAL_H
#include "R_control.h"
#include "MOC_CommonMacro.h"

    struct R_NUC 
	{
        int nuc_id;
        FP_PRECISION nuc_den;
		//int ifres;
	};

	//struct NUC_FSR 
	//{
 //       int fsr_ID;
	//	FP_PRECISION tem;
	//	int nuc_n;
	//	struct R_NUC nuc[300];
 //      };

    struct R_MAT 
	{
        int id;
     	FP_PRECISION mtem;
		int nuc_number;
		int Num_Res;
		int Num_Non;
        struct R_NUC nuc[1000];
    };

	struct R_TRISO
	{
		int id;
		int lay_n;
		FP_PRECISION vol_ratio;
		FP_PRECISION radi[20];
		int mat_id[20];
		int sub_lay[20];
	};

	struct R_COM
	{
		int id;
		int mtx_id;
		int if_homo;
		int Num_triso;
		int _mat_id;
		struct R_TRISO R_tio[100];
	};


	struct PIN_TYPE
	{
		unsigned char if_ring;
	};

   struct FLUX_INFO
   {
	   char fname[NAME_N];
   };


   struct RES_CATE_NUC
   {
	   int nuc_name[NucNumbersInWorld];
	   int nuc_num;
   };
   struct RES_INFO
   {
	   int res_num;
	   int cate_num;
	   int res_cate[NucNumbersInWorld];
	   struct RES_CATE_NUC res_cate_nuc[NucNumbersInWorld];

	   int res_id[NucNumbersInWorld];
	   //int ****res_order;
   };
   struct NUC_INFO
   {
	   int nuc_num;
	   int nuc_id[NucNumbersInWorld];
   };
   struct TRISO_INFO
   {
	   int lay_n;
	   int mat_id[30];
	   FP_PRECISION radi[30];
	   int sublay_n[30];
   };
   struct DH_INFO
   {
	   int triso_num;
	   int triso_ID[100];
	   struct TRISO_INFO triso_info[100];
   };


#endif // MATERIAL_H
