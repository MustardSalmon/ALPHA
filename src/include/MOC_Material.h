
#ifndef MOC_MATERIAL_H
#define MOC_MATERIAL_H
#include "MOC_CommonMacro.h"

    struct XS {
        FP_PRECISION xs_num;
        int xsname;
        };
    struct MAT {
        int id;
        int type;
        int ty_den;
        FP_PRECISION den;
        FP_PRECISION tem;
        struct XS xs[XS_N];
        };
/**
 * specify the axial materials for GPU_TSKernelType rod
 */
struct CR_MAT_DIS {
    int mat_id;
    FP_PRECISION mat_bound[2];
};
struct CR_MAT {
    int id;
    int num_mat;
    struct CR_MAT_DIS *mat_dis;
};
void copy_material(struct MAT *_copy_mat, struct MAT *_mat);
void free_coreMaterials(struct MAT *_coreMat, char *_parameter);
void free_contRodMat(int _num, struct CR_MAT *_CRMat, char *_parameter);

#endif // MATERIAL_H
