
#include "zres_common_func.h"

FILE* ZRES_MemLog = NULL;
rsuint ZRES_MemUsage = 0;
double ZRES_limit = 0.1;

void zres_init_memlog(char* fileName) {
    ZRES_MemUsage = 0;
    ZRES_MemLog = fopen(fileName, "w");
    fprintf(ZRES_MemLog, "%-40s %18s %18s %18s\n", "varName", "varDim", "varSize [MB]", "TotalUsage [MB]");
    fprintf(ZRES_MemLog, "------------------------------------------------------------------------------\n");
}

void zres_close_memlog() {
    fclose(ZRES_MemLog);
}

// need to check this function, when use it, O_o
void* zres_malloc(size_t dim, 
                  size_t dataSize, 
                  char* varName) {
    void* ptr = calloc(dim, dataSize);
    double mb = ZresBit2MB(dim * dataSize);
    ZRES_MemUsage += dim * dataSize;

    if (mb > ZRES_limit) {
        fprintf(ZRES_MemLog, "%-40s %15zu %15f MB %15f MB\n", varName, dim, mb, ZresBit2MB(ZRES_MemUsage));
    }

    return ptr;
}

void zres_free_(void* ptr, 
                size_t dim, 
                char* varName) {
    free(ptr);
    ptr = NULL;
    double mb = -ZresBit2MB(dim * sizeof(ptr));
    ZRES_MemUsage -= dim * sizeof(ptr);
    
    if (mb > ZRES_limit) {
        fprintf(ZRES_MemLog, "%-40s %15f MB %15f MB\n", varName, mb, ZresBit2MB(ZRES_MemUsage));
    }
}

void zres_raise_error(char* information) {
    printf("%s", information);
    exit(1000);
}