//
// Created by hangZou on 2022/7/14.
//
#include "auxiliary.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "MOC_CommonMacro.h"
#include "MOC_Geometry.h" // For FSR Information
#include "MOC_Track.h"    // For FSR Information

void checkSysMemUsage() {
#ifdef SCHED_LINUX
#ifdef MPI_USED
    int myproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
    if (myproc == 0) {
#endif
    struct sysinfo s_info;
    char info_buff[100];
    float total_mem, free_mem;
    if (sysinfo(&s_info) == 0) {

        total_mem = (float) (s_info.totalram) / 1024 / 1024 / 1024;
        free_mem = (float) (s_info.freeram) / 1024 / 1024 / 1024;

        sprintf(info_buff, "\n [SYS-INFO ]  SYSTEM RAM USAGE: %.1f G / %.1f G\n", total_mem - free_mem, total_mem);
        printf("%s", info_buff);

    }
#ifdef MPI_USED
    }
#endif
#endif
}

// for fission spectrum
// for nu-fission, tot cross section
// for scattering matrix
char *cleanComments(char *_src_inp_name, char *_dst_inp_name, char _comment_flag) {

    char *tmp_file_name = _dst_inp_name;
    char line[MAX_TEXT_LINE];
    char line_tmp[MAX_TEXT_LINE];
    int num_lines = 0;

    FILE *usr_inp_fp = fopen(_src_inp_name, "r");
    if (usr_inp_fp == NULL) {
        printf("\n[**ERROR**]  cannot open %s", _src_inp_name);
        exit(-1);
    }

    FILE *eff_inp_fp = fopen(tmp_file_name, "w");
    if (eff_inp_fp == NULL) {
        printf("\n[**ERROR**]  cannot open %s", tmp_file_name);
        exit(-1);
    }

    while (fgets(line, MAX_TEXT_LINE, usr_inp_fp) != NULL) {
        num_lines++;
        char *a = strchr(line, _comment_flag);

        unsigned int ptr_offset;

        if (strlen(line) == strspn(line, " \t\n")) {

        } else if (NULL == a) {
            fprintf(eff_inp_fp, "%s", line);
        } else {
            ptr_offset = (a - line) / (unsigned int) sizeof(char);
            if (ptr_offset > 0) {
                strncpy(line_tmp, line, ptr_offset);
                line_tmp[ptr_offset] = '\n';
                line_tmp[ptr_offset + 1] = '\0';
                fprintf(eff_inp_fp, "%s", line_tmp);
            }
        }
    }

    fclose(usr_inp_fp);
    fclose(eff_inp_fp);

    return tmp_file_name;

}
