#!/bin/bash

gpuID=0
exe=${ALPHA_EXE:-../../build/alpha_2d}

# export FisRate=FisRate.out

# implement given vaccum boundary trk
# export InpBoundCond=boundary.dat
rm *.out
FsrCluster=1 GPU_AzimSweepMode=1 GPU_TSKernelTypeOverride=13 APKernel=3 CUDA_VISIBLE_DEVICES=$gpuID $exe mainTsp.inp
rm 0_0 ^_^ v_v

# rm fsrCgXs.inp
