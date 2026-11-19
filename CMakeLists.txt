cmake_minimum_required(VERSION 3.18)

project(paper_transport_reproducibility LANGUAGES C CXX CUDA)

# This repository is intentionally a 2-D, GPU-only publication snapshot.
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architecture; RTX 4090 is sm_89")
set(CMAKE_CUDA_FLAGS "-lineinfo" CACHE STRING "CUDA flags used by Nsight Compute")

find_package(CUDAToolkit REQUIRED)

add_compile_definitions(
        CUDA_USED
        FP32
        M_2DMOC
        TESTR
        PAPER_TRANSPORT_ONLY
        SCHED_LINUX
        _CRT_SECURE_NO_WARNINGS
)

# Keep optional legacy branches from pulling unrelated resonance/CMFD code
# into the publication executable.  PAPER_TRANSPORT_ONLY is consumed by the
# CUDA dispatch/source units and by the output module to exclude experimental
# Type 17--21, CMFD, RTM and historical report branches from this snapshot.
# The supported external-XS transport path remains numerically identical to
# the retained Type 13--16 implementation.
add_compile_options(-ffunction-sections -fdata-sections)

set(PAPER_C_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/main.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper_compat.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/auxiliary.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/MOC_AllocAndFreeMem.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_Assembly.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_Boundary.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_Cell.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_Core.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_EigenValueEvaluation.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_ExpEvaluation.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_FluxEvaluation.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/MOC_GPUCPUInterdface.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/MOC_Geometry.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_Input_trackfile.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_Lattice.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_Macro_XSs_fsr.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_NTE_Solver.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_Output_transport.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_Pin_Cell.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_Point.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/MOC_Quadrature_Set.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_ReacRateEvaluation.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_Surface.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_FSRCluster.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_Track_file.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/MOC_VectorOperation.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/MOC_Visualization.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/MOC_main.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/paper_transport_compat.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/paper/paper_restored_shared.c")

set(PAPER_CUDA_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/MOC_GPUConstants.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/paper_gpu_globals.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/MOC_GPUMatVec.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/MOC_GPUQuery.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/MOC_GPUSolver.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/MOC_GPU_TSSolver.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/MOC_GPU_TSSolver_Cluster.cu")
set(PAPER_CLUSTER_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/resonance/common/general_info.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/resonance/common/interface.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/resonance/cluster/zres_cluster.c")

add_executable(alpha_2d
        ${PAPER_C_SOURCES}
        ${PAPER_CUDA_SOURCES}
        ${PAPER_CLUSTER_SOURCES})

target_include_directories(alpha_2d PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/resonance/inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/resonance/inc/gpu"
        "${CUDAToolkit_INCLUDE_DIRS}")

target_link_libraries(alpha_2d PRIVATE
        CUDA::cublas
        CUDA::cusparse
        m)

target_link_options(alpha_2d PRIVATE
        -Wl,--gc-sections
        -Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/alpha_2d.map)

set_target_properties(alpha_2d PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
