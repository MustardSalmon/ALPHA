# ALPHA GPU-MOC Transport Reproducibility Package

This repository provides a self-contained CUDA implementation and benchmark case for reproducing the two-dimensional GPU method-of-characteristics (MOC) transport calculations discussed in the accompanying work.

The package focuses on the high-order scattering-source calculation and its transport integration. It contains a pre-generated track file and an external FSR-wise CSR cross-section file, so the benchmark can be built and executed without running resonance reconstruction, slowing-down, group collapsing, or track generation.

The code is a transport-focused reproducibility snapshot of ALPHA,
the **Advanced Lattice Physics code based on Heterogeneous Architecture**,
developed by the ALPHA Reactor Team, Zhejiang Institute of Modern Physics,
School of Physics, Zhejiang University, Hangzhou, China.

## Software information

| Item | Information |
|---|---|
| Software | ALPHA (Advanced Lattice Physics code based on Heterogeneous Architecture) |
| Version | 0.1.0 (initial release) |
| Release date | 2026-09-17 |
| Author | CaoWei |
| Development | ALPHA Reactor Team, Zhejiang Institute of Modern Physics, School of Physics, Zhejiang University, Hangzhou, China |
| Supported platforms of ALPHA | Windows and Linux |
| Reference build environment | Linux, CMake, CUDA Toolkit, and a CUDA-compatible C/C++ compiler |
| Repository | [github.com/MustardSalmon/ALPHA](https://github.com/MustardSalmon/ALPHA) |

The present repository is a CUDA transport-only snapshot. Its provided
shell scripts and documented benchmark workflow target Linux; the full
ALPHA project may support additional platform-specific workflows. Python
3.12.3 belongs to the SMRT preprocessing-tool environment and is not a
runtime requirement of the CUDA transport executable in this repository.

## Overview

The reproducibility package supports the following workflow:

1. Configure and compile the standalone 2-D CUDA transport executable.
2. Read a pre-generated track file through `UseTrackFile=1`.
3. Read FSR-wise multigroup cross sections from an external CSR file.
4. Optionally cluster FSR cross sections during the transport calculation.
5. Select one of the four high-order scattering-source implementations.
6. Select full-angle or per-azimuth transport sweeping.
7. Record convergence, eigenvalue, clustering time, source-calculation time, transport time, total time, and wall-clock time.

The package is intended for algorithmic reproducibility and performance comparison. It is not a complete replacement for the full ALPHA resonance and multigroup cross-section-generation workflow.

## Included benchmark case

The bundled case is:

```text
case/minimal/
```

Its main characteristics are:

| Property | Value |
|---|---:|
| Geometry | 2-D transport problem |
| Energy groups | 26 |
| FSRs | 610 |
| Scattering order | P1 |
| Azimuthal directions | 48 |
| Polar quadrature order | 3 |
| Precision | FP32 |
| Track data | Pre-generated |
| Cross-section format | FSR-wise CSR |
| Resonance calculation | Disabled |
| Transport mode | CUDA/GPU |

The case is deliberately small enough for a smoke test while still exercising the complete GPU transport path, FSR clustering, high-order scattering-source calculation, transport sweeping, and convergence loop.

The four required benchmark inputs are:

```text
case/minimal/mainTsp.inp
case/minimal/FsrIdx.inp
case/minimal/GeomInfo.trk
case/minimal/fsrCgXs.csr
```

Generated output files are ignored by Git and are not required as input data.

## Access to the 2165-group benchmark

The repository includes only the minimal benchmark case described above.
The 2165-group benchmark is not included because its FSR-wise
multigroup cross-section files are too large to be uploaded to GitHub
under the repository's file-size constraints. The complete 2165-group
input data can be provided upon reasonable request. Researchers who need
these data for validation should contact Cao Wei at
`1031090394@qq.com`.

## Repository layout

```text
paper-transport-reproducibility/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── CITATION.cff
├── case/
│   └── minimal/
│       ├── mainTsp.inp
│       ├── FsrIdx.inp
│       ├── GeomInfo.trk
│       └── fsrCgXs.csr
├── scripts/
│   ├── build.sh
│   └── run_type13_16.sh
├── src/
│   ├── include/
│   └── paper/
└── build/
    └── alpha_2d                 # generated after compilation
```

The source tree under `src/` is the transport-only publication subset. It retains the external track-file and FSR-wise CSR input path required by the benchmark and excludes unrelated production workflows such as resonance reconstruction, slowing-down, 3-D transport, MPI production execution, and dynamic track generation.

## Requirements

### Software

- Linux
- CMake 3.18 or newer
- NVIDIA CUDA Toolkit with `nvcc`
- CUDA libraries required by the build, including cuBLAS and cuSPARSE
- A C/C++ compiler supported by the selected CUDA Toolkit

### Hardware

- NVIDIA GPU with CUDA support
- The default configuration targets NVIDIA RTX 4090-class hardware (`sm_89`)
- Other architectures may require changing `CMAKE_CUDA_ARCHITECTURES` in
  `CMakeLists.txt`

Before running a case, verify that the GPU is visible:

```bash
nvidia-smi
ls -l /dev/nvidia*
```

Select the GPU explicitly when running:

```bash
export CUDA_VISIBLE_DEVICES=0
```

## Build

From the package root:

```bash
cd paper-transport-reproducibility
./scripts/build.sh
```

The script configures a Release build and creates:

```text
build/alpha_2d
```

If CUDA is installed in a non-default location, specify the compiler explicitly:

```bash
CUDACXX=/path/to/cuda/bin/nvcc ./scripts/build.sh
```

The build can be parallelized with:

```bash
BUILD_JOBS=8 ./scripts/build.sh
```

A successful build should produce the executable:

```bash
test -x build/alpha_2d
```

## Input-file contract

The benchmark uses a pre-generated transport track and external FSR-wise cross sections.

### Transport input

`case/minimal/mainTsp.inp` enables the external track-file path:

```text
UseTrackFile 1
TrackFileName GeomInfo.trk
If_Resonance 0
ProblemType 0
```

The relative file name `GeomInfo.trk` is resolved inside the case directory.

### FSR-wise cross sections

`case/minimal/FsrIdx.inp` selects FSR-wise cross sections and points to the CSR file:

```text
INPUT_XSs_MATorFSR 1
Num_Group 26
IfSetXSsForEveryFSR 1
Num_of_XSsTypes_FSR 610
Num_of_ScattOrder 1
FSR_XS_File_List 1
fsrCgXs.csr
```

The group count, FSR count, and scattering order in the input cards must remain consistent with the supplied `fsrCgXs.csr` file.

## High-order scattering-source paths

The publication package exposes four paths through `GPU_TSKernelTypeOverride`:

| Type | Organization | Description |
|---:|---|---|
| 13 | LM/AP-out | LM computes the group-coupled moment source with outgoing-group expansion; AP performs the angular projection. |
| 14 | LM/AP-in | LM computes the group-coupled moment source with incoming-group expansion; AP performs the angular projection. |
| 15 | Conventional-in | Conventional fused high-order scattering-source calculation with incoming-group expansion and the corresponding reduction organization. |
| 16 | Conventional-out | Conventional fused high-order scattering-source calculation with outgoing-group expansion. |

Types 13 and 14 separate the group-coupling and angular-projection stages. Types 15 and 16 provide the conventional fused implementations used as comparison baselines.

The package intentionally supports only Types 13--16. Other paths from the full development code are outside this reproducibility package.

## Runtime controls

The following environment variables control the benchmark without recompilation.

| Environment variable | Values | Meaning |
|---|---|---|
| `GPU_TSKernelTypeOverride` | `13`--`16` | Selects the high-order scattering-source path. |
| `APKernel` | `0`--`3` | Selects the angular-projection implementation. The paper benchmark uses `3`. |
| `FsrCluster` or `FSR_CLUSTER` | `0` or `1` | Disables or enables transport-time FSR clustering. |
| `FsrClusterSseLimit` or `FSR_CLUSTER_SSE_LIMIT` | positive number | SSE threshold used by the FSR clustering procedure. The benchmark uses `100`. |
| `GPU_AzimSweepMode` | `0` or a positive integer | `0` selects full-angle sweeping; a positive value selects per-azimuth batching. `1` means one azimuth at a time. |

When using `build/alpha_2d` directly, use the program variable names in
the table. When using `scripts/run_type13_16.sh`, the script-level aliases
are `FSR_CLUSTER`, `FSR_CLUSTER_SSE_LIMIT`, `GPU_AZIM_SWEEP_MODE`, and
`AP_KERNEL`; the script translates them to the program variables.

The clustering implementation also accepts advanced controls used by the transport code, including the maximum number of clusters per material, maximum iteration count, convergence tolerance, random seed, and clustering algorithm. These should be changed only when a controlled sensitivity study is intended. For a direct reproduction of the bundled benchmark, keep the defaults used by the provided script.

For a fair comparison, keep the following fixed while changing only the scattering-source type:

- GPU device
- precision
- input track file
- FSR and group counts
- scattering order
- azimuthal and polar quadrature
- AP kernel
- clustering mode and SSE threshold
- sweep mode
- convergence criteria
- CUDA and compiler configuration

## Run one algorithm

From the package root, change to the case directory before launching the
executable so that the relative track, FSR-index, and CSR file names are
resolved correctly:

```bash
(cd case/minimal && \
 CUDA_VISIBLE_DEVICES=0 \
 FsrCluster=1 \
 FsrClusterSseLimit=100 \
 GPU_AzimSweepMode=1 \
 GPU_TSKernelTypeOverride=13 \
 APKernel=3 \
 ../../build/alpha_2d mainTsp.inp)
```

Alternatively, run from inside the case directory:

```bash
cd case/minimal

CUDA_VISIBLE_DEVICES=0 \
FsrCluster=1 \
FsrClusterSseLimit=100 \
GPU_AzimSweepMode=1 \
GPU_TSKernelTypeOverride=13 \
APKernel=3 \
../../build/alpha_2d mainTsp.inp > type13.out 2>&1
```

The executable writes its normal transport log to standard output. Redirect the output to a file when the final eigenvalue, convergence history, and timing information need to be archived.

To run a different path, change only `GPU_TSKernelTypeOverride`:

```bash
GPU_TSKernelTypeOverride=14 ../../build/alpha_2d mainTsp.inp
GPU_TSKernelTypeOverride=15 ../../build/alpha_2d mainTsp.inp
GPU_TSKernelTypeOverride=16 ../../build/alpha_2d mainTsp.inp
```

## Run all four paths

The supplied script runs Types 13--16 with the same case and runtime configuration:

```bash
./scripts/run_type13_16.sh case/minimal
```

By default, the script uses:

```text
GPU device              CUDA_VISIBLE_DEVICES=0
FSR clustering          enabled
Clustering SSE limit    100
Azimuthal sweep         one azimuth per batch
AP kernel                3
Scattering paths         13, 14, 15, 16
```

The script creates:

```text
case/minimal/type13.out
case/minimal/type14.out
case/minimal/type15.out
case/minimal/type16.out
case/minimal/summary.tsv
```

To use full-angle sweeping and disable clustering:

```bash
CUDA_VISIBLE_DEVICES=0 \
FSR_CLUSTER=0 \
GPU_AZIM_SWEEP_MODE=0 \
FSR_CLUSTER_SSE_LIMIT=100 \
AP_KERNEL=3 \
./scripts/run_type13_16.sh case/minimal
```

To select another GPU:

```bash
CUDA_VISIBLE_DEVICES=1 ./scripts/run_type13_16.sh case/minimal
```

The script does not impose a timeout. A run is allowed to finish according to the convergence criteria in the input card.

## Summary output

The generated `summary.tsv` contains one row for each scattering-source path and the following columns:

| Column | Meaning |
|---|---|
| `type` | Scattering-source path, 13--16. |
| `iteration` | Final iteration reported by the executable. |
| `keff` | Final eigenvalue. |
| `cluster_s` | Program-reported FSR clustering time. |
| `fission_s` | Program-reported fission-source time. |
| `scattering_s` | Program-reported high-order scattering-source time. |
| `lm_s` | LM-stage time when separately reported. |
| `ap_s` | AP-stage time when separately reported. |
| `transport_s` | Program-reported transport-sweep time. |
| `total_s` | Program-reported total solution time. |
| `wall_s` | Program-reported wall-clock time extracted from the run log. |

The program-level time fields include the work reported by the executable for the corresponding stage. They should be compared only when the same case, GPU, precision, convergence criteria, clustering settings, and sweep mode are used.

For an operating-system-level wall-clock measurement independent of the program's own report, wrap a single run with `/usr/bin/time`:

```bash
/usr/bin/time -f 'external_wall_s=%e' \
  env CUDA_VISIBLE_DEVICES=0 \
      FsrCluster=1 \
      FsrClusterSseLimit=100 \
      GPU_AzimSweepMode=1 \
      GPU_TSKernelTypeOverride=13 \
      APKernel=3 \
      ./build/alpha_2d case/minimal/mainTsp.inp \
      > case/minimal/type13.external.out 2>&1
```

The external wall time includes process startup, input reading, CUDA initialization, GPU execution, CPU waiting, data transfers, and output. It should not be confused with an individual GPU kernel duration.

## Reproducibility protocol

For a reproducible algorithm comparison:

1. Build the executable once using a fixed CUDA Toolkit, compiler, build type, and GPU architecture.
2. Select one physical GPU with `CUDA_VISIBLE_DEVICES`.
3. Use the same `case/minimal` input files for all four paths.
4. Keep clustering, SSE threshold, sweep mode, AP kernel, quadrature, precision, and convergence criteria unchanged.
5. Change only `GPU_TSKernelTypeOverride`.
6. Record the final `keff`, iteration count, stage times, total time, and wall time.
7. Confirm that all runs converge and that the eigenvalues agree within the expected floating-point and algorithmic tolerance.
8. Repeat a run if the GPU is concurrently occupied by another process; report the selected GPU and system load with performance results.

For performance conclusions, do not compare runs made under different GPU occupancy or different convergence iteration counts. If the GPU is shared, report that condition explicitly or repeat the measurement on an idle device.

## Generated files and version-control policy

The benchmark input files are the reproducibility data and should be kept under version control:

```text
case/minimal/mainTsp.inp
case/minimal/FsrIdx.inp
case/minimal/GeomInfo.trk
case/minimal/fsrCgXs.csr
```

The following are generated artifacts and should normally remain untracked:

- `build/`
- `case/*/*.out`
- `case/*/*.log`
- `case/*/summary.tsv`
- transport scratch directories and runtime diagnostic files

The included `.gitignore` is configured to exclude these local build and run products.

## Scope and limitations

This package is intentionally limited to the reproducible transport benchmark:

- 2-D CUDA transport
- FP32 execution
- Pre-generated track-file input
- External FSR-wise CSR cross sections
- Optional transport-time FSR clustering
- High-order scattering-source Types 13--16
- Full-angle or per-azimuth sweeping

The following workflows are not included:

- Resonance reconstruction
- Slowing-down calculations
- Energy-group collapsing
- Automatic generation of `SlowDownFlux*`
- CSR cross-section generation
- Dynamic track generation
- 3-D transport
- Production MPI execution
- Type 17--21 scattering-source paths
- General-purpose input conversion for arbitrary geometries

To reproduce a different energy-group count, geometry, track file, scattering order, or FSR count, the corresponding `mainTsp.inp`, `FsrIdx.inp`, track file, and CSR file must be supplied as a mutually consistent case. The executable does not regenerate or validate those data products.

## Reproducibility note

The package is intended to accompany the transport-method publication. The benchmark case and runtime controls document the exact data path used for the included example; numerical and performance results should be reported together with the GPU model, CUDA version, compiler version, precision, input dimensions, sweep mode, clustering settings, AP kernel, and convergence criteria.
