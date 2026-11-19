#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case_dir="${1:-${repo_dir}/case/minimal}"
gpu_id="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -x "${repo_dir}/build/alpha_2d" ]]; then
    echo "Executable not found. Run scripts/build.sh first." >&2
    exit 2
fi

if [[ ! -f "${case_dir}/mainTsp.inp" || ! -f "${case_dir}/FsrIdx.inp" || \
      ! -f "${case_dir}/GeomInfo.trk" || ! -f "${case_dir}/fsrCgXs.csr" ]]; then
    echo "Case must contain mainTsp.inp, FsrIdx.inp, GeomInfo.trk and fsrCgXs.csr." >&2
    exit 3
fi

case_dir="$(cd "${case_dir}" && pwd)"
summary_file="${SUMMARY_FILE:-${case_dir}/summary.tsv}"

pushd "${case_dir}" >/dev/null
printf '%s\n' $'type\titeration\tkeff\tcluster_s\tfission_s\tscattering_s\tlm_s\tap_s\ttransport_s\ttotal_s\twall_s' >"${summary_file}"
for type in 13 14 15 16; do
    output="type${type}.out"
    echo "=== Type ${type} (GPU ${gpu_id}) ==="
    env CUDA_VISIBLE_DEVICES="${gpu_id}" \
        FsrCluster="${FSR_CLUSTER:-1}" \
        FsrClusterSseLimit="${FSR_CLUSTER_SSE_LIMIT:-100}" \
        GPU_AzimSweepMode="${GPU_AZIM_SWEEP_MODE:-1}" \
        GPU_TSKernelTypeOverride="${type}" \
        APKernel="${AP_KERNEL:-3}" \
        "${repo_dir}/build/alpha_2d" mainTsp.inp >"${output}" 2>&1
    awk '/k_eff|Total Program Wall Time|Higher-Order|Transport Sweep|Cluster/ {print}' "${output}" || true
    awk -v type="${type}" '
        /\[ITER-/ {iteration=$1; keff=$3}
        /Fission Source Time:/ {fission=$(NF-1)}
        /Scattering Source Time:/ {scattering=$(NF-1)}
        /LM Kernel Time:/ {lm=$(NF-1)}
        /AP Kernel Time:/ {ap=$(NF-1)}
        /Total Transport Calculation Time:/ {transport=$(NF-1)}
        /Total Time to Solution:/ {total=$(NF-1)}
        /Total Program Wall Time:/ {wall=$(NF-1)}
        /[Cc]luster.*[Tt]ime:/ {cluster=$(NF-1)}
        END {
            sub(/^\[ITER-/, "", iteration)
            sub(/[^0-9].*$/, "", iteration)
            if (iteration == "") iteration="-"
            if (keff == "") keff="-"
            if (fission == "") fission="-"
            if (scattering == "") scattering="-"
            if (lm == "") lm="-"
            if (ap == "") ap="-"
            if (transport == "") transport="-"
            if (total == "") total="-"
            if (wall == "") wall="-"
            if (cluster == "") cluster="-"
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
                type, iteration, keff, cluster, fission, scattering, lm, ap, \
                transport, total, wall
        }
    ' "${output}" >>"${summary_file}"
done
echo "Summary: ${summary_file}"
popd >/dev/null
