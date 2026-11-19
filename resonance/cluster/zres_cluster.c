/**
 * ============================================================
 * @File     : zres_cluster.c
 * @Author   : CaoWei
 * @Colleges : Zhejiang University
 * @Email    : 12245028@zju.edu.cn
 * @Time     : 2026-07-29 14:30
 * ============================================================
 */

#include <float.h>
#include "zres_cluster.h"
#include "zres_common_func.h"

/*
 * FSR clustering module.
 *
 * The current implementation clusters FSRs material-by-material using one
 * scalar feature: sum_g Sigma_t(fsr, g).  The legacy K-Means++ path and an
 * exact one-dimensional SSE dynamic-programming path are both available; each
 * final cluster is represented by the real FSR nearest to its center.
 */

#ifndef ZRES_FSR_CLUSTER_RESTART
    #define ZRES_FSR_CLUSTER_RESTART 10
#endif

#ifndef ZRES_FSR_CLUSTER_ELBOW_WINDOW
    #define ZRES_FSR_CLUSTER_ELBOW_WINDOW 15
#endif

#define ZRES_FSR_CLUSTER_DEFAULT_SEED 20240727u

typedef struct ZresFsrClusterInfo {
    int ready;
    int numFsr;
    int numGroup;
    int numCluster;
    int allocCluster;
    int* clusterIdOfFsr;
    int* repFsrOfFsr;
    int* repFsrList;
    int* clusterMatId;
    int* clusterSize;
    FP_RES_XS* clusterSse;
} ZRES_FSR_CLUSTER_INFO;

static ZRES_FSR_CLUSTER_INFO _zresFsrClusterInfo = {0};

typedef struct ZresFsrClusterSample {
    FP_RES_XS value;
    int fsr;
} ZRES_FSR_CLUSTER_SAMPLE;

static int zres_fsr_cluster_compare_sample(const void* left, const void* right) {
    const ZRES_FSR_CLUSTER_SAMPLE* a = (const ZRES_FSR_CLUSTER_SAMPLE*) left;
    const ZRES_FSR_CLUSTER_SAMPLE* b = (const ZRES_FSR_CLUSTER_SAMPLE*) right;

    if (a->value < b->value) return -1;
    if (a->value > b->value) return 1;
    if (a->fsr < b->fsr) return -1;
    if (a->fsr > b->fsr) return 1;
    return 0;
}

static void zres_fsr_cluster_sort_samples(FP_RES_XS* scalarValues,
                                          int* fsrList,
                                          int numSample) {
    if (numSample <= 0) {
        zres_raise_error(" [  ERROR  ]  Cannot sort an empty FSR clustering sample list\n");
    }
    if (scalarValues == NULL || fsrList == NULL) {
        zres_raise_error(" [  ERROR  ]  Null FSR clustering sample buffer before sorting\n");
    }

    ZRES_FSR_CLUSTER_SAMPLE* samples = (ZRES_FSR_CLUSTER_SAMPLE *)
            zres_malloc((size_t) numSample, sizeof (ZRES_FSR_CLUSTER_SAMPLE), RNAME(samples));
    if (samples == NULL) {
        zres_raise_error(" [  ERROR  ]  Failed to allocate FSR clustering sort buffer\n");
    }

    for (int isample = 0; isample < numSample; isample++) {
        samples[isample].value = scalarValues[isample];
        samples[isample].fsr = fsrList[isample];
    }
    qsort(samples, (size_t) numSample, sizeof (ZRES_FSR_CLUSTER_SAMPLE),
          zres_fsr_cluster_compare_sample);
    for (int isample = 0; isample < numSample; isample++) {
        scalarValues[isample] = samples[isample].value;
        fsrList[isample] = samples[isample].fsr;
    }

    zres_free(samples, (size_t) numSample);
}

/*
 * Select the active macro total cross section array.
 *
 * Slow-down mode stores resonance-group total XS in resMacro_T; otherwise the
 * transport-group array macro_T is used directly.
 */
static FP_RES_XS* zres_fsr_cluster_macro_t(ZRES_GI* gi) {
    if (gi->ifSlowDown) {
        return gi->resMacro_T;
    }
    return gi->macro_T;
}

/* Select the active macro nu-fission cross section array. */
static FP_RES_XS* zres_fsr_cluster_macro_nuf(ZRES_GI* gi) {
    if (gi->ifSlowDown) {
        return gi->resMacro_Nuf;
    }
    return gi->macro_Nuf;
}

/* Return whether a floating-point value is finite enough for clustering. */
static int zres_fsr_cluster_valid_double(double value) {
    return value == value && value <= DBL_MAX && value >= -DBL_MAX;
}

/* Advance the deterministic linear congruential generator state. */
static unsigned int zres_fsr_cluster_rng_next(unsigned int* randomState) {
    *randomState = (*randomState * 1664525u) + 1013904223u;
    return *randomState;
}

/* Draw a deterministic uniform value in [0, 1). */
static double zres_fsr_cluster_rng_uniform(unsigned int* randomState) {
    return ((double) zres_fsr_cluster_rng_next(randomState) + 0.5) / 4294967296.0;
}

/* Mix user seed with material id, K, and restart id for reproducible K-Means++. */
static unsigned int zres_fsr_cluster_mix_seed(unsigned int randomSeed,
                                              int imat,
                                              int localK,
                                              int restartId) {
    unsigned int seed = randomSeed;

    seed ^= (unsigned int) (imat + 1) * 2246822519u;
    seed ^= (unsigned int) (localK + 1) * 3266489917u;
    seed ^= (unsigned int) (restartId + 1) * 668265263u;
    seed ^= seed >> 13;
    seed *= 1274126177u;
    seed ^= seed >> 16;

    return seed;
}

/*
 * Build the only clustering feature used here:
 *
 *     scalarValue(fsr) = sum_g Sigma_t(fsr, g)
 *
 * This follows the Python scripts: no log scaling, no material normalization,
 * no volume weighting, and no multigroup validation.
 */
static void zres_fsr_cluster_build_sum_t(const FP_RES_XS* macroT,
                                         const int* fsrList,
                                         int matFsrNum,
                                         int numGroup,
                                         FP_RES_XS* scalarValues) {
    char errseq[1024];

    for (int isample = 0; isample < matFsrNum; isample++) {
        int ifsr = fsrList[isample];
        double sumXs = 0.;

        for (int igrp = 0; igrp < numGroup; igrp++) {
            double xs = (double) macroT[ifsr * numGroup + igrp];

            if (!zres_fsr_cluster_valid_double(xs)) {
                sprintf(errseq, " [  ERROR  ]  Invalid macro total xs in FSR clustering: fsr=%d, group=%d\n",
                        ifsr, igrp);
                zres_raise_error(errseq);
            }

            sumXs += xs;
        }

        if (!zres_fsr_cluster_valid_double(sumXs)) {
            sprintf(errseq, " [  ERROR  ]  Invalid sumT in FSR clustering: fsr=%d\n", ifsr);
            zres_raise_error(errseq);
        }

        scalarValues[isample] = (FP_RES_XS) sumXs;
    }
}

/* Test whether a sample was already chosen as an initial center. */
static int zres_fsr_cluster_is_selected(const int* selectedSamples,
                                        int selectedNum,
                                        int sampleId) {
    for (int isel = 0; isel < selectedNum; isel++) {
        if (selectedSamples[isel] == sampleId) {
            return 1;
        }
    }

    return 0;
}

/*
 * Pick one unselected sample uniformly.
 *
 * This is used as a fallback when K-Means++ scores collapse to zero, which can
 * happen when all remaining scalar values are identical.
 */
static int zres_fsr_cluster_pick_random_unselected(int numSample,
                                                   const int* selectedSamples,
                                                   int selectedNum,
                                                   unsigned int* randomState) {
    int remainNum = numSample - selectedNum;

    if (remainNum <= 0) {
        return 0;
    }

    int target = (int) (zres_fsr_cluster_rng_uniform(randomState) * (double) remainNum);
    if (target >= remainNum) {
        target = remainNum - 1;
    }

    for (int isample = 0; isample < numSample; isample++) {
        if (zres_fsr_cluster_is_selected(selectedSamples, selectedNum, isample)) {
            continue;
        }
        if (target == 0) {
            return isample;
        }
        target -= 1;
    }

    return numSample - 1;
}

/* Initialize 1D K-Means centers using the K-Means++ distance-squared rule. */
static void zres_fsr_cluster_init_kmeanspp(const FP_RES_XS* scalarValues,
                                           int numSample,
                                           int localK,
                                           FP_RES_XS* centers,
                                           unsigned int randomSeed) {
    int* selectedSamples = (int *) zres_malloc((size_t) localK, sizeof (int), RNAME(selectedSamples));
    unsigned int randomState = randomSeed;

    for (int ic = 0; ic < localK; ic++) {
        selectedSamples[ic] = -1;
    }

    selectedSamples[0] = zres_fsr_cluster_pick_random_unselected(numSample, selectedSamples, 0, &randomState);
    centers[0] = scalarValues[selectedSamples[0]];

    for (int ic = 1; ic < localK; ic++) {
        double totalScore = 0.;

        for (int isample = 0; isample < numSample; isample++) {
            if (zres_fsr_cluster_is_selected(selectedSamples, ic, isample)) {
                continue;
            }

            double minDist = DBL_MAX;
            for (int jc = 0; jc < ic; jc++) {
                double diff = (double) scalarValues[isample] - (double) centers[jc];
                double dist = diff * diff;

                if (dist < minDist) {
                    minDist = dist;
                }
            }

            totalScore += minDist;
        }

        if (totalScore <= 0.) {
            selectedSamples[ic] = zres_fsr_cluster_pick_random_unselected(numSample, selectedSamples, ic, &randomState);
            centers[ic] = scalarValues[selectedSamples[ic]];
            continue;
        }

        double target = zres_fsr_cluster_rng_uniform(&randomState) * totalScore;
        double accumulated = 0.;
        int selected = -1;

        for (int isample = 0; isample < numSample; isample++) {
            if (zres_fsr_cluster_is_selected(selectedSamples, ic, isample)) {
                continue;
            }

            double minDist = DBL_MAX;
            for (int jc = 0; jc < ic; jc++) {
                double diff = (double) scalarValues[isample] - (double) centers[jc];
                double dist = diff * diff;

                if (dist < minDist) {
                    minDist = dist;
                }
            }

            accumulated += minDist;
            if (accumulated >= target) {
                selected = isample;
                break;
            }
        }

        if (selected < 0) {
            selected = zres_fsr_cluster_pick_random_unselected(numSample, selectedSamples, ic, &randomState);
        }

        selectedSamples[ic] = selected;
        centers[ic] = scalarValues[selected];
    }

    zres_free(selectedSamples, (size_t) localK);
}

/*
 * Assign each scalar sample to the nearest 1D center.
 *
 * Returns the number of labels changed in this assignment pass and updates
 * counts[ic] for every local cluster.
 */
static int zres_fsr_cluster_assign_samples(const FP_RES_XS* scalarValues,
                                           int numSample,
                                           int localK,
                                           const FP_RES_XS* centers,
                                           int* labels,
                                           int* counts) {
    int changed = 0;

    memset(counts, 0, sizeof (int) * (size_t) localK);

    for (int isample = 0; isample < numSample; isample++) {
        int bestCluster = 0;
        double bestDist = DBL_MAX;

        for (int ic = 0; ic < localK; ic++) {
            double diff = (double) scalarValues[isample] - (double) centers[ic];
            double dist = diff * diff;

            if (dist < bestDist) {
                bestDist = dist;
                bestCluster = ic;
            }
        }

        if (labels[isample] != bestCluster) {
            labels[isample] = bestCluster;
            changed += 1;
        }
        counts[bestCluster] += 1;
    }

    return changed;
}

/*
 * Find the sample farthest from its currently assigned center.
 *
 * Empty clusters are reinitialized to this farthest sample to avoid losing a
 * cluster during K-Means updates.
 */
static int zres_fsr_cluster_find_farthest(const FP_RES_XS* scalarValues,
                                          const FP_RES_XS* centers,
                                          const int* labels,
                                          int numSample) {
    int farthest = 0;
    double maxDist = -1.;

    for (int isample = 0; isample < numSample; isample++) {
        int label = labels[isample];
        double dist = 0.;

        if (label >= 0) {
            double diff = (double) scalarValues[isample] - (double) centers[label];
            dist = diff * diff;
        }

        if (dist > maxDist) {
            maxDist = dist;
            farthest = isample;
        }
    }

    return farthest;
}

/*
 * Recompute centers as arithmetic means.
 *
 * Non-empty centers move to the mean of their assigned samples.  Empty centers
 * are reset to a far sample and reported through hadEmptyCluster.
 */
static double zres_fsr_cluster_update_centers(const FP_RES_XS* scalarValues,
                                              const int* labels,
                                              const int* counts,
                                              int numSample,
                                              int localK,
                                              FP_RES_XS* centers,
                                              FP_RES_XS* newCenters,
                                              int* hadEmptyCluster) {
    double centerMove = 0.;

    memset(newCenters, 0, sizeof (FP_RES_XS) * (size_t) localK);
    *hadEmptyCluster = 0;

    for (int isample = 0; isample < numSample; isample++) {
        int label = labels[isample];
        newCenters[label] += scalarValues[isample];
    }

    for (int ic = 0; ic < localK; ic++) {
        if (counts[ic] > 0) {
            newCenters[ic] /= (FP_RES_XS) counts[ic];
        }
        else {
            int farthest = zres_fsr_cluster_find_farthest(scalarValues, centers, labels, numSample);
            newCenters[ic] = scalarValues[farthest];
            *hadEmptyCluster = 1;
        }

        double diff = (double) newCenters[ic] - (double) centers[ic];
        centerMove += diff * diff;
        centers[ic] = newCenters[ic];
    }

    return centerMove;
}

/* Evaluate the K-Means inertia, i.e. SSE to arithmetic cluster centers. */
static FP_RES_XS zres_fsr_cluster_eval_center_sse(const FP_RES_XS* scalarValues,
                                                  const FP_RES_XS* centers,
                                                  const int* labels,
                                                  int numSample) {
    double sse = 0.;

    for (int isample = 0; isample < numSample; isample++) {
        int label = labels[isample];
        double diff = (double) scalarValues[isample] - (double) centers[label];

        sse += diff * diff;
    }

    return (FP_RES_XS) sse;
}

/*
 * Choose one real FSR per non-empty cluster.
 *
 * K-Means centers are arithmetic means and may not correspond to real FSRs.
 * This function maps each center to the closest actual sample so later code
 * can reuse an existing FSR ID as the representative.
 */
static void zres_fsr_cluster_choose_representatives(const FP_RES_XS* scalarValues,
                                                    const FP_RES_XS* centers,
                                                    const int* labels,
                                                    const int* counts,
                                                    int numSample,
                                                    int localK,
                                                    int* representatives) {
    char errseq[1024];

    for (int ic = 0; ic < localK; ic++) {
        representatives[ic] = -1;

        if (counts[ic] == 0) {
            continue;
        }

        double bestDist = DBL_MAX;
        for (int isample = 0; isample < numSample; isample++) {
            if (labels[isample] != ic) {
                continue;
            }

            double diff = (double) scalarValues[isample] - (double) centers[ic];
            double dist = diff * diff;

            if (dist < bestDist) {
                bestDist = dist;
                representatives[ic] = isample;
            }
        }

        if (representatives[ic] < 0) {
            sprintf(errseq, " [  ERROR  ]  Invalid representative in FSR scalar clustering: localCluster=%d, count=%d\n",
                    ic, counts[ic]);
            zres_raise_error(errseq);
        }
    }
}

/* SSE after replacing every FSR by its nearest real representative FSR. */
static FP_RES_XS zres_fsr_cluster_eval_rep_sse(const FP_RES_XS* scalarValues,
                                               const FP_RES_XS* centers,
                                               const int* labels,
                                               const int* counts,
                                               int numSample,
                                               int localK) {
    int* representatives = (int *) zres_malloc((size_t) localK, sizeof (int), RNAME(representatives));
    double sse = 0.;

    zres_fsr_cluster_choose_representatives(scalarValues, centers, labels, counts, numSample, localK, representatives);

    for (int isample = 0; isample < numSample; isample++) {
        int label = labels[isample];
        int rep = representatives[label];
        double diff = (double) scalarValues[isample] - (double) scalarValues[rep];

        sse += diff * diff;
    }

    zres_free(representatives, (size_t) localK);

    return (FP_RES_XS) sse;
}

/* Run one K-Means++ initialized 1D K-Means solve for a fixed K and seed. */
static FP_RES_XS zres_fsr_cluster_run_kmeans_once(const FP_RES_XS* scalarValues,
                                                  int numSample,
                                                  int localK,
                                                  int maxIter,
                                                  double tolerance2,
                                                  unsigned int randomSeed,
                                                  FP_RES_XS* centers,
                                                  int* labels,
                                                  int* counts,
                                                  FP_RES_XS* repSse) {
    FP_RES_XS* newCenters = (FP_RES_XS *) zres_malloc((size_t) localK, sizeof (FP_RES_XS), RNAME(newCenters));

    for (int isample = 0; isample < numSample; isample++) {
        labels[isample] = -1;
    }

    zres_fsr_cluster_init_kmeanspp(scalarValues, numSample, localK, centers, randomSeed);

    for (int iter = 0; iter < maxIter; iter++) {
        int hadEmptyCluster = 0;
        int changed = zres_fsr_cluster_assign_samples(scalarValues, numSample, localK, centers, labels, counts);
        double centerMove = zres_fsr_cluster_update_centers(scalarValues, labels, counts, numSample, localK,
                                                           centers, newCenters, &hadEmptyCluster);

        if (!hadEmptyCluster && (changed == 0 || centerMove <= tolerance2)) {
            break;
        }
    }

    zres_fsr_cluster_assign_samples(scalarValues, numSample, localK, centers, labels, counts);

    FP_RES_XS centerSse = zres_fsr_cluster_eval_center_sse(scalarValues, centers, labels, numSample);
    if (repSse != NULL) {
        *repSse = zres_fsr_cluster_eval_rep_sse(scalarValues, centers, labels, counts, numSample, localK);
    }

    zres_free(newCenters, (size_t) localK);

    return centerSse;
}

/*
 * Exact one-dimensional SSE DP state.
 *
 * The samples must be sorted.  A cluster is therefore a contiguous interval
 * in this order.  The divide-and-conquer row optimization uses the Monge
 * property of the squared-error interval cost.  Only the last DP row is kept;
 * split rows are retained for reconstruction of the selected K.
 */
typedef struct ZresFsrClusterDpState {
    int numSample;
    int maxK;
    int builtK;
    double* prefix1;
    double* prefix2;
    double* previous;
    double* current;
    double* sse;
    int** split;
} ZRES_FSR_CLUSTER_DP_STATE;

static double zres_fsr_cluster_interval_sse(const double* prefix1,
                                            const double* prefix2,
                                            int begin,
                                            int end) {
    int count = end - begin;
    double sum = prefix1[end] - prefix1[begin];
    double sum2 = prefix2[end] - prefix2[begin];
    double value;

    if (count <= 1) {
        return 0.;
    }

    value = sum2 - sum * sum / (double) count;
    /* Prefix subtraction can produce a tiny negative value in FP32 cases. */
    return value > 0. ? value : 0.;
}

static void zres_fsr_cluster_dp_compute_row(const double* prefix1,
                                            const double* prefix2,
                                            const double* previous,
                                            double* current,
                                            int* splitRow,
                                            int clusterNum,
                                            int leftEnd,
                                            int rightEnd,
                                            int leftOpt,
                                            int rightOpt) {
    if (leftEnd > rightEnd) {
        return;
    }

    int midEnd = leftEnd + (rightEnd - leftEnd) / 2;
    int minSplit = clusterNum - 1 > leftOpt ? clusterNum - 1 : leftOpt;
    int maxSplit = midEnd - 1 < rightOpt ? midEnd - 1 : rightOpt;
    int bestSplit = minSplit;
    double bestValue = DBL_MAX;

    for (int split = minSplit; split <= maxSplit; split++) {
        double candidate = previous[split] +
                           zres_fsr_cluster_interval_sse(prefix1, prefix2, split, midEnd);
        if (candidate < bestValue) {
            bestValue = candidate;
            bestSplit = split;
        }
    }

    current[midEnd] = bestValue;
    splitRow[midEnd] = bestSplit;

    zres_fsr_cluster_dp_compute_row(prefix1, prefix2, previous, current, splitRow,
                                     clusterNum, leftEnd, midEnd - 1,
                                     leftOpt, bestSplit);
    zres_fsr_cluster_dp_compute_row(prefix1, prefix2, previous, current, splitRow,
                                     clusterNum, midEnd + 1, rightEnd,
                                     bestSplit, rightOpt);
}

static void zres_fsr_cluster_dp_init(ZRES_FSR_CLUSTER_DP_STATE* state,
                                     const FP_RES_XS* scalarValues,
                                     int numSample,
                                     int maxK) {
    memset(state, 0, sizeof (*state));
    state->numSample = numSample;
    state->maxK = maxK;
    state->prefix1 = (double*) zres_malloc((size_t) (numSample + 1), sizeof (double), RNAME(prefix1));
    state->prefix2 = (double*) zres_malloc((size_t) (numSample + 1), sizeof (double), RNAME(prefix2));
    state->previous = (double*) zres_malloc((size_t) (numSample + 1), sizeof (double), RNAME(previous));
    state->current = (double*) zres_malloc((size_t) (numSample + 1), sizeof (double), RNAME(current));
    state->sse = (double*) zres_malloc((size_t) (maxK + 1), sizeof (double), RNAME(sse));
    state->split = (int**) zres_malloc((size_t) (maxK + 1), sizeof (int*), RNAME(split));
    memset(state->split, 0, sizeof (int*) * (size_t) (maxK + 1));

    state->prefix1[0] = 0.;
    state->prefix2[0] = 0.;
    for (int isample = 0; isample < numSample; isample++) {
        double value = (double) scalarValues[isample];
        state->prefix1[isample + 1] = state->prefix1[isample] + value;
        state->prefix2[isample + 1] = state->prefix2[isample] + value * value;
    }

    state->previous[0] = 0.;
    for (int end = 1; end <= numSample; end++) {
        state->previous[end] = DBL_MAX;
    }
}

static void zres_fsr_cluster_dp_build_next(ZRES_FSR_CLUSTER_DP_STATE* state) {
    int clusterNum = state->builtK + 1;
    int* splitRow;

    if (clusterNum > state->maxK) {
        zres_raise_error(" [  ERROR  ]  SSE DP requested K beyond allocated maximum\n");
    }

    splitRow = (int*) zres_malloc((size_t) (state->numSample + 1), sizeof (int), RNAME(splitRow));
    for (int end = 0; end <= state->numSample; end++) {
        splitRow[end] = -1;
        state->current[end] = DBL_MAX;
    }

    if (clusterNum == 1) {
        splitRow[0] = -1;
        for (int end = 1; end <= state->numSample; end++) {
            state->current[end] = zres_fsr_cluster_interval_sse(
                    state->prefix1, state->prefix2, 0, end);
            splitRow[end] = 0;
        }
    }
    else {
        zres_fsr_cluster_dp_compute_row(state->prefix1, state->prefix2,
                                        state->previous, state->current, splitRow,
                                        clusterNum, clusterNum, state->numSample,
                                        clusterNum - 1, state->numSample - 1);
    }

    state->split[clusterNum] = splitRow;
    state->sse[clusterNum] = state->current[state->numSample];
    {
        double* swap = state->previous;
        state->previous = state->current;
        state->current = swap;
    }
    state->builtK = clusterNum;
}

static void zres_fsr_cluster_dp_destroy(ZRES_FSR_CLUSTER_DP_STATE* state) {
    if (state->split != NULL) {
        for (int clusterNum = 0; clusterNum <= state->builtK; clusterNum++) {
            if (state->split[clusterNum] != NULL) {
                zres_free(state->split[clusterNum], (size_t) (state->numSample + 1));
            }
        }
        zres_free(state->split, (size_t) (state->maxK + 1));
    }
    if (state->prefix1 != NULL) zres_free(state->prefix1, (size_t) (state->numSample + 1));
    if (state->prefix2 != NULL) zres_free(state->prefix2, (size_t) (state->numSample + 1));
    if (state->previous != NULL) zres_free(state->previous, (size_t) (state->numSample + 1));
    if (state->current != NULL) zres_free(state->current, (size_t) (state->numSample + 1));
    if (state->sse != NULL) zres_free(state->sse, (size_t) (state->maxK + 1));
    memset(state, 0, sizeof (*state));
}

static void zres_fsr_cluster_dp_reconstruct(const ZRES_FSR_CLUSTER_DP_STATE* state,
                                            const FP_RES_XS* scalarValues,
                                            int localK,
                                            FP_RES_XS* centers,
                                            int* labels,
                                            int* counts,
                                            FP_RES_XS* repSse) {
    int end = state->numSample;

    if (localK <= 0 || localK > state->builtK) {
        zres_raise_error(" [  ERROR  ]  Invalid SSE DP reconstruction K\n");
    }

    for (int clusterNum = localK; clusterNum >= 1; clusterNum--) {
        int begin = state->split[clusterNum][end];
        if (begin < clusterNum - 1 || begin >= end) {
            zres_raise_error(" [  ERROR  ]  Invalid SSE DP split during reconstruction\n");
        }

        counts[clusterNum - 1] = end - begin;
        centers[clusterNum - 1] = (FP_RES_XS) (
                (state->prefix1[end] - state->prefix1[begin]) / (double) counts[clusterNum - 1]);
        for (int isample = begin; isample < end; isample++) {
            labels[isample] = clusterNum - 1;
        }
        end = begin;
    }

    if (end != 0) {
        zres_raise_error(" [  ERROR  ]  SSE DP reconstruction did not cover all samples\n");
    }
    if (repSse != NULL) {
        *repSse = zres_fsr_cluster_eval_rep_sse(
                scalarValues, centers, labels, counts, state->numSample, localK);
    }
}

/*
 * Run K-Means++ several times.
 *
 * The average center SSE follows sklearn's averaged inertia_ in the Python
 * script and is used for K selection.  When output buffers are provided, the
 * lowest-center-SSE run is kept for final representatives.
 */
static FP_RES_XS zres_fsr_cluster_run_kmeans_repeated(const FP_RES_XS* scalarValues,
                                                      int numSample,
                                                      int localK,
                                                      int maxIter,
                                                      double tolerance2,
                                                      int restartNum,
                                                      unsigned int randomSeed,
                                                      int imat,
                                                      FP_RES_XS* centers,
                                                      int* labels,
                                                      int* counts,
                                                      FP_RES_XS* bestCenterSse,
                                                      FP_RES_XS* bestRepSse) {
    FP_RES_XS* trialCenters = (FP_RES_XS *) zres_malloc((size_t) localK, sizeof (FP_RES_XS), RNAME(trialCenters));
    int* trialLabels = (int *) zres_malloc((size_t) numSample, sizeof (int), RNAME(trialLabels));
    int* trialCounts = (int *) zres_malloc((size_t) localK, sizeof (int), RNAME(trialCounts));
    double avgSse = 0.;
    double bestSse = DBL_MAX;
    FP_RES_XS bestRep = (FP_RES_XS) 0.;

    for (int irestart = 0; irestart < restartNum; irestart++) {
        unsigned int seed = zres_fsr_cluster_mix_seed(randomSeed, imat, localK, irestart);
        FP_RES_XS trialRepSse = (FP_RES_XS) 0.;
        FP_RES_XS* trialRepSsePtr = bestRepSse != NULL ? &trialRepSse : NULL;
        FP_RES_XS trialCenterSse = zres_fsr_cluster_run_kmeans_once(scalarValues, numSample, localK,
                                                                    maxIter, tolerance2, seed,
                                                                    trialCenters, trialLabels, trialCounts,
                                                                    trialRepSsePtr);

        avgSse += (double) trialCenterSse;

        if ((double) trialCenterSse < bestSse) {
            bestSse = (double) trialCenterSse;
            if (bestRepSse != NULL) {
                bestRep = trialRepSse;
            }
            if (centers != NULL) {
                memcpy(centers, trialCenters, sizeof (FP_RES_XS) * (size_t) localK);
            }
            if (labels != NULL) {
                memcpy(labels, trialLabels, sizeof (int) * (size_t) numSample);
            }
            if (counts != NULL) {
                memcpy(counts, trialCounts, sizeof (int) * (size_t) localK);
            }
        }
    }

    zres_free(trialCenters, (size_t) localK);
    zres_free(trialLabels, (size_t) numSample);
    zres_free(trialCounts, (size_t) localK);

    if (bestCenterSse != NULL) {
        *bestCenterSse = (FP_RES_XS) bestSse;
    }
    if (bestRepSse != NULL) {
        *bestRepSse = bestRep;
    }

    return (FP_RES_XS) (avgSse / (double) restartNum);
}

static FP_RES_XS zres_fsr_cluster_run_algorithm(const FP_RES_XS* scalarValues,
                                                int numSample,
                                                int localK,
                                                int maxIter,
                                                double tolerance2,
                                                int restartNum,
                                                unsigned int randomSeed,
                                                int imat,
                                                int sorted1d,
                                                FP_RES_XS* centers,
                                                int* labels,
                                                int* counts,
                                                FP_RES_XS* bestCenterSse,
                                                FP_RES_XS* bestRepSse) {
    /* sorted1d is handled by the exact DP path in the main driver. */
    (void) sorted1d;
    return zres_fsr_cluster_run_kmeans_repeated(
            scalarValues, numSample, localK, maxIter, tolerance2,
            restartNum, randomSeed, imat, centers, labels, counts,
            bestCenterSse, bestRepSse);
}

/* Commit one singleton cluster per FSR for a material. */
static int zres_fsr_cluster_commit_singletons(int imat,
                                              const int* fsrList,
                                              int matFsrNum,
                                              int globalCluster) {
    for (int isample = 0; isample < matFsrNum; isample++) {
        int ifsr = fsrList[isample];
        int cid = globalCluster;

        _zresFsrClusterInfo.clusterIdOfFsr[ifsr] = cid;
        _zresFsrClusterInfo.repFsrOfFsr[ifsr] = ifsr;
        _zresFsrClusterInfo.repFsrList[cid] = ifsr;
        _zresFsrClusterInfo.clusterMatId[cid] = imat;
        _zresFsrClusterInfo.clusterSize[cid] = 1;
        _zresFsrClusterInfo.clusterSse[cid] = (FP_RES_XS) 0.;
        globalCluster += 1;
    }

    return globalCluster;
}

/* Commit labels and store representative-FSR scalar SSE for each cluster. */
static int zres_fsr_cluster_commit_kmeans(int imat,
                                          const int* fsrList,
                                          int matFsrNum,
                                          int localK,
                                          const FP_RES_XS* scalarValues,
                                          const FP_RES_XS* centers,
                                          const int* labels,
                                          const int* counts,
                                          int globalCluster) {
    int* localClusterId = (int *) zres_malloc((size_t) localK, sizeof (int), RNAME(localClusterId));
    int* representatives = (int *) zres_malloc((size_t) localK, sizeof (int), RNAME(representatives));

    zres_fsr_cluster_choose_representatives(scalarValues, centers, labels, counts, matFsrNum, localK, representatives);

    for (int ic = 0; ic < localK; ic++) {
        localClusterId[ic] = -1;

        if (counts[ic] == 0) {
            continue;
        }

        int cid = globalCluster;
        int repFsr = fsrList[representatives[ic]];

        localClusterId[ic] = cid;
        _zresFsrClusterInfo.repFsrList[cid] = repFsr;
        _zresFsrClusterInfo.clusterMatId[cid] = imat;
        _zresFsrClusterInfo.clusterSize[cid] = 0;
        _zresFsrClusterInfo.clusterSse[cid] = (FP_RES_XS) 0.;
        globalCluster += 1;
    }

    for (int isample = 0; isample < matFsrNum; isample++) {
        int label = labels[isample];
        int cid = localClusterId[label];
        int ifsr = fsrList[isample];
        int repLocal = representatives[label];

        if (cid < 0 || repLocal < 0) {
            zres_raise_error(" [  ERROR  ]  Invalid empty cluster label in FSR scalar clustering\n");
        }

        _zresFsrClusterInfo.clusterIdOfFsr[ifsr] = cid;
        _zresFsrClusterInfo.repFsrOfFsr[ifsr] = _zresFsrClusterInfo.repFsrList[cid];
        _zresFsrClusterInfo.clusterSize[cid] += 1;

        double diff = (double) scalarValues[isample] - (double) scalarValues[repLocal];
        _zresFsrClusterInfo.clusterSse[cid] += (FP_RES_XS) (diff * diff);
    }

    zres_free(localClusterId, (size_t) localK);
    zres_free(representatives, (size_t) localK);

    return globalCluster;
}

/*
 * Release the cached cluster map and reset module state.
 *
 * After this call, representative FSR IDs returned by the getter arrays are no
 * longer valid.
 */
void zres_free_fsr_cluster_info() {
    if (_zresFsrClusterInfo.clusterIdOfFsr != NULL) {
        zres_free(_zresFsrClusterInfo.clusterIdOfFsr, (size_t) _zresFsrClusterInfo.numFsr);
    }
    if (_zresFsrClusterInfo.repFsrOfFsr != NULL) {
        zres_free(_zresFsrClusterInfo.repFsrOfFsr, (size_t) _zresFsrClusterInfo.numFsr);
    }
    if (_zresFsrClusterInfo.repFsrList != NULL) {
        zres_free(_zresFsrClusterInfo.repFsrList, (size_t) _zresFsrClusterInfo.allocCluster);
    }
    if (_zresFsrClusterInfo.clusterMatId != NULL) {
        zres_free(_zresFsrClusterInfo.clusterMatId, (size_t) _zresFsrClusterInfo.allocCluster);
    }
    if (_zresFsrClusterInfo.clusterSize != NULL) {
        zres_free(_zresFsrClusterInfo.clusterSize, (size_t) _zresFsrClusterInfo.allocCluster);
    }
    if (_zresFsrClusterInfo.clusterSse != NULL) {
        zres_free(_zresFsrClusterInfo.clusterSse, (size_t) _zresFsrClusterInfo.allocCluster);
    }

    memset(&_zresFsrClusterInfo, 0, sizeof (ZRES_FSR_CLUSTER_INFO));
}

/* Return whether a cluster map has been built and cached. */
int zres_is_fsr_cluster_ready() {
    return _zresFsrClusterInfo.ready;
}





/*
 * Return the representative FSR for one FSR.
 *
 * If clustering has not run, the original FSR id is returned so callers can use
 * this function without a separate feature-flag branch.
 */
int zres_get_fsr_cluster_rep_fsr(int ifsr) {
    if (!_zresFsrClusterInfo.ready) {
        return ifsr;
    }
    if (ifsr < 0 || ifsr >= _zresFsrClusterInfo.numFsr) {
        return -1;
    }
    return _zresFsrClusterInfo.repFsrOfFsr[ifsr];
}


/*
 * Main driver.
 *
 * For each material:
 *   1. Build sumT(fsr) = sum_g Sigma_t(fsr,g).
 *   2. Increase K until center SSE <= sseLimit.  The sorted1d mode uses
 *      exact one-dimensional SSE DP and therefore has a monotone SSE curve.
 *   3. If the first accepted K is larger than 2, scan
 *      [K, K + ZRES_FSR_CLUSTER_ELBOW_WINDOW) and select the largest
 *      absolute second-difference point as the elbow K.
 *   4. Cluster once more with this K and choose the real FSR nearest to each
 *      final center as the representative.
 */
static int zres_cluster_fsr_by_macro_t_core(int autoBySse,
                                            int clusterPerMat,
                                            FP_RES_XS sseLimit,
                                            int maxClusterPerMat,
                                            int maxIter,
                                            FP_RES_XS tolerance,
                                            int restartNum,
                                            unsigned int randomSeed,
                                            int materialNumOverride) {
    ZRES_GI* gi = zres_get_gen_info();
    ZRES_MTC* mtc = zres_get_local_mat();
    FP_RES_XS* macroT = NULL;

    if (gi == NULL || (mtc == NULL && materialNumOverride <= 0)) {
        zres_raise_error(" [  ERROR  ]  FSR clustering requires initialized general/material info\n");
    }

    macroT = zres_fsr_cluster_macro_t(gi);
    if (macroT == NULL || gi->fsr2matMap == NULL) {
        zres_raise_error(" [  ERROR  ]  FSR clustering requires macro total xs and fsr2mat map\n");
    }

    if (!autoBySse && clusterPerMat <= 0) {
        zres_raise_error(" [  ERROR  ]  clusterPerMat must be positive in fixed-K FSR clustering\n");
    }
    if (autoBySse && maxClusterPerMat <= 0) {
        zres_raise_error(" [  ERROR  ]  maxClusterPerMat must be positive in FSR clustering\n");
    }
    if (autoBySse && sseLimit < (FP_RES_XS) 0.) {
        zres_raise_error(" [  ERROR  ]  sseLimit must be non-negative in FSR clustering\n");
    }
    if (maxIter <= 0) {
        maxIter = 100;
    }
    if (tolerance <= (FP_RES_XS) 0.) {
        tolerance = (FP_RES_XS) 1e-8;
    }
    if (restartNum <= 0) {
        restartNum = ZRES_FSR_CLUSTER_RESTART;
    }

    zres_free_fsr_cluster_info();

    int numFsr = gi->numFsr;
    int numGroup = gi->resGrpNum;
    int numMat = materialNumOverride > 0 ? materialNumOverride : mtc->matNum;
    int* fsr2matMap = gi->fsr2matMap;
    int clusterLimit = autoBySse ? maxClusterPerMat : clusterPerMat;
    int sorted1d = gi->fsrClusterAlgorithm == 1;
    int maxCluster = 0;

    for (int imat = 0; imat < numMat; imat++) {
        int matFsrNum = 0;

        for (int ifsr = 0; ifsr < numFsr; ifsr++) {
            if (fsr2matMap[ifsr] == imat) {
                matFsrNum += 1;
            }
        }

        if (matFsrNum > 0) {
            maxCluster += matFsrNum < clusterLimit ? matFsrNum : clusterLimit;
        }
    }

    if (maxCluster == 0) {
        zres_raise_error(" [  ERROR  ]  No active FSR found in FSR clustering\n");
    }

    _zresFsrClusterInfo.numFsr = numFsr;
    _zresFsrClusterInfo.numGroup = numGroup;
    _zresFsrClusterInfo.numCluster = 0;
    _zresFsrClusterInfo.allocCluster = maxCluster;
    _zresFsrClusterInfo.clusterIdOfFsr = (int *) zres_malloc((size_t) numFsr, sizeof (int), RNAME(_zresFsrClusterInfo.clusterIdOfFsr));
    _zresFsrClusterInfo.repFsrOfFsr = (int *) zres_malloc((size_t) numFsr, sizeof (int), RNAME(_zresFsrClusterInfo.repFsrOfFsr));
    _zresFsrClusterInfo.repFsrList = (int *) zres_malloc((size_t) maxCluster, sizeof (int), RNAME(_zresFsrClusterInfo.repFsrList));
    _zresFsrClusterInfo.clusterMatId = (int *) zres_malloc((size_t) maxCluster, sizeof (int), RNAME(_zresFsrClusterInfo.clusterMatId));
    _zresFsrClusterInfo.clusterSize = (int *) zres_malloc((size_t) maxCluster, sizeof (int), RNAME(_zresFsrClusterInfo.clusterSize));
    _zresFsrClusterInfo.clusterSse = (FP_RES_XS *) zres_malloc((size_t) maxCluster, sizeof (FP_RES_XS), RNAME(_zresFsrClusterInfo.clusterSse));

    for (int ifsr = 0; ifsr < numFsr; ifsr++) {
        _zresFsrClusterInfo.clusterIdOfFsr[ifsr] = -1;
        _zresFsrClusterInfo.repFsrOfFsr[ifsr] = ifsr;
    }

    printf(" [  CLUST  ]  Start FSR sumT clustering by material: numFsr=%d, numGroup=%d, numMat=%d\n",
           numFsr, numGroup, numMat);
    printf(" [  CLUST  ]  Feature=sum_g(Sigma_t); algorithm=%s, restart=%d, elbowWindow=%d, seed=%u\n",
           sorted1d ? "sse-dp1d" : "kmeans++", sorted1d ? 0 : restartNum,
           ZRES_FSR_CLUSTER_ELBOW_WINDOW, randomSeed);
    if (autoBySse) {
        printf(" [  CLUST  ]  Initial-K criterion: average center SSE <= %.6e\n", (double) sseLimit);
    }
    fflush(stdout);

    int globalCluster = 0;
    double tolerance2 = (double) tolerance * (double) tolerance;

    for (int imat = 0; imat < numMat; imat++) {
        int matFsrNum = 0;

        for (int ifsr = 0; ifsr < numFsr; ifsr++) {
            if (fsr2matMap[ifsr] == imat) {
                matFsrNum += 1;
            }
        }

        if (matFsrNum == 0) {
            continue;
        }

        int* fsrList = (int *) zres_malloc((size_t) matFsrNum, sizeof (int), RNAME(fsrList));
        int listPos = 0;

        for (int ifsr = 0; ifsr < numFsr; ifsr++) {
            if (fsr2matMap[ifsr] == imat) {
                fsrList[listPos] = ifsr;
                listPos += 1;
            }
        }

        int localKMax = matFsrNum < clusterLimit ? matFsrNum : clusterLimit;

        if ((!autoBySse && localKMax == matFsrNum) || matFsrNum == 1) {
            globalCluster = zres_fsr_cluster_commit_singletons(imat, fsrList, matFsrNum, globalCluster);
            printf(" [  CLUST  ]  Mat %d: fsr=%d, Kmax=%d, selected K=%d, "
                   "average center SSE=0.000000e+00, representative SSE=0.000000e+00;\n",
                   imat, matFsrNum, localKMax, matFsrNum);
            fflush(stdout);
            zres_free(fsrList, (size_t) matFsrNum);
            continue;
        }

        FP_RES_XS* scalarValues = (FP_RES_XS *) zres_malloc((size_t) matFsrNum, sizeof (FP_RES_XS), RNAME(scalarValues));
        zres_fsr_cluster_build_sum_t(macroT, fsrList, matFsrNum, numGroup, scalarValues);
        if (sorted1d) {
            zres_fsr_cluster_sort_samples(scalarValues, fsrList, matFsrNum);
        }

        int selectedK = 0;
        int selectedUseSingletons = 0;
        FP_RES_XS selectedCenterSse = (FP_RES_XS) 0.;
        FP_RES_XS selectedRepSse = (FP_RES_XS) 0.;
        FP_RES_XS* selectedCenters = NULL;
        int* selectedLabels = NULL;
        int* selectedCounts = NULL;
        ZRES_FSR_CLUSTER_DP_STATE dpState;
        int dpReady = 0;

        if (sorted1d) {
            /*
             * The exact DP path computes all required K values in one
             * monotone sequence.  K==numSample is known analytically to have
             * zero SSE, so no O(N^2) split table is built for that row.
             */
            int dpMaxK = localKMax < matFsrNum ? localKMax : matFsrNum - 1;
            if (dpMaxK > 0) {
                size_t splitBytes = (size_t) (dpMaxK + 1) *
                                    (size_t) (matFsrNum + 1) * sizeof (int);
                printf(" [  CLUST  ]  Mat %d: exact SSE-DP fsr=%d, maxK=%d, splitMemory=%.1f MB\n",
                       imat, matFsrNum, dpMaxK,
                       (double) splitBytes / (1024. * 1024.));
                fflush(stdout);
                zres_fsr_cluster_dp_init(&dpState, scalarValues, matFsrNum, dpMaxK);
                dpReady = 1;
            }

            if (!autoBySse) {
                selectedK = localKMax;
                while (dpReady && dpState.builtK < selectedK) {
                    zres_fsr_cluster_dp_build_next(&dpState);
                }
            }
            else {
                int initialK = 0;

                for (int testK = 1; testK <= dpMaxK; testK++) {
                    zres_fsr_cluster_dp_build_next(&dpState);
                    if (testK % 100 == 0) {
                        printf(" [  CLUST  ]  Mat %d: SSE-DP built K=%d, SSE=%.6e\n",
                               imat, testK, dpState.sse[testK]);
                        fflush(stdout);
                    }
                    if (dpState.sse[testK] <= (double) sseLimit) {
                        initialK = testK;
                        break;
                    }
                }
                if (initialK == 0) {
                    initialK = localKMax >= matFsrNum ? matFsrNum : localKMax;
                }

                selectedK = initialK;
                if (initialK > 2) {
                    int rangeLen = localKMax - initialK + 1;
                    if (rangeLen > ZRES_FSR_CLUSTER_ELBOW_WINDOW) {
                        rangeLen = ZRES_FSR_CLUSTER_ELBOW_WINDOW;
                    }

                    FP_RES_XS* kAvgSse = (FP_RES_XS *) zres_malloc(
                            (size_t) rangeLen, sizeof (FP_RES_XS), RNAME(kAvgSse));
                    for (int ik = 0; ik < rangeLen; ik++) {
                        int testK = initialK + ik;
                        if (testK == matFsrNum) {
                            kAvgSse[ik] = (FP_RES_XS) 0.;
                        }
                        else {
                            while (dpReady && dpState.builtK < testK) {
                                zres_fsr_cluster_dp_build_next(&dpState);
                            }
                            kAvgSse[ik] = (FP_RES_XS) dpState.sse[testK];
                        }
                    }

                    if (rangeLen >= 3) {
                        int elbowOffset = 1;
                        double maxAbsDiff2 = -1.;

                        for (int ik = 0; ik < rangeLen - 2; ik++) {
                            double diff2 = (double) kAvgSse[ik + 2]
                                         - 2. * (double) kAvgSse[ik + 1]
                                         + (double) kAvgSse[ik];
                            double absDiff2 = fabs(diff2);

                            if (absDiff2 > maxAbsDiff2) {
                                maxAbsDiff2 = absDiff2;
                                elbowOffset = ik + 1;
                            }
                        }
                        selectedK = initialK + elbowOffset;
                    }
                    zres_free(kAvgSse, (size_t) rangeLen);
                }
            }
        }
        else if (autoBySse) {
            int initialK = 0;

            for (int testK = 1; testK <= localKMax; testK++) {
                if (testK == matFsrNum) {
                    initialK = testK;
                    break;
                }

                FP_RES_XS* testCenters = (FP_RES_XS *) zres_malloc((size_t) testK, sizeof (FP_RES_XS), RNAME(testCenters));
                int* testLabels = (int *) zres_malloc((size_t) matFsrNum, sizeof (int), RNAME(testLabels));
                int* testCounts = (int *) zres_malloc((size_t) testK, sizeof (int), RNAME(testCounts));
                FP_RES_XS avgCenterSse = zres_fsr_cluster_run_algorithm(
                        scalarValues, matFsrNum, testK, maxIter, tolerance2,
                        restartNum, randomSeed, imat, sorted1d,
                        testCenters, testLabels, testCounts, NULL, NULL);
                int testPassed = avgCenterSse <= sseLimit;

                zres_free(testCenters, (size_t) testK);
                zres_free(testLabels, (size_t) matFsrNum);
                zres_free(testCounts, (size_t) testK);

                if (testPassed || testK == localKMax) {
                    initialK = testK;
                    break;
                }
            }

            selectedK = initialK;

            if (initialK > 2) {
                int rangeLen = localKMax - initialK + 1;
                if (rangeLen > ZRES_FSR_CLUSTER_ELBOW_WINDOW) {
                    rangeLen = ZRES_FSR_CLUSTER_ELBOW_WINDOW;
                }

                FP_RES_XS* kAvgSse = (FP_RES_XS *) zres_malloc((size_t) rangeLen, sizeof (FP_RES_XS), RNAME(kAvgSse));

                for (int ik = 0; ik < rangeLen; ik++) {
                    int testK = initialK + ik;

                    if (testK == matFsrNum) {
                        kAvgSse[ik] = (FP_RES_XS) 0.;
                        continue;
                    }

                    FP_RES_XS* testCenters = (FP_RES_XS *) zres_malloc((size_t) testK, sizeof (FP_RES_XS), RNAME(testCenters));
                    int* testLabels = (int *) zres_malloc((size_t) matFsrNum, sizeof (int), RNAME(testLabels));
                    int* testCounts = (int *) zres_malloc((size_t) testK, sizeof (int), RNAME(testCounts));

                    kAvgSse[ik] = zres_fsr_cluster_run_algorithm(
                            scalarValues, matFsrNum, testK, maxIter, tolerance2,
                            restartNum, randomSeed, imat, sorted1d,
                            testCenters, testLabels, testCounts, NULL, NULL);

                    zres_free(testCenters, (size_t) testK);
                    zres_free(testLabels, (size_t) matFsrNum);
                    zres_free(testCounts, (size_t) testK);
                }

                if (rangeLen >= 3) {
                    int elbowOffset = 1;
                    double maxAbsDiff2 = -1.;

                    for (int ik = 0; ik < rangeLen - 2; ik++) {
                        double diff2 = (double) kAvgSse[ik + 2]
                                     - 2. * (double) kAvgSse[ik + 1]
                                     + (double) kAvgSse[ik];
                        double absDiff2 = fabs(diff2);

                        if (absDiff2 > maxAbsDiff2) {
                            maxAbsDiff2 = absDiff2;
                            elbowOffset = ik + 1;
                        }
                    }

                    selectedK = initialK + elbowOffset;
                }

                zres_free(kAvgSse, (size_t) rangeLen);
            }
        }
        else {
            selectedK = localKMax;
        }

        if (selectedK <= 0) {
            zres_raise_error(" [  ERROR  ]  Failed to select FSR cluster number\n");
        }

        if (selectedK >= matFsrNum) {
            selectedUseSingletons = 1;
        }
        else {
            selectedCenters = (FP_RES_XS *) zres_malloc((size_t) selectedK, sizeof (FP_RES_XS), RNAME(selectedCenters));
            selectedLabels = (int *) zres_malloc((size_t) matFsrNum, sizeof (int), RNAME(selectedLabels));
            selectedCounts = (int *) zres_malloc((size_t) selectedK, sizeof (int), RNAME(selectedCounts));
            if (sorted1d) {
                if (!dpReady || selectedK > dpState.builtK) {
                    zres_raise_error(" [  ERROR  ]  SSE DP did not build the selected K\n");
                }
                selectedCenterSse = (FP_RES_XS) dpState.sse[selectedK];
                zres_fsr_cluster_dp_reconstruct(&dpState, scalarValues, selectedK,
                                                selectedCenters, selectedLabels,
                                                selectedCounts, &selectedRepSse);
            }
            else {
                selectedCenterSse = zres_fsr_cluster_run_algorithm(
                    scalarValues, matFsrNum, selectedK, maxIter, tolerance2,
                    restartNum, randomSeed, imat, sorted1d,
                    selectedCenters, selectedLabels, selectedCounts,
                    NULL, &selectedRepSse);
            }
        }

        if (dpReady) {
            zres_fsr_cluster_dp_destroy(&dpState);
        }

        if (selectedUseSingletons) {
            globalCluster = zres_fsr_cluster_commit_singletons(imat, fsrList, matFsrNum, globalCluster);
        }
        else {
            globalCluster = zres_fsr_cluster_commit_kmeans(imat, fsrList, matFsrNum, selectedK,
                                                           scalarValues, selectedCenters, selectedLabels, selectedCounts,
                                                           globalCluster);
        }

        printf(" [  CLUST  ]  Mat %d: fsr=%d, Kmax=%d, selected K=%d, "
               "average center SSE=%.6e, representative SSE=%.6e;\n",
               imat, matFsrNum, localKMax, selectedK,
               (double) selectedCenterSse, (double) selectedRepSse);
        fflush(stdout);

        zres_free(fsrList, (size_t) matFsrNum);
        zres_free(scalarValues, (size_t) matFsrNum);
        if (!selectedUseSingletons) {
            zres_free(selectedCenters, (size_t) selectedK);
            zres_free(selectedLabels, (size_t) matFsrNum);
            zres_free(selectedCounts, (size_t) selectedK);
        }
    }

    _zresFsrClusterInfo.numCluster = globalCluster;
    _zresFsrClusterInfo.ready = 1;

    printf(" [  CLUST  ]  FSR clustering finished, total cluster number: %d\n", globalCluster);
    fflush(stdout);

    return globalCluster;
}

/*
 * Cluster each material with a fixed upper cluster count.
 *
 * The actual cluster number for one material is min(clusterPerMat, matFsrNum).
 */
int zres_cluster_fsr_by_macro_t_sse_seed_external(FP_RES_XS sseLimit,
                                                  int maxClusterPerMat,
                                                  int maxIter,
                                                  FP_RES_XS tolerance,
                                                  unsigned int randomSeed,
                                                  int materialNum) {
    return zres_cluster_fsr_by_macro_t_core(1, 0, sseLimit,
                                            maxClusterPerMat, maxIter, tolerance,
                                            ZRES_FSR_CLUSTER_RESTART, randomSeed,
                                            materialNum);
}

void zres_apply_fsr_cluster_macro_t_nuf(int applyTotal,
                                        int applyNuf) {
    ZRES_GI* gi = zres_get_gen_info();

    if (!_zresFsrClusterInfo.ready) {
        zres_raise_error(" [  ERROR  ]  FSR cluster map is not ready for macro xs replacement\n");
    }

    int numFsr = _zresFsrClusterInfo.numFsr;
    int numGroup = _zresFsrClusterInfo.numGroup;
    FP_RES_XS* macroT = zres_fsr_cluster_macro_t(gi);
    FP_RES_XS* macroNuf = zres_fsr_cluster_macro_nuf(gi);

    if (applyTotal && macroT == NULL) {
        zres_raise_error(" [  ERROR  ]  Macro total xs is NULL in cluster replacement\n");
    }
    if (applyNuf && macroNuf == NULL) {
        zres_raise_error(" [  ERROR  ]  Macro nuf xs is NULL in cluster replacement\n");
    }

    for (int ifsr = 0; ifsr < numFsr; ifsr++) {
        int repFsr = _zresFsrClusterInfo.repFsrOfFsr[ifsr];

        if (repFsr == ifsr) {
            continue;
        }

        if (applyTotal) {
            memcpy(macroT + ifsr * numGroup, macroT + repFsr * numGroup, sizeof (FP_RES_XS) * (size_t) numGroup);
        }
        if (applyNuf) {
            memcpy(macroNuf + ifsr * numGroup, macroNuf + repFsr * numGroup, sizeof (FP_RES_XS) * (size_t) numGroup);
        }
    }
}

/*
 * Replace the transport fission spectrum by the value of the representative
 * FSR.  The transport flux and geometry remain indexed by the original FSR;
 * only the FSR-dependent fission-spectrum data are clustered.
 */
void zres_apply_fsr_cluster_chi() {
    ZRES_GI* gi = zres_get_gen_info();

    if (!_zresFsrClusterInfo.ready) {
        zres_raise_error(" [  ERROR  ]  FSR cluster map is not ready for Chi replacement\n");
    }
    if (gi == NULL || gi->chi == NULL) {
        zres_raise_error(" [  ERROR  ]  Chi array is NULL in FSR cluster replacement\n");
    }

    const int numFsr = _zresFsrClusterInfo.numFsr;
    const int numGroup = _zresFsrClusterInfo.numGroup;
#ifdef FIS_MTX
    const size_t chiStride = (size_t) numGroup * (size_t) numGroup;
#else
    const size_t chiStride = (size_t) numGroup;
#endif

    for (int ifsr = 0; ifsr < numFsr; ++ifsr) {
        const int repFsr = _zresFsrClusterInfo.repFsrOfFsr[ifsr];

        if (repFsr < 0 || repFsr >= numFsr) {
            zres_raise_error(" [  ERROR  ]  Invalid representative FSR in Chi replacement\n");
        }
        if (repFsr == ifsr) {
            continue;
        }

        memcpy(gi->chi + (size_t) ifsr * chiStride,
               gi->chi + (size_t) repFsr * chiStride,
               sizeof(FP_RES_XS) * chiStride);
    }
}

/*
 * Replace slow-down escape XS by representative-FSR values.
 *
 * The escape XS array keeps the original numFsr x resGrpNum layout.  Only the
 * values are copied, so downstream scatter matrix code can keep indexing by the
 * original FSR id.
 */
