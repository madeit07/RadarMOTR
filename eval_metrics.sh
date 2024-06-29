#!/usr/bin/env bash

python ./TrackEval/scripts/run_mot_challenge.py \
    --BENCHMARK RDTrack \
    --DO_PREPROC False \
    --GT_FOLDER ./data/dataset/RDTrack/ \
    --SPLIT_TO_EVAL $1 \
    --SKIP_SPLIT_FOL False \
    --USE_PARALLEL True \
    --NUM_PARALLEL_CORES 4 \
    --PLOT_CURVES False \
    --METRICS HOTA CLEAR Identity  \
    --TRACKERS_TO_EVAL $2 \
    --TRACKER_SUB_FOLDER '' \
    --TRACKERS_FOLDER ./data/trackers/ | tee -a ./data/trackers/rdtrack-"$1"/"$2"/eval.log


# ./eval_metrics.sh val run4
