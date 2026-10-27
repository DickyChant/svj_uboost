#!/bin/bash

# configurable parameters
sels=()
# needed to call merge and smooth on histograms after producing them
hists_date=$(date +%Y%m%d)
skim_dir="skims_20241030_hadd"

# common binning parameters
mt_wide="--mtmin 130 --mtmax 700"
mt_reg="--mtmin 180 --mtmax 650"
mt_bin="--mtbinw 10"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --sels) IFS=' ' read -r -a sels <<< "$2"; shift ;;
        --hists_date) hists_date="$2"; shift ;;
        --skim_dir) skim_dir="$2"; shift ;;
        *) "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

actual_sels=()
for sel in ${sels[@]}; do
    # define all regions for this selection
    actual_sels+=(${sel} anti${sel} antiloose${sel})
done

set -x
for sel in ${actual_sels[@]}; do
    # DATA: just build and merge
    python3 build_datacard.py build_all_histograms ${sel} "${skim_dir}/Run20*/*.npz" ${mt_bin}
    python3 build_datacard.py merge_histograms ${sel} hists_${hists_date} --cat data

    # BKG: build w/ extended mt range for smoothing (to make toys); then produce merged set w/ regular mt range
    python3 build_datacard.py build_all_histograms ${sel} "${skim_dir}/Summer*/*.npz" ${mt_wide} ${mt_bin}
    python3 build_datacard.py merge_histograms ${sel} hists_${hists_date} --cat bkg
    bkg_merged=merged_${hists_date}/bkg_sel-${sel}_mt
    mv ${bkg_merged}.json ${bkg_merged}_wide.json
    python3 build_datacard.py smooth_shapes --optimize 1000 --spanmin 0.11 --default bkg ${mt_reg} ${bkg_merged}_wide.json
    python3 build_datacard.py merge_histograms ${sel} hists_${hists_date} --cat bkg ${mt_reg} ${mt_bin}
    mv ${bkg_merged}_binw10.00_range180.0-650.0.json ${bkg_merged}.json

    # SIG: build w/ extended mt range for smoothing
    python3 build_datacard.py build_all_histograms ${sel} "${skim_dir}/Private3D*/*pythia8.npz" ${mt_wide} ${mt_bin}
    python3 build_datacard.py merge_histograms ${sel} hists_${hists_date} --cat sig
    # span of the ratio smoothing per systematic: smallest span with the best held-out score across the years, capped at 0.5
    python3 ratio_span_cv.py hists_${hists_date}/SVJ_*_sel-${sel}_year-*_mt.json --span-max 0.5 --out merged_${hists_date}/ratio_span_sel-${sel}.json
    python3 build_datacard.py smooth_shapes --optimize 1000 --target central ${mt_reg} --ratio-span-file merged_${hists_date}/ratio_span_sel-${sel}.json merged_${hists_date}/SVJ_*_sel-${sel}_*
done
