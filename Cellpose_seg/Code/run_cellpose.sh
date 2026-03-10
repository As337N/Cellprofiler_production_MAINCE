#!/bin/bash
set -e # El script fallara si hay un error
START_TIME=$(date +%s)
set -a
source /workspace/variables.env
SECTIONS=(2) # 2
run_section() { [[ " ${SECTIONS[*]} " == *" $1 "* ]]; }

if run_section 2; then
  #### --- 2) Cellpose segmentation --- ####
  echo "*** $PWD ***"
  python3 $SCRIPT_PY_CELLPOSE $IMAGES_WORKSPACE /output/CellProfiler_files/Cellpose_seg --rna_channel 4 --batch-size 36
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf "[INFO] Cellpose seg complete in: %02d:%02d:%02d\n" \
  $((ELAPSED/3600)) $(( (ELAPSED%3600)/60 )) $((ELAPSED%60))
