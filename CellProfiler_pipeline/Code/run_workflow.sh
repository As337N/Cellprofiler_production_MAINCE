#!/bin/bash
set -Eeuo pipefail
trap 'echo "[FATAL] línea $LINENO: $BASH_COMMAND (rc=$?)" >&2' ERR
START_TIME=$(date +%s)

set -a
source /config/variables.env
set +a
source /workspace/Code/bash_functions.sh

#sections 3-12
run_section() { [[ " ${SECTIONS[*]} " == *" $1 "* ]]; }

# ============================================================
create_output_dirs $OUTPUT $IMAGES_WORKSPACE
BATCH_SIZE=1000

if run_section 3; then
  echo "===***=== [3] Prepare QC ===***==="
  NAME_QC="III_QC"
  echo "[Z-PLANE] $ZPLANE"
  #TODO: Si corremos MAINCE con cellpose, hay que poner el parámetro --masks
  python $SCRIPT_PY_CELLPROFILER -i $IMAGES_WORKSPACE -o $PATH_CSV --name_csv $NAME_QC --illum --channel_dict "$CHANNEL_DICT" --filename_regex "$FILENAME_REGEX" --zplane "$ZPLANE"

  echo "[DEBUG 12.10.2026] Path csv: $PATH_CSV"

  generar_batchfiles "$PATH_CSV/$NAME_QC.csv" "$TEMPLATE_CPPIPE_QC" "$PATH_CPPIPE" "$PATH_BATCH_PIPELINES" "$PATH_QC_IMAGES" 1
  mv "$PATH_BATCH_PIPELINES/Batch_data.h5" \
     "$PATH_BATCH_PIPELINES/Batch_data_QC.h5"
  echo "[INFO] Batchfiles generated for QC"

  ejecutar_pipeline --pipeline "$PATH_BATCH_PIPELINES/Batch_data_QC.h5" --illum 0 --out "$PATH_QC_IMAGES" --metadata "$PATH_CSV/$NAME_QC.csv" --batch_size $BATCH_SIZE
fi

#### --- 4) Obtain plate collage for quality control --- ####
if run_section 4; then
  echo "===***=== [4] Generate QC report ===***==="
  python $SCRIPT_PY_AGGREGATE_QC_OUTPUT --input "$PATH_QC_IMAGES"
  cd ./Code
  python -m qc.cli -i "$PATH_QC_IMAGES" -o "$PATH_QC_COLLAGES" \
    --platemap "/workspace_images/platemap_${COHORT}.csv" --no-pdf
  cd -
  echo "[INFO] QC report generated in ${PATH_QC_COLLAGES}"
fi

#### --- 5) Calculate CellProfiler features --- ####
if run_section 5; then
  echo "===***=== [5] Profile generation ===***==="
  NAME_MP="V_MP"
  #TODO: Falta integrar --discard_sites, para descartar sitios flageados
  python $SCRIPT_PY_CELLPROFILER -i $IMAGES_WORKSPACE -o $PATH_CSV --name_csv $NAME_MP --illum --channel_dict "$CHANNEL_DICT" --filename_regex "$FILENAME_REGEX" --zplane "$ZPLANE"

  CSV_COUNT=$(find "$PATH_CPPIPE" -maxdepth 1 -name "*.csv" | wc -l)
  echo "[INFO] Amount of CSV files to be processed: $CSV_COUNT in: $PATH_CPPIPE"

  generar_batchfiles "$PATH_CSV/$NAME_MP.csv" "$TEMPLATE_CPPIPE_PROFILING" "$PATH_CPPIPE" "$PATH_BATCH_PIPELINES" "$PATH_PROFILES" 1
  mv "$PATH_BATCH_PIPELINES/Batch_data.h5" \
     "$PATH_BATCH_PIPELINES/Batch_data_MP.h5"
  echo "[INFO] Batchfiles generated"

  ejecutar_pipeline --pipeline "$PATH_BATCH_PIPELINES/Batch_data_MP.h5" --illum 0 --out "$PATH_PROFILES" --metadata "$PATH_CSV/$NAME_MP.csv" --batch_size $BATCH_SIZE
fi

#### --- 6) Feature postprocessing [Aggregation, Normalization and Reduction] --- ####
if run_section 6; then
  echo "===***=== [6] Feature processing ===***==="
  python $SCRIPT_PY_FEAT_PROCESS -i $PATH_PROFILES -o $PATH_FINAL_PROFILES -c $COHORT -m $PATH_PLATEMAP
fi

#### --- 7) Clustering --- ####
if run_section 7; then
  echo "===***=== [7] Clustering generation ===***==="
  python $SCRIPT_PY_CLUSTERING -i $PATH_FINAL_PROFILES -o $PATH_CLUSTERS -c $COHORT
fi

#### --- 8) Reproducibility --- ####
if run_section 8; then
  echo "===***=== [8] Reproducibility analysis ===***==="
  python $SCRIPT_PY_REPRODUCIBILITY -i $PATH_FINAL_PROFILES -o $PATH_REPRODUCIBILITY -c $COHORT
fi

#### --- 9) Subprofile analysis --- ####
if run_section 9; then
  echo "===***=== [9] Subprofile clustering ===***==="
  python $SCRIPT_PY_SUBPROFILES \
    -i $PATH_FINAL_PROFILES \
    -o $PATH_SUBPROFILES \
    -c $COHORT \
    --fraction 0.85 \
    --magnitude 0.5 \
    --fdr 0.05 \
    --metric spearman \
    --reference-col Metadata_Reference
fi

# ── 10) Morphological Map ───────────────────────────────────────────────

if run_section 10; then
  echo "===***=== [10] Morphological Map ===***==="
  python $SCRIPT_PY_MORPHOMAP \
    -i "$PATH_SUBPROFILES/subprofiles_norm" \
    -o $PATH_MORPHOMAP \
    -c $COHORT \
    --prefix-filenames \
    --metric spearman \
    --fdr 0.05 \
    --secondary-threshold 0.60
fi


# -- 11) Random forest ----------------

if run_section 11; then
  echo "===***=== [11] Random Forest ===***==="
  python $SCRIPT_PY_RANDOMFOREST \
    -i "$PATH_FINAL_PROFILES" \
    -o "$PATH_RANDOMFOREST" \
    -c $COHORT
fi

# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf "[INFO] Pipeline complete in: %02d:%02d:%02d\n" \
  $((ELAPSED/3600)) $(( (ELAPSED%3600)/60 )) $((ELAPSED%60))