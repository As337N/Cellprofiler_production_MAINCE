#!/bin/bash
set -e
START_TIME=$(date +%s)

set -a
source /config/variables.env
set +a
source /workspace/Code/bash_functions.sh

SECTIONS=(10) #3-9
run_section() { [[ " ${SECTIONS[*]} " == *" $1 "* ]]; }

# ============================================================
create_output_dirs $OUTPUT $IMAGES_WORKSPACE
BATCH_SIZE=5000

#### --- 3) Obtain plate collage for quality control --- ####
if run_section 3; then
  echo "===***=== [3] QC outlines generation ===***==="
  NAME_QC="III_QC"
  python $SCRIPT_PY_CELLPROFILER -i $IMAGES_WORKSPACE -o $PATH_CSV --name_csv $NAME_QC --illum --masks

  CSV_COUNT=$(find "$PATH_CPPIPE" -maxdepth 1 -name "*.csv" | wc -l)
  echo "[INFO] Se van a procesar $CSV_COUNT archivos CSV en $PATH_CPPIPE"

  generar_batchfiles "$PATH_CSV/$NAME_QC.csv" "$TEMPLATE_CPPIPE_QC" "$PATH_CPPIPE" "$PATH_BATCH_PIPELINES" "$PATH_QC_IMAGES" 1
  mv "$PATH_BATCH_PIPELINES/Batch_data.h5" \
     "$PATH_BATCH_PIPELINES/Batch_data_QC.h5"
  echo "[INFO] Batchfiles generated"

  ejecutar_pipeline "$PATH_BATCH_PIPELINES/Batch_data_QC.h5" 0 "$PATH_QC_IMAGES" "$PATH_CSV/$NAME_QC.csv" $BATCH_SIZE
  python $SCRIPT_PY_COLLAGE -i $PATH_QC_IMAGES -o $PATH_QC_COLLAGES --platemap /workspace_images/platemap_${COHORT}.csv
fi

#### --- 4) Calculate CellProfiler features --- ####
if run_section 4; then
  echo "===***=== [4] Profile generation ===***==="
  NAME_MP="IV_MP"
  python $SCRIPT_PY_CELLPROFILER -i $IMAGES_WORKSPACE -o $PATH_CSV --name_csv $NAME_MP --illum --masks

  CSV_COUNT=$(find "$PATH_CPPIPE" -maxdepth 1 -name "*.csv" | wc -l)
  echo "[INFO] Amount of CSV files to be processed: $CSV_COUNT in: $PATH_CPPIPE"

  generar_batchfiles "$PATH_CSV/$NAME_MP.csv" "$TEMPLATE_CPPIPE_PROFILING" "$PATH_CPPIPE" "$PATH_BATCH_PIPELINES" "$PATH_PROFILES" 1
  mv "$PATH_BATCH_PIPELINES/Batch_data.h5" \
     "$PATH_BATCH_PIPELINES/Batch_data_MP.h5"
  echo "[INFO] Batchfiles generated"

  ejecutar_pipeline "$PATH_BATCH_PIPELINES/Batch_data_MP.h5" 0 "$PATH_PROFILES" "$PATH_CSV/$NAME_MP.csv" $BATCH_SIZE
fi

#### --- 5) Feature postprocessing [Aggregation, Normalization and Reduction] --- ####
if run_section 5; then
  echo "===***=== [5] Feature processing ===***==="
  python $SCRIPT_PY_FEAT_PROCESS -i $PATH_PROFILES -o $PATH_FINAL_PROFILES -c $COHORT -m $PATH_PLATEMAP
fi

#### --- 6) Clustering --- ####
if run_section 6; then
  echo "===***=== [6] Clustering generation ===***==="
  python $SCRIPT_PY_CLUSTERING -i $PATH_FINAL_PROFILES -o $PATH_CLUSTERS -c $COHORT
fi

#### --- 7) Reproducibility --- ####
if run_section 7; then
  echo "===***=== [7] Reproducibility analysis ===***==="
  python $SCRIPT_PY_REPRODUCIBILITY -i $PATH_FINAL_PROFILES -o $PATH_REPRODUCIBILITY -c $COHORT
fi

#### --- 8) Subprofile analysis --- ####
if run_section 8; then
  echo "===***=== [8] Subprofile clustering ===***==="
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

# ── Step IX: Morphological Map ───────────────────────────────────────────────

if run_section 9; then
  echo "===***=== [9] Morphological Map ===***==="
  python $SCRIPT_PY_MORPHOMAP \
    -i "$PATH_SUBPROFILES/subprofiles_norm" \
    -o $PATH_MORPHOMAP \
    -c $COHORT \
    --prefix-filenames \
    --metric spearman \
    --fdr 0.05 \
    --secondary-threshold 0.60
fi


# -- Step X: Random forest ----------------

if run_section 10; then
  echo "===***=== [10] Random Forest ===***==="
  python $SCRYPT_PY_RANDOMFOREST \
    -i "$PATH_FINAL_PROFILES" \
    -o "$PATH_RANDOMFOREST" \
    -c $COHORT
fi

# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf "[INFO] Pipeline complete in: %02d:%02d:%02d\n" \
  $((ELAPSED/3600)) $(( (ELAPSED%3600)/60 )) $((ELAPSED%60))