#!/bin/bash
set -e
START_TIME=$(date +%s)

set -a
source /workspace/variables.env
set +a
source /workspace/Code/bash_functions.sh

SECTIONS=(5 6)
run_section() { [[ " ${SECTIONS[*]} " == *" $1 "* ]]; }

# ============================================================
create_output_dirs $OUTPUT $IMAGES_WORKSPACE
BATCH_SIZE=5000

#### --- 3) Obtain plate collage for quality control --- ####
if run_section 3; then
  echo "===***=== [3] QC outlines generation ===***==="
  NAME_QC="III_QC"
  python $SCRIPT_PY_CELLPROFILER $IMAGES_WORKSPACE $PATH_CSV $NAME_QC 1 1

  CSV_COUNT=$(find "$PATH_CPPIPE" -maxdepth 1 -name "*.csv" | wc -l)
  echo "[INFO] Se van a procesar $CSV_COUNT archivos CSV en $PATH_CPPIPE"

  generar_batchfiles "$PATH_CSV/$NAME_QC.csv" "$TEMPLATE_CPPIPE_QC" "$PATH_CPPIPE" "$PATH_BATCH_PIPELINES" "$PATH_QC_IMAGES" 1
  mv "$PATH_BATCH_PIPELINES/Batch_data.h5" \
     "$PATH_BATCH_PIPELINES/Batch_data_QC.h5"
  echo "[INFO] Batchfiles generated"

  ejecutar_pipeline "$PATH_BATCH_PIPELINES/Batch_data_QC.h5" 0 "$PATH_QC_IMAGES" "$PATH_CSV/$NAME_QC.csv" $BATCH_SIZE
  python $SCRIPT_PY_COLLAGE -i $PATH_QC_IMAGES -o $PATH_QC_COLLAGES
fi

#### --- 4) Calculate CellProfiler features --- ####
if run_section 4; then
  echo "===***=== [4] Profile generation ===***==="
  NAME_MP="IV_MP"
  python $SCRIPT_PY_CELLPROFILER $IMAGES_WORKSPACE $PATH_CSV $NAME_MP 1 1

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
  python $SCRIPT_PY_CLUSTERING -i $PATH_FINAL_PROFILES -o $PATH_CLUSTERS
fi

# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf "[INFO] Pipeline complete in: %02d:%02d:%02d\n" \
  $((ELAPSED/3600)) $(( (ELAPSED%3600)/60 )) $((ELAPSED%60))