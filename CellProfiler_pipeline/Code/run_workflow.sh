#!/bin/bash
set -e

START_TIME=$(date +%s)
set -a
source /workspace/variables.env
set +a
source /workspace/Code/bash_functions.sh

create_output_dirs $IMAGES_WORKSPACE

BATCH_SIZE=5000 

#echo "[DEBUG] $(ls /workspace_images)"

#### --- 1) Creación de los CSV [Illum] --- ####

python $SCRIPT_PY $IMAGES_WORKSPACE $PATH_CSV_ILLUM 0

#### --- 2) Creación de los CPPIPE y ejecución de CellProfiler [Illum] --- ####

echo "===***=== Obtención de illum files ===***==="
generar_batchfiles "$PATH_CSV_ILLUM/metadata.csv" "$TEMPLATE_CPPIPE_ILLUM" "$PATH_CPPIPE_ILLUM" "$BATCH_PIPELINES" "$PATH_ILLUM_FILES" 0
mv $BATCH_PIPELINES/Batch_data.h5 \
   $BATCH_PIPELINES/Batch_data_Illum.h5
echo "Batch files generated"
ejecutar_pipeline "$BATCH_PIPELINES/Batch_data_Illum.h5" 1

#### --- 3) Creación de los CSV [Profiling] --- ####

echo "===***=== Obtención de csv prof ===***==="
python $SCRIPT_PY $IMAGES_WORKSPACE $PATH_CSV_PROF 1

CSV_COUNT=$(find "$PATH_CPPIPE_PROF" -maxdepth 1 -name "*.csv" | wc -l)
echo "[INFO] Se van a procesar $CSV_COUNT archivos CSV en $PATH_CPPIPE_PROF"

#### --- 4) Creación de los CPPIPE y ejecución de CellProfiler [Profiling] --- ####
echo "===***=== Obtención de profiles ===***==="
generar_batchfiles "$PATH_CSV_PROF/metadata.csv" "$TEMPLATE_CPPIPE_PROFILING" "$PATH_CPPIPE_PROF" "$BATCH_PIPELINES" "$PATH_PROFILES" 1
mv $BATCH_PIPELINES/Batch_data.h5 \
   $BATCH_PIPELINES/Batch_data_MP.h5
echo "batchfiles generated"
ejecutar_pipeline "$BATCH_PIPELINES/Batch_data_MP.h5" 0 "$PATH_PROFILES" "$PATH_CSV_PROF/metadata.csv" 5000

END_TIME=$(date +%s)
echo "[INFO] Tiempo total: $((END_TIME - START_TIME))s"