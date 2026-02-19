#!/bin/bash
set -e

START_TIME=$(date +%s)
set -a
source /workspace/variables.env
set +a
source /workspace/Code/bash_functions.sh

create_output_dirs $OUTPUT $IMAGES_WORKSPACE

BATCH_SIZE=5000 

#### --- 1) Calculate illumination correction files --- ####

python $SCRIPT_PY_CELLPROFILER $IMAGES_WORKSPACE $PATH_CSV_ILLUM 0 0

echo "===***=== Generating Illumination correction files ===***==="
generar_batchfiles "$PATH_CSV_ILLUM/metadata.csv" "$TEMPLATE_CPPIPE_ILLUM" "$PATH_CPPIPE_ILLUM" "$BATCH_PIPELINES" "$PATH_ILLUM_FILES" 0
mv $BATCH_PIPELINES/Batch_data.h5 \
   $BATCH_PIPELINES/Batch_data_Illum.h5
echo "Batch files generated"
ejecutar_pipeline "$BATCH_PIPELINES/Batch_data_Illum.h5" 1

