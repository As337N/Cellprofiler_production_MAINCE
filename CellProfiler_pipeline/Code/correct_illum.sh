#!/bin/bash
set -e

START_TIME=$(date +%s)
set -a
source /workspace/variables.env
set +a
source /workspace/Code/bash_functions.sh

create_output_dirs $OUTPUT $IMAGES_WORKSPACE

BATCH_SIZE=5000 

SECTIONS=( )
run_section() { [[ " ${SECTIONS[*]} " == *" $1 "* ]]; }

#### --- 1) Calculate illumination correction files --- ####
if run_section 1; then
   echo "===***=== Generating Illumination correction files ===***==="
   NAME_ILLUM="I_illum"
   python $SCRIPT_PY_CELLPROFILER $IMAGES_WORKSPACE $PATH_CSV $NAME_ILLUM 0 0

   generar_batchfiles "$PATH_CSV/$NAME_ILLUM.csv" "$TEMPLATE_CPPIPE_ILLUM" "$PATH_CPPIPE" "$PATH_BATCH_PIPELINES" "$PATH_ILLUM_FILES" 0
   mv $PATH_BATCH_PIPELINES/Batch_data.h5 \
      $PATH_BATCH_PIPELINES/Batch_data_Illum.h5
   echo "Batch files generated"
   ejecutar_pipeline "$PATH_BATCH_PIPELINES/Batch_data_Illum.h5" 1
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
printf "[INFO] Cellpose illum complete in: %02d:%02d:%02d\n" \
$((ELAPSED/3600)) $(( (ELAPSED%3600)/60 )) $((ELAPSED%60))