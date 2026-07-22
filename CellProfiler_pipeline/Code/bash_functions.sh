detectar_max_jobs() {
  local n_cores mem_kb mem_gb jobs_by_cpu jobs_by_mem max_jobs

  n_cores=$(nproc --all)

  mem_kb=$(grep -i MemTotal /proc/meminfo | awk '{print $2}')
  mem_gb=$((mem_kb / 1024 / 1024))

  # 8 GB por CP job → seguro para BBBC047
  jobs_by_mem=$((mem_gb / 8))
  if [ "$jobs_by_mem" -lt 1 ]; then
    jobs_by_mem=1
  fi

  jobs_by_cpu=$((n_cores - 1))
  if [ "$jobs_by_cpu" -lt 1 ]; then
    jobs_by_cpu=1
  fi

  if [ "$jobs_by_mem" -lt "$jobs_by_cpu" ]; then
    max_jobs=$jobs_by_mem
  else
    max_jobs=$jobs_by_cpu
  fi

  echo "$max_jobs"
}

generar_batchfiles() {
  local metadata_csv_path="$1"
  local TEMPLATE_CPPIPE="$2"
  local PATH_CPPIPE="$3"
  local PATH_BATCH_FILE="$4"
  local PATH_OUTPUT="$5"
  local PROFILES="${6:-}"

  if [ -z "$metadata_csv_path" ] || [ -z "$TEMPLATE_CPPIPE" ] || [ -z "$PATH_CPPIPE" ]; then
    echo "[ERROR] A parameter is missing, path csv: $metadata_csv_path, template cppipe: $TEMPLATE_CPPIPE, path cppipe: $PATH_CPPIPE"
    return 1
  fi
  
  (
    local parent_dir_csv file_name_csv file_name OUTPUT_DIR PATH_IMAGES CURRENT_CPPIPE

    parent_dir_csv=$(dirname "$metadata_csv_path")
    file_name_csv=$(basename "$metadata_csv_path")
    file_name="${file_name_csv%.csv}"

    PATH_IMAGES=$(find "/workspace_images" -mindepth 1 -maxdepth 1 -type d ! -name "Output" | head -n 1)
    echo "[INFO] path images: $PATH_IMAGES"
    CURRENT_CPPIPE="$PATH_CPPIPE/pipeline_${file_name}.cppipe"

  sed -e "s|INPUT_PATH_CSV|$parent_dir_csv|g" \
      -e "s|SAVING_OUTPUT_PATH|$PATH_OUTPUT|g" \
      -e "s|SAVING_MEASUREMENTS_OUTPUT_PATH|$PATH_OUTPUT/Measurements|g" \
      -e "s|SAVING_BATCH_PATH|$PATH_BATCH_FILE|g" \
      -e "s|FILE_CSV|$file_name_csv|g" \
      -e "s|INPUT_PATH_IMAGES|$PATH_IMAGES|g" \
      -e "s|TEMPLATE_CPPIPE|$CURRENT_CPPIPE|g" \
      "$TEMPLATE_CPPIPE" > "$CURRENT_CPPIPE"

  echo "[INFO] CPPipe generated in: $CURRENT_CPPIPE"
  echo "[INFO] Saving h5 in: $PATH_BATCH_FILE"

  cellprofiler -c -r \
  --data-file "$metadata_csv_path" \
  -o "$PATH_BATCH_FILE" \
  -p "$CURRENT_CPPIPE" \
  -i "$PATH_IMAGES"

  echo "[INFO] Generando batch file : $file_name_csv"
  ) &
  wait
}

create_output_dirs() {
    local OUTPUT="$1"
    local IMAGES_WORKSPACE="$2"
    local REGEX='_P([0-9]{2})_'

    mapfile -t PLATES < <(
        find "$IMAGES_WORKSPACE" -maxdepth 1 -type d -printf '%f\n' \
        | grep -oP "$REGEX" \
        | sed -E 's/_P([0-9]{2})_/\1/' \
        | sort -u
    )

    for PLATE in "${PLATES[@]}"; do

        PATH_CELLPOSE_SEG="$OUTPUT/CellProfiler_files/Cellpose_seg"
        PATH_CELLPOSE_LOCAL_MODEL="$OUTPUT/CellProfiler_files/cellpose_models_cache"
        PATH_CSV="$OUTPUT/CellProfiler_files/CSVs"
        PATH_ILLUM_FILES="$OUTPUT/CellProfiler_files/Illum_files"
        PATH_CPPIPE="$OUTPUT/CellProfiler_files/Pipelines"
        PATH_BATCH_PIPELINES="$OUTPUT/CellProfiler_files/Batch_files"
        PATH_PROFILES="$OUTPUT/CellProfiler_files/MP"

        PATH_QC_IMAGES="$OUTPUT/QC/Images"
        PATH_QC_COLLAGES="$OUTPUT/QC/Collages"
        PATH_QC_MEASUREMENTS="$OUTPUT/QC/Measurements"
        PATH_QC_REPORTS="$OUTPUT/QC/Reports"

        PATH_FINAL_PROFILES="$OUTPUT/Profiles/Treated_profiles"
        PATH_CLUSTERS="$OUTPUT/Clustering"

        PATH_REPRODUCIBILITY="$OUTPUT/Reproducibility"

        PATH_SUBPROFILES="$OUTPUT/Subprofiles"

        PATH_MORPHOMAP="$OUTPUT/MorphologicalMap"

        PATH_RANDOMFOREST="$OUTPUT/Random_forest"

        for folder in \
            "$PATH_CELLPOSE_SEG" \
            "$PATH_CELLPOSE_LOCAL_MODEL" \
            "$PATH_CSV" \
            "$PATH_ILLUM_FILES" \
            "$PATH_CPPIPE" \
            "$PATH_BATCH_PIPELINES" \
            "$PATH_PROFILES" \
            "$PATH_QC_IMAGES" \
            "$PATH_QC_COLLAGES" \
            "$PATH_QC_MEASUREMENTS" \
            "$PATH_QC_REPORTS" \
            "$PATH_FINAL_PROFILES" \
            "$PATH_CLUSTERS" \
            "$PATH_REPRODUCIBILITY" \
            "$PATH_SUBPROFILES" \
            "$PATH_MORPHOMAP" \
            "$PATH_RANDOMFOREST"
        do
            mkdir -p "$folder"
        done
        export PATH_CSV 
    done
}


# ============================================================
#  Resource detection → prints MAX_JOBS to stdout
# ============================================================
calculate_max_jobs() {
  local ram_per_batch="${CP_RAM_PER_BATCH:-4000}"   # MiB per worker

  local nproc mem_free max_by_ram max_jobs
  nproc=$(nproc)
  mem_free=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)

  max_by_ram=$(( mem_free / ram_per_batch ))
  [ "$max_by_ram" -lt 1 ] && max_by_ram=1

  local cpu_fraction="${CP_CPU_FRACTION:-75}"
  local usable_cores=$(( nproc * cpu_fraction / 100 ))
  [ "$usable_cores" -lt 1 ] && usable_cores=1

  max_jobs=$(( max_by_ram < usable_cores ? max_by_ram : usable_cores ))

  # Logs to stderr so they don't contaminate the stdout return value
  {
    echo "[INFO] Available RAM: ${mem_free} MiB"
    echo "[INFO] Available cores: ${nproc}"
    echo "[INFO] Max jobs (RAM):  ${max_by_ram}"
    echo "[INFO] Max jobs (CPU):  ${usable_cores}"
    echo "[INFO] Effective parallel workers: ${max_jobs}"
  } >&2

  echo "$max_jobs"
}

# ============================================================
#  Number of image sets = CSV rows - 1 (header)
# ============================================================
count_image_sets() {
  local csv="$1"
  echo $(( $(wc -l < "$csv") - 1 ))
}

# ============================================================
#  Launch batches in parallel, respecting MAX_JOBS
# ============================================================
launch_batches() {
  local batch_data="$1"
  local out_root="$2"
  local total_sets="$3"
  local batch_size="$4"
  local max_jobs="$5"

  local start=1 end outdir
  while [ "$start" -le "$total_sets" ]; do
    end=$(( start + batch_size - 1 ))
    [ "$end" -gt "$total_sets" ] && end="$total_sets"

    outdir="$out_root/batch_${start}_${end}"
    mkdir -p "$outdir"

    # Wait if MAX_JOBS are already running (real count, no manual counter)
    while [ "$(jobs -r | wc -l)" -ge "$max_jobs" ]; do
      wait -n
    done

    echo "[INFO] Launching batch: $start → $end"
    cellprofiler -c -r \
      -p "$batch_data" \
      -f "$start" \
      -l "$end" \
      -o "$outdir" &

    start=$(( end + 1 ))
  done

  wait
  echo "[INFO] All batch processing done"
}

# ============================================================
#  Orchestrator: parse flags and delegate
# ============================================================
ejecutar_pipeline() {
  local BATCH_DATA=""
  local ILLUMINATION=1
  local OUT_ROOT=""
  local METADATA_CSV=""
  local USER_BATCH_SIZE=0

  # --- Flag parsing ---
  while [ $# -gt 0 ]; do
    case "$1" in
      -p|--pipeline)   BATCH_DATA="$2";      shift 2 ;;
      -i|--illum)      ILLUMINATION="$2";    shift 2 ;;
      -o|--out)        OUT_ROOT="$2";        shift 2 ;;
      -m|--metadata)   METADATA_CSV="$2";    shift 2 ;;
      -b|--batch_size) USER_BATCH_SIZE="$2"; shift 2 ;;
      -h|--help)
        echo "Usage: ejecutar_pipeline -p <batch.h5> [-i 0|1] [-o <outdir>] [-m <metadata.csv>] [-b <batch_size>]"
        return 0 ;;
      *)
        echo "[ERROR] Unknown flag: $1" >&2
        return 1 ;;
    esac
  done

  # --- Common validations ---
  if [ -z "$BATCH_DATA" ]; then
    echo "[ERROR] Missing -p/--pipeline" >&2
    return 1
  fi
  if [ ! -f "$BATCH_DATA" ]; then
    echo "[ERROR] Pipeline not found: $BATCH_DATA" >&2
    return 1
  fi

  # --- Illumination mode (no splitting) ---
  if [ "$ILLUMINATION" -eq 1 ]; then
    echo "[INFO] Illumination mode (no splitting)"
    cellprofiler -c -r -p "$BATCH_DATA" -o "$OUT_ROOT"
    return
  fi

  # --- Batch mode validations ---
  if [ -z "$OUT_ROOT" ]; then
    echo "[ERROR] Missing -o/--out in batch mode" >&2
    return 1
  fi
  if [ ! -f "$METADATA_CSV" ]; then
    echo "[ERROR] METADATA_CSV not found: $METADATA_CSV" >&2
    return 1
  fi
  if ! [[ "$USER_BATCH_SIZE" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --batch_size must be an integer" >&2
    return 1
  fi

  # --- Parallelism and batch size calculation ---
  local TOTAL_SETS MAX_JOBS BATCH_SIZE
  TOTAL_SETS=$(count_image_sets "$METADATA_CSV")
  MAX_JOBS=$(calculate_max_jobs)

  BATCH_SIZE="$USER_BATCH_SIZE"
  if [ "$BATCH_SIZE" -le 0 ]; then
    BATCH_SIZE=$(( (TOTAL_SETS + MAX_JOBS - 1) / MAX_JOBS ))
  fi

  echo "[INFO] Total image sets: $TOTAL_SETS"
  echo "[INFO] Batch size: $BATCH_SIZE"

  # --- Execution ---
  launch_batches "$BATCH_DATA" "$OUT_ROOT" "$TOTAL_SETS" "$BATCH_SIZE" "$MAX_JOBS"
}