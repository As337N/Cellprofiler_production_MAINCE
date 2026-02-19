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
        PATH_CSV="$OUTPUT/CellProfiler_files/CSVs"
        PATH_ILLUM_FILES="$OUTPUT/CellProfiler_files/Illum_files"
        PATH_CPPIPE="$OUTPUT/CellProfiler_files/Pipelines"
        PATH_BATCH_PIPELINES="$OUTPUT/CellProfiler_files/Batch_files"
        PATH_PROFILES="$OUTPUT/CellProfiler_files/MP"

        PATH_QC_IMAGES="$OUTPUT/QC/Images"
        PATH_QC_COLLAGES="$OUTPUT/QC/Collages"
        PATH_QC_REPORTS="$OUTPUT/QC/Reports"

        PATH_FINAL_PROFILES="$OUTPUT/Profiles/Treated_profiles"
        PATH_CLUSTERS="$OUTPUT/Clustering"

        for folder in \
            "$PATH_CELLPOSE_SEG" \
            "$PATH_CSV" \
            "$PATH_ILLUM_FILES" \
            "$PATH_CPPIPE" \
            "$PATH_BATCH_PIPELINES" \
            "$PATH_PROFILES" \
            "$PATH_QC_IMAGES" \
            "$PATH_QC_COLLAGES" \
            "$PATH_QC_REPORTS" \
            "$PATH_FINAL_PROFILES" \
            "$PATH_CLUSTERS"
        do
            mkdir -p "$folder"
        done
    done
}


ejecutar_pipeline() {
  local BATCH_DATA="$1"
  local ILUMINACION="${2:-1}"
  local OUT_ROOT="${3:-}"
  local METADATA_CSV="${4:-}"
  local USER_BATCH_SIZE="${5:-0}"   # si > 0, sobrescribe el cálculo automático

  # --- MODO ILUMINACIÓN (sin batches) ---
  if [ "$ILUMINACION" -eq 1 ]; then
    echo "[INFO] Modo iluminación (sin fraccionar)"
    cellprofiler -c -r -p "$BATCH_DATA" -o "$OUT_ROOT"
    return
  fi

  # --- Validaciones ---
  if [ ! -f "$METADATA_CSV" ]; then
    echo "[ERROR] No se encontró METADATA_CSV"
    return 1
  fi

  local TOTAL_SETS=$(( $(wc -l < "$METADATA_CSV") - 1 ))

  # ================================================================
  #   DETECCIÓN AUTOMÁTICA DE CPU + RAM PARALELIZACIÓN REAL
  # ================================================================

  # núcleos
  local NPROC=$(nproc)

  # RAM total y libre (en MiB)
  local MEM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{print int($2/1024)}')
  local MEM_FREE=$(grep MemAvailable /proc/meminfo | awk '{print int($2/1024)}')

  echo "[INFO] Total RAM: ${MEM_TOTAL} MiB"
  echo "[INFO] Available RAM: ${MEM_FREE} MiB"
  echo "[INFO] Available cores: $NPROC"

  # ---------- Estimación de RAM por batch ----------
  # regla empírica: CellProfiler suele usar 200–600 MB por batch dependiendo del pipeline
  # ajustable si quieres
  local RAM_PER_BATCH="${CP_RAM_PER_BATCH:-500}"   # en MiB

  # máximo de jobs por RAM
  local MAX_BY_RAM=$(( MEM_FREE / RAM_PER_BATCH ))
  [ "$MAX_BY_RAM" -lt 1 ] && MAX_BY_RAM=1

  # máximo total (limitado también por CPU)
  local MAX_JOBS=$(( MAX_BY_RAM < NPROC ? MAX_BY_RAM : NPROC ))

  echo "[INFO]   Max jobs per RAM: $MAX_BY_RAM"
  echo "[INFO]   Max jobs per CPU: $NPROC"
  echo "[INFO]   Effective parallel works: $MAX_JOBS"

  # ---------- Tamaño automático del batch ----------
  local BATCH_SIZE="$USER_BATCH_SIZE"
  if [ "$BATCH_SIZE" -le 0 ]; then
      BATCH_SIZE=$(( (TOTAL_SETS + MAX_JOBS - 1) / MAX_JOBS ))
  fi

  echo "[INFO] Batch size: $BATCH_SIZE"

  # ================================================================
  #   EJECUCIÓN EN PARALELO AUTOMÁTICO
  # ================================================================
  local start=1
  local running=0

  while [ "$start" -le "$TOTAL_SETS" ]; do
    local end=$(( start + BATCH_SIZE - 1 ))
    [ "$end" -gt "$TOTAL_SETS" ] && end="$TOTAL_SETS"

    local OUTDIR="$OUT_ROOT/batch_${start}_${end}"
    mkdir -p "$OUTDIR"

    echo "[INFO] Launching batch: $start → $end"

    cellprofiler -c -r \
      -p "$BATCH_DATA" \
      -f "$start" \
      -l "$end" \
      -o "$OUTDIR" &

    running=$(( running + 1 ))

    # Si ya hay muchos jobs corriendo, esperar
    if [ "$running" -ge "$MAX_JOBS" ]; then
      wait -n
      running=$(( running - 1 ))
    fi

    start=$(( end + 1 ))
  done

  wait
  echo "[INFO] All batch processing done"
}

