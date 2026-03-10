#!/usr/bin/env bash
set -euo pipefail

update_or_add() {
    local key=$1
    local value=$2
    local file=$3
    if grep -q "^${key}=" "$file"; then
        sed -i "s|^${key}=.*|${key}=${value}|" "$file"
    else
        echo "${key}=${value}" >> "$file"
    fi
}

create_variables() {
    cat > variables.env << EOF
COHORT=${COHORT}
WELLS_PER_PLATE=${WELLS_PER_PLATE}

ILLUM_PIPE=${ILLUM_PIPE}
PROFILING_PIPE=${PROFILING_PIPE}
QC_PIPE=${QC_PIPE}
TEMPLATE_CPPIPE_ILLUM=/workspace/Pipelines/template_${ILLUM_PIPE}
TEMPLATE_CPPIPE_PROFILING=/workspace/Pipelines/template_${PROFILING_PIPE}
TEMPLATE_CPPIPE_QC=/workspace/Pipelines/template_${QC_PIPE}

IMAGES_WORKSPACE=/workspace_images
OUTPUT=/output
PATH_PLATEMAP=/workspace_images/platemap_${COHORT}.csv

SCRIPT_PY_CELLPROFILER=/workspace/Code/main.py
SCRIPT_PY_CELLPOSE=/workspace/Code/II_cellpose_seg.py
SCRIPT_PY_COLLAGE=/workspace/Code/III_QC_collage.py
SCRIPT_PY_FEAT_PROCESS=/workspace/Code/V_feat_processing.py
SCRIPT_PY_CLUSTERING=/workspace/Code/VI_Clustering.py
EOF
    echo "Variables.env generated"
    chmod 755 ./variables.env
}

update_or_add "DOCKER_UID" "$(id -u)" .env
update_or_add "DOCKER_GID" "$(id -g)" .env

source .env

create_variables

LOG_FILE="./logs/${COHORT}.log"
mkdir -p "$(dirname "$LOG_FILE")"

nohup docker compose up > "$LOG_FILE" 2>&1 &
echo "Docker compose running in background. Log saved in ${LOG_FILE}"
