#!/usr/bin/env bash
set -euo pipefail

# Exportar UID/GID reales
export DOCKER_UID="$(id -u)"
export DOCKER_GID="$(id -g)"
echo $DOCKER_GID $DOCKER_UID
printf "DOCKER_UID=%s\nDOCKER_GID=%s\n" "$(id -u)" "$(id -g)" wha> .env
nohup docker compose up > ./logs/cohort2.log 2>&1 &
echo "Docker compose running in background. Logs under ./logs"
