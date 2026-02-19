#!/usr/bin/env bash
set -euo pipefail

# Exportar UID/GID reales
export DOCKER_UID="$(id -u)"
export DOCKER_GID="$(id -g)"
echo $DOCKER_GID $DOCKER_UID
printf "DOCKER_UID=%s\nDOCKER_GID=%s\n" "$(id -u)" "$(id -g)" > .env
# Levantar en background con logs
#nohup docker compose up --build cellprofiler_maince > c3_test.log 2>&1 &
nohup docker compose up > ./logs/container_test.log 2>&1 &
echo "Docker Compose corriendo en background. Logs en docker.log"
