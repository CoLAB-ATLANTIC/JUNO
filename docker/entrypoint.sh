#!/usr/bin/env bash
set -euo pipefail

# ensures the /data and /logs directories exist
export JUNO_BASE_DIR="${JUNO_BASE_DIR:-/app}"
export JUNO_DATA_DIR="${JUNO_DATA_DIR:-/data}"
export JUNO_LOG_DIR="${JUNO_LOG_DIR:-/logs}"

mkdir -p "${JUNO_DATA_DIR}" "${JUNO_LOG_DIR}"

#function to write the environment variables to a file (/etc/cron.d/juno-env) that can be sourced by the cron jobs
#otherwise variables such EDDIE_USER, COPERNICUS_USERNAME, ... might not be visible to cron jobs
write_cron_file() {
  : > /etc/cron.d/juno-env
  while IFS='=' read -r name value; do
    if [[ "${name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      printf 'export %s=%q\n' "${name}" "${value}" >> /etc/cron.d/juno-env
    fi
  done < <(env)
  cat /app/docker/juno.cron > /etc/cron.d/juno
  chmod 0644 /etc/cron.d/juno /etc/cron.d/juno-env
  crontab /etc/cron.d/juno
}

#It also supports running the individual scripts directly by passing the script name as an argument to the container. 
#For example, to run the MUR script directly, we can use: docker compose run --rm juno mur
case "${1:-cron}" in
  cron)
    write_cron_file
    echo "Starting JUNO cron scheduler"
    exec cron -f
    ;;
  mur)
    exec python /app/src/MUR_daily_fronts_netcdf3.py
    ;;
  cmems)
    exec python /app/src/CMEMS_daily_fronts_netcdf.py
    ;;
  eddies)
    exec python /app/src/eddies_cron_complete.py
    ;;
  *)
    exec "$@"
    ;;
esac
