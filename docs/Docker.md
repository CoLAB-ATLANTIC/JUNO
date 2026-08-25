# Docker Deployment

This branch contains a production-oriented Docker setup for the daily JUNO jobs.

The image contains only the code, Python dependencies, and required system tools. Runtime data, logs, credentials, and shapefiles stay outside the image and are mounted into the container.

## Runtime Layout

- `/app`: repository code copied into the image.
- `/data`: persistent JUNO data directory, mounted from the host.
- `/logs`: persistent cron logs, mounted from the host.

By default, `docker-compose.yml` maps:

- `./juno_data:/data`
- `./juno_logs:/logs`

## Environment Variables

Copy `.env.example` to `.env` and fill in the real values.

Required for eddies:

- `EDDIES_USER`
- `EDDIES_PASS`

Required for CMEMS, unless `/data/copernicus_login.txt` exists:

- `COPERNICUS_USERNAME`
- `COPERNICUS_PASSWORD`

Optional path settings:

- `JUNO_DATA_DIR`, default `/data`
- `JUNO_LOG_DIR`, default `/logs`
- `JUNO_COPERNICUS_LOGIN_FILE`, default `/data/copernicus_login.txt`
- `JUNO_ATLANTIC_SHAPEFILE`, default `/data/atlantic_shapefile/aoi_atlantic_clip.shp`
- `TZ`, default `Europe/Lisbon`

## Build And Run

```bash
cp .env.example .env
mkdir -p juno_data juno_logs
docker compose build
docker compose up -d
```

The container starts cron in the foreground. The configured schedule mirrors the existing server crontab:

- `09:15`: MUR daily fronts
- `09:20`: CMEMS daily fronts
- `09:25`: AVISO eddies tracking

The Docker image sets the container timezone from `TZ`, so these cron times are local to that timezone.

## One-Off Commands

These are useful for testing without waiting for cron:

```bash
docker compose run --rm juno mur
docker compose run --rm juno cmems
docker compose run --rm juno eddies
docker compose run --rm juno eddies --start-date 20180101 --end-date 20260810
```

The scheduled eddies cron job processes only the latest AVISO file end date minus 15 days through the latest AVISO file end date, skipping any `eddies_YYYYMMDD.nc` files that already exist in `/data/AVISO_netcdf`. Use `--start-date` and `--end-date` when you want to manually backfill a wider date range.

## Notes

The eddies job requires the `ncks` command, provided by the Debian `nco` package in the image.

The CMEMS clipping step expects the Atlantic shapefile at `/data/atlantic_shapefile/aoi_atlantic_clip.shp` unless `JUNO_ATLANTIC_SHAPEFILE` points elsewhere. Make sure all shapefile sidecar files are present in the same directory, especially `.shp`, `.shx`, `.dbf`, and `.prj`.
