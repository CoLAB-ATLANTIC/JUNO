"""Script to apply the 3 front detection algorithms for any type of latitude/longitude SST product (any resolution) 


It accepts an xarray Dataset with any regular 1-D latitude/longitude grid and returns SST, Canny, BOA and CCA
on the original grid. Spatial tuning is expressed in kilometres rather than in pixels, so changing the 
source resolution does not silently change the scale of the detectors.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

import BOA
import CayulaCornillon_xarray


@dataclass(frozen=True)
class FrontConfig:
    smoothing_km: float = 10.0   # SST smoothing scale; Increase it -> Fewer, smoother fronts
    canny_low: int = 8           # Weak Canny edge acceptance; Increase it -> Removes weak connected edges
    canny_high: int = 16         # Strong Canny edge detection; Increase it -> Requires stronger edges
    boa_threshold_c_per_km: float = 0.1  #Minimum physical SST gradient; Increase it -> Detects only stronger fronts
    cca_base_window_km: float = 90.0   # CCA analysis scale; Increase it -> Focuses on larger structures
    coast_buffer_km: float = 5.0     # Coastal exclusion distance; Increase it -> Removes fronts farther from land


def normalise_sst_dataset(ds: xr.Dataset, variable: str) -> xr.Dataset:
    """Return one time/depth slice named analysed_sst on ascending lat/lon."""
    rename = {}
    for candidate in ("latitude", "nav_lat", "y"):
        if candidate in ds.coords and "lat" not in ds.coords:
            rename[candidate] = "lat"
            break
    for candidate in ("longitude", "nav_lon", "x"):
        if candidate in ds.coords and "lon" not in ds.coords:
            rename[candidate] = "lon"
            break
    ds = ds.rename(rename)
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("The input must contain 1-D latitude and longitude coordinates")
    if ds.lat.ndim != 1 or ds.lon.ndim != 1:
        raise ValueError("Only regular grids with 1-D latitude/longitude are supported")

    da = ds[variable]
    for dim in list(da.dims):
        if dim not in ("lat", "lon"):
            da = da.isel({dim: 0}, drop=True)
    da = da.transpose("lat", "lon").astype("float64")
    if da.attrs.get("units", "").lower() in {"k", "kelvin"} or float(da.median(skipna=True)) > 100:
        da = da - 273.15
        da.attrs["units"] = "degree_Celsius"
    out = da.to_dataset(name="analysed_sst")
    if out.lat[0] > out.lat[-1]:
        out = out.sortby("lat")
    if out.lon[0] > out.lon[-1]:
        out = out.sortby("lon")
    return out


def grid_spacing_km(ds: xr.Dataset) -> tuple[float, float]:
    """Median north/south and east/west cell sizes in kilometres."""
    if ds.lat.size < 2 or ds.lon.size < 2:
        raise ValueError("At least two latitude and longitude cells are required")
    dy = abs(float(np.nanmedian(np.diff(ds.lat.values)))) * 111.195
    mean_lat = float(np.nanmean(ds.lat.values))
    dx = abs(float(np.nanmedian(np.diff(ds.lon.values)))) * 111.195 * np.cos(np.deg2rad(mean_lat))
    if dx <= 0 or dy <= 0:
        raise ValueError("Invalid latitude/longitude spacing")
    return dy, dx


def _coast_mask(sst: np.ndarray, dy: float, dx: float, buffer_km: float) -> np.ndarray:
    radius = max(1, int(round(buffer_km / min(dx, dy))))
    size = 2 * radius + 1
    return cv2.dilate(np.isnan(sst).astype("uint8"), np.ones((size, size), np.uint8)).astype(bool)


def _binary(values: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    answer = np.where(values, 1.0, np.nan)
    answer[invalid] = np.nan
    return answer.astype("float32")


def canny_fronts(sst: np.ndarray, dy: float, dx: float, cfg: FrontConfig) -> np.ndarray:
    valid = np.isfinite(sst)
    if valid.sum() < 2 or np.nanmax(sst) == np.nanmin(sst):
        return np.full(sst.shape, np.nan, dtype="float32")
    filled = np.where(valid, sst, np.nanmedian(sst))
    sigma = (cfg.smoothing_km / dy, cfg.smoothing_km / dx)
    smooth = gaussian_filter(filled, sigma=sigma)
    image = np.clip((smooth - np.nanmin(smooth)) / np.ptp(smooth) * 255, 0, 255).astype("uint8")
    edges = cv2.Canny(image, cfg.canny_low, cfg.canny_high, apertureSize=3, L2gradient=True) > 0
    return _binary(edges, _coast_mask(sst, dy, dx, cfg.coast_buffer_km))


def boa_fronts(sst: np.ndarray, dy: float, dx: float, cfg: FrontConfig) -> np.ndarray:
    """Run the supplied BOA and convert its per-cell gradient to degC/km."""
    lat = np.arange(sst.shape[0], dtype="float64")
    lon = np.arange(sst.shape[1], dtype="float64")
    raw = BOA.boa(lon=lon, lat=lat, ingrid=sst.copy(), nodata=np.nan, direction=False)
    # BOA returns (lon, lat), with longitude reversed; restore (lat, lon).
    magnitude = raw.T[:, ::-1] / np.sqrt(dx * dy)
    fronts = magnitude >= cfg.boa_threshold_c_per_km
    return _binary(fronts, _coast_mask(sst, dy, dx, cfg.coast_buffer_km))


def cca_fronts(ds: xr.Dataset, dy: float, dx: float, cfg: FrontConfig) -> np.ndarray:
    """Run CCA after resampling to a fixed physical window scale, then map back."""
    # The supplied CCA implementation uses a 16-pixel base window. Resampling
    # makes those 16 pixels represent the requested physical size on every grid.
    target_km = cfg.cca_base_window_km / 16.0
    lat_step = target_km / 111.195
    mean_lat = float(ds.lat.mean())
    lon_step = target_km / (111.195 * np.cos(np.deg2rad(mean_lat)))
    lat = np.arange(float(ds.lat.min()), float(ds.lat.max()) + lat_step / 2, lat_step)
    lon = np.arange(float(ds.lon.min()), float(ds.lon.max()) + lon_step / 2, lon_step)
    work = ds.interp(lat=lat, lon=lon).expand_dims(time=[0])
    x, y = CayulaCornillon_xarray.CCA_SIED(work)
    raster = np.zeros((lat.size, lon.size), dtype="uint8")
    if len(x):
        cols = np.abs(lon[:, None] - np.asarray(x)[None, :]).argmin(axis=0)
        rows = np.abs(lat[:, None] - np.asarray(y)[None, :]).argmin(axis=0)
        good = (rows >= 0) & (rows < lat.size) & (cols >= 0) & (cols < lon.size)
        raster[rows[good], cols[good]] = 1
    coarse = xr.DataArray(raster, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    restored = coarse.interp(lat=ds.lat, lon=ds.lon, method="nearest").values > 0
    sst = ds.analysed_sst.values
    return _binary(restored, _coast_mask(sst, dy, dx, cfg.coast_buffer_km))


def detect_fronts(ds: xr.Dataset, variable: str, cfg: FrontConfig = FrontConfig()) -> xr.Dataset:
    time_value = None
    if "time" in ds.coords and ds.coords["time"].size:
        time_value = np.asarray(ds.coords["time"].values).reshape(-1)[0]
    ds = normalise_sst_dataset(ds, variable)
    dy, dx = grid_spacing_km(ds)
    sst = ds.analysed_sst.values
    result = xr.Dataset(
        {
            "sst": ds.analysed_sst.astype("float32"),
            "Canny": (("lat", "lon"), canny_fronts(sst, dy, dx, cfg)),
            "BOA": (("lat", "lon"), boa_fronts(sst, dy, dx, cfg)),
            "CCA": (("lat", "lon"), cca_fronts(ds, dy, dx, cfg)),
        },
        coords={"lat": ds.lat, "lon": ds.lon},
    )
    result.sst.attrs.update(units="degree_Celsius", long_name="sea surface temperature")
    for name in ("Canny", "BOA", "CCA"):
        result[name].attrs.update(flag_values=np.array([1], dtype="int8"), comment="1=front; NaN=no front")
    result.attrs.update(
        title="Resolution-aware SST ocean-front detection",
        source_grid_spacing_km=f"latitude={dy:.4f}, longitude={dx:.4f}",
    )
    if time_value is not None:
        result = result.expand_dims(time=[time_value])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect Canny, BOA and CCA ocean fronts in a regular lat/lon SST NetCDF"
    )
    parser.add_argument("input", type=Path, help="Input SST NetCDF")
    parser.add_argument("output", type=Path, help="Output front NetCDF")
    parser.add_argument("--variable", required=True, help="SST variable, e.g. analysed_sst or thetao")
    parser.add_argument("--smoothing-km", type=float, default=5.0)
    parser.add_argument("--boa-threshold", type=float, default=0.055, help="degrees Celsius per kilometre")
    parser.add_argument("--cca-window-km", type=float, default=90.0)
    parser.add_argument("--coast-buffer-km", type=float, default=5.0)
    args = parser.parse_args()

    config = FrontConfig(
        smoothing_km=args.smoothing_km,
        boa_threshold_c_per_km=args.boa_threshold,
        cca_base_window_km=args.cca_window_km,
        coast_buffer_km=args.coast_buffer_km,
    )
    with xr.open_dataset(args.input) as source:
        result = detect_fronts(source.load(), args.variable, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_netcdf(args.output)


if __name__ == "__main__":
    main()