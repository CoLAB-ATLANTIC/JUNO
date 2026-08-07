"""SST input data resolution independent ocean front detection for regular latitude/longitude SST grids.

The public entry point is ``detect_fronts``.  It accepts an xarray Dataset with
any regular 1-D latitude/longitude grid and returns SST, Canny, BOA and CCA on
the *original* grid. Spatial tuning is expressed in kilometres rather than in
pixels, so changing the source resolution does not silently change the scale
of the detectors.

We can run this script from the command line with:
    python src/resolution_independent_fronts.py INPUT.nc OUTPUT.nc --variable analysed_sst
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
    smoothing_km: float = 5.0
    canny_low: int = 10
    canny_high: int = 15
    boa_threshold_c_per_km: float = 0.055
    cca_base_window_km: float = 90.0
    coast_buffer_km: float = 10.0


LAT_NAMES = ("lat", "latitude", "nav_lat", "y")
LON_NAMES = ("lon", "longitude", "nav_lon", "x")
TIME_NAMES = ("time",)
DEPTH_NAMES = ("depth", "deptht", "depthu", "depthv", "depthw", "z", "lev", "level")
SST_VARIABLE_NAMES = ("analysed_sst", "thetao", "temperature", "temp", "sst")


def _first_existing_name(ds: xr.Dataset, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in ds.coords or name in ds.dims:
            return name
    return None


def _coord_endpoint(coord: xr.DataArray, index: int) -> float:
    return float(np.asarray(coord.values).reshape(-1)[index])


def _finite_median(da: xr.DataArray) -> float:
    values = np.asarray(da.values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("The SST variable does not contain any finite values")
    return float(np.median(finite))


def _normalised_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _is_depth_dim(dim: str) -> bool:
    name = _normalised_name(dim)
    return name in DEPTH_NAMES or "depth" in name


def _is_time_dim(dim: str) -> bool:
    return _normalised_name(dim) in TIME_NAMES


def _select_surface_index(da: xr.DataArray, dim: str) -> int:
    if dim not in da.coords:
        return 0

    values = np.asarray(da.coords[dim].values, dtype="float64").reshape(-1)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return 0

    finite_indexes = np.where(np.isfinite(values))[0]
    finite_values = values[finite_indexes]
    return int(finite_indexes[np.argmin(np.abs(finite_values))])


def infer_sst_variable(ds: xr.Dataset) -> str:
    """Return a likely SST variable name when one is not supplied."""
    for name in SST_VARIABLE_NAMES:
        if name in ds.data_vars:
            return name

    lower_lookup = {_normalised_name(name): name for name in ds.data_vars}
    for name in SST_VARIABLE_NAMES:
        normalised = _normalised_name(name)
        if normalised in lower_lookup:
            return lower_lookup[normalised]

    raise ValueError(
        "Could not infer an SST variable. Use --variable with one of the "
        f"available variables: {', '.join(ds.data_vars)}"
    )


def resolve_sst_variable(ds: xr.Dataset, variable: str | None) -> str:
    if variable is None:
        return infer_sst_variable(ds)
    if variable in ds.data_vars:
        return variable

    normalised = _normalised_name(variable)
    for name in ds.data_vars:
        if _normalised_name(name) == normalised:
            return name

    raise ValueError(f"Variable {variable!r} was not found in the input dataset")


def _validate_config(cfg: FrontConfig) -> None:
    if cfg.smoothing_km < 0:
        raise ValueError("smoothing_km must be greater than or equal to 0")
    if cfg.coast_buffer_km < 0:
        raise ValueError("coast_buffer_km must be greater than or equal to 0")
    if cfg.cca_base_window_km <= 0:
        raise ValueError("cca_base_window_km must be greater than 0")
    if cfg.boa_threshold_c_per_km < 0:
        raise ValueError("boa_threshold_c_per_km must be greater than or equal to 0")
    if not 0 <= cfg.canny_low <= 255 or not 0 <= cfg.canny_high <= 255:
        raise ValueError("Canny thresholds must be in the range 0..255")
    if cfg.canny_low >= cfg.canny_high:
        raise ValueError("canny_low must be smaller than canny_high")


def normalise_sst_dataset(ds: xr.Dataset, variable: str | None = None) -> xr.Dataset:
    """Return one time/depth slice named analysed_sst on ascending lat/lon."""
    variable = resolve_sst_variable(ds, variable)

    rename = {}
    lat_name = _first_existing_name(ds, LAT_NAMES)
    lon_name = _first_existing_name(ds, LON_NAMES)
    if lat_name and lat_name != "lat":
        rename[lat_name] = "lat"
    if lon_name and lon_name != "lon":
        rename[lon_name] = "lon"
    ds = ds.rename(rename)
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("The input must contain 1-D latitude and longitude coordinates")
    if ds.lat.ndim != 1 or ds.lon.ndim != 1:
        raise ValueError("Only regular grids with 1-D latitude/longitude are supported")

    da = ds[variable]
    for dim in list(da.dims):
        if dim not in ("lat", "lon"):
            if _is_depth_dim(dim):
                da = da.isel({dim: _select_surface_index(da, dim)}, drop=True)
            elif _is_time_dim(dim):
                da = da.isel({dim: 0}, drop=True)
            else:
                da = da.isel({dim: 0}, drop=True)
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"Variable {variable!r} must have latitude and longitude dimensions")
    da = da.transpose("lat", "lon").astype("float64")
    units = da.attrs.get("units", "").strip().lower()
    if units in {"k", "kelvin", "degrees_kelvin"} or _finite_median(da) > 100:
        da = da - 273.15
        da.attrs["units"] = "degree_Celsius"
    out = da.to_dataset(name="analysed_sst")
    if _coord_endpoint(out.lat, 0) > _coord_endpoint(out.lat, -1):
        out = out.sortby("lat")
    if _coord_endpoint(out.lon, 0) > _coord_endpoint(out.lon, -1):
        out = out.sortby("lon")
    return out


def _regular_spacing_degrees(coord: xr.DataArray, name: str) -> float:
    values = np.asarray(coord.values, dtype="float64")
    if values.size < 2:
        raise ValueError(f"At least two {name} cells are required")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"The {name} coordinate contains non-finite values")

    diffs = np.diff(values)
    if np.any(diffs == 0) or not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError(f"The {name} coordinate must be strictly monotonic")

    abs_diffs = np.abs(diffs)
    spacing = float(np.median(abs_diffs))
    if not np.allclose(abs_diffs, spacing, rtol=0.05, atol=1e-9):
        raise ValueError(f"The {name} coordinate must be regular within 5 percent")
    return spacing


def grid_spacing_km(ds: xr.Dataset) -> tuple[float, float]:
    """Median north/south and east/west cell sizes in kilometres."""
    dy = _regular_spacing_degrees(ds.lat, "latitude") * 111.195
    mean_lat = float(np.nanmean(ds.lat.values))
    longitude_scale = np.cos(np.deg2rad(mean_lat))
    if longitude_scale <= 0.001:
        raise ValueError("Longitude spacing is too small near the poles for this lat/lon approximation")
    dx = _regular_spacing_degrees(ds.lon, "longitude") * 111.195 * longitude_scale
    if dx <= 0 or dy <= 0:
        raise ValueError("Invalid latitude/longitude spacing")
    return dy, dx


def _coast_mask(sst: np.ndarray, dy: float, dx: float, buffer_km: float) -> np.ndarray:
    invalid = np.isnan(sst)
    if buffer_km == 0 or not invalid.any():
        return invalid
    radius = max(1, int(round(buffer_km / min(dx, dy))))
    size = 2 * radius + 1
    return cv2.dilate(invalid.astype("uint8"), np.ones((size, size), np.uint8)).astype(bool)


def _binary(values: np.ndarray, invalid: np.ndarray) -> np.ndarray:
    answer = np.where(values, 1.0, np.nan)
    answer[invalid] = np.nan
    return answer.astype("float32")


def canny_fronts(sst: np.ndarray, dy: float, dx: float, cfg: FrontConfig) -> np.ndarray:
    valid = np.isfinite(sst)
    if valid.sum() < 2:
        return np.full(sst.shape, np.nan, dtype="float32")
    valid_values = sst[valid]
    if np.ptp(valid_values) == 0:
        return np.full(sst.shape, np.nan, dtype="float32")
    filled = np.where(valid, sst, np.median(valid_values))
    sigma = (cfg.smoothing_km / dy, cfg.smoothing_km / dx)
    smooth = gaussian_filter(filled, sigma=sigma)
    smooth_range = np.ptp(smooth)
    if smooth_range == 0:
        return np.full(sst.shape, np.nan, dtype="float32")
    image = np.clip((smooth - np.nanmin(smooth)) / smooth_range * 255, 0, 255).astype("uint8")
    edges = cv2.Canny(image, cfg.canny_low, cfg.canny_high, apertureSize=3, L2gradient=True) > 0
    return _binary(edges, _coast_mask(sst, dy, dx, cfg.coast_buffer_km))


def boa_fronts(sst: np.ndarray, dy: float, dx: float, cfg: FrontConfig) -> np.ndarray:
    """Run the supplied BOA and convert its per-cell gradient to degC/km."""
    valid = np.isfinite(sst)
    if valid.sum() < 2 or np.ptp(sst[valid]) == 0:
        return np.full(sst.shape, np.nan, dtype="float32")
    lat = np.arange(sst.shape[0], dtype="float64")
    lon = np.arange(sst.shape[1], dtype="float64")
    raw = BOA.boa(lon=lon, lat=lat, ingrid=sst.copy(), nodata=np.nan, direction=False)
    # BOA returns (lon, lat), with longitude reversed; restore (lat, lon).
    pixel_size_km = (dx + dy) / 2.0
    magnitude = raw.T[:, ::-1] / pixel_size_km
    fronts = magnitude >= cfg.boa_threshold_c_per_km
    return _binary(fronts, _coast_mask(sst, dy, dx, cfg.coast_buffer_km))


def cca_fronts(ds: xr.Dataset, dy: float, dx: float, cfg: FrontConfig) -> np.ndarray:
    """Run CCA after resampling to a fixed physical window scale, then map back."""
    if min(ds.lat.size, ds.lon.size) < 2:
        return np.full(ds.analysed_sst.shape, np.nan, dtype="float32")
    # The supplied CCA implementation uses a 16-pixel base window. Resampling
    # makes those 16 pixels represent the requested physical size on every grid.
    target_km = cfg.cca_base_window_km / 16.0
    lat_step = target_km / 111.195
    mean_lat = float(ds.lat.mean())
    longitude_scale = np.cos(np.deg2rad(mean_lat))
    if longitude_scale <= 0.001:
        raise ValueError("CCA longitude resampling is not valid near the poles")
    lon_step = target_km / (111.195 * longitude_scale)
    lat = np.arange(float(ds.lat.min()), float(ds.lat.max()) + lat_step / 2, lat_step)
    lon = np.arange(float(ds.lon.min()), float(ds.lon.max()) + lon_step / 2, lon_step)
    if lat.size < 2 or lon.size < 2:
        return np.full(ds.analysed_sst.shape, np.nan, dtype="float32")
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


def detect_fronts(ds: xr.Dataset, variable: str | None = None, cfg: FrontConfig = FrontConfig()) -> xr.Dataset:
    _validate_config(cfg)
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
    parser.add_argument(
        "--variable",
        help="SST variable, e.g. analysed_sst, thetao, temperature or sst. If omitted, a common SST name is inferred.",
    )
    parser.add_argument("--smoothing-km", type=float, default=5.0)
    parser.add_argument("--canny-low", type=int, default=5)
    parser.add_argument("--canny-high", type=int, default=15)
    parser.add_argument("--boa-threshold", type=float, default=0.055, help="degrees Celsius per kilometre")
    parser.add_argument("--cca-window-km", type=float, default=90.0)
    parser.add_argument("--coast-buffer-km", type=float, default=10.0)
    args = parser.parse_args()

    config = FrontConfig(
        smoothing_km=args.smoothing_km,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
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
