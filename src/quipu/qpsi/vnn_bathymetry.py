"""vnn_bathymetry — Bathymetry, Vertical-Dynamics (v0.33) & Temporal-Spatial Reconstruction.

Mathematical specification
--------------------------
1. **Unified Temporal-Spatial Physical State**:

       S(t) = [ x(t), y(t), z(t), v(t), F(t), B(x,y,t), θ(t), P(t) ]^T

   kept strictly separate from QUIPU's 7-D semantic representation (`mesh_slm_embed`).
   Inverse historical reconstruction operator:

       S(t_0) = R( S(t_1), E, θ ),    p(S | E) ∝ p(E | S) · p(S)

2. **Bathymetric Grid Interpolation & Depth-Dependent Current Attenuation**:
   * Positive water depth: H(x,y,t) = −B(x,y,t)
   * Bilinear interpolation of H(x,y,t) and central-difference slope:
         s(x,y) = ‖∇H(x,y)‖_2 = sqrt( (∂H/∂x)² + (∂H/∂y)² )
   * Depth-dependent horizontal current attenuation:
         U(z) = U_0 · exp(−z / L),     L > 0

3. **Stochastic Surface Drift (Disabled Vessel / Floating Object)**:

       dX_t = ( U_o(X_t, t) + α U_a(X_t, t) + U_w(X_t, t) ) dt + sqrt(2 K) dW_t

   preserving independent displacement accounting for ocean current (U_o),
   atmospheric wind/leeway (α U_a), wave/Stokes drift (U_w), and eddy diffusion.

4. **Vertical Motion & Multi-Regime Hull / Person Physics**:

       (m + m_a) dv/dt = (m − ρ_w V) g − ½ ρ_w C_D A v |v|,     dz/dt = v

   Supports distinct parameter sets for illustrative v0.33 objects (reaching a
   100 m artificial seabed in ~31.48 s), the 1850 iron screw steamer *Helena Sloman*,
   and a person in water, across buoyancy/phase states (`floating`, `flooded_hull`,
   `submerged`, `grounded`) and both point-object contact (z(t) ≥ H(x,y,t)) and
   extended 3-D hull orientation contact (`HullGeometry`).

5. **Temporal Seabed Reclamation, Exner Morphodynamics & Wreck Burial**:
   * Exner sediment continuity:
         (1 − λ_p) ∂B/∂t = −∇ · q_s + S
   * Historical bathymetry back-projection:
         B_{t_0}(x,y) = B_{modern}(x,y) − ∫_{t_0}^{t_{modern}} (∂B/∂t) dt
   * Temporal depression (multiscale relational elevation difference, not a
     quantum-gravitational mechanism):
         D_T(x,y,t_0,t_1) = B(x,y,t_0) − B(x,y,t_1)
   * Burial depth & 7-state wreck lifecycle:
         d_burial(t) = max(0, B(x_w, y_w, t) − Z_top(t))
     tracking `initial_seabed_contact`, `partial_burial`, `scour`,
     `sediment_accumulation`, `structural_collapse`, `fragmentation`, and `re_exposure`.

6. **6-Feature VNN Bathymetric Adapter**:

       (modern_depth_m, historical_depth_m, depth_uncertainty_m,
        sediment_flux_x, sediment_flux_y, seabed_slope)

   Exposed as a typed physical feature block without overwriting 7-D semantic vectors.

7. **Geodesic Reconstruction Graph, Pruned Probability-Mass Ledger & 8-Ensemble Shadow MC**:
   * Geodesic Graph G = (V, E) with maximum-probability Dijkstra search via
     c_ij = −log(p_ij), Geodesic Voronoi partitioning, greedy trajectory
     simplification, fuzzy interpolation, and explicit accounting of pruned
     probability mass.
   * 8-Ensemble Shadow Monte Carlo (`raw_ueqgm_quipu`, `ocean_only`,
     `atmosphere_only`, `combined_forcing`, `vertical_descent`,
     `historical_bathymetry`, `sediment_evolution`, `acquisition`) with
     GeoJSON and CSV exporters.
"""
from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass, field
import heapq
import io
import json
import math
import random
import sqlite3
import statistics
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np


KV_VNN_BATHYMETRY_LATEST: str = "entirety:vnn:bathymetry:latest"

BATHYMETRIC_FEATURE_NAMES: tuple[str, ...] = (
    "modern_depth_m",
    "historical_depth_m",
    "depth_uncertainty_m",
    "sediment_flux_x",
    "sediment_flux_y",
    "seabed_slope",
)

BUOYANCY_PHASE_STATES: tuple[str, ...] = (
    "floating",
    "flooded_hull",
    "submerged",
    "grounded",
)

WRECK_LIFECYCLE_STATES: tuple[str, ...] = (
    "initial_seabed_contact",
    "partial_burial",
    "scour",
    "sediment_accumulation",
    "structural_collapse",
    "fragmentation",
    "re_exposure",
)

SHADOW_ENSEMBLE_NAMES: tuple[str, ...] = (
    "raw_ueqgm_quipu",
    "ocean_only",
    "atmosphere_only",
    "combined_forcing",
    "vertical_descent",
    "historical_bathymetry",
    "sediment_evolution",
    "acquisition",
)

EARTH_RADIUS_M: float = 6_371_008.8


# ---------------------------------------------------------------------------
# 1. Unified Temporal-Spatial Physical State
# ---------------------------------------------------------------------------

@dataclass
class TemporalSpatialState:
    """Unified physical state vector S(t) = [x, y, z, v, F, B, θ, P]^T.

    Kept strictly separate from QUIPU's 7-D semantic embedding (`mesh_slm_embed`).
    Coordinates (x, y) are local easting/northing meters (or projected coordinates),
    with optional geographic (latitude, longitude) metadata; z >= 0 is positive
    downward depth (or negative elevation relative to datum).
    """
    x: float
    y: float
    z: float
    v: tuple[float, float, float] = (0.0, 0.0, 0.0)          # (u, v, w) velocity [m/s]
    forcing: dict[str, float] = field(default_factory=dict)  # environmental forcing F(t)
    seabed_elevation_m: float = -100.0                       # B(x,y,t) [m, negative below sea level]
    parameters: dict[str, float] = field(default_factory=dict)  # physical model params θ(t)
    covariance_diag: tuple[float, ...] = (1.0, 1.0, 0.5)     # state uncertainty P(t) diagonal
    timestamp: float = 0.0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    phase_state: str = "floating"

    @property
    def water_depth_m(self) -> float:
        """Positive water depth H(x,y,t) = -B(x,y,t)."""
        return -float(self.seabed_elevation_m)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "v": [float(c) for c in self.v],
            "forcing": dict(self.forcing),
            "seabed_elevation_m": float(self.seabed_elevation_m),
            "water_depth_m": self.water_depth_m,
            "parameters": dict(self.parameters),
            "covariance_diag": [float(c) for c in self.covariance_diag],
            "timestamp": float(self.timestamp),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "phase_state": self.phase_state,
        }


# ---------------------------------------------------------------------------
# 2. Bathymetric Grid Interpolation & NOAA / GEBCO Data Integration
# ---------------------------------------------------------------------------

@dataclass
class BathymetricGrid:
    """2-D bathymetric grid storing seabed elevation B(x,y,t) [m, negative below datum]
    and depth uncertainty σ_H(x,y) [m]. Positive water depth is H(x,y,t) = -B(x,y,t).
    """
    x_coords: np.ndarray               # shape (Nx,), strictly increasing [m or deg]
    y_coords: np.ndarray               # shape (Ny,), strictly increasing [m or deg]
    elevation_b: np.ndarray            # shape (Nx, Ny), B(x,y,t) <= 0 for submerged seabed
    uncertainty_m: np.ndarray          # shape (Nx, Ny), σ_H(x,y) >= 0
    timestamp: float = 2026.0
    source: str = "synthetic"

    def __post_init__(self) -> None:
        self.x_coords = np.asarray(self.x_coords, dtype=float)
        self.y_coords = np.asarray(self.y_coords, dtype=float)
        self.elevation_b = np.asarray(self.elevation_b, dtype=float)
        self.uncertainty_m = np.asarray(self.uncertainty_m, dtype=float)
        if self.x_coords.ndim != 1 or self.y_coords.ndim != 1:
            raise ValueError("x_coords and y_coords must be 1-D arrays")
        if self.x_coords.size < 2 or self.y_coords.size < 2:
            raise ValueError("BathymetricGrid requires at least 2 points along each axis")
        if np.any(np.diff(self.x_coords) <= 0.0) or np.any(np.diff(self.y_coords) <= 0.0):
            raise ValueError("x_coords and y_coords must be strictly increasing")
        expected_shape = (self.x_coords.size, self.y_coords.size)
        if self.elevation_b.shape != expected_shape:
            raise ValueError(f"elevation_b shape {self.elevation_b.shape} != {expected_shape}")
        if self.uncertainty_m.shape != expected_shape:
            raise ValueError(f"uncertainty_m shape {self.uncertainty_m.shape} != {expected_shape}")

    @property
    def depth_h(self) -> np.ndarray:
        """Positive downward water depth H(x,y,t) = -B(x,y,t)."""
        return -self.elevation_b

    @classmethod
    def uniform(
        cls,
        depth_m: float = 100.0,
        nx: int = 21,
        ny: int = 21,
        x_span: tuple[float, float] = (0.0, 1000.0),
        y_span: tuple[float, float] = (0.0, 1000.0),
        uncertainty_m: float = 0.5,
        timestamp: float = 2026.0,
        source: str = "synthetic_uniform",
    ) -> "BathymetricGrid":
        xs = np.linspace(float(x_span[0]), float(x_span[1]), int(nx))
        ys = np.linspace(float(y_span[0]), float(y_span[1]), int(ny))
        b = np.full((int(nx), int(ny)), -float(depth_m), dtype=float)
        u = np.full((int(nx), int(ny)), max(0.0, float(uncertainty_m)), dtype=float)
        return cls(x_coords=xs, y_coords=ys, elevation_b=b, uncertainty_m=u, timestamp=timestamp, source=source)

    @classmethod
    def sloped_shelf(
        cls,
        base_depth_m: float = 100.0,
        slope_x: float = 0.02,
        slope_y: float = -0.01,
        nx: int = 21,
        ny: int = 21,
        x_span: tuple[float, float] = (0.0, 1000.0),
        y_span: tuple[float, float] = (0.0, 1000.0),
        uncertainty_m: float = 1.2,
        timestamp: float = 2026.0,
        source: str = "synthetic_shelf",
    ) -> "BathymetricGrid":
        xs = np.linspace(float(x_span[0]), float(x_span[1]), int(nx))
        ys = np.linspace(float(y_span[0]), float(y_span[1]), int(ny))
        x_mid = 0.5 * (xs[0] + xs[-1])
        y_mid = 0.5 * (ys[0] + ys[-1])
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        depth = float(base_depth_m) + float(slope_x) * (xx - x_mid) + float(slope_y) * (yy - y_mid)
        b = -np.maximum(1.0, depth)
        u = np.full_like(b, max(0.0, float(uncertainty_m)))
        return cls(x_coords=xs, y_coords=ys, elevation_b=b, uncertainty_m=u, timestamp=timestamp, source=source)

    def _bilinear_sample(self, field_2d: np.ndarray, x: float, y: float) -> float:
        xc = float(np.clip(x, self.x_coords[0], self.x_coords[-1]))
        yc = float(np.clip(y, self.y_coords[0], self.y_coords[-1]))
        ix = int(np.searchsorted(self.x_coords, xc, side="right")) - 1
        iy = int(np.searchsorted(self.y_coords, yc, side="right")) - 1
        ix = max(0, min(self.x_coords.size - 2, ix))
        iy = max(0, min(self.y_coords.size - 2, iy))

        x0, x1 = float(self.x_coords[ix]), float(self.x_coords[ix + 1])
        y0, y1 = float(self.y_coords[iy]), float(self.y_coords[iy + 1])
        tx = (xc - x0) / max(1e-12, x1 - x0)
        ty = (yc - y0) / max(1e-12, y1 - y0)

        f00 = float(field_2d[ix, iy])
        f10 = float(field_2d[ix + 1, iy])
        f01 = float(field_2d[ix, iy + 1])
        f11 = float(field_2d[ix + 1, iy + 1])
        return (
            (1.0 - tx) * (1.0 - ty) * f00
            + tx * (1.0 - ty) * f10
            + (1.0 - tx) * ty * f01
            + tx * ty * f11
        )

    def interpolate_elevation(self, x: float, y: float) -> float:
        """Bilinearly interpolate seabed elevation B(x,y,t) [m]."""
        return self._bilinear_sample(self.elevation_b, x, y)

    def interpolate_depth(self, x: float, y: float) -> float:
        """Bilinearly interpolate positive water depth H(x,y,t) = -B(x,y,t) [m]."""
        return -self.interpolate_elevation(x, y)

    def interpolate_uncertainty(self, x: float, y: float) -> float:
        """Bilinearly interpolate depth uncertainty σ_H(x,y) [m]."""
        return max(0.0, self._bilinear_sample(self.uncertainty_m, x, y))

    def depth_gradient(self, x: float, y: float, eps: Optional[float] = None) -> tuple[float, float]:
        """Central-difference gradient ∇H(x,y) = (∂H/∂x, ∂H/∂y)."""
        dx_grid = float(np.mean(np.diff(self.x_coords)))
        dy_grid = float(np.mean(np.diff(self.y_coords)))
        hx = float(eps) if eps is not None else max(1e-4, 0.25 * dx_grid)
        hy = float(eps) if eps is not None else max(1e-4, 0.25 * dy_grid)

        x_lo = max(float(self.x_coords[0]), float(x) - hx)
        x_hi = min(float(self.x_coords[-1]), float(x) + hx)
        y_lo = max(float(self.y_coords[0]), float(y) - hy)
        y_hi = min(float(self.y_coords[-1]), float(y) + hy)

        dh_dx = (self.interpolate_depth(x_hi, y) - self.interpolate_depth(x_lo, y)) / max(1e-12, x_hi - x_lo)
        dh_dy = (self.interpolate_depth(x, y_hi) - self.interpolate_depth(x, y_lo)) / max(1e-12, y_hi - y_lo)
        return float(dh_dx), float(dh_dy)

    def seabed_slope(self, x: float, y: float) -> float:
        """Seabed slope magnitude s(x,y) = ‖∇H(x,y)‖_2."""
        dh_dx, dh_dy = self.depth_gradient(x, y)
        return math.hypot(dh_dx, dh_dy)


def load_esri_ascii_grid(
    ascii_text: str,
    *,
    source: str = "NOAA",
    default_uncertainty_m: float = 1.5,
    timestamp: float = 2026.0,
) -> BathymetricGrid:
    """Parse an ESRI ASCII raster grid (standard NOAA NCEI / ETOPO export format)
    into a `BathymetricGrid`.
    """
    lines = [ln.strip() for ln in ascii_text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    header: dict[str, float] = {}
    data_start = 0
    for idx, ln in enumerate(lines):
        parts = ln.split()
        if len(parts) == 2 and parts[0].lower() in {
            "ncols", "nrows", "xllcorner", "yllcorner", "xllcenter", "yllcenter", "cellsize", "nodata_value"
        }:
            header[parts[0].lower()] = float(parts[1])
            data_start = idx + 1
        else:
            break

    ncols = int(header.get("ncols", 0))
    nrows = int(header.get("nrows", 0))
    cellsize = float(header.get("cellsize", 1.0))
    nodata = header.get("nodata_value", -9999.0)
    if ncols < 2 or nrows < 2:
        raise ValueError("ESRI ASCII grid requires ncols >= 2 and nrows >= 2")

    x0 = float(header.get("xllcenter", header.get("xllcorner", 0.0)))
    y0 = float(header.get("yllcenter", header.get("yllcorner", 0.0)))
    xs = x0 + np.arange(ncols, dtype=float) * cellsize
    ys = y0 + np.arange(nrows, dtype=float) * cellsize

    tokens: list[float] = []
    for ln in lines[data_start:]:
        for tok in ln.split():
            tokens.append(float(tok))
    if len(tokens) != nrows * ncols:
        raise ValueError(f"Expected {nrows * ncols} raster values, got {len(tokens)}")

    # ESRI ASCII rows are ordered north-to-south (top-to-bottom); flip so y increases
    raster = np.array(tokens, dtype=float).reshape((nrows, ncols))[::-1, :]
    valid = raster != nodata
    fill_val = float(np.median(raster[valid])) if np.any(valid) else -100.0
    raster = np.where(valid, raster, fill_val)
    # Transpose from (nrows=Ny, ncols=Nx) to (Nx, Ny)
    elev_b = raster.T
    uncert = np.where(valid.T, float(default_uncertainty_m), float(default_uncertainty_m) * 3.0)
    return BathymetricGrid(
        x_coords=xs,
        y_coords=ys,
        elevation_b=elev_b,
        uncertainty_m=uncert,
        timestamp=timestamp,
        source=source,
    )


def load_xyz_csv_grid(
    csv_text: str,
    *,
    source: str = "GEBCO",
    default_uncertainty_m: float = 3.5,
    timestamp: float = 2026.0,
) -> BathymetricGrid:
    """Parse a rectilinear XYZ / CSV bathymetry table (`x,y,elevation_b[,uncertainty_m]`)
    used by GEBCO and NOAA bathymetric extracts.
    """
    reader = csv.reader(io.StringIO(csv_text.strip()))
    rows: list[tuple[float, float, float, float]] = []
    for raw in reader:
        if not raw or raw[0].strip().startswith("#"):
            continue
        try:
            x = float(raw[0])
            y = float(raw[1])
            b = float(raw[2])
            u = float(raw[3]) if len(raw) > 3 and raw[3].strip() else float(default_uncertainty_m)
            rows.append((x, y, b, u))
        except ValueError:
            # Skip header line
            continue

    if not rows:
        raise ValueError("No numeric XYZ bathymetry rows found")
    xs = np.array(sorted({r[0] for r in rows}), dtype=float)
    ys = np.array(sorted({r[1] for r in rows}), dtype=float)
    x_idx = {float(v): i for i, v in enumerate(xs)}
    y_idx = {float(v): j for j, v in enumerate(ys)}
    elev_b = np.full((xs.size, ys.size), np.nan, dtype=float)
    uncert = np.full((xs.size, ys.size), float(default_uncertainty_m), dtype=float)
    for x, y, b, u in rows:
        elev_b[x_idx[x], y_idx[y]] = b
        uncert[x_idx[x], y_idx[y]] = max(1e-6, u)

    if np.any(np.isnan(elev_b)):
        med = float(np.nanmedian(elev_b))
        elev_b = np.where(np.isnan(elev_b), med, elev_b)
    return BathymetricGrid(
        x_coords=xs,
        y_coords=ys,
        elevation_b=elev_b,
        uncertainty_m=uncert,
        timestamp=timestamp,
        source=source,
    )


def fuse_bathymetric_grids(
    primary: BathymetricGrid,
    secondary: BathymetricGrid,
    *,
    source_label: str = "fused_NOAA_GEBCO",
) -> BathymetricGrid:
    """Fuse two bathymetric grids (e.g., high-resolution NOAA multibeam and regional
    GEBCO compilation) onto `primary`'s coordinate lattice using Bayesian inverse-variance
    weighting:
        σ_fused^{-2} = σ_1^{-2} + σ_2^{-2}
        B_fused      = σ_fused^2 · (σ_1^{-2} B_1 + σ_2^{-2} B_2)
    """
    nx, ny = primary.x_coords.size, primary.y_coords.size
    b_fused = np.zeros((nx, ny), dtype=float)
    u_fused = np.zeros((nx, ny), dtype=float)

    for i, x in enumerate(primary.x_coords):
        for j, y in enumerate(primary.y_coords):
            b1 = float(primary.elevation_b[i, j])
            s1 = max(1e-3, float(primary.uncertainty_m[i, j]))
            b2 = secondary.interpolate_elevation(float(x), float(y))
            s2 = max(1e-3, secondary.interpolate_uncertainty(float(x), float(y)))
            w1 = 1.0 / (s1 * s1)
            w2 = 1.0 / (s2 * s2)
            var_f = 1.0 / (w1 + w2)
            b_fused[i, j] = var_f * (w1 * b1 + w2 * b2)
            u_fused[i, j] = math.sqrt(var_f)

    return BathymetricGrid(
        x_coords=primary.x_coords.copy(),
        y_coords=primary.y_coords.copy(),
        elevation_b=b_fused,
        uncertainty_m=u_fused,
        timestamp=primary.timestamp,
        source=source_label,
    )


# ---------------------------------------------------------------------------
# 3. Depth-Dependent Current Attenuation & Stochastic Surface Drift
# ---------------------------------------------------------------------------

def attenuate_current(
    u_surface: Sequence[float] | np.ndarray,
    depth_z_m: float,
    attenuation_length_m: float = 40.0,
) -> np.ndarray:
    """Depth-dependent horizontal current attenuation:
        U(z) = U_0 · exp(−z / L)
    for positive downward depth z >= 0 and characteristic attenuation depth L > 0.
    """
    if not (math.isfinite(attenuation_length_m) and attenuation_length_m > 0.0):
        raise ValueError("attenuation_length_m must be finite and > 0")
    u0 = np.asarray(u_surface, dtype=float)
    z_eff = max(0.0, float(depth_z_m))
    return u0 * math.exp(-z_eff / float(attenuation_length_m))


@dataclass(frozen=True)
class SurfaceDriftConfig:
    """Configuration for stochastic surface drift:
        dX_t = (U_o + α U_a + U_w) dt + sqrt(2 K) dW_t
    """
    leeway_alpha: float = 0.03         # wind leeway coefficient α
    diffusivity_k_m2_s: tuple[float, float] = (2.5, 2.5)  # eddy diffusivity diag(K_x, K_y) [m²/s]
    dt_s: float = 60.0                 # integration step [s]
    enable_ocean: bool = True
    enable_atmosphere: bool = True
    enable_waves: bool = True
    enable_diffusion: bool = True
    seed: int = 42


def step_surface_drift(
    position_xy: tuple[float, float],
    u_ocean: tuple[float, float],
    u_wind: tuple[float, float],
    u_wave: tuple[float, float],
    cfg: SurfaceDriftConfig = SurfaceDriftConfig(),
    *,
    step_index: int = 0,
) -> dict[str, Any]:
    """Advance horizontal surface drift by one step Δt while recording the separate
    ocean, atmospheric leeway, wave/Stokes, and stochastic diffusion displacements.
    """
    dt = float(cfg.dt_s)
    if dt <= 0.0:
        raise ValueError("dt_s must be > 0")
    rng = random.Random(int(cfg.seed) ^ (int(step_index) * 1000003))

    uo = (float(u_ocean[0]), float(u_ocean[1])) if cfg.enable_ocean else (0.0, 0.0)
    ua = (
        float(cfg.leeway_alpha) * float(u_wind[0]),
        float(cfg.leeway_alpha) * float(u_wind[1]),
    ) if cfg.enable_atmosphere else (0.0, 0.0)
    uw = (float(u_wave[0]), float(u_wave[1])) if cfg.enable_waves else (0.0, 0.0)

    if cfg.enable_diffusion:
        kx = max(0.0, float(cfg.diffusivity_k_m2_s[0]))
        ky = max(0.0, float(cfg.diffusivity_k_m2_s[1]))
        dx_diff = math.sqrt(2.0 * kx * dt) * rng.gauss(0.0, 1.0)
        dy_diff = math.sqrt(2.0 * ky * dt) * rng.gauss(0.0, 1.0)
    else:
        dx_diff, dy_diff = 0.0, 0.0

    delta_ocean = (uo[0] * dt, uo[1] * dt)
    delta_atmosphere = (ua[0] * dt, ua[1] * dt)
    delta_wave = (uw[0] * dt, uw[1] * dt)
    delta_diffusion = (dx_diff, dy_diff)

    next_x = float(position_xy[0]) + delta_ocean[0] + delta_atmosphere[0] + delta_wave[0] + delta_diffusion[0]
    next_y = float(position_xy[1]) + delta_ocean[1] + delta_atmosphere[1] + delta_wave[1] + delta_diffusion[1]

    return {
        "position_xy": (next_x, next_y),
        "delta_ocean": delta_ocean,
        "delta_atmosphere": delta_atmosphere,
        "delta_wave": delta_wave,
        "delta_diffusion": delta_diffusion,
        "total_displacement": (next_x - float(position_xy[0]), next_y - float(position_xy[1])),
    }


# ---------------------------------------------------------------------------
# 4. Vertical Motion, Multi-Regime Physics & Extended Hull Contact
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HullGeometry:
    """Extended 3-D hull geometry for orientation-aware seabed contact detection."""
    length_m: float = 62.5             # overall length (LOA) [m]
    beam_m: float = 8.4                # beam [m]
    draft_m: float = 4.6               # hull draft from waterline to keel [m]
    pitch_rad: float = 0.0             # bow-down / stern-down pitch angle [rad]
    roll_rad: float = 0.0              # port / starboard roll angle [rad]
    yaw_rad: float = 0.0               # heading angle from +x axis [rad]

    def control_points_local(self) -> list[tuple[str, float, float, float]]:
        """Return 9 structural control points (label, dx_local, dy_local, dz_downward)
        relative to the hull center of mass.
        """
        hl = 0.5 * float(self.length_m)
        hb = 0.5 * float(self.beam_m)
        hd = 0.5 * float(self.draft_m)
        return [
            ("keel_center", 0.0, 0.0, hd),
            ("bow_keel", hl, 0.0, hd),
            ("stern_keel", -hl, 0.0, hd),
            ("port_mid", 0.0, hb, hd),
            ("starboard_mid", 0.0, -hb, hd),
            ("bow_port", 0.7 * hl, 0.7 * hb, hd),
            ("bow_starboard", 0.7 * hl, -0.7 * hb, hd),
            ("stern_port", -0.7 * hl, 0.7 * hb, hd),
            ("stern_starboard", -0.7 * hl, -0.7 * hb, hd),
        ]

    def world_control_points(self, x: float, y: float, z: float) -> list[tuple[str, float, float, float]]:
        """Transform local hull control points into world (x_w, y_w, z_w) coordinates
        using yaw, pitch, and roll rotations (z_w positive downward).
        """
        cp, sp = math.cos(self.pitch_rad), math.sin(self.pitch_rad)
        cr, sr = math.cos(self.roll_rad), math.sin(self.roll_rad)
        cy, sy = math.cos(self.yaw_rad), math.sin(self.yaw_rad)

        pts: list[tuple[str, float, float, float]] = []
        for label, lx, ly, lz in self.control_points_local():
            # Roll about longitudinal x-axis, then pitch about y-axis, then yaw about z-axis
            y1 = cr * ly - sr * lz
            z1 = sr * ly + cr * lz
            x2 = cp * lx + sp * z1
            z2 = -sp * lx + cp * z1
            xw = float(x) + (cy * x2 - sy * y1)
            yw = float(y) + (sy * x2 + cy * y1)
            zw = float(z) + z2
            pts.append((label, xw, yw, zw))
        return pts


@dataclass(frozen=True)
class DescentObjectParams:
    """Physical parameters for the vertical-motion ODE:
        (m + m_a) dv/dt = (m − ρ_w V) g − ½ ρ_w C_D A v |v|
    """
    name: str
    object_class: str                  # "illustrative", "vessel", "person"
    mass_kg: float                     # m > 0
    volume_m3: float                   # displaced volume V > 0
    drag_coeff: float                  # C_D > 0
    projected_area_m2: float           # A > 0
    added_mass_coeff: float = 0.50     # C_a >= 0, so m_a = C_a * ρ_w * V
    rho_water_kg_m3: float = 1025.0    # seawater density ρ_w
    gravity_m_s2: float = 9.81         # gravitational acceleration g
    phase_state: str = "submerged"     # "floating", "flooded_hull", "submerged", "grounded"
    hull_geometry: Optional[HullGeometry] = None

    def validate(self) -> None:
        if self.mass_kg <= 0.0 or self.volume_m3 <= 0.0:
            raise ValueError("mass_kg and volume_m3 must be > 0")
        if self.drag_coeff <= 0.0 or self.projected_area_m2 <= 0.0:
            raise ValueError("drag_coeff and projected_area_m2 must be > 0")
        if self.added_mass_coeff < 0.0:
            raise ValueError("added_mass_coeff must be >= 0")
        if self.phase_state not in BUOYANCY_PHASE_STATES:
            raise ValueError(f"phase_state must be one of {BUOYANCY_PHASE_STATES}")

    @property
    def added_mass_kg(self) -> float:
        return float(self.added_mass_coeff) * float(self.rho_water_kg_m3) * float(self.volume_m3)

    @property
    def effective_mass_kg(self) -> float:
        return float(self.mass_kg) + self.added_mass_kg

    @property
    def net_buoyancy_weight_n(self) -> float:
        """Net downward force (m − ρ_w V) g [N]. Positive = negatively buoyant (sinking)."""
        return (float(self.mass_kg) - float(self.rho_water_kg_m3) * float(self.volume_m3)) * float(self.gravity_m_s2)

    @property
    def terminal_velocity_m_s(self) -> float:
        """Analytical terminal vertical velocity v_T = sign(W_net) * sqrt(2 |W_net| / (ρ_w C_D A))."""
        w_net = self.net_buoyancy_weight_n
        denom = 0.5 * float(self.rho_water_kg_m3) * float(self.drag_coeff) * float(self.projected_area_m2)
        if denom <= 0.0:
            return 0.0
        return math.copysign(math.sqrt(abs(w_net) / denom), w_net)

    def with_phase_state(self, new_phase: str, flood_fraction: float = 1.0) -> "DescentObjectParams":
        """Transition object between buoyancy/phase states (`floating`, `flooded_hull`,
        `submerged`, `grounded`) by updating effective displaced buoyancy volume.
        """
        if new_phase not in BUOYANCY_PHASE_STATES:
            raise ValueError(f"Unknown phase state: {new_phase}")
        ff = max(0.0, min(1.0, float(flood_fraction)))
        if new_phase == "floating":
            # Positively buoyant: displaced volume exceeds mass / rho_w
            vol = max(self.volume_m3, (self.mass_kg / self.rho_water_kg_m3) * 1.15)
        elif new_phase == "flooded_hull":
            # Partially flooded hull: progressive loss of reserve buoyancy
            v_float = (self.mass_kg / self.rho_water_kg_m3) * 1.05
            v_steel = self.mass_kg / 7850.0
            vol = (1.0 - ff) * v_float + ff * max(v_steel * 2.5, self.volume_m3)
        else:
            vol = min(self.volume_m3, (self.mass_kg / self.rho_water_kg_m3) * 0.92)
        return dataclasses.replace(self, volume_m3=vol, phase_state=new_phase)

    @classmethod
    def illustrative_v033(cls) -> "DescentObjectParams":
        """v0.33 synthetic descent benchmark parameters: reaches a 100 m artificial
        seabed in 31.48 seconds (terminal velocity v_T ≈ 3.216 m/s).
        """
        return cls(
            name="illustrative_v033_test_object",
            object_class="illustrative",
            mass_kg=500.0,
            volume_m3=0.15,
            drag_coeff=0.80,
            projected_area_m2=0.8012,
            added_mass_coeff=0.50,
            rho_water_kg_m3=1025.0,
            gravity_m_s2=9.81,
            phase_state="submerged",
            hull_geometry=None,
        )

    @classmethod
    def helena_sloman_hull(
        cls,
        *,
        phase_state: str = "flooded_hull",
        pitch_rad: float = 0.12,
        roll_rad: float = 0.08,
        yaw_rad: float = 0.0,
    ) -> "DescentObjectParams":
        """Physical parameter preset for the 1850 iron screw steamer *Helena Sloman*
        (~800 tons displacement, LOA 62.5 m, beam 8.4 m, draft 4.6 m).
        """
        hull = HullGeometry(
            length_m=62.5,
            beam_m=8.4,
            draft_m=4.6,
            pitch_rad=pitch_rad,
            roll_rad=roll_rad,
            yaw_rad=yaw_rad,
        )
        # Flooded iron hull + machinery + cargo after progressive water ingress
        mass_kg = 820_000.0
        volume_m3 = 520.0 if phase_state in ("flooded_hull", "submerged", "grounded") else 920.0
        return cls(
            name="helena_sloman_1850_iron_steamer",
            object_class="vessel",
            mass_kg=mass_kg,
            volume_m3=volume_m3,
            drag_coeff=1.15,
            projected_area_m2=360.0,
            added_mass_coeff=0.65,
            rho_water_kg_m3=1026.0,
            gravity_m_s2=9.81,
            phase_state=phase_state,
            hull_geometry=hull,
        )

    @classmethod
    def person_in_water(cls, *, positively_buoyant: bool = True) -> "DescentObjectParams":
        """Physical parameter preset for a person in water (distinct from a vessel)."""
        return cls(
            name="person_in_water",
            object_class="person",
            mass_kg=80.0,
            volume_m3=0.082 if positively_buoyant else 0.074,
            drag_coeff=1.05,
            projected_area_m2=0.45,
            added_mass_coeff=0.35,
            rho_water_kg_m3=1025.0,
            gravity_m_s2=9.81,
            phase_state="floating" if positively_buoyant else "submerged",
            hull_geometry=None,
        )


def vertical_acceleration(v_z: float, params: DescentObjectParams) -> float:
    """Compute instantaneous vertical acceleration dv/dt (positive downward):
        dv/dt = [ (m − ρ_w V) g − ½ ρ_w C_D A v |v| ] / (m + m_a)
    """
    params.validate()
    f_buoy_grav = params.net_buoyancy_weight_n
    f_drag = 0.5 * params.rho_water_kg_m3 * params.drag_coeff * params.projected_area_m2 * float(v_z) * abs(float(v_z))
    return (f_buoy_grav - f_drag) / params.effective_mass_kg


def check_seabed_contact(
    x: float,
    y: float,
    z: float,
    grid: BathymetricGrid,
    hull: Optional[HullGeometry] = None,
) -> dict[str, Any]:
    """Evaluate point-object or extended-hull contact with the interpolated seabed:
    * Point-object contact: z(t) >= H(x(t), y(t), t).
    * Extended-hull contact: any hull control point k satisfies z_k(t) >= H(x_k(t), y_k(t), t).
    """
    center_depth = grid.interpolate_depth(x, y)
    center_clearance = center_depth - float(z)
    if hull is None:
        contact = float(z) >= center_depth
        return {
            "contact": bool(contact),
            "mode": "point_object",
            "local_depth_m": center_depth,
            "clearance_m": max(0.0, center_clearance),
            "penetration_m": max(0.0, float(z) - center_depth),
            "contact_fraction": 1.0 if contact else 0.0,
            "contact_points": ["center"] if contact else [],
        }

    pts = hull.world_control_points(x, y, z)
    contact_labels: list[str] = []
    min_clearance = math.inf
    max_penetration = 0.0
    for label, xw, yw, zw in pts:
        h_loc = grid.interpolate_depth(xw, yw)
        clr = h_loc - zw
        if clr < min_clearance:
            min_clearance = clr
        if zw >= h_loc:
            contact_labels.append(label)
            max_penetration = max(max_penetration, zw - h_loc)

    frac = len(contact_labels) / float(len(pts))
    return {
        "contact": len(contact_labels) > 0,
        "mode": "extended_hull",
        "local_depth_m": center_depth,
        "clearance_m": max(0.0, min_clearance),
        "penetration_m": max_penetration,
        "contact_fraction": frac,
        "contact_points": contact_labels,
    }


def simulate_descent(
    params: DescentObjectParams,
    grid: BathymetricGrid,
    *,
    x0: float = 500.0,
    y0: float = 500.0,
    z0: float = 0.0,
    v0: float = 0.0,
    u_surface: tuple[float, float] = (0.6, 0.2),
    attenuation_length_m: float = 40.0,
    dt_s: float = 0.01,
    t_max_s: float = 600.0,
    use_extended_hull: bool = False,
) -> dict[str, Any]:
    """Numerically integrate vertical motion and depth-attenuated horizontal transport:
        (m + m_a) dv/dt = (m − ρ_w V) g − ½ ρ_w C_D A v |v|
        dz/dt = v,    dx/dt = U_x(z),    dy/dt = U_y(z)
    with exact linear fractional-step interpolation at the instant of seabed contact.
    """
    params.validate()
    if dt_s <= 0.0 or t_max_s <= 0.0:
        raise ValueError("dt_s and t_max_s must be > 0")

    x, y, z, v = float(x0), float(y0), float(z0), float(v0)
    t = 0.0
    hull_ref = params.hull_geometry if use_extended_hull else None
    trajectory: list[dict[str, float]] = [
        {"t": t, "x": x, "y": y, "z": z, "vz": v, "local_depth_m": grid.interpolate_depth(x, y)}
    ]

    max_steps = int(math.ceil(t_max_s / dt_s))
    contact_info = check_seabed_contact(x, y, z, grid, hull=hull_ref)
    if contact_info["contact"]:
        return {
            "seabed_contact": True,
            "contact_time_s": 0.0,
            "landing_x": x,
            "landing_y": y,
            "landing_z_m": z,
            "impact_velocity_m_s": v,
            "horizontal_drift_m": 0.0,
            "phase_state": "grounded",
            "contact_details": contact_info,
            "trajectory": trajectory,
        }

    for step_idx in range(1, max_steps + 1):
        prev_x, prev_y, prev_z, prev_v, prev_t = x, y, z, v, t
        prev_clear = contact_info["clearance_m"]

        # Semi-implicit Euler / RK2 midpoint vertical update + depth-attenuated advection
        a1 = vertical_acceleration(v, params)
        v_mid = v + 0.5 * dt_s * a1
        z_mid = max(0.0, z + 0.5 * dt_s * v)
        a2 = vertical_acceleration(v_mid, params)
        v = v + dt_s * a2
        z_next = z + dt_s * v_mid
        if z_next < 0.0 and params.phase_state == "floating":
            z_next = 0.0
            v = max(0.0, v)

        u_horiz = attenuate_current(u_surface, max(0.0, z_mid), attenuation_length_m)
        x = x + dt_s * float(u_horiz[0])
        y = y + dt_s * float(u_horiz[1])
        z = max(0.0, z_next)
        t = prev_t + dt_s

        contact_info = check_seabed_contact(x, y, z, grid, hull=hull_ref)
        if step_idx % max(1, int(round(1.0 / dt_s))) == 0 or contact_info["contact"]:
            trajectory.append(
                {"t": round(t, 4), "x": x, "y": y, "z": z, "vz": v, "local_depth_m": contact_info["local_depth_m"]}
            )

        if contact_info["contact"]:
            pen = contact_info["penetration_m"]
            denom = prev_clear + pen
            alpha = prev_clear / denom if denom > 1e-12 else 1.0
            alpha = max(0.0, min(1.0, alpha))
            t_hit = prev_t + alpha * dt_s
            x_hit = prev_x + alpha * (x - prev_x)
            y_hit = prev_y + alpha * (y - prev_y)
            z_hit = prev_z + alpha * (z - prev_z)
            v_hit = prev_v + alpha * (v - prev_v)
            drift_m = math.hypot(x_hit - float(x0), y_hit - float(y0))
            return {
                "seabed_contact": True,
                "contact_time_s": round(t_hit, 4),
                "landing_x": x_hit,
                "landing_y": y_hit,
                "landing_z_m": z_hit,
                "impact_velocity_m_s": v_hit,
                "horizontal_drift_m": drift_m,
                "phase_state": "grounded",
                "contact_details": contact_info,
                "trajectory": trajectory,
            }

    drift_m = math.hypot(x - float(x0), y - float(y0))
    return {
        "seabed_contact": False,
        "contact_time_s": None,
        "landing_x": x,
        "landing_y": y,
        "landing_z_m": z,
        "impact_velocity_m_s": v,
        "horizontal_drift_m": drift_m,
        "phase_state": params.phase_state,
        "contact_details": contact_info,
        "trajectory": trajectory,
    }


# ---------------------------------------------------------------------------
# 5. Temporal Seabed Reclamation, Exner Morphodynamics & Wreck Burial
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExnerConfig:
    """Configuration for the Exner sediment-continuity equation:
        (1 − λ_p) ∂B/∂t = −∇ · q_s + S
    """
    porosity_lambda_p: float = 0.40    # bed porosity λ_p ∈ [0, 1)
    dt_years: float = 1.0              # morphodynamic time step [years]

    def validate(self) -> None:
        if not (0.0 <= self.porosity_lambda_p < 1.0):
            raise ValueError("porosity_lambda_p must lie in [0, 1)")
        if self.dt_years == 0.0 or not math.isfinite(self.dt_years):
            raise ValueError("dt_years must be finite and non-zero")


def _divergence_2d_nonperiodic(
    qx: np.ndarray,
    qy: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    """Compute ∇ · q_s = ∂q_sx/∂x + ∂q_sy/∂y using second-order central differences
    in the interior and one-sided differences on the boundaries.
    """
    dqx_dx = np.gradient(qx, x_coords, axis=0)
    dqy_dy = np.gradient(qy, y_coords, axis=1)
    return dqx_dx + dqy_dy


def exner_step(
    grid: BathymetricGrid,
    q_sx: np.ndarray | float,
    q_sy: np.ndarray | float,
    source_sink: np.ndarray | float = 0.0,
    cfg: ExnerConfig = ExnerConfig(),
) -> tuple[BathymetricGrid, dict[str, Any]]:
    """Advance seabed elevation B(x,y,t) by one Exner step:
        (1 − λ_p) ∂B/∂t = −∇ · q_s + S
    where `q_sx`, `q_sy` are volumetric sediment fluxes [m²/year] and `source_sink` S
    is external deposition/erosion rate [m/year].
    """
    cfg.validate()
    shape = grid.elevation_b.shape
    qx_arr = np.full(shape, float(q_sx), dtype=float) if np.ndim(q_sx) == 0 else np.asarray(q_sx, dtype=float)
    qy_arr = np.full(shape, float(q_sy), dtype=float) if np.ndim(q_sy) == 0 else np.asarray(q_sy, dtype=float)
    s_arr = np.full(shape, float(source_sink), dtype=float) if np.ndim(source_sink) == 0 else np.asarray(source_sink, dtype=float)
    if qx_arr.shape != shape or qy_arr.shape != shape or s_arr.shape != shape:
        raise ValueError("q_sx, q_sy, and source_sink must match grid shape")

    div_q = _divergence_2d_nonperiodic(qx_arr, qy_arr, grid.x_coords, grid.y_coords)
    db_dt = (-div_q + s_arr) / (1.0 - cfg.porosity_lambda_p)
    b_next = grid.elevation_b + cfg.dt_years * db_dt

    # Propagate morphodynamic uncertainty proportionally to bed elevation change
    sigma_morph = 0.10 * np.abs(cfg.dt_years * db_dt)
    u_next = np.sqrt(grid.uncertainty_m ** 2 + sigma_morph ** 2)

    next_grid = BathymetricGrid(
        x_coords=grid.x_coords.copy(),
        y_coords=grid.y_coords.copy(),
        elevation_b=b_next,
        uncertainty_m=u_next,
        timestamp=grid.timestamp + cfg.dt_years,
        source=grid.source,
    )
    diag = {
        "max_abs_db_dt": float(np.max(np.abs(db_dt))),
        "mean_db_dt": float(np.mean(db_dt)),
        "max_abs_delta_b": float(np.max(np.abs(b_next - grid.elevation_b))),
        "zero_flux_stable": bool(np.max(np.abs(b_next - grid.elevation_b)) <= 1e-12),
    }
    return next_grid, diag


def reconstruct_historical_bathymetry(
    modern_grid: BathymetricGrid,
    *,
    t_historical: float = 1850.0,
    t_modern: Optional[float] = None,
    q_sx: np.ndarray | float = 0.0,
    q_sy: np.ndarray | float = 0.0,
    source_sink: np.ndarray | float = 0.004,
    porosity_lambda_p: float = 0.40,
    scenario_factors: Sequence[float] = (0.7, 1.0, 1.3),
) -> tuple[BathymetricGrid, dict[str, Any]]:
    """Reconstruct historical seabed elevation:
        B_{t_0}(x,y) = B_modern(x,y) − ∫_{t_0}^{t_modern} (∂B/∂t) dt
    across an ensemble of sediment-transport scenarios to propagate historical
    bathymetric uncertainty.
    """
    t_mod = float(modern_grid.timestamp if t_modern is None else t_modern)
    span_years = t_mod - float(t_historical)
    shape = modern_grid.elevation_b.shape
    qx_base = np.full(shape, float(q_sx), dtype=float) if np.ndim(q_sx) == 0 else np.asarray(q_sx, dtype=float)
    qy_base = np.full(shape, float(q_sy), dtype=float) if np.ndim(q_sy) == 0 else np.asarray(q_sy, dtype=float)
    s_base = np.full(shape, float(source_sink), dtype=float) if np.ndim(source_sink) == 0 else np.asarray(source_sink, dtype=float)

    div_base = _divergence_2d_nonperiodic(qx_base, qy_base, modern_grid.x_coords, modern_grid.y_coords)
    db_dt_base = (-div_base + s_base) / (1.0 - float(porosity_lambda_p))

    scenario_grids: list[np.ndarray] = []
    for fac in scenario_factors:
        b_hist = modern_grid.elevation_b - float(fac) * span_years * db_dt_base
        scenario_grids.append(b_hist)

    stack = np.stack(scenario_grids, axis=0)
    b_mean = np.mean(stack, axis=0)
    b_std = np.std(stack, axis=0)
    u_hist = np.sqrt(modern_grid.uncertainty_m ** 2 + b_std ** 2 + (0.01 * abs(span_years)) ** 2)

    hist_grid = BathymetricGrid(
        x_coords=modern_grid.x_coords.copy(),
        y_coords=modern_grid.y_coords.copy(),
        elevation_b=b_mean,
        uncertainty_m=u_hist,
        timestamp=float(t_historical),
        source=f"{modern_grid.source}_reconstructed_{int(t_historical)}",
    )
    summary = {
        "t_historical": float(t_historical),
        "t_modern": t_mod,
        "span_years": span_years,
        "mean_elevation_change_m": float(np.mean(modern_grid.elevation_b - b_mean)),
        "mean_historical_uncertainty_m": float(np.mean(u_hist)),
        "n_scenarios": len(scenario_factors),
    }
    return hist_grid, summary


def temporal_depression(
    grid_t0: BathymetricGrid,
    grid_t1: BathymetricGrid,
    x: Optional[float] = None,
    y: Optional[float] = None,
) -> dict[str, Any]:
    """Compute the Temporal Depression metric:
        D_T(x, y, t_0, t_1) = B(x, y, t_0) − B(x, y, t_1)

    Interpretation guardrail: D_T is a multiscale relational elevation difference
    between historical epoch t_0 and epoch t_1 (negative when sediment/tephra has
    accumulated so B(t_1) > B(t_0), positive when erosion/subsidence has lowered
    the bed). It is NOT a quantum-gravitational mechanism.
    """
    if x is not None and y is not None:
        b0 = grid_t0.interpolate_elevation(float(x), float(y))
        b1 = grid_t1.interpolate_elevation(float(x), float(y))
        u0 = grid_t0.interpolate_uncertainty(float(x), float(y))
        u1 = grid_t1.interpolate_uncertainty(float(x), float(y))
        dt_val = b0 - b1
        return {
            "x": float(x),
            "y": float(y),
            "t0": grid_t0.timestamp,
            "t1": grid_t1.timestamp,
            "b_t0_m": b0,
            "b_t1_m": b1,
            "temporal_depression_m": dt_val,
            "uncertainty_m": math.hypot(u0, u1),
            "interpretation": "multiscale_relational_elevation_difference",
        }

    diff_field = grid_t0.elevation_b - grid_t1.elevation_b
    unc_field = np.hypot(grid_t0.uncertainty_m, grid_t1.uncertainty_m)
    return {
        "t0": grid_t0.timestamp,
        "t1": grid_t1.timestamp,
        "mean_temporal_depression_m": float(np.mean(diff_field)),
        "min_temporal_depression_m": float(np.min(diff_field)),
        "max_temporal_depression_m": float(np.max(diff_field)),
        "mean_uncertainty_m": float(np.mean(unc_field)),
        "interpretation": "multiscale_relational_elevation_difference",
    }


@dataclass(frozen=True)
class WreckMorphodynamicState:
    """Wreck burial, exposure, and 7-state morphodynamic lifecycle evaluation.
    Vertical sediment accumulation and burial depth are tracked independently of
    horizontal wreck displacement.
    """
    seabed_elevation_b_m: float
    wreck_top_elevation_z_top_m: float
    burial_depth_m: float              # d_burial(t) = max(0, B(x_w,y_w,t) - Z_top(t))
    exposed_relief_m: float            # max(0, Z_top(t) - B(x_w,y_w,t))
    local_scour_depth_m: float
    structural_collapse_fraction: float
    fragmentation_index: float
    horizontal_displacement_m: float   # kept strictly separate from vertical burial
    active_lifecycle_states: tuple[str, ...]
    dominant_state: str


def evaluate_wreck_burial(
    seabed_elevation_b_m: float,
    keel_elevation_b0_m: float,
    hull_height_m: float = 4.6,
    *,
    elapsed_years: float = 176.0,
    bottom_current_speed_m_s: float = 0.25,
    collapse_rate_per_year: float = 0.0035,
    prior_max_burial_m: float = 0.0,
    horizontal_displacement_m: float = 0.0,
) -> WreckMorphodynamicState:
    """Evaluate wreck burial depth:
        d_burial(t) = max(0, B(x_w, y_w, t) − Z_top(t))
    and classify the 7 wreck morphodynamic lifecycle states without conflating
    vertical sediment accumulation with horizontal wreck displacement.
    """
    t_yr = max(0.0, float(elapsed_years))
    collapse_frac = 1.0 - math.exp(-max(0.0, float(collapse_rate_per_year)) * t_yr)
    collapse_frac = max(0.0, min(0.85, collapse_frac))
    effective_height = float(hull_height_m) * (1.0 - collapse_frac)
    z_top = float(keel_elevation_b0_m) + effective_height

    b_now = float(seabed_elevation_b_m)
    d_burial = max(0.0, b_now - z_top)
    exposed_relief = max(0.0, z_top - b_now)
    sediment_acc = b_now - float(keel_elevation_b0_m)

    # Local horseshoe-vortex scour depth around exposed wreck relief
    u_bot = max(0.0, float(bottom_current_speed_m_s))
    scour_depth = 0.8 * exposed_relief * math.tanh(u_bot / 0.30) if exposed_relief > 0.05 else 0.0
    frag_idx = max(0.0, min(1.0, 0.6 * collapse_frac + 0.4 * math.tanh(u_bot * t_yr / 80.0)))

    states: list[str] = ["initial_seabed_contact"]
    if scour_depth > 0.15:
        states.append("scour")
    if sediment_acc > 0.10:
        states.append("sediment_accumulation")
    if sediment_acc > 0.10 and d_burial == 0.0:
        states.append("partial_burial")
    if collapse_frac >= 0.20:
        states.append("structural_collapse")
    if frag_idx >= 0.25:
        states.append("fragmentation")
    if prior_max_burial_m > d_burial + 0.10:
        states.append("re_exposure")

    if "re_exposure" in states:
        dominant = "re_exposure"
    elif d_burial > 0.0:
        dominant = "sediment_accumulation"
    elif "structural_collapse" in states:
        dominant = "structural_collapse"
    elif "partial_burial" in states:
        dominant = "partial_burial"
    elif "scour" in states:
        dominant = "scour"
    else:
        dominant = "initial_seabed_contact"

    return WreckMorphodynamicState(
        seabed_elevation_b_m=b_now,
        wreck_top_elevation_z_top_m=z_top,
        burial_depth_m=d_burial,
        exposed_relief_m=exposed_relief,
        local_scour_depth_m=scour_depth,
        structural_collapse_fraction=collapse_frac,
        fragmentation_index=frag_idx,
        horizontal_displacement_m=float(horizontal_displacement_m),
        active_lifecycle_states=tuple(states),
        dominant_state=dominant,
    )


# ---------------------------------------------------------------------------
# 6. 6-Feature VNN Bathymetric Adapter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BathymetricFeatureVector:
    """The 6 physical bathymetry & sediment features exposed to QUIPU VNN adapters.
    Never overwrites the 7-D semantic sense vector (`mesh_slm_embed`).
    """
    modern_depth_m: float
    historical_depth_m: float
    depth_uncertainty_m: float
    sediment_flux_x: float
    sediment_flux_y: float
    seabed_slope: float

    def to_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (
            float(self.modern_depth_m),
            float(self.historical_depth_m),
            float(self.depth_uncertainty_m),
            float(self.sediment_flux_x),
            float(self.sediment_flux_y),
            float(self.seabed_slope),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "modern_depth_m": float(self.modern_depth_m),
            "historical_depth_m": float(self.historical_depth_m),
            "depth_uncertainty_m": float(self.depth_uncertainty_m),
            "sediment_flux_x": float(self.sediment_flux_x),
            "sediment_flux_y": float(self.sediment_flux_y),
            "seabed_slope": float(self.seabed_slope),
        }


def extract_bathymetric_features(
    x: float,
    y: float,
    modern_grid: BathymetricGrid,
    historical_grid: Optional[BathymetricGrid] = None,
    *,
    sediment_flux_x: float = 0.0,
    sediment_flux_y: float = 0.0,
) -> BathymetricFeatureVector:
    """Extract the 6-feature VNN bathymetric vector at coordinates (x, y)."""
    h_mod = modern_grid.interpolate_depth(x, y)
    h_hist = historical_grid.interpolate_depth(x, y) if historical_grid is not None else h_mod
    u_mod = modern_grid.interpolate_uncertainty(x, y)
    u_hist = historical_grid.interpolate_uncertainty(x, y) if historical_grid is not None else u_mod
    slope = modern_grid.seabed_slope(x, y)
    return BathymetricFeatureVector(
        modern_depth_m=h_mod,
        historical_depth_m=h_hist,
        depth_uncertainty_m=max(u_mod, u_hist),
        sediment_flux_x=float(sediment_flux_x),
        sediment_flux_y=float(sediment_flux_y),
        seabed_slope=slope,
    )


# ---------------------------------------------------------------------------
# 7. Geodesic Reconstruction Graph & Pruned Probability-Mass Ledger
# ---------------------------------------------------------------------------

def geodesic_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle geodesic distance [m] on the spherical Earth."""
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(0.5 * dp) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(0.5 * dl) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(max(0.0, a))))


@dataclass
class ReconstructionNode:
    """Typed node in the QUIPU Geodesic Reconstruction Graph G = (V, E)."""
    node_id: str
    entity_type: str                   # e.g. "vessel_state", "evidence_sighting", "seabed_cell", "settlement"
    latitude: float
    longitude: float
    depth: float                       # positive downward depth [m] (negative for elevation above datum)
    timestamp: float                   # epoch year or unix timestamp
    physical_state: dict[str, Any]
    evidence_source: str
    uncertainty: float                 # positional/depth uncertainty radius [m]
    model_version: str = "quipu-vnn-bathymetry-v0.33"
    bathymetry_features: Optional[BathymetricFeatureVector] = None


@dataclass
class ReconstructionEdge:
    """Typed directed edge in the QUIPU Geodesic Reconstruction Graph G = (V, E)."""
    source_node: str
    target_node: str
    relationship_type: str             # e.g. "surface_drift", "vertical_descent", "morphodynamic_evolution", "spatial_neighbor"
    transition_probability: float      # p_ij ∈ (0, 1]
    geodesic_distance: float           # [m]
    physical_cost: float               # c_ij = -log(p_ij) >= 0
    temporal_interval: float           # Δt
    evidence_weight: float = 1.0

    @classmethod
    def from_probability(
        cls,
        source_node: str,
        target_node: str,
        relationship_type: str,
        transition_probability: float,
        geodesic_distance: float,
        temporal_interval: float,
        evidence_weight: float = 1.0,
    ) -> "ReconstructionEdge":
        p = max(1e-12, min(1.0, float(transition_probability)))
        return cls(
            source_node=source_node,
            target_node=target_node,
            relationship_type=relationship_type,
            transition_probability=p,
            geodesic_distance=float(geodesic_distance),
            physical_cost=-math.log(p),
            temporal_interval=float(temporal_interval),
            evidence_weight=float(evidence_weight),
        )


@dataclass(frozen=True)
class PrunedMassLedger:
    """Explicit accounting of retained vs. pruned branch probability mass during
    Monte Carlo decision pruning and geodesic graph screening.
    """
    total_branches: int
    retained_branches: int
    pruned_branches: int
    total_probability_mass: float
    retained_probability_mass: float
    pruned_probability_mass: float
    pruned_mass_fraction: float
    renormalized: bool


class GeodesicReconstructionGraph:
    """Geodesic Graph G = (V, E) supporting Voronoi spatial partitioning,
    maximum-probability Dijkstra routing via c_ij = −log(p_ij), probability-mass-aware
    pruning, greedy trajectory simplification, and fuzzy temporal-spatial interpolation.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, ReconstructionNode] = {}
        self.edges: list[ReconstructionEdge] = []

    def add_node(self, node: ReconstructionNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: ReconstructionEdge) -> None:
        if edge.source_node not in self.nodes or edge.target_node not in self.nodes:
            raise KeyError("Both source_node and target_node must exist in graph")
        self.edges.append(edge)

    def assign_voronoi_cells(
        self,
        sample_points_latlon: Sequence[tuple[float, float]],
        seed_node_ids: Optional[Sequence[str]] = None,
    ) -> dict[str, list[tuple[float, float]]]:
        """Partition sample (lat, lon) coordinates into Geodesic Voronoi regions
        around the graph's seed nodes using great-circle distance.
        """
        seeds = list(seed_node_ids) if seed_node_ids is not None else list(self.nodes.keys())
        cells: dict[str, list[tuple[float, float]]] = {nid: [] for nid in seeds}
        if not seeds:
            return cells
        for lat, lon in sample_points_latlon:
            best_nid = min(
                seeds,
                key=lambda nid: (
                    geodesic_distance_m(lat, lon, self.nodes[nid].latitude, self.nodes[nid].longitude),
                    nid,
                ),
            )
            cells[best_nid].append((float(lat), float(lon)))
        return cells

    def build_neighbor_edges(self, k_neighbors: int = 3, length_scale_m: float = 15_000.0) -> int:
        """Connect each node to its k nearest geodesic neighbors with transition
        probabilities p_ij ∝ exp(−d_ij / length_scale_m).
        """
        nids = list(self.nodes.keys())
        added = 0
        for u_id in nids:
            u = self.nodes[u_id]
            dists: list[tuple[float, str]] = []
            for v_id in nids:
                if u_id == v_id:
                    continue
                v = self.nodes[v_id]
                d = geodesic_distance_m(u.latitude, u.longitude, v.latitude, v.longitude)
                dists.append((d, v_id))
            dists.sort(key=lambda item: (item[0], item[1]))
            chosen = dists[: max(1, int(k_neighbors))]
            raw_weights = [math.exp(-d / max(1.0, float(length_scale_m))) for d, _ in chosen]
            z_sum = sum(raw_weights) or 1.0
            for (d, v_id), w in zip(chosen, raw_weights):
                v = self.nodes[v_id]
                p_ij = max(1e-9, min(1.0, w / z_sum))
                self.add_edge(
                    ReconstructionEdge.from_probability(
                        source_node=u_id,
                        target_node=v_id,
                        relationship_type="spatial_neighbor",
                        transition_probability=p_ij,
                        geodesic_distance=d,
                        temporal_interval=v.timestamp - u.timestamp,
                        evidence_weight=1.0,
                    )
                )
                added += 1
        return added

    def prune_low_probability_edges(
        self,
        min_probability: float = 0.05,
    ) -> tuple[list[ReconstructionEdge], PrunedMassLedger]:
        """Screen candidate edges by minimum transition probability while recording
        the exact pruned and retained probability mass in a `PrunedMassLedger`.
        """
        retained: list[ReconstructionEdge] = []
        total_mass = sum(e.transition_probability for e in self.edges)
        retained_mass = 0.0
        pruned_mass = 0.0
        pruned_count = 0

        for e in self.edges:
            if e.transition_probability >= float(min_probability):
                retained.append(e)
                retained_mass += e.transition_probability
            else:
                pruned_count += 1
                pruned_mass += e.transition_probability

        # Guarantee at least the highest-probability outgoing edge per source survives
        by_src: dict[str, list[ReconstructionEdge]] = {}
        for e in self.edges:
            by_src.setdefault(e.source_node, []).append(e)
        retained_set = {(e.source_node, e.target_node, e.relationship_type) for e in retained}
        for src_id, cands in by_src.items():
            if not any(e.source_node == src_id for e in retained):
                best_e = max(cands, key=lambda x: (x.transition_probability, -x.physical_cost))
                key = (best_e.source_node, best_e.target_node, best_e.relationship_type)
                if key not in retained_set:
                    retained.append(best_e)
                    retained_set.add(key)
                    retained_mass += best_e.transition_probability
                    pruned_mass = max(0.0, pruned_mass - best_e.transition_probability)
                    pruned_count = max(0, pruned_count - 1)

        frac = pruned_mass / total_mass if total_mass > 0.0 else 0.0
        ledger = PrunedMassLedger(
            total_branches=len(self.edges),
            retained_branches=len(retained),
            pruned_branches=pruned_count,
            total_probability_mass=total_mass,
            retained_probability_mass=retained_mass,
            pruned_probability_mass=pruned_mass,
            pruned_mass_fraction=frac,
            renormalized=False,
        )
        return retained, ledger

    def dijkstra_max_probability_path(
        self,
        source_id: str,
        target_id: str,
        *,
        min_edge_probability: float = 0.0,
    ) -> dict[str, Any]:
        """Maximum-probability trajectory search via Dijkstra with edge costs:
            c_ij = −log(p_ij) >= 0
        so minimizing Σ c_ij maximizes Π p_ij.
        """
        eligible_edges, ledger = self.prune_low_probability_edges(min_edge_probability)
        adj: dict[str, list[ReconstructionEdge]] = {}
        for e in eligible_edges:
            adj.setdefault(e.source_node, []).append(e)

        dist: dict[str, float] = {source_id: 0.0}
        prev: dict[str, tuple[str, ReconstructionEdge]] = {}
        heap: list[tuple[float, str]] = [(0.0, source_id)]
        visited: set[str] = set()

        while heap:
            cost_u, u_id = heapq.heappop(heap)
            if u_id in visited:
                continue
            visited.add(u_id)
            if u_id == target_id:
                break
            for e in adj.get(u_id, ()):
                c_ij = max(0.0, -math.log(max(1e-12, min(1.0, e.transition_probability))))
                nd = cost_u + c_ij
                if nd < dist.get(e.target_node, math.inf):
                    dist[e.target_node] = nd
                    prev[e.target_node] = (u_id, e)
                    heapq.heappush(heap, (nd, e.target_node))

        if target_id not in dist:
            return {
                "found": False,
                "path": [],
                "total_neg_log_prob": math.inf,
                "path_probability": 0.0,
                "geodesic_distance_m": 0.0,
                "pruned_mass_ledger": dataclasses.asdict(ledger),
            }

        path = [target_id]
        total_geo_m = 0.0
        while path[-1] != source_id:
            p_id, edge = prev[path[-1]]
            total_geo_m += edge.geodesic_distance
            path.append(p_id)
        path.reverse()

        neg_log_p = dist[target_id]
        return {
            "found": True,
            "path": path,
            "total_neg_log_prob": neg_log_p,
            "path_probability": math.exp(-neg_log_p),
            "geodesic_distance_m": total_geo_m,
            "pruned_mass_ledger": dataclasses.asdict(ledger),
        }

    def simplify_trajectory_greedy(
        self,
        path_node_ids: Sequence[str],
        tolerance_m: float = 500.0,
    ) -> list[str]:
        """Greedy trajectory simplification: retain waypoints whose cross-track
        geodesic deviation exceeds `tolerance_m`.
        """
        if len(path_node_ids) <= 2:
            return list(path_node_ids)
        kept = [path_node_ids[0]]
        anchor = self.nodes[path_node_ids[0]]
        for idx in range(1, len(path_node_ids) - 1):
            curr = self.nodes[path_node_ids[idx]]
            nxt = self.nodes[path_node_ids[idx + 1]]
            d_direct = geodesic_distance_m(anchor.latitude, anchor.longitude, nxt.latitude, nxt.longitude)
            d_via = (
                geodesic_distance_m(anchor.latitude, anchor.longitude, curr.latitude, curr.longitude)
                + geodesic_distance_m(curr.latitude, curr.longitude, nxt.latitude, nxt.longitude)
            )
            detour = max(0.0, d_via - d_direct)
            if detour > float(tolerance_m):
                kept.append(curr.node_id)
                anchor = curr
        kept.append(path_node_ids[-1])
        return kept

    def fuzzy_interpolate_nodes(
        self,
        node_a_id: str,
        node_b_id: str,
        alpha: float = 0.5,
    ) -> ReconstructionNode:
        """Fuzzy temporal-spatial interpolation between two reconstruction nodes,
        inflating uncertainty at intermediate times (brownian bridge envelope).
        """
        w = max(0.0, min(1.0, float(alpha)))
        na = self.nodes[node_a_id]
        nb = self.nodes[node_b_id]
        lat = (1.0 - w) * na.latitude + w * nb.latitude
        lon = (1.0 - w) * na.longitude + w * nb.longitude
        depth = (1.0 - w) * na.depth + w * nb.depth
        ts = (1.0 - w) * na.timestamp + w * nb.timestamp
        bridge_spread = 0.15 * geodesic_distance_m(na.latitude, na.longitude, nb.latitude, nb.longitude) * w * (1.0 - w)
        unc = (1.0 - w) * na.uncertainty + w * nb.uncertainty + bridge_spread
        return ReconstructionNode(
            node_id=f"interp_{node_a_id}_{node_b_id}_{int(round(w * 100))}",
            entity_type="fuzzy_interpolated_state",
            latitude=lat,
            longitude=lon,
            depth=depth,
            timestamp=ts,
            physical_state={"alpha": w, "source_a": node_a_id, "source_b": node_b_id},
            evidence_source="fuzzy_temporal_spatial_bridge",
            uncertainty=unc,
            model_version=na.model_version,
        )


# ---------------------------------------------------------------------------
# 8. Bayesian Evidence Update, Inverse Operator & 8-Ensemble Shadow Monte Carlo
# ---------------------------------------------------------------------------

def bayesian_evidence_update(
    prior_weights: Sequence[float],
    candidate_positions_xy: Sequence[tuple[float, float]],
    evidence_observations: Sequence[tuple[tuple[float, float], float]] = (),
) -> list[float]:
    """Bayesian posterior update over an ensemble of candidate historical states:
        p(S_i | E) ∝ p(E | S_i) · p(S_i)
    where each observation in `evidence_observations` is `((obs_x, obs_y), obs_sigma_m)`.
    """
    n = len(prior_weights)
    if n == 0:
        return []
    if len(candidate_positions_xy) != n:
        raise ValueError("prior_weights and candidate_positions_xy must have the same length")

    log_post: list[float] = []
    for w, (cx, cy) in zip(prior_weights, candidate_positions_xy):
        lp = math.log(max(1e-15, float(w)))
        for (ox, oy), sigma in evidence_observations:
            s = max(1.0, float(sigma))
            d2 = (float(cx) - float(ox)) ** 2 + (float(cy) - float(oy)) ** 2
            lp += -0.5 * d2 / (s * s)
        log_post.append(lp)

    max_lp = max(log_post)
    unnorm = [math.exp(lp - max_lp) for lp in log_post]
    z = sum(unnorm) or 1.0
    return [u / z for u in unnorm]


def reconstruct_historical_state(
    present_state: TemporalSpatialState,
    evidence_observations: Sequence[tuple[tuple[float, float], float]],
    modern_grid: BathymetricGrid,
    *,
    t_historical: float = 1850.0,
    n_particles: int = 24,
    u_ocean: tuple[float, float] = (0.35, 0.15),
    u_wind: tuple[float, float] = (8.0, 3.5),
    u_wave: tuple[float, float] = (0.08, 0.04),
    drift_duration_hours: float = 6.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Inverse historical reconstruction operator:
        S(t_0) = R( S(t_1), E, θ )
    Retains a full probability-weighted ensemble of plausible historical states S_i(t_0)
    rather than collapsing to a single deterministic point estimate.
    """
    hist_grid, hist_summary = reconstruct_historical_bathymetry(
        modern_grid, t_historical=t_historical, t_modern=present_state.timestamp or 2026.0
    )
    rng = random.Random(int(seed))
    dt_total = float(drift_duration_hours) * 3600.0
    leeway = 0.03
    mean_vx = float(u_ocean[0]) + leeway * float(u_wind[0]) + float(u_wave[0])
    mean_vy = float(u_ocean[1]) + leeway * float(u_wind[1]) + float(u_wave[1])

    candidates: list[TemporalSpatialState] = []
    positions_xy: list[tuple[float, float]] = []
    priors: list[float] = []

    for i in range(max(2, int(n_particles))):
        # Back-project surface drift + descent offset with stochastic dispersion
        spread = 180.0
        x0 = present_state.x - mean_vx * dt_total + rng.gauss(0.0, spread)
        y0 = present_state.y - mean_vy * dt_total + rng.gauss(0.0, spread)
        b_hist = hist_grid.interpolate_elevation(x0, y0)
        u_hist = hist_grid.interpolate_uncertainty(x0, y0)
        st = TemporalSpatialState(
            x=x0,
            y=y0,
            z=0.0,
            v=(mean_vx, mean_vy, 0.0),
            forcing={"u_ocean_x": u_ocean[0], "u_ocean_y": u_ocean[1], "u_wind_x": u_wind[0], "u_wind_y": u_wind[1]},
            seabed_elevation_m=b_hist,
            parameters={"t_historical": float(t_historical), "particle_id": float(i)},
            covariance_diag=(spread ** 2, spread ** 2, u_hist ** 2),
            timestamp=float(t_historical),
            latitude=present_state.latitude,
            longitude=present_state.longitude,
            phase_state="flooded_hull",
        )
        candidates.append(st)
        positions_xy.append((x0, y0))
        priors.append(1.0 / float(max(2, int(n_particles))))

    post_weights = bayesian_evidence_update(priors, positions_xy, evidence_observations)
    post_mean_x = sum(w * p[0] for w, p in zip(post_weights, positions_xy))
    post_mean_y = sum(w * p[1] for w, p in zip(post_weights, positions_xy))
    post_var_x = sum(w * ((p[0] - post_mean_x) ** 2) for w, p in zip(post_weights, positions_xy))
    post_var_y = sum(w * ((p[1] - post_mean_y) ** 2) for w, p in zip(post_weights, positions_xy))

    return {
        "t_historical": float(t_historical),
        "t_present": present_state.timestamp,
        "ensemble_size": len(candidates),
        "posterior_mean_xy": (post_mean_x, post_mean_y),
        "posterior_std_xy": (math.sqrt(post_var_x), math.sqrt(post_var_y)),
        "posterior_weights": post_weights,
        "historical_states": [c.to_dict() for c in candidates],
        "historical_bathymetry_summary": hist_summary,
    }


def run_shadow_mc_ensembles(
    modern_grid: BathymetricGrid,
    *,
    x0: float = 500.0,
    y0: float = 500.0,
    lat0: float = 40.50,
    lon0: float = -69.20,
    u_ocean: tuple[float, float] = (0.40, 0.15),
    u_wind: tuple[float, float] = (9.5, 4.0),
    u_wave: tuple[float, float] = (0.10, 0.05),
    n_Rollouts: int = 16,
    t_historical: float = 1850.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute the 8-Ensemble Shadow Monte Carlo architecture:
    1. `raw_ueqgm_quipu`     — relational prior without environmental physics
    2. `ocean_only`          — ocean current transport only
    3. `atmosphere_only`     — wind leeway transport only
    4. `combined_forcing`    — coupled ocean + atmosphere + wave + diffusion
    5. `vertical_descent`    — water-column descent to seabed contact
    6. `historical_bathymetry` — 1850 back-projected seabed geometry
    7. `sediment_evolution`  — Exner burial & 7-state wreck morphodynamics
    8. `acquisition`         — Bayesian uncertainty-weighted search priority
    """
    hist_grid, hist_info = reconstruct_historical_bathymetry(modern_grid, t_historical=t_historical)
    vessel_params = DescentObjectParams.helena_sloman_hull(phase_state="flooded_hull")
    descent_mod = simulate_descent(
        vessel_params, modern_grid, x0=x0, y0=y0, u_surface=u_ocean, dt_s=0.05, use_extended_hull=True
    )
    descent_hist = simulate_descent(
        vessel_params, hist_grid, x0=x0, y0=y0, u_surface=u_ocean, dt_s=0.05, use_extended_hull=True
    )

    ensembles: dict[str, dict[str, Any]] = {}
    for ens_idx, name in enumerate(SHADOW_ENSEMBLE_NAMES):
        cfg = SurfaceDriftConfig(
            enable_ocean=name in ("ocean_only", "combined_forcing", "vertical_descent", "historical_bathymetry", "sediment_evolution", "acquisition"),
            enable_atmosphere=name in ("atmosphere_only", "combined_forcing", "acquisition"),
            enable_waves=name in ("combined_forcing", "acquisition"),
            enable_diffusion=(name != "raw_ueqgm_quipu"),
            dt_s=600.0,
            seed=int(seed) + ens_idx * 97,
        )
        endpoints: list[tuple[float, float]] = []
        for r in range(max(4, int(n_Rollouts))):
            out = step_surface_drift((x0, y0), u_ocean, u_wind, u_wave, cfg, step_index=r)
            px, py = out["position_xy"]
            if name in ("vertical_descent", "historical_bathymetry", "sediment_evolution", "acquisition"):
                ref_d = descent_hist if name == "historical_bathymetry" else descent_mod
                px += ref_d["landing_x"] - x0
                py += ref_d["landing_y"] - y0
            endpoints.append((px, py))

        xs = [p[0] for p in endpoints]
        ys = [p[1] for p in endpoints]
        mean_x = statistics.fmean(xs)
        mean_y = statistics.fmean(ys)
        std_x = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        std_y = statistics.pstdev(ys) if len(ys) > 1 else 0.0
        # Split-half Monte Carlo convergence diagnostic
        half = max(1, len(xs) // 2)
        split_diff = math.hypot(
            statistics.fmean(xs[:half]) - statistics.fmean(xs[half:]),
            statistics.fmean(ys[:half]) - statistics.fmean(ys[half:]),
        )
        grid_ref = hist_grid if name == "historical_bathymetry" else modern_grid
        feat = extract_bathymetric_features(mean_x, mean_y, modern_grid, hist_grid, sediment_flux_x=0.01, sediment_flux_y=-0.005)
        b_now = modern_grid.interpolate_elevation(mean_x, mean_y)
        b_1850 = hist_grid.interpolate_elevation(mean_x, mean_y)
        burial = evaluate_wreck_burial(
            seabed_elevation_b_m=b_now,
            keel_elevation_b0_m=b_1850,
            hull_height_m=4.6,
            elapsed_years=max(1.0, 2026.0 - t_historical),
            horizontal_displacement_m=math.hypot(mean_x - x0, mean_y - y0),
        )
        # Convert local meter offsets to geographic lat/lon around (lat0, lon0)
        dlat = (mean_y - y0) / 111_320.0
        dlon = (mean_x - x0) / (111_320.0 * max(0.1, math.cos(math.radians(lat0))))
        ensembles[name] = {
            "ensemble_name": name,
            "rollouts": len(endpoints),
            "mean_xy": (mean_x, mean_y),
            "std_xy": (std_x, std_y),
            "latitude": lat0 + dlat,
            "longitude": lon0 + dlon,
            "depth_m": grid_ref.interpolate_depth(mean_x, mean_y),
            "uncertainty_m": math.hypot(std_x, std_y) + grid_ref.interpolate_uncertainty(mean_x, mean_y),
            "split_half_error_m": split_diff,
            "converged": bool(split_diff <= max(25.0, 1.5 * math.hypot(std_x, std_y) + 1.0)),
            "bathymetry_features": feat.to_dict(),
            "burial_depth_m": burial.burial_depth_m,
            "wreck_lifecycle_state": burial.dominant_state,
            "parameter_manifest": {
                "enable_ocean": cfg.enable_ocean,
                "enable_atmosphere": cfg.enable_atmosphere,
                "enable_waves": cfg.enable_waves,
                "t_historical": t_historical,
            },
        }

    return {
        "case": "shadow_mc_8_ensembles",
        "t_historical": t_historical,
        "modern_descent": {
            "contact_time_s": descent_mod["contact_time_s"],
            "horizontal_drift_m": descent_mod["horizontal_drift_m"],
            "phase_state": descent_mod["phase_state"],
        },
        "historical_bathymetry": hist_info,
        "ensembles": ensembles,
    }


def build_helena_sloman_1850_case() -> dict[str, Any]:
    """Construct the integrated *Helena Sloman* (November 1850 North Atlantic)
    historical reconstruction benchmark case.
    """
    modern_grid = BathymetricGrid.sloped_shelf(
        base_depth_m=115.0, slope_x=0.015, slope_y=-0.008, nx=25, ny=25
    )
    shadow = run_shadow_mc_ensembles(
        modern_grid,
        x0=500.0,
        y0=500.0,
        lat0=40.55,
        lon0=-69.15,
        t_historical=1850.0,
    )
    hist_grid, _ = reconstruct_historical_bathymetry(modern_grid, t_historical=1850.0, source_sink=0.005)
    dep = temporal_depression(hist_grid, modern_grid, 500.0, 500.0)
    hull_params = DescentObjectParams.helena_sloman_hull(phase_state="flooded_hull")
    person_params = DescentObjectParams.person_in_water(positively_buoyant=True)
    hull_descent = simulate_descent(hull_params, hist_grid, x0=500.0, y0=500.0, dt_s=0.05, use_extended_hull=True)
    person_descent = simulate_descent(person_params, hist_grid, x0=500.0, y0=500.0, dt_s=0.05, t_max_s=30.0)

    return {
        "case_id": "helena_sloman_1850",
        "vessel": hull_params.name,
        "hull_descent": {
            "seabed_contact": hull_descent["seabed_contact"],
            "contact_time_s": hull_descent["contact_time_s"],
            "horizontal_drift_m": hull_descent["horizontal_drift_m"],
            "contact_mode": hull_descent["contact_details"]["mode"],
        },
        "person_drift": {
            "seabed_contact": person_descent["seabed_contact"],
            "phase_state": person_descent["phase_state"],
            "horizontal_drift_m": person_descent["horizontal_drift_m"],
        },
        "temporal_depression": dep,
        "shadow_mc": shadow,
    }


def build_pompeii_79ce_case() -> dict[str, Any]:
    """Construct the Pompeii–Vesuvius (79 CE tephra/pyroclastic burial vs. modern
    topography/shoreline reclamation) benchmark case.
    """
    modern_grid = BathymetricGrid.sloped_shelf(
        base_depth_m=-28.0,  # subaerial elevation B = +28 m (H = -28 m)
        slope_x=-0.01,
        slope_y=0.005,
        nx=21,
        ny=21,
        source="pompeii_modern_dem",
    )
    hist_grid, hist_summary = reconstruct_historical_bathymetry(
        modern_grid,
        t_historical=79.0,
        t_modern=2026.0,
        source_sink=0.0022,  # net tephra/alluvial aggradation equivalent over 1947 yrs
    )
    dep = temporal_depression(hist_grid, modern_grid, 500.0, 500.0)
    burial = evaluate_wreck_burial(
        seabed_elevation_b_m=modern_grid.interpolate_elevation(500.0, 500.0),
        keel_elevation_b0_m=hist_grid.interpolate_elevation(500.0, 500.0),
        hull_height_m=3.5,
        elapsed_years=2026.0 - 79.0,
        bottom_current_speed_m_s=0.0,
    )
    return {
        "case_id": "pompeii_vesuvius_79ce",
        "historical_epoch": 79.0,
        "modern_epoch": 2026.0,
        "temporal_depression": dep,
        "burial_depth_m": burial.burial_depth_m,
        "lifecycle_states": list(burial.active_lifecycle_states),
        "historical_summary": hist_summary,
    }


def export_reconstruction_geojson(shadow_result: Mapping[str, Any]) -> str:
    """Export the 8-Ensemble Shadow Monte Carlo reconstruction results as a valid
    GeoJSON FeatureCollection for GIS ingestion.
    """
    features: list[dict[str, Any]] = []
    ensembles = shadow_result.get("ensembles", {})
    for name, data in ensembles.items():
        lat = float(data.get("latitude", 0.0))
        lon = float(data.get("longitude", 0.0))
        depth = float(data.get("depth_m", 0.0))
        props = {
            "ensemble_name": name,
            "rollouts": data.get("rollouts", 0),
            "depth_m": depth,
            "uncertainty_m": data.get("uncertainty_m", 0.0),
            "burial_depth_m": data.get("burial_depth_m", 0.0),
            "wreck_lifecycle_state": data.get("wreck_lifecycle_state", "initial_seabed_contact"),
            "converged": data.get("converged", True),
            **(data.get("bathymetry_features") or {}),
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat, -depth]},
                "properties": props,
            }
        )
    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2, sort_keys=True)


def export_reconstruction_csv(shadow_result: Mapping[str, Any]) -> str:
    """Export the 8-Ensemble Shadow Monte Carlo reconstruction summary as CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        [
            "ensemble_name",
            "latitude",
            "longitude",
            "depth_m",
            "uncertainty_m",
            "burial_depth_m",
            "wreck_lifecycle_state",
            "modern_depth_m",
            "historical_depth_m",
            "depth_uncertainty_m",
            "sediment_flux_x",
            "sediment_flux_y",
            "seabed_slope",
            "converged",
        ]
    )
    ensembles = shadow_result.get("ensembles", {})
    for name in sorted(ensembles.keys()):
        d = ensembles[name]
        bf = d.get("bathymetry_features") or {}
        writer.writerow(
            [
                name,
                f"{float(d.get('latitude', 0.0)):.6f}",
                f"{float(d.get('longitude', 0.0)):.6f}",
                f"{float(d.get('depth_m', 0.0)):.3f}",
                f"{float(d.get('uncertainty_m', 0.0)):.3f}",
                f"{float(d.get('burial_depth_m', 0.0)):.3f}",
                str(d.get("wreck_lifecycle_state", "")),
                f"{float(bf.get('modern_depth_m', 0.0)):.3f}",
                f"{float(bf.get('historical_depth_m', 0.0)):.3f}",
                f"{float(bf.get('depth_uncertainty_m', 0.0)):.3f}",
                f"{float(bf.get('sediment_flux_x', 0.0)):.5f}",
                f"{float(bf.get('sediment_flux_y', 0.0)):.5f}",
                f"{float(bf.get('seabed_slope', 0.0)):.5f}",
                str(bool(d.get("converged", True))),
            ]
        )
    return buf.getvalue()


def run_v033_synthetic_checks() -> dict[str, Any]:
    """Execute the five v0.33 synthetic verification checks on an artificial
    100-meter-deep seabed and verify the ~31.48-second illustrative descent benchmark.
    """
    grid = BathymetricGrid.uniform(depth_m=100.0, nx=21, ny=21)
    params = DescentObjectParams.illustrative_v033()

    # 1. Bathymetric interpolation check
    interp_depth = grid.interpolate_depth(500.0, 500.0)
    slope_val = grid.seabed_slope(500.0, 500.0)
    c1_pass = math.isclose(interp_depth, 100.0, rel_tol=1e-9, abs_tol=1e-9) and math.isclose(slope_val, 0.0, abs_tol=1e-9)

    # 2. Surface-current attenuation check
    u0 = np.array([1.0, 0.0], dtype=float)
    u_surface = attenuate_current(u0, 0.0, 40.0)
    u_40m = attenuate_current(u0, 40.0, 40.0)
    c2_pass = math.isclose(float(u_surface[0]), 1.0, abs_tol=1e-9) and math.isclose(float(u_40m[0]), math.exp(-1.0), rel_tol=1e-6)

    # 3. Initial downward acceleration check
    a0 = vertical_acceleration(0.0, params)
    c3_pass = a0 > 0.0 and math.isfinite(a0)

    # 4. Seabed-contact detection check (~31.48 seconds to 100 m seabed)
    descent = simulate_descent(params, grid, x0=500.0, y0=500.0, z0=0.0, v0=0.0, dt_s=0.01, t_max_s=120.0)
    t_contact = descent["contact_time_s"]
    c4_pass = bool(descent["seabed_contact"]) and t_contact is not None and abs(float(t_contact) - 31.48) <= 0.15

    # 5. Zero-flux sediment stability check
    _, ex_diag = exner_step(grid, q_sx=0.0, q_sy=0.0, source_sink=0.0)
    c5_pass = bool(ex_diag["zero_flux_stable"])

    checks = {
        "bathymetric_interpolation": "PASS" if c1_pass else "FAIL",
        "surface_current_attenuation": "PASS" if c2_pass else "FAIL",
        "initial_downward_acceleration": "PASS" if c3_pass else "FAIL",
        "seabed_contact_detection": "PASS" if c4_pass else "FAIL",
        "zero_flux_sediment_stability": "PASS" if c5_pass else "FAIL",
    }
    return {
        "all_passed": all(v == "PASS" for v in checks.values()),
        "checks": checks,
        "synthetic_descent_time_s": t_contact,
        "initial_acceleration_m_s2": round(a0, 6),
        "terminal_velocity_m_s": round(params.terminal_velocity_m_s, 6),
    }


def publish_bathymetry_report(cn: sqlite3.Connection, report: Mapping[str, Any]) -> dict[str, Any]:
    """Persist the latest VNN bathymetry & reconstruction report to
    `brain_kv["entirety:vnn:bathymetry:latest"]` (read-only on the 7-D semantic mesh).
    """
    payload = {"at": time.time(), **dict(report)}
    cn.execute(
        "CREATE TABLE IF NOT EXISTS brain_kv (key TEXT PRIMARY KEY, value TEXT, updated_at REAL)"
    )
    cn.execute(
        "INSERT INTO brain_kv(key, value, updated_at) VALUES(?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (KV_VNN_BATHYMETRY_LATEST, json.dumps(payload, sort_keys=True), payload["at"]),
    )
    return payload


__all__ = [
    "KV_VNN_BATHYMETRY_LATEST",
    "BATHYMETRIC_FEATURE_NAMES",
    "BUOYANCY_PHASE_STATES",
    "WRECK_LIFECYCLE_STATES",
    "SHADOW_ENSEMBLE_NAMES",
    "TemporalSpatialState",
    "BathymetricGrid",
    "load_esri_ascii_grid",
    "load_xyz_csv_grid",
    "fuse_bathymetric_grids",
    "attenuate_current",
    "SurfaceDriftConfig",
    "step_surface_drift",
    "HullGeometry",
    "DescentObjectParams",
    "vertical_acceleration",
    "check_seabed_contact",
    "simulate_descent",
    "ExnerConfig",
    "exner_step",
    "reconstruct_historical_bathymetry",
    "temporal_depression",
    "WreckMorphodynamicState",
    "evaluate_wreck_burial",
    "BathymetricFeatureVector",
    "extract_bathymetric_features",
    "geodesic_distance_m",
    "ReconstructionNode",
    "ReconstructionEdge",
    "PrunedMassLedger",
    "GeodesicReconstructionGraph",
    "bayesian_evidence_update",
    "reconstruct_historical_state",
    "run_shadow_mc_ensembles",
    "build_helena_sloman_1850_case",
    "build_pompeii_79ce_case",
    "export_reconstruction_geojson",
    "export_reconstruction_csv",
    "run_v033_synthetic_checks",
    "publish_bathymetry_report",
]
