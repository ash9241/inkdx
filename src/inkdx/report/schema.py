"""The machine-readable report schema (pydantic models == the contract).

Downstream tools (VC3D overlays, dashboards, batch triage) consume
report.json; schema_version gates breaking changes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


class GridInfo(BaseModel):
    tile_px: int
    n_tiles: tuple[int, int]  # (rows, cols)
    samples_per_tile: int
    halfwidth: int


class RegionInfo(BaseModel):
    id: int
    verdict: str
    uv_bbox: tuple[int, int, int, int]  # tile-grid coords (r0, c0, r1, c1)
    grid_bbox: tuple[int, int, int, int]  # stored-grid coords
    n_tiles: int
    confidence: float
    explanation: str


class Summary(BaseModel):
    verdict_fractions: dict[str, float]
    dominant_failure: str | None
    headline: str
    regions: list[RegionInfo]


class TileTable(BaseModel):
    """Columnar per-tile data, row-major over the tile grid. None = NaN
    (e.g. NO_DATA tiles have no scores)."""

    verdict: list[int]
    confidence: list[float | None]
    scores: dict[str, list[float | None]]
    metrics: dict[str, list[float | None]]


class Report(BaseModel):
    inkdx_version: str
    schema_version: int = SCHEMA_VERSION
    created: str
    inputs: dict[str, dict] = Field(default_factory=dict)
    calibration: dict = Field(default_factory=dict)
    grid: GridInfo
    summary: Summary
    tiles: TileTable
    maps: dict[str, str] = Field(default_factory=dict)  # metric -> sidecar path
