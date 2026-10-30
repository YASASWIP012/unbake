"""The recipe: what an effect is, once the source clip is gone.

A recipe holds measurements and plain words. It keeps no pixels or audio from
the reference clip, so it can be shared without sharing the clip.
"""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field

CURVE_POINTS = 64


class SourceInfo(BaseModel):
    duration: float
    fps: float
    width: int
    height: int
    shots: list[tuple[float, float]]
    analyzed_shot: int = 0


class CameraLayer(BaseModel):
    enabled: bool = True
    move: str = "static"
    easing: str = "linear"
    # Cumulative camera motion over the shot, CURVE_POINTS samples each.
    # x and y are fractions of frame width, zoom is log scale, roll is radians.
    x: list[float] = Field(default_factory=list)
    y: list[float] = Field(default_factory=list)
    zoom: list[float] = Field(default_factory=list)
    roll: list[float] = Field(default_factory=list)
    jitter: float = 0.0
    confidence: float = 1.0
    gain: float = 1.0


class TransformLayer(BaseModel):
    enabled: bool = True
    present: bool = False
    amount: float = 0.0
    before: str = ""
    after: str = ""
    start: float = 0.0
    end: float = 1.0


class LookLayer(BaseModel):
    enabled: bool = True
    lab_mean: list[float] = Field(default_factory=list)
    lab_std: list[float] = Field(default_factory=list)
    palette: list[str] = Field(default_factory=list)
    saturation: float = 0.0
    contrast: float = 0.0
    strength: float = 0.35


class PacingLayer(BaseModel):
    enabled: bool = True
    energy: list[float] = Field(default_factory=list)
    ramps: list[tuple[float, float]] = Field(default_factory=list)


class SoundLayer(BaseModel):
    enabled: bool = True
    present: bool = False
    envelope: list[float] = Field(default_factory=list)
    onsets: list[float] = Field(default_factory=list)
    description: str = ""


class Recipe(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:10])
    name: str = "Untitled effect"
    source: SourceInfo
    camera: CameraLayer
    transform: TransformLayer
    look: LookLayer
    pacing: PacingLayer
    sound: SoundLayer
    unsupported: list[str] = Field(default_factory=list)
    lineage: list[str] = Field(default_factory=list)
