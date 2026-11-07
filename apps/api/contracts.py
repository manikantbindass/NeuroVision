"""Public HTTP and queue contracts."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from neurovision.core.pipeline import RESEARCH_NOTICE

Kind = Literal["preprocess", "inference", "metrics", "maps", "reports"]
ShortText = Annotated[str, Field(min_length=1, max_length=160, pattern=r"\S")]
RegionCode = Annotated[int, Field(ge=1, le=65535)]
Row = tuple[float, float, float, float]


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PreprocessRequest(Payload):
    scan_id: UUID
    normalization: Literal["zscore", "percentile", "none"] = "zscore"
    seed: int = Field(default=0, ge=0, lt=2**63, strict=True)


class InferenceRequest(Payload):
    scan_id: UUID
    model_id: ShortText


class MetricsRequest(Payload):
    segmentation_id: UUID
    target_scan_id: UUID


class AtlasRegionInput(Payload):
    code: RegionCode
    name: ShortText
    hemisphere: Literal["left", "right", "bilateral", "midline", "unknown"]


class MapRequest(Payload):
    segmentation_id: UUID
    atlas_scan_id: UUID
    atlas_name: ShortText
    atlas_version: ShortText
    atlas_license: ShortText
    atlas_space: ShortText
    registration_provenance: ShortText
    segmentation_to_atlas_world: tuple[Row, Row, Row, Row]
    regions: list[AtlasRegionInput] = Field(min_length=1, max_length=256)


class ReportRequest(Payload):
    artifact_ids: list[UUID] = Field(min_length=1, max_length=12)


JobInput = (
    PreprocessRequest | InferenceRequest | MetricsRequest | MapRequest | ReportRequest
)


class JobView(BaseModel):
    id: UUID
    kind: Kind
    status: Literal["queued", "running", "succeeded", "failed"]
    result_url: str | None = None
    error: ErrorDetail | None = None
    notice: str = RESEARCH_NOTICE
