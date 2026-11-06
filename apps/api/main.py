"""FastAPI entry point for bounded, authenticated MRI research services."""

import io
import logging
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, Header, Request, Response, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect

from apps.api.adapters import (
    ArtifactStore,
    Authenticator,
    BearerTokens,
    FixedWindow,
    Job,
    JobEvent,
    JobQueue,
    JobRunner,
    LocalJobs,
    MemoryStore,
    RateLimiter,
)
from apps.api.contracts import (
    APIError,
    ErrorResponse,
    InferenceRequest,
    JobInput,
    JobView,
    Kind,
    MapRequest,
    MetricsRequest,
    PreprocessRequest,
    ReportRequest,
)
from apps.api.middleware import RequestBoundary, error_response, log_event
from apps.api.service import (
    Metadata,
    ModelEntry,
    ResearchService,
    Role,
    demo_models,
    nifti_bytes,
)
from neurovision.core.pipeline import RESEARCH_NOTICE
from neurovision.imaging.limits import IngestionLimits
from neurovision.imaging.volume import Modality
from neurovision.reporting.export import pdf_available, report_html, report_pdf
from neurovision.reporting.research import PDFUnavailable, report_json

PREFIX = "/api/v1"
DEFAULT_LIMITS = IngestionLimits(
    max_input_bytes=16 * 1024**2, max_decoded_bytes=32 * 1024**2, max_voxels=2_000_000
)
_ERRORS: dict[int | str, dict[str, Any]] = {
    status: {"model": ErrorResponse}
    for status in (400, 401, 404, 405, 409, 413, 415, 422, 429, 500, 503, 507)
}


def _owner(request: Request) -> str:
    owner = getattr(request.state, "owner", None)
    if not isinstance(owner, str):
        raise APIError(401, "unauthorized", "A valid bearer token is required.")
    return owner


Owner = Annotated[str, Depends(_owner)]


def _logger() -> logging.Logger:
    logger = logging.Logger("neurovision.api", logging.INFO)
    logger.addHandler(logging.StreamHandler())
    return logger


def create_app(
    *,
    authenticator: Authenticator | None = None,
    limiter: RateLimiter | None = None,
    store: ArtifactStore | None = None,
    models: Mapping[str, ModelEntry] | None = None,
    limits: IngestionLimits = DEFAULT_LIMITS,
    queue_factory: Callable[[JobRunner, JobEvent], JobQueue] = LocalJobs,
    logger: logging.Logger | None = None,
) -> FastAPI:
    """Create one local service instance or inject deployment-owned adapters."""
    token = os.environ.get("NEUROVISION_API_TOKEN")
    auth = (
        authenticator
        if authenticator is not None
        else BearerTokens({token: "local-research"} if token else {})
    )
    storage = store if store is not None else MemoryStore()
    log = logger if logger is not None else _logger()
    service = ResearchService(
        storage, models if models is not None else demo_models(), limits
    )

    def job_event(status: str, job: Job, failure: str | None) -> None:
        log_event(
            log,
            "job_complete",
            request_id=job.request_id,
            job_id=str(job.id),
            kind=job.kind,
            status=status,
            exception=failure,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        queue = queue_factory(service.run, job_event)
        app.state.queue = queue
        try:
            yield
        finally:
            await run_in_threadpool(queue.close)

    app = FastAPI(
        title="NeuroVision Research API",
        version="1.0.0",
        description=RESEARCH_NOTICE,
        lifespan=lifespan,
        redirect_slashes=False,
        openapi_url=PREFIX + "/openapi.json",
    )
    app.add_middleware(
        RequestBoundary,
        auth=auth,
        limiter=limiter if limiter is not None else FixedWindow(),
        logger=log,
        upload_bytes=limits.max_input_bytes,
    )

    @app.exception_handler(APIError)
    async def api_error(request: Request, exc: APIError) -> JSONResponse:
        return error_response(exc, request.state.request_id)

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return error_response(
            APIError(422, "invalid_request", "Request does not match the API schema."),
            request.state.request_id,
        )

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        code, message = {
            404: ("not_found", "Route not found."),
            405: ("method_not_allowed", "Method not allowed."),
            413: ("body_too_large", "Request body exceeds its byte limit."),
        }.get(exc.status_code, ("http_error", "HTTP request could not be completed."))
        return error_response(
            APIError(exc.status_code, code, message),
            request.state.request_id,
            exc.headers,
        )

    @app.get(PREFIX + "/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "notice": RESEARCH_NOTICE}

    router = APIRouter(
        prefix=PREFIX,
        responses=_ERRORS,
        dependencies=[Depends(_owner), Security(HTTPBearer(auto_error=False))],
    )

    def queue(request: Request) -> JobQueue:
        return cast(JobQueue, request.app.state.queue)

    def submit(request: Request, owner: str, kind: Kind, body: JobInput) -> JobView:
        service.check_inputs(owner, body)
        return queue(request).submit(owner, kind, body, request.state.request_id)

    @router.post(
        "/scans",
        status_code=201,
        tags=["scans"],
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/octet-stream": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            }
        },
    )
    async def upload(
        request: Request,
        owner: Owner,
        reviewed: Annotated[Literal["true"], Header(alias="X-Deidentified-Research")],
        format: Literal["nii", "nii.gz"] = "nii",
        modality: Modality = Modality.UNKNOWN,
        role: Role = "image",
    ) -> Metadata:
        if (
            request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            != "application/octet-stream"
            or request.headers.get("content-encoding", "identity").lower() != "identity"
        ):
            raise APIError(
                415,
                "unsupported_media_type",
                "Send raw NIfTI bytes as application/octet-stream.",
            )
        with io.BytesIO() as stream:
            try:
                async for chunk in request.stream():
                    stream.write(chunk)
            except ClientDisconnect:
                raise APIError(
                    400, "upload_disconnected", "Upload was interrupted."
                ) from None
            stream.seek(0)
            return await run_in_threadpool(
                service.upload, owner, stream, format, modality, role
            )

    @router.get("/scans", tags=["scans"])
    def scans(owner: Owner) -> dict[str, list[UUID]]:
        return {"ids": storage.keys(owner, "scan")}

    @router.get("/scans/{key}", tags=["scans"])
    def scan(key: UUID, owner: Owner) -> Metadata:
        return service.summary(owner, key, "scan")

    @router.get("/scans/{key}/volume.nii", tags=["scans"], response_class=Response)
    def scan_volume(key: UUID, owner: Owner) -> Response:
        service.summary(owner, key, "scan")
        volume = service.scan(owner, key).volume
        return Response(
            nifti_bytes(volume, volume.data),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="volume.nii"'},
        )

    @router.delete("/scans/{key}", status_code=204, tags=["scans"])
    def delete_scan(key: UUID, owner: Owner) -> Response:
        service.summary(owner, key, "scan")
        storage.remove(owner, key)
        return Response(status_code=204)

    @router.post("/preprocess", status_code=202, tags=["jobs"])
    def preprocess(body: PreprocessRequest, request: Request, owner: Owner) -> JobView:
        return submit(request, owner, "preprocess", body)

    @router.get("/preprocess/{key}", tags=["jobs"])
    def preprocessed(key: UUID, owner: Owner) -> Metadata:
        return service.summary(owner, key, "preprocess")

    @router.get("/preprocess/{key}/volume.nii", tags=["jobs"], response_class=Response)
    def preprocessed_volume(key: UUID, owner: Owner) -> Response:
        service.summary(owner, key, "preprocess")
        volume = service.scan(owner, key).volume
        return Response(
            nifti_bytes(volume, volume.data),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="preprocessed.nii"'},
        )

    @router.get("/models", tags=["models"])
    def model_catalog() -> Metadata:
        return {"models": service.catalog(), "notice": RESEARCH_NOTICE}

    @router.post("/inference", status_code=202, tags=["jobs"])
    def inference(body: InferenceRequest, request: Request, owner: Owner) -> JobView:
        return submit(request, owner, "inference", body)

    @router.get("/segmentations", tags=["results"])
    def segmentations(owner: Owner) -> dict[str, list[UUID]]:
        return {"ids": storage.keys(owner, "inference")}

    @router.get("/segmentations/{key}", tags=["results"])
    def segmentation(key: UUID, owner: Owner) -> Metadata:
        return service.segmentation(owner, key).metadata

    @router.get("/segmentations/{key}/labels.nii", tags=["results"])
    def labels(key: UUID, owner: Owner) -> Response:
        result = service.segmentation(owner, key)
        return Response(
            nifti_bytes(result.volume, result.labels),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="labels.nii"'},
        )

    @router.get("/uncertainty", tags=["results"])
    def uncertainty_results(owner: Owner) -> dict[str, list[UUID]]:
        return {"ids": storage.keys(owner, "inference")}

    @router.get("/uncertainty/{key}", tags=["results"])
    def uncertainty(key: UUID, owner: Owner) -> Metadata:
        result = service.segmentation(owner, key)
        return {
            "id": str(key),
            "notice": RESEARCH_NOTICE,
            **result.metadata["uncertainty"],
        }

    @router.get("/uncertainty/{key}/{map_kind}.nii", tags=["results"])
    def uncertainty_map(
        key: UUID, map_kind: Literal["confidence", "entropy"], owner: Owner
    ) -> Response:
        result = service.segmentation(owner, key)
        data = result.confidence if map_kind == "confidence" else result.entropy
        return Response(
            nifti_bytes(result.volume, data),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{map_kind}.nii"'},
        )

    @router.post("/metrics", status_code=202, tags=["jobs"])
    def metrics(body: MetricsRequest, request: Request, owner: Owner) -> JobView:
        return submit(request, owner, "metrics", body)

    @router.get("/metrics/{key}", tags=["results"])
    def metric_result(key: UUID, owner: Owner) -> Metadata:
        return service.summary(owner, key, "metrics")

    @router.post("/maps", status_code=202, tags=["jobs"])
    def maps(body: MapRequest, request: Request, owner: Owner) -> JobView:
        return submit(request, owner, "maps", body)

    @router.get("/maps/{key}", tags=["results"])
    def map_result(key: UUID, owner: Owner) -> Metadata:
        return service.summary(owner, key, "maps")

    @router.post("/reports", status_code=202, tags=["jobs"])
    def reports(body: ReportRequest, request: Request, owner: Owner) -> JobView:
        return submit(request, owner, "reports", body)

    @router.get("/reports/{key}", tags=["results"])
    def report_result(key: UUID, owner: Owner) -> Metadata:
        return service.summary(owner, key, "reports")

    @router.get("/reports/{key}/exports", tags=["results"])
    def report_exports(key: UUID, owner: Owner) -> dict[str, list[str]]:
        service.summary(owner, key, "reports")
        return {"formats": ["json", "html", *(["pdf"] if pdf_available() else [])]}

    @router.get(
        "/reports/{key}/export.{format}", tags=["results"], response_class=Response
    )
    def export_report(
        key: UUID, format: Literal["json", "html", "pdf"], owner: Owner
    ) -> Response:
        report = service.summary(owner, key, "reports")
        try:
            if format == "pdf":
                data = report_pdf(report)
                media_type = "application/pdf"
            elif format == "html":
                data = report_html(report)
                media_type = "text/html"
            else:
                data = report_json(report)
                media_type = "application/json"
        except PDFUnavailable:
            raise APIError(
                503, "pdf_unavailable", "The optional PDF renderer is unavailable."
            ) from None
        return Response(
            data,
            media_type=media_type,
            headers={
                "Content-Disposition": (
                    f'attachment; filename="research-report.{format}"'
                ),
                "Content-Security-Policy": (
                    "default-src 'none'; style-src 'unsafe-inline'; sandbox"
                ),
            },
        )

    @router.get("/jobs", tags=["jobs"])
    def jobs(request: Request, owner: Owner) -> list[JobView]:
        return queue(request).list(owner)

    @router.get("/jobs/{key}", tags=["jobs"])
    def job_status(key: UUID, request: Request, owner: Owner) -> JobView:
        return queue(request).get(owner, key)

    @router.delete("/jobs/{key}", status_code=204, tags=["jobs"])
    def delete_job(key: UUID, request: Request, owner: Owner) -> Response:
        queue(request).remove(owner, key)
        storage.remove(owner, key)
        return Response(status_code=204)

    app.include_router(router)
    return app
