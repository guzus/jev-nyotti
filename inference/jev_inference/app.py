from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Receive, Scope, Send

from .engine import Engine, InputTooLong, QwenEngine
from .schemas import ScoreRequest, ScoreResponse
from .settings import MAX_BODY_BYTES, MAX_INPUT_TOKENS, MAX_JOBS, MAX_OPTIONS, MODEL_ID, Settings

logger = logging.getLogger("jev_inference")


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


class GuardedBody:
    """Authenticate before reading a bounded request body, including chunked input."""

    def __init__(self, app: ASGIApp, api_key: str):
        self.app = app
        self.expected = f"Bearer {api_key}".encode("ascii")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope["path"] == "/healthz" and scope["method"] == "GET":
            await self.app(scope, receive, send)
            return
        auth_headers = [value for key, value in scope["headers"] if key.lower() == b"authorization"]
        if len(auth_headers) != 1 or not hmac.compare_digest(auth_headers[0], self.expected):
            await error(401, "unauthorized", "Valid inference credentials are required")(scope, receive, send)
            return
        size = 0
        chunks: list[bytes] = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                await error(413, "request_too_large", "Request body exceeds 64 KiB")(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def replay() -> dict:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def reject_non_json_number(value: str) -> None:
    raise ValueError("non-finite numbers are not JSON")


def create_app(settings: Settings | None = None, *, engine_factory: Callable[[], Engine] | None = None) -> FastAPI:
    configuration = settings or Settings.from_env()
    # Dependency injection exists only as a Python argument for unit tests. There
    # is deliberately no environment variable that enables a fake production model.
    engine = engine_factory() if engine_factory else QwenEngine(configuration)
    gpu_lock = asyncio.Semaphore(1)
    pending = 0

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        await run_in_threadpool(engine.load)
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False

    app = FastAPI(title="Jev Qwen classifier", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.ready = False
    app.add_middleware(GuardedBody, api_key=configuration.api_key)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _exc: RequestValidationError):
        return error(422, "invalid_request", "Request does not match the scoring schema")

    @app.get("/healthz")
    async def healthz():
        return JSONResponse({
            "ready": app.state.ready,
            "model": MODEL_ID,
            "revision": configuration.provenance_revision,
            "fineTuned": configuration.adapter_id is not None,
            "adapterVerification": getattr(engine, "adapter_verification", None),
            "limits": {"jobs": MAX_JOBS, "options": MAX_OPTIONS, "inputTokensPerJob": MAX_INPUT_TOKENS, "bodyBytes": MAX_BODY_BYTES},
        }, status_code=200 if app.state.ready else 503)

    @app.post("/score", response_model=ScoreResponse)
    async def score(request: Request):
        nonlocal pending
        if not app.state.ready:
            return error(503, "not_ready", "Model is not ready")
        # Bound waiting work as well as active work. The gateway should retry 429
        # with backoff; do not fan out into unbounded GPU memory or queue growth.
        if pending >= 8:
            return error(429, "busy", "Inference queue is full")
        pending += 1
        started = time.perf_counter()
        try:
            try:
                payload = json.loads(await request.body(), parse_constant=reject_non_json_number)
                parsed = ScoreRequest.model_validate(payload)
            except (ValueError, UnicodeError, RecursionError, ValidationError):
                return error(422, "invalid_request", "Request does not match the scoring schema")
            async with gpu_lock:
                prepared = await run_in_threadpool(engine.prepare, parsed.jobs)
                scores = await run_in_threadpool(engine.score, prepared)
            return ScoreResponse(
                model=MODEL_ID,
                revision=configuration.provenance_revision,
                scores=scores,
                elapsedMs=round((time.perf_counter() - started) * 1000, 3),
            )
        except InputTooLong:
            return error(422, "input_too_long", f"Each formatted job must fit within {MAX_INPUT_TOKENS} input tokens")
        except Exception as exc:
            # Log a type only: no caller state, credentials, or exception payloads.
            logger.error("Inference failed (%s)", type(exc).__name__)
            return error(503, "inference_unavailable", "Inference failed; retry later")
        finally:
            pending -= 1

    return app
