FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    TOKENIZERS_PARALLELISM=false \
    INFERENCE_DEVICE=cuda

WORKDIR /app
COPY requirements-api.txt requirements-gpu.txt ./
RUN pip install --no-cache-dir -r requirements-gpu.txt \
    && useradd --create-home --uid 10001 inference \
    && mkdir -p /models/huggingface \
    && chown -R inference:inference /models
COPY --chown=inference:inference jev_inference ./jev_inference
USER inference
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"
CMD ["uvicorn", "jev_inference.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--timeout-keep-alive", "5"]
