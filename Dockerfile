FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_CONFIG=/app/configs/app.yaml
ENV MODEL_PATH=/app/models/best.pt
ENV MAX_UPLOAD_BYTES=15728640
ENV CORS_ALLOW_ORIGINS=*

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api api
COPY configs configs
COPY scripts scripts
COPY src src

RUN mkdir -p data models outputs/uploads outputs/predictions outputs/reports

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
