# syntax=docker/dockerfile:1
# QUIPU Observer — learning hub for the Loadopoly-OCR / Bakugo OCR fleet.

FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    SCB_DB_PATH=/data/local_brain.sqlite \
    QUIPU_BRAIN_OWNER=1 \
    QUIPU_HOST=0.0.0.0 \
    QUIPU_PORT=7100

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
# docs/ is QUIPU's self-knowledge (the local_docs ingest source) and where doc
# annealing writes the living map; compose mounts the repo's docs over it.
COPY docs/ ./docs/

RUN mkdir -p /data
VOLUME /data

EXPOSE 7100
HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=5 \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7100/health',timeout=4).status==200 else 1)"

# The whole Entirety, one writer of the brain (v0.48.0): observer + expansion +
# the operator's pulse + doc annealing.  See src/quipu/entirety_service.py.
CMD ["python", "-m", "src.quipu.entirety_service"]
