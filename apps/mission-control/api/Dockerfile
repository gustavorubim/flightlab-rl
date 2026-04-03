FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY apps/mission-control/api/requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt

COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
COPY apps/mission-control /workspace/apps/mission-control

RUN python -m pip install -e .

WORKDIR /workspace/apps/mission-control/api

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
