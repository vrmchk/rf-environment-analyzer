# Backend image: CPU build of torch (Linux), no TorchSig; scenes and the model are mounted.
# Build from the repository root: docker build -f backend/Dockerfile .
FROM ghcr.io/astral-sh/uv:0.13.9 AS uv

FROM python:3.14-slim
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# The workspace lock needs every member's pyproject; only rfea-backend is installed.
COPY pyproject.toml uv.lock ./
COPY ml/pyproject.toml ml/pyproject.toml
COPY packages/rfea_core packages/rfea_core
COPY backend backend
RUN uv sync --frozen --no-dev --package rfea-backend

ENV RFEA_SCENES_DIR=/data/scenes RFEA_MODEL_DIR=/model
EXPOSE 8000
CMD ["uv", "run", "--frozen", "--no-sync", "--package", "rfea-backend", "uvicorn", \
     "rfea_backend.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
