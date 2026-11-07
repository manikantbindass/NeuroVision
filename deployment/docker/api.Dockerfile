FROM python:3.12-slim AS wheel
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY neurovision ./neurovision
COPY apps/__init__.py ./apps/__init__.py
COPY apps/api ./apps/api
RUN python -m pip wheel --no-deps --wheel-dir /wheels .

FROM python:3.12-slim
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp OMP_NUM_THREADS=1
COPY --from=wheel /wheels /wheels
RUN python -m pip install --no-cache-dir --no-deps "torch>=2.8,<3" --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install --no-cache-dir /wheels/*.whl \
    && python -m pip install --no-cache-dir "neurovision-2028[api,imaging,training,distributed,pdf]" \
    && python -m pip check \
    && rm -rf /wheels
USER 10001:10001
EXPOSE 8000 9000
CMD ["python", "-m", "uvicorn", "apps.api.deployed:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
