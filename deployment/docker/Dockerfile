FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY neurovision ./neurovision
RUN python -m pip wheel --no-deps --wheel-dir /wheels .

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1
COPY --from=build /wheels /wheels
RUN python -m pip install --no-index --no-deps /wheels/*.whl && rm -rf /wheels
USER 10001:10001
ENTRYPOINT ["python", "-m", "neurovision"]
