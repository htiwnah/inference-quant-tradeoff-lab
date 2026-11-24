FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build
COPY pyproject.toml setup.cfg LICENSE README.md ./
COPY src/ ./src/
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir /wheels .

FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN python3 -m pip install --no-cache-dir --no-index --find-links=/wheels inference-quant-tradeoff-lab \
    && rm -rf /wheels

COPY Makefile LICENSE README.md CHANGELOG.md ./
COPY policy/ ./policy/
COPY fixtures/ ./fixtures/
COPY sql/ ./sql/
COPY scripts/ ./scripts/
COPY experiments/ ./experiments/
COPY tools/ ./tools/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY assets/ ./assets/

ENTRYPOINT ["python3", "-m", "inference_quant_tradeoff_lab.cli"]
CMD ["audit", "--fixture", "fixtures/quant_eval_cases.jsonl", "--policy", "policy/quant_policy.json"]
