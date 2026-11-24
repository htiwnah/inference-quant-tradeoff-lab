# ADR-003: Zero-Dependency Stdlib Core and Python 3.12 Build Isolation

## Context

Python 3.12 stripped pre-installed `setuptools` from base CI virtualenvs.

## Decision

Keep core analysis on Python stdlib and install `pip setuptools wheel` explicitly in CI and Docker.

## Alternatives Considered

* Pulling PyTorch/TensorRT into offline CI analysis rejected.
