# ADR-002: Explicit Three-Way Exit Code Contract

## Context

CI deployment pipelines need to separate a quantization regression from malformed JSON.

## Decision

Return `0` on pass, `1` when edge divergences exceed budget, and `2` on invalid input.

## Alternatives Considered

* Single non-zero exit code rejected.
