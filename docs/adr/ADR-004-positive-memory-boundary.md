# ADR-004: Reject Non-Positive Memory Footprints at Parse Time

## Context

Zero or negative memory values cause division-by-zero or fabricated savings percentages.

## Decision

Raise `ValueError` during fixture loading whenever `fp16_mem_mb <= 0` or `quant_mem_mb <= 0`.

## Alternatives Considered

* Clamping invalid rows silently rejected.
