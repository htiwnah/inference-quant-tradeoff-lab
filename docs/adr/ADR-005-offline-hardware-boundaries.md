# ADR-005: Offline Hardware Timing Boundaries and Resolution Floor

## Context

Offline fixtures cannot measure live TensorRT-LLM/vLLM kernel timing on unprovisioned GPUs.

## Decision

Document offline scope clearly and publish the `2.5% resolution floor` (`1 / 40`).

## Alternatives Considered

* Synthesizing live CUDA wall-clock measurements rejected.
