# ADR-001: Separate Tail Edge-Case Erosion From Global Mean Accuracy

## Context

Bulk easy prompts keep mean accuracy drop below 0.05 even while edge slices drop by 0.18.

## Decision

Gate explicitly on `edge_divergence_cases_count` alongside `mean_accuracy_drop`.

## Alternatives Considered

* Gating on aggregate mean accuracy alone rejected.
