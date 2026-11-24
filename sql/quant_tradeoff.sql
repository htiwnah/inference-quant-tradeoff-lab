CREATE TABLE IF NOT EXISTS quant_cases (
    case_id TEXT PRIMARY KEY,
    slice_name TEXT NOT NULL,
    is_edge_case INTEGER NOT NULL,
    accuracy_drop REAL NOT NULL,
    mem_saved_pct REAL NOT NULL,
    edge_divergence INTEGER NOT NULL
);

SELECT slice_name, COUNT(*) AS edge_losses, ROUND(AVG(accuracy_drop), 4) AS avg_drop
FROM quant_cases
WHERE edge_divergence = 1
GROUP BY slice_name;
