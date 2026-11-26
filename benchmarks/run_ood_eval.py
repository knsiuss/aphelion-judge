import json
import time
import numpy as np
from aphelion_judge import AphelionJudge


def evaluate_ood(split_file: str = "test_ood_pure.jsonl", limit: int | None = None):
    judge = AphelionJudge(backend="transformers")
    rows = [json.loads(line) for line in open(split_file, encoding="utf-8")]
    if limit:
        rows = rows[:limit]

    strict_json = 0
    tp = fp = fn = 0
    latencies = []

    for row in rows:
        t0 = time.perf_counter()
        verdict = judge.verify(
            context=row["context"],
            question=row.get("question", ""),
            response=row["response"],
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

        if isinstance(verdict, dict) and isinstance(verdict.get("grounded"), bool):
            strict_json += 1
        pred_fail = not bool(verdict.get("grounded", False))
        gold_fail = not bool(row["grounded"])

        if pred_fail and gold_fail:
            tp += 1
        elif pred_fail and not gold_fail:
            fp += 1
        elif not pred_fail and gold_fail:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    summary = {
        "samples": len(rows),
        "strict_json_pct": round(100.0 * strict_json / len(rows), 2),
        "f1_fail": round(f1, 4),
        "precision_fail": round(precision, 4),
        "recall_fail": round(recall, 4),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    evaluate_ood()
