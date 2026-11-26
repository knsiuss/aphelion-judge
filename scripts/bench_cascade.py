import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from huggingface_hub import HfApi
import aphelion_judge as aj

try:
    import train_baseline as tb
    if hasattr(tb, "SYSTEM_PROMPT"):
        aj.BASELINE_SYSTEM_PROMPT = tb.SYSTEM_PROMPT
except Exception:
    pass


def main():
    judge = aj.AphelionJudge(device="cuda:0", baseline_sys_prompt=aj.BASELINE_SYSTEM_PROMPT)
    df = pd.read_json("test_ood_pure.jsonl", lines=True)

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Cascade Eval (test_ood_pure)"):
        verdict = judge.evaluate(
            context=r.get("context", ""),
            question=r.get("question", ""),
            response=r.get("response", r.get("answer", "")),
            mode="cascade",
        )
        rec = verdict.to_dict()
        rec["true_grounded"] = bool(r.get("grounded", r.get("label")))
        rows.append(rec)

    res = pd.DataFrame(rows)
    res.to_json("eval_cascade_test_ood_pure.jsonl", orient="records", lines=True)

    y_true = (~res["true_grounded"]).astype(int)
    y_pred = (~res["grounded"]).astype(int)

    summary = {
        "strict_json_pct": float(res["valid_json"].mean() * 100.0),
        "f1_fail": float(f1_score(y_true, y_pred)),
        "precision_fail": float(precision_score(y_true, y_pred)),
        "recall_fail": float(recall_score(y_true, y_pred)),
        "fast_early_exit_pct": float((res["route"] == "fast").mean() * 100.0),
        "avg_output_tokens": float(res["output_tokens"].mean()),
        "latency_p50_ms": float(np.percentile(res["latency_ms"], 50)),
        "latency_mean_ms": float(res["latency_ms"].mean()),
    }
    print("\n=== CASCADE OOD BENCHMARK SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
