import json
import re
import time
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import train_micro_cot as tmc

MODEL_ID = getattr(tmc, "MODEL_ID", "microsoft/Phi-3.5-mini-instruct")
ADAPTER_DIR = "/kaggle/working/phi35-qlora-micro-cot/best_adapter"
MAX_CTX_TOKENS = getattr(tmc, "MAX_CTX_TOKENS", 1250)
SYSTEM_PROMPT = tmc.SYSTEM_PROMPT


def parse_grounded_json(text: str):
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and isinstance(obj.get("grounded"), bool):
            return True, obj["grounded"], str(obj.get("reason", ""))
    except Exception:
        pass
    return False, None, ""


def extract_true_grounded(row) -> bool:
    for col in ["grounded", "label", "is_grounded", "true_grounded"]:
        if col in row and pd.notna(row[col]):
            val = row[col]
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return bool(val)
            s = str(val).strip().lower()
            if s in {"true", "1", "grounded", "pass", "yes"}:
                return True
            if s in {"false", "0", "hallucinated", "fail", "no"}:
                return False
    raise ValueError(f"Cannot determine ground-truth label from row keys: {list(row.index)}")


def format_prompt(tokenizer, row) -> str:
    ctx = tmc.truncate_context(tokenizer, row.get("context", ""), MAX_CTX_TOKENS)
    q = str(row.get("question", ""))[:600]
    ans = str(row.get("response", row.get("answer", "")))[:1200]
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion:\n{q}\n\nResponse:\n{ans}"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def evaluate_split(model, tokenizer, eos_ids, df: pd.DataFrame, split_name: str, batch_size: int = 2, measure_probe: bool = False):
    prompts = [format_prompt(tokenizer, row) for _, row in df.iterrows()]
    records = []
    probe_latencies_ms = []

    for start in tqdm(range(0, len(prompts), batch_size), desc=f"Eval {split_name}"):
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        for idx_in_batch, out_ids in enumerate(outputs):
            row_idx = start + idx_in_batch
            row = df.iloc[row_idx]
            gen_ids = out_ids[prompt_len:]
            # Trim trailing pad/eos tokens for accurate token count
            non_pad = [tid for tid in gen_ids.tolist() if tid != tokenizer.pad_token_id]
            raw_out = tokenizer.decode(gen_ids, skip_special_tokens=True)
            valid_json, pred_grounded, reason = parse_grounded_json(raw_out)
            true_grounded = extract_true_grounded(row)

            if measure_probe and batch_size == 1:
                probe_latencies_ms.append(elapsed_ms)

            records.append({
                "row_idx": row_idx,
                "source_ds": row.get("source_ds", "unknown"),
                "true_grounded": true_grounded,
                "pred_grounded": pred_grounded,
                "valid_json": valid_json,
                "reason": reason,
                "output_tokens": len(non_pad),
                "raw_output": raw_out,
            })

    # Dedicated single-item latency probe on first 30 OOD rows for clean p50 measurement
    if measure_probe and not probe_latencies_ms:
        for p in prompts[:30]:
            inp = tokenizer([p], return_tensors="pt").to(model.device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model.generate(
                    **inp,
                    max_new_tokens=96,
                    do_sample=False,
                    eos_token_id=eos_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            probe_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    out_path = f"eval_micro_cot_{split_name}.jsonl"
    res_df = pd.DataFrame(records)
    res_df.to_json(out_path, orient="records", lines=True)

    valid_df = res_df[res_df["valid_json"]]
    strict_pct = float(res_df["valid_json"].mean() * 100.0)

    # Positive class = FAIL / HALLUCINATED (grounded == False)
    y_true = (~valid_df["true_grounded"].astype(bool)).astype(int)
    y_pred = (~valid_df["pred_grounded"].astype(bool)).astype(int)

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    avg_tok = float(res_df["output_tokens"].mean())
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        f"{split_name}/strict_json_pct": strict_pct,
        f"{split_name}/f1_fail": f1,
        f"{split_name}/precision_fail": prec,
        f"{split_name}/recall_fail": rec,
        f"{split_name}/avg_output_tokens": avg_tok,
    }
    if probe_latencies_ms:
        metrics[f"{split_name}/probe_latency_p50_ms"] = float(np.percentile(probe_latencies_ms, 50))

    print(f"\nResults [{split_name}] (n={len(res_df)}):")
    for k, v in metrics.items():
        print(f"  {k:<32}: {v:.3f}")
    print("Confusion Matrix [TN, FP / FN, TP]:\n", cm)

    if "source_ds" in valid_df.columns:
        print("Per-domain F1 (FAIL):")
        for dom, grp in valid_df.groupby("source_ds"):
            d_yt = (~grp["true_grounded"].astype(bool)).astype(int)
            d_yp = (~grp["pred_grounded"].astype(bool)).astype(int)
            print(f"  {str(dom):<15} (n={len(grp):<3}): F1={f1_score(d_yt, d_yp, zero_division=0):.3f} | Rec={recall_score(d_yt, d_yp, zero_division=0):.3f}")

    return metrics


def main():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_ids = [tokenizer.eos_token_id]
    end_id = tokenizer.convert_tokens_to_ids("<|end|>")
    if isinstance(end_id, int) and end_id not in eos_ids:
        eos_ids.append(end_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model and Micro-CoT best_adapter on dedicated GPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    val_df = pd.read_json("val_v2.jsonl", lines=True)
    test_df = pd.read_json("test_ood_pure.jsonl", lines=True)

    val_metrics = evaluate_split(model, tokenizer, eos_ids, val_df, "val_v2", batch_size=2, measure_probe=False)
    test_metrics = evaluate_split(model, tokenizer, eos_ids, test_df, "test_ood_pure", batch_size=2, measure_probe=True)

    summary = {"val_v2": val_metrics, "test_ood_pure": test_metrics}
    with open("eval_micro_cot_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== MICRO-COT FINAL EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
