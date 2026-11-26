import os
import sys
import gc
import json
import time
import torch
import wandb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

assert "unsloth" not in sys.modules, "Unsloth must not be imported!"

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

if local_rank != 0:
    os.environ["WANDB_MODE"] = "disabled"

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
MAX_SEQ_LEN = 1280
OUTPUT_DIR = "/kaggle/working/phi35-qlora-baseline"

SYSTEM_PROMPT = (
    "You are a strict RAG hallucination evaluator. "
    "Determine whether the Answer is completely grounded in the Context without unsupported claims or contradictions. "
    'Respond ONLY with a single-line valid JSON object: {"grounded": true} or {"grounded": false}. '
    "Do not include explanations or extra text."
)

def build_prompt_text(tokenizer, context, question, answer):
    user_msg = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

class GroundednessSFTDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=1280):
        self.samples = []
        end_token = "<|end|>"

        for row in df.itertuples():
            target_str = json.dumps({"grounded": bool(row.grounded)}) + end_token
            target_ids = tokenizer.encode(target_str, add_special_tokens=False)

            user_header = "Context:\n"
            user_footer = f"\n\nQuestion:\n{row.question}\n\nAnswer:\n{row.answer}"
            prefix = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}],
                tokenize=False,
                add_generation_prompt=False,
            ) + "<|user|>\n" + user_header
            suffix = user_footer + "<|end|>\n<|assistant|>\n"

            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
            ctx_ids = tokenizer.encode(str(row.context), add_special_tokens=False)

            budget_for_ctx = max_len - (len(prefix_ids) + len(suffix_ids) + len(target_ids))
            if budget_for_ctx < 32:
                continue
            if len(ctx_ids) > budget_for_ctx:
                ctx_ids = ctx_ids[:budget_for_ctx]

            prompt_ids = prefix_ids + ctx_ids + suffix_ids
            input_ids = prompt_ids + target_ids
            labels = ([-100] * len(prompt_ids)) + target_ids

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def make_collator(pad_token_id):
    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        max_len = ((max_len + 7) // 8) * 8

        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate_fn

def parse_strict_json(text):
    clean = str(text).strip()
    try:
        obj = json.loads(clean)
        if set(obj.keys()) == {"grounded"} and isinstance(obj["grounded"], bool):
            return True, obj["grounded"]
    except Exception:
        pass
    return False, None

def evaluate_split(model, tokenizer, eos_ids, df, split_name, batch_size=4, measure_probe=False):
    prompts = [build_prompt_text(tokenizer, r.context, r.question, r.answer) for r in df.itertuples()]

    probe_p50 = None
    if measure_probe:
        lats = []
        for p in prompts[:20]:
            inp = tokenizer(p, return_tensors="pt", truncation=True, max_length=1260).to("cuda:0")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(
                    **inp,
                    max_new_tokens=16,
                    do_sample=False,
                    eos_token_id=eos_ids,
                    pad_token_id=tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000.0)
            del inp, out
        probe_p50 = float(np.percentile(lats, 50))

    order = np.argsort([len(p) for p in prompts])
    records = [None] * len(prompts)

    for start in tqdm(range(0, len(order), batch_size), desc=f"Eval {split_name}"):
        batch_idx = order[start : start + batch_size]
        batch_prompts = [prompts[i] for i in batch_idx]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1260,
        ).to("cuda:0")
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i, orig_idx in enumerate(batch_idx):
            row = df.iloc[orig_idx]
            gen_ids = outputs[i][prompt_len:]
            non_pad = int((gen_ids != tokenizer.pad_token_id).sum().item())
            raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            is_valid, pred_grounded = parse_strict_json(raw_text)

            rec = {
                "true_grounded": bool(row["grounded"]),
                "pred_grounded": pred_grounded,
                "valid_json": is_valid,
                "output_tokens": non_pad,
                "raw_output": raw_text,
            }
            if "source_ds" in df.columns:
                rec["source_ds"] = row["source_ds"]
            records[orig_idx] = rec

        del inputs, outputs

    res_df = pd.DataFrame(records)
    res_df.to_json(f"eval_qlora_baseline_{split_name}.jsonl", orient="records", lines=True)

    valid_sub = res_df[res_df["valid_json"]]
    yt = (~valid_sub["true_grounded"].astype(bool)).astype(int)
    yp = (~valid_sub["pred_grounded"].astype(bool)).astype(int)

    summary = {
        f"{split_name}/strict_json_pct": float(res_df["valid_json"].mean() * 100),
        f"{split_name}/f1_fail": float(f1_score(yt, yp)),
        f"{split_name}/precision_fail": float(precision_score(yt, yp, zero_division=0)),
        f"{split_name}/recall_fail": float(recall_score(yt, yp, zero_division=0)),
        f"{split_name}/avg_output_tokens": float(res_df["output_tokens"].mean()),
    }
    if probe_p50 is not None:
        summary[f"{split_name}/probe_latency_p50_ms"] = probe_p50

    print(f"\nResults [{split_name}] (n={len(res_df)}):")
    for k, v in summary.items():
        print(f"  {k:<36}: {v:.3f}")
    print("Confusion Matrix [TN, FP / FN, TP]:")
    print(confusion_matrix(yt, yp))

    if "source_ds" in valid_sub.columns:
        print("Per-domain F1 (FAIL):")
        for dom, grp in valid_sub.groupby("source_ds"):
            d_yt = (~grp["true_grounded"].astype(bool)).astype(int)
            d_yp = (~grp["pred_grounded"].astype(bool)).astype(int)
            print(f"  {dom:<15} (n={len(grp):<3}): F1={f1_score(d_yt, d_yp, zero_division=0):.3f} | Rec={recall_score(d_yt, d_yp, zero_division=0):.3f}")

    return summary

def main():
    if local_rank == 0:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_df = pd.read_json("train_v2_balanced.jsonl", lines=True)
    val_df = pd.read_json("val_v2.jsonl", lines=True)
    test_df = pd.read_json("test_ood_pure.jsonl", lines=True)

    train_ds = GroundednessSFTDataset(train_df, tokenizer, max_len=MAX_SEQ_LEN)
    val_ds = GroundednessSFTDataset(val_df, tokenizer, max_len=MAX_SEQ_LEN)
    if local_rank == 0:
        print(f"Dual-T4 DDP | train: {len(train_ds)} | val: {len(val_ds)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_config)
    if local_rank == 0:
        model.print_trainable_parameters()
        wandb.init(project="aphelion-rag-judge", name="07-train-phi35-qlora-baseline", job_type="train")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        weight_decay=0.01,
        fp16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=80,
        save_strategy="steps",
        save_steps=80,
        save_total_limit=2,
        prediction_loss_only=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        report_to="wandb" if local_rank == 0 else "none",
        seed=42,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collator(tokenizer.pad_token_id),
    )

    trainer.train()

    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    if local_rank == 0:
        print(f"Peak VRAM on GPU 0: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        unwrapped_model.save_pretrained(f"{OUTPUT_DIR}/best_adapter")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_adapter")

        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        unwrapped_model.eval()
        unwrapped_model.config.use_cache = True
        tokenizer.padding_side = "left"

        end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
        eos_ids = list({tokenizer.eos_token_id, end_token_id})

        val_metrics = evaluate_split(unwrapped_model, tokenizer, eos_ids, val_df, "val_v2", batch_size=4, measure_probe=False)
        ood_metrics = evaluate_split(unwrapped_model, tokenizer, eos_ids, test_df, "test_ood_pure", batch_size=4, measure_probe=True)

        wandb.log({**val_metrics, **ood_metrics})
        wandb.finish()

if __name__ == "__main__":
    main()
