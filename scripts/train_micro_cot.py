import os
import json
import torch
import pandas as pd
import wandb
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
OUTPUT_DIR = "/kaggle/working/phi35-qlora-micro-cot"
MAX_SEQ_LEN = 1792
MAX_CTX_TOKENS = 1250  # Context-only budget so Question, Response, and JSON target are never truncated

SYSTEM_PROMPT = (
    "You are a strict RAG groundedness judge. Determine whether the Response is fully supported "
    "by the Context alone—do NOT use outside knowledge. If any entity, number, or claim in the "
    "Response is not explicitly supported by the Context, mark grounded as false. "
    'Respond ONLY with valid JSON: {"reason": "<1 concise sentence>", "grounded": true/false}'
)


def truncate_context(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer.encode(str(text), add_special_tokens=False)
    if len(ids) <= max_tokens:
        return str(text)
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True) + " ..."


def build_messages(tokenizer, row: dict, include_target: bool = True):
    ctx = truncate_context(tokenizer, row.get("context", ""), MAX_CTX_TOKENS)
    q = str(row.get("question", ""))[:600]
    ans = str(row.get("response", row.get("answer", "")))[:1200]

    user_content = f"Context:\n{ctx}\n\nQuestion:\n{q}\n\nResponse:\n{ans}"
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if include_target:
        msgs.append({"role": "assistant", "content": row["target_json"]})
    return msgs


def tokenize_example(example, tokenizer):
    prompt_msgs = build_messages(tokenizer, example, include_target=False)
    full_msgs = build_messages(tokenizer, example, include_target=True)

    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]

    # Mask prompt tokens so loss is computed ONLY on the assistant JSON response
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids, batch_mask, batch_labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch_input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch_mask.append(f["attention_mask"] + [0] * pad_len)
            batch_labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    if is_main and os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_df = pd.read_json("train_v2_micro_cot.jsonl", lines=True)
    val_df = pd.read_json("val_v2_micro_cot.jsonl", lines=True)
    if is_main:
        print(f"Dual-T4 DDP Micro-CoT | train: {len(train_df)} | val: {len(val_df)}")

    train_ds = Dataset.from_pandas(train_df).map(
        lambda x: tokenize_example(x, tokenizer),
        remove_columns=list(train_df.columns),
    )
    val_ds = Dataset.from_pandas(val_df).map(
        lambda x: tokenize_example(x, tokenizer),
        remove_columns=list(val_df.columns),
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 * 2 GPUs = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=80,
        save_strategy="steps",
        save_steps=80,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name="08-train-phi35-qlora-micro-cot",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    trainer.train()

    if is_main:
        best_dir = os.path.join(OUTPUT_DIR, "best_adapter")
        trainer.model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        peak_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        print(f"Saved best Micro-CoT adapter -> {best_dir} | Peak VRAM GPU 0: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
