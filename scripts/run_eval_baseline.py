import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import train_baseline as tb

MODEL_ID = getattr(tb, "MODEL_ID", "microsoft/Phi-3.5-mini-instruct")
ADAPTER_DIR = "/kaggle/working/phi35-qlora-baseline/best_adapter"


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

    print("Loading base model and best_adapter on dedicated GPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    val_df = pd.read_json("val_v2.jsonl", lines=True)
    test_df = pd.read_json("test_ood_pure.jsonl", lines=True)

    val_metrics = tb.evaluate_split(
        model, tokenizer, eos_ids, val_df, "val_v2", batch_size=2, measure_probe=False
    )
    test_metrics = tb.evaluate_split(
        model, tokenizer, eos_ids, test_df, "test_ood_pure", batch_size=2, measure_probe=True
    )

    summary = {"val_v2": val_metrics, "test_ood_pure": test_metrics}
    with open("eval_qlora_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== FINAL EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
