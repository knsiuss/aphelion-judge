import json
import re
import time
from dataclasses import dataclass, asdict
from typing import Optional
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
DEFAULT_REPO_ID = "knss12/aphelion-judge"

BASELINE_SYSTEM_PROMPT = (
    "You are a strict RAG groundedness judge. Determine whether the Response is fully supported "
    "by the Context alone. Respond ONLY with valid JSON: {\"grounded\": true/false}"
)

MICRO_COT_SYSTEM_PROMPT = (
    "You are a strict RAG groundedness judge. Determine whether the Response is fully supported "
    "by the Context alone—do NOT use outside knowledge. If any entity, number, or claim in the "
    "Response is not explicitly supported by the Context, mark grounded as false. "
    'Respond ONLY with valid JSON: {"reason": "<1 concise sentence>", "grounded": true/false}'
)


@dataclass
class JudgeVerdict:
    grounded: bool
    reason: str
    route: str
    valid_json: bool
    latency_ms: float
    output_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_json(text: str) -> tuple[bool, Optional[bool], str]:
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE
    ).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and isinstance(obj.get("grounded"), bool):
            return True, obj["grounded"], str(obj.get("reason", "")).strip()
    except Exception:
        pass
    return False, None, ""


class AphelionJudge:
    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        base_model_id: str = DEFAULT_BASE_MODEL,
        device: str = "cuda:0",
        max_ctx_tokens: int = 1250,
        baseline_sys_prompt: str = BASELINE_SYSTEM_PROMPT,
    ):
        self.max_ctx_tokens = max_ctx_tokens
        self.baseline_sys_prompt = baseline_sys_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eos_ids = [self.tokenizer.eos_token_id]
        end_id = self.tokenizer.convert_tokens_to_ids("<|end|>")
        if isinstance(end_id, int) and end_id not in self.eos_ids:
            self.eos_ids.append(end_id)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map={"": device},
            dtype=torch.float16,
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            repo_id,
            subfolder="experiments/01_qlora_baseline/best_adapter",
            adapter_name="fast",
        )
        self.model.load_adapter(
            repo_id,
            adapter_name="micro_cot",
            subfolder="experiments/02_micro_cot/best_adapter",
        )
        self.model.eval()

    def _truncate_context(self, context: str) -> str:
        ids = self.tokenizer.encode(str(context), add_special_tokens=False)
        if len(ids) <= self.max_ctx_tokens:
            return str(context)
        return self.tokenizer.decode(ids[: self.max_ctx_tokens], skip_special_tokens=True) + " ..."

    def _build_prompt(self, sys_prompt: str, context: str, question: str, response: str) -> str:
        ctx = self._truncate_context(context)
        user_msg = (
            f"Context:\n{ctx}\n\n"
            f"Question:\n{str(question)[:600]}\n\n"
            f"Response:\n{str(response)[:1200]}"
        )
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _run_adapter(self, adapter_name: str, sys_prompt: str, context: str, question: str, response: str, max_new_tokens: int):
        self.model.set_adapter(adapter_name)
        prompt = self._build_prompt(sys_prompt, context, question, response)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.eos_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        gen_ids = out[0][prompt_len:]
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        valid, grounded, reason = _parse_json(raw_text)
        return valid, grounded, reason, len(gen_ids), raw_text

    def evaluate(self, context: str, response: str, question: str = "", mode: str = "cascade") -> JudgeVerdict:
        t0 = time.perf_counter()
        total_tokens = 0

        if mode in {"fast", "cascade"}:
            valid, grounded, _, toks, _ = self._run_adapter(
                "fast", self.baseline_sys_prompt, context, question, response, max_new_tokens=16
            )
            total_tokens += toks
            if mode == "fast" or (valid and grounded is False):
                return JudgeVerdict(
                    grounded=bool(grounded) if valid else True,
                    reason="Flagged as unsupported by fast-path verifier." if grounded is False else "Supported (fast-path).",
                    route="fast",
                    valid_json=valid,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    output_tokens=total_tokens,
                )

        valid, grounded, reason, toks, raw = self._run_adapter(
            "micro_cot", MICRO_COT_SYSTEM_PROMPT, context, question, response, max_new_tokens=96
        )
        total_tokens += toks
        return JudgeVerdict(
            grounded=bool(grounded) if valid else True,
            reason=reason if valid else f"Parse fallback: {raw[:80]}",
            route="micro_cot",
            valid_json=valid,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            output_tokens=total_tokens,
        )
