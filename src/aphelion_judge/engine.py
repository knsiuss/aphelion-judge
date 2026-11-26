import json
from typing import Literal
from .prompts import JUDGE_SYSTEM_PROMPT, format_verification_prompt

DEFAULT_REPO = "knss12/Aphelion-Judge-3.8B"
DEFAULT_GGUF = "gguf/aphelion-judge-3.8b-Q4_K_M.gguf"


class AphelionJudge:
    def __init__(
        self,
        model_id: str = DEFAULT_REPO,
        backend: Literal["transformers", "gguf"] = "transformers",
        gguf_filename: str = DEFAULT_GGUF,
        n_ctx: int = 2048,
    ):
        self.backend = backend
        if backend == "transformers":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        elif backend == "gguf":
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama

            path = hf_hub_download(repo_id=model_id, filename=gguf_filename)
            self.llm = Llama(model_path=path, n_ctx=n_ctx, verbose=False)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def verify(self, context: str, response: str, question: str = "") -> dict:
        user_msg = format_verification_prompt(context=context, response=response, question=question)
        if self.backend == "transformers":
            import torch

            inputs = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.inference_mode():
                out = self.model.generate(inputs, max_new_tokens=96, do_sample=False)
            raw = self.tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
        else:
            prompt = (
                f"<|system|>\n{JUDGE_SYSTEM_PROMPT}<|end|>\n"
                f"<|user|>\n{user_msg}<|end|>\n"
                f"<|assistant|>\n"
            )
            res = self.llm(
                prompt,
                max_tokens=96,
                temperature=0.0,
                stop=["<|end|>", "<|endoftext|>"],
                response_format={"type": "json_object"},
            )
            raw = res["choices"][0]["text"].strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"reason": raw, "grounded": False}
