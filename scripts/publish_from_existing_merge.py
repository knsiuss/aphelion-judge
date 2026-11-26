import os
import shutil
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MERGED_DIR = "/tmp/aphelion_judge_merged_fp16"
Q4_GGUF = "/tmp/aphelion-judge-micro-cot-Q4_K_M.gguf"
RELEASE_REPO = "knss12/Aphelion-Judge-3.8B"
MODELFILE_OUT = "/tmp/Modelfile_3_8B"

JUDGE_SYSTEM_PROMPT = """You are Aphelion Judge, an autonomous, first-principles RAG groundedness and factuality verifier.

IDENTITY & MISSION:
Your sole function is to audit whether an AI-generated Response is 100% factually grounded in the provided Context. You act as a deterministic verification gate, not a conversational assistant.

FIRST-PRINCIPLES VERIFICATION PROTOCOL:
1. ZERO OUTSIDE KNOWLEDGE: Treat the provided Context as the entire universe of truth. Never rely on pre-trained world knowledge, common assumptions, or external facts to validate or rescue a claim.
2. ATOMIC CLAIM DECOMPOSITION: Break the Response down into individual atomic assertions—every entity, numerical figure, unit, date, causal relationship, and conditional qualifier.
3. EXPLICIT EVIDENCE MAPPING: Verify each atomic assertion directly against explicit statements in the Context. Paraphrasing is valid only when logical equivalence is preserved without adding new information.
4. UNCERTAINTY & AMBIGUITY = UNGROUNDED: If any assertion lacks explicit textual proof, extrapolates beyond the Context, contains subtle numerical/entity distortion, or leaves room for doubt, you MUST reject it and set "grounded": false. Never give the benefit of the doubt.

OUTPUT CONTRACT:
Respond ONLY with a single valid JSON object matching this exact schema (no markdown fences, no extra text):
{"reason": "<1 concise first-principles sentence citing the exact supporting evidence or identifying the specific unsupported/contradictory claim>", "grounded": true/false}"""

DEFAULT_SYS_ESCAPED = (
    JUDGE_SYSTEM_PROMPT.replace("\\", "\\\\")
    .replace('"', '\\"')
    .replace("\n", "\\n")
)

CHAT_TEMPLATE = (
    '{% set default_sys = "' + DEFAULT_SYS_ESCAPED + '" %}'
    "{% if messages[0]['role'] != 'system' %}"
    "{{ '<|system|>\\n' + default_sys + '<|end|>\\n' }}"
    "{% endif %}"
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|assistant|>\\n' }}"
    "{% endif %}"
)


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = token
    return token


def build_model_card(total_params_b: float, exact_params: int) -> str:
    fence = "`" * 3
    lines = [
        "---",
        "license: mit",
        "language:",
        "  - en",
        "pipeline_tag: text-generation",
        "base_model: microsoft/Phi-3.5-mini-instruct",
        "tags:",
        "  - rag",
        "  - groundedness",
        "  - hallucination-detection",
        "  - llm-as-a-judge",
        "  - micro-cot",
        "  - first-principles",
        "  - phi-3.5",
        "model-index:",
        "  - name: Aphelion-Judge-3.8B",
        "    results:",
        "      - task:",
        "          type: text-classification",
        "          name: RAG Groundedness & Hallucination Verification",
        "        dataset:",
        "          type: aphelion-rag-ood",
        "          name: Aphelion OOD RAG Benchmark (test_ood_pure, n=468)",
        "        metrics:",
        "          - type: f1",
        "            value: 0.741",
        "            name: OOD F1 (Micro-CoT Solo)",
        "          - type: f1",
        "            value: 0.744",
        "            name: OOD F1 (Cascade Fast -> Micro-CoT)",
        "          - type: accuracy",
        "            value: 0.9979",
        "            name: Strict JSON Compliance",
        "---",
        "",
        f"# Aphelion-Judge-{total_params_b:.1f}B",
        "",
        f"**Aphelion-Judge-{total_params_b:.1f}B** (`{exact_params:,}` parameters) is a specialized, first-principles RAG groundedness and hallucination verification model. Fine-tuned from `microsoft/Phi-3.5-mini-instruct` using **Micro-Chain-of-Thought (Micro-CoT)** supervision, it audits AI-generated responses strictly against retrieved `Context` documents and outputs a deterministic JSON verdict (`reason` + `grounded`).",
        "",
        "---",
        "",
        "## Architecture & First-Principles Design Rationale",
        "",
        "### 1. Model Architecture Specifications",
        "* **Backbone:** Dense Decoder-Only Causal Transformer (`microsoft/Phi-3.5-mini-instruct`)",
        f"* **Total Parameters:** `{exact_params:,}` (`~{total_params_b:.2f}B`, FP16 Safetensors + `Q4_K_M` GGUF)",
        "* **Depth & Width:** `32` Transformer blocks, hidden dimension `d_model = 3072`, SwiGLU MLP intermediate dimension `8192`",
        "* **Attention & Positional Encoding:** `32` attention heads with **LongRoPE** (`max_position_embeddings = 131,072` tokens) and **RMSNorm** pre-normalization",
        "* **Adaptation & Merging:** Trained via 4-bit NF4 **QLoRA** (`rank = 16`, `lora_alpha = 32`, targeting `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`) and mathematically merged (`W_merged = W_0 + (alpha / r) * B @ A`) into standalone FP16 weights.",
        "",
        "### 2. Why This Architecture? (First-Principles Rationale)",
        "1. **Why Dense 3.8B + LongRoPE over 7B–70B Judges?**",
        "   RAG verification is a **closed-book contextual entailment task**, not an open-domain world-knowledge recall task. Massive 70B models waste compute storing encyclopedic facts that must actually be *suppressed* during RAG auditing. At `3.82B` parameters with native `128K` LongRoPE context, Aphelion-Judge fits comfortably on single edge/consumer GPUs (or CPU via 4-bit GGUF) while handling long passages at `<3s` latency.",
        "2. **Why Causal Micro-CoT (`reason` before `grounded`)?**",
        "   Predicting a boolean token immediately after a long context compresses multi-claim comparison into one forward pass, causing high false-negative rates (`0.562` OOD F1 in our Direct-JSON baseline). Generating a **single-sentence atomic audit (`reason`) before emitting `grounded`** gives the model explicit working memory, boosting OOD F1 by **+17.9 points (`0.562 -> 0.741`)** while keeping output compact (`~40 tokens` vs `130+ tokens` in verbose judges).",
        "3. **Why Baked-In First-Principles System Prompt?**",
        "   The tokenizer's `chat_template` embeds a strict default system prompt enforcing **zero outside knowledge**, **atomic claim decomposition**, and **uncertainty-as-ungrounded (`false`)**, guaranteeing consistent behavior across `transformers`, `vLLM`, and `llama.cpp`.",
        "",
        "---",
        "",
        "## Out-of-Distribution Benchmark (`test_ood_pure`, n=468)",
        "",
        "| Model / Variant | Params | Strict JSON | OOD F1 (FAIL) | Precision | Recall | Avg Output Tokens | Latency `p50` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        "| Phi-3.5-mini Zero-Shot | 3.82B | 0.00% | 0.652 | — | — | — | 1,740.0 ms |",
        "| Flow-Judge-v0.1 | 3.82B | — | 0.752 | — | — | ~130+ | 5,858.0 ms |",
        "| Exp-01: Direct-JSON Baseline | 3.82B | 96.37% | 0.562 | **0.872** | 0.415 | **7.33** | **645.6 ms** |",
        "| **Aphelion-Judge-3.8B (Micro-CoT)** | **3.82B** | **99.79%** | **0.741** | 0.752 | 0.731 | 40.31 | **2,753.3 ms** |",
        "| Cascade (`Exp-01 -> Aphelion-Judge`) | 3.82B | **99.79%** | **0.744** | 0.749 | **0.740** | 36.53 | 3,435.8 ms |",
        "",
        "---",
        "",
        "## Quickstart",
        "",
        "### 1. Python (`transformers`)",
        "",
        f"{fence}python",
        "import json",
        "import torch",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'model_id = "{RELEASE_REPO}"',
        "tokenizer = AutoTokenizer.from_pretrained(model_id)",
        "model = AutoModelForCausalLM.from_pretrained(",
        "    model_id,",
        "    torch_dtype=torch.float16,",
        '    device_map="auto",',
        ")",
        "",
        "prompt = (",
        '    "Context:\\nApple reported $23,646 million in cash and cash equivalents as of September 2023.\\n\\n"',
        '    "Question:\\nHow much cash did Apple report?\\n\\n"',
        '    "Response:\\nApple reported $50 billion in cash reserves."',
        ")",
        "",
        "inputs = tokenizer.apply_chat_template(",
        '    [{"role": "user", "content": prompt}],',
        "    tokenize=True,",
        "    add_generation_prompt=True,",
        '    return_tensors="pt",',
        ").to(model.device)",
        "",
        "with torch.inference_mode():",
        "    output_ids = model.generate(inputs)",
        "",
        "verdict_raw = tokenizer.decode(output_ids[0][inputs.shape[-1]:], skip_special_tokens=True).strip()",
        "verdict = json.loads(verdict_raw)",
        "print(verdict)",
        f"{fence}",
        "",
        "### 2. Local Deployment (`Ollama` / `llama.cpp` `Q4_K_M`)",
        "",
        f"{fence}bash",
        f'huggingface-cli download {RELEASE_REPO} --include "gguf/*" --local-dir .',
        "ollama create aphelion-judge -f gguf/Modelfile",
        f"{fence}",
        "",
        "---",
        "",
        "## Experimental Archive",
        "",
        "Training splits, QLoRA adapters, and cascade evaluation logs are archived at [`knss12/aphelion-judge`](https://huggingface.co/knss12/aphelion-judge).",
    ]
    return "\n".join(lines) + "\n"


def main():
    assert os.path.exists(MERGED_DIR), f"Folder {MERGED_DIR} belum ada—tunggu cell export_gguf selesai dulu!"
    token = get_hf_token()
    api = HfApi(token=token)

    print("1/3 Updating tokenizer chat_template & generation_config on existing merged build...")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(MERGED_DIR)

    gen_cfg = GenerationConfig.from_pretrained(MERGED_DIR)
    gen_cfg.do_sample = False
    gen_cfg.temperature = None
    gen_cfg.top_p = None
    gen_cfg.top_k = None
    gen_cfg.max_new_tokens = 96
    gen_cfg.save_pretrained(MERGED_DIR)

    # Exact Phi-3.5-mini-instruct parameter count (3,821,079,552 = 3.82B)
    exact_params = 3_821_079_552
    total_params_b = exact_params / 1e9

    with open(os.path.join(MERGED_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(build_model_card(total_params_b, exact_params))

    print(f"2/3 Creating & uploading Safetensors to {RELEASE_REPO}...")
    api.create_repo(repo_id=RELEASE_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=MERGED_DIR,
        path_in_repo=".",
        repo_id=RELEASE_REPO,
        repo_type="model",
        commit_message="release: Aphelion-Judge-3.8B standalone merged model with first-principles system prompt",
    )

    if os.path.exists(Q4_GGUF):
        print("3/3 Uploading Q4_K_M GGUF & Modelfile into release repo...")
        modelfile_txt = (
            "FROM ./aphelion-judge-3.8b-Q4_K_M.gguf\n"
            f'SYSTEM """{JUDGE_SYSTEM_PROMPT}"""\n'
            "PARAMETER temperature 0\n"
            "PARAMETER num_predict 96\n"
            'PARAMETER stop "<|end|>"\n'
            'PARAMETER stop "<|endoftext|>"\n'
        )
        with open(MODELFILE_OUT, "w", encoding="utf-8") as f:
            f.write(modelfile_txt)

        api.upload_file(
            path_or_fileobj=Q4_GGUF,
            path_in_repo="gguf/aphelion-judge-3.8b-Q4_K_M.gguf",
            repo_id=RELEASE_REPO,
            repo_type="model",
            commit_message="feat(gguf): add Q4_K_M quantized GGUF",
        )
        api.upload_file(
            path_or_fileobj=MODELFILE_OUT,
            path_in_repo="gguf/Modelfile",
            repo_id=RELEASE_REPO,
            repo_type="model",
            commit_message="feat(gguf): add Modelfile with baked-in first-principles system prompt",
        )

    print(f"DONE!!!!!!!!!!!!!!!!!!!!! https://huggingface.co/{RELEASE_REPO}")


if __name__ == "__main__":
    main()
