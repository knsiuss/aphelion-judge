# Aphelion-Judge-3.8B

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Aphelion--Judge--3.8B-blue)](https://huggingface.co/knss12/Aphelion-Judge-3.8B)
[![Live Demo Space](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-HF%20Space-indigo)](https://huggingface.co/spaces/knss12/Aphelion-Judge-3.8B-Demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Aphelion-Judge-3.8B** is a lightweight, first-principles RAG groundedness and hallucination verification model (`3.82B` parameters) fine-tuned from `microsoft/Phi-3.5-mini-instruct` using QLoRA and single-sentence **Micro-Chain-of-Thought (Micro-CoT)** supervision.

It audits candidate AI responses strictly against retrieved `Context` passages—without relying on outside parametric knowledge—and outputs deterministic, schema-compliant JSON verdicts (`99.79%` strict JSON compliance on out-of-distribution data).

---

## Out-of-Distribution Benchmark (`test_ood_pure`, n=468)

| Model / Variant | Params | Strict JSON | OOD F1 (FAIL) | Precision | Recall | Avg Output Tokens | Latency `p50` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Phi-3.5-mini Zero-Shot | 3.82B | 0.00% | 0.652 | — | — | — | 1,740.0 ms |
| Flow-Judge-v0.1 | 3.82B | — | 0.752 | — | — | ~130+ | 5,858.0 ms |
| Exp-01: Direct-JSON Baseline | 3.82B | 96.37% | 0.562 | **0.872** | 0.415 | **7.33** | **645.6 ms** |
| **Aphelion-Judge-3.8B (Micro-CoT)** | **3.82B** | **99.79%** | **0.741** | 0.752 | 0.731 | 40.31 | **2,753.3 ms** |
| Cascade (`Exp-01 -> Aphelion-Judge`) | 3.82B | **99.79%** | **0.744** | 0.749 | **0.740** | 36.53 | 3,435.8 ms |

---

## Quickstart

```bash
pip install -e .
```

```python
from aphelion_judge import AphelionJudge

judge = AphelionJudge(backend="transformers")  # or backend="gguf" for CPU/local
verdict = judge.verify(
    context="Apple reported $23,646 million in cash and cash equivalents as of September 2023.",
    question="How much cash did Apple report?",
    response="Apple reported $50 billion in cash reserves.",
)
print(verdict)
# {"reason": "The context states Apple reported $23,646 million in cash, contradicting the claim of $50 billion.", "grounded": false}
```

---

## Repository Structure

- `src/aphelion_judge/`: Clean Python inference SDK supporting both `Transformers` (FP16 GPU) and `llama.cpp` (`Q4_K_M` GGUF CPU/GPU).
- `scripts/`: Training (`train_micro_cot.py`), weight merging (`publish_from_existing_merge.py`), GGUF quantization (`export_gguf.py`), and cascade benchmarking scripts.
- `benchmarks/`: Out-of-distribution evaluation harness measuring strict JSON compliance, F1/Precision/Recall, and p50 latency.
- `space/`: Gradio web UI and JSON-constrained REST endpoint deployed on Hugging Face Spaces.
