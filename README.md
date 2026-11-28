# Aphelion-Judge-3.8B

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Aphelion--Judge--3.8B-blue)](https://huggingface.co/knss12/Aphelion-Judge-3.8B)
[![HF Downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fknss12%2FAphelion-Judge-3.8B&query=%24.downloads&label=HF%20Downloads&color=yellow&logo=huggingface)](https://huggingface.co/knss12/Aphelion-Judge-3.8B)
[![Eval Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Eval%20Dataset-aphelion--judge--eval--v1-orange)](https://huggingface.co/datasets/knss12/aphelion-judge-eval-v1)
[![Live Demo Space](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-HF%20Space-indigo)](https://huggingface.co/spaces/knss12/Aphelion-Judge-3.8B-Demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Aphelion-Judge-3.8B** is a lightweight, first-principles RAG groundedness and hallucination verification model (`3.82B` parameters) fine-tuned from `microsoft/Phi-3.5-mini-instruct` using QLoRA and single-sentence **Micro-Chain-of-Thought (Micro-CoT)** supervision.

It audits candidate AI responses strictly against retrieved `Context` passages—without relying on outside parametric knowledge—and outputs deterministic, schema-compliant JSON verdicts (`100.0%` strict JSON compliance in FP16 and `99.57%` in `Q4_K_M` GGUF on the frozen out-of-distribution audit).

---

## Reproducible Out-of-Distribution Benchmark (`test_ood_pure`, `n=468`)

All release metrics are frozen and publicly auditable at [`knss12/aphelion-judge-eval-v1`](https://huggingface.co/datasets/knss12/aphelion-judge-eval-v1).

* **Upstream Provenance:** Derived from `PatronusAI/HaluBench` / RAGTruth OOD split (filtered zero-leakage `test_ood_pure`, SHA-256: `40057dd26c37157f...`)
* **Class Distribution:** `218` Supported (`grounded: true`) / `250` Unsupported (`grounded: false`)
* **Pinned Release Commit:** [`52715d8aa2b1bbc9ee728c78ec31815679796167`](https://huggingface.co/knss12/Aphelion-Judge-3.8B/commit/52715d8aa2b1bbc9ee728c78ec31815679796167)
* **Decoding Contract:** `temperature=0.0`, `do_sample=False`, `max_new_tokens=96`
* **Metric Definition:** Binary response-level classification where **Positive Class = `FAIL` (`grounded == false`, hallucinated/ungrounded)**.

### 1. Frozen Release Audit: FP16 Safetensors vs. Quantized GGUF (`Q4_K_M`)

| Release Artifact | Strict JSON | OOD F1 (`FAIL`) | Precision (`FAIL`) | Recall (`FAIL`) | Accuracy | Avg Output Tokens | Latency `p50` | Latency `p95` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Aphelion-Judge-3.8B (`FP16` Safetensors, GPU)** | **100.0%** | **0.7669** | 0.7398 | **0.7960** | 74.15% | 42.86 | **1,994.3 ms** | 3,138.4 ms |
| **Aphelion-Judge-3.8B (`Q4_K_M` GGUF, CPU)** | 99.57% | **0.7686** | **0.7538** | 0.7840 | **74.79%** | **42.68** | 16,481.3 ms | 62,077.3 ms |

> **Quantization Fidelity (`FP16 -> Q4_K_M`):** Delta OOD F1 is `+0.0017`, confirming virtually zero accuracy degradation when compressing the `7.64 GB` FP16 model down to the `2.40 GB` `Q4_K_M` GGUF artifact.

### 2. Ablation & Baseline Comparison (`test_ood_pure`, `n=468`)

| Model / Experiment | Params | Strict JSON | OOD F1 (`FAIL`) | Precision | Recall | Avg Output Tokens | Latency `p50` (GPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Phi-3.5-mini-instruct` (Zero-Shot) | 3.82B | 0.00% | 0.652 | — | — | — | 1,740.0 ms |
| `Flow-Judge-v0.1` | 3.82B | — | 0.752 | — | — | ~130+ | 5,858.0 ms |
| Exp-01: Direct-JSON Baseline (No CoT) | 3.82B | 96.37% | 0.562 | **0.872** | 0.415 | **7.33** | **645.6 ms** |
| **Aphelion-Judge-3.8B (Merged FP16 Release)** | **3.82B** | **100.0%** | **0.7669** | 0.7398 | **0.7960** | 42.86 | 1,994.3 ms |

---

## Quickstart

### 1. Python SDK

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

### 2. Local Deployment (`Ollama` / `Q4_K_M` GGUF)

```bash
huggingface-cli download knss12/Aphelion-Judge-3.8B --include "gguf/*" --local-dir .
ollama create aphelion-judge -f gguf/Modelfile
```

---

## Repository Structure

- `src/aphelion_judge/`: Python inference SDK supporting both `Transformers` (`FP16` GPU) and `llama.cpp` (`Q4_K_M` GGUF CPU/GPU).
- `scripts/`: Training (`train_micro_cot.py`), weight merging (`publish_from_existing_merge.py`), GGUF quantization (`export_gguf.py`), and benchmarking scripts.
- `benchmarks/`: Out-of-distribution evaluation harness measuring strict JSON compliance, F1/Precision/Recall, and `p50`/`p95` latency.
- `space/`: Gradio web UI app reference (live browser-side WebAssembly demo hosted at [`knss12/Aphelion-Judge-3.8B-Demo`](https://huggingface.co/spaces/knss12/Aphelion-Judge-3.8B-Demo)).

---

## Citation

If you use **Aphelion-Judge-3.8B** or its frozen evaluation benchmark in your work, please cite:

```bibtex
@misc{bagaskara2026aphelionjudge,
  title     = {Aphelion-Judge-3.8B: First-Principles Micro-CoT Judge for RAG Groundedness Verification},
  author    = {Bagaskara, P. Kanisius},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/knss12/Aphelion-Judge-3.8B},
  note      = {Merged QLoRA fine-tune of microsoft/Phi-3.5-mini-instruct for RAG groundedness verification}
}
```
