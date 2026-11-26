import os
import subprocess
import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
REPO_ID = "knss12/aphelion-judge"
MERGED_DIR = "/tmp/aphelion_judge_merged_fp16"
LLAMA_CPP_DIR = "/tmp/llama.cpp"
F16_GGUF = "/tmp/aphelion-judge-micro-cot-f16.gguf"
Q4_GGUF = "/tmp/aphelion-judge-micro-cot-Q4_K_M.gguf"
MODELFILE_PATH = "/tmp/Modelfile"

JUDGE_SYSTEM_PROMPT = (
    "You are Aphelion Judge, a strict, first-principles RAG groundedness and factuality verifier.\n\n"
    "CORE OPERATING RULES:\n"
    "1. ZERO OUTSIDE KNOWLEDGE: Rely exclusively on the provided Context. Never assume, extrapolate, or fill gaps with external or pre-trained knowledge.\n"
    "2. FIRST-PRINCIPLES VERIFICATION: Decompose the Response into atomic claims (entities, numbers, causal links, conditions) and verify each one directly against explicit evidence in the Context.\n"
    "3. UNCERTAINTY = UNGROUNDED: If any claim lacks explicit support, is ambiguous, or leaves room for doubt, treat it as unsupported and set \"grounded\": false. Never give the benefit of the doubt.\n\n"
    'Respond ONLY with valid JSON: {"reason": "<1 concise first-principles sentence citing exact support or missing evidence>", "grounded": true/false}'
)


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            return token
    except Exception as e:
        raise RuntimeError(f"HF_TOKEN tidak ditemukan di env maupun Kaggle Secrets: {e}")
    raise RuntimeError("HF_TOKEN kosong.")


def run_cmd(cmd: list[str]):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    token = get_hf_token()

    print("1/4 Loading base FP16 model & merging 02_micro_cot LoRA adapter on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        REPO_ID,
        subfolder="experiments/02_micro_cot/best_adapter",
    )
    merged = model.merge_and_unload()
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    del merged, model, base_model

    print("2/4 Building llama.cpp quantizer...")
    if not os.path.exists(LLAMA_CPP_DIR):
        run_cmd(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", LLAMA_CPP_DIR])
    run_cmd(["cmake", "-B", f"{LLAMA_CPP_DIR}/build", "-S", LLAMA_CPP_DIR, "-DGGML_CUDA=OFF"])
    run_cmd(["cmake", "--build", f"{LLAMA_CPP_DIR}/build", "--target", "llama-quantize", "-j4"])

    print("3/4 Converting HF -> F16 GGUF -> Q4_K_M GGUF...")
    run_cmd(["python", f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py", MERGED_DIR, "--outfile", F16_GGUF, "--outtype", "f16"])
    run_cmd([f"{LLAMA_CPP_DIR}/build/bin/llama-quantize", F16_GGUF, Q4_GGUF, "Q4_K_M"])

    if os.path.exists(F16_GGUF):
        os.remove(F16_GGUF)

    modelfile_content = (
        f"FROM ./aphelion-judge-micro-cot-Q4_K_M.gguf\n"
        f'SYSTEM """{JUDGE_SYSTEM_PROMPT}"""\n'
        f"PARAMETER temperature 0\n"
        f"PARAMETER num_predict 96\n"
        f'PARAMETER stop "<|end|>"\n'
        f'PARAMETER stop "<|endoftext|>"\n'
    )
    with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print("4/4 Uploading Q4_K_M GGUF & Modelfile to Hugging Face...")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=Q4_GGUF,
        path_in_repo="gguf/aphelion-judge-micro-cot-Q4_K_M.gguf",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="feat(gguf): add Q4_K_M quantized merged Micro-CoT model",
    )
    api.upload_file(
        path_or_fileobj=MODELFILE_PATH,
        path_in_repo="gguf/Modelfile",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="feat(gguf): add Modelfile with first-principles & zero-assumption system prompt",
    )
    print(f"DONE! GGUF + Modelfile siap dipakai lokal -> https://huggingface.co/{REPO_ID}/tree/main/gguf")


if __name__ == "__main__":
    main()
