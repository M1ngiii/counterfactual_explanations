from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = ROOT / "data" / "models" / "trained_3.1.8b"
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
OFFLOAD_DIR = ROOT / ".cache" / "offload"


def load_countereval_model():
    """Load base LLaMA 3.1 8B and attach the adapter."""
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        offload_folder=str(OFFLOAD_DIR),
    )

    # Attach the LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
        device_map="auto",
        offload_folder=str(OFFLOAD_DIR),
    )

    model.eval()
    return model, tokenizer


def main():
    print("Loading CounterEval LLM...")
    model, tokenizer = load_countereval_model()


if __name__ == "__main__":
    main()