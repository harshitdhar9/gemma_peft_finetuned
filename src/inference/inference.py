import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig

BASE = "google/gemma-2b-it"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER = os.path.join(PROJECT_ROOT, "train", "simplifier_ckpt", "checkpoint-2000")

print(f"[INFO] Using adapter path: {ADAPTER}")

tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
config_path = os.path.join(ADAPTER, "adapter_config.json")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"[ERROR] adapter_config.json not found at {config_path}")

with open(config_path, "r") as f:
    config = LoraConfig(**json.load(f))

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER,
    config=config,
    local_files_only=True
)

model.eval()

def clean_output(text: str):
    text = text.strip()
    for sep in ["\n", ". "]:
        if sep in text:
            return text.split(sep)[0].strip()

    return text

def simplify(text: str):
    prompt = (
        "Rewrite the following medical text in short simple language "
        "for a non-medical person.\n\n"
        f"[Medical] {text}\n\n[Simple]"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,            
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            max_new_tokens=120,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    if "[Simple]" in decoded:
        decoded = decoded.split("[Simple]")[-1].strip()

    return clean_output(decoded)

if __name__ == "__main__":
    test = "The lungs are clear of focal consolidation, effusion or pneumothorax."
    print("[INPUT]:", test)
    print("[OUTPUT]:", simplify(test))
