import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "google/gemma-2b-it"
ADAPTER = "simplifier_ckpt/checkpoint-2000"  

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer=AutoTokenizer.from_pretrained(BASE)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

def simplify(text: str):
    prompt = f"Simplify the medical text:\n{text}\nSimplified:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.9,
            do_sample=False
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return result.split("Simplified:")[-1].strip()

if __name__ == "__main__":
    test = "The lungs are clear of focal consolidation, pleural effusion or pneumothorax."
    print("Input:", test)
    print("Output:", simplify(test))
