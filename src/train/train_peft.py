import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
from tokenize_dataset import SimplificationDataset

MODEL = "google/gemma-2b-it"
DATA_PATH = "simplification_clean.csv"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,  
        device_map="auto"
    )

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora)

    train_data = SimplificationDataset(DATA_PATH, tokenizer)

    args = TrainingArguments(
        output_dir="simplifier_ckpt",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=20,
        save_steps=200,
        bf16=True,         
        fp16=False,           
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,  
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
    )

    trainer.train()

    model.save_pretrained("simplifier_peft_model")
    print("\nModel saved to simplifier_peft_model")

if __name__ == "__main__":
    main()
