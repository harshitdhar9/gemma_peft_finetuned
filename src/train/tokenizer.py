import os
from transformers import AutoTokenizer

MODEL_NAME = "google/gemma-2b-it"
LOCAL_PATH = "tokenizer/"

def get_tokenizer():
    if os.path.exists(LOCAL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)
        print("Loaded tokenizer from local folder.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("Downloaded tokenizer from Hugging Face.")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        tokenizer.save_pretrained(LOCAL_PATH)
        print(f"Saved tokenizer to {LOCAL_PATH}")
    
    return tokenizer

if __name__ == "__main__":
    get_tokenizer()

