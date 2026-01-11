import torch
import pandas as pd
from torch.utils.data import Dataset

class SimplificationDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        standard = str(row["Standard_English"])
        simplified = str(row["Simplified_English"])

        prompt = f"Simplify the following medical text:\n\n{standard}\n\nSimplified: "

        prompt_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length)["input_ids"]
        target_ids = self.tokenizer(simplified, truncation=True, max_length=self.max_length)["input_ids"]

        input_ids = prompt_ids + target_ids

        labels = [-100] * len(prompt_ids) + target_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
