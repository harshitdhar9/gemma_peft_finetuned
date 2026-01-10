import pandas as pd
from datasets import load_dataset
import os
os.makedirs("data", exist_ok=True)

ds = load_dataset("vishnukantshukla/medical-complex-to-simple-10k")["train"]
df = ds.to_pandas()
df.to_csv("data/simplification.csv", index=False)

ds1 = load_dataset("huhucheck/medical_jargon")["train"]
df1 = ds1.to_pandas()
df1.to_csv("data/medical_jargon.csv", index=False)
