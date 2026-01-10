import pandas as pd

def normalize_noteaid(df):
    df = df[["ann_text", "gpt_generated"]].rename(
        columns={"ann_text": "term", "gpt_generated": "definition"}
    )
    df = df.dropna(subset=["term", "definition"])
    df["term"] = df["term"].str.lower().str.strip()
    df["definition"] = df["definition"].str.strip()
    df = df.drop_duplicates(subset=["term"])
    return df

def normalize_jargon(df):
    df = df[["input", "output"]].rename(
        columns={"input": "term", "output": "definition"}
    )
    df = df.dropna(subset=["term", "definition"])
    df["term"] = df["term"].str.lower().str.strip()
    df["definition"] = df["definition"].str.strip()
    df = df.drop_duplicates(subset=["term"])
    return df

if __name__ == "__main__":
    syn_path = "/Users/harshitdhar/Documents/gemma_peft_finetuned/scripts/data/readme_syn_good.csv"
    exp_path = "/Users/harshitdhar/Documents/gemma_peft_finetuned/scripts/data/readme_exp_good.csv"
    jargon_path = "/Users/harshitdhar/Documents/gemma_peft_finetuned/scripts/data/medical_jargon.csv"

    syn_df = pd.read_csv(syn_path)
    exp_df = pd.read_csv(exp_path)

    syn_clean = normalize_noteaid(syn_df)
    exp_clean = normalize_noteaid(exp_df)
    noteaid_combined = pd.concat([syn_clean, exp_clean], ignore_index=True)
    noteaid_combined = noteaid_combined.drop_duplicates(subset=["term"])

    jargon_df = pd.read_csv(jargon_path)
    jargon_clean = normalize_jargon(jargon_df)

    final_df = pd.concat([noteaid_combined, jargon_clean], ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["term"])
    final_df.to_csv("dictionary.csv", index=False)

    print("Saved dictionary.csv with", len(final_df), "rows")
    print(final_df.head())
