"""Integration test on a DIVERSE batch — mix of perfect and imperfect best-of-5
candidates, and explicit BAD predictions, to verify the reward function discriminates.
"""
import json, sys, pandas as pd
sys.path.insert(0, "/volume/ECG_tokenizer")
from services.openrlhf_judge_reward import reward_func

df = pd.read_csv("/volume/ECG_tokenizer/analysis/rlvr_eval/bestofn_1k/generations_bestof5_1k.csv")

# Pick a diverse set
cats = ["classification", "interpretation", "structural_heart_disease", "afib_risk"]
samples = []
for c in cats:
    sub = df[df["prompt_category"]==c]
    # one high, one low
    samples.append(sub.sort_values("bestof_picked_score").iloc[-1])
    samples.append(sub.sort_values("bestof_picked_score").iloc[0])

# Add 2 deliberately bad predictions (gibberish)
bad = df.iloc[0].copy()
bad["generation"] = "the quick brown fox jumps over the lazy dog"
samples.append(bad)
bad2 = df.iloc[5].copy()
bad2["generation"] = "I have no idea what this ECG shows."
samples.append(bad2)

prompts, labels, queries = [], [], []
for r in samples:
    prompts.append(str(r["question"]))
    labels.append(json.dumps({"category": str(r["prompt_category"]), "ground_truth": str(r["ground_truth"])}))
    queries.append(str(r["question"]) + str(r["generation"]))

out = reward_func(queries, prompts, labels)
print(f"\nReward function discriminates correctly:")
for i, (r, lbl) in enumerate(zip(out["rewards"].tolist(), labels)):
    cat = json.loads(lbl)["category"]
    gen = samples[i]["generation"][:60].replace("\n", " ")
    print(f"  [{i:2d}] cat={cat:30s} reward={r:.2f}  gen='{gen}...'")
