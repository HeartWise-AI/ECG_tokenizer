import os
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from utils.generate_ecg_cf_eval_dataset import (
    create_cf_multicategory_dataset,
)
from utils.metrics.cf_evaluator import CFEvaluator


class DummyTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0}
        self.pad_token_id = 0

    def _encode_words(self, text: str):
        tokens = str(text).split()
        ids = []
        for t in tokens:
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab)
            ids.append(self.vocab[t])
        return ids

    def __call__(self, texts, return_tensors=None, padding=False):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self._encode_words(t) for t in texts]
        max_len = max(len(x) for x in encoded) if padding else None
        if max_len is None:
            max_len = max(len(x) for x in encoded)
        out_ids = []
        for seq in encoded:
            pad = [self.pad_token_id] * (max_len - len(seq))
            out_ids.append(seq + pad)
        result = {"input_ids": torch.tensor(out_ids, dtype=torch.long)}
        result["attention_mask"] = (result["input_ids"] != self.pad_token_id).long()
        return result

    def decode(self, ids, skip_special_tokens=True):  # pragma: no cover - unused
        inv = {i: t for t, i in self.vocab.items()}
        return " ".join(inv.get(i, "?") for i in ids)


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int, good_token_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.good = good_token_id

    def forward(self, ecg_signal, input_ids, attention_mask=None, labels=None):
        # Produce logits biased toward the NEXT token (HF shift) in labels; extra bias for 'good' token
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size, dtype=torch.float32, device=input_ids.device) - 3.0
        if labels is None:
            labels = input_ids.clone()
        for b in range(B):
            for t in range(T):
                if t + 1 >= T:
                    continue
                lid = int(labels[b, t + 1].item())
                if lid >= 0:
                    logits[b, t, lid] += 6.0
                    if lid == self.good:
                        logits[b, t, lid] += 6.0  # extra boost
        return {"logits": logits}


def test_cf_dataset_has_six_records_per_ecg(tmp_path):
    # Minimal two-ECG dataframe with one clear rhythm label
    import pandas as pd
    df = pd.DataFrame([
        {
            "waveform_path_psa": str(tmp_path / "a.npy"),
            "Afib": 1,
            "Left bundle branch block": 0,
            "Left atrial enlargement": 0,
            "Acute pericarditis": 0,
            "ST elevation (inferior - II, III, aVF)": 0,
            "Monomorph": 0,
        },
        {
            "waveform_path_psa": str(tmp_path / "b.npy"),
            "Afib": 0,
            "Left bundle branch block": 1,
            "Left atrial enlargement": 1,
            "Acute pericarditis": 0,
            "ST elevation (inferior - II, III, aVF)": 0,
            "Monomorph": 1,
        },
    ])
    # Create dummy signals
    for p in ["a.npy", "b.npy"]:
        sig = np.random.randn(2500, 12).astype(np.float32)
        np.save(tmp_path / p, sig)

    records = create_cf_multicategory_dataset(df, split_name="test")
    # 2 ECGs × 6 categories
    assert len(records) == 12
    # Check a known ground-truth
    rhythm = [r for r in records if r["category"] == "RHYTHM" and r["ecg_id"] == os.path.basename(df.loc[0, "waveform_path_psa"])][0]
    assert rhythm["ground_truth_answer"] in rhythm["candidate_answers"]


def test_batched_scoring_prefers_good_candidate(tmp_path):
    # Prepare evaluator
    # Create a tiny CF dataset file
    cf_path = tmp_path / "cf.json"
    cf_data = [
        {
            "ecg_id": "x",
            "signal_path": str(tmp_path / "x.npy"),
            "category": "RHYTHM",
            "question": "What is the rhythm?",
            "candidate_answers": ["bad", "verybad", "good"],
            "ground_truth_answer": "good",
            "ground_truth_index": 2,
        }
    ]
    with cf_path.open("w") as f:
        json.dump(cf_data, f)
    # Save ECG
    np.save(tmp_path / "x.npy", np.random.randn(2500, 12).astype(np.float32))

    evaluator = CFEvaluator(str(cf_path), device="cpu")
    tok = DummyTokenizer()
    rec = evaluator.cf_data[0]
    base = evaluator.build_cf_prompt(rec)
    # Prime tokenizer vocab with the exact CF completions we will score
    _ = tok([base + " bad.", base + " verybad.", base + " good."], return_tensors="pt", padding=True)
    good_id = tok.vocab.get("good.")
    assert good_id is not None
    model = DummyModel(vocab_size=len(tok.vocab), good_token_id=good_id)

    signal = evaluator._load_ecg_signal(rec["signal_path"])
    scores = evaluator.score_all_candidates_batched(
        model=model,
        tokenizer=tok,
        ecg_signal=signal,
        base_prompt=base,
        candidates=rec["candidate_answers"],
        category=rec.get("category"),
        model_device=torch.device("cpu"),
    )
    assert int(torch.tensor(scores).argmax().item()) == 2
