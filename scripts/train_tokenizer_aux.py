#!/usr/bin/env python3
"""Train the ECG VQ tokenizer with reconstruction + MASKED AUXILIARY losses.

Rationale (see goal.md): the production tokenizer optimises only
    rec = |recon - signal|.mean()  +  1.0 * commit
which is dominated by large-amplitude QRS deflections and is nearly indifferent to the
small, lead-*relative* amplitude relationships that QRS axis and LVEF depend on. Gradients
provably cannot reach the encoder from the LLM (the Q-Former path consumes int64 code
indices), so the information must be put into the codes at THIS stage.

Auxiliary heads sit on the POST-QUANTIZATION tensor so the codebook itself is forced to
carry the information the LLM will later read.

Self-contained: does not modify the shared runner/config stack.

Launch (3 GPUs):
  torchrun --nproc_per_node=3 scripts/train_tokenizer_aux.py --aux_weight 0.3
"""
import argparse, os, sys, json, math, time
sys.path.insert(0, "/volume/ECG_tokenizer")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import models  # noqa: F401  — import populates the ModelRegistry
from utils.registry import ModelRegistry
from utils.constants import ECG_PATTERNS

PHYS = ["hr", "qrs_dur", "qtc", "pr", "qt"]
ENDPOINTS = ["deepecho_Visually_Estimated_EF", "afib_label_5y", "echonext_shd_binary"]          # z-scored regression targets
TOK_TRAIN = "/media/data1/datasets/ECG_Tokenizer/parquets/train/mimic_mhi_code15_train_updated.parquet"
AUX_TRAIN = "/volume/ECG_tokenizer/output/tokenizer_aux_targets_train.parquet"


class AuxDataset(Dataset):
    """Waveform + 77 binary labels + physiologic regression targets. NaN = 'no target'."""

    def __init__(self, tok_parquet, aux_parquet, length=2500, phys_stats=None,
                 path_col="waveform_path_psa", scale=1.0, labels_from=None):
        import pyarrow.parquet as _pq
        have = _pq.read_schema(tok_parquet).names
        cols = [c for c in ([path_col, "waveform_name"] + ECG_PATTERNS + ENDPOINTS) if c in have]
        df = pd.read_parquet(tok_parquet, columns=cols)
        if labels_from is not None and not all(c in df.columns for c in ECG_PATTERNS):
            lab = pd.read_parquet(labels_from, columns=["waveform_name"] + ECG_PATTERNS).drop_duplicates("waveform_name")
            df = df.merge(lab, on="waveform_name", how="left")
        aux = pd.read_parquet(aux_parquet)
        jk = "waveform_name" if "waveform_name" in aux.columns else path_col
        df = df.merge(aux[[jk] + [c for c in PHYS if c in aux.columns]].drop_duplicates(jk), on=jk, how="left")
        self.path_col, self.scale = path_col, float(scale)
        self.df = df.reset_index(drop=True)
        self.length = length
        self.labels = self.df[ECG_PATTERNS].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        P = self.df[PHYS].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        # z-score physiologic targets using TRAIN stats so heads see comparable scales
        if phys_stats is None:
            mu = np.nanmean(P, 0); sd = np.nanstd(P, 0); sd[sd == 0] = 1.0
            phys_stats = {"mu": mu.tolist(), "sd": sd.tolist()}
        self.phys_stats = phys_stats
        self.phys = (P - np.array(phys_stats["mu"], np.float32)) / np.array(phys_stats["sd"], np.float32)
        # endpoint targets: EF scaled to [0,1] for BCE-on-logit; afib/shd already binary
        E = np.full((len(self.df), len(ENDPOINTS)), np.nan, np.float32)
        for k, c in enumerate(ENDPOINTS):
            if c in self.df.columns:
                v = pd.to_numeric(self.df[c], errors="coerce").to_numpy(np.float32)
                E[:, k] = (v / 100.0) if "EF" in c else (v >= 1).astype(np.float32) * np.where(np.isfinite(v), 1, np.nan)
        self.endp = E
        self.paths = self.df[self.path_col].to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        try:
            s = np.load(self.paths[i]).astype(np.float32) * self.scale
        except Exception:
            s = np.zeros((self.length, 12), np.float32)
        if s.ndim == 3:
            s = s.squeeze(-1)
        if s.shape[0] > self.length:
            s = s[:: max(1, s.shape[0] // self.length), :]
        if s.shape[0] < self.length:
            s = np.pad(s, ((0, self.length - s.shape[0]), (0, 0)))
        s = s[: self.length, :]
        return {"signal": np.transpose(s, (1, 0)).copy(),
                "labels": self.labels[i], "phys": self.phys[i], "endp": self.endp[i]}



class ScalableEncoder(nn.Module):
    """Wider/deeper residual conv encoder with the SAME output interface (B,128,82).

    The production encoder is only 173K params — 15-70x smaller than any ECG model in the
    literature, and smaller than the 811K quantizer it feeds. This scales extraction capacity
    while keeping tokens/codebooks/bitrate identical, so the downstream bridge is unchanged.
    """

    def __init__(self, in_ch=12, width=64, blocks=(2, 2, 2, 2), out_ch=128, out_len=82):
        super().__init__()
        chs = [width, width * 2, width * 4, width * 6]
        layers = [nn.Conv1d(in_ch, chs[0], 15, stride=2, padding=7), nn.BatchNorm1d(chs[0]), nn.GELU()]
        c = chs[0]
        for ci, nb in zip(chs, blocks):
            for b in range(nb):
                stride = 2 if b == 0 else 1
                layers += [nn.Conv1d(c, ci, 7, stride=stride, padding=3), nn.BatchNorm1d(ci), nn.GELU(),
                           nn.Conv1d(ci, ci, 7, stride=1, padding=3), nn.BatchNorm1d(ci), nn.GELU()]
                c = ci
        self.body = nn.Sequential(*layers)
        self.proj = nn.Conv1d(c, out_ch, 1)
        self.out_len = out_len

    def forward(self, x):
        h = self.body(x)
        h = nn.functional.adaptive_avg_pool1d(h, self.out_len)   # guarantees (B, C, 82)
        return self.proj(h)                                       # (B, 128, 82)


class AuxHeads(nn.Module):
    """Small heads on the pooled post-quantization representation (B, 128)."""

    def __init__(self, d=128, n_diag=len(ECG_PATTERNS), n_phys=len(PHYS), n_endp=len(ENDPOINTS)):
        super().__init__()
        def mlp(out):
            return nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, out))
        self.diag = mlp(n_diag)
        self.phys = mlp(n_phys)
        self.endp = mlp(n_endp)

    def forward(self, q):                      # q: (B, 128, 82)
        z = q.mean(dim=-1)                     # (B, 128)  — 128 is the sequence dim
        return self.diag(z), self.phys(z), self.endp(z)


def masked_bce(logits, target):
    """BCE over finite targets only. Returns a graph-connected zero when none are valid
    (required: without it DDP all-reduce buckets mismatch across ranks and NCCL hangs)."""
    m = torch.isfinite(target)
    if m.sum() == 0:
        return 0.0 * logits.sum()
    return nn.functional.binary_cross_entropy_with_logits(logits[m], target[m])


def masked_huber(pred, target):
    m = torch.isfinite(target)
    if m.sum() == 0:
        return 0.0 * pred.sum()
    return nn.functional.smooth_l1_loss(pred[m], target[m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux_weight", type=float, default=0.3)
    ap.add_argument("--diag_weight", type=float, default=1.0)
    ap.add_argument("--phys_weight", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_quantizers", type=int, default=8)
    ap.add_argument("--codebook_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out", default="/volume/ECG_tokenizer/checkpoints/tokenizer_aux")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--manifest", default=None, help="parquet with the signal path column (mV run: mhi_mv_manifest)")
    ap.add_argument("--path_col", default="waveform_path_psa")
    ap.add_argument("--scale", type=float, default=1.0, help="raw->mV factor (MHI 0.00488, MIMIC 0.001)")
    ap.add_argument("--sem_codebooks", type=int, default=0, help="#1 split latent: route aux heads to first-N codebooks; recon uses all")
    ap.add_argument("--endp_weight", type=float, default=0.0, help="E2: weight on direct LVEF/AFib/SHD heads")
    ap.add_argument("--enc_width", type=int, default=0, help=">0 uses ScalableEncoder with this base width")
    ap.add_argument("--labels_from", default=None, help="parquet supplying the 77 labels if the manifest lacks them)")
    a = ap.parse_args()

    rank = int(os.environ.get("RANK", 0)); local = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world > 1
    if ddp:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    ref = rank == 0
    os.makedirs(a.out, exist_ok=True)

    ds = AuxDataset(a.manifest or TOK_TRAIN, AUX_TRAIN, path_col=a.path_col,
                    scale=a.scale, labels_from=a.labels_from)
    if ref:
        print(f"[data] {len(ds):,} ECGs | phys coverage: " +
              ", ".join(f"{p}={np.isfinite(ds.phys[:, i]).mean()*100:.1f}%" for i, p in enumerate(PHYS)), flush=True)
        json.dump(ds.phys_stats, open(f"{a.out}/phys_stats.json", "w"))
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if ddp else None
    dl = DataLoader(ds, batch_size=a.batch_size, sampler=sampler, shuffle=sampler is None,
                    num_workers=a.num_workers, pin_memory=True, drop_last=True, persistent_workers=a.num_workers > 0)

    enc = (ScalableEncoder(width=a.enc_width) if a.enc_width > 0
           else ModelRegistry.get("Residual_Conv_Encoder")()).to(dev)
    if ref:
        print(f"[model] encoder params: {sum(p.numel() for p in enc.parameters()):,}"
              f" ({'scaled w=' + str(a.enc_width) if a.enc_width > 0 else 'baseline 173K'})", flush=True)
    quant = ModelRegistry.get("ECG_Tokenizer_Quantizer")(
        num_quantizers=a.num_quantizers, codebook_size=a.codebook_size).to(dev)
    dec = ModelRegistry.get("Conv_Decoder")().to(dev)
    heads = AuxHeads().to(dev)
    mods = nn.ModuleDict({"enc": enc, "quant": quant, "dec": dec, "heads": heads})
    if ddp:
        mods = DDP(mods, device_ids=[local], find_unused_parameters=False)
    core = mods.module if ddp else mods

    opt = torch.optim.AdamW(mods.parameters(), lr=a.lr, weight_decay=1e-4)
    total = (a.max_steps or len(dl) * a.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)

    wb = None
    if a.wandb and ref:
        import wandb as _w
        wb = _w; _w.init(project="ECG_Tokenizer_Reconstruction", entity="mhi_ai",
                         name=f"tokenizer_aux_w{a.aux_weight}", config=vars(a))

    step = 0; t0 = time.time()
    for ep in range(a.epochs):
        if sampler: sampler.set_epoch(ep)
        mods.train()
        for b in dl:
            sig = b["signal"].to(dev, non_blocking=True)
            lab = b["labels"].to(dev, non_blocking=True)
            phy = b["phys"].to(dev, non_blocking=True)

            feats = core["enc"](sig)
            if a.sem_codebooks > 0:
                q, idx, commit, all_codes = core["quant"](feats, return_all_codes=True)
                head_in = all_codes[:a.sem_codebooks].sum(0)   # first-N codebooks = semantic latent
            else:
                q, idx, commit = core["quant"](feats)
                head_in = q
            recon = core["dec"](q)
            rec = (recon - sig).abs().mean()
            cmt = commit.mean()
            dlog, plog, elog = core["heads"](head_in)
            l_d = masked_bce(dlog, lab)
            l_p = masked_huber(plog, phy)
            l_e = masked_bce(elog, b["endp"].to(dev, non_blocking=True)) if a.endp_weight > 0 else 0.0 * elog.sum()
            loss = rec + 1.0 * cmt + a.aux_weight * (a.diag_weight * l_d + a.phys_weight * l_p
                                                     + a.endp_weight * l_e)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mods.parameters(), 1.0)
            opt.step()
            if step < total - 1: sched.step()
            step += 1

            if ref and step % 100 == 0:
                act = idx.unique().numel() / a.codebook_size * 100
                msg = (f"[ep{ep} step {step}/{total}] loss {loss.item():.4f} rec {rec.item():.4f} "
                       f"cmt {cmt.item():.4f} diag {float(l_d):.4f} phys {float(l_p):.4f} endp {float(l_e):.4f} "
                       f"active_codes {act:.1f}% {(time.time()-t0)/60:.1f}min")
                print(msg, flush=True)
                if wb: wb.log({"train/loss": loss.item(), "train/rec_loss": rec.item(),
                               "train/cmt_loss": cmt.item(), "train/diag_loss": float(l_d),
                               "train/phys_loss": float(l_p), "train/active_codes": act,
                               "train/lr": sched.get_last_lr()[0]}, step=step)
            if ref and step % 2000 == 0:
                torch.save({"encoder": core["enc"].state_dict(), "quantizer": core["quant"].state_dict(),
                            "decoder": core["dec"].state_dict(), "heads": core["heads"].state_dict(),
                            "step": step, "args": vars(a), "phys_stats": ds.phys_stats},
                           f"{a.out}/tokenizer_aux_step{step}.pt")
            if a.max_steps and step >= a.max_steps: break
        if a.max_steps and step >= a.max_steps: break

    if ref:
        torch.save({"encoder": core["enc"].state_dict(), "quantizer": core["quant"].state_dict(),
                    "decoder": core["dec"].state_dict(), "heads": core["heads"].state_dict(),
                    "step": step, "args": vars(a), "phys_stats": ds.phys_stats},
                   f"{a.out}/tokenizer_aux_final.pt")
        print(f"[done] {step} steps in {(time.time()-t0)/60:.1f} min -> {a.out}/tokenizer_aux_final.pt", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
