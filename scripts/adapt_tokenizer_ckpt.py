#!/usr/bin/env python
"""Convert a standalone train_tokenizer_aux.py checkpoint into the production
`pretrained_tokenizer_path` format consumed by LLMFinetuningProject.

Standalone format: {encoder, quantizer, decoder, heads, args, phys_stats, step}
  - encoder  keys: <ResidualConvEncoder or ScalableEncoder>.state_dict()
  - quantizer keys: ECG_Tokenizer_Quantizer.state_dict()  -> 'quantizer.layers.*'
  - decoder  keys: Conv_Decoder.state_dict()

Production format: {model_state_dict, config, epoch}
  - model_state_dict keys: 'encoder.*', 'quantizer.quantizer.*', 'decoder.*'
  - config: object/dict with encoder_name, quantizer_name, num_quantizers, codebook_size

The wrapper's _load_pretrained_weights only consumes 'encoder.*'/'quantizer.*'
keys (by name+shape) and ignores the rest, so aux heads and the conv decoder
are dropped automatically. This script also VALIDATES that the produced
encoder.*/quantizer.* keys match a reference production checkpoint exactly, so a
successful run guarantees a drop-in load.
"""
import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ref config unpickle

PROD_REF = "/media/data1/models/ECG_Tokenizer/ECG_Tokenizer_Reconstruction/tfq5q94l_20250622-004552/best_model_epoch_10.pt"


def build_model_state_dict(ckpt):
    msd = {}
    for k, v in ckpt["encoder"].items():
        msd[f"encoder.{k}"] = v
    for k, v in ckpt["quantizer"].items():
        msd[f"quantizer.{k}"] = v          # -> quantizer.quantizer.layers.*
    for k, v in ckpt.get("decoder", {}).items():
        msd[f"decoder.{k}"] = v
    return msd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="standalone tokenizer_aux checkpoint")
    ap.add_argument("--out", required=True, help="output production-format checkpoint")
    ap.add_argument("--ref", default=PROD_REF, help="reference production checkpoint for key validation")
    ap.add_argument("--no-validate", action="store_true")
    a = ap.parse_args()

    ckpt = torch.load(a.inp, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {}) or {}
    enc_width = int(args.get("enc_width", 0) or 0)
    encoder_name = "ScalableEncoder" if enc_width > 0 else "Residual_Conv_Encoder"
    nq = int(args.get("num_quantizers", 8) or 8)
    cb = int(args.get("codebook_size", 512) or 512)

    msd = build_model_state_dict(ckpt)
    config = {
        "encoder_name": encoder_name,
        "quantizer_name": "ECG_Tokenizer_Quantizer",
        "model_name": "ECG_Tokenizer_Wrapper",
        "num_quantizers": nq,
        "codebook_size": cb,
    }
    out = {"model_state_dict": msd, "config": config, "epoch": int(args.get("epochs", 0) or 0),
           "source_checkpoint": a.inp, "sem_codebooks": int(args.get("sem_codebooks", 0) or 0)}

    print(f"[adapt] source={a.inp}")
    print(f"[adapt] encoder_name={encoder_name} nq={nq} cb={cb} sem_codebooks={out['sem_codebooks']}")
    print(f"[adapt] model_state_dict: {len(msd)} keys "
          f"(encoder={sum(k.startswith('encoder.') for k in msd)}, "
          f"quantizer={sum(k.startswith('quantizer.') for k in msd)}, "
          f"decoder={sum(k.startswith('decoder.') for k in msd)})")

    if not a.no_validate:
        import models  # noqa: F401 — populates ModelRegistry
        from utils.registry import ModelRegistry
        ref = torch.load(a.ref, map_location="cpu", weights_only=False)["model_state_dict"]
        # Reference keys: quantizer always from the production ref (same class).
        # Encoder: from the ref if Residual_Conv (drop-in), else from a freshly-built
        # registered instance of the named encoder (must be registered for the pipeline to build it).
        ref_keys = {k: v.shape for k, v in ref.items() if k.startswith("quantizer.")}
        if encoder_name == "Residual_Conv_Encoder":
            ref_keys.update({k: v.shape for k, v in ref.items() if k.startswith("encoder.")})
        else:
            enc_cls = ModelRegistry.get(encoder_name)
            if enc_cls is None:
                print(f"[validate] FAILED — encoder '{encoder_name}' is NOT registered; the pipeline "
                      f"cannot build it. Register it in models/ first.", file=sys.stderr)
                sys.exit(1)
            for k, v in enc_cls().state_dict().items():
                ref_keys[f"encoder.{k}"] = v.shape
            print(f"[validate] encoder ref = freshly-built registered {encoder_name}() ({len(ref_keys)-158} enc keys)")
        ok = True
        for prefix in ("encoder.", "quantizer."):
            rk = {k: v for k, v in ref_keys.items() if k.startswith(prefix)}
            mk = {k: v.shape for k, v in msd.items() if k.startswith(prefix)}
            miss = set(rk) - set(mk); extra = set(mk) - set(rk)
            shape_mismatch = {k for k in set(rk) & set(mk) if rk[k] != mk[k]}
            status = "OK" if not (miss or extra or shape_mismatch) else "MISMATCH"
            if status != "OK":
                ok = False
            print(f"[validate] {prefix:11s} ref={len(rk)} mine={len(mk)} "
                  f"missing={len(miss)} extra={len(extra)} shape_mismatch={len(shape_mismatch)} -> {status}")
            if miss:   print("    missing sample:", list(miss)[:3])
            if extra:  print("    extra sample:  ", list(extra)[:3])
            if shape_mismatch: print("    shape sample: ", {k: (rk[k], mk[k]) for k in list(shape_mismatch)[:3]})
        if not ok:
            print("[validate] FAILED — encoder/quantizer keys differ from reference.", file=sys.stderr)
            sys.exit(1)
        print(f"[validate] PASS — encoder.*({encoder_name})/quantizer.* match reference exactly; guaranteed load.")

    torch.save(out, a.out)
    print(f"[adapt] wrote {a.out}")


if __name__ == "__main__":
    main()
