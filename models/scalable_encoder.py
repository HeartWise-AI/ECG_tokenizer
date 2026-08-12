"""ScalableEncoder — wider residual-conv ECG encoder with the SAME output
interface (B,128,82) as Residual_Conv_Encoder, registered so the tokenizer
wrapper / LLM-finetuning pipeline can build it by name from a checkpoint config.

The class body is kept byte-identical to the definition used at training time in
`scripts/train_tokenizer_aux.py` so that state_dicts (`body.*`, `proj.*`) load
without remapping. Default `width=64` matches the production winner (x1_split,
enc_width=64, 5,977,408 params); the wrapper instantiates it with no args.
"""
import torch.nn as nn
from utils.registry import ModelRegistry


@ModelRegistry.register("ScalableEncoder")
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
