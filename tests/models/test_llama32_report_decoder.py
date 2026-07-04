import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from models.llama32_report_decoder import Llama32ReportDecoder
from utils.enums import AdapterName


def _tiny_llama_base() -> LlamaForCausalLM:
    """Build a CPU-friendly random-init Llama model.

    Uses a hidden size that matches SequenceAdapter's output (we pass
    llm_input_embedding_size=64 in tests), 2 layers, 4 heads with head_dim=16.
    """
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
    )
    return LlamaForCausalLM(cfg)


class TestLlama32ReportDecoder(unittest.TestCase):

    def setUp(self):
        self.base_patcher = patch(
            "models.llama32_report_decoder.LlamaForCausalLM.from_pretrained",
            return_value=_tiny_llama_base(),
        )
        self.base_patcher.start()

        self.decoder = Llama32ReportDecoder(
            huggingface_model_name="dummy/llama-tiny",
            llm_input_embedding_size=64,
            quantized_feature_shape=(128, 82),
            adapter_name=AdapterName.LLAMA32_SEQUENCE_ADAPTER,
            adapter_dropout=0.0,
            lora_r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            lora_target_modules=("q_proj", "v_proj"),
        )

        self.batch_size = 2
        self.seq_len = 5
        self.quantized = torch.randn(self.batch_size, 128, 82)
        # Keep input tokens away from pad/bos/eos/ecg special ids
        self.input_ids = torch.randint(5, 256, (self.batch_size, self.seq_len))
        self.attention_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)
        self.labels = self.input_ids.clone()

    def tearDown(self):
        self.base_patcher.stop()

    def _llm_decoder_layers(self):
        """Access the raw Llama decoder layers through the PEFT wrapper."""
        return self.decoder.llm_model.base_model.model.model.layers

    def test_forward_shapes_and_loss(self):
        out = self.decoder(
            quantized_features=self.quantized,
            input_ids=self.input_ids,
            labels=self.labels,
            attention_mask=self.attention_mask,
        )
        self.assertTrue(torch.is_tensor(out.loss))
        self.assertEqual(out.loss.dim(), 0)
        self.assertTrue(out.loss.requires_grad)
        # +1 for the prepended <ecg> token; vocab grew by 1 too
        self.assertEqual(out.logits.shape, (self.batch_size, self.seq_len + 1, 256 + 1))

    def test_lora_applied_to_q_and_v_only(self):
        layers = self._llm_decoder_layers()
        self.assertGreater(len(layers), 0)
        for i, layer in enumerate(layers):
            attn = layer.self_attn
            for proj_name in ("q_proj", "v_proj"):
                proj = getattr(attn, proj_name)
                self.assertTrue(
                    hasattr(proj, "lora_A"),
                    f"layer {i} {proj_name} missing LoRA A",
                )
                self.assertTrue(
                    hasattr(proj, "lora_B"),
                    f"layer {i} {proj_name} missing LoRA B",
                )
            for proj_name in ("k_proj", "o_proj"):
                proj = getattr(attn, proj_name)
                self.assertFalse(
                    hasattr(proj, "lora_A") and len(proj.lora_A) > 0,
                    f"layer {i} {proj_name} unexpectedly has LoRA adapters",
                )
            mlp = layer.mlp
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(mlp, proj_name)
                self.assertFalse(
                    hasattr(proj, "lora_A") and len(proj.lora_A) > 0,
                    f"layer {i} MLP {proj_name} unexpectedly has LoRA adapters",
                )

    def test_trainable_params_are_lora_adapter_and_ecg_row(self):
        trainable_names = {
            name for name, p in self.decoder.named_parameters() if p.requires_grad
        }
        # Adapter is fully trainable
        adapter_count = sum(1 for n in trainable_names if n.startswith("adapter."))
        self.assertGreater(adapter_count, 0)
        # LoRA params (live under llm_model.base_model...)
        lora_count = sum(1 for n in trainable_names if "lora_" in n)
        self.assertGreater(lora_count, 0)
        # Embedding table is marked trainable (we mask the gradient to <ecg> row only)
        embed_names = [n for n in trainable_names if n.endswith("embed_tokens.weight")]
        self.assertEqual(len(embed_names), 1)

    def test_only_ecg_row_receives_gradient(self):
        out = self.decoder(
            quantized_features=self.quantized,
            input_ids=self.input_ids,
            labels=self.labels,
            attention_mask=self.attention_mask,
        )
        out.loss.backward()
        embed_weight = self.decoder.llm_model.get_input_embeddings().weight
        self.assertIsNotNone(embed_weight.grad)
        ecg_id = self.decoder.ecg_token_id
        mask = torch.zeros(embed_weight.shape[0], dtype=torch.bool)
        mask[ecg_id] = True
        non_ecg_grad_norm = embed_weight.grad[~mask].abs().sum().item()
        self.assertEqual(non_ecg_grad_norm, 0.0)

    def test_base_llama_params_are_frozen(self):
        # Everything inside llm_model that is not LoRA or the embedding table
        # should be frozen.
        for name, p in self.decoder.llm_model.named_parameters():
            if "lora_" in name:
                self.assertTrue(p.requires_grad, f"LoRA param {name} not trainable")
            elif name.endswith("embed_tokens.weight"):
                self.assertTrue(p.requires_grad, f"embed_tokens not trainable (needed for <ecg> row)")
            else:
                self.assertFalse(p.requires_grad, f"base param {name} should be frozen")

    def test_generate_report_shape_greedy(self):
        gen = self.decoder.generate_report(
            quantized_features=self.quantized,
            max_token_length=6,
            do_sample=False,
            eos_token_id=9999,  # unreachable so we generate the full budget
        )
        self.assertEqual(gen.shape[0], self.batch_size)
        self.assertLessEqual(gen.shape[1], 6)
        self.assertEqual(gen.shape[1], 6)  # no early stop with unreachable EOS
        self.assertEqual(gen.dtype, torch.long)

    def test_generate_report_stops_on_eos(self):
        fixed_eos = 42

        def _always_eos(logits, do_sample, temperature, top_p, top_k):
            return torch.full((logits.size(0),), fixed_eos, dtype=torch.long, device=logits.device)

        with patch.object(Llama32ReportDecoder, "_sample_next", staticmethod(_always_eos)):
            gen = self.decoder.generate_report(
                quantized_features=self.quantized,
                max_token_length=8,
                do_sample=False,
                eos_token_id=fixed_eos,
            )

        # First sampled token is EOS so we stop after one token
        self.assertEqual(gen.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(gen == fixed_eos))

    def test_generate_report_uses_kv_cache(self):
        forward_calls = []
        original_call = self.decoder.llm_model.__class__.__call__

        def spy_call(self_model, *args, **kwargs):
            forward_calls.append({
                "has_past": kwargs.get("past_key_values") is not None,
                "has_inputs_embeds": kwargs.get("inputs_embeds") is not None,
                "input_ids_len": (
                    kwargs["input_ids"].shape[1] if kwargs.get("input_ids") is not None else None
                ),
            })
            return original_call(self_model, *args, **kwargs)

        with patch.object(self.decoder.llm_model.__class__, "__call__", spy_call):
            self.decoder.generate_report(
                quantized_features=self.quantized,
                max_token_length=3,
                do_sample=False,
                eos_token_id=9999,
            )

        # First call: inputs_embeds, no past. Subsequent calls: input_ids of length 1, with past.
        self.assertGreaterEqual(len(forward_calls), 2)
        self.assertTrue(forward_calls[0]["has_inputs_embeds"])
        self.assertFalse(forward_calls[0]["has_past"])
        for call in forward_calls[1:]:
            self.assertTrue(call["has_past"], "subsequent calls should pass past_key_values")
            self.assertEqual(call["input_ids_len"], 1, "subsequent calls should feed only the new token")

    def test_peft_wrapper_embedding_has_vocab_plus_one_rows(self):
        embed = self.decoder.llm_model.get_input_embeddings()
        self.assertEqual(embed.weight.shape[0], 256 + 1)
        self.assertEqual(embed.weight.shape[1], 64)


if __name__ == "__main__":
    unittest.main()
