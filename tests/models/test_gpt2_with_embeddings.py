import unittest
import torch
from unittest.mock import patch, MagicMock

from models.decoder import GPT2Decoder
from utils.registry import ModelRegistry
from utils.enums import BridgeName

class TestGPT2WithEmbedding(unittest.TestCase):
    
    @patch('models.decoder.gpt2_decoder.GPT2LMHeadModel')
    @patch('models.decoder.gpt2_decoder.ModelRegistry')
    def setUp(self, mock_registry, mock_gpt2):
        # Mock the GPT2 model and embedding adapter
        self.mock_gpt2_instance = MagicMock()
        self.mock_gpt2_instance.config.n_embd = 768
        self.mock_gpt2_instance.config.eos_token_id = 50256
        self.mock_gpt2_instance.get_input_embeddings().weight.size = lambda: torch.Size([50257, 768])
        mock_gpt2.from_pretrained.return_value = self.mock_gpt2_instance
        
        # Mock the embedding adapter
        self.mock_adapter = MagicMock()
        mock_registry.get.return_value.return_value = self.mock_adapter
        
        # Create model instance
        self.model = GPT2Decoder(
            huggingface_model_name='gpt2',
            llm_input_embedding_size=768,
            quantized_feature_shape=(8, 128, 160),
            bridge_name=BridgeName.GPT2_EMBEDDING_BRIDGE,
            adapter_dropout=0.2
        )
        
        # Set up common test variables
        self.batch_size = 4
        self.quantized_features = torch.randn(self.batch_size, 8, 128, 160)
        self.input_ids = torch.randint(0, 50256, (self.batch_size, 10))
        self.attention_mask = torch.ones(self.batch_size, 10)
        self.labels = torch.randint(0, 50256, (self.batch_size, 10))
        
        # Configure mock returns
        self.mock_adapter.return_value = torch.randn(self.batch_size, 768)
        self.mock_gpt2_instance.get_input_embeddings().return_value = torch.randn(self.batch_size, 11, 768)
        
    def test_init(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, GPT2Decoder)
        self.assertEqual(self.model.huggingface_model_name, 'gpt2')
        self.assertEqual(self.model.llm_input_embedding_size, 768)
        self.assertEqual(self.model.quantized_feature_shape, (8, 128, 160))
        self.assertEqual(self.model.bridge_name, BridgeName.GPT2_EMBEDDING_BRIDGE)
        self.assertEqual(self.model.adapter_dropout, 0.2)
        
    def test_forward(self):
        """Test forward pass"""
        # Configure mock return for GPT-2 forward
        expected_output = MagicMock()
        expected_output.loss = torch.tensor(0.5)
        self.mock_gpt2_instance.return_value = expected_output
        
        # Run forward pass
        outputs = self.model(
            quantized_features=self.quantized_features,
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels
        )
        
        # Assertions
        self.mock_adapter.assert_called_once_with(self.quantized_features)
        self.assertIsNotNone(outputs)
        self.mock_gpt2_instance.assert_called_once()
        
    def test_forward_without_attention_mask(self):
        """Test forward pass without providing attention mask"""
        # Configure mock
        expected_output = MagicMock()
        self.mock_gpt2_instance.return_value = expected_output
        
        # Run forward without attention mask
        outputs = self.model(
            quantized_features=self.quantized_features,
            input_ids=self.input_ids,
            labels=self.labels
        )
        
        # Assertions
        self.assertIsNotNone(outputs)
        self.mock_gpt2_instance.assert_called_once()
        
    def test_generate_report(self):
        """Report generation runs a manual autoregressive loop (forward +
        _sample_next); `.generate()` is intentionally not used, so the mocked
        forward must return real logit tensors."""
        fwd = MagicMock()
        fwd.logits = torch.randn(self.batch_size, 11, 50257)
        fwd.past_key_values = None
        self.mock_gpt2_instance.return_value = fwd
        # wte(ecg_token) must be a real tensor for the in-place ECG embedding write.
        self.mock_gpt2_instance.get_input_embeddings().return_value = torch.randn(self.batch_size, 1, 768)

        # Unreachable EOS -> loop runs the full budget, so length is exact and a
        # truncation/off-by-one regression in the manual loop would be caught.
        generated = self.model.generate_report(
            quantized_features=self.quantized_features,
            max_token_length=20,
            eos_token_id=999999,
        )

        # Assertions — exact shape/dtype, not a stale `.generate` call.
        self.mock_adapter.assert_called_with(self.quantized_features)
        self.assertEqual(generated.shape, (self.batch_size, 20))
        self.assertEqual(generated.dtype, torch.long)
        
if __name__ == '__main__':
    unittest.main() 
