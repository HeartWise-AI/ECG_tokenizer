import unittest
import torch
from unittest.mock import patch, MagicMock

from models.decoder import GPT2Decoder
from models.bridge import (
    EmbeddingBridge,
    LinearBridge,
    SimpleEmbeddingBridge
)
from utils.registry import ModelRegistry
from utils.enums import BridgeName

class TestModelsIntegration(unittest.TestCase):
    
    def setUp(self):
        # Keep the GPT-2 mock active for the whole test. The previous @patch on
        # setUp expired before the test bodies constructed GPT2Decoder, so they
        # loaded real GPT-2 from HuggingFace (network/cache-dependent).
        gpt2_patcher = patch('models.decoder.gpt2_decoder.GPT2LMHeadModel')
        mock_gpt2 = gpt2_patcher.start()
        self.addCleanup(gpt2_patcher.stop)

        self.batch_size = 4
        self.mock_gpt2_instance = mock_gpt2.from_pretrained.return_value
        self.mock_gpt2_instance.config.n_embd = 768
        self.mock_gpt2_instance.config.eos_token_id = 50256
        self.mock_gpt2_instance.get_input_embeddings().weight.size = lambda: torch.Size([50257, 768])
        self.mock_gpt2_instance.get_input_embeddings().return_value = torch.randn(self.batch_size, 11, 768)

        # Forward output carries real tensors so both the forward tests and the
        # manual generation loop run without touching real HuggingFace weights.
        fwd = MagicMock()
        fwd.loss = torch.tensor(0.5)
        fwd.logits = torch.randn(self.batch_size, 11, 50257)
        fwd.past_key_values = None
        self.mock_gpt2_instance.return_value = fwd

        # Set up common test variables
        self.quantized_features = torch.randn(self.batch_size, 8, 128, 160)
        self.input_ids = torch.randint(0, 50256, (self.batch_size, 10))
        self.attention_mask = torch.ones(self.batch_size, 10)
        
    def test_integration_with_embedding_bridge(self):
        """Test integration of GPT2WithEmbedding with EmbeddingBridge"""
        # Register EmbeddingBridge
        bridge_name = BridgeName.GPT2_EMBEDDING_BRIDGE

        # Create GPT2WithEmbedding model with EmbeddingBridge
        with patch.object(ModelRegistry, 'get', return_value=EmbeddingBridge):
            model = GPT2Decoder(
                huggingface_model_name='gpt2',
                llm_input_embedding_size=768,
                quantized_feature_shape=(8, 128, 160),
                bridge_name=bridge_name,
                adapter_dropout=0.2
            )
            
            # Check model initialized correctly
            self.assertIsInstance(model.embedding_adapter, EmbeddingBridge)
            
            # Test forward pass
            outputs = model(
                quantized_features=self.quantized_features,
                input_ids=self.input_ids,
                attention_mask=self.attention_mask
            )
            
            # Check outputs
            self.assertIsNotNone(outputs)
    
    def test_integration_with_linear_bridge(self):
        """Test integration of GPT2WithEmbedding with LinearBridge"""
        # Register LinearBridge
        bridge_name = BridgeName.GPT2_LINEAR_BRIDGE

        # Create GPT2WithEmbedding model with LinearBridge
        with patch.object(ModelRegistry, 'get', return_value=LinearBridge):
            model = GPT2Decoder(
                huggingface_model_name='gpt2',
                llm_input_embedding_size=768,
                quantized_feature_shape=(8, 128, 160),
                bridge_name=bridge_name,
                adapter_dropout=0.2
            )
            
            # Check model initialized correctly
            self.assertIsInstance(model.embedding_adapter, LinearBridge)
            
            # Test forward pass
            outputs = model(
                quantized_features=self.quantized_features,
                input_ids=self.input_ids,
                attention_mask=self.attention_mask
            )
            
            # Check outputs
            self.assertIsNotNone(outputs)
    
    def test_integration_with_simple_embedding_bridge(self):
        """Test integration of GPT2WithEmbedding with SimpleEmbeddingBridge"""
        # Register SimpleEmbeddingBridge
        bridge_name = BridgeName.GPT2_SIMPLE_EMBEDDING_BRIDGE

        # Create GPT2WithEmbedding model with SimpleEmbeddingBridge
        with patch.object(ModelRegistry, 'get', return_value=SimpleEmbeddingBridge):
            model = GPT2Decoder(
                huggingface_model_name='gpt2',
                llm_input_embedding_size=768,
                quantized_feature_shape=(8, 128, 160),
                bridge_name=bridge_name,
                adapter_dropout=0.2
            )
            
            # Check model initialized correctly
            self.assertIsInstance(model.embedding_adapter, SimpleEmbeddingBridge)
            
            # Test forward pass
            outputs = model(
                quantized_features=self.quantized_features,
                input_ids=self.input_ids,
                attention_mask=self.attention_mask
            )
            
            # Check outputs
            self.assertIsNotNone(outputs)
    
    def test_report_generation_with_different_bridges(self):
        """Report generation runs a manual autoregressive loop (not `.generate`);
        assert output shape/dtype per bridge."""
        bridges = [EmbeddingBridge, LinearBridge, SimpleEmbeddingBridge]
        bridge_names = [BridgeName.GPT2_EMBEDDING_BRIDGE, BridgeName.GPT2_LINEAR_BRIDGE, BridgeName.GPT2_SIMPLE_EMBEDDING_BRIDGE]
        max_token_length = 20

        for bridge_cls, bridge_name in zip(bridges, bridge_names):
            with self.subTest(bridge=bridge_name):
                with patch.object(ModelRegistry, 'get', return_value=bridge_cls):
                    model = GPT2Decoder(
                        huggingface_model_name='gpt2',
                        llm_input_embedding_size=768,
                        quantized_feature_shape=(8, 128, 160),
                        bridge_name=bridge_name,
                        adapter_dropout=0.2
                    )

                    # Unreachable EOS -> full budget, so every bridge must drive
                    # multi-token generation to the exact length (a bridge that
                    # degenerates to an early stop would be caught).
                    generated = model.generate_report(
                        quantized_features=self.quantized_features,
                        max_token_length=max_token_length,
                        eos_token_id=999999,
                    )

                    self.assertEqual(generated.shape, (self.batch_size, max_token_length))
                    self.assertEqual(generated.dtype, torch.long)

if __name__ == '__main__':
    unittest.main()
