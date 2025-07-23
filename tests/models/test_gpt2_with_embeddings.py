import unittest
import torch
from unittest.mock import patch, MagicMock

from models.gpt2_tokenizer_decoder import GPT2Decoder
from utils.registry import ModelRegistry
from utils.enums import AdapterName

class TestGPT2WithEmbedding(unittest.TestCase):
    
    @patch('models.gpt2_tokenizer_decoder.GPT2LMHeadModel')
    @patch('models.gpt2_tokenizer_decoder.ModelRegistry')
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
            ecg_embedding_size=(8, 128, 82),
            adapter_name=AdapterName.GPT2_EMBEDDING_ADAPTER,
            adapter_dropout=0.2
        )
        
        # Set up common test variables
        self.batch_size = 4
        self.ecg_embeddings = torch.randn(self.batch_size, 8, 128, 82)
        self.input_ids = torch.randint(0, 50256, (self.batch_size, 10))
        self.attention_mask = torch.ones(self.batch_size, 10)
        self.labels = torch.randint(0, 50256, (self.batch_size, 10))
        
        # Configure mock returns
        self.mock_adapter.return_value = torch.randn(self.batch_size, 768)
        self.mock_gpt2_instance.get_input_embeddings().return_value = torch.randn(self.batch_size, 11, 768)
        
    def test_init(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, GPT2WithEmbedding)
        self.assertEqual(self.model.gpt2, self.mock_gpt2_instance)
        self.assertEqual(self.model.embedding_adapter, self.mock_adapter)
        
    def test_forward(self):
        """Test forward pass"""
        # Configure mock return for GPT-2 forward
        expected_output = MagicMock()
        expected_output.loss = torch.tensor(0.5)
        self.mock_gpt2_instance.return_value = expected_output
        
        # Run forward pass
        outputs = self.model(
            ecg_embeddings=self.ecg_embeddings,
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels
        )
        
        # Assertions
        self.mock_adapter.assert_called_once_with(self.ecg_embeddings)
        self.assertIsNotNone(outputs)
        self.mock_gpt2_instance.assert_called_once()
        
    def test_forward_without_attention_mask(self):
        """Test forward pass without providing attention mask"""
        # Configure mock
        expected_output = MagicMock()
        self.mock_gpt2_instance.return_value = expected_output
        
        # Run forward without attention mask
        outputs = self.model(
            ecg_embeddings=self.ecg_embeddings,
            input_ids=self.input_ids,
            labels=self.labels
        )
        
        # Assertions
        self.assertIsNotNone(outputs)
        self.mock_gpt2_instance.assert_called_once()
        
    def test_generate_report(self):
        """Test report generation"""
        # Configure mocks
        expected_output = torch.randint(0, 50256, (self.batch_size, 19))
        self.mock_gpt2_instance.generate.return_value = expected_output
        
        # Run generate_report
        generated = self.model.generate_report(
            ecg_embeddings=self.ecg_embeddings,
            max_token_length=20
        )
        
        # Assertions
        self.mock_adapter.assert_called_with(self.ecg_embeddings)
        self.mock_gpt2_instance.generate.assert_called_once()
        self.assertTrue(torch.equal(generated, expected_output))
        
if __name__ == '__main__':
    unittest.main() 