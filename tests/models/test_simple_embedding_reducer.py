import unittest
import torch

from models.simple_embedding_reducer import SimpleEmbeddingReducer

class TestSimpleEmbeddingReducer(unittest.TestCase):
    
    def setUp(self):
        self.input_shape = (8, 128, 160)
        self.output_size = 768
        self.dropout = 0.2
        self.model = SimpleEmbeddingReducer(
            input_shape=self.input_shape,
            output_size=self.output_size,
            dropout=self.dropout
        )
        self.batch_size = 4
    
    def test_init(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, SimpleEmbeddingReducer)
        
        # Test components are correctly initialized
        self.assertIsInstance(self.model.global_pool, torch.nn.AdaptiveAvgPool2d)
        self.assertIsInstance(self.model.fc, torch.nn.Linear)
        self.assertIsInstance(self.model.dropout, torch.nn.Dropout)
        
        # Check linear layer dimensions
        self.assertEqual(self.model.fc.in_features, self.input_shape[0])
        self.assertEqual(self.model.fc.out_features, self.output_size)
        
        # Check dropout value
        self.assertEqual(self.model.dropout.p, self.dropout)
    
    def test_forward(self):
        """Test forward pass with input tensor"""
        # Create input tensor
        x = torch.randn(self.batch_size, *self.input_shape)
        
        # Pass through model
        output = self.model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_size))
        
        # Check output values are reasonable (not NaN or inf)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_global_pooling(self):
        """Test that global pooling works correctly"""
        # Create a tensor with known values
        x = torch.ones(self.batch_size, *self.input_shape)
        
        # Forward through global_pool only
        pooled = self.model.global_pool(x)
        
        # Check shape after pooling (should be [batch_size, channels, 1, 1])
        expected_shape = (self.batch_size, self.input_shape[0], 1, 1)
        self.assertEqual(pooled.shape, expected_shape)
        
        # Check values (should still be ones since we're averaging ones)
        self.assertTrue(torch.allclose(pooled, torch.ones(expected_shape)))
    
    def test_forward_different_batch_size(self):
        """Test forward pass with different batch sizes"""
        batch_sizes = [1, 8, 16]
        
        for bs in batch_sizes:
            with self.subTest(batch_size=bs):
                x = torch.randn(bs, *self.input_shape)
                output = self.model(x)
                self.assertEqual(output.shape, (bs, self.output_size))

if __name__ == '__main__':
    unittest.main() 