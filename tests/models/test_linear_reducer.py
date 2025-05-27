import unittest
import torch

from models.linear_reducer import LinearReducer

class TestLinearReducer(unittest.TestCase):
    
    def setUp(self):
        self.input_shape = (8, 128, 160)
        self.output_size = 768
        self.dropout = 0.2
        self.model = LinearReducer(
            input_shape=self.input_shape,
            output_size=self.output_size,
            dropout=self.dropout
        )
        self.batch_size = 4
    
    def test_init(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, LinearReducer)
        
        # Test that flatten dim is correctly calculated
        expected_flatten_dim = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        self.assertEqual(self.model.flatten_dim, expected_flatten_dim)
        
        # Test that layers are correctly initialized
        self.assertIsNotNone(self.model.reducer)
        self.assertEqual(len(self.model.reducer), 5)  # Flatten, Linear, ReLU, Dropout, Linear
        
        # Check first linear layer
        linear1 = self.model.reducer[1]
        self.assertIsInstance(linear1, torch.nn.Linear)
        self.assertEqual(linear1.in_features, expected_flatten_dim)
        self.assertEqual(linear1.out_features, 1024)
        
        # Check dropout layer
        dropout = self.model.reducer[3]
        self.assertIsInstance(dropout, torch.nn.Dropout)
        self.assertEqual(dropout.p, self.dropout)
        
        # Check last linear layer
        linear2 = self.model.reducer[4]
        self.assertIsInstance(linear2, torch.nn.Linear)
        self.assertEqual(linear2.in_features, 1024)
        self.assertEqual(linear2.out_features, self.output_size)
    
    def test_init_no_dropout(self):
        """Test initialization with no dropout"""
        model = LinearReducer(
            input_shape=self.input_shape,
            output_size=self.output_size,
            dropout=0.0
        )
        
        # Check that dropout layer is replaced with Identity
        dropout_or_identity = model.reducer[3]
        self.assertIsInstance(dropout_or_identity, torch.nn.Identity)
    
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