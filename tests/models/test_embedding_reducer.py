import unittest
import torch

from models.embedding_reducer import EmbeddingReducer

class TestEmbeddingReducer(unittest.TestCase):
    
    def setUp(self):
        self.input_shape = (8, 128, 160)
        self.output_size = 768
        self.dropout = 0.2
        self.model = EmbeddingReducer(
            input_shape=self.input_shape,
            output_size=self.output_size,
            dropout=self.dropout
        )
        self.batch_size = 4
    
    def test_init(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, EmbeddingReducer)
        self.assertEqual(self.model.input_shape, self.input_shape)
        self.assertEqual(self.model.output_size, self.output_size)
        
        # Test that layers are correctly initialized
        self.assertEqual(len(self.model.conv_layers), 9)  # 3 conv blocks with 3 layers each
        self.assertIsNotNone(self.model.avgpool)
        self.assertIsNotNone(self.model.flatten)
        self.assertEqual(len(self.model.fc_layers), 4)  # 2 linear layers + ReLU + dropout
        
        # Check the first conv layer
        first_conv = self.model.conv_layers[0]
        self.assertEqual(first_conv.in_channels, self.input_shape[0])
        self.assertEqual(first_conv.out_channels, 32)
        
        # Check the last linear layer
        last_linear = self.model.fc_layers[3]
        self.assertEqual(last_linear.out_features, self.output_size)
    
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
    
    def test_forward_normalization(self):
        """Test that normalization is applied in forward pass"""
        # Create input with known values
        x = torch.ones(self.batch_size, *self.input_shape) * 1024.0
        
        # Get the first layer of conv_layers to check input
        with torch.no_grad():
            # We'll monkey patch the first layer to capture its input
            original_forward = self.model.conv_layers[0].forward
            
            # Capture normalized input
            normalized_input = None
            def capture_input(input_tensor):
                nonlocal normalized_input
                normalized_input = input_tensor.clone()
                return original_forward(input_tensor)
            
            self.model.conv_layers[0].forward = capture_input
            
            # Run forward pass
            self.model(x)
            
            # Check normalization
            self.assertIsNotNone(normalized_input)
            self.assertTrue(torch.allclose(normalized_input, torch.ones_like(normalized_input)))
            
            # Restore original forward
            self.model.conv_layers[0].forward = original_forward

if __name__ == '__main__':
    unittest.main() 