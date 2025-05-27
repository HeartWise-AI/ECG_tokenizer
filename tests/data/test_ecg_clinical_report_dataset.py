import os
import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from data.ecg_clinical_report_dataset import (
    ECGClinicalReportDataset,
    custom_collate_fn,
    get_clinical_report_dataloader,
    get_distributed_clinical_report_dataloader
)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    mock = MagicMock()
    mock.encode_plus.return_value = {
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
    }
    return mock


@pytest.fixture
def mock_reports_df():
    """Create a mock dataframe with clinical reports."""
    return pd.DataFrame({
        'waveform_path': ['path/to/waveform1.npy', 'path/to/waveform2.npy', 'path/to/waveform3.npy'],
        'report': ['Sample report 1', 'Sample report 2', 'Sample report 3']
    })


@pytest.fixture
def mock_embedding():
    """Create a mock ECG embedding for testing."""
    # Create an embedding with shape (8, 128, 160)
    return np.random.randn(8, 128, 160).astype(np.float32)


class TestECGClinicalReportDataset:
    
    @patch('pandas.read_parquet')
    def test_init(self, mock_read_parquet, mock_reports_df, mock_tokenizer):
        """Test dataset initialization."""
        mock_read_parquet.return_value = mock_reports_df
        
        dataset = ECGClinicalReportDataset(
            embeddings_path="/embeddings",
            reports_path="reports.parquet",
            tokenizer=mock_tokenizer,
            max_length=512
        )
        
        mock_read_parquet.assert_called_once_with("reports.parquet")
        assert dataset.embeddings_path == "/embeddings"
        assert dataset.tokenizer == mock_tokenizer
        assert dataset.max_length == 512
        assert len(dataset.df) == 3
        
    def test_len(self, mock_reports_df):
        """Test the __len__ method."""
        with patch('pandas.read_parquet', return_value=mock_reports_df):
            dataset = ECGClinicalReportDataset(
                embeddings_path="/embeddings",
                reports_path="reports.parquet",
                tokenizer=MagicMock(),
                max_length=512
            )
            assert len(dataset) == 3
    
    @patch('os.path.join')
    @patch('numpy.load')
    def test_getitem(self, mock_load, mock_path_join, mock_reports_df, mock_tokenizer, mock_embedding):
        """Test the __getitem__ method."""
        mock_path_join.return_value = "/embeddings/waveform1_embedding.npy"
        mock_load.return_value = mock_embedding
        
        with patch('pandas.read_parquet', return_value=mock_reports_df):
            dataset = ECGClinicalReportDataset(
                embeddings_path="/embeddings",
                reports_path="reports.parquet",
                tokenizer=mock_tokenizer,
                max_length=512
            )
            
            result = dataset[0]
            
            # Verify that the result contains expected keys
            assert 'embedding' in result
            assert 'input_ids' in result
            assert 'attention_mask' in result
            assert 'waveform_name' in result
            
            # Verify that the embedding has the correct shape
            assert result['embedding'].shape == (8, 128, 160)
            
            # Verify that the tokenizer was called with the correct report
            mock_tokenizer.encode_plus.assert_called_once_with(
                'Sample report 1',
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Verify the waveform name extraction logic
            assert result['waveform_name'] == 'waveform1'
    
    @patch('os.path.join')
    def test_getitem_with_missing_embedding(self, mock_path_join, mock_reports_df, mock_tokenizer):
        """Test the __getitem__ method when embedding file is missing."""
        mock_path_join.return_value = "/embeddings/nonexistent_embedding.npy"
        
        with patch('pandas.read_parquet', return_value=mock_reports_df):
            dataset = ECGClinicalReportDataset(
                embeddings_path="/embeddings",
                reports_path="reports.parquet",
                tokenizer=mock_tokenizer,
                max_length=512
            )
            
            # Should return None when embedding file is not found
            with patch('numpy.load', side_effect=FileNotFoundError):
                result = dataset[0]
                assert result is None
    
    def test_custom_collate_fn(self):
        """Test the custom_collate_fn function."""
        # Create a batch with some valid items and some None items
        batch = [
            {
                'embedding': torch.tensor(np.random.randn(8, 128, 160)),
                'input_ids': torch.tensor([1, 2, 3, 4, 5]),
                'attention_mask': torch.tensor([1, 1, 1, 1, 1]),
                'waveform_name': 'waveform1'
            },
            None,  # Invalid item
            {
                'embedding': torch.tensor(np.random.randn(8, 128, 160)),
                'input_ids': torch.tensor([6, 7, 8, 9, 10]),
                'attention_mask': torch.tensor([1, 1, 1, 1, 1]),
                'waveform_name': 'waveform3'
            }
        ]
        
        result = custom_collate_fn(batch)
        
        # Should filter out None items
        assert len(result) == 4  # 4 keys in the dictionary
        assert len(result['embedding']) == 2  # 2 valid items
        assert len(result['input_ids']) == 2
        assert len(result['attention_mask']) == 2
        assert len(result['waveform_name']) == 2
    
    def test_custom_collate_fn_all_none(self):
        """Test the custom_collate_fn function with all None items."""
        batch = [None, None, None]
        
        # Should raise ValueError when all items are None
        with pytest.raises(ValueError):
            custom_collate_fn(batch)
    
    @patch('data.ecg_clinical_report_dataset.DataLoader')
    def test_get_clinical_report_dataloader(self, mock_dataloader, mock_tokenizer):
        """Test the get_clinical_report_dataloader function."""
        mock_config = MagicMock()
        mock_config.embeddings_path = "/embeddings"
        mock_config.reports_path = "reports.parquet"
        mock_config.batch_size = 32
        mock_config.num_workers = 4
        mock_config.max_length = 512
        mock_config.tokenizer = mock_tokenizer
        
        # Create a mock dataset with a non-zero length
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        
        with patch('pandas.read_parquet'), patch('data.ecg_clinical_report_dataset.ECGClinicalReportDataset', return_value=mock_dataset):
            get_clinical_report_dataloader(
                config=mock_config,
                shuffle=True,
                pin_memory=True
            )
            
            # Verify that DataLoader was called with correct parameters
            mock_dataloader.assert_called_once()
            
            # Check the DataLoader parameters
            args, kwargs = mock_dataloader.call_args
            assert args[0] == mock_dataset
            
            # Check the other parameters
            assert kwargs['batch_size'] == 32
            assert kwargs['shuffle'] is True
            assert kwargs['pin_memory'] is True
            assert kwargs['collate_fn'] == custom_collate_fn
    
    @patch('utils.ddp.DistributedUtils.get_distributed_dataloader')
    def test_get_distributed_clinical_report_dataloader(self, mock_get_dataloader, mock_tokenizer):
        """Test the get_distributed_clinical_report_dataloader function."""
        # Create a mock dataset with a non-zero length
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        
        with patch('pandas.read_parquet'), patch('data.ecg_clinical_report_dataset.ECGClinicalReportDataset', return_value=mock_dataset):
            get_distributed_clinical_report_dataloader(
                reports_path="reports.parquet",
                embeddings_path="/embeddings",
                tokenizer=mock_tokenizer,
                max_token_length=512,
                batch_size=32,
                num_workers=4,
                num_replicas=2,
                rank=0,
                shuffle=True,
                pin_memory=True
            )
            
            # Verify the dataloader was created with correct parameters
            mock_get_dataloader.assert_called_once()
            
            # Check that dataset was passed correctly using kwargs
            _, kwargs = mock_get_dataloader.call_args
            assert kwargs['dataset'] == mock_dataset
            
            # Check the other parameters
            assert kwargs['batch_size'] == 32
            assert kwargs['num_workers'] == 4
            assert kwargs['num_replicas'] == 2
            assert kwargs['rank'] == 0
            assert kwargs['shuffle'] is True
            assert kwargs['pin_memory'] is True
            assert kwargs['collate_fn'] == custom_collate_fn 