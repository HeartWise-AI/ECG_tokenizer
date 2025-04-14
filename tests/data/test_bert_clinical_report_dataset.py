import os
import pytest
import pandas as pd
import torch
from unittest.mock import patch, MagicMock

from data.bert_clinical_report_dataset import (
    BertClinicalReportDataset,
    get_clinical_report_dataloader,
    get_distributed_clinical_report_dataloader
)


@pytest.fixture
def mock_bert_tokenizer():
    """Create a mock BERT tokenizer for testing."""
    mock = MagicMock()
    mock.return_value = {
        'input_ids': torch.tensor([[101, 2054, 2003, 1037, 3231, 102]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1]])
    }
    return mock


@pytest.fixture
def mock_reports_csv():
    """Create a mock CSV file with predicted and reference reports."""
    return pd.DataFrame({
        'predicted_report': [
            'This is a predicted report 1',
            'This is a predicted report 2',
            'This is a predicted report 3'
        ],
        'reference_report': [
            'This is a reference report 1',
            'This is a reference report 2',
            'This is a reference report 3'
        ]
    })


class TestBertClinicalReportDataset:
    
    @patch('pandas.read_csv')
    def test_init(self, mock_read_csv, mock_reports_csv, mock_bert_tokenizer):
        """Test dataset initialization."""
        mock_read_csv.return_value = mock_reports_csv
        
        dataset = BertClinicalReportDataset(
            df_path="reports.csv",
            tokenizer=mock_bert_tokenizer
        )
        
        mock_read_csv.assert_called_once_with("reports.csv")
        assert len(dataset.predicted_report) == 3
        assert len(dataset.reference_report) == 3
        assert dataset.tokenizer == mock_bert_tokenizer
        
    def test_len(self, mock_reports_csv):
        """Test the __len__ method."""
        with patch('pandas.read_csv', return_value=mock_reports_csv):
            dataset = BertClinicalReportDataset(
                df_path="reports.csv",
                tokenizer=MagicMock()
            )
            assert len(dataset) == 3
    
    def test_getitem(self, mock_reports_csv, mock_bert_tokenizer):
        """Test the __getitem__ method."""
        with patch('pandas.read_csv', return_value=mock_reports_csv):
            dataset = BertClinicalReportDataset(
                df_path="reports.csv",
                tokenizer=mock_bert_tokenizer
            )
            
            result = dataset[0]
            
            # Verify that the result contains expected keys
            assert 'encoded_predicted_report' in result
            assert 'encoded_reference_report' in result
            
            # Verify that the tokenizer was called with the correct reports
            mock_bert_tokenizer.assert_any_call(
                'This is a predicted report 1',
                padding='max_length',
                max_length=512,
                truncation=True,
                return_tensors='pt'
            )
            
            mock_bert_tokenizer.assert_any_call(
                'This is a reference report 1',
                padding='max_length',
                max_length=512,
                truncation=True,
                return_tensors='pt'
            )
    
    @patch('data.bert_clinical_report_dataset.DataLoader')
    def test_get_clinical_report_dataloader(self, mock_dataloader, mock_bert_tokenizer, mock_reports_csv):
        """Test the get_clinical_report_dataloader function."""
        mock_config = MagicMock()
        mock_config.predicted_reports_path = "reports.csv"
        mock_config.batch_size = 32
        
        # Mock return value for DataLoader
        mock_dataloader_instance = MagicMock()
        mock_dataloader.return_value = mock_dataloader_instance
        
        # Properly mock pandas.read_csv to return our mock dataset
        with patch('pandas.read_csv', return_value=mock_reports_csv):
            result = get_clinical_report_dataloader(
                config=mock_config,
                tokenizer=mock_bert_tokenizer,
                shuffle=True,
                pin_memory=True
            )
            
            # Verify that DataLoader was called with correct parameters
            mock_dataloader.assert_called_once()
            
            # Get the keyword arguments used to call DataLoader
            _, kwargs = mock_dataloader.call_args
            
            # Check the parameters
            assert kwargs['batch_size'] == 32
            assert kwargs['shuffle'] is True
            assert kwargs['pin_memory'] is True
            
            # Verify that the result is the mock dataloader instance
            assert result == mock_dataloader_instance
    
    @patch('utils.ddp.DistributedUtils.get_distributed_dataloader')
    def test_get_distributed_clinical_report_dataloader(self, mock_get_dataloader, mock_bert_tokenizer, mock_reports_csv):
        """Test the get_distributed_clinical_report_dataloader function."""
        # Configure mock_get_dataloader to return a proper value
        mock_dataloader = MagicMock()
        mock_get_dataloader.return_value = mock_dataloader
        
        # Mock pandas.read_csv to return our mock dataset
        with patch('pandas.read_csv', return_value=mock_reports_csv):
            result = get_distributed_clinical_report_dataloader(
                predicted_reports_path="reports.csv",
                tokenizer=mock_bert_tokenizer,
                batch_size=32,
                num_workers=4,
                num_replicas=2,
                rank=0,
                shuffle=True,
                pin_memory=True
            )
            
            # Verify the dataloader was created with correct parameters
            mock_get_dataloader.assert_called_once()
            
            # Get the keyword arguments used to call get_distributed_dataloader
            kwargs = mock_get_dataloader.call_args.kwargs
            
            # Check the parameters using kwargs dictionary directly
            assert kwargs['batch_size'] == 32
            assert kwargs['num_workers'] == 4
            assert kwargs['num_replicas'] == 2
            assert kwargs['rank'] == 0
            assert kwargs['shuffle'] is True
            assert kwargs['pin_memory'] is True
            
            # Check that dataset is an instance of BertClinicalReportDataset
            assert isinstance(kwargs['dataset'], BertClinicalReportDataset)
            
            # Verify the result is the mock dataloader
            assert result == mock_dataloader 