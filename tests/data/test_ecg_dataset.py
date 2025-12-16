import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

from data.ecg_dataset import ECGDataset, get_distributed_ecg_dataloader
from utils.constants import lead_to_idx


@pytest.fixture
def mock_parquet_data():
    """Create a mock parquet dataframe for testing."""
    return pd.DataFrame({
        'waveform_path': ['test_path_1.npy', 'test_path_2.npy', 'test_path_3.npy']
    })


@pytest.fixture
def mock_lead_stats():
    """Create mock lead statistics for normalization."""
    stats = {}
    for lead_name in lead_to_idx.keys():
        stats[lead_name] = {
            "mean": 0.0,
            "std": 1.0
        }
    return stats


@pytest.fixture
def mock_signal():
    """Create a mock ECG signal for testing."""
    # Create a signal with shape (2500, 12) - typical dimensions for ECG
    return np.random.randn(2500, 12).astype(np.float32)


class TestECGDataset:
    
    @patch('pandas.read_parquet')
    def test_init(self, mock_read_parquet, mock_parquet_data):
        """Test dataset initialization."""
        mock_read_parquet.return_value = mock_parquet_data
        
        lead_stats = {
            "I": {
                "mean": -0.982577,
                "std": 33.551235
            },
            "II": {
                "mean": -0.683182,
                "std": 35.616466
            },
            "III": {
                "mean": 0.299394,
                "std": 40.365378
            },
            "aVR": {
                "mean": 0.832880,
                "std": 28.102812
            },
            "aVL": {
                "mean": -0.640986,
                "std": 32.563651
            },
            "aVF": {
                "mean": -0.191894,
                "std": 34.169092
            },
            "V1": {
                "mean": -1.351383,
                "std": 54.328896
            },
            "V2": {
                "mean": -1.366718,
                "std": 72.522918
            },
            "V3": {
                "mean": -1.325758,
                "std": 66.522440
            },
            "V4": {
                "mean": -1.091183,
                "std": 61.623552
            },
            "V5": {
                "mean": -0.937534,
                "std": 57.341500
            },
            "V6": {
                "mean": -0.882349,
                "std": 52.247728
            }
        }
        
        dataset = ECGDataset(
            parquet_file="dummy.parquet",
            expected_waveform_length=2500,
            num_leads=12,
            normalize_waveforms=True,
            lead_stats=lead_stats
        )
        
        mock_read_parquet.assert_called_once_with("dummy.parquet")
        assert dataset.expected_waveform_length == 2500
        assert dataset.num_leads == 12
        assert dataset.normalize_waveforms is True
        assert dataset.lead_stats == lead_stats
        
    def test_len(self, mock_parquet_data):
        """Test the __len__ method."""
        with patch('pandas.read_parquet', return_value=mock_parquet_data):
            dataset = ECGDataset(parquet_file="dummy.parquet", normalize_waveforms=False)
            assert len(dataset) == 3
            
    @patch('os.path.exists')
    @patch('numpy.load')
    def test_getitem_with_normalization(self, mock_load, mock_exists, mock_signal, mock_lead_stats):
        """Test the __getitem__ method with normalization."""
        mock_exists.return_value = True
        mock_load.return_value = mock_signal
        
        mock_df = pd.DataFrame({
            'waveform_path': ['test_path.npy']
        })
        
        with patch('pandas.read_parquet', return_value=mock_df):
            dataset = ECGDataset(
                parquet_file="dummy.parquet",
                expected_waveform_length=2500,
                num_leads=12,
                normalize_waveforms=True,
                lead_stats=mock_lead_stats,
                signal_path_column='waveform_path'
            )
            
            result = dataset[0]
            
            # Check that result is a dictionary with 'signal' key
            assert isinstance(result, dict)
            assert 'signal' in result
            
            # Check that signal has the correct shape (transposed: 12, 2500)
            assert result['signal'].shape == (12, 2500)
            
            # Verify normalization was applied
            mock_load.assert_called_once_with('test_path.npy')
    
    @patch('os.path.exists')
    @patch('numpy.load')
    def test_getitem_without_normalization(self, mock_load, mock_exists, mock_signal):
        """Test the __getitem__ method without normalization."""
        mock_exists.return_value = True
        mock_load.return_value = mock_signal
        
        mock_df = pd.DataFrame({
            'waveform_path': ['test_path.npy']
        })
        
        with patch('pandas.read_parquet', return_value=mock_df):
            dataset = ECGDataset(
                parquet_file="dummy.parquet",
                expected_waveform_length=2500,
                num_leads=12,
                normalize_waveforms=False,
                signal_path_column='waveform_path'
            )
            
            result = dataset[0]
            
            # Verify signal shape and that it's not normalized
            assert result['signal'].shape == (12, 2500)
            
            # The transposed signal should match our mock signal (transposed)
            np.testing.assert_array_equal(result['signal'], np.transpose(mock_signal, (1, 0)))
    
    @patch('os.path.exists')
    @patch('numpy.load')
    def test_getitem_wrong_shape(self, mock_load, mock_exists, mock_signal):
        """Test the __getitem__ method with incorrect signal shape."""
        mock_exists.return_value = True
        # Create a signal with wrong shape and a correct one
        wrong_shape_signal = np.random.randn(1000, 12)  # Wrong length
        correct_shape_signal = np.random.randn(2500, 12)  # Correct length
        
        # First return wrong shape, then correct shape
        mock_load.side_effect = [wrong_shape_signal, correct_shape_signal]
        
        mock_df = pd.DataFrame({
            'waveform_path': ['test_path_1.npy', 'test_path_2.npy']
        })
        
        with patch('pandas.read_parquet', return_value=mock_df):
            dataset = ECGDataset(
                parquet_file="dummy.parquet",
                expected_waveform_length=2500,  # Expect 2500, but first signal is 1000
                num_leads=12,
                normalize_waveforms=False,
                signal_path_column='waveform_path'
            )
            
            # The first signal has wrong shape, so it should skip to the second item
            result = dataset[0]
            
            # Verify that numpy.load was called twice
            assert mock_load.call_count == 2
            
            # Check that the result has the correct shape
            assert result['signal'].shape == (12, 2500)
            assert result['waveform_path'] == 'test_path_2.npy'
    
    @patch('utils.ddp.DistributedUtils.get_distributed_dataloader')
    def test_get_distributed_ecg_dataloader(self, mock_get_dataloader, mock_parquet_data, mock_lead_stats):
        """Test the get_distributed_ecg_dataloader function."""
        with patch('pandas.read_parquet', return_value=mock_parquet_data):
            get_distributed_ecg_dataloader(
                parquet_file="dummy.parquet",
                expected_waveform_length=2500,
                num_leads=12,
                normalize_waveforms=True,
                lead_stats=mock_lead_stats,
                batch_size=32,
                num_workers=4,
                num_replicas=2,
                rank=0,
                shuffle=True,
                pin_memory=True,
                shuffle_rows=True,
                shuffle_seed=123
            )
            
            # Verify the dataloader was created with correct parameters
            mock_get_dataloader.assert_called_once()

            args, kwargs = mock_get_dataloader.call_args
            dataset_arg = kwargs['dataset']
            assert isinstance(dataset_arg, ECGDataset)

            expected_df = mock_parquet_data.sample(frac=1.0, random_state=123).reset_index(drop=True)
            pd.testing.assert_frame_equal(dataset_arg.data, expected_df)

            assert kwargs['batch_size'] == 32
            assert kwargs['num_workers'] == 4
            assert kwargs['num_replicas'] == 2
            assert kwargs['rank'] == 0
            assert kwargs['shuffle'] is True
            assert kwargs['pin_memory'] is True

    @patch('pandas.read_parquet')
    def test_dataset_row_shuffle(self, mock_read_parquet, mock_parquet_data):
        shuffled = mock_parquet_data.copy()
        mock_read_parquet.return_value = shuffled

        dataset = ECGDataset(
            parquet_file="dummy.parquet",
            normalize_waveforms=False,
            shuffle_rows=True,
            shuffle_seed=7
        )

        assert len(dataset.data) == len(shuffled)
        expected = shuffled.sample(frac=1.0, random_state=7).reset_index(drop=True)
        pd.testing.assert_frame_equal(dataset.data, expected)
