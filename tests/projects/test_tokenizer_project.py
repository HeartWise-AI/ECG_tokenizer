import unittest
from unittest.mock import MagicMock, patch

from projects.tokenizer_project import ECGTokenizerTrainingProject
from utils.config import ECGTokenizerTrainingConfig

class TestECGTokenizerTrainingProject(unittest.TestCase):
    @patch('projects.tokenizer_project.get_distributed_ecg_dataloader')
    @patch('projects.tokenizer_project.ModelRegistry')
    @patch('projects.tokenizer_project.DistributedUtils')
    @patch('projects.tokenizer_project.RunnerRegistry')
    @patch('projects.tokenizer_project.torch.optim.AdamW')
    @patch('projects.tokenizer_project.torch.optim.RAdam')
    @patch('projects.tokenizer_project.torch.optim.lr_scheduler.StepLR')
    @patch('projects.tokenizer_project.torch.optim.lr_scheduler.CosineAnnealingLR')
    @patch('projects.tokenizer_project.torch.amp.GradScaler')
    def test_setup_training_objects(self, mock_grad_scaler, mock_cosine_lr, mock_step_lr, 
                                   mock_radam, mock_adamw, mock_runner_registry, 
                                   mock_distributed_utils, mock_model_registry, mock_dataloader):
        # Mock the config and wandb_wrapper
        mock_config = MagicMock(spec=ECGTokenizerTrainingConfig)
        mock_config.run_mode = "train"
        mock_config.optimizer = "AdamW"
        mock_config.scheduler_name = "step"
        # Add missing attributes
        mock_config.train_dataset_path = "path/to/train"
        mock_config.validation_dataset_path = "path/to/validation" 
        mock_config.waveform_length = 5000
        mock_config.num_leads = 12
        mock_config.normalize_waveforms = True
        mock_config.lead_stats = {"mean": 0, "std": 1}
        mock_config.batch_size = 32
        mock_config.num_workers = 4
        mock_config.world_size = 1
        mock_config.device = 0
        mock_config.pipeline_project = "ECG_Tokenizer"
        mock_config.encoder_name = "encoder"
        mock_config.quantizer_name = "quantizer"
        mock_config.decoder_name = "decoder"
        mock_config.num_quantizers = 4
        mock_config.codebook_size = 512
        mock_config.lr = 0.001
        mock_config.weight_decay = 0.01
        mock_config.step_size = 10
        mock_config.gamma = 0.1
        mock_config.num_epochs = 100
        
        mock_wandb_wrapper = MagicMock()
        
        # Mock return values
        mock_dataloader.return_value = MagicMock()
        mock_model_registry.get.return_value.return_value = MagicMock()
        mock_distributed_utils.DDP.return_value = MagicMock()
        mock_runner_registry.get.return_value.return_value = MagicMock()
        
        # Mock the optimizers and scheduler
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer
        mock_radam.return_value = mock_optimizer
        mock_scheduler = MagicMock()
        mock_step_lr.return_value = mock_scheduler
        mock_cosine_lr.return_value = mock_scheduler
        mock_scaler = MagicMock()
        mock_grad_scaler.return_value = mock_scaler
        
        # Create the project and call run
        project = ECGTokenizerTrainingProject(mock_config, mock_wandb_wrapper)
        
        # Test _setup_training_objects
        training_objects = project._setup_training_objects()
        
        # Verify expected objects are returned
        self.assertIn("optimizer", training_objects)
        self.assertIn("scheduler", training_objects)
        self.assertIn("scaler", training_objects)
        self.assertIn("ecg_tokenizer", training_objects)
        self.assertIn("train_dataloader", training_objects)
        self.assertIn("validation_dataloader", training_objects)
    
    @patch('projects.tokenizer_project.RunnerRegistry')
    def test_run_method(self, mock_runner_registry):
        # Mock the config and wandb_wrapper
        mock_config = MagicMock(spec=ECGTokenizerTrainingConfig)
        mock_config.run_mode = "train"
        # Add missing attribute
        mock_config.pipeline_project = "ECG_Tokenizer"
        mock_wandb_wrapper = MagicMock()
        
        # Create a mock runner that will be returned
        mock_runner = MagicMock()
        mock_runner_registry.get.return_value.return_value = mock_runner
        
        # Create partial mock of the project class to avoid actually setting up training objects
        with patch.object(ECGTokenizerTrainingProject, '_setup_training_objects') as mock_setup:
            mock_setup.return_value = {
                "optimizer": MagicMock(),
                "scheduler": MagicMock(),
                "scaler": MagicMock(),
                "ecg_tokenizer": MagicMock(),
                "train_dataloader": MagicMock(),
                "validation_dataloader": MagicMock()
            }
            
            # Create the project and call run
            project = ECGTokenizerTrainingProject(mock_config, mock_wandb_wrapper)
            project.run()
            
            # Verify the runner was executed
            mock_runner.execute.assert_called_once_with(mode="train")
    
    def test_setup_inference_objects_not_implemented(self):
        # Mock the config and wandb_wrapper
        mock_config = MagicMock(spec=ECGTokenizerTrainingConfig)
        mock_wandb_wrapper = MagicMock()
        
        # Create the project
        project = ECGTokenizerTrainingProject(mock_config, mock_wandb_wrapper)
        
        # Verify that _setup_inference_objects raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            project._setup_inference_objects()

if __name__ == "__main__":
    unittest.main() 