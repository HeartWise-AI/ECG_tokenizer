import unittest
from unittest.mock import MagicMock, patch, mock_open
import os

from projects.llm_finetuning_project import LLMFinetuningProject
from utils.config import LLMFinetuningConfig

class TestLLMFinetuningProject(unittest.TestCase):
    @patch('projects.llm_finetuning_project.GPT2Tokenizer')
    @patch('projects.llm_finetuning_project.get_distributed_clinical_report_dataloader')
    @patch('projects.llm_finetuning_project.ModelRegistry')
    @patch('projects.llm_finetuning_project.DistributedUtils')
    @patch('projects.llm_finetuning_project.torch.optim.AdamW')
    @patch('projects.llm_finetuning_project.torch.optim.RAdam')
    @patch('projects.llm_finetuning_project.torch.optim.lr_scheduler.StepLR')
    @patch('projects.llm_finetuning_project.torch.optim.lr_scheduler.CosineAnnealingLR')
    @patch('projects.llm_finetuning_project.torch.amp.GradScaler')
    def test_setup_training_objects(self, mock_grad_scaler, mock_cosine_lr, mock_step_lr,
                                   mock_radam, mock_adamw, mock_distributed_utils, 
                                   mock_model_registry, mock_dataloader, mock_tokenizer):
        # Mock the config and wandb_wrapper
        mock_config = MagicMock(spec=LLMFinetuningConfig)
        mock_config.optimizer = "AdamW"
        mock_config.scheduler_type = "step"
        mock_config.train_dataset_path = "/path/to/train"
        mock_config.train_embeddings_path = "/path/to/train_embeddings"
        mock_config.validation_dataset_path = "/path/to/validation"
        mock_config.validation_embeddings_path = "/path/to/validation_embeddings"
        mock_config.max_token_length = 512
        mock_config.batch_size = 32
        mock_config.num_workers = 4
        mock_config.world_size = 1
        mock_config.device = 0
        mock_config.trainable_model_name = "GPT2_with_embeddings"
        mock_config.huggingface_model_name = "gpt2"
        mock_config.gpt2_embedding_size = 768
        mock_config.embedding_reducer_name = "mlp_reducer"
        mock_config.reducer_dropout = 0.1
        mock_config.lr = 0.001
        mock_config.step_size = 10
        mock_config.gamma = 0.1
        mock_config.num_epochs = 100
        mock_wandb_wrapper = MagicMock()
        
        # Mock return values
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_dataloader.return_value = MagicMock()
        mock_model_registry.get.return_value.return_value = MagicMock()
        mock_distributed_utils.DDP.return_value = MagicMock()
        
        # Mock the optimizers and scheduler
        mock_optimizer = MagicMock()
        mock_adamw.return_value = mock_optimizer
        mock_radam.return_value = mock_optimizer
        mock_scheduler = MagicMock()
        mock_step_lr.return_value = mock_scheduler
        mock_cosine_lr.return_value = mock_scheduler
        mock_scaler = MagicMock()
        mock_grad_scaler.return_value = mock_scaler
        
        # Patch _get_embedding_size to return a dummy value
        with patch.object(LLMFinetuningProject, '_get_embedding_size') as mock_get_embedding_size:
            mock_get_embedding_size.return_value = (768,)
            
            # Create the project and call _setup_training_objects
            project = LLMFinetuningProject(mock_config, mock_wandb_wrapper)
            training_objects = project._setup_training_objects()
            
            # Verify expected objects are returned
            self.assertIn("train_dataloader", training_objects)
            self.assertIn("val_dataloader", training_objects)
            self.assertIn("optimizer", training_objects)
            self.assertIn("scheduler", training_objects)
            self.assertIn("scaler", training_objects)
            self.assertIn("model", training_objects)
    
    @patch('projects.llm_finetuning_project.os.path.exists')
    @patch('projects.llm_finetuning_project.torch.load')
    def test_load_checkpoint(self, mock_torch_load, mock_exists):
        # Mock the config and wandb_wrapper
        mock_config = MagicMock(spec=LLMFinetuningConfig)
        mock_wandb_wrapper = MagicMock()
        
        # Set up mocks
        mock_exists.return_value = True
        mock_checkpoint = {"model_state_dict": MagicMock()}
        mock_torch_load.return_value = mock_checkpoint
        
        # Create the project and call _load_checkpoint
        project = LLMFinetuningProject(mock_config, mock_wandb_wrapper)
        result = project._load_checkpoint("fake/checkpoint.pt")
        
        # Verify result
        self.assertEqual(result, mock_checkpoint)
        mock_torch_load.assert_called_once()
    
    @patch('projects.llm_finetuning_project.os.listdir')
    @patch('projects.llm_finetuning_project.os.path.join')
    @patch('projects.llm_finetuning_project.np.load')
    def test_get_embedding_size(self, mock_np_load, mock_join, mock_listdir):
        # Mock the config and wandb_wrapper
        mock_config = MagicMock(spec=LLMFinetuningConfig)
        mock_wandb_wrapper = MagicMock()
        
        # Set up mocks
        mock_listdir.return_value = ["embedding.npy"]
        mock_join.return_value = "path/to/embedding.npy"
        mock_array = MagicMock()
        mock_array.shape = (512, 768)
        mock_np_load.return_value = mock_array
        
        # Create the project and call _get_embedding_size
        project = LLMFinetuningProject(mock_config, mock_wandb_wrapper)
        result = project._get_embedding_size("fake/embeddings")
        
        # Verify result
        self.assertEqual(result, (512, 768))
    
    @patch('projects.llm_finetuning_project.RunnerRegistry')
    def test_run_method_train(self, mock_runner_registry):
        # Mock the config and wandb_wrapper
        mock_config = MagicMock(spec=LLMFinetuningConfig)
        mock_config.run_mode = "train"
        # Add missing attribute
        mock_config.runner_name = "LLM_Training_Runner"
        mock_config.is_ref_device = True
        mock_wandb_wrapper = MagicMock()
    
        # Create a mock runner
        mock_runner = MagicMock()
        mock_runner_registry.get.return_value.return_value = mock_runner
    
        # Create partial mock of the project class to avoid actually setting up training objects
        with patch.object(LLMFinetuningProject, '_setup_training_objects') as mock_setup, \
             patch('projects.llm_finetuning_project.backup_config'), \
             patch('projects.llm_finetuning_project.generate_output_dir_name', return_value="mock_output_dir"):
            mock_setup.return_value = {
                "train_dataloader": MagicMock(),
                "val_dataloader": MagicMock(),
                "optimizer": MagicMock(),
                "scheduler": MagicMock(),
                "scaler": MagicMock(),
                "model": MagicMock()
            }
            
            # Create the project and call run
            project = LLMFinetuningProject(mock_config, mock_wandb_wrapper)
            project.run()
            
            # Verify the runner was executed
            mock_runner.execute.assert_called_once_with(mode="train")

if __name__ == "__main__":
    unittest.main() 