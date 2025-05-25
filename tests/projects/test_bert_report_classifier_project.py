import unittest
from unittest.mock import MagicMock, patch

from projects.bert_report_classifier_project import BertReportClassifierProject
from utils.config import BertReportClassifierConfig

class TestBertReportClassifierProject(unittest.TestCase):
    @patch('projects.bert_report_classifier_project.load_api_keys')
    @patch('projects.bert_report_classifier_project.HuggingFaceWrapper')
    @patch('projects.bert_report_classifier_project.BertTokenizer')
    @patch('projects.bert_report_classifier_project.ModelRegistry')
    @patch('projects.bert_report_classifier_project.get_distributed_clinical_report_dataloader')
    def test_setup_inference_objects(self, mock_dataloader, mock_model_registry, 
                                    mock_tokenizer, mock_huggingface, mock_load_api_keys):
        # Mock the config and wandb_wrapper
        mock_config = MagicMock(spec=BertReportClassifierConfig)
        mock_config.run_mode = "inference"
        # Add missing attributes
        mock_config.api_keys_path = "/path/to/api_keys.json"
        mock_config.huggingface_model_name = "bert-base-uncased"
        mock_config.store_model_path = "/path/to/model"
        mock_config.num_classes = 5
        mock_config.device = 0
        mock_config.predictions_reports_path = "/path/to/predictions"
        mock_config.batch_size = 32
        mock_config.num_workers = 4
        mock_config.world_size = 1
        mock_config.pipeline_project = "BERT_Classifier"
        
        mock_wandb_wrapper = MagicMock()
        
        # Mock return values
        mock_load_api_keys.return_value = {"HUGGING_FACE_TOKEN": "fake_token"}
        mock_huggingface.get_model.return_value = "model/path"
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model_registry.get.return_value.return_value = MagicMock()
        mock_dataloader.return_value = MagicMock()
        
        # Create the project
        project = BertReportClassifierProject(mock_config, mock_wandb_wrapper)
        
        # Test _setup_inference_objects
        inference_objects = project._setup_inference_objects()
        
        # Verify expected objects are returned
        self.assertIn("val_dataloader", inference_objects)
        self.assertIn("model", inference_objects)
    
    def test_setup_training_objects_not_implemented(self):
        # Mock the config and wandb_wrapper
        mock_config = MagicMock(spec=BertReportClassifierConfig)
        mock_wandb_wrapper = MagicMock()
        
        # Create the project
        project = BertReportClassifierProject(mock_config, mock_wandb_wrapper)
        
        # Verify that _setup_training_objects raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            project._setup_training_objects()
    
    @patch('projects.bert_report_classifier_project.RunnerRegistry')
    def test_run_method_inference(self, mock_runner_registry):
        # Mock the config and wandb_wrapper
        mock_config = MagicMock(spec=BertReportClassifierConfig)
        mock_config.run_mode = "inference"
        # Add missing attribute
        mock_config.pipeline_project = "BERT_Classifier"
        mock_wandb_wrapper = MagicMock()
        
        # Create a mock runner
        mock_runner = MagicMock()
        mock_runner_registry.get.return_value.return_value = mock_runner
        
        # Create partial mock of the project class to avoid actually setting up inference objects
        with patch.object(BertReportClassifierProject, '_setup_inference_objects') as mock_setup:
            mock_setup.return_value = {
                "val_dataloader": MagicMock(),
                "model": MagicMock(),
            }
            
            # Create the project and call run
            project = BertReportClassifierProject(mock_config, mock_wandb_wrapper)
            project.run()
            
            # Verify the runner was executed
            mock_runner.execute.assert_called_once_with(mode="inference")
    
    @patch('projects.bert_report_classifier_project.RunnerRegistry')
    def test_run_method_train_not_implemented(self, mock_runner_registry):
        # Mock the config and wandb_wrapper
        mock_config = MagicMock(spec=BertReportClassifierConfig)
        mock_config.run_mode = "train"
        mock_wandb_wrapper = MagicMock()
        
        # Create the project
        project = BertReportClassifierProject(mock_config, mock_wandb_wrapper)
        
        # Verify that run with train mode raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            project.run()

if __name__ == "__main__":
    unittest.main() 