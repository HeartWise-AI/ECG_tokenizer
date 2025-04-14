import unittest
from unittest.mock import MagicMock, patch
import torch

from utils.schedulers import get_scheduler, scheduler_is_per_iteration
from utils.config import HeartWiseConfig

class TestSchedulers(unittest.TestCase):
    def setUp(self):
        # Create mock objects for testing
        self.optimizer = MagicMock(spec=torch.optim.Optimizer)
        self.train_dataloader = MagicMock(spec=torch.utils.data.DataLoader)
        self.train_dataloader.__len__.return_value = 100  # 100 batches per epoch
        self.num_epochs = 10
        
    def test_get_scheduler_cosine(self):
        with patch('torch.optim.lr_scheduler.CosineAnnealingLR') as mock_scheduler:
            # Call the function with 'cosine' scheduler
            get_scheduler('cosine', self.optimizer, self.num_epochs, self.train_dataloader)
            # Verify the scheduler was created with correct parameters
            mock_scheduler.assert_called_once()
            args, kwargs = mock_scheduler.call_args
            self.assertEqual(args[0], self.optimizer)
            self.assertEqual(kwargs['T_max'], 1000)  # 100 batches * 10 epochs

    def test_get_scheduler_step(self):
        with patch('torch.optim.lr_scheduler.StepLR') as mock_scheduler:
            # Call the function with 'step' scheduler
            get_scheduler('step', self.optimizer, self.num_epochs, self.train_dataloader, gamma=0.5, step_size=2)
            # Verify the scheduler was created with correct parameters
            mock_scheduler.assert_called_once()
            args, kwargs = mock_scheduler.call_args
            self.assertEqual(args[0], self.optimizer)
            self.assertEqual(kwargs['gamma'], 0.5)
            self.assertEqual(kwargs['step_size'], 2)

    def test_get_scheduler_cosine_warm_restart(self):
        with patch('torch.optim.lr_scheduler.CosineAnnealingWarmRestarts') as mock_scheduler:
            # Call the function with 'cosine_warm_restart' scheduler
            get_scheduler('cosine_warm_restart', self.optimizer, self.num_epochs, 
                          self.train_dataloader, warm_restart_tmult=3, num_restarts=5)
            # Verify the scheduler was created with correct parameters
            mock_scheduler.assert_called_once()
            args, kwargs = mock_scheduler.call_args
            self.assertEqual(args[0], self.optimizer)
            self.assertEqual(kwargs['T_mult'], 3)
            self.assertEqual(kwargs['T_0'], 200)  # 1000 total steps / 5 restarts = 200

    def test_get_scheduler_linear_warmup(self):
        with patch('utils.schedulers.get_linear_schedule_with_warmup') as mock_scheduler:
            # Call the function with 'linear_warmup' scheduler
            get_scheduler('linear_warmup', self.optimizer, self.num_epochs, 
                          self.train_dataloader, num_warmup_percent=0.2)
            # Verify the scheduler was created with correct parameters
            mock_scheduler.assert_called_once()
            args, kwargs = mock_scheduler.call_args
            self.assertEqual(args[0], self.optimizer)
            self.assertEqual(kwargs['num_warmup_steps'], 200)  # 20% of 1000 total steps
            self.assertEqual(kwargs['num_training_steps'], 1000)

    def test_get_scheduler_cosine_with_warmup(self):
        with patch('utils.schedulers.get_cosine_schedule_with_warmup') as mock_scheduler:
            # Call the function with 'cosine_with_warmup' scheduler
            get_scheduler('cosine_with_warmup', self.optimizer, self.num_epochs, 
                          self.train_dataloader, num_warmup_percent=0.1)
            # Verify the scheduler was created with correct parameters
            mock_scheduler.assert_called_once()
            args, kwargs = mock_scheduler.call_args
            self.assertEqual(args[0], self.optimizer)
            self.assertEqual(kwargs['num_warmup_steps'], 100)  # 10% of 1000 total steps
            self.assertEqual(kwargs['num_training_steps'], 1000)

    def test_get_scheduler_cosine_with_hard_restarts(self):
        with patch('utils.schedulers.get_cosine_with_hard_restarts_schedule_with_warmup') as mock_scheduler:
            # Call the function with 'cosine_with_hard_restarts_with_warmup' scheduler
            get_scheduler('cosine_with_hard_restarts_with_warmup', self.optimizer, self.num_epochs, 
                          self.train_dataloader, num_warmup_percent=0.1, num_hard_restarts_cycles=2.0)
            # Verify the scheduler was created with correct parameters
            mock_scheduler.assert_called_once()
            args, kwargs = mock_scheduler.call_args
            self.assertEqual(args[0], self.optimizer)
            self.assertEqual(kwargs['num_warmup_steps'], 100)  # 10% of 1000 total steps
            self.assertEqual(kwargs['num_training_steps'], 1000)
            self.assertEqual(kwargs['num_cycles'], 2.0)

    def test_get_scheduler_invalid(self):
        # Test with invalid scheduler name
        with self.assertRaises(ValueError):
            get_scheduler('invalid_scheduler', self.optimizer, self.num_epochs, self.train_dataloader)

    def test_scheduler_with_gradient_accumulation(self):
        with patch('torch.optim.lr_scheduler.CosineAnnealingLR') as mock_scheduler:
            # Call the function with gradient accumulation steps = 2
            get_scheduler('cosine', self.optimizer, self.num_epochs, 
                          self.train_dataloader, gradient_accumulation_steps=2)
            # Verify the total steps are adjusted correctly
            mock_scheduler.assert_called_once()
            args, kwargs = mock_scheduler.call_args
            self.assertEqual(kwargs['T_max'], 500)  # (100 batches * 10 epochs) / 2 gradient_accumulation_steps

    def test_scheduler_is_per_iteration(self):
        # Test step scheduler (should be per epoch)
        config = MagicMock(spec=HeartWiseConfig)
        config.scheduler_name = "step"
        self.assertFalse(scheduler_is_per_iteration(config))
        
        # Test cosine scheduler (should be per iteration)
        config.scheduler_name = "cosine"
        self.assertTrue(scheduler_is_per_iteration(config))
        
        # Test cosine_warm_restart scheduler (should be per iteration)
        config.scheduler_name = "cosine_warm_restart"
        self.assertTrue(scheduler_is_per_iteration(config))
        
        # Test with mixed case
        config.scheduler_name = "COSINE"
        self.assertTrue(scheduler_is_per_iteration(config))
        
        # Test with no scheduler defined
        config = MagicMock(spec=HeartWiseConfig)
        delattr(config, "scheduler_name")
        self.assertTrue(scheduler_is_per_iteration(config))

if __name__ == "__main__":
    unittest.main() 