"""
Continue Trainer Module

Continue training existing models with additional timesteps or fine-tuning.
Provides incremental learning and model improvement capabilities.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional

from stable_baselines3 import PPO

import sys
from pathlib import Path

# Add both parent directory and project root to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
project_root = parent_dir.parent

sys.path.insert(0, str(parent_dir))  # ml_training directory
sys.path.insert(0, str(project_root))  # project root

from model_utils import find_model_file, list_available_models
from .base_trainer import BaseTrainer, TrainingError


class ContinueTrainer(BaseTrainer):
    """
    Continue training existing models with additional timesteps or fine-tuning.
    
    Features:
    - Load and continue from existing models
    - Incremental learning capabilities
    - Hyperparameter adjustment for fine-tuning
    - Backup and versioning support
    - Configurable environment selection (simple, fast, test)
    """
    
    def get_training_type(self) -> str:
        """Get training type identifier."""
        return "standard"  # Use standard config for continued training
    
    def get_default_environment_type(self) -> str:
        """Get default environment type for continue trainer."""
        return "simple"  # Conservative choice for continuing existing models
    
    def continue_training(self, model_name: str, additional_steps: int = 25000, 
                         save_backup: bool = True) -> Dict[str, Any]:
        """
        Continue training an existing model with additional timesteps.
        
        Args:
            model_name: Name of the model to continue training
            additional_steps: Additional training timesteps (default: 25000)
            save_backup: Whether to create backup of original model
            
        Returns:
            Training results dictionary
            
        Raises:
            TrainingError: If model not found or training fails
        """
        print(f"Continuing training for model: {model_name}")
        print(f"Environment: {self.environment_type}")
        print(f"Additional training steps: {additional_steps:,}")
        
        # Find the model file
        model_path = find_model_file(model_name)
        if not model_path:
            available_models = list_available_models()
            error_msg = f"Model '{model_name}' not found!"
            if available_models:
                error_msg += f"\nAvailable models: {', '.join(available_models)}"
            else:
                error_msg += "\nNo models found."
            raise TrainingError(error_msg)
        
        try:
            with self.managed_environment() as env:
                # Load the existing model
                print(f"Loading model from: {model_path}")
                ai_agent = PPO.load(str(model_path), env=env)
                
                # Create backup if requested
                if save_backup:
                    backup_path = self.models_dir / f"{model_name}_backup.zip"
                    ai_agent.save(str(backup_path))
                    print(f"Backup created: {backup_path}")
                
                print("Continuing training with existing model...")
                
                # Continue training
                start_time = time.time()
                ai_agent.learn(total_timesteps=additional_steps)
                training_time = time.time() - start_time
                
                # Save the updated model
                continued_name = f"{model_name}_continued"
                continued_path = self.models_dir / f"{continued_name}.zip"
                ai_agent.save(str(continued_path))
                
                print(f"Continued training complete! AI saved as '{continued_path}'")
                print(f"Additional training took {training_time:.1f} seconds")
                
                results = {
                    'original_model': model_name,
                    'continued_model': continued_name,
                    'additional_steps': additional_steps,
                    'training_time': training_time,
                    'backup_created': save_backup,
                    'model_path': str(continued_path),
                    'environment_type': self.environment_type,
                    'trainer_type': self.get_training_type()
                }
                
                return results
                
        except Exception as e:
            self.logger.error(f"Continue training failed: {e}")
            raise TrainingError(f"Failed to continue training: {e}") from e
    
    def fine_tune(self, model_name: str, fine_tune_steps: int = 15000) -> Dict[str, Any]:
        """
        Fine-tune an existing model with conservative hyperparameters.
        
        Args:
            model_name: Name of the model to fine-tune
            fine_tune_steps: Number of fine-tuning steps (default: 15000)
            
        Returns:
            Fine-tuning results dictionary
            
        Raises:
            TrainingError: If model not found or fine-tuning fails
        """
        print(f"Fine-tuning model: {model_name}")
        print(f"Environment: {self.environment_type}")
        print(f"Fine-tuning steps: {fine_tune_steps:,}")
        
        # Find the model file
        model_path = find_model_file(model_name)
        if not model_path:
            available_models = list_available_models()
            error_msg = f"Model '{model_name}' not found!"
            if available_models:
                error_msg += f"\nAvailable models: {', '.join(available_models)}"
            else:
                error_msg += "\nNo models found."
            raise TrainingError(error_msg)
        
        try:
            with self.managed_environment() as env:
                # Load the existing model
                print(f"Loading model from: {model_path}")
                ai_agent = PPO.load(str(model_path), env=env)
                
                # Use conservative learning rate for fine-tuning
                ai_agent.learning_rate = 0.0001
                print("Using lower learning rate (0.0001) for fine-tuning")
                
                print("Fine-tuning the AI with more conservative learning...")
                
                start_time = time.time()
                ai_agent.learn(total_timesteps=fine_tune_steps)
                training_time = time.time() - start_time
                
                # Save the fine-tuned model
                finetuned_name = f"{model_name}_finetuned"
                finetuned_path = self.models_dir / f"{finetuned_name}.zip"
                ai_agent.save(str(finetuned_path))
                
                print(f"Fine-tuning complete! AI saved as '{finetuned_path}'")
                print(f"Fine-tuning took {training_time:.1f} seconds")
                
                results = {
                    'original_model': model_name,
                    'finetuned_model': finetuned_name,
                    'fine_tune_steps': fine_tune_steps,
                    'training_time': training_time,
                    'model_path': str(finetuned_path),
                    'environment_type': self.environment_type,
                    'trainer_type': self.get_training_type()
                }
                
                return results
                
        except Exception as e:
            self.logger.error(f"Fine-tuning failed: {e}")
            raise TrainingError(f"Failed to fine-tune model: {e}") from e
    
    def list_available_models(self) -> list[str]:
        """
        List all available saved models.
        
        Returns:
            List of available model names
        """
        return list_available_models() 