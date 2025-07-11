"""
Base Trainer Module

Abstract base class for all training strategies with shared functionality.
Provides standardized interfaces and resource management.
"""

import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback
)

import sys
from pathlib import Path

# Add both parent directory and project root to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
project_root = parent_dir.parent

sys.path.insert(0, str(parent_dir))  # ml_training directory
sys.path.insert(0, str(project_root))  # project root

try:
    from base_environment import SimpleEscapeCageEnv, FastEscapeCageEnv, TestEscapeCageEnv
    from config.hyperparameters import get_hyperparameters, validate_hyperparameters
except ImportError:
    # Fallback for different execution contexts
    try:
        from ml_training.base_environment import SimpleEscapeCageEnv, FastEscapeCageEnv, TestEscapeCageEnv
        from ml_training.config.hyperparameters import get_hyperparameters, validate_hyperparameters
    except ImportError:
        # Final fallback with absolute imports
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from base_environment import SimpleEscapeCageEnv, FastEscapeCageEnv, TestEscapeCageEnv
        from config.hyperparameters import get_hyperparameters, validate_hyperparameters


# Global environment registry for all trainers
ENVIRONMENT_REGISTRY = {
    "simple": SimpleEscapeCageEnv,
    "fast": FastEscapeCageEnv,
    "debug": TestEscapeCageEnv  # Renamed from "test" to "debug" for clarity
}


class TrainingError(Exception):
    """Custom exception for training-related errors."""
    pass


class TrainingMetricsCallback(BaseCallback):
    """Enhanced callback for detailed training metrics collection."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.success_count: int = 0
        self.total_episodes: int = 0
        
    def _on_step(self) -> bool:
        """Called at each step."""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])
                self.total_episodes += 1
                
                # Track success if available
                if 'success' in info and info['success']:
                    self.success_count += 1


class BaseTrainer(ABC):
    """
    Abstract base trainer class with shared functionality.
    
    Provides:
    - Standardized environment and agent creation
    - Resource management with context managers
    - Consistent training session handling
    - Proper error handling and logging
    - Unified environment selection for all trainers
    """
    
    def __init__(self, models_dir: str = "models", environment_type: Optional[str] = None):
        """
        Initialize base trainer with configurable environment.
        
        Args:
            models_dir: Directory to save trained models
            environment_type: Type of environment ("simple", "fast", "test").
                            If None, uses get_default_environment_type()
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        
        # Set environment type
        if environment_type is None:
            environment_type = self.get_default_environment_type()
        
        if environment_type not in ENVIRONMENT_REGISTRY:
            available = list(ENVIRONMENT_REGISTRY.keys())
            raise ValueError(f"Unknown environment type '{environment_type}'. Available: {available}")
        
        self.environment_type = environment_type
        self.logger.info(f"Initialized {self.__class__.__name__} with {environment_type} environment")
        
    @abstractmethod
    def get_training_type(self) -> str:
        """
        Get the training type identifier for this trainer.
        
        Returns:
            Training type string (e.g., "standard", "fast")
        """
        pass
    
    def get_default_environment_type(self) -> str:
        """
        Get the default environment type for this trainer.
        Subclasses can override to set trainer-specific defaults.
        
        Returns:
            Default environment type string
        """
        return "simple"  # Conservative default
    
    def get_environment(self):
        """
        Get environment instance based on configured type.
        
        Returns:
            Environment instance
        """
        env_class = ENVIRONMENT_REGISTRY[self.environment_type]
        return env_class()
    
    @contextmanager
    def managed_environment(self):
        """
        Context manager for proper environment resource management.
        
        Yields:
            Environment instance with automatic cleanup
        """
        env = None
        try:
            env = self.get_environment()
            yield env
        except Exception as e:
            self.logger.error(f"Environment error: {e}")
            raise TrainingError(f"Environment setup failed: {e}") from e
        finally:
            if env:
                try:
                    env.close()
                    self.logger.debug("Environment closed successfully")
                except Exception as e:
                    self.logger.warning(f"Error closing environment: {e}")
    
    def create_agent(self, env, custom_params: Optional[Dict[str, Any]] = None) -> PPO:
        """
        Create PPO agent with trainer-specific configuration.
        
        Args:
            env: Training environment
            custom_params: Custom hyperparameters to override defaults
            
        Returns:
            Configured PPO agent
            
        Raises:
            TrainingError: If agent creation fails
        """
        try:
            # Get hyperparameters for this trainer type
            config = get_hyperparameters(
                training_type=self.get_training_type(),
                custom_params=custom_params
            )
            
            # Validate configuration
            validate_hyperparameters(config)
            
            # Create agent
            agent = PPO("MlpPolicy", env, verbose=1, **config.to_dict())
            
            self.logger.info(f"Created PPO agent with {self.get_training_type()} configuration")
            self.logger.debug(f"Hyperparameters: {config.to_dict()}")
            
            return agent
            
        except Exception as e:
            self.logger.error(f"Agent creation failed: {e}")
            raise TrainingError(f"Failed to create agent: {e}") from e
    
    def run_training_session(self, agent: PPO, total_timesteps: int, 
                           save_path: str, eval_env=None, 
                           eval_freq: int = 10000) -> Dict[str, Any]:
        """
        Run complete training session with monitoring and saving.
        
        Args:
            agent: PPO agent to train
            total_timesteps: Number of timesteps to train
            save_path: Path to save the trained model
            eval_env: Optional evaluation environment
            eval_freq: Frequency of evaluation episodes
            
        Returns:
            Dictionary containing training results and metrics
            
        Raises:
            TrainingError: If training fails
        """
        start_time = time.time()
        
        try:
            # Setup callbacks
            callbacks = []
            metrics_callback = TrainingMetricsCallback(verbose=1)
            callbacks.append(metrics_callback)
            
            # Add evaluation callback if eval environment provided
            if eval_env:
                eval_callback = EvalCallback(
                    eval_env, 
                    best_model_save_path=str(self.models_dir / f"{save_path}_best"),
                    log_path=str(self.models_dir / "logs"),
                    eval_freq=eval_freq,
                    deterministic=True,
                    render=False
                )
                callbacks.append(eval_callback)
            
            # Train the model
            self.logger.info(f"Starting training for {total_timesteps:,} timesteps")
            agent.learn(total_timesteps=total_timesteps, callback=callbacks)
            training_time = time.time() - start_time
            
            # Save the model
            model_path = self.models_dir / f"{save_path}.zip"
            agent.save(str(model_path))
            
            # Calculate training metrics
            total_reward = sum(metrics_callback.episode_rewards) if metrics_callback.episode_rewards else 0
            avg_reward = total_reward / len(metrics_callback.episode_rewards) if metrics_callback.episode_rewards else 0
            success_rate = (metrics_callback.success_count / metrics_callback.total_episodes * 100) if metrics_callback.total_episodes else 0
            
            results = {
                'training_time': training_time,
                'total_timesteps': total_timesteps,
                'model_path': str(model_path),
                'total_episodes': metrics_callback.total_episodes,
                'success_count': metrics_callback.success_count,
                'success_rate': success_rate,
                'avg_reward': avg_reward,
                'environment_type': self.environment_type,
                'trainer_type': self.get_training_type()
            }
            
            self.logger.info(f"Training completed successfully")
            self.logger.info(f"Success rate: {success_rate:.1f}% ({metrics_callback.success_count}/{metrics_callback.total_episodes})")
            self.logger.info(f"Average reward: {avg_reward:.2f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training session failed: {e}")
            raise TrainingError(f"Training session failed: {e}") from e
    
    def train(self, total_timesteps: int, model_name: str, 
              custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main training method with proper resource management.
        
        Args:
            total_timesteps: Number of training timesteps
            model_name: Name for the saved model
            custom_params: Custom hyperparameters to override defaults
            
        Returns:
            Training results dictionary
            
        Raises:
            TrainingError: If training fails
        """
        self.logger.info(f"Starting {self.__class__.__name__} training")
        self.logger.info(f"Environment: {self.environment_type}")
        self.logger.info(f"Training for {total_timesteps:,} steps")
        
        try:
            with self.managed_environment() as env:
                # Create agent
                agent = self.create_agent(env, custom_params)
                
                # Run training
                results = self.run_training_session(
                    agent=agent,
                    total_timesteps=total_timesteps,
                    save_path=model_name
                )
                
                self.logger.info(f"Training complete! AI saved as '{self.models_dir / model_name}.zip'")
                return results
                
        except KeyboardInterrupt:
            self.logger.info("Training stopped by user")
            return {'status': 'interrupted'}
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            error_msg = (
                "Training failed. Troubleshooting tips:\n"
                "  - Make sure Unity is running with the escape cage scene loaded\n"
                "  - Check that no firewall is blocking port 9999\n"
                "  - Verify Python has the required packages installed"
            )
            print(error_msg)
            raise 