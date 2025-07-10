"""
AI Escape Cage Trainer - Comprehensive Training Module

Consolidated training system with multiple training strategies and comprehensive utilities.
Features all training methods: standard, fast, continue, and improved training.
"""

import os
import argparse
import logging
import time
import datetime
import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np
import tensorflow as tf
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from base_environment import SimpleEscapeCageEnv, FastEscapeCageEnv, TestEscapeCageEnv
from logger_setup import TrainingLogger


class TrainingMetricsCallback(BaseCallback):
    """Custom callback for detailed training metrics collection."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
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
    Base trainer class with common functionality for all training types.
    
    Provides shared methods for:
    - Environment setup
    - Model creation and saving
    - Testing and evaluation
    - Progress tracking and logging
    """
    
    def __init__(self, models_dir: str = "models"):
        """Initialize base trainer."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f'{self.__class__.__name__}')
        
    @abstractmethod
    def create_agent(self, env) -> PPO:
        """Create PPO agent with trainer-specific hyperparameters."""
        pass
    
    @abstractmethod
    def get_environment(self):
        """Get the appropriate environment for this trainer."""
        pass
    
    def create_ppo_agent(self, env, training_type: str = "standard", 
                        custom_params: Optional[Dict[str, Any]] = None) -> PPO:
        """
        Create a PPO agent with optimized hyperparameters.
        
        Args:
            env: Training environment
            training_type: Type of training ("standard", "fast", or "conservative")
            custom_params: Custom hyperparameters to override defaults
            
        Returns:
            Configured PPO agent
        """
        # Default hyperparameters for different training types
        hyperparams = {
            "standard": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5
            },
            "fast": {
                "learning_rate": 1e-3,
                "n_steps": 512,
                "batch_size": 32,
                "n_epochs": 4,
                "gamma": 0.95,
                "gae_lambda": 0.9,
                "clip_range": 0.3,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5
            },
            "conservative": {
                "learning_rate": 1e-4,
                "n_steps": 4096,
                "batch_size": 128,
                "n_epochs": 20,
                "gamma": 0.995,
                "gae_lambda": 0.98,
                "clip_range": 0.1,
                "ent_coef": 0.001,
                "vf_coef": 0.5,
                "max_grad_norm": 0.3
            }
        }
        
        # Get base parameters
        params = hyperparams.get(training_type, hyperparams["standard"]).copy()
        
        # Override with custom parameters
        if custom_params:
            params.update(custom_params)
        
        # Create agent
        agent = PPO("MlpPolicy", env, verbose=1, **params)
        
        self.logger.info(f"Created PPO agent with {training_type} configuration")
        self.logger.debug(f"Hyperparameters: {params}")
        
        return agent
    
    def run_training_session(self, agent: PPO, total_timesteps: int, 
                           save_path: str, eval_env=None, eval_freq: int = 10000) -> Dict[str, Any]:
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
        """
        start_time = time.time()
        
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
        try:
            agent.learn(total_timesteps=total_timesteps, callback=callbacks)
            training_time = time.time() - start_time
            
            # Save the model
            model_path = self.models_dir / save_path
            agent.save(str(model_path))
            
            # Calculate results
            results = {
                'training_time': training_time,
                'total_timesteps': total_timesteps,
                'total_episodes': metrics_callback.total_episodes,
                'success_count': metrics_callback.success_count,
                'success_rate': (metrics_callback.success_count / max(metrics_callback.total_episodes, 1)) * 100,
                'average_reward': np.mean(metrics_callback.episode_rewards) if metrics_callback.episode_rewards else 0,
                'average_episode_length': np.mean(metrics_callback.episode_lengths) if metrics_callback.episode_lengths else 0,
                'model_path': str(model_path)
            }
            
            self.logger.info(f"Training completed in {training_time:.1f} seconds")
            self.logger.info(f"Model saved to: {model_path}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def test_trained_model(self, ai_agent, env, episodes: int = 3) -> Dict[str, Any]:
        """
        Test the trained model and provide detailed feedback.
        
        Args:
            ai_agent: The trained PPO agent
            env: Environment instance
            episodes: Number of test episodes to run
            
        Returns:
            Dictionary containing test results
        """
        successes = 0
        total_steps = 0
        total_rewards = []
        episode_results = []
        
        print(f"Testing trained AI for {episodes} episodes...")
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_steps = 0
            episode_reward = 0
            
            print(f"\nTest Episode {episode + 1}")
            
            for step in range(500):  # Max steps per episode
                action, _ = ai_agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_steps += 1
                episode_reward += reward
                
                if terminated or truncated:
                    success = info.get('success', False)
                    if success:
                        successes += 1
                        print(f"SUCCESS in {episode_steps} steps! Reward: {episode_reward:.2f}")
                    else:
                        reason = "Time limit reached" if truncated else "Episode ended"
                        print(f"{reason} after {episode_steps} steps. Reward: {episode_reward:.2f}")
                    
                    episode_results.append({
                        'episode': episode + 1,
                        'steps': episode_steps,
                        'reward': episode_reward,
                        'success': success,
                        'terminated': terminated,
                        'truncated': truncated
                    })
                    break
            
            total_steps += episode_steps
            total_rewards.append(episode_reward)
            time.sleep(1)
        
        # Calculate statistics
        success_rate = (successes / episodes) * 100
        avg_steps = total_steps / episodes
        avg_reward = np.mean(total_rewards)
        
        results = {
            'episodes_tested': episodes,
            'successes': successes,
            'success_rate': success_rate,
            'average_steps': avg_steps,
            'average_reward': avg_reward,
            'episode_details': episode_results
        }
        
        print(f"\nTEST RESULTS:")
        print(f"Success Rate: {success_rate:.1f}% ({successes}/{episodes})")
        print(f"Average Steps: {avg_steps:.1f}")
        print(f"Average Reward: {avg_reward:.2f}")
        
        # Provide performance feedback
        if success_rate >= 80:
            print("Excellent performance! Your AI is well-trained.")
        elif success_rate >= 60:
            print("Good performance. Your AI is learning well.")
        elif success_rate >= 40:
            print("Moderate performance. Consider more training or hyperparameter tuning.")
        else:
            print("Poor performance. The AI needs significantly more training.")
        
        return results
    
    def train(self, total_timesteps: int, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Main training method - to be implemented by specific trainers.
        
        Args:
            total_timesteps: Number of training timesteps
            model_name: Name for the saved model
            **kwargs: Additional trainer-specific arguments
            
        Returns:
            Training results dictionary
        """
        print(f"Starting {self.__class__.__name__} training...")
        print(f"Training for {total_timesteps:,} steps")
        
        try:
            # Create environment
            env = self.get_environment()
            
            # Create agent
            ai_agent = self.create_agent(env)
            
            # Run training
            results = self.run_training_session(
                agent=ai_agent,
                total_timesteps=total_timesteps,
                save_path=model_name
            )
            
            print(f"Training complete! AI saved as '{self.models_dir / model_name}'")
            
            # Quick test
            print("\nQuick test of trained model...")
            test_results = self.test_trained_model(ai_agent, env, episodes=3)
            results['test_results'] = test_results
            
            env.close()
            return results
            
        except KeyboardInterrupt:
            print("\nTraining stopped by user")
            try:
                partial_model_path = self.models_dir / f"{model_name}_partial"
                ai_agent.save(str(partial_model_path))
                print(f"Partial training saved as '{partial_model_path}'")
                
                # Test partial model
                print("\nTesting partial model...")
                self.test_trained_model(ai_agent, env, episodes=2)
            except Exception as e:
                print(f"Warning: Could not save partial model: {e}")
            finally:
                if 'env' in locals():
                    env.close()
            return {}
        
        except Exception as e:
            print(f"Training failed: {e}")
            print("Troubleshooting tips:")
            print("   - Make sure Unity is running and the escape cage scene is loaded")
            print("   - Check that no firewall is blocking port 9999")
            print("   - Verify Python has the required packages installed")
            if 'env' in locals():
                env.close()
            raise


class StandardTrainer(BaseTrainer):
    """
    Standard training with balanced reward structure and parameters.
    
    Features:
    - Balanced hyperparameters for stable learning
    - 15-30 minute training time
    - 85-95% expected success rate
    - Comprehensive logging and testing
    """
    
    def get_environment(self):
        """Get SimpleEscapeCageEnv for balanced training."""
        return SimpleEscapeCageEnv()
    
    def create_agent(self, env) -> PPO:
        """Create PPO agent with standard hyperparameters."""
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0003,
            verbose=1
        )


class FastTrainer(BaseTrainer):
    """
    Fast training optimized for quick results and prototyping.
    
    Features:
    - Aggressive hyperparameters for faster convergence
    - 5-10 minute training time
    - 70-85% expected success rate
    - Optimized for rapid development cycles
    """
    
    def get_environment(self):
        """Get FastEscapeCageEnv for accelerated training."""
        return FastEscapeCageEnv()
    
    def create_agent(self, env) -> PPO:
        """Create PPO agent with aggressive hyperparameters for fast learning."""
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=0.001,       # Higher learning rate
            n_steps=512,               # Smaller rollouts
            batch_size=32,             # Smaller batches
            n_epochs=4,                # Fewer epochs
            gamma=0.95,                # Focus on immediate rewards
            gae_lambda=0.9,            # Lower GAE
            clip_range=0.3,            # Higher clip range
            ent_coef=0.01,             # Encourage exploration
            verbose=1
        )
    
    def train(self, total_timesteps: int = 25000, model_name: str = "fast_trained_ai", **kwargs) -> Dict[str, Any]:
        """Train with fast settings - optimized for 5-10 minute runtime."""
        print("FAST TRAINING MODE")
        print("This version should show results in 5-10 minutes!")
        print("Watch for key pickups and escapes!")
        
        return super().train(total_timesteps, model_name, **kwargs)


class ContinueTrainer(BaseTrainer):
    """
    Continue training existing models with additional timesteps or fine-tuning.
    
    Features:
    - Load and continue from existing models
    - Incremental learning capabilities
    - Hyperparameter adjustment
    - Backup and versioning support
    """
    
    def get_environment(self):
        """Get SimpleEscapeCageEnv for continued training."""
        return SimpleEscapeCageEnv()
    
    def create_agent(self, env) -> PPO:
        """This will be overridden when loading existing model."""
        return PPO("MlpPolicy", env, learning_rate=0.0003, verbose=1)
    
    def find_model_file(self, model_name: str) -> Optional[Path]:
        """Find a model file by searching in common locations."""
        search_paths = [
            self.models_dir / f"{model_name}.zip",
            self.models_dir / model_name,
            Path(f"{model_name}.zip"),
            Path(model_name),
            Path(".") / f"{model_name}.zip",
            Path(".") / model_name
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        return None
    
    def list_available_models(self) -> List[str]:
        """List all available saved models."""
        models = []
        
        # Check current directory
        for file in Path(".").glob("*.zip"):
            models.append(file.stem)
        
        # Check models directory
        if self.models_dir.exists():
            for file in self.models_dir.glob("*.zip"):
                models.append(file.stem)
        
        return sorted(set(models))
    
    def continue_training(self, model_name: str, additional_steps: int = 25000, 
                         save_backup: bool = True) -> Dict[str, Any]:
        """
        Continue training from a saved model.
        
        Args:
            model_name: Name of the model to continue training
            additional_steps: Number of additional training steps
            save_backup: Whether to create a backup of the original model
            
        Returns:
            Training results dictionary
        """
        print(f"Loading AI model: {model_name}")
        print(f"Will train for {additional_steps:,} additional steps")
        
        # Find the model file
        model_path = self.find_model_file(model_name)
        if not model_path:
            print(f"Model '{model_name}' not found!")
            print("Available models:")
            available_models = self.list_available_models()
            if available_models:
                for model in available_models:
                    print(f"   - {model}")
            else:
                print("   - No models found")
            print("\nTrain a model first using: python escape_cage_trainer.py")
            return {}
        
        try:
            # Create environment
            env = self.get_environment()
            
            # Load the existing model
            print(f"Loading model from: {model_path}")
            ai_agent = PPO.load(str(model_path), env=env)
            print(f"Model '{model_name}' loaded successfully!")
            
            # Create backup if requested
            if save_backup:
                try:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_name = f"{model_name}_backup_{timestamp}"
                    backup_path = self.models_dir / f"{backup_name}.zip"
                    
                    shutil.copy2(str(model_path), str(backup_path))
                    print(f"Backup created: {backup_path}")
                except Exception as e:
                    print(f"Warning: Could not create backup: {e}")
            
            print("Continuing training from where we left off...")
            
            # Continue training
            start_time = time.time()
            ai_agent.learn(total_timesteps=additional_steps)
            training_time = time.time() - start_time
            
            # Save the improved model
            improved_name = f"{model_name}_improved"
            improved_path = self.models_dir / improved_name
            ai_agent.save(str(improved_path))
            print(f"Training complete! Improved AI saved as '{improved_path}'")
            print(f"Training took {training_time:.1f} seconds")
            
            # Test the improved model
            print("\nTesting improved model...")
            test_results = self.test_trained_model(ai_agent, env, episodes=5)
            
            results = {
                'original_model': model_name,
                'improved_model': improved_name,
                'additional_steps': additional_steps,
                'training_time': training_time,
                'test_results': test_results
            }
            
            env.close()
            return results
            
        except Exception as e:
            print(f"Training failed: {e}")
            print("Troubleshooting tips:")
            print("   - Ensure Unity is running with the escape cage scene")
            print("   - Check that the model file is not corrupted")
            print("   - Verify sufficient disk space for saving models")
            if 'env' in locals():
                env.close()
            return {}
    
    def fine_tune(self, model_name: str, fine_tune_steps: int = 15000) -> Dict[str, Any]:
        """
        Fine-tune with conservative hyperparameters.
        
        Args:
            model_name: Name of the model to fine-tune
            fine_tune_steps: Number of fine-tuning steps
            
        Returns:
            Fine-tuning results dictionary
        """
        print(f"Fine-tuning AI model: {model_name}")
        
        model_path = self.find_model_file(model_name)
        if not model_path:
            print(f"Model '{model_name}' not found!")
            return {}
        
        try:
            env = self.get_environment()
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
            finetuned_path = self.models_dir / finetuned_name
            ai_agent.save(str(finetuned_path))
            print(f"Fine-tuning complete! AI saved as '{finetuned_path}'")
            print(f"Fine-tuning took {training_time:.1f} seconds")
            
            # Test the fine-tuned model
            print("\nTesting fine-tuned model...")
            test_results = self.test_trained_model(ai_agent, env, episodes=5)
            
            results = {
                'original_model': model_name,
                'finetuned_model': finetuned_name,
                'fine_tune_steps': fine_tune_steps,
                'training_time': training_time,
                'test_results': test_results
            }
            
            env.close()
            return results
            
        except Exception as e:
            print(f"Fine-tuning failed: {e}")
            if 'env' in locals():
                env.close()
            return {}


class ImprovedTrainer(BaseTrainer):
    """
    Enhanced training with advanced techniques and comprehensive analytics.
    
    Features:
    - Advanced hyperparameters and techniques
    - Comprehensive logging and monitoring
    - Adaptive training strategies
    - Performance tracking and analysis
    """
    
    def __init__(self, models_dir: str = "models"):
        super().__init__(models_dir)
        self.session_name: Optional[str] = None
        
    def get_environment(self):
        """Get environment based on training type."""
        return SimpleEscapeCageEnv()
    
    def create_agent(self, env) -> PPO:
        """Create PPO agent with advanced configuration."""
        return self.create_ppo_agent(env, training_type="standard")
    
    def train_with_analytics(self, training_type: str = "standard", 
                            total_timesteps: int = 50000,
                            model_name: str = "escape_ai_improved",
                            session_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training function with improved architecture and analytics.
        
        Args:
            training_type: Type of training ("standard" or "fast")
            total_timesteps: Number of training timesteps
            model_name: Name for the saved model
            session_name: Custom session name for logging
            
        Returns:
            Comprehensive training results
        """
        # Setup logging
        try:
            from logger_setup import setup_logging
            logger_manager = setup_logging(
                session_name=session_name or f"{training_type}_training",
                log_level="INFO"
            )
            logger = logger_manager.logger
        except ImportError:
            logger = self.logger
            logger_manager = None
        
        try:
            logger.info(f"Starting {training_type} training for {total_timesteps:,} timesteps")
            
            # Create environment based on training type
            if training_type == "fast":
                env = FastEscapeCageEnv()
                logger.info("Using FastEscapeCageEnv for accelerated training")
            else:
                env = SimpleEscapeCageEnv()
                logger.info("Using SimpleEscapeCageEnv for balanced training")
            
            # Create PPO agent
            agent = self.create_ppo_agent(env, training_type=training_type)
            
            # Run training session
            training_results = self.run_training_session(
                agent=agent,
                total_timesteps=total_timesteps,
                save_path=model_name,
                eval_env=None,
                eval_freq=10000
            )
            
            # Log training results
            logger.info("Training Results Summary:")
            for key, value in training_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"   {key}: {value:.3f}")
                else:
                    logger.info(f"   {key}: {value}")
            
            # Save model metadata if logger manager available
            if logger_manager and hasattr(logger_manager, 'log_model_save'):
                logger_manager.log_model_save(
                    model_name=model_name,
                    episode=training_results.get('total_episodes', 0),
                    performance_metrics=training_results
                )
            
            logger.info("Training completed successfully!")
            logger.info(f"Model saved as: {model_name}")
            
            # Quick evaluation
            model_path = self.models_dir / f"{model_name}.zip"
            if model_path.exists():
                logger.info("Running quick evaluation...")
                eval_results = self.evaluate_model(
                    model_path=str(model_path),
                    env=env,
                    num_episodes=5
                )
                
                logger.info("Evaluation Results:")
                logger.info(f"   Success Rate: {eval_results.get('success_rate', 0):.1f}%")
                logger.info(f"   Average Reward: {eval_results.get('average_reward', 0):.2f}")
                
                training_results['evaluation'] = eval_results
            
            env.close()
            return training_results
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return {}
        except Exception as e:
            if logger_manager and hasattr(logger_manager, 'log_error'):
                logger_manager.log_error(f"Training failed: {e}", "TRAINING_ERROR", e)
            else:
                logger.error(f"Training failed: {e}")
            raise
        finally:
            # Close environment
            if 'env' in locals():
                env.close()
            
            # Close logger
            if logger_manager and hasattr(logger_manager, 'close'):
                logger_manager.close()
    
    def evaluate_model(self, model_path: str, env, num_episodes: int = 5) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to the model file
            env: Environment for evaluation
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation results dictionary
        """
        try:
            model = PPO.load(model_path, env=env)
            return self.test_trained_model(model, env, episodes=num_episodes)
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return {'success_rate': 0, 'average_reward': 0, 'error': str(e)}


def load_and_test_model(model_name: str = "trained_escape_ai", episodes: int = 5):
    """
    Load a saved model and test it.
    
    Args:
        model_name: Name of the model to load
        episodes: Number of test episodes
    """
    trainer = StandardTrainer()
    
    try:
        print(f"Loading model: {model_name}")
        
        # Create environment
        env = SimpleEscapeCageEnv()
        
        # Try to load from models directory first, then current directory
        model_paths = [
            trainer.models_dir / model_name,
            Path(model_name),
            trainer.models_dir / f"{model_name}.zip",
            Path(f"{model_name}.zip")
        ]
        
        model_loaded = False
        for path in model_paths:
            try:
                ai_agent = PPO.load(str(path), env=env)
                print(f"Model loaded from: {path}")
                model_loaded = True
                break
            except FileNotFoundError:
                continue
        
        if not model_loaded:
            print(f"Model '{model_name}' not found in any of these locations:")
            for path in model_paths:
                print(f"  - {path}")
            return
        
        # Test the model
        trainer.test_trained_model(ai_agent, env, episodes)
        env.close()
        
    except Exception as e:
        print(f"Error loading or testing model: {e}")


def main():
    """Main function with comprehensive training options."""
    parser = argparse.ArgumentParser(description='AI Escape Cage Trainer - All Training Methods')
    
    # Training type selection
    parser.add_argument('--trainer', '-t', 
                       choices=['standard', 'fast', 'continue', 'improved'],
                       default='standard',
                       help='Type of trainer to use (default: standard)')
    
    # Common arguments
    parser.add_argument('--steps', '-s', type=int, default=50000,
                       help='Number of training steps (default: 50000)')
    parser.add_argument('--model', '-m', default='trained_escape_ai',
                       help='Model name to save/load (default: trained_escape_ai)')
    parser.add_argument('--episodes', '-e', type=int, default=5,
                       help='Number of test episodes (default: 5)')
    
    # Action selection
    parser.add_argument('--test', action='store_true',
                       help='Test existing model instead of training')
    parser.add_argument('--list-models', action='store_true',
                       help='List available models')
    
    # Continue trainer specific
    parser.add_argument('--continue-steps', type=int, default=25000,
                       help='Additional steps for continue training (default: 25000)')
    parser.add_argument('--fine-tune', action='store_true',
                       help='Fine-tune with conservative settings (continue trainer)')
    parser.add_argument('--fine-steps', type=int, default=15000,
                       help='Steps for fine-tuning (default: 15000)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup (continue trainer)')
    
    # Improved trainer specific
    parser.add_argument('--session-name', type=str,
                       help='Custom session name for logging (improved trainer)')
    
    args = parser.parse_args()
    
    # Handle list models
    if args.list_models:
        trainer = ContinueTrainer()
        print("Available models:")
        models = trainer.list_available_models()
        if models:
            for model in models:
                print(f"   - {model}")
        else:
            print("   - No models found")
        return
    
    # Handle test existing model
    if args.test:
        load_and_test_model(args.model, args.episodes)
        return
    
    # Create appropriate trainer
    if args.trainer == 'standard':
        trainer = StandardTrainer()
        print("AI Escape Cage Trainer - Standard Training")
        print("Balanced parameters for stable, production-ready models")
        trainer.train(args.steps, args.model)
        
    elif args.trainer == 'fast':
        trainer = FastTrainer()
        print("AI Escape Cage Trainer - Fast Training")
        print("Optimized for quick results and prototyping")
        trainer.train(args.steps, args.model)
        
    elif args.trainer == 'continue':
        trainer = ContinueTrainer()
        print("AI Escape Cage Trainer - Continue Training")
        print("Load and improve existing models")
        
        if args.fine_tune:
            trainer.fine_tune(args.model, args.fine_steps)
        else:
            trainer.continue_training(args.model, args.continue_steps, not args.no_backup)
            
    elif args.trainer == 'improved':
        trainer = ImprovedTrainer()
        print("AI Escape Cage Trainer - Improved Training")
        print("Enhanced training with advanced techniques and analytics")
        
        # Determine training type based on steps
        training_type = "fast" if args.steps < 40000 else "standard"
        
        trainer.train_with_analytics(
            training_type=training_type,
            total_timesteps=args.steps,
            model_name=args.model,
            session_name=args.session_name
        )


if __name__ == "__main__":
    main()