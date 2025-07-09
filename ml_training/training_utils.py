"""
🛠️ Training Utilities Module
Comprehensive utilities and helper functions for ML training pipeline.

This module provides essential tools for training reinforcement learning models:

Core Components:
- PPO model creation with optimized hyperparameters for different training scenarios
- Advanced training callback functions with detailed metrics tracking
- Model evaluation utilities with comprehensive performance analysis
- Environment performance monitoring and benchmarking tools
- Complete training session management with error handling and progress tracking

Key Classes:
- TrainingMetricsCallback: Custom callback for detailed training metrics collection
  Tracks episode rewards, lengths, success rates, learning progress, and timing

Utility Functions:
- create_ppo_agent(): Factory function for creating optimized PPO agents
- run_training_session(): Complete training workflow with monitoring and saving
- evaluate_model(): Comprehensive model evaluation with statistical analysis
- benchmark_environment(): Performance benchmarking for environment validation
- create_training_schedule(): Advanced training schedule creation for curriculum learning

Usage Examples:
    # Create a fast training agent
    agent = create_ppo_agent(env, training_type="fast")
    
    # Run complete training session with metrics
    results = run_training_session(
        agent, 
        total_timesteps=100000,
        save_path="trained_model.zip"
    )
    
    # Evaluate trained model
    eval_results = evaluate_model("trained_model.zip", env, num_episodes=20)

Note: All functions include comprehensive error handling and logging for robust operation.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, List, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import logging
import time
import datetime
from pathlib import Path


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback to track and log training metrics.
    
    This callback monitors:
    - Episode rewards and lengths
    - Success rates
    - Learning progress
    - Performance metrics
    """
    
    def __init__(self, 
                 logger: Optional[logging.Logger] = None,
                 log_interval: int = 100,
                 verbose: int = 0):
        super().__init__(verbose)
        self.logger = logger or logging.getLogger('training_metrics')
        self.log_interval = log_interval
        
        # Metrics tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_count = 0
        self.total_episodes = 0
        
        # Timing
        self.start_time = None
        self.last_log_time = None
    
    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.logger.info("🏋️ Training started with metrics callback")
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Check if episode is done
        if self.locals.get('done', False):
            # Extract episode info
            info = self.locals.get('info', {})
            if isinstance(info, list) and len(info) > 0:
                info = info[0]  # Take first environment's info
            
            # Record episode metrics
            if 'episode' in info:
                episode_info = info['episode']
                self.episode_rewards.append(episode_info['r'])
                self.episode_lengths.append(episode_info['l'])
                
                # Track success
                if info.get('success', False):
                    self.success_count += 1
                
                self.total_episodes += 1
                
                # Log periodically
                if self.total_episodes % self.log_interval == 0:
                    self._log_metrics()
        
        return True
    
    def _log_metrics(self) -> None:
        """Log current training metrics."""
        if not self.episode_rewards:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate statistics
        recent_rewards = self.episode_rewards[-self.log_interval:]
        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(self.episode_lengths[-self.log_interval:])
        success_rate = (self.success_count / self.total_episodes) * 100 if self.total_episodes > 0 else 0
        
        # Calculate FPS
        steps_since_last = self.log_interval * np.mean(self.episode_lengths[-self.log_interval:])
        time_since_last = current_time - self.last_log_time
        fps = steps_since_last / time_since_last if time_since_last > 0 else 0
        
        # Log metrics
        self.logger.info(
            f"Episodes: {self.total_episodes:4d} | "
            f"Avg Reward: {avg_reward:7.2f} | "
            f"Avg Length: {avg_length:5.1f} | "
            f"Success Rate: {success_rate:5.1f}% | "
            f"FPS: {fps:5.1f} | "
            f"Elapsed: {elapsed_time/3600:.1f}h"
        )
        
        self.last_log_time = current_time
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_episodes': self.total_episodes,
            'total_successes': self.success_count,
            'success_rate': (self.success_count / self.total_episodes) * 100 if self.total_episodes > 0 else 0,
            'average_reward': np.mean(self.episode_rewards),
            'average_length': np.mean(self.episode_lengths),
            'best_reward': np.max(self.episode_rewards),
            'worst_reward': np.min(self.episode_rewards),
            'reward_std': np.std(self.episode_rewards),
            'total_training_time': time.time() - self.start_time if self.start_time else 0
        }


def create_ppo_agent(
    env,
    training_type: str = "standard",
    custom_params: Optional[Dict[str, Any]] = None
) -> PPO:
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
    agent = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        **params
    )
    
    logger = logging.getLogger('training_utils')
    logger.info(f"Created PPO agent with {training_type} configuration")
    logger.debug(f"Hyperparameters: {params}")
    
    return agent


def run_training_session(
    agent: PPO,
    total_timesteps: int,
    save_path: Optional[str] = None,
    eval_env=None,
    eval_freq: int = 10000,
    callbacks: Optional[List[BaseCallback]] = None
) -> Dict[str, Any]:
    """
    Run a complete training session with monitoring and saving.
    
    Args:
        agent: PPO agent to train
        total_timesteps: Total number of training timesteps
        save_path: Path to save the trained model
        eval_env: Environment for evaluation (optional)
        eval_freq: Frequency of evaluation episodes
        callbacks: Additional callbacks for training
        
    Returns:
        Dictionary containing training results and metrics
    """
    logger = logging.getLogger('training_utils')
    
    # Prepare callbacks
    all_callbacks = callbacks or []
    
    # Add metrics callback
    metrics_callback = TrainingMetricsCallback(logger)
    all_callbacks.append(metrics_callback)
    
    # Add evaluation callback if eval environment provided
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(Path(save_path).parent / "best_model") if save_path else None,
            log_path=str(Path(save_path).parent / "eval_logs") if save_path else None,
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        all_callbacks.append(eval_callback)
    
    # Start training
    start_time = time.time()
    logger.info(f"🚀 Starting training for {total_timesteps:,} timesteps")
    
    try:
        agent.learn(
            total_timesteps=total_timesteps,
            callback=all_callbacks
        )
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time/3600:.2f} hours")
        
        # Save model if path provided
        if save_path:
            agent.save(save_path)
            logger.info(f"💾 Model saved to: {save_path}")
        
        # Get final metrics
        final_metrics = metrics_callback.get_metrics_summary()
        final_metrics['total_training_time'] = training_time
        final_metrics['timesteps_per_second'] = total_timesteps / training_time
        
        return final_metrics
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        logger.info("⏹️ Training interrupted by user")
        
        # Save partial model
        if save_path:
            partial_save_path = save_path + "_partial"
            agent.save(partial_save_path)
            logger.info(f"💾 Partial model saved to: {partial_save_path}")
        
        # Get partial metrics
        partial_metrics = metrics_callback.get_metrics_summary()
        partial_metrics['total_training_time'] = training_time
        partial_metrics['interrupted'] = True
        
        return partial_metrics


def evaluate_model(
    model_path: str,
    env,
    num_episodes: int = 10,
    render: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a trained model on the given environment.
    
    Args:
        model_path: Path to the saved model
        env: Environment for evaluation
        num_episodes: Number of episodes to run
        render: Whether to render the episodes
        
    Returns:
        Dictionary containing evaluation results
    """
    logger = logging.getLogger('training_utils')
    
    # Load model
    try:
        model = PPO.load(model_path)
        logger.info(f"📋 Loaded model from: {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return {}
    
    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    logger.info(f"🧪 Starting evaluation for {num_episodes} episodes")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Check for success
        if info.get('success', False):
            success_count += 1
        
        logger.info(f"Episode {episode + 1:2d}: Reward={episode_reward:7.2f}, "
                   f"Length={episode_length:3d}, Success={info.get('success', False)}")
    
    # Calculate statistics
    results = {
        'num_episodes': num_episodes,
        'average_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'average_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': (success_count / num_episodes) * 100,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }
    
    logger.info(f"📊 Evaluation Results:")
    logger.info(f"   Average Reward: {results['average_reward']:.2f} ± {results['std_reward']:.2f}")
    logger.info(f"   Success Rate: {results['success_rate']:.1f}%")
    logger.info(f"   Average Length: {results['average_length']:.1f} steps")
    
    return results


def benchmark_environment(env, num_steps: int = 1000) -> Dict[str, float]:
    """
    Benchmark environment performance.
    
    Args:
        env: Environment to benchmark
        num_steps: Number of steps to run for benchmarking
        
    Returns:
        Dictionary containing performance metrics
    """
    logger = logging.getLogger('training_utils')
    
    logger.info(f"⏱️ Benchmarking environment performance for {num_steps} steps")
    
    # Reset timing
    start_time = time.time()
    env.reset()
    reset_time = time.time() - start_time
    
    # Step timing
    start_time = time.time()
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            env.reset()
    
    step_time = time.time() - start_time
    
    results = {
        'reset_time_ms': reset_time * 1000,
        'total_step_time_s': step_time,
        'average_step_time_ms': (step_time / num_steps) * 1000,
        'steps_per_second': num_steps / step_time
    }
    
    logger.info(f"📈 Benchmark Results:")
    logger.info(f"   Reset Time: {results['reset_time_ms']:.2f} ms")
    logger.info(f"   Avg Step Time: {results['average_step_time_ms']:.2f} ms")
    logger.info(f"   Steps/Second: {results['steps_per_second']:.1f}")
    
    return results


def create_training_schedule(
    total_timesteps: int,
    phases: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Create a training schedule with different phases.
    
    Args:
        total_timesteps: Total timesteps to distribute across phases
        phases: Custom phase definitions
        
    Returns:
        List of training phases with timestep allocations
    """
    if phases is None:
        # Default 3-phase training schedule
        phases = [
            {
                'name': 'exploration',
                'proportion': 0.3,
                'learning_rate_multiplier': 1.5,
                'ent_coef_multiplier': 2.0,
                'description': 'High exploration phase'
            },
            {
                'name': 'learning',
                'proportion': 0.5,
                'learning_rate_multiplier': 1.0,
                'ent_coef_multiplier': 1.0,
                'description': 'Balanced learning phase'
            },
            {
                'name': 'refinement',
                'proportion': 0.2,
                'learning_rate_multiplier': 0.5,
                'ent_coef_multiplier': 0.5,
                'description': 'Fine-tuning phase'
            }
        ]
    
    # Calculate timesteps for each phase
    schedule = []
    remaining_timesteps = total_timesteps
    
    for i, phase in enumerate(phases):
        if i == len(phases) - 1:  # Last phase gets remaining timesteps
            phase_timesteps = remaining_timesteps
        else:
            phase_timesteps = int(total_timesteps * phase['proportion'])
            remaining_timesteps -= phase_timesteps
        
        schedule.append({
            **phase,
            'timesteps': phase_timesteps,
            'start_step': total_timesteps - remaining_timesteps - phase_timesteps,
            'end_step': total_timesteps - remaining_timesteps
        })
    
    return schedule 