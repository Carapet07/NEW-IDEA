"""
Hyperparameter Configuration Module

Centralized hyperparameter management for different training strategies.
Provides type-safe configuration classes and validation.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class HyperparameterConfig:
    """
    Type-safe hyperparameter configuration.
    
    Attributes:
        learning_rate: Learning rate for the optimizer
        n_steps: Number of steps per rollout
        batch_size: Batch size for training
        n_epochs: Number of training epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm for clipping
    """
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for PPO agent creation."""
        return {
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm
        }


# Predefined hyperparameter configurations
HYPERPARAMETER_CONFIGS = {
    "standard": HyperparameterConfig(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5
    ),
    "fast": HyperparameterConfig(
        learning_rate=1e-3,
        n_steps=512,
        batch_size=32,
        n_epochs=4,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=0.3,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5
    ),
    "conservative": HyperparameterConfig(
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=20,
        gamma=0.995,
        gae_lambda=0.98,
        clip_range=0.1,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.3
    )
}


def get_hyperparameters(training_type: str, 
                       custom_params: Optional[Dict[str, Any]] = None) -> HyperparameterConfig:
    """
    Get hyperparameter configuration for a training type.
    
    Args:
        training_type: Type of training ("standard", "fast", or "conservative")
        custom_params: Custom parameters to override defaults
        
    Returns:
        HyperparameterConfig with the specified or default parameters
        
    Raises:
        ValueError: If training_type is not recognized
    """
    if training_type not in HYPERPARAMETER_CONFIGS:
        available_types = list(HYPERPARAMETER_CONFIGS.keys())
        raise ValueError(f"Unknown training type '{training_type}'. "
                        f"Available types: {available_types}")
    
    # Get base configuration
    config = HYPERPARAMETER_CONFIGS[training_type]
    
    # Apply custom parameters if provided
    if custom_params:
        config_dict = config.to_dict()
        config_dict.update(custom_params)
        config = HyperparameterConfig(**config_dict)
    
    return config


def validate_hyperparameters(config: HyperparameterConfig) -> None:
    """
    Validate hyperparameter configuration.
    
    Args:
        config: Hyperparameter configuration to validate
        
    Raises:
        ValueError: If any parameter is invalid
    """
    if config.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    if config.n_steps <= 0:
        raise ValueError("Number of steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if config.n_epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    if not 0 <= config.gamma <= 1:
        raise ValueError("Gamma must be between 0 and 1")
    if not 0 <= config.gae_lambda <= 1:
        raise ValueError("GAE lambda must be between 0 and 1")
    if config.clip_range <= 0:
        raise ValueError("Clip range must be positive")
    if config.ent_coef < 0:
        raise ValueError("Entropy coefficient must be non-negative")
    if config.vf_coef < 0:
        raise ValueError("Value function coefficient must be non-negative")
    if config.max_grad_norm <= 0:
        raise ValueError("Max gradient norm must be positive") 