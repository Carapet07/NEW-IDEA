"""
Trainer Factory Module

Factory pattern for creating trainer instances based on training type.
Provides centralized trainer instantiation and configuration.
"""

from typing import Dict, Type

from .base_trainer import BaseTrainer
from .standard_trainer import StandardTrainer
from .fast_trainer import FastTrainer
from .continue_trainer import ContinueTrainer


# Registry of available trainers
TRAINER_REGISTRY: Dict[str, Type[BaseTrainer]] = {
    'standard': StandardTrainer,
    'fast': FastTrainer,
    'continue': ContinueTrainer,
}


def create_trainer(trainer_type: str, **kwargs) -> BaseTrainer:
    """
    Create a trainer instance based on the specified type.
    
    Args:
        trainer_type: Type of trainer to create ("standard", "fast", "continue")
        **kwargs: Additional arguments to pass to trainer constructor.
                 All trainers support environment_type ("simple", "fast", "test")
                 and models_dir parameters.
        
    Returns:
        Trainer instance of the specified type
        
    Raises:
        ValueError: If trainer_type is not recognized
        
    Examples:
        # Standard trainer with default simple environment
        trainer = create_trainer("standard")
        
        # Standard trainer with fast environment (aggressive rewards!)
        trainer = create_trainer("standard", environment_type="fast")
        
        # Fast trainer with default fast environment  
        trainer = create_trainer("fast")
        
        # Fast trainer with simple environment (fast learning + simple rewards)
        trainer = create_trainer("fast", environment_type="simple")
        
        # Continue trainer with fast environment (continue existing model in aggressive rewards)
        trainer = create_trainer("continue", environment_type="fast")
        
        # Any trainer with debug environment (slow, detailed logging)
        trainer = create_trainer("standard", environment_type="debug")
        trainer = create_trainer("fast", environment_type="debug")
        trainer = create_trainer("continue", environment_type="debug")
    """
    if trainer_type not in TRAINER_REGISTRY:
        available_types = list(TRAINER_REGISTRY.keys())
        raise ValueError(f"Unknown trainer type '{trainer_type}'. "
                        f"Available types: {available_types}")
    
    trainer_class = TRAINER_REGISTRY[trainer_type]
    return trainer_class(**kwargs)


def get_available_trainer_types() -> list[str]:
    """
    Get list of available trainer types.
    
    Returns:
        List of available trainer type names
    """
    return list(TRAINER_REGISTRY.keys())


def get_available_environments() -> list[str]:
    """
    Get list of available environment types for all trainers.
    
    Returns:
        List of available environment type names
    """
    return ["simple", "fast", "debug"]


def register_trainer(name: str, trainer_class: Type[BaseTrainer]) -> None:
    """
    Register a new trainer type.
    
    Args:
        name: Name for the trainer type
        trainer_class: Trainer class to register
        
    Raises:
        ValueError: If trainer_class is not a BaseTrainer subclass
    """
    if not issubclass(trainer_class, BaseTrainer):
        raise ValueError(f"Trainer class must be a subclass of BaseTrainer, "
                        f"got {trainer_class}")
    
    TRAINER_REGISTRY[name] = trainer_class 