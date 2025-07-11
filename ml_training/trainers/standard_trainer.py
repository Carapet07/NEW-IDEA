"""
Standard Trainer Module

Standard training with balanced reward structure and parameters.
Provides stable, production-ready model training.
"""

import sys
from pathlib import Path

# Add both parent directory and project root to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
project_root = parent_dir.parent

sys.path.insert(0, str(parent_dir))  # ml_training directory
sys.path.insert(0, str(project_root))  # project root

from .base_trainer import BaseTrainer


class StandardTrainer(BaseTrainer):
    """
    Standard training with balanced reward structure and parameters.
    
    Features:
    - Balanced hyperparameters for stable learning
    - 15-30 minute training time
    - 85-95% expected success rate
    - Comprehensive logging and testing
    - Configurable environment selection (simple, fast, test)
    """
    
    def get_training_type(self) -> str:
        """Get training type identifier."""
        return "standard"
    
    def get_default_environment_type(self) -> str:
        """Get default environment type for standard trainer."""
        return "simple"  # Conservative choice for production-ready models 