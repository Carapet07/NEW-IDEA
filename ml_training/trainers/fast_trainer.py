"""
Fast Trainer Module

Fast training optimized for quick results and prototyping.
Uses aggressive hyperparameters for rapid convergence.
"""

from typing import Dict, Any

import sys
from pathlib import Path

# Add both parent directory and project root to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
project_root = parent_dir.parent

sys.path.insert(0, str(parent_dir))  # ml_training directory
sys.path.insert(0, str(project_root))  # project root

from .base_trainer import BaseTrainer


class FastTrainer(BaseTrainer):
    """
    Fast training optimized for quick results and prototyping.
    
    Features:
    - Aggressive hyperparameters for faster convergence
    - 5-10 minute training time
    - 70-85% expected success rate
    - Optimized for rapid development cycles
    - Configurable environment selection (simple, fast, test)
    """
    
    def get_training_type(self) -> str:
        """Get training type identifier."""
        return "fast"
    
    def get_default_environment_type(self) -> str:
        """Get default environment type for fast trainer."""
        return "fast"  # Aggressive choice for rapid prototyping
    
    def train(self, total_timesteps: int = 25000, model_name: str = "fast_trained_ai", 
              custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train with fast settings - optimized for 5-10 minute runtime.
        
        Args:
            total_timesteps: Number of training timesteps (default: 25000)
            model_name: Name for saved model (default: "fast_trained_ai")
            custom_params: Custom hyperparameters
            
        Returns:
            Training results dictionary
        """
        print("FAST TRAINING MODE")
        print(f"Environment: {self.environment_type}")
        print("This version should show results in 5-10 minutes!")
        print("Watch for key pickups and escapes!")
        
        return super().train(total_timesteps, model_name, custom_params) 