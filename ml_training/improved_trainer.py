"""
🚀 Improved Escape Cage Trainer
Enhanced training script using modular components and comprehensive logging.

This script demonstrates the improved architecture with:
- Modular environment base classes
- Comprehensive logging system
- Advanced training utilities
- Performance monitoring
"""

import sys
import os
from typing import Optional
import argparse

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from base_environment import StandardEscapeCageEnv, FastEscapeCageEnv
from logger_setup import setup_logging
from training_utils import create_ppo_agent, run_training_session, evaluate_model


def train_escape_ai(
    training_type: str = "standard",
    total_timesteps: int = 50000,
    model_name: str = "escape_ai_improved",
    session_name: Optional[str] = None
):
    """
    Main training function with improved architecture.
    
    Args:
        training_type: Type of training ("standard" or "fast")
        total_timesteps: Number of training timesteps
        model_name: Name for the saved model
        session_name: Custom session name for logging
    """
    
    # Setup logging
    logger_manager = setup_logging(
        session_name=session_name or f"{training_type}_training",
        log_level="INFO"
    )
    logger = logger_manager.logger
    
    try:
        logger.info(f"🚀 Starting {training_type} training for {total_timesteps:,} timesteps")
        
        # Create environment based on training type
        if training_type == "fast":
            env = FastEscapeCageEnv()
            logger.info("⚡ Using FastEscapeCageEnv for accelerated training")
        else:
            env = StandardEscapeCageEnv()
            logger.info("🎯 Using StandardEscapeCageEnv for balanced training")
        
        # Create PPO agent
        agent = create_ppo_agent(env, training_type=training_type)
        
        # Run training session
        training_results = run_training_session(
            agent=agent,
            total_timesteps=total_timesteps,
            save_path=model_name,
            eval_env=None,  # Could add evaluation environment here
            eval_freq=10000
        )
        
        # Log training results
        logger.info("📊 Training Results Summary:")
        for key, value in training_results.items():
            if isinstance(value, (int, float)):
                logger.info(f"   {key}: {value:.3f}")
            else:
                logger.info(f"   {key}: {value}")
        
        # Save model metadata
        logger_manager.log_model_save(
            model_name=model_name,
            episode=training_results.get('total_episodes', 0),
            performance_metrics=training_results
        )
        
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"💾 Model saved as: {model_name}")
        
        # Quick evaluation
        if os.path.exists(f"{model_name}.zip"):
            logger.info("🧪 Running quick evaluation...")
            eval_results = evaluate_model(
                model_path=model_name,
                env=env,
                num_episodes=5
            )
            
            logger.info("📈 Evaluation Results:")
            logger.info(f"   Success Rate: {eval_results.get('success_rate', 0):.1f}%")
            logger.info(f"   Average Reward: {eval_results.get('average_reward', 0):.2f}")
        
        return training_results
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        return {}
    except Exception as e:
        logger_manager.log_error(f"Training failed: {e}", "TRAINING_ERROR", e)
        raise
    finally:
        # Close environment
        if 'env' in locals():
            env.close()
        
        # Close logger
        logger_manager.close()


def main():
    """Command line interface for the improved trainer."""
    parser = argparse.ArgumentParser(description="🤖 Improved Escape Cage AI Trainer")
    
    parser.add_argument(
        "--type",
        choices=["standard", "fast"],
        default="standard",
        help="Training type (default: standard)"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Number of training timesteps (default: 50000)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="escape_ai_improved",
        help="Name for the saved model (default: escape_ai_improved)"
    )
    
    parser.add_argument(
        "--session-name",
        type=str,
        help="Custom session name for logging"
    )
    
    args = parser.parse_args()
    
    print("🤖 Improved Escape Cage AI Trainer")
    print("=" * 50)
    print(f"Training Type: {args.type}")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Model Name: {args.model_name}")
    print(f"Session Name: {args.session_name or 'auto-generated'}")
    print("=" * 50)
    
    # Run training
    train_escape_ai(
        training_type=args.type,
        total_timesteps=args.timesteps,
        model_name=args.model_name,
        session_name=args.session_name
    )


if __name__ == "__main__":
    main() 