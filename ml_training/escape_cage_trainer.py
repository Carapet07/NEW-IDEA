"""
AI Escape Cage Trainer - Streamlined Training CLI

Focused command-line interface for different training strategies.
Delegates to specialized trainer classes for clean separation of concerns.
"""

import argparse
import sys
from typing import Dict, Any

from trainers import create_trainer, get_available_trainer_types
from trainers.trainer_factory import get_available_environments
from trainers.continue_trainer import ContinueTrainer
# Testing functionality now integrated into test_system.py


def load_and_test_model(model_name: str = "trained_escape_ai", episodes: int = 5):
    """
    Load a saved model and test it using the comprehensive test system.
    
    Args:
        model_name: Name of the model to load
        episodes: Number of test episodes
    """
    print(f"⚠️  Individual model testing moved to comprehensive test system")
    print(f"   Run: python test_system.py --component models")
    print(f"   Or:  python test_system.py --verbose  # Full test suite")


def main():
    """Main function with streamlined training options."""
    parser = argparse.ArgumentParser(description='AI Escape Cage Trainer - Streamlined Interface')
    
    # Training type selection
    available_trainers = get_available_trainer_types()
    parser.add_argument('--trainer', '-t', 
                       choices=available_trainers,
                       default='standard',
                       help=f'Type of trainer to use (default: standard). Available: {available_trainers}')
    
    # Environment selection (now works with all trainers!)
    available_environments = get_available_environments()
    parser.add_argument('--environment', '-env',
                       choices=available_environments,
                       help=f'Training environment type. simple=basic rewards, fast=enhanced rewards, debug=detailed logging. '
                            f'Defaults: standard->simple, fast->fast, continue->simple')
    
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
    parser.add_argument('--compare', nargs=2, metavar=('MODEL1', 'MODEL2'),
                       help='Compare two models')
    
    # Continue trainer specific
    parser.add_argument('--continue-steps', type=int, default=25000,
                       help='Additional steps for continue training (default: 25000)')
    parser.add_argument('--fine-tune', action='store_true',
                       help='Fine-tune with conservative settings (continue trainer)')
    parser.add_argument('--fine-steps', type=int, default=15000,
                       help='Steps for fine-tuning (default: 15000)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup (continue trainer)')
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.test:
        load_and_test_model(args.model, args.episodes)
        return
    

    
    if args.compare:
        print(f"⚠️  Model comparison moved to comprehensive test system")
        print(f"   Run: python test_system.py --component models --verbose")
        return
    
    print(f"AI Escape Cage Training System")
    print(f"Trainer: {args.trainer}")
    if args.environment:
        print(f"Environment: {args.environment}")
    else:
        defaults = {"standard": "simple", "fast": "fast", "continue": "simple"}
        default_env = defaults.get(args.trainer, "simple")
        print(f"Environment: {default_env} (default)")
    print(f"Training steps: {args.steps:,}")
    print(f"Model name: {args.model}")
    print()
    
    # Handle training
    if args.trainer == 'continue':
        # Create trainer with environment configuration
        trainer_kwargs = {}
        if args.environment:
            trainer_kwargs['environment_type'] = args.environment
        
        trainer = ContinueTrainer(**trainer_kwargs)
        print("AI Escape Cage Trainer - Continue Training")
        print("Load and improve existing models")
        
        if args.fine_tune:
            results = trainer.fine_tune(args.model, args.fine_steps)
        else:
            results = trainer.continue_training(
                args.model, 
                args.continue_steps, 
                not args.no_backup
            )
    else:
        # Create trainer with environment configuration
        trainer_kwargs = {}
        if args.environment:
            trainer_kwargs['environment_type'] = args.environment
                
        # Create and run trainer
        trainer = create_trainer(args.trainer, **trainer_kwargs)
        
        trainer_descriptions = {
            'standard': "Balanced parameters for stable, production-ready models",
            'fast': "Optimized for quick results and prototyping"
        }
        
        description = trainer_descriptions.get(args.trainer, "")
        print(f"AI Escape Cage Trainer - {args.trainer.title()} Training")
        if description:
            print(description)
        
        # Adjust default steps for fast trainer
        timesteps = args.steps
        if args.trainer == 'fast' and args.steps == 50000:
            timesteps = 25000  # Default for fast training
        
        model_name = args.model
        if args.trainer == 'fast' and args.model == 'trained_escape_ai':
            model_name = 'fast_trained_ai'  # Default for fast trainer
        
        results = trainer.train(timesteps, model_name)
    
    # Print results summary
    if results and isinstance(results, dict):
        print(f"\nTraining Results:")
        for key, value in results.items():
            if key != 'model_path':  # Skip long paths
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
    
    print(f"\n✅ Training complete!")
    print(f"   Test your model: python test_system.py --component models")
    print(f"   Full system test: python test_system.py --verbose")


if __name__ == "__main__":
    main() 