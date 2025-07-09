"""
🔄 Continue AI Training
Load a saved AI model and continue training it further with enhanced error handling
"""

import numpy as np
from stable_baselines3 import PPO
import time
import os
import datetime
from pathlib import Path
from typing import Optional, List

# Import the new base environment
from base_environment import SimpleEscapeCageEnv


def find_model_file(model_name: str) -> Optional[Path]:
    """
    Find a model file by searching in common locations.
    
    Args:
        model_name: Name of the model to find
        
    Returns:
        Path to the model file if found, None otherwise
    """
    # Common locations to search for models
    search_paths = [
        Path("models") / f"{model_name}.zip",
        Path("models") / model_name,
        Path(f"{model_name}.zip"),
        Path(model_name),
        Path(".") / f"{model_name}.zip",
        Path(".") / model_name
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None


def list_available_models() -> List[str]:
    """
    List all available saved models.
    
    Returns:
        List of model names without extensions
    """
    models = []
    
    # Check current directory
    for file in Path(".").glob("*.zip"):
        models.append(file.stem)
    
    # Check models directory
    models_dir = Path("models")
    if models_dir.exists():
        for file in models_dir.glob("*.zip"):
            models.append(file.stem)
    
    return sorted(set(models))


def continue_training(model_name: str = "trained_escape_ai", 
                     additional_steps: int = 25000,
                     save_backup: bool = True) -> bool:
    """
    Continue training from a saved model with enhanced error handling.
    
    Args:
        model_name: Name of the model to continue training
        additional_steps: Number of additional training steps
        save_backup: Whether to create a backup of the original model
        
    Returns:
        True if training succeeded, False otherwise
    """
    
    print(f"🔄 Loading AI model: {model_name}")
    print(f"📈 Will train for {additional_steps:,} additional steps")
    
    # Find the model file
    model_path = find_model_file(model_name)
    if not model_path:
        print(f"❌ Model '{model_name}' not found!")
        print("💡 Available models:")
        available_models = list_available_models()
        if available_models:
            for model in available_models:
                print(f"   - {model}")
        else:
            print("   - No models found")
        print("\n🏋️ Train a model first using: python escape_cage_trainer.py")
        return False
    
    try:
        # Create environment
        env = SimpleEscapeCageEnv()
        
        # Load the existing model
        print(f"📂 Loading model from: {model_path}")
        ai_agent = PPO.load(str(model_path), env=env)
        print(f"✅ Model '{model_name}' loaded successfully!")
        
        # Create backup if requested
        if save_backup:
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{model_name}_backup_{timestamp}"
                models_dir = Path("models")
                models_dir.mkdir(exist_ok=True)
                backup_path = models_dir / f"{backup_name}.zip"
                
                # Copy the original model
                import shutil
                shutil.copy2(str(model_path), str(backup_path))
                print(f"💾 Backup created: {backup_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not create backup: {e}")
        
        print("🧠 Continuing training from where we left off...")
        
        # Continue training
        start_time = time.time()
        ai_agent.learn(total_timesteps=additional_steps)
        training_time = time.time() - start_time
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save the improved model
        improved_name = f"{model_name}_improved"
        improved_path = models_dir / improved_name
        ai_agent.save(str(improved_path))
        print(f"✅ Training complete! Improved AI saved as '{improved_path}'")
        print(f"⏱️ Training took {training_time:.1f} seconds")
        
        # Test the improved model
        print("\n🧪 Testing improved model...")
        test_model_performance(ai_agent, env, episodes=5)
        
        env.close()
        return True
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("🔧 Troubleshooting tips:")
        print("   - Ensure Unity is running with the escape cage scene")
        print("   - Check that the model file is not corrupted")
        print("   - Verify sufficient disk space for saving models")
        return False
        
    except KeyboardInterrupt:
        print("\n⏹️ Training stopped by user")
        try:
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            partial_name = f"{model_name}_partial_improved"
            partial_path = models_dir / partial_name
            ai_agent.save(str(partial_path))
            print(f"💾 Partial improved training saved as '{partial_path}'")
            
            # Test partial model
            print("\n🧪 Testing partial improved model...")
            test_model_performance(ai_agent, env, episodes=3)
        except Exception as e:
            print(f"⚠️ Could not save partial model: {e}")
        finally:
            if 'env' in locals():
                env.close()
        return False


def train_with_different_settings(model_name: str = "trained_escape_ai", 
                                fine_tune_steps: int = 15000) -> bool:
    """
    Continue training with different hyperparameters for fine-tuning.
    
    Args:
        model_name: Name of the model to fine-tune
        fine_tune_steps: Number of fine-tuning steps
        
    Returns:
        True if fine-tuning succeeded, False otherwise
    """
    
    print(f"🔧 Fine-tuning AI model: {model_name}")
    
    # Find the model file
    model_path = find_model_file(model_name)
    if not model_path:
        print(f"❌ Model '{model_name}' not found!")
        return False
    
    try:
        # Create environment
        env = SimpleEscapeCageEnv()
        
        # Load the existing model
        ai_agent = PPO.load(str(model_path), env=env)
        
        # Change learning rate for fine-tuning (lower for stability)
        ai_agent.learning_rate = 0.0001
        print("🎛️ Using lower learning rate (0.0001) for fine-tuning")
        
        print("🧠 Fine-tuning the AI with more conservative learning...")
        
        start_time = time.time()
        # Fine-tune with fewer steps but more precision
        ai_agent.learn(total_timesteps=fine_tune_steps)
        training_time = time.time() - start_time
        
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save the fine-tuned model
        finetuned_name = f"{model_name}_finetuned"
        finetuned_path = models_dir / finetuned_name
        ai_agent.save(str(finetuned_path))
        print(f"✅ Fine-tuning complete! AI saved as '{finetuned_path}'")
        print(f"⏱️ Fine-tuning took {training_time:.1f} seconds")
        
        # Test the fine-tuned model
        print("\n🧪 Testing fine-tuned model...")
        test_model_performance(ai_agent, env, episodes=5)
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        return False
        
    except KeyboardInterrupt:
        print("\n⏹️ Fine-tuning stopped by user")
        try:
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            partial_name = f"{model_name}_partial_finetuned"
            partial_path = models_dir / partial_name
            ai_agent.save(str(partial_path))
            print(f"💾 Partial fine-tuned model saved as '{partial_path}'")
        except Exception as e:
            print(f"⚠️ Could not save partial model: {e}")
        finally:
            if 'env' in locals():
                env.close()
        return False


def test_model_performance(ai_agent, env, episodes: int = 5):
    """
    Test model performance and provide detailed feedback.
    
    Args:
        ai_agent: The trained PPO agent
        env: Environment instance
        episodes: Number of test episodes
    """
    
    successes = 0
    total_steps = 0
    total_rewards = []
    
    print(f"🧪 Testing model for {episodes} episodes...")
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_steps = 0
        episode_reward = 0
        
        print(f"\n🎯 Test Episode {episode + 1}")
        
        for step in range(500):  # Max steps per episode
            action, _ = ai_agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            episode_reward += reward
            
            if terminated or truncated:
                if info.get('success', False):
                    successes += 1
                    print(f"✅ SUCCESS in {episode_steps} steps! Reward: {episode_reward:.2f}")
                else:
                    reason = "Time limit" if truncated else "Failed"
                    print(f"❌ {reason} after {episode_steps} steps. Reward: {episode_reward:.2f}")
                break
        
        total_steps += episode_steps
        total_rewards.append(episode_reward)
        time.sleep(0.5)  # Brief pause
    
    # Calculate and display statistics
    success_rate = (successes / episodes) * 100
    avg_steps = total_steps / episodes
    avg_reward = np.mean(total_rewards)
    
    print(f"\n📊 PERFORMANCE RESULTS:")
    print(f"🎯 Success Rate: {success_rate:.1f}% ({successes}/{episodes})")
    print(f"📏 Average Steps: {avg_steps:.1f}")
    print(f"🏆 Average Reward: {avg_reward:.2f}")
    
    # Provide improvement suggestions
    if success_rate >= 90:
        print("🌟 Outstanding performance! The model is highly optimized.")
    elif success_rate >= 70:
        print("👍 Good performance! Minor improvements possible with more training.")
    elif success_rate >= 50:
        print("⚠️ Moderate performance. Consider additional training or hyperparameter tuning.")
    else:
        print("🔄 Poor performance. The model needs significantly more training.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Continue training saved AI models')
    parser.add_argument('--model', '-m', default='trained_escape_ai',
                       help='Name of model to continue training (default: trained_escape_ai)')
    parser.add_argument('--steps', '-s', type=int, default=25000,
                       help='Additional training steps (default: 25000)')
    parser.add_argument('--finetune', '-f', action='store_true',
                       help='Fine-tune with conservative settings')
    parser.add_argument('--fine-steps', type=int, default=15000,
                       help='Steps for fine-tuning (default: 15000)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup of original model')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available models')
    
    args = parser.parse_args()
    
    if args.list:
        print("📋 Available models:")
        models = list_available_models()
        if models:
            for model in models:
                print(f"   - {model}")
        else:
            print("   - No models found")
    elif args.finetune:
        train_with_different_settings(args.model, args.fine_steps)
    else:
        continue_training(args.model, args.steps, not args.no_backup) 