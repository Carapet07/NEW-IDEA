"""
🤖 AI Escape Cage Trainer
Simple script that trains an AI to escape from a cage with enhanced error handling and logging
"""

import numpy as np
from stable_baselines3 import PPO
import time
import os
from pathlib import Path

# Import the new base environment
from base_environment import SimpleEscapeCageEnv


def train_ai(total_timesteps: int = 50000, model_name: str = "trained_escape_ai"):
    """
    Main training function with improved error handling and logging.
    
    Args:
        total_timesteps: Number of training steps to perform
        model_name: Name to save the trained model under
    """
    
    print("🧠 Creating AI agent...")
    
    try:
        # Create environment using the new base class
        env = SimpleEscapeCageEnv()
        
        # Create AI agent (PPO algorithm)
        ai_agent = PPO(
            "MlpPolicy",           # Type of neural network
            env,                   # Environment to learn in
            learning_rate=0.0003,  # How fast AI learns
            verbose=1              # Show training progress
        )
        
        print("🏋️ Starting AI training...")
        print(f"💡 Training for {total_timesteps:,} steps (estimated time: 10-15 minutes)")
        print("💡 Tip: The AI will be bad at first, then gradually get better!")
        
        # Train the AI
        ai_agent.learn(total_timesteps=total_timesteps)
        
        # Create models directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save the trained AI
        model_path = models_dir / model_name
        ai_agent.save(str(model_path))
        print(f"✅ Training complete! AI saved as '{model_path}'")
        
        # Quick test of the trained model
        print("\n🧪 Quick test of trained model...")
        test_trained_model(ai_agent, env, episodes=3)
        
    except KeyboardInterrupt:
        print("\n⏹️ Training stopped by user")
        try:
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            partial_model_path = models_dir / f"{model_name}_partial"
            ai_agent.save(str(partial_model_path))
            print(f"💾 Partial training saved as '{partial_model_path}'")
            
            # Test partial model
            print("\n🧪 Testing partial model...")
            test_trained_model(ai_agent, env, episodes=2)
        except Exception as e:
            print(f"⚠️ Could not save partial model: {e}")
    
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("🔧 Troubleshooting tips:")
        print("   - Make sure Unity is running and the escape cage scene is loaded")
        print("   - Check that no firewall is blocking port 9999")
        print("   - Verify Python has the required packages installed")
        return False
    
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"⚠️ Warning closing environment: {e}")
    
    return True


def test_trained_model(ai_agent, env, episodes: int = 3):
    """
    Test the trained model and provide detailed feedback.
    
    Args:
        ai_agent: The trained PPO agent
        env: Environment instance
        episodes: Number of test episodes to run
    """
    
    successes = 0
    total_steps = 0
    total_rewards = []
    
    print(f"🧪 Testing trained AI for {episodes} episodes...")
    
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
                    reason = "Time limit reached" if truncated else "Episode ended"
                    print(f"❌ {reason} after {episode_steps} steps. Reward: {episode_reward:.2f}")
                break
        
        total_steps += episode_steps
        total_rewards.append(episode_reward)
        time.sleep(1)
    
    # Calculate statistics
    success_rate = (successes / episodes) * 100
    avg_steps = total_steps / episodes
    avg_reward = np.mean(total_rewards)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"🎯 Success Rate: {success_rate:.1f}% ({successes}/{episodes})")
    print(f"📏 Average Steps: {avg_steps:.1f}")
    print(f"🏆 Average Reward: {avg_reward:.2f}")
    
    # Provide performance feedback
    if success_rate >= 80:
        print("🌟 Excellent! The AI has learned the task very well.")
    elif success_rate >= 60:
        print("👍 Good performance! The AI understands the basic strategy.")
    elif success_rate >= 30:
        print("⚠️ Moderate performance. Consider more training or hyperparameter tuning.")
    else:
        print("🔄 Poor performance. The AI needs significantly more training.")


def load_and_test_model(model_name: str = "trained_escape_ai", episodes: int = 5):
    """
    Load a saved model and test it.
    
    Args:
        model_name: Name of the model to load
        episodes: Number of test episodes
    """
    
    try:
        print(f"📂 Loading model: {model_name}")
        
        # Create environment
        env = SimpleEscapeCageEnv()
        
        # Try to load from models directory first, then current directory
        model_paths = [
            Path("models") / model_name,
            Path(model_name),
            Path(f"models/{model_name}.zip"),
            Path(f"{model_name}.zip")
        ]
        
        model_loaded = False
        for path in model_paths:
            try:
                ai_agent = PPO.load(str(path), env=env)
                print(f"✅ Model loaded from: {path}")
                model_loaded = True
                break
            except FileNotFoundError:
                continue
        
        if not model_loaded:
            print(f"❌ Model '{model_name}' not found in any of these locations:")
            for path in model_paths:
                print(f"   - {path}")
            return
        
        # Test the model
        test_trained_model(ai_agent, env, episodes)
        env.close()
        
    except Exception as e:
        print(f"❌ Error loading or testing model: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AI to escape from cage')
    parser.add_argument('--steps', '-s', type=int, default=50000,
                       help='Number of training steps (default: 50000)')
    parser.add_argument('--model', '-m', default='trained_escape_ai',
                       help='Model name to save/load (default: trained_escape_ai)')
    parser.add_argument('--test', '-t', action='store_true',
                       help='Test existing model instead of training')
    parser.add_argument('--episodes', '-e', type=int, default=5,
                       help='Number of test episodes (default: 5)')
    
    args = parser.parse_args()
    
    if args.test:
        load_and_test_model(args.model, args.episodes)
    else:
        train_ai(args.steps, args.model)