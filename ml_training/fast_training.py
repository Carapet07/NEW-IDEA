"""
⚡ Fast AI Training - Optimized for Quick Results
Aggressive training settings for faster learning
"""

import numpy as np
from stable_baselines3 import PPO
import time
import os

# Import the new base environment
from base_environment import FastEscapeCageEnv


def fast_train_ai():
    """Optimized training function for faster results"""
    
    print("⚡ Creating FAST AI agent...")
    
    # Create environment using the new base class
    env = FastEscapeCageEnv()
    
    # AGGRESSIVE PPO settings for faster learning
    ai_agent = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.001,       # HIGHER learning rate
        n_steps=512,               # Smaller rollouts for faster updates
        batch_size=32,             # Smaller batches for frequent updates
        n_epochs=4,                # Fewer epochs but more frequent
        gamma=0.95,                # Slightly lower discount (focus on immediate rewards)
        gae_lambda=0.9,            # Lower GAE for faster policy updates
        clip_range=0.3,            # Higher clip range for aggressive updates
        ent_coef=0.01,             # Encourage exploration
        verbose=1
    )
    
    print("⚡ Starting FAST AI training...")
    print("💡 This version should show results in 5-10 minutes!")
    print("🎯 Watch for key pickups and escapes!")
    
    try:
        # Start with shorter training but more intensive
        ai_agent.learn(total_timesteps=25000)
        
        # Save the trained AI
        model_path = os.path.join("models", "fast_trained_ai")
        os.makedirs("models", exist_ok=True)
        ai_agent.save(model_path)
        print(f"✅ FAST training complete! AI saved as '{model_path}'")
        
        # Test the model briefly
        print("\n🧪 Quick test of trained model...")
        test_fast_model(ai_agent, env)
        
    except KeyboardInterrupt:
        print("\n⏹️ Training stopped by user")
        partial_path = os.path.join("models", "fast_ai_partial")
        os.makedirs("models", exist_ok=True)
        ai_agent.save(partial_path)
        print(f"💾 Partial FAST training saved as '{partial_path}'")
        
        # Still test what we have
        print("\n🧪 Testing partial model...")
        test_fast_model(ai_agent, env)
    
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        print("🔧 Check Unity connection and try again")
        
    finally:
        env.close()


def test_fast_model(ai_agent, env, episodes=5):
    """
    Quick test of the trained model with enhanced feedback.
    
    Args:
        ai_agent: Trained PPO agent
        env: Environment instance
        episodes: Number of test episodes to run
    """
    
    successes = 0
    total_steps = 0
    total_rewards = []
    
    print(f"\n🧪 Testing model for {episodes} episodes...")
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_steps = 0
        episode_reward = 0
        
        print(f"\n🎯 Test Episode {episode + 1}")
        
        for step in range(200):  # Max 200 steps per test
            action, _ = ai_agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1
            episode_reward += reward
            
            if terminated or truncated:
                if info.get('success', False):
                    successes += 1
                    print(f"✅ SUCCESS in {episode_steps} steps! Reward: {episode_reward:.1f}")
                else:
                    reason = "Time limit" if truncated else "Failed"
                    print(f"❌ {reason} after {episode_steps} steps. Reward: {episode_reward:.1f}")
                break
        
        total_steps += episode_steps
        total_rewards.append(episode_reward)
        time.sleep(1)  # Brief pause between episodes
    
    # Calculate and display statistics
    success_rate = (successes / episodes) * 100
    avg_steps = total_steps / episodes
    avg_reward = np.mean(total_rewards)
    
    print(f"\n📊 FAST TRAINING RESULTS:")
    print(f"🎯 Success Rate: {success_rate:.1f}% ({successes}/{episodes})")
    print(f"📏 Average Steps: {avg_steps:.1f}")
    print(f"🏆 Average Reward: {avg_reward:.1f}")
    
    if success_rate >= 80:
        print("🌟 Excellent performance! The AI has learned well.")
    elif success_rate >= 60:
        print("👍 Good performance! Consider additional training for improvement.")
    elif success_rate >= 40:
        print("⚠️ Moderate performance. More training recommended.")
    else:
        print("🔄 Low performance. Try training for more timesteps or adjusting hyperparameters.")


def analyze_training_progress(episodes=10):
    """
    Analyze the training progress by testing the model multiple times.
    
    Args:
        episodes: Number of episodes to test for analysis
    """
    print("📈 Analyzing training progress...")
    
    # Try to load the saved model
    try:
        env = FastEscapeCageEnv()
        model_path = os.path.join("models", "fast_trained_ai")
        ai_agent = PPO.load(model_path, env=env)
        
        test_fast_model(ai_agent, env, episodes)
        env.close()
        
    except FileNotFoundError:
        print("❌ No trained model found. Run fast training first.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast AI training for escape cage')
    parser.add_argument('--analyze', '-a', action='store_true',
                       help='Analyze existing trained model instead of training')
    parser.add_argument('--episodes', '-e', type=int, default=10,
                       help='Number of episodes for analysis (default: 10)')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_training_progress(args.episodes)
    else:
        fast_train_ai() 