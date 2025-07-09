"""
🧪 Test Trained AI
Load a saved AI model and watch it play with enhanced analytics and visualization
"""

import numpy as np
from stable_baselines3 import PPO
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import datetime

# Import the new base environment
from base_environment import TestEscapeCageEnv


class TestingAnalytics:
    """
    Class to track and analyze AI testing performance with detailed metrics.
    """
    
    def __init__(self):
        self.episodes_data: List[Dict[str, Any]] = []
        self.start_time = time.time()
        
    def record_episode(self, episode_num: int, steps: int, reward: float, 
                      success: bool, action_history: List[int], 
                      total_reward: float, terminated: bool, truncated: bool):
        """
        Record data from a completed episode.
        
        Args:
            episode_num: Episode number
            steps: Number of steps taken
            reward: Final step reward
            success: Whether the episode was successful
            action_history: List of actions taken
            total_reward: Total episode reward
            terminated: Whether episode terminated naturally
            truncated: Whether episode was truncated
        """
        episode_data = {
            'episode': episode_num,
            'steps': steps,
            'final_reward': reward,
            'total_reward': total_reward,
            'success': success,
            'terminated': terminated,
            'truncated': truncated,
            'action_count': len(action_history),
            'action_distribution': self._analyze_actions(action_history),
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.episodes_data.append(episode_data)
    
    def _analyze_actions(self, actions: List[int]) -> Dict[str, int]:
        """Analyze the distribution of actions taken."""
        action_names = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        distribution = {name: 0 for name in action_names.values()}
        
        for action in actions:
            if action in action_names:
                distribution[action_names[action]] += 1
                
        return distribution
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive testing report."""
        if not self.episodes_data:
            return {}
        
        successes = sum(1 for ep in self.episodes_data if ep['success'])
        total_episodes = len(self.episodes_data)
        
        # Calculate statistics
        steps_list = [ep['steps'] for ep in self.episodes_data]
        rewards_list = [ep['total_reward'] for ep in self.episodes_data]
        
        report = {
            'summary': {
                'total_episodes': total_episodes,
                'successes': successes,
                'success_rate': (successes / total_episodes) * 100,
                'total_test_time': time.time() - self.start_time
            },
            'performance': {
                'avg_steps': np.mean(steps_list),
                'min_steps': np.min(steps_list),
                'max_steps': np.max(steps_list),
                'avg_reward': np.mean(rewards_list),
                'min_reward': np.min(rewards_list),
                'max_reward': np.max(rewards_list)
            },
            'completion_types': {
                'natural_success': sum(1 for ep in self.episodes_data 
                                     if ep['success'] and ep['terminated']),
                'timeout_failure': sum(1 for ep in self.episodes_data 
                                     if not ep['success'] and ep['truncated']),
                'other_failure': sum(1 for ep in self.episodes_data 
                                   if not ep['success'] and ep['terminated'])
            }
        }
        
        # Action analysis for successful episodes
        successful_episodes = [ep for ep in self.episodes_data if ep['success']]
        if successful_episodes:
            all_actions = {}
            for ep in successful_episodes:
                for action, count in ep['action_distribution'].items():
                    all_actions[action] = all_actions.get(action, 0) + count
            
            total_actions = sum(all_actions.values())
            if total_actions > 0:
                report['successful_strategy'] = {
                    action: (count / total_actions) * 100 
                    for action, count in all_actions.items()
                }
        
        return report


def find_model_file(model_name: str) -> Optional[Path]:
    """Find a model file in common locations."""
    search_paths = [
        Path("models") / f"{model_name}.zip",
        Path("models") / model_name,
        Path(f"{model_name}.zip"),
        Path(model_name)
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    return None


def test_saved_ai(model_name: str = "trained_escape_ai", episodes: int = 10,
                 detailed_analysis: bool = False, save_report: bool = False):
    """
    Test a saved AI model with comprehensive analytics and visualization.
    
    Args:
        model_name: Name of the model to test
        episodes: Number of test episodes to run
        detailed_analysis: Whether to perform detailed action analysis
        save_report: Whether to save the testing report to file
    """
    
    print(f"🤖 Loading saved AI: {model_name}")
    print(f"🧪 Running {episodes} test episodes with analytics...")
    
    # Find the model file
    model_path = find_model_file(model_name)
    if not model_path:
        print(f"❌ Model '{model_name}' not found!")
        list_available_models()
        return
    
    try:
        # Create test environment
        env = TestEscapeCageEnv()
        
        # Load the saved AI model
        ai_agent = PPO.load(str(model_path), env=env)
        print(f"✅ AI model loaded from: {model_path}")
        print("🎮 Watch the trained AI play! (Press Ctrl+C to stop early)")
        
        # Initialize analytics
        analytics = TestingAnalytics()
        
        # Test the AI for multiple episodes
        for episode in range(episodes):
            print(f"\n🎯 Episode {episode + 1}/{episodes}")
            obs, _ = env.reset()
            episode_steps = 0
            
            start_time = time.time()
            
            while True:
                # AI makes decision (deterministic for consistent testing)
                action, _states = ai_agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_steps += 1
                
                # Display real-time progress for detailed analysis
                if detailed_analysis and episode_steps % 10 == 0:
                    print(f"   Step {episode_steps}: Action {action}, Reward: {reward:.3f}")
                
                if terminated or truncated:
                    episode_time = time.time() - start_time
                    success = info.get('success', False)
                    total_reward = info.get('total_reward', 0)
                    action_history = info.get('action_history', [])
                    
                    # Record episode data
                    analytics.record_episode(
                        episode + 1, episode_steps, reward, success,
                        action_history, total_reward, terminated, truncated
                    )
                    
                    # Display episode result
                    if success:
                        print(f"✅ SUCCESS in {episode_steps} steps! "
                              f"Total reward: {total_reward:.2f}, Time: {episode_time:.1f}s")
                    else:
                        reason = "Time limit" if truncated else "Failed"
                        print(f"❌ {reason} after {episode_steps} steps. "
                              f"Total reward: {total_reward:.2f}, Time: {episode_time:.1f}s")
                    
                    # Show action distribution for detailed analysis
                    if detailed_analysis and action_history:
                        action_dist = analytics._analyze_actions(action_history)
                        print(f"   Action distribution: {action_dist}")
                    
                    break
                
                if episode_steps > 500:  # Safety limit
                    print(f"⏰ Episode {episode + 1} exceeded 500 steps, ending...")
                    analytics.record_episode(
                        episode + 1, episode_steps, reward, False,
                        info.get('action_history', []), 
                        info.get('total_reward', 0), False, True
                    )
                    break
            
            # Brief pause between episodes for observation
            if episode < episodes - 1:  # Don't pause after last episode
                time.sleep(1)
        
        # Generate and display comprehensive report
        report = analytics.generate_report()
        display_test_report(report)
        
        # Save report if requested
        if save_report:
            save_test_report(report, model_name)
        
        env.close()
        
    except FileNotFoundError:
        print(f"❌ Model '{model_name}' not found!")
        list_available_models()
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Testing stopped by user after {len(analytics.episodes_data)} episodes")
        if analytics.episodes_data:
            report = analytics.generate_report()
            display_test_report(report)
            if save_report:
                save_test_report(report, model_name, suffix="_interrupted")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        print("🔧 Make sure Unity is running with the escape cage scene loaded")


def display_test_report(report: Dict[str, Any]):
    """Display a comprehensive test report."""
    if not report:
        print("❌ No test data to report")
        return
    
    print(f"\n{'='*60}")
    print("📊 COMPREHENSIVE TEST REPORT")
    print(f"{'='*60}")
    
    # Summary statistics
    summary = report['summary']
    print(f"\n📈 SUMMARY:")
    print(f"🎯 Episodes Tested: {summary['total_episodes']}")
    print(f"✅ Successes: {summary['successes']}")
    print(f"📊 Success Rate: {summary['success_rate']:.1f}%")
    print(f"⏱️ Total Test Time: {summary['total_test_time']:.1f} seconds")
    
    # Performance metrics
    performance = report['performance']
    print(f"\n🏃 PERFORMANCE METRICS:")
    print(f"📏 Average Steps: {performance['avg_steps']:.1f}")
    print(f"🔄 Steps Range: {performance['min_steps']} - {performance['max_steps']}")
    print(f"🏆 Average Reward: {performance['avg_reward']:.2f}")
    print(f"💰 Reward Range: {performance['min_reward']:.2f} - {performance['max_reward']:.2f}")
    
    # Completion analysis
    completion = report['completion_types']
    print(f"\n🎭 COMPLETION ANALYSIS:")
    print(f"🌟 Natural Successes: {completion['natural_success']}")
    print(f"⏰ Timeout Failures: {completion['timeout_failure']}")
    print(f"❌ Other Failures: {completion['other_failure']}")
    
    # Strategy analysis for successful episodes
    if 'successful_strategy' in report:
        strategy = report['successful_strategy']
        print(f"\n🧠 SUCCESSFUL STRATEGY ANALYSIS:")
        for action, percentage in strategy.items():
            print(f"   {action.upper()}: {percentage:.1f}%")
    
    # Performance assessment
    success_rate = summary['success_rate']
    print(f"\n🎓 PERFORMANCE ASSESSMENT:")
    if success_rate >= 90:
        print("🌟 OUTSTANDING! The AI has mastered the task exceptionally well.")
    elif success_rate >= 75:
        print("🏅 EXCELLENT! The AI performs the task very reliably.")
    elif success_rate >= 60:
        print("👍 GOOD! The AI understands the task well with room for improvement.")
    elif success_rate >= 40:
        print("⚠️ MODERATE! The AI shows basic understanding but needs more training.")
    elif success_rate >= 20:
        print("🔄 POOR! The AI struggles with the task and needs significant training.")
    else:
        print("❌ VERY POOR! The AI shows little understanding of the task.")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if success_rate < 60:
        print("   - Consider additional training with more timesteps")
        print("   - Try adjusting hyperparameters for better learning")
        print("   - Check if the reward function encourages the right behavior")
    elif success_rate < 85:
        print("   - Fine-tune the model with conservative learning rates")
        print("   - Consider collecting more diverse training scenarios")
    else:
        print("   - The model performs excellently!")
        print("   - Consider testing in more challenging scenarios")


def save_test_report(report: Dict[str, Any], model_name: str, suffix: str = ""):
    """Save the test report to a JSON file."""
    try:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_report_{model_name}_{timestamp}{suffix}.json"
        filepath = reports_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Test report saved to: {filepath}")
        
    except Exception as e:
        print(f"⚠️ Could not save test report: {e}")


def list_available_models():
    """Show all available saved models with details."""
    print("\n🗂️ Available AI models:")
    
    models_found = False
    
    # Check current directory
    for file in Path(".").glob("*.zip"):
        model_name = file.stem
        file_size = file.stat().st_size / 1024  # KB
        modified = datetime.datetime.fromtimestamp(file.stat().st_mtime)
        print(f"   📦 {model_name} ({file_size:.1f} KB, modified: {modified.strftime('%Y-%m-%d %H:%M')})")
        models_found = True
    
    # Check models directory
    models_dir = Path("models")
    if models_dir.exists():
        for file in models_dir.glob("*.zip"):
            model_name = file.stem
            file_size = file.stat().st_size / 1024  # KB
            modified = datetime.datetime.fromtimestamp(file.stat().st_mtime)
            print(f"   📦 {model_name} ({file_size:.1f} KB, modified: {modified.strftime('%Y-%m-%d %H:%M')})")
            models_found = True
    
    if not models_found:
        print("   ❌ No trained models found.")
        print("   🏋️ Train a model first using: python escape_cage_trainer.py")


def compare_models(model1: str, model2: str, episodes: int = 10):
    """
    Compare the performance of two AI models side by side.
    
    Args:
        model1: Name of the first model
        model2: Name of the second model 
        episodes: Number of episodes to test each model
    """
    print(f"⚖️ Comparing models: '{model1}' vs '{model2}'")
    print(f"🧪 Testing {episodes} episodes for each model...")
    
    results = {}
    
    for model_name in [model1, model2]:
        print(f"\n🤖 Testing model: {model_name}")
        model_path = find_model_file(model_name)
        
        if not model_path:
            print(f"❌ Model '{model_name}' not found!")
            continue
        
        try:
            env = TestEscapeCageEnv()
            ai_agent = PPO.load(str(model_path), env=env)
            analytics = TestingAnalytics()
            
            # Run test episodes quickly (less verbose)
            for episode in range(episodes):
                obs, _ = env.reset()
                episode_steps = 0
                
                while episode_steps < 500:
                    action, _ = ai_agent.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_steps += 1
                    
                    if terminated or truncated:
                        success = info.get('success', False)
                        total_reward = info.get('total_reward', 0)
                        action_history = info.get('action_history', [])
                        
                        analytics.record_episode(
                            episode + 1, episode_steps, reward, success,
                            action_history, total_reward, terminated, truncated
                        )
                        break
            
            results[model_name] = analytics.generate_report()
            env.close()
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
    
    # Display comparison
    if len(results) == 2:
        print(f"\n{'='*70}")
        print("📊 MODEL COMPARISON RESULTS")
        print(f"{'='*70}")
        
        model_names = list(results.keys())
        model1_report = results[model_names[0]]
        model2_report = results[model_names[1]]
        
        print(f"\n{'Metric':<25} {'Model 1':<20} {'Model 2':<20}")
        print(f"{'':<25} {model_names[0]:<20} {model_names[1]:<20}")
        print("-" * 70)
        
        # Compare key metrics
        metrics = [
            ('Success Rate', 'summary', 'success_rate', '%.1f%%'),
            ('Avg Steps', 'performance', 'avg_steps', '%.1f'),
            ('Avg Reward', 'performance', 'avg_reward', '%.2f'),
            ('Natural Successes', 'completion_types', 'natural_success', '%d')
        ]
        
        for metric_name, category, key, format_str in metrics:
            val1 = model1_report.get(category, {}).get(key, 0)
            val2 = model2_report.get(category, {}).get(key, 0)
            
            print(f"{metric_name:<25} {format_str % val1:<20} {format_str % val2:<20}")
        
        # Determine winner
        success1 = model1_report.get('summary', {}).get('success_rate', 0)
        success2 = model2_report.get('summary', {}).get('success_rate', 0)
        
        print(f"\n🏆 WINNER:")
        if success1 > success2:
            print(f"   {model_names[0]} performs better with {success1:.1f}% success rate!")
        elif success2 > success1:
            print(f"   {model_names[1]} performs better with {success2:.1f}% success rate!")
        else:
            print(f"   It's a tie! Both models have {success1:.1f}% success rate.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained AI models with analytics')
    parser.add_argument('--model', '-m', default='trained_escape_ai', 
                       help='Name of model to test (default: trained_escape_ai)')
    parser.add_argument('--episodes', '-e', type=int, default=10,
                       help='Number of test episodes (default: 10)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available models')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Enable detailed analysis during testing')
    parser.add_argument('--save-report', '-s', action='store_true',
                       help='Save test report to file')
    parser.add_argument('--compare', '-c', nargs=2, metavar=('MODEL1', 'MODEL2'),
                       help='Compare two models side by side')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    elif args.compare:
        compare_models(args.compare[0], args.compare[1], args.episodes)
    else:
        test_saved_ai(args.model, args.episodes, args.detailed, args.save_report) 