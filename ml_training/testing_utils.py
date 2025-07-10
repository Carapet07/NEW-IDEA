"""
Testing Utilities Module
Comprehensive utilities for testing, evaluation, and benchmarking AI models.

This module provides:
- Automated testing frameworks
- Performance benchmarking tools
- Model evaluation and validation
- A/B testing utilities
- Test report generation
- Environment testing helpers
- Interactive CLI for testing trained models
"""

import numpy as np
import time
import datetime
import argparse
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from pathlib import Path
import csv
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import PPO


@dataclass
class TestConfiguration:
    """
    Configuration for test runs.
    
    Attributes:
        test_name: Name identifier for the test
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        deterministic: Whether to use deterministic policy
        render: Whether to render during testing
        save_actions: Whether to save action sequences
        save_observations: Whether to save observations
        timeout_seconds: Timeout for entire test run
    """
    test_name: str
    num_episodes: int = 10
    max_steps_per_episode: int = 500
    deterministic: bool = True
    render: bool = False
    save_actions: bool = True
    save_observations: bool = False
    timeout_seconds: int = 3600  # 1 hour default


@dataclass
class TestResult:
    """
    Results from a single test episode.
    
    Attributes:
        episode_num: Episode number
        steps_taken: Number of steps in episode
        total_reward: Total reward accumulated
        final_reward: Final step reward
        success: Whether episode was successful
        terminated: Whether episode terminated naturally
        truncated: Whether episode was truncated
        duration: Episode duration in seconds
        actions: List of actions taken (if recorded)
        observations: List of observations (if recorded)
        info: Additional information from environment
        error: Error message if episode failed
    """
    episode_num: int
    steps_taken: int
    total_reward: float
    final_reward: float
    success: bool
    terminated: bool
    truncated: bool
    duration: float
    actions: Optional[List[int]] = None
    observations: Optional[List[np.ndarray]] = None
    info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()


class TestingAnalytics:
    """
    Class to track and analyze AI testing performance with detailed metrics.
    Provides comprehensive analytics for model evaluation and comparison.
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
        action_counts = {name: 0 for name in action_names.values()}
        
        for action in actions:
            if action in action_names:
                action_counts[action_names[action]] += 1
        
        return action_counts
    
    def _analyze_success_pattern(self) -> Dict[str, Any]:
        """Analyze patterns in successful episodes."""
        successful_episodes = [ep for ep in self.episodes_data if ep['success']]
        
        if not successful_episodes:
            return {'message': 'No successful episodes to analyze'}
        
        # Average steps in successful episodes
        avg_success_steps = np.mean([ep['steps'] for ep in successful_episodes])
        
        # Most common action pattern in successful episodes
        combined_actions = {}
        for ep in successful_episodes:
            for action, count in ep['action_distribution'].items():
                combined_actions[action] = combined_actions.get(action, 0) + count
        
        total_actions = sum(combined_actions.values())
        action_percentages = {
            action: (count / total_actions) * 100 
            for action, count in combined_actions.items()
        }
        
        return {
            'avg_steps': avg_success_steps,
            'action_strategy': action_percentages,
            'success_count': len(successful_episodes)
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not self.episodes_data:
            return {'error': 'No episode data available'}
        
        total_episodes = len(self.episodes_data)
        successes = sum(1 for ep in self.episodes_data if ep['success'])
        success_rate = (successes / total_episodes) * 100
        
        steps_list = [ep['steps'] for ep in self.episodes_data]
        rewards_list = [ep['total_reward'] for ep in self.episodes_data]
        
        # Calculate completion types
        natural_success = sum(1 for ep in self.episodes_data 
                            if ep['success'] and ep['terminated'])
        timeout_failure = sum(1 for ep in self.episodes_data 
                            if not ep['success'] and ep['truncated'])
        other_failure = total_episodes - successes - timeout_failure
        
        report = {
            'summary': {
                'total_episodes': total_episodes,
                'successes': successes,
                'success_rate': success_rate,
                'total_test_time': time.time() - self.start_time
            },
            'performance': {
                'avg_steps': np.mean(steps_list),
                'min_steps': min(steps_list),
                'max_steps': max(steps_list),
                'avg_reward': np.mean(rewards_list),
                'min_reward': min(rewards_list),
                'max_reward': max(rewards_list)
            },
            'completion_types': {
                'natural_success': natural_success,
                'timeout_failure': timeout_failure,
                'other_failure': other_failure
            }
        }
        
        # Add success pattern analysis
        success_analysis = self._analyze_success_pattern()
        if 'action_strategy' in success_analysis:
            report['successful_strategy'] = success_analysis['action_strategy']
        
        return report


class TestRunner:
    """
    Main class for running comprehensive model tests.
    
    Provides functionality for:
    - Running single and batch model tests
    - Performance benchmarking
    - Statistical analysis of results
    - Test report generation
    """
    
    def __init__(self, config: TestConfiguration):
        """
        Initialize test runner.
        
        Args:
            config: Test configuration
        """
        self.config = config
        self.results: List[TestResult] = []
        self.start_time: Optional[datetime.datetime] = None
        self.end_time: Optional[datetime.datetime] = None
        self.logger = logging.getLogger(f'test_runner.{config.test_name}')
    
    def run_model_test(self, model: BaseAlgorithm, env, 
                      progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run comprehensive test on a model.
        
        Args:
            model: Trained model to test
            env: Environment to test in
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing test summary and results
        """
        self.results.clear()
        self.start_time = datetime.datetime.now()
        
        self.logger.info(f"Starting test '{self.config.test_name}' with {self.config.num_episodes} episodes")
        
        for episode in range(self.config.num_episodes):
            try:
                episode_start = time.time()
                
                # Reset environment
                obs, info = env.reset()
                episode_reward = 0
                step_count = 0
                actions_taken = []
                observations = []
                
                # Run episode
                for step in range(self.config.max_steps_per_episode):
                    # Get action from model
                    action, _ = model.predict(obs, deterministic=self.config.deterministic)
                    
                    # Store data if requested
                    if self.config.save_actions:
                        actions_taken.append(int(action))
                    if self.config.save_observations:
                        observations.append(obs.copy())
                    
                    # Step environment
                    obs, reward, terminated, truncated, step_info = env.step(action)
                    episode_reward += reward
                    step_count += 1
                    
                    # Check for episode end
                    if terminated or truncated:
                        break
                
                episode_duration = time.time() - episode_start
                
                # Create test result
                result = TestResult(
                    episode_num=episode + 1,
                    steps_taken=step_count,
                    total_reward=episode_reward,
                    final_reward=reward,
                    success=step_info.get('success', False),
                    terminated=terminated,
                    truncated=truncated,
                    duration=episode_duration,
                    actions=actions_taken if self.config.save_actions else None,
                    observations=observations if self.config.save_observations else None,
                    info=step_info
                )
                
                self.results.append(result)
                
                # Progress callback
                if progress_callback:
                    progress_callback(episode + 1, self.config.num_episodes, result)
                
                self.logger.debug(f"Episode {episode + 1}: {step_count} steps, "
                                f"reward {episode_reward:.2f}, "
                                f"success: {result.success}")
                
            except Exception as e:
                self.logger.error(f"Episode {episode + 1} failed: {e}")
                error_result = TestResult(
                    episode_num=episode + 1,
                    steps_taken=0,
                    total_reward=0,
                    final_reward=0,
                    success=False,
                    terminated=False,
                    truncated=False,
                    duration=0,
                    error=str(e)
                )
                self.results.append(error_result)
        
        self.end_time = datetime.datetime.now()
        
        # Generate summary
        return self._generate_summary()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary from results."""
        if not self.results:
            return {'error': 'No test results available'}
        
        # Filter out error results for statistics
        valid_results = [r for r in self.results if r.error is None]
        
        if not valid_results:
            return {'error': 'No valid test results'}
        
        # Calculate statistics
        total_episodes = len(self.results)
        successes = sum(1 for r in valid_results if r.success)
        success_rate = (successes / len(valid_results)) * 100
        
        avg_steps = np.mean([r.steps_taken for r in valid_results])
        avg_reward = np.mean([r.total_reward for r in valid_results])
        avg_duration = np.mean([r.duration for r in valid_results])
        
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            'test_name': self.config.test_name,
            'total_episodes': total_episodes,
            'valid_episodes': len(valid_results),
            'errors': total_episodes - len(valid_results),
            'success_rate': success_rate,
            'successes': successes,
            'avg_steps': avg_steps,
            'avg_reward': avg_reward,
            'avg_episode_duration': avg_duration,
            'total_test_duration': total_duration,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'results': [asdict(r) for r in self.results]
        }
    
    def run_quick_test(self, model: BaseAlgorithm, env, episodes: int = 5) -> Dict[str, Any]:
        """
        Run a quick test with simplified reporting.
        
        Args:
            model: Model to test
            env: Environment to test in
            episodes: Number of episodes to run
            
        Returns:
            Quick test results
        """
        # Update config for quick test
        original_episodes = self.config.num_episodes
        self.config.num_episodes = episodes
        
        try:
            results = self.run_model_test(model, env)
            return {
                'quick_test': True,
                'episodes_tested': episodes,
                'success_rate': results.get('success_rate', 0),
                'avg_reward': results.get('avg_reward', 0),
                'avg_steps': results.get('avg_steps', 0)
            }
        finally:
            # Restore original config
            self.config.num_episodes = original_episodes


class ModelBenchmark:
    """
    Comprehensive benchmarking suite for model comparison.
    
    Provides functionality for:
    - Comparing multiple models
    - Statistical significance testing
    - Performance ranking
    - Benchmark report generation
    """
    
    def __init__(self, benchmark_name: str = "model_benchmark"):
        """
        Initialize model benchmark.
        
        Args:
            benchmark_name: Name for this benchmark suite
        """
        self.benchmark_name = benchmark_name
        self.model_results: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f'benchmark.{benchmark_name}')
    
    def add_model_results(self, model_name: str, test_results: Dict[str, Any]) -> None:
        """
        Add test results for a model.
        
        Args:
            model_name: Name of the model
            test_results: Results from TestRunner
        """
        self.model_results[model_name] = test_results
        self.logger.info(f"Added results for model: {model_name}")
    
    def run_benchmark(self, models: Dict[str, BaseAlgorithm], env, 
                     config: TestConfiguration) -> Dict[str, Any]:
        """
        Run benchmark on multiple models.
        
        Args:
            models: Dictionary mapping model names to model instances
            env: Environment to test in
            config: Test configuration
            
        Returns:
            Benchmark results
        """
        self.logger.info(f"Starting benchmark '{self.benchmark_name}' with {len(models)} models")
        
        for model_name, model in models.items():
            self.logger.info(f"Testing model: {model_name}")
            
            # Create test runner for this model
            test_config = TestConfiguration(
                test_name=f"{self.benchmark_name}_{model_name}",
                num_episodes=config.num_episodes,
                max_steps_per_episode=config.max_steps_per_episode,
                deterministic=config.deterministic
            )
            
            runner = TestRunner(test_config)
            results = runner.run_model_test(model, env)
            self.add_model_results(model_name, results)
        
        return self.generate_benchmark_report()
    
    def generate_benchmark_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        if not self.model_results:
            return {'error': 'No model results available'}
        
        # Rank models by success rate
        model_rankings = []
        for model_name, results in self.model_results.items():
            if 'success_rate' in results:
                model_rankings.append({
                    'model': model_name,
                    'success_rate': results['success_rate'],
                    'avg_reward': results.get('avg_reward', 0),
                    'avg_steps': results.get('avg_steps', 0)
                })
        
        # Sort by success rate (descending)
        model_rankings.sort(key=lambda x: x['success_rate'], reverse=True)
        
        # Find best performing model
        best_model = model_rankings[0] if model_rankings else None
        
        return {
            'benchmark_name': self.benchmark_name,
            'models_tested': len(self.model_results),
            'rankings': model_rankings,
            'best_model': best_model,
            'detailed_results': self.model_results,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def compare_models(self, model1_name: str, model2_name: str) -> Dict[str, Any]:
        """
        Compare two specific models.
        
        Args:
            model1_name: Name of first model
            model2_name: Name of second model
            
        Returns:
            Comparison results
        """
        if model1_name not in self.model_results or model2_name not in self.model_results:
            return {'error': 'One or both models not found in benchmark results'}
        
        results1 = self.model_results[model1_name]
        results2 = self.model_results[model2_name]
        
        comparison = {
            'model1': model1_name,
            'model2': model2_name,
            'success_rate_diff': results1.get('success_rate', 0) - results2.get('success_rate', 0),
            'reward_diff': results1.get('avg_reward', 0) - results2.get('avg_reward', 0),
            'steps_diff': results1.get('avg_steps', 0) - results2.get('avg_steps', 0),
            'better_model': model1_name if results1.get('success_rate', 0) > results2.get('success_rate', 0) else model2_name
        }
        
        return comparison


class TestReportGenerator:
    """
    Utility for generating comprehensive test reports.
    
    Supports multiple output formats:
    - JSON reports
    - CSV data exports
    - HTML reports (if jinja2 available)
    - Text summaries
    """
    
    def __init__(self):
        self.logger = logging.getLogger('test_reporter')
    
    def generate_json_report(self, test_results: Dict[str, Any], 
                           output_path: str) -> bool:
        """
        Generate JSON test report.
        
        Args:
            test_results: Results from TestRunner or ModelBenchmark
            output_path: Path to save the report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            self.logger.info(f"JSON report saved to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate JSON report: {e}")
            return False
    
    def generate_csv_report(self, test_results: Dict[str, Any], 
                          output_path: str) -> bool:
        """
        Generate CSV report with episode data.
        
        Args:
            test_results: Results from TestRunner
            output_path: Path to save the CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if 'results' not in test_results:
                self.logger.error("No episode results found in test data")
                return False
            
            with open(output_path, 'w', newline='') as csvfile:
                fieldnames = ['episode_num', 'steps_taken', 'total_reward', 
                            'success', 'terminated', 'truncated', 'duration']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for result in test_results['results']:
                    # Filter to only include relevant fields
                    row = {k: v for k, v in result.items() if k in fieldnames}
                    writer.writerow(row)
            
            self.logger.info(f"CSV report saved to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate CSV report: {e}")
            return False
    
    def generate_text_summary(self, test_results: Dict[str, Any]) -> str:
        """
        Generate human-readable text summary.
        
        Args:
            test_results: Results from TestRunner or ModelBenchmark
            
        Returns:
            Formatted text summary
        """
        if 'benchmark_name' in test_results:
            # Benchmark report
            return self._format_benchmark_summary(test_results)
        else:
            # Single model test report
            return self._format_test_summary(test_results)
    
    def _format_test_summary(self, results: Dict[str, Any]) -> str:
        """Format single model test summary."""
        summary = f"""
Test Report: {results.get('test_name', 'Unknown Test')}
{'='*50}

Overview:
  Episodes Tested: {results.get('total_episodes', 0)}
  Success Rate: {results.get('success_rate', 0):.1f}%
  Successes: {results.get('successes', 0)}

Performance:
  Average Steps: {results.get('avg_steps', 0):.1f}
  Average Reward: {results.get('avg_reward', 0):.2f}
  Test Duration: {results.get('total_test_duration', 0):.1f} seconds

Status: {'PASSED' if results.get('success_rate', 0) >= 70 else 'NEEDS IMPROVEMENT'}
"""
        return summary
    
    def _format_benchmark_summary(self, results: Dict[str, Any]) -> str:
        """Format benchmark summary."""
        summary = f"""
Benchmark Report: {results.get('benchmark_name', 'Unknown Benchmark')}
{'='*50}

Models Tested: {results.get('models_tested', 0)}

Rankings:
"""
        
        for i, model in enumerate(results.get('rankings', []), 1):
            summary += f"  {i}. {model['model']} - {model['success_rate']:.1f}% success rate\n"
        
        best_model = results.get('best_model', {})
        if best_model:
            summary += f"\nBest Performing Model: {best_model.get('model', 'Unknown')}\n"
        
        return summary


# Utility functions for common testing operations

def find_model_file(model_name: str) -> Optional[Path]:
    """Find a model file by searching in common locations."""
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
    """List all available saved models."""
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
    from base_environment import TestEscapeCageEnv
    
    print(f"Loading saved AI: {model_name}")
    print(f"Running {episodes} test episodes with analytics...")
    
    # Find the model file
    model_path = find_model_file(model_name)
    if not model_path:
        print(f"Model '{model_name}' not found!")
        print("Available models:")
        available_models = list_available_models()
        if available_models:
            for model in available_models:
                print(f"   - {model}")
        else:
            print("   - No models found")
        return
    
    try:
        # Create test environment
        env = TestEscapeCageEnv()
        
        # Load the saved AI model
        ai_agent = PPO.load(str(model_path), env=env)
        print(f"AI model loaded from: {model_path}")
        print("Watch the trained AI play! (Press Ctrl+C to stop early)")
        
        # Initialize analytics
        analytics = TestingAnalytics()
        
        # Test the AI for multiple episodes
        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}/{episodes}")
            obs, _ = env.reset()
            episode_steps = 0
            action_history = []
            total_reward = 0
            
            start_time = time.time()
            
            while True:
                # AI makes decision (deterministic for consistent testing)
                action, _states = ai_agent.predict(obs, deterministic=True)
                action_history.append(int(action))
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_steps += 1
                total_reward += reward
                
                # Display real-time progress for detailed analysis
                if detailed_analysis and episode_steps % 10 == 0:
                    print(f"   Step {episode_steps}: Action {action}, Reward: {reward:.3f}")
                
                if terminated or truncated:
                    success = info.get('success', False)
                    
                    # Record episode data
                    analytics.record_episode(
                        episode + 1, episode_steps, reward, success,
                        action_history, total_reward, terminated, truncated
                    )
                    
                    # Display episode result
                    if success:
                        print(f"   SUCCESS! Completed in {episode_steps} steps")
                        print(f"   Total reward: {total_reward:.2f}")
                    else:
                        reason = "Time limit" if truncated else "Failed"
                        print(f"   {reason} after {episode_steps} steps")
                        print(f"   Total reward: {total_reward:.2f}")
                    
                    break
            
            # Brief pause between episodes
            time.sleep(0.5)
        
        # Generate and display comprehensive report
        report = analytics.generate_report()
        display_test_report(report)
        
        # Save report if requested
        if save_report:
            save_test_report(report, model_name)
        
        env.close()
        
    except FileNotFoundError:
        print(f"Model '{model_name}' not found!")
        print("Available models:")
        available_models = list_available_models()
        if available_models:
            for model in available_models:
                print(f"   - {model}")
        else:
            print("   - No models found")
        
    except KeyboardInterrupt:
        print(f"\nTesting stopped by user after {len(analytics.episodes_data)} episodes")
        if analytics.episodes_data:
            report = analytics.generate_report()
            display_test_report(report)
            if save_report:
                save_test_report(report, model_name, suffix="_interrupted")
        
    except Exception as e:
        print(f"Testing failed: {e}")
        print("Make sure Unity is running with the escape cage scene loaded")


def display_test_report(report: Dict[str, Any]):
    """Display a comprehensive test report."""
    if not report:
        print("No test data to report")
        return
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST REPORT")
    print(f"{'='*60}")
    
    # Summary statistics
    summary = report['summary']
    print(f"\nSUMMARY:")
    print(f"Episodes Tested: {summary['total_episodes']}")
    print(f"Successes: {summary['successes']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Total Test Time: {summary['total_test_time']:.1f} seconds")
    
    # Performance metrics
    performance = report['performance']
    print(f"\nPERFORMANCE METRICS:")
    print(f"Average Steps: {performance['avg_steps']:.1f}")
    print(f"Steps Range: {performance['min_steps']} - {performance['max_steps']}")
    print(f"Average Reward: {performance['avg_reward']:.2f}")
    print(f"Reward Range: {performance['min_reward']:.2f} - {performance['max_reward']:.2f}")
    
    # Completion analysis
    completion = report['completion_types']
    print(f"\nCOMPLETION ANALYSIS:")
    print(f"Natural Successes: {completion['natural_success']}")
    print(f"Timeout Failures: {completion['timeout_failure']}")
    print(f"Other Failures: {completion['other_failure']}")
    
    # Strategy analysis for successful episodes
    if 'successful_strategy' in report:
        strategy = report['successful_strategy']
        print(f"\nSUCCESSFUL STRATEGY ANALYSIS:")
        for action, percentage in strategy.items():
            print(f"   {action.upper()}: {percentage:.1f}%")
    
    # Performance assessment
    success_rate = summary['success_rate']
    print(f"\nPERFORMANCE ASSESSMENT:")
    if success_rate >= 90:
        print("   EXCELLENT: Your AI performs exceptionally well!")
        print("   This model is ready for production use.")
    elif success_rate >= 70:
        print("   GOOD: Your AI shows strong performance.")
        print("   Minor improvements could push it to excellence.")
    elif success_rate >= 50:
        print("   MODERATE: Your AI is learning but needs improvement.")
        print("   Consider more training or hyperparameter adjustment.")
    elif success_rate >= 25:
        print("   POOR: Your AI struggles with the task.")
        print("   Significant additional training recommended.")
    else:
        print("   VERY POOR: Your AI rarely succeeds.")
        print("   Check training setup and increase training time significantly.")
    
    # Specific recommendations
    avg_steps = performance['avg_steps']
    print(f"\nRECOMMENDATIONS:")
    if avg_steps > 300:
        print("   - AI takes many steps - consider training for efficiency")
    elif avg_steps < 50:
        print("   - Very efficient! AI finds solutions quickly")
    
    if completion['timeout_failure'] > completion['natural_success']:
        print("   - Many timeouts suggest AI gets stuck - needs more exploration training")


def save_test_report(report: Dict[str, Any], model_name: str, suffix: str = ""):
    """
    Save test report to file.
    
    Args:
        report: Test report data
        model_name: Name of the model tested
        suffix: Optional suffix for filename
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_report_{model_name}_{timestamp}{suffix}.json"
    
    try:
        # Create reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / filename
        
        # Add metadata
        report_with_metadata = {
            'model_name': model_name,
            'test_timestamp': timestamp,
            'report_version': '1.0',
            **report
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_with_metadata, f, indent=2, default=str)
        
        print(f"\nTest report saved to: {report_path}")
        
    except Exception as e:
        print(f"Failed to save test report: {e}")


def compare_models(model1: str, model2: str, episodes: int = 10):
    """
    Compare the performance of two AI models side by side.
    
    Args:
        model1: Name of the first model
        model2: Name of the second model 
        episodes: Number of episodes to test each model
    """
    from base_environment import TestEscapeCageEnv
    
    print(f"Comparing models: '{model1}' vs '{model2}'")
    print(f"Testing {episodes} episodes for each model...")
    
    results = {}
    
    for model_name in [model1, model2]:
        print(f"\nTesting model: {model_name}")
        model_path = find_model_file(model_name)
        
        if not model_path:
            print(f"Model '{model_name}' not found!")
            continue
        
        try:
            env = TestEscapeCageEnv()
            ai_agent = PPO.load(str(model_path), env=env)
            analytics = TestingAnalytics()
            
            # Run test episodes quickly (less verbose)
            for episode in range(episodes):
                obs, _ = env.reset()
                episode_steps = 0
                action_history = []
                total_reward = 0
                
                while episode_steps < 500:
                    action, _ = ai_agent.predict(obs, deterministic=True)
                    action_history.append(int(action))
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_steps += 1
                    total_reward += reward
                    
                    if terminated or truncated:
                        success = info.get('success', False)
                        
                        analytics.record_episode(
                            episode + 1, episode_steps, reward, success,
                            action_history, total_reward, terminated, truncated
                        )
                        break
            
            results[model_name] = analytics.generate_report()
            env.close()
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            continue
    
    # Display comparison
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON RESULTS")
        print(f"{'='*60}")
        
        model_names = list(results.keys())
        r1, r2 = results[model_names[0]], results[model_names[1]]
        
        print(f"\n{'Metric':<20} | {model_names[0]:<15} | {model_names[1]:<15} | Winner")
        print("-" * 70)
        
        # Success rate comparison
        sr1 = r1['summary']['success_rate']
        sr2 = r2['summary']['success_rate']
        winner = model_names[0] if sr1 > sr2 else model_names[1]
        print(f"{'Success Rate':<20} | {sr1:<14.1f}% | {sr2:<14.1f}% | {winner}")
        
        # Average steps comparison
        as1 = r1['performance']['avg_steps']
        as2 = r2['performance']['avg_steps']
        winner = model_names[0] if as1 < as2 else model_names[1]  # Fewer steps is better
        print(f"{'Avg Steps':<20} | {as1:<14.1f}  | {as2:<14.1f}  | {winner}")
        
        # Average reward comparison
        ar1 = r1['performance']['avg_reward']
        ar2 = r2['performance']['avg_reward']
        winner = model_names[0] if ar1 > ar2 else model_names[1]
        print(f"{'Avg Reward':<20} | {ar1:<14.2f}  | {ar2:<14.2f}  | {winner}")
        
        # Overall winner
        scores = {model_names[0]: 0, model_names[1]: 0}
        if sr1 > sr2: scores[model_names[0]] += 1
        else: scores[model_names[1]] += 1
        
        if as1 < as2: scores[model_names[0]] += 1
        else: scores[model_names[1]] += 1
        
        if ar1 > ar2: scores[model_names[0]] += 1
        else: scores[model_names[1]] += 1
        
        overall_winner = max(scores, key=scores.get)
        print(f"\nOVERALL WINNER: {overall_winner}")
        print(f"Score: {overall_winner} wins {scores[overall_winner]}/3 metrics")
        
    else:
        print("Could not compare models - insufficient valid results")


def run_batch_tests(model_names: List[str], episodes: int = 10) -> Dict[str, Any]:
    """
    Run tests on multiple models and generate comparative report.
    
    Args:
        model_names: List of model names to test
        episodes: Number of episodes per model
        
    Returns:
        Batch test results
    """
    print(f"Running batch tests on {len(model_names)} models...")
    
    # Create benchmark
    benchmark = ModelBenchmark("batch_test")
    
    for model_name in model_names:
        print(f"\nTesting {model_name}...")
        
        # Create test config
        config = TestConfiguration(
            test_name=f"batch_test_{model_name}",
            num_episodes=episodes,
            deterministic=True
        )
        
        model_path = find_model_file(model_name)
        if not model_path:
            print(f"Model {model_name} not found, skipping...")
            continue
        
        try:
            from base_environment import TestEscapeCageEnv
            env = TestEscapeCageEnv()
            model = PPO.load(str(model_path), env=env)
            
            runner = TestRunner(config)
            results = runner.run_model_test(model, env)
            benchmark.add_model_results(model_name, results)
            
            env.close()
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
    
    return benchmark.generate_benchmark_report()


def main():
    """Main function for command-line testing interface."""
    parser = argparse.ArgumentParser(description="AI Model Testing Utilities")
    
    # Action selection
    parser.add_argument('action', choices=['test', 'compare', 'batch', 'list'], 
                       help='Action to perform')
    
    # Common arguments
    parser.add_argument('--model', '-m', default='trained_escape_ai',
                       help='Model name to test (default: trained_escape_ai)')
    parser.add_argument('--episodes', '-e', type=int, default=10,
                       help='Number of test episodes (default: 10)')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Enable detailed analysis during testing')
    parser.add_argument('--save-report', '-s', action='store_true',
                       help='Save test report to file')
    
    # Compare specific arguments
    parser.add_argument('--model2', help='Second model for comparison')
    
    # Batch testing arguments
    parser.add_argument('--models', nargs='+', help='List of models for batch testing')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        print("Available models:")
        models = list_available_models()
        if models:
            for model in models:
                print(f"   - {model}")
        else:
            print("   - No models found")
    
    elif args.action == 'test':
        test_saved_ai(args.model, args.episodes, args.detailed, args.save_report)
    
    elif args.action == 'compare':
        if not args.model2:
            print("--model2 required for comparison")
            return
        compare_models(args.model, args.model2, args.episodes)
    
    elif args.action == 'batch':
        if not args.models:
            print("--models required for batch testing")
            return
        
        results = run_batch_tests(args.models, args.episodes)
        
        # Display results
        generator = TestReportGenerator()
        summary = generator.generate_text_summary(results)
        print(summary)
        
        # Save report if requested
        if args.save_report:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_test_report_{timestamp}.json"
            generator.generate_json_report(results, filename)


if __name__ == "__main__":
    main() 