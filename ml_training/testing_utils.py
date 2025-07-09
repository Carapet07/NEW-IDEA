"""
🧪 Testing Utilities Module
Comprehensive utilities for testing, evaluation, and benchmarking AI models.

This module provides:
- Automated testing frameworks
- Performance benchmarking tools
- Model evaluation and validation
- A/B testing utilities
- Test report generation
- Environment testing helpers
"""

import numpy as np
import time
import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from pathlib import Path
import json
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
                # Check timeout
                if self._check_timeout():
                    self.logger.warning("Test timed out")
                    break
                
                result = self._run_single_episode(model, env, episode + 1)
                self.results.append(result)
                
                # Progress callback
                if progress_callback:
                    progress_callback(episode + 1, self.config.num_episodes, result)
                
                # Log progress
                if (episode + 1) % max(1, self.config.num_episodes // 10) == 0:
                    self.logger.info(f"Completed episode {episode + 1}/{self.config.num_episodes}")
                    
            except Exception as e:
                self.logger.error(f"Error in episode {episode + 1}: {e}")
                error_result = TestResult(
                    episode_num=episode + 1,
                    steps_taken=0,
                    total_reward=0.0,
                    final_reward=0.0,
                    success=False,
                    terminated=False,
                    truncated=False,
                    duration=0.0,
                    error=str(e)
                )
                self.results.append(error_result)
        
        self.end_time = datetime.datetime.now()
        
        # Generate summary
        summary = self._generate_test_summary()
        success_rate = summary.get('success_metrics', {}).get('success_rate', 0)
        self.logger.info(f"Test completed. Success rate: {success_rate:.1f}%")
        
        return {
            'config': asdict(self.config),
            'summary': summary,
            'results': [asdict(result) for result in self.results]
        }
    
    def _run_single_episode(self, model: BaseAlgorithm, env, episode_num: int) -> TestResult:
        """Run a single test episode."""
        episode_start = time.time()
        
        # Reset environment
        obs, _ = env.reset()
        
        # Initialize tracking
        actions = [] if self.config.save_actions else None
        observations = [] if self.config.save_observations else None
        total_reward = 0.0
        steps = 0
        
        # Run episode
        while steps < self.config.max_steps_per_episode:
            # Get action from model
            action, _states = model.predict(obs, deterministic=self.config.deterministic)
            
            # Record data if requested
            if actions is not None:
                actions.append(int(action))
            if observations is not None:
                observations.append(obs.copy())
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        episode_duration = time.time() - episode_start
        
        # Create result
        result = TestResult(
            episode_num=episode_num,
            steps_taken=steps,
            total_reward=total_reward,
            final_reward=reward,
            success=info.get('success', False),
            terminated=terminated,
            truncated=truncated,
            duration=episode_duration,
            actions=actions,
            observations=observations,
            info=info
        )
        
        return result
    
    def _check_timeout(self) -> bool:
        """Check if test has timed out."""
        if self.start_time is None:
            return False
        
        elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
        return elapsed > self.config.timeout_seconds
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        if not self.results:
            return {}
        
        # Filter out error results for statistics
        valid_results = [r for r in self.results if r.error is None]
        successful_results = [r for r in valid_results if r.success]
        
        total_episodes = len(self.results)
        valid_episodes = len(valid_results)
        successful_episodes = len(successful_results)
        
        # Calculate basic statistics
        if valid_results:
            steps_data = [r.steps_taken for r in valid_results]
            rewards_data = [r.total_reward for r in valid_results]
            durations_data = [r.duration for r in valid_results]
        else:
            steps_data = rewards_data = durations_data = []
        
        # Calculate success-specific statistics
        if successful_results:
            success_steps = [r.steps_taken for r in successful_results]
            success_rewards = [r.total_reward for r in successful_results]
        else:
            success_steps = success_rewards = []
        
        summary = {
            'test_info': {
                'test_name': self.config.test_name,
                'total_episodes': total_episodes,
                'valid_episodes': valid_episodes,
                'error_episodes': total_episodes - valid_episodes,
                'test_duration': (self.end_time - self.start_time).total_seconds() if self.end_time else 0
            },
            'success_metrics': {
                'successful_episodes': successful_episodes,
                'success_rate': (successful_episodes / valid_episodes) * 100 if valid_episodes > 0 else 0,
                'failure_rate': ((valid_episodes - successful_episodes) / valid_episodes) * 100 if valid_episodes > 0 else 0
            },
            'performance_metrics': {
                'avg_steps': np.mean(steps_data) if steps_data else 0,
                'min_steps': np.min(steps_data) if steps_data else 0,
                'max_steps': np.max(steps_data) if steps_data else 0,
                'std_steps': np.std(steps_data) if steps_data else 0,
                'avg_reward': np.mean(rewards_data) if rewards_data else 0,
                'min_reward': np.min(rewards_data) if rewards_data else 0,
                'max_reward': np.max(rewards_data) if rewards_data else 0,
                'std_reward': np.std(rewards_data) if rewards_data else 0,
                'avg_duration': np.mean(durations_data) if durations_data else 0,
                'total_test_time': sum(durations_data) if durations_data else 0
            },
            'success_specific_metrics': {
                'avg_success_steps': np.mean(success_steps) if success_steps else 0,
                'min_success_steps': np.min(success_steps) if success_steps else 0,
                'avg_success_reward': np.mean(success_rewards) if success_rewards else 0
            },
            'termination_analysis': {
                'natural_terminations': sum(1 for r in valid_results if r.terminated),
                'truncations': sum(1 for r in valid_results if r.truncated),
                'successful_natural': sum(1 for r in valid_results if r.success and r.terminated),
                'timeout_failures': sum(1 for r in valid_results if not r.success and r.truncated)
            }
        }
        
        return summary


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
    
    def run_model_benchmark(self, models: Dict[str, BaseAlgorithm], env,
                          test_config: TestConfiguration) -> Dict[str, Any]:
        """
        Run benchmark tests on multiple models.
        
        Args:
            models: Dictionary mapping model names to model instances
            env: Environment to test in
            test_config: Test configuration
            
        Returns:
            Dictionary containing benchmark results
        """
        self.logger.info(f"Starting benchmark with {len(models)} models")
        
        for model_name, model in models.items():
            self.logger.info(f"Testing model: {model_name}")
            
            # Create test runner for this model
            model_config = TestConfiguration(
                test_name=f"{self.benchmark_name}_{model_name}",
                num_episodes=test_config.num_episodes,
                max_steps_per_episode=test_config.max_steps_per_episode,
                deterministic=test_config.deterministic,
                render=False,  # Disable rendering for benchmarks
                save_actions=test_config.save_actions,
                save_observations=test_config.save_observations
            )
            
            test_runner = TestRunner(model_config)
            results = test_runner.run_model_test(model, env)
            self.add_model_results(model_name, results)
        
        # Generate comparison report
        comparison = self._generate_comparison_report()
        
        return {
            'benchmark_name': self.benchmark_name,
            'models_tested': list(models.keys()),
            'test_config': asdict(test_config),
            'individual_results': self.model_results,
            'comparison': comparison
        }
    
    def _generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        if not self.model_results:
            return {
                'rankings': {},
                'statistical_comparison': {},
                'recommendations': []
            }
        
        comparison = {
            'rankings': {},
            'statistical_comparison': {},
            'recommendations': []
        }
        
        # Extract key metrics for comparison
        metrics = {}
        for model_name, results in self.model_results.items():
            summary = results.get('summary', {})
            if summary:
                success_metrics = summary.get('success_metrics', {})
                performance_metrics = summary.get('performance_metrics', {})
                success_specific_metrics = summary.get('success_specific_metrics', {})
                
                metrics[model_name] = {
                    'success_rate': success_metrics.get('success_rate', 0),
                    'avg_steps': performance_metrics.get('avg_steps', 0),
                    'avg_reward': performance_metrics.get('avg_reward', 0),
                    'avg_success_steps': success_specific_metrics.get('avg_success_steps', 0)
                }
        
        if not metrics:
            return comparison
        
        # Generate rankings
        comparison['rankings'] = {
            'by_success_rate': sorted(metrics.items(), 
                                    key=lambda x: x[1]['success_rate'], reverse=True),
            'by_efficiency': sorted(metrics.items(), 
                                  key=lambda x: x[1]['avg_success_steps']),
            'by_reward': sorted(metrics.items(), 
                              key=lambda x: x[1]['avg_reward'], reverse=True)
        }
        
        # Best performers
        best_success = comparison['rankings']['by_success_rate'][0]
        best_efficiency = comparison['rankings']['by_efficiency'][0]
        best_reward = comparison['rankings']['by_reward'][0]
        
        comparison['best_performers'] = {
            'highest_success_rate': best_success[0],
            'most_efficient': best_efficiency[0],
            'highest_reward': best_reward[0]
        }
        
        # Generate recommendations
        if best_success[0] == best_efficiency[0] == best_reward[0]:
            comparison['recommendations'].append(
                f"{best_success[0]} is the clear winner across all metrics"
            )
        else:
            comparison['recommendations'].append(
                f"For success rate: {best_success[0]} ({best_success[1]['success_rate']:.1f}%)"
            )
            comparison['recommendations'].append(
                f"For efficiency: {best_efficiency[0]} ({best_efficiency[1]['avg_success_steps']:.1f} steps)"
            )
            comparison['recommendations'].append(
                f"For reward: {best_reward[0]} ({best_reward[1]['avg_reward']:.2f} avg reward)"
            )
        
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
        Generate CSV report with episode-level data.
        
        Args:
            test_results: Results from TestRunner
            output_path: Path to save the CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            results = test_results.get('results', [])
            if not results:
                return False
            
            # Get fieldnames from first result
            fieldnames = list(results[0].keys())
            
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    # Convert complex types to strings for CSV
                    csv_row = {}
                    for key, value in result.items():
                        if isinstance(value, (list, dict)):
                            csv_row[key] = json.dumps(value)
                        else:
                            csv_row[key] = value
                    writer.writerow(csv_row)
            
            self.logger.info(f"CSV report saved to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate CSV report: {e}")
            return False
    
    def generate_text_summary(self, test_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable text summary.
        
        Args:
            test_results: Results from TestRunner or ModelBenchmark
            
        Returns:
            Text summary string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("TEST REPORT SUMMARY")
        lines.append("=" * 60)
        
        # Handle different result types
        if 'benchmark_name' in test_results:
            # Benchmark results
            lines.append(f"Benchmark: {test_results['benchmark_name']}")
            lines.append(f"Models tested: {', '.join(test_results['models_tested'])}")
            
            comparison = test_results.get('comparison', {})
            if 'best_performers' in comparison:
                lines.append("\nBest Performers:")
                for metric, model in comparison['best_performers'].items():
                    lines.append(f"  {metric}: {model}")
            
        else:
            # Single model test results
            config = test_results.get('config', {})
            summary = test_results.get('summary', {})
            
            lines.append(f"Test: {config.get('test_name', 'Unknown')}")
            lines.append(f"Episodes: {config.get('num_episodes', 0)}")
            
            if summary:
                success_metrics = summary.get('success_metrics', {})
                perf_metrics = summary.get('performance_metrics', {})
                
                lines.append(f"\nSuccess Rate: {success_metrics.get('success_rate', 0):.1f}%")
                lines.append(f"Average Steps: {perf_metrics.get('avg_steps', 0):.1f}")
                lines.append(f"Average Reward: {perf_metrics.get('avg_reward', 0):.2f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# Utility functions for common testing operations

def quick_model_test(model_path: str, env, episodes: int = 10) -> Dict[str, Any]:
    """
    Quick test of a model with minimal configuration.
    
    Args:
        model_path: Path to the model file
        env: Environment to test in
        episodes: Number of episodes to run
        
    Returns:
        Test results dictionary
    """
    try:
        model = PPO.load(model_path)
        
        config = TestConfiguration(
            test_name=f"quick_test_{Path(model_path).stem}",
            num_episodes=episodes,
            deterministic=True
        )
        
        runner = TestRunner(config)
        return runner.run_model_test(model, env)
        
    except Exception as e:
        return {'error': str(e)}


def compare_models_quick(model_paths: List[str], env, 
                        episodes: int = 10) -> Dict[str, Any]:
    """
    Quick comparison of multiple models.
    
    Args:
        model_paths: List of paths to model files
        env: Environment to test in
        episodes: Number of episodes per model
        
    Returns:
        Benchmark results dictionary
    """
    models = {}
    
    for path in model_paths:
        try:
            model_name = Path(path).stem
            model = PPO.load(path)
            models[model_name] = model
        except Exception as e:
            logging.error(f"Failed to load model {path}: {e}")
    
    if not models:
        return {'error': 'No models loaded successfully'}
    
    config = TestConfiguration(
        test_name="quick_comparison",
        num_episodes=episodes,
        deterministic=True
    )
    
    benchmark = ModelBenchmark("quick_benchmark")
    return benchmark.run_model_benchmark(models, env, config)


def validate_environment_performance(env, num_episodes: int = 100) -> Dict[str, Any]:
    """
    Test environment performance and stability.
    
    Args:
        env: Environment to test
        num_episodes: Number of episodes to run
        
    Returns:
        Environment performance metrics
    """
    results = {
        'total_episodes': num_episodes,
        'successful_resets': 0,
        'successful_steps': 0,
        'total_steps': 0,
        'reset_times': [],
        'step_times': [],
        'errors': []
    }
    
    for episode in range(num_episodes):
        try:
            # Test reset
            start_time = time.time()
            obs, _ = env.reset()
            reset_time = time.time() - start_time
            results['reset_times'].append(reset_time)
            results['successful_resets'] += 1
            
            # Test steps
            for step in range(100):  # Test 100 steps per episode
                start_time = time.time()
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                step_time = time.time() - start_time
                
                results['step_times'].append(step_time)
                results['successful_steps'] += 1
                results['total_steps'] += 1
                
                if terminated or truncated:
                    break
                    
        except Exception as e:
            results['errors'].append(f"Episode {episode}: {str(e)}")
    
    # Calculate summary statistics
    if results['reset_times']:
        results['avg_reset_time'] = np.mean(results['reset_times'])
        results['max_reset_time'] = np.max(results['reset_times'])
    
    if results['step_times']:
        results['avg_step_time'] = np.mean(results['step_times'])
        results['max_step_time'] = np.max(results['step_times'])
        results['steps_per_second'] = 1.0 / np.mean(results['step_times'])
    
    results['reset_success_rate'] = (results['successful_resets'] / num_episodes) * 100
    results['step_success_rate'] = (results['successful_steps'] / results['total_steps']) * 100 if results['total_steps'] > 0 else 0
    
    return results 