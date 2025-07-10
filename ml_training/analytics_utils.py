"""
Analytics Utilities Module

Comprehensive performance analysis and visualization tools for AI training sessions.
Provides detailed analytics, learning curves, action pattern analysis, and statistical reporting.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import datetime
from pathlib import Path
import pandas as pd
from dataclasses import dataclass, asdict
import logging


@dataclass
class EpisodeMetrics:
    """
    Dataclass to store metrics for a single episode.
    
    Attributes:
        episode_num: Episode number
        steps: Number of steps taken
        total_reward: Total reward accumulated
        success: Whether episode was successful
        terminated: Whether episode terminated naturally
        truncated: Whether episode was truncated
        duration: Episode duration in seconds
        action_distribution: Distribution of actions taken
        key_pickup_step: Step when key was picked up (if any)
        escape_step: Step when escape occurred (if any)
    """
    episode_num: int
    steps: int
    total_reward: float
    success: bool
    terminated: bool
    truncated: bool
    duration: float
    action_distribution: Dict[str, int]
    key_pickup_step: Optional[int] = None
    escape_step: Optional[int] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for training and testing sessions.
    
    This class provides detailed analytics including:
    - Episode-by-episode performance tracking
    - Success rate and efficiency metrics
    - Action pattern analysis
    - Learning curve visualization
    - Statistical performance summaries
    """
    
    def __init__(self, session_name: str = "training_session"):
        """
        Initialize performance analyzer.
        
        Args:
            session_name: Name identifier for this analysis session
        """
        self.session_name = session_name
        self.episodes: List[EpisodeMetrics] = []
        self.start_time = datetime.datetime.now()
        self.logger = logging.getLogger(f'analytics.{session_name}')
    
    def add_episode(self, metrics: EpisodeMetrics) -> None:
        """Add episode metrics to the analyzer."""
        self.episodes.append(metrics)
        self.logger.debug(f"Added episode {metrics.episode_num} with {metrics.steps} steps")
    
    def add_episode_data(self, episode_num: int, steps: int, total_reward: float,
                        success: bool, terminated: bool, truncated: bool,
                        duration: float, actions: List[int],
                        key_pickup_step: Optional[int] = None,
                        escape_step: Optional[int] = None) -> None:
        """
        Add episode data by providing individual parameters.
        
        Args:
            episode_num: Episode number
            steps: Number of steps taken
            total_reward: Total accumulated reward
            success: Whether episode was successful
            terminated: Natural termination flag
            truncated: Truncation flag
            duration: Episode duration in seconds
            actions: List of actions taken during episode
            key_pickup_step: Step when key was picked up
            escape_step: Step when escape occurred
        """
        action_distribution = self._calculate_action_distribution(actions)
        
        metrics = EpisodeMetrics(
            episode_num=episode_num,
            steps=steps,
            total_reward=total_reward,
            success=success,
            terminated=terminated,
            truncated=truncated,
            duration=duration,
            action_distribution=action_distribution,
            key_pickup_step=key_pickup_step,
            escape_step=escape_step
        )
        
        self.add_episode(metrics)
    
    def _calculate_action_distribution(self, actions: List[int]) -> Dict[str, int]:
        """Calculate distribution of actions taken."""
        action_names = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        distribution = {name: 0 for name in action_names.values()}
        
        for action in actions:
            if action in action_names:
                distribution[action_names[action]] += 1
        
        return distribution
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics.
        
        Returns:
            Dictionary containing detailed performance statistics
        """
        if not self.episodes:
            return {}
        
        # Basic metrics
        total_episodes = len(self.episodes)
        successful_episodes = [ep for ep in self.episodes if ep.success]
        success_count = len(successful_episodes)
        
        # Calculate various statistics
        steps_data = [ep.steps for ep in self.episodes]
        rewards_data = [ep.total_reward for ep in self.episodes]
        durations_data = [ep.duration for ep in self.episodes]
        
        # Success-specific metrics
        if successful_episodes:
            successful_steps = [ep.steps for ep in successful_episodes]
            successful_rewards = [ep.total_reward for ep in successful_episodes]
            key_pickup_times = [ep.key_pickup_step for ep in successful_episodes 
                              if ep.key_pickup_step is not None]
            escape_times = [ep.escape_step for ep in successful_episodes 
                          if ep.escape_step is not None]
        else:
            successful_steps = []
            successful_rewards = []
            key_pickup_times = []
            escape_times = []
        
        summary = {
            'session_info': {
                'name': self.session_name,
                'total_episodes': total_episodes,
                'analysis_duration': (datetime.datetime.now() - self.start_time).total_seconds(),
                'episodes_per_minute': total_episodes / max(1, (datetime.datetime.now() - self.start_time).total_seconds() / 60)
            },
            'success_metrics': {
                'success_count': success_count,
                'success_rate': (success_count / total_episodes) * 100,
                'failure_count': total_episodes - success_count,
                'failure_rate': ((total_episodes - success_count) / total_episodes) * 100
            },
            'performance_metrics': {
                'avg_steps': np.mean(steps_data),
                'min_steps': np.min(steps_data),
                'max_steps': np.max(steps_data),
                'std_steps': np.std(steps_data),
                'avg_reward': np.mean(rewards_data),
                'min_reward': np.min(rewards_data),
                'max_reward': np.max(rewards_data),
                'std_reward': np.std(rewards_data),
                'avg_duration': np.mean(durations_data),
                'total_duration': np.sum(durations_data)
            },
            'success_specific_metrics': {
                'avg_successful_steps': np.mean(successful_steps) if successful_steps else 0,
                'min_successful_steps': np.min(successful_steps) if successful_steps else 0,
                'avg_successful_reward': np.mean(successful_rewards) if successful_rewards else 0,
                'avg_key_pickup_time': np.mean(key_pickup_times) if key_pickup_times else 0,
                'avg_escape_time': np.mean(escape_times) if escape_times else 0
            },
            'termination_analysis': {
                'natural_terminations': sum(1 for ep in self.episodes if ep.terminated),
                'truncations': sum(1 for ep in self.episodes if ep.truncated),
                'successful_terminations': sum(1 for ep in self.episodes if ep.success and ep.terminated),
                'timeout_failures': sum(1 for ep in self.episodes if not ep.success and ep.truncated)
            }
        }
        
        return summary
    
    def get_learning_curve_data(self, window_size: int = 100) -> Dict[str, List[float]]:
        """
        Generate data for learning curve visualization.
        
        Args:
            window_size: Size of moving average window
            
        Returns:
            Dictionary with learning curve data
        """
        if not self.episodes:
            return {}
        
        episodes_nums = [ep.episode_num for ep in self.episodes]
        rewards = [ep.total_reward for ep in self.episodes]
        success_rates = []
        avg_steps = []
        
        # Calculate moving averages
        for i in range(len(self.episodes)):
            start_idx = max(0, i - window_size + 1)
            window_episodes = self.episodes[start_idx:i+1]
            
            # Success rate in window
            window_successes = sum(1 for ep in window_episodes if ep.success)
            success_rate = (window_successes / len(window_episodes)) * 100
            success_rates.append(success_rate)
            
            # Average steps in window
            window_steps = [ep.steps for ep in window_episodes]
            avg_step = np.mean(window_steps)
            avg_steps.append(avg_step)
        
        return {
            'episodes': episodes_nums,
            'rewards': rewards,
            'success_rates': success_rates,
            'avg_steps': avg_steps
        }
    
    def generate_action_analysis(self) -> Dict[str, Any]:
        """
        Analyze action patterns across episodes.
        
        Returns:
            Dictionary containing action pattern analysis
        """
        if not self.episodes:
            return {}
        
        all_actions = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
        successful_actions = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
        
        for episode in self.episodes:
            for action, count in episode.action_distribution.items():
                all_actions[action] += count
                if episode.success:
                    successful_actions[action] += count
        
        total_actions = sum(all_actions.values())
        total_successful_actions = sum(successful_actions.values())
        
        analysis = {
            'overall_distribution': {
                action: (count / total_actions) * 100 if total_actions > 0 else 0
                for action, count in all_actions.items()
            },
            'successful_distribution': {
                action: (count / total_successful_actions) * 100 if total_successful_actions > 0 else 0
                for action, count in successful_actions.items()
            },
            'action_efficiency': {},
            'total_actions': total_actions,
            'successful_actions': total_successful_actions
        }
        
        # Calculate action efficiency (success rate per action type)
        for action in all_actions:
            if all_actions[action] > 0:
                efficiency = (successful_actions[action] / all_actions[action]) * 100
                analysis['action_efficiency'][action] = efficiency
            else:
                analysis['action_efficiency'][action] = 0
        
        return analysis
    
    def export_to_json(self, filepath: str) -> None:
        """
        Export all analytics data to JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        export_data = {
            'session_info': {
                'name': self.session_name,
                'start_time': self.start_time.isoformat(),
                'export_time': datetime.datetime.now().isoformat(),
                'total_episodes': len(self.episodes)
            },
            'episodes': [asdict(ep) for ep in self.episodes],
            'summary_statistics': self.get_summary_statistics(),
            'learning_curve_data': self.get_learning_curve_data(),
            'action_analysis': self.generate_action_analysis()
        }
        
        with open(filepath, 'w') as f:
                            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Analytics data exported to {filepath}")
    
    def create_visualization_plots(self, save_dir: str = "analytics_plots") -> None:
        """
        Create comprehensive visualization plots.
        
        Args:
            save_dir: Directory to save plot images
        """
        if not self.episodes:
            self.logger.warning("No episodes to visualize")
            return
        
        # Ensure save directory exists
        Path(save_dir).mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Learning curve plot
        self._plot_learning_curves(save_dir)
        
        # 2. Performance distribution plots
        self._plot_performance_distributions(save_dir)
        
        # 3. Action analysis plots
        self._plot_action_analysis(save_dir)
        
        # 4. Success/failure analysis
        self._plot_success_analysis(save_dir)
        
        self.logger.info(f"Visualization plots saved to {save_dir}")
    
    def _plot_learning_curves(self, save_dir: str) -> None:
        """Create learning curve plots."""
        curve_data = self.get_learning_curve_data()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Learning Curves - {self.session_name}', fontsize=16)
        
        # Episode rewards
        ax1.plot(curve_data['episodes'], curve_data['rewards'], alpha=0.6, linewidth=0.5)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True, alpha=0.3)
        
        # Success rate
        ax2.plot(curve_data['episodes'], curve_data['success_rates'], color='green', linewidth=2)
        ax2.set_title('Success Rate (Moving Average)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate (%)')
        ax2.grid(True, alpha=0.3)
        
        # Average steps
        ax3.plot(curve_data['episodes'], curve_data['avg_steps'], color='orange', linewidth=2)
        ax3.set_title('Average Steps per Episode')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Steps')
        ax3.grid(True, alpha=0.3)
        
        # Success vs Steps scatter
        successful_eps = [ep for ep in self.episodes if ep.success]
        failed_eps = [ep for ep in self.episodes if not ep.success]
        
        if successful_eps:
            ax4.scatter([ep.episode_num for ep in successful_eps], 
                       [ep.steps for ep in successful_eps], 
                       color='green', alpha=0.6, label='Success')
        if failed_eps:
            ax4.scatter([ep.episode_num for ep in failed_eps], 
                       [ep.steps for ep in failed_eps], 
                       color='red', alpha=0.6, label='Failure')
        
        ax4.set_title('Episode Outcomes')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Steps Taken')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/learning_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_distributions(self, save_dir: str) -> None:
        """Create performance distribution plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Performance Distributions - {self.session_name}', fontsize=16)
        
        steps_data = [ep.steps for ep in self.episodes]
        rewards_data = [ep.total_reward for ep in self.episodes]
        durations_data = [ep.duration for ep in self.episodes]
        
        # Steps distribution
        ax1.hist(steps_data, bins=30, alpha=0.7, edgecolor='black')
        ax1.set_title('Distribution of Steps per Episode')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Frequency')
        ax1.axvline(np.mean(steps_data), color='red', linestyle='--', label=f'Mean: {np.mean(steps_data):.1f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Rewards distribution
        ax2.hist(rewards_data, bins=30, alpha=0.7, edgecolor='black', color='green')
        ax2.set_title('Distribution of Total Rewards')
        ax2.set_xlabel('Total Reward')
        ax2.set_ylabel('Frequency')
        ax2.axvline(np.mean(rewards_data), color='red', linestyle='--', label=f'Mean: {np.mean(rewards_data):.1f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Duration distribution
        ax3.hist(durations_data, bins=30, alpha=0.7, edgecolor='black', color='orange')
        ax3.set_title('Distribution of Episode Durations')
        ax3.set_xlabel('Duration (seconds)')
        ax3.set_ylabel('Frequency')
        ax3.axvline(np.mean(durations_data), color='red', linestyle='--', label=f'Mean: {np.mean(durations_data):.1f}s')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Success vs failure comparison
        successful_steps = [ep.steps for ep in self.episodes if ep.success]
        failed_steps = [ep.steps for ep in self.episodes if not ep.success]
        
        if successful_steps and failed_steps:
            ax4.hist([successful_steps, failed_steps], bins=20, alpha=0.7, 
                    label=['Successful', 'Failed'], color=['green', 'red'])
            ax4.set_title('Steps Distribution: Success vs Failure')
            ax4.set_xlabel('Steps')
            ax4.set_ylabel('Frequency')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for comparison', 
                    ha='center', va='center', transform=ax4.transAxes)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/performance_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_action_analysis(self, save_dir: str) -> None:
        """Create action analysis plots."""
        action_data = self.generate_action_analysis()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Action Analysis - {self.session_name}', fontsize=16)
        
        actions = list(action_data['overall_distribution'].keys())
        overall_values = list(action_data['overall_distribution'].values())
        successful_values = list(action_data['successful_distribution'].values())
        efficiency_values = list(action_data['action_efficiency'].values())
        
        # Overall action distribution
        ax1.pie(overall_values, labels=actions, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Overall Action Distribution')
        
        # Successful action distribution
        ax2.pie(successful_values, labels=actions, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Action Distribution in Successful Episodes')
        
        # Action efficiency bar chart
        ax3.bar(actions, efficiency_values, color=['blue', 'orange', 'green', 'red'])
        ax3.set_title('Action Efficiency (Success Rate per Action)')
        ax3.set_xlabel('Action')
        ax3.set_ylabel('Efficiency (%)')
        ax3.grid(True, alpha=0.3)
        
        # Comparison of overall vs successful
        x = np.arange(len(actions))
        width = 0.35
        
        ax4.bar(x - width/2, overall_values, width, label='Overall', alpha=0.7)
        ax4.bar(x + width/2, successful_values, width, label='Successful', alpha=0.7)
        ax4.set_title('Action Distribution Comparison')
        ax4.set_xlabel('Action')
        ax4.set_ylabel('Percentage (%)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(actions)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/action_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_success_analysis(self, save_dir: str) -> None:
        """Create success/failure analysis plots."""
        summary = self.get_summary_statistics()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Success Analysis - {self.session_name}', fontsize=16)
        
        # Success rate pie chart
        success_data = [summary['success_metrics']['success_count'], 
                       summary['success_metrics']['failure_count']]
        labels = ['Success', 'Failure']
        colors = ['green', 'red']
        
        ax1.pie(success_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f"Overall Success Rate: {summary['success_metrics']['success_rate']:.1f}%")
        
        # Termination types
        term_data = summary['termination_analysis']
        term_labels = ['Natural\nTerminations', 'Truncations', 'Successful\nTerminations', 'Timeout\nFailures']
        term_values = [term_data['natural_terminations'], term_data['truncations'], 
                      term_data['successful_terminations'], term_data['timeout_failures']]
        
        ax2.bar(term_labels, term_values, color=['blue', 'orange', 'green', 'red'])
        ax2.set_title('Episode Termination Types')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Success rate over time (if enough episodes)
        if len(self.episodes) >= 10:
            window_size = max(10, len(self.episodes) // 10)
            success_rates = []
            episode_windows = []
            
            for i in range(window_size, len(self.episodes) + 1, window_size):
                window_episodes = self.episodes[i-window_size:i]
                successes = sum(1 for ep in window_episodes if ep.success)
                rate = (successes / len(window_episodes)) * 100
                success_rates.append(rate)
                episode_windows.append(i)
            
            ax3.plot(episode_windows, success_rates, marker='o', linewidth=2, markersize=8)
            ax3.set_title(f'Success Rate Over Time (Window: {window_size})')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Success Rate (%)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Not enough episodes\nfor trend analysis', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Success Rate Trend (Insufficient Data)')
        
        # Key performance metrics
        perf_metrics = summary['performance_metrics']
        metrics_names = ['Avg Steps', 'Avg Reward', 'Avg Duration (s)']
        metrics_values = [perf_metrics['avg_steps'], perf_metrics['avg_reward'], 
                         perf_metrics['avg_duration']]
        
        # Normalize values for better visualization
        normalized_values = [(val - min(metrics_values)) / (max(metrics_values) - min(metrics_values)) 
                           if max(metrics_values) != min(metrics_values) else 0.5 
                           for val in metrics_values]
        
        bars = ax4.bar(metrics_names, normalized_values, color=['blue', 'green', 'orange'])
        ax4.set_title('Key Performance Metrics (Normalized)')
        ax4.set_ylabel('Normalized Value')
        
        # Add actual values as text on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/success_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def compare_performance_sessions(sessions: List[PerformanceAnalyzer], 
                               session_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compare performance across multiple training/testing sessions.
    
    Args:
        sessions: List of PerformanceAnalyzer instances to compare
        session_names: Optional custom names for sessions
        
    Returns:
        Dictionary containing comparison analysis
    """
    if not sessions:
        return {}
    
    if session_names is None:
        session_names = [f"Session {i+1}" for i in range(len(sessions))]
    
    comparison = {
        'session_summaries': {},
        'comparative_metrics': {},
        'recommendations': []
    }
    
    # Collect summaries from each session
    for session, name in zip(sessions, session_names):
        summary = session.get_summary_statistics()
        comparison['session_summaries'][name] = summary
    
    # Calculate comparative metrics
    success_rates = []
    avg_steps = []
    avg_rewards = []
    
    for name in session_names:
        summary = comparison['session_summaries'][name]
        if summary:
            success_rates.append(summary['success_metrics']['success_rate'])
            avg_steps.append(summary['performance_metrics']['avg_steps'])
            avg_rewards.append(summary['performance_metrics']['avg_reward'])
    
    if success_rates:
        best_success_idx = np.argmax(success_rates)
        best_efficiency_idx = np.argmin(avg_steps)
        best_reward_idx = np.argmax(avg_rewards)
        
        comparison['comparative_metrics'] = {
            'best_success_rate': {
                'session': session_names[best_success_idx],
                'value': success_rates[best_success_idx]
            },
            'most_efficient': {
                'session': session_names[best_efficiency_idx],
                'value': avg_steps[best_efficiency_idx]
            },
            'highest_reward': {
                'session': session_names[best_reward_idx],
                'value': avg_rewards[best_reward_idx]
            }
        }
        
        # Generate recommendations
        if best_success_idx == best_efficiency_idx == best_reward_idx:
            comparison['recommendations'].append(
                f"{session_names[best_success_idx]} performs best across all metrics"
            )
        else:
            comparison['recommendations'].append(
                f"Consider {session_names[best_success_idx]} for success rate, "
                f"{session_names[best_efficiency_idx]} for efficiency"
            )
    
    return comparison


def export_comparison_report(comparison: Dict[str, Any], filepath: str) -> None:
    """
    Export session comparison to JSON file.
    
    Args:
        comparison: Comparison data from compare_performance_sessions
        filepath: Path to save the comparison report
    """
    with open(filepath, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)


# Utility functions for quick analytics
def quick_episode_analysis(episodes_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Quick analysis function for episode data in dictionary format.
    
    Args:
        episodes_data: List of episode dictionaries
        
    Returns:
        Dictionary with basic analytics
    """
    if not episodes_data:
        return {}
    
    success_count = sum(1 for ep in episodes_data if ep.get('success', False))
    total_episodes = len(episodes_data)
    
    return {
        'total_episodes': total_episodes,
        'success_count': success_count,
        'success_rate': (success_count / total_episodes) * 100,
        'avg_steps': np.mean([ep.get('steps', 0) for ep in episodes_data]),
        'avg_reward': np.mean([ep.get('total_reward', 0) for ep in episodes_data])
    } 