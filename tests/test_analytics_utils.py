"""
🧪 Analytics Utilities Tests
Comprehensive unit tests for analytics_utils module functionality.

Test coverage:
- EpisodeMetrics dataclass functionality
- PerformanceAnalyzer comprehensive analysis
- Learning curve data generation
- Action pattern analysis
- Visualization plot creation
- Data export functionality
- Performance comparison between sessions
- Quick analysis utility functions
"""

import unittest
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add ml_training to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_training'))

try:
    from analytics_utils import (
        EpisodeMetrics, PerformanceAnalyzer, compare_performance_sessions,
        export_comparison_report, quick_episode_analysis
    )
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False


@unittest.skipIf(not ANALYTICS_AVAILABLE, "Analytics utilities not available")
class TestEpisodeMetrics(unittest.TestCase):
    """Test cases for EpisodeMetrics dataclass."""
    
    def test_episode_metrics_creation(self):
        """Test basic creation of EpisodeMetrics."""
        metrics = EpisodeMetrics(
            episode_num=1,
            steps=100,
            total_reward=50.5,
            success=True,
            terminated=True,
            truncated=False,
            duration=15.3,
            action_distribution={'up': 25, 'down': 25, 'left': 25, 'right': 25}
        )
        
        self.assertEqual(metrics.episode_num, 1)
        self.assertEqual(metrics.steps, 100)
        self.assertEqual(metrics.total_reward, 50.5)
        self.assertTrue(metrics.success)
        self.assertTrue(metrics.terminated)
        self.assertFalse(metrics.truncated)
        self.assertEqual(metrics.duration, 15.3)
        self.assertIsInstance(metrics.timestamp, str)
    
    def test_episode_metrics_with_optional_fields(self):
        """Test EpisodeMetrics with optional fields."""
        metrics = EpisodeMetrics(
            episode_num=5,
            steps=150,
            total_reward=75.2,
            success=True,
            terminated=True,
            truncated=False,
            duration=20.5,
            action_distribution={'up': 40, 'down': 30, 'left': 40, 'right': 40},
            key_pickup_step=50,
            escape_step=150
        )
        
        self.assertEqual(metrics.key_pickup_step, 50)
        self.assertEqual(metrics.escape_step, 150)
    
    def test_episode_metrics_timestamp_auto_generation(self):
        """Test that timestamp is automatically generated."""
        metrics = EpisodeMetrics(
            episode_num=1,
            steps=50,
            total_reward=25.0,
            success=False,
            terminated=False,
            truncated=True,
            duration=10.0,
            action_distribution={'up': 12, 'down': 13, 'left': 12, 'right': 13}
        )
        
        self.assertIsNotNone(metrics.timestamp)
        self.assertNotEqual(metrics.timestamp, "")


@unittest.skipIf(not ANALYTICS_AVAILABLE, "Analytics utilities not available")
class TestPerformanceAnalyzer(unittest.TestCase):
    """Test cases for PerformanceAnalyzer class."""
    
    def setUp(self):
        """Set up test analyzer with sample data."""
        self.analyzer = PerformanceAnalyzer("test_session")
        
        # Add sample episodes
        sample_episodes = [
            EpisodeMetrics(1, 100, 45.5, True, True, False, 12.3, 
                         {'up': 25, 'down': 25, 'left': 25, 'right': 25}, 50, 100),
            EpisodeMetrics(2, 150, -5.2, False, False, True, 18.7,
                         {'up': 40, 'down': 35, 'left': 35, 'right': 40}),
            EpisodeMetrics(3, 80, 62.1, True, True, False, 10.5,
                         {'up': 20, 'down': 20, 'left': 20, 'right': 20}, 40, 80),
            EpisodeMetrics(4, 200, -15.8, False, False, True, 25.2,
                         {'up': 50, 'down': 50, 'left': 50, 'right': 50}),
            EpisodeMetrics(5, 90, 55.3, True, True, False, 11.8,
                         {'up': 22, 'down': 23, 'left': 22, 'right': 23}, 45, 90)
        ]
        
        for episode in sample_episodes:
            self.analyzer.add_episode(episode)
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        analyzer = PerformanceAnalyzer("custom_session")
        self.assertEqual(analyzer.session_name, "custom_session")
        self.assertEqual(len(analyzer.episodes), 0)
    
    def test_add_episode_data_method(self):
        """Test adding episode data via convenience method."""
        analyzer = PerformanceAnalyzer("test")
        
        analyzer.add_episode_data(
            episode_num=1,
            steps=100,
            total_reward=50.0,
            success=True,
            terminated=True,
            truncated=False,
            duration=15.0,
            actions=[0, 1, 2, 3, 0, 1, 2, 3],
            key_pickup_step=40,
            escape_step=100
        )
        
        self.assertEqual(len(analyzer.episodes), 1)
        episode = analyzer.episodes[0]
        self.assertEqual(episode.episode_num, 1)
        self.assertEqual(episode.steps, 100)
        self.assertEqual(episode.total_reward, 50.0)
        self.assertTrue(episode.success)
        self.assertEqual(episode.key_pickup_step, 40)
        self.assertEqual(episode.escape_step, 100)
    
    def test_summary_statistics_generation(self):
        """Test comprehensive summary statistics generation."""
        summary = self.analyzer.get_summary_statistics()
        
        # Check session info
        self.assertEqual(summary['session_info']['name'], "test_session")
        self.assertEqual(summary['session_info']['total_episodes'], 5)
        
        # Check success metrics
        self.assertEqual(summary['success_metrics']['success_count'], 3)
        self.assertEqual(summary['success_metrics']['success_rate'], 60.0)
        
        # Check performance metrics
        expected_avg_steps = (100 + 150 + 80 + 200 + 90) / 5
        self.assertAlmostEqual(summary['performance_metrics']['avg_steps'], expected_avg_steps, places=2)
        
        # Check success-specific metrics
        expected_avg_success_steps = (100 + 80 + 90) / 3
        self.assertAlmostEqual(summary['success_specific_metrics']['avg_successful_steps'], expected_avg_success_steps, places=2)
        
        # Check termination analysis
        self.assertEqual(summary['termination_analysis']['natural_terminations'], 3)
        self.assertEqual(summary['termination_analysis']['truncations'], 2)
    
    def test_learning_curve_data_generation(self):
        """Test learning curve data generation."""
        curve_data = self.analyzer.get_learning_curve_data(window_size=3)
        
        self.assertIn('episodes', curve_data)
        self.assertIn('rewards', curve_data)
        self.assertIn('success_rates', curve_data)
        self.assertIn('avg_steps', curve_data)
        
        self.assertEqual(len(curve_data['episodes']), 5)
        self.assertEqual(len(curve_data['rewards']), 5)
        self.assertEqual(len(curve_data['success_rates']), 5)
        self.assertEqual(len(curve_data['avg_steps']), 5)
    
    def test_action_analysis_generation(self):
        """Test action pattern analysis."""
        action_analysis = self.analyzer.generate_action_analysis()
        
        self.assertIn('overall_distribution', action_analysis)
        self.assertIn('successful_distribution', action_analysis)
        self.assertIn('action_efficiency', action_analysis)
        
        # Check that all action types are present
        for action in ['up', 'down', 'left', 'right']:
            self.assertIn(action, action_analysis['overall_distribution'])
            self.assertIn(action, action_analysis['successful_distribution'])
            self.assertIn(action, action_analysis['action_efficiency'])
        
        # Check that percentages sum to approximately 100
        total_overall = sum(action_analysis['overall_distribution'].values())
        self.assertAlmostEqual(total_overall, 100.0, places=1)
    
    def test_empty_analyzer_handling(self):
        """Test analyzer handles empty data gracefully."""
        empty_analyzer = PerformanceAnalyzer("empty_test")
        
        summary = empty_analyzer.get_summary_statistics()
        self.assertEqual(summary, {})
        
        curve_data = empty_analyzer.get_learning_curve_data()
        self.assertEqual(curve_data, {})
        
        action_analysis = empty_analyzer.generate_action_analysis()
        self.assertEqual(action_analysis, {})
    
    def test_export_to_json(self):
        """Test JSON export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "test_export.json"
            
            self.analyzer.export_to_json(str(export_path))
            
            self.assertTrue(export_path.exists())
            
            # Verify exported data
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            
            self.assertIn('session_info', exported_data)
            self.assertIn('episodes', exported_data)
            self.assertIn('summary_statistics', exported_data)
            self.assertEqual(len(exported_data['episodes']), 5)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_visualization_plots_creation(self, mock_close, mock_savefig):
        """Test visualization plots creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.analyzer.create_visualization_plots(temp_dir)
            
            # Check that savefig was called for each plot type
            expected_calls = 4  # learning_curves, performance_distributions, action_analysis, success_analysis
            self.assertEqual(mock_savefig.call_count, expected_calls)
            self.assertEqual(mock_close.call_count, expected_calls)


@unittest.skipIf(not ANALYTICS_AVAILABLE, "Analytics utilities not available")
class TestPerformanceComparison(unittest.TestCase):
    """Test cases for performance comparison functionality."""
    
    def setUp(self):
        """Set up test analyzers for comparison."""
        # Create first analyzer with better performance
        self.analyzer1 = PerformanceAnalyzer("good_model")
        good_episodes = [
            EpisodeMetrics(i, 80 + i*5, 50 + i*5, True, True, False, 10 + i, 
                         {'up': 20, 'down': 20, 'left': 20, 'right': 20})
            for i in range(1, 6)
        ]
        for episode in good_episodes:
            self.analyzer1.add_episode(episode)
        
        # Create second analyzer with worse performance
        self.analyzer2 = PerformanceAnalyzer("poor_model")
        poor_episodes = [
            EpisodeMetrics(i, 150 + i*10, -10 - i*2, False, False, True, 20 + i*2,
                         {'up': 25, 'down': 25, 'left': 25, 'right': 25})
            for i in range(1, 6)
        ]
        for episode in poor_episodes:
            self.analyzer2.add_episode(episode)
    
    def test_compare_performance_sessions(self):
        """Test comparison of multiple performance sessions."""
        comparison = compare_performance_sessions(
            [self.analyzer1, self.analyzer2],
            ["Good Model", "Poor Model"]
        )
        
        self.assertIn('session_summaries', comparison)
        self.assertIn('comparative_metrics', comparison)
        self.assertIn('recommendations', comparison)
        
        # Check that both sessions are included
        self.assertIn("Good Model", comparison['session_summaries'])
        self.assertIn("Poor Model", comparison['session_summaries'])
        
        # Check comparative metrics
        self.assertIn('best_success_rate', comparison['comparative_metrics'])
        self.assertIn('most_efficient', comparison['comparative_metrics'])
        self.assertIn('highest_reward', comparison['comparative_metrics'])
        
        # Good model should win in all categories
        self.assertEqual(comparison['comparative_metrics']['best_success_rate']['session'], "Good Model")
        self.assertEqual(comparison['comparative_metrics']['highest_reward']['session'], "Good Model")
    
    def test_empty_comparison(self):
        """Test comparison with empty session list."""
        comparison = compare_performance_sessions([])
        self.assertEqual(comparison, {})
    
    def test_export_comparison_report(self):
        """Test exporting comparison report to file."""
        comparison = compare_performance_sessions(
            [self.analyzer1, self.analyzer2],
            ["Model A", "Model B"]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "comparison_report.json"
            export_comparison_report(comparison, str(export_path))
            
            self.assertTrue(export_path.exists())
            
            # Verify exported data
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            
            # Check key structure is preserved (noting that numpy types are converted to strings)
            self.assertIn('session_summaries', exported_data)
            self.assertIn('comparative_metrics', exported_data)
            self.assertIn('recommendations', exported_data)


@unittest.skipIf(not ANALYTICS_AVAILABLE, "Analytics utilities not available")
class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_quick_episode_analysis(self):
        """Test quick analysis utility function."""
        episodes_data = [
            {'steps': 100, 'success': True, 'total_reward': 50.0},
            {'steps': 150, 'success': False, 'total_reward': -10.0},
            {'steps': 80, 'success': True, 'total_reward': 60.0},
            {'steps': 200, 'success': False, 'total_reward': -20.0},
            {'steps': 90, 'success': True, 'total_reward': 55.0}
        ]
        
        analysis = quick_episode_analysis(episodes_data)
        
        self.assertEqual(analysis['total_episodes'], 5)
        self.assertEqual(analysis['success_count'], 3)
        self.assertEqual(analysis['success_rate'], 60.0)
        self.assertAlmostEqual(analysis['avg_steps'], 124.0, places=1)
        self.assertAlmostEqual(analysis['avg_reward'], 27.0, places=1)
    
    def test_quick_episode_analysis_empty(self):
        """Test quick analysis with empty data."""
        analysis = quick_episode_analysis([])
        self.assertEqual(analysis, {})
    
    def test_quick_episode_analysis_missing_fields(self):
        """Test quick analysis with missing fields."""
        episodes_data = [
            {'steps': 100},  # Missing success and total_reward
            {'success': True, 'total_reward': 50.0},  # Missing steps
        ]
        
        analysis = quick_episode_analysis(episodes_data)
        
        self.assertEqual(analysis['total_episodes'], 2)
        self.assertEqual(analysis['success_count'], 1)
        # Should handle missing fields gracefully with defaults


class TestAnalyticsIntegration(unittest.TestCase):
    """Integration tests for analytics utilities."""
    
    @unittest.skipIf(not ANALYTICS_AVAILABLE, "Analytics utilities not available")
    def test_full_analytics_workflow(self):
        """Test complete analytics workflow from data collection to export."""
        # Create analyzer
        analyzer = PerformanceAnalyzer("integration_test")
        
        # Simulate training session data
        for episode in range(1, 21):
            # Simulate improving performance over time
            success = episode > 10  # Succeed after episode 10
            steps = max(50, 200 - episode * 5)  # Decreasing steps over time
            reward = 50 if success else -10
            
            analyzer.add_episode_data(
                episode_num=episode,
                steps=steps,
                total_reward=reward,
                success=success,
                terminated=success,
                truncated=not success,
                duration=steps * 0.1,
                actions=[0, 1, 2, 3] * (steps // 4),
                key_pickup_step=steps // 2 if success else None,
                escape_step=steps if success else None
            )
        
        # Generate comprehensive analysis
        summary = analyzer.get_summary_statistics()
        curve_data = analyzer.get_learning_curve_data()
        action_analysis = analyzer.generate_action_analysis()
        
        # Verify analysis quality
        self.assertEqual(summary['session_info']['total_episodes'], 20)
        self.assertEqual(summary['success_metrics']['success_count'], 10)
        self.assertEqual(summary['success_metrics']['success_rate'], 50.0)
        
        # Test export functionality
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "integration_test.json"
            analyzer.export_to_json(str(export_path))
            
            self.assertTrue(export_path.exists())
            
            # Verify exported data completeness
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            
            self.assertIn('summary_statistics', exported_data)
            self.assertIn('learning_curve_data', exported_data)
            self.assertIn('action_analysis', exported_data)
            self.assertEqual(len(exported_data['episodes']), 20)


def run_analytics_tests():
    """Run all analytics utility tests."""
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEpisodeMetrics,
        TestPerformanceAnalyzer,
        TestPerformanceComparison,
        TestUtilityFunctions,
        TestAnalyticsIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    if ANALYTICS_AVAILABLE:
        print("🧪 Running Analytics Utilities Tests...")
        success = run_analytics_tests()
        if success:
            print("✅ All analytics tests passed!")
        else:
            print("❌ Some analytics tests failed!")
    else:
        print("⚠️ Analytics utilities not available for testing")
        unittest.main() 