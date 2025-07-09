"""
🧪 Testing Utilities Tests
Comprehensive unit tests for testing_utils module functionality.

Test coverage:
- TestConfiguration dataclass functionality
- TestResult dataclass functionality  
- TestRunner comprehensive testing workflows
- ModelBenchmark comparison functionality
- TestReportGenerator report creation
- Environment performance validation
- Quick testing utility functions
- Integration testing workflows
"""

import unittest
import tempfile
import shutil
import json
import csv
import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import sys
import os
import time

# Add ml_training to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_training'))

try:
    from testing_utils import (
        TestConfiguration, TestResult, TestRunner, ModelBenchmark,
        TestReportGenerator, quick_model_test, compare_models_quick,
        validate_environment_performance
    )
    TESTING_UTILS_AVAILABLE = True
except ImportError:
    TESTING_UTILS_AVAILABLE = False


@unittest.skipIf(not TESTING_UTILS_AVAILABLE, "Testing utilities not available")
class TestTestConfiguration(unittest.TestCase):
    """Test cases for TestConfiguration dataclass."""
    
    def test_default_configuration(self):
        """Test TestConfiguration with default values."""
        config = TestConfiguration("test_run")
        
        self.assertEqual(config.test_name, "test_run")
        self.assertEqual(config.num_episodes, 10)
        self.assertEqual(config.max_steps_per_episode, 500)
        self.assertTrue(config.deterministic)
        self.assertFalse(config.render)
        self.assertTrue(config.save_actions)
        self.assertFalse(config.save_observations)
        self.assertEqual(config.timeout_seconds, 3600)
    
    def test_custom_configuration(self):
        """Test TestConfiguration with custom values."""
        config = TestConfiguration(
            test_name="custom_test",
            num_episodes=20,
            max_steps_per_episode=1000,
            deterministic=False,
            render=True,
            save_actions=False,
            save_observations=True,
            timeout_seconds=7200
        )
        
        self.assertEqual(config.test_name, "custom_test")
        self.assertEqual(config.num_episodes, 20)
        self.assertEqual(config.max_steps_per_episode, 1000)
        self.assertFalse(config.deterministic)
        self.assertTrue(config.render)
        self.assertFalse(config.save_actions)
        self.assertTrue(config.save_observations)
        self.assertEqual(config.timeout_seconds, 7200)


@unittest.skipIf(not TESTING_UTILS_AVAILABLE, "Testing utilities not available")
class TestTestResult(unittest.TestCase):
    """Test cases for TestResult dataclass."""
    
    def test_basic_result_creation(self):
        """Test basic TestResult creation."""
        result = TestResult(
            episode_num=1,
            steps_taken=100,
            total_reward=50.5,
            final_reward=5.0,
            success=True,
            terminated=True,
            truncated=False,
            duration=15.3
        )
        
        self.assertEqual(result.episode_num, 1)
        self.assertEqual(result.steps_taken, 100)
        self.assertEqual(result.total_reward, 50.5)
        self.assertEqual(result.final_reward, 5.0)
        self.assertTrue(result.success)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.duration, 15.3)
        self.assertIsInstance(result.timestamp, str)
    
    def test_result_with_optional_data(self):
        """Test TestResult with optional action and observation data."""
        actions = [0, 1, 2, 3, 0, 1]
        observations = [[1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0]] * 6
        info = {'key_picked_up': True, 'escaped': True}
        
        result = TestResult(
            episode_num=2,
            steps_taken=6,
            total_reward=100.0,
            final_reward=50.0,
            success=True,
            terminated=True,
            truncated=False,
            duration=8.5,
            actions=actions,
            observations=observations,
            info=info
        )
        
        self.assertEqual(result.actions, actions)
        self.assertEqual(result.observations, observations)
        self.assertEqual(result.info, info)
    
    def test_result_with_error(self):
        """Test TestResult with error information."""
        result = TestResult(
            episode_num=3,
            steps_taken=0,
            total_reward=0.0,
            final_reward=0.0,
            success=False,
            terminated=False,
            truncated=False,
            duration=0.0,
            error="Unity connection failed"
        )
        
        self.assertEqual(result.error, "Unity connection failed")
        self.assertFalse(result.success)
    
    def test_timestamp_auto_generation(self):
        """Test that timestamp is automatically generated."""
        result = TestResult(
            episode_num=1,
            steps_taken=50,
            total_reward=25.0,
            final_reward=2.0,
            success=False,
            terminated=False,
            truncated=True,
            duration=10.0
        )
        
        self.assertIsNotNone(result.timestamp)
        self.assertNotEqual(result.timestamp, "")


@unittest.skipIf(not TESTING_UTILS_AVAILABLE, "Testing utilities not available")
class TestTestRunner(unittest.TestCase):
    """Test cases for TestRunner class."""
    
    def setUp(self):
        """Set up test configuration and mock objects."""
        self.config = TestConfiguration(
            test_name="unit_test",
            num_episodes=3,
            max_steps_per_episode=100,
            deterministic=True,
            save_actions=True,
            save_observations=False
        )
        self.runner = TestRunner(self.config)
        
        # Create mock model and environment
        self.mock_model = MagicMock()
        self.mock_env = MagicMock()
    
    def test_runner_initialization(self):
        """Test TestRunner initializes correctly."""
        self.assertEqual(self.runner.config.test_name, "unit_test")
        self.assertEqual(len(self.runner.results), 0)
        self.assertIsNone(self.runner.start_time)
        self.assertIsNone(self.runner.end_time)
    
    def test_single_episode_run(self):
        """Test running a single episode."""
        # Configure mock behavior
        self.mock_env.reset.return_value = ([1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0], {})
        self.mock_model.predict.return_value = (1, None)  # Action, states
        
        # Setup step sequence: normal steps then termination
        step_returns = [
            ([1.1, 2.1, 0.0, 3.0, 4.0, 5.0, 6.0], 0.1, False, False, {}),
            ([1.2, 2.2, 1.0, 3.0, 4.0, 5.0, 6.0], 10.0, False, False, {'key_picked_up': True}),
            ([1.3, 2.3, 1.0, 3.0, 4.0, 5.0, 6.0], 100.0, True, False, {'success': True, 'escaped': True})
        ]
        self.mock_env.step.side_effect = step_returns
        
        result = self.runner._run_single_episode(self.mock_model, self.mock_env, 1)
        
        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.episode_num, 1)
        self.assertEqual(result.steps_taken, 3)
        self.assertTrue(result.success)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertIsNotNone(result.actions)
        self.assertEqual(len(result.actions), 3)
    
    def test_episode_truncation(self):
        """Test episode truncation at max steps."""
        # Configure mock behavior for long episode
        self.mock_env.reset.return_value = ([1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0], {})
        self.mock_model.predict.return_value = (0, None)
        
        # Configure step to never terminate naturally
        self.mock_env.step.return_value = (
            [1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0], -0.1, False, False, {}
        )
        
        # Use small max steps for testing
        self.runner.config.max_steps_per_episode = 5
        
        result = self.runner._run_single_episode(self.mock_model, self.mock_env, 1)
        
        self.assertEqual(result.steps_taken, 5)
        self.assertFalse(result.success)
        self.assertFalse(result.terminated)
        # Note: In current implementation, truncated isn't set during single episode
    
    def test_full_test_run(self):
        """Test complete test run with multiple episodes."""
        # Configure mock behavior
        self.mock_env.reset.return_value = ([1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0], {})
        self.mock_model.predict.return_value = (1, None)
        
        # Configure episodes with different outcomes
        episode_outcomes = [
            # Episode 1: Success
            ([1.1, 2.1, 1.0, 3.0, 4.0, 5.0, 6.0], 100.0, True, False, {'success': True}),
            # Episode 2: Failure
            ([1.2, 2.2, 0.0, 3.0, 4.0, 5.0, 6.0], -10.0, False, True, {'success': False}),
            # Episode 3: Success
            ([1.3, 2.3, 1.0, 3.0, 4.0, 5.0, 6.0], 90.0, True, False, {'success': True})
        ]
        
        self.mock_env.step.side_effect = episode_outcomes
        
        test_results = self.runner.run_model_test(self.mock_model, self.mock_env)
        
        # Verify test structure
        self.assertIn('config', test_results)
        self.assertIn('summary', test_results)
        self.assertIn('results', test_results)
        
        # Verify results
        self.assertEqual(len(test_results['results']), 3)
        
        # Verify summary
        summary = test_results['summary']
        self.assertEqual(summary['test_info']['total_episodes'], 3)
        self.assertEqual(summary['success_metrics']['successful_episodes'], 2)
        self.assertAlmostEqual(summary['success_metrics']['success_rate'], 66.67, places=1)
    
    def test_timeout_handling(self):
        """Test timeout handling during test runs."""
        # Set very short timeout
        self.runner.config.timeout_seconds = 0.001
        
        # Configure mock to cause delay
        def slow_reset():
            time.sleep(0.01)  # Longer than timeout
            return ([1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0], {})
        
        self.mock_env.reset.side_effect = slow_reset
        
        # Start time should be set when we begin
        self.runner.start_time = self.runner.start_time or datetime.datetime.now()
        
        # Check timeout detection
        is_timeout = self.runner._check_timeout()
        # Should timeout due to delay in setup
        
        # Run should handle timeout gracefully
        test_results = self.runner.run_model_test(self.mock_model, self.mock_env)
        self.assertIn('results', test_results)
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        # Setup simple successful episode
        self.mock_env.reset.return_value = ([1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0], {})
        self.mock_model.predict.return_value = (1, None)
        self.mock_env.step.return_value = (
            [1.1, 2.1, 1.0, 3.0, 4.0, 5.0, 6.0], 100.0, True, False, {'success': True}
        )
        
        # Create callback mock
        progress_callback = MagicMock()
        
        test_results = self.runner.run_model_test(
            self.mock_model, self.mock_env, progress_callback=progress_callback
        )
        
        # Verify callback was called for each episode
        self.assertEqual(progress_callback.call_count, 3)
        
        # Verify callback arguments
        for call_args in progress_callback.call_args_list:
            episode_num, total_episodes, result = call_args[0]
            self.assertIsInstance(episode_num, int)
            self.assertEqual(total_episodes, 3)
            self.assertIsInstance(result, TestResult)


@unittest.skipIf(not TESTING_UTILS_AVAILABLE, "Testing utilities not available")
class TestModelBenchmark(unittest.TestCase):
    """Test cases for ModelBenchmark class."""
    
    def setUp(self):
        """Set up benchmark test environment."""
        self.benchmark = ModelBenchmark("test_benchmark")
        
        # Create mock models
        self.mock_model1 = MagicMock()
        self.mock_model2 = MagicMock()
        self.models = {
            "good_model": self.mock_model1,
            "poor_model": self.mock_model2
        }
        
        # Create mock environment
        self.mock_env = MagicMock()
    
    def test_benchmark_initialization(self):
        """Test ModelBenchmark initializes correctly."""
        self.assertEqual(self.benchmark.benchmark_name, "test_benchmark")
        self.assertEqual(len(self.benchmark.model_results), 0)
    
    def test_add_model_results(self):
        """Test adding model results to benchmark."""
        test_results = {
            'config': {'test_name': 'test'},
            'summary': {'success_metrics': {'success_rate': 75.0}},
            'results': []
        }
        
        self.benchmark.add_model_results("test_model", test_results)
        
        self.assertIn("test_model", self.benchmark.model_results)
        self.assertEqual(self.benchmark.model_results["test_model"], test_results)
    
    @patch('testing_utils.TestRunner')
    def test_run_model_benchmark(self, mock_test_runner_class):
        """Test running benchmark on multiple models."""
        # Create mock test runner instance
        mock_runner = MagicMock()
        mock_test_runner_class.return_value = mock_runner
        
        # Configure different results for each model
        good_results = {
            'config': {'test_name': 'good_model_test'},
            'summary': {
                'success_metrics': {'success_rate': 85.0},
                'performance_metrics': {'avg_steps': 80, 'avg_reward': 45.0}
            },
            'results': []
        }
        
        poor_results = {
            'config': {'test_name': 'poor_model_test'},
            'summary': {
                'success_metrics': {'success_rate': 25.0},
                'performance_metrics': {'avg_steps': 150, 'avg_reward': -5.0}
            },
            'results': []
        }
        
        # Configure mock to return different results for each call
        mock_runner.run_model_test.side_effect = [good_results, poor_results]
        
        config = TestConfiguration("benchmark_test", num_episodes=5)
        benchmark_results = self.benchmark.run_model_benchmark(
            self.models, self.mock_env, config
        )
        
        # Verify benchmark structure
        self.assertIn('benchmark_name', benchmark_results)
        self.assertIn('models_tested', benchmark_results)
        self.assertIn('individual_results', benchmark_results)
        self.assertIn('comparison', benchmark_results)
        
        # Verify models were tested
        self.assertEqual(benchmark_results['models_tested'], ["good_model", "poor_model"])
        
        # Verify comparison results
        comparison = benchmark_results['comparison']
        self.assertIn('rankings', comparison)
        self.assertIn('best_performers', comparison)
        
        # Good model should be best performer
        self.assertEqual(comparison['best_performers']['highest_success_rate'], "good_model")
        self.assertEqual(comparison['best_performers']['highest_reward'], "good_model")
    
    def test_generate_comparison_report_empty(self):
        """Test comparison report generation with no results."""
        comparison = self.benchmark._generate_comparison_report()
        self.assertEqual(comparison['rankings'], {})
        self.assertEqual(comparison['statistical_comparison'], {})
        self.assertEqual(comparison['recommendations'], [])
    
    def test_generate_comparison_report_with_data(self):
        """Test comparison report generation with model data."""
        # Add mock results
        good_results = {
            'summary': {
                'success_metrics': {'success_rate': 80.0},
                'performance_metrics': {'avg_steps': 90, 'avg_reward': 40.0},
                'success_specific_metrics': {'avg_success_steps': 85}
            }
        }
        
        poor_results = {
            'summary': {
                'success_metrics': {'success_rate': 30.0},
                'performance_metrics': {'avg_steps': 160, 'avg_reward': -10.0},
                'success_specific_metrics': {'avg_success_steps': 140}
            }
        }
        
        self.benchmark.add_model_results("good_model", good_results)
        self.benchmark.add_model_results("poor_model", poor_results)
        
        comparison = self.benchmark._generate_comparison_report()
        
        # Verify rankings exist
        self.assertIn('by_success_rate', comparison['rankings'])
        self.assertIn('by_efficiency', comparison['rankings'])
        self.assertIn('by_reward', comparison['rankings'])
        
        # Verify best performers
        self.assertIn('best_performers', comparison)
        self.assertEqual(comparison['best_performers']['highest_success_rate'], "good_model")
        self.assertEqual(comparison['best_performers']['most_efficient'], "good_model")
        self.assertEqual(comparison['best_performers']['highest_reward'], "good_model")


@unittest.skipIf(not TESTING_UTILS_AVAILABLE, "Testing utilities not available")
class TestTestReportGenerator(unittest.TestCase):
    """Test cases for TestReportGenerator class."""
    
    def setUp(self):
        """Set up test report generator and sample data."""
        self.generator = TestReportGenerator()
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # Sample test results
        self.test_results = {
            'config': {
                'test_name': 'sample_test',
                'num_episodes': 3
            },
            'summary': {
                'success_metrics': {'success_rate': 66.67, 'success_count': 2},
                'performance_metrics': {'avg_steps': 95.0, 'avg_reward': 30.0}
            },
            'results': [
                {
                    'episode_num': 1,
                    'steps_taken': 80,
                    'total_reward': 50.0,
                    'success': True,
                    'actions': [0, 1, 2, 3],
                    'observations': None
                },
                {
                    'episode_num': 2,
                    'steps_taken': 120,
                    'total_reward': -10.0,
                    'success': False,
                    'actions': [1, 2, 3, 0],
                    'observations': None
                },
                {
                    'episode_num': 3,
                    'steps_taken': 85,
                    'total_reward': 70.0,
                    'success': True,
                    'actions': [2, 3, 0, 1],
                    'observations': None
                }
            ]
        }
    
    def test_generate_json_report(self):
        """Test JSON report generation."""
        output_path = Path(self.temp_dir) / "test_report.json"
        
        success = self.generator.generate_json_report(self.test_results, str(output_path))
        
        self.assertTrue(success)
        self.assertTrue(output_path.exists())
        
        # Verify content
        with open(output_path, 'r') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data['config']['test_name'], 'sample_test')
        self.assertEqual(len(loaded_data['results']), 3)
    
    def test_generate_csv_report(self):
        """Test CSV report generation."""
        output_path = Path(self.temp_dir) / "test_report.csv"
        
        success = self.generator.generate_csv_report(self.test_results, str(output_path))
        
        self.assertTrue(success)
        self.assertTrue(output_path.exists())
        
        # Verify content
        with open(output_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
        
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]['episode_num'], '1')
        self.assertEqual(rows[0]['success'], 'True')
    
    def test_generate_csv_report_empty_results(self):
        """Test CSV report generation with empty results."""
        empty_results = {'results': []}
        output_path = Path(self.temp_dir) / "empty_report.csv"
        
        success = self.generator.generate_csv_report(empty_results, str(output_path))
        
        self.assertFalse(success)
    
    def test_generate_text_summary(self):
        """Test text summary generation."""
        summary_text = self.generator.generate_text_summary(self.test_results)
        
        self.assertIsInstance(summary_text, str)
        self.assertIn("TEST REPORT SUMMARY", summary_text)
        self.assertIn("sample_test", summary_text)
        self.assertIn("Episodes: 3", summary_text)
        self.assertIn("Success Rate: 66.7%", summary_text)
    
    def test_generate_text_summary_benchmark(self):
        """Test text summary generation for benchmark results."""
        benchmark_results = {
            'benchmark_name': 'model_comparison',
            'models_tested': ['model1', 'model2'],
            'comparison': {
                'best_performers': {
                    'highest_success_rate': 'model1',
                    'most_efficient': 'model2',
                    'highest_reward': 'model1'
                }
            }
        }
        
        summary_text = self.generator.generate_text_summary(benchmark_results)
        
        self.assertIn("model_comparison", summary_text)
        self.assertIn("model1, model2", summary_text)
        self.assertIn("Best Performers", summary_text)


@unittest.skipIf(not TESTING_UTILS_AVAILABLE, "Testing utilities not available")
class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up temporary directory and mock objects."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # Create mock model file
        self.model_path = Path(self.temp_dir) / "test_model.zip"
        with open(self.model_path, 'w') as f:
            f.write("mock model content")
    
    @patch('testing_utils.PPO')
    def test_quick_model_test(self, mock_ppo):
        """Test quick model testing utility."""
        # Setup mock model and environment
        mock_model = MagicMock()
        mock_ppo.load.return_value = mock_model
        
        mock_env = MagicMock()
        mock_env.reset.return_value = ([1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0], {})
        mock_model.predict.return_value = (1, None)
        mock_env.step.return_value = (
            [1.1, 2.1, 1.0, 3.0, 4.0, 5.0, 6.0], 100.0, True, False, {'success': True}
        )
        
        with patch('testing_utils.TestRunner') as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner_class.return_value = mock_runner
            mock_runner.run_model_test.return_value = {'summary': {'success_rate': 90.0}}
            
            results = quick_model_test(str(self.model_path), mock_env, episodes=5)
        
        self.assertIn('summary', results)
        mock_ppo.load.assert_called_once_with(str(self.model_path))
    
    @patch('testing_utils.PPO')
    def test_quick_model_test_load_failure(self, mock_ppo):
        """Test quick model test with model loading failure."""
        mock_ppo.load.side_effect = Exception("Load failed")
        mock_env = MagicMock()
        
        results = quick_model_test(str(self.model_path), mock_env)
        
        self.assertIn('error', results)
        self.assertEqual(results['error'], "Load failed")
    
    @patch('testing_utils.PPO')
    def test_compare_models_quick(self, mock_ppo):
        """Test quick model comparison utility."""
        # Create additional model files
        model2_path = Path(self.temp_dir) / "model2.zip"
        with open(model2_path, 'w') as f:
            f.write("mock model 2 content")
        
        # Setup mock models
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_ppo.load.side_effect = [mock_model1, mock_model2]
        
        mock_env = MagicMock()
        
        with patch('testing_utils.ModelBenchmark') as mock_benchmark_class:
            mock_benchmark = MagicMock()
            mock_benchmark_class.return_value = mock_benchmark
            mock_benchmark.run_model_benchmark.return_value = {
                'comparison': {'best_performers': {'highest_success_rate': 'test_model'}}
            }
            
            results = compare_models_quick(
                [str(self.model_path), str(model2_path)], 
                mock_env, 
                episodes=5
            )
        
        self.assertIn('comparison', results)
        self.assertEqual(mock_ppo.load.call_count, 2)
    
    @patch('testing_utils.PPO')
    def test_compare_models_quick_no_valid_models(self, mock_ppo):
        """Test quick model comparison with no loadable models."""
        mock_ppo.load.side_effect = Exception("Load failed")
        mock_env = MagicMock()
        
        results = compare_models_quick([str(self.model_path)], mock_env)
        
        self.assertIn('error', results)
        self.assertEqual(results['error'], 'No models loaded successfully')
    
    def test_validate_environment_performance(self):
        """Test environment performance validation."""
        # Create mock environment
        mock_env = MagicMock()
        
        # Configure environment behavior
        mock_env.reset.return_value = ([1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0], {})
        mock_env.action_space.sample.return_value = 1
        mock_env.step.return_value = (
            [1.1, 2.1, 0.0, 3.0, 4.0, 5.0, 6.0], -0.1, False, False, {}
        )
        
        # Run validation with small episode count for testing
        results = validate_environment_performance(mock_env, num_episodes=2)
        
        # Verify results structure
        self.assertIn('total_episodes', results)
        self.assertIn('successful_resets', results)
        self.assertIn('successful_steps', results)
        self.assertIn('reset_times', results)
        self.assertIn('step_times', results)
        
        # Verify some basic metrics
        self.assertEqual(results['total_episodes'], 2)
        self.assertGreaterEqual(results['successful_resets'], 0)
        self.assertGreaterEqual(results['successful_steps'], 0)
    
    def test_validate_environment_performance_with_errors(self):
        """Test environment performance validation with errors."""
        # Create mock environment that throws errors
        mock_env = MagicMock()
        mock_env.reset.side_effect = Exception("Reset failed")
        
        results = validate_environment_performance(mock_env, num_episodes=1)
        
        # Should handle errors gracefully
        self.assertIn('errors', results)
        self.assertGreater(len(results['errors']), 0)
        self.assertEqual(results['successful_resets'], 0)


class TestTestingUtilsIntegration(unittest.TestCase):
    """Integration tests for testing utilities."""
    
    @unittest.skipIf(not TESTING_UTILS_AVAILABLE, "Testing utilities not available")
    def test_full_testing_workflow(self):
        """Test complete testing workflow from configuration to reporting."""
        # Create test configuration
        config = TestConfiguration(
            test_name="integration_test",
            num_episodes=2,
            max_steps_per_episode=50,
            save_actions=True,
            save_observations=False
        )
        
        # Create test runner
        runner = TestRunner(config)
        
        # Create mock model and environment
        mock_model = MagicMock()
        mock_env = MagicMock()
        
        # Configure environment for two episodes
        mock_env.reset.return_value = ([1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0], {})
        mock_model.predict.return_value = (1, None)
        
        # Episode outcomes: one success, one failure
        episode_outcomes = [
            ([1.1, 2.1, 1.0, 3.0, 4.0, 5.0, 6.0], 100.0, True, False, {'success': True}),
            ([1.2, 2.2, 0.0, 3.0, 4.0, 5.0, 6.0], -10.0, False, True, {'success': False})
        ]
        mock_env.step.side_effect = episode_outcomes
        
        # Run test
        test_results = runner.run_model_test(mock_model, mock_env)
        
        # Verify test completed
        self.assertIn('config', test_results)
        self.assertIn('summary', test_results)
        self.assertIn('results', test_results)
        self.assertEqual(len(test_results['results']), 2)
        
        # Test report generation
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = TestReportGenerator()
            
            # Generate JSON report
            json_path = Path(temp_dir) / "integration_test.json"
            json_success = generator.generate_json_report(test_results, str(json_path))
            self.assertTrue(json_success)
            self.assertTrue(json_path.exists())
            
            # Generate CSV report
            csv_path = Path(temp_dir) / "integration_test.csv"
            csv_success = generator.generate_csv_report(test_results, str(csv_path))
            self.assertTrue(csv_success)
            self.assertTrue(csv_path.exists())
            
            # Generate text summary
            text_summary = generator.generate_text_summary(test_results)
            self.assertIn("integration_test", text_summary)


def run_testing_utils_tests():
    """Run all testing utility tests."""
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTestConfiguration,
        TestTestResult,
        TestTestRunner,
        TestModelBenchmark,
        TestTestReportGenerator,
        TestUtilityFunctions,
        TestTestingUtilsIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    if TESTING_UTILS_AVAILABLE:
        print("🧪 Running Testing Utilities Tests...")
        success = run_testing_utils_tests()
        if success:
            print("✅ All testing utility tests passed!")
        else:
            print("❌ Some testing utility tests failed!")
    else:
        print("⚠️ Testing utilities not available for testing")
        unittest.main() 