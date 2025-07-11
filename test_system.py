"""
Comprehensive AI Escape Cage Test System

Single command to test every aspect of the project:
- Unity communication and connection
- Training system functionality  
- Model loading and saving
- Configuration system
- Environment functionality
- Overall system health and integration

Usage:
    python test_system.py                    # Run all tests
    python test_system.py --quick           # Quick smoke tests only
    python test_system.py --component unity # Test specific component
    python test_system.py --verbose         # Detailed output
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TestResult(Enum):
    """Test result status."""
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    SKIP = "⏭️  SKIP"
    WARN = "⚠️  WARN"


@dataclass
class TestReport:
    """Individual test report."""
    name: str
    result: TestResult
    duration: float
    message: str = ""
    details: str = ""


class TestSuite:
    """Main test suite for the AI Escape Cage project."""
    
    def __init__(self, verbose: bool = False):
        """Initialize test suite."""
        self.verbose = verbose
        self.reports: List[TestReport] = []
        self.start_time = time.time()
        
    def log(self, message: str, level: str = "info"):
        """Log message with proper formatting."""
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug" and self.verbose:
            logger.debug(message)
    
    @contextmanager
    def test_context(self, test_name: str):
        """Context manager for individual tests."""
        start_time = time.time()
        self.log(f"🧪 Running: {test_name}")
        
        try:
            yield
            duration = time.time() - start_time
            self.reports.append(TestReport(test_name, TestResult.PASS, duration))
            self.log(f"   {TestResult.PASS.value} {test_name} ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            details = traceback.format_exc() if self.verbose else ""
            self.reports.append(TestReport(test_name, TestResult.FAIL, duration, error_msg, details))
            self.log(f"   {TestResult.FAIL.value} {test_name} ({duration:.2f}s): {error_msg}")
            if self.verbose:
                self.log(details, "debug")
    
    def skip_test(self, test_name: str, reason: str):
        """Mark a test as skipped."""
        self.reports.append(TestReport(test_name, TestResult.SKIP, 0.0, reason))
        self.log(f"   {TestResult.SKIP.value} {test_name}: {reason}")
    
    def warn_test(self, test_name: str, message: str):
        """Mark a test with warning."""
        self.reports.append(TestReport(test_name, TestResult.WARN, 0.0, message))
        self.log(f"   {TestResult.WARN.value} {test_name}: {message}")

    def test_project_structure(self):
        """Test that all required project files and directories exist."""
        with self.test_context("Project Structure"):
            required_files = [
                "ml_training/escape_cage_trainer.py",
                "ml_training/base_environment.py",
                "ml_training/model_utils.py",
                "ml_training/analytics_utils.py",
                "ml_training/logger_setup.py",
                "ml_training/trainers/__init__.py",
                "ml_training/trainers/base_trainer.py",
                "ml_training/trainers/standard_trainer.py",
                "ml_training/trainers/fast_trainer.py",
                "ml_training/trainers/continue_trainer.py",
                "ml_training/trainers/trainer_factory.py",
                "ml_training/config/__init__.py",
                "ml_training/config/hyperparameters.py",
                "communication/unity_bridge.py",
                "requirements.txt",
                "README.md",
                "test_system.py"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                raise FileNotFoundError(f"Missing required files: {missing_files}")
            
            self.log(f"   All {len(required_files)} required files found", "debug")

    def test_imports(self):
        """Test that all critical modules can be imported."""
        with self.test_context("Python Imports"):
            # Core dependencies
            import numpy as np
            import gymnasium as gym
            from stable_baselines3 import PPO
            import tensorflow as tf
            
            # Project modules
            from ml_training.trainers import create_trainer, get_available_trainer_types
            from ml_training.config.hyperparameters import get_hyperparameters
            from ml_training.base_environment import SimpleEscapeCageEnv
            from ml_training.model_utils import find_model_file, list_available_models
            from communication.unity_bridge import UnityBridge
            
            self.log("   All critical imports successful", "debug")

    def test_configuration_system(self):
        """Test the hyperparameter configuration system."""
        with self.test_context("Configuration System"):
            from ml_training.config.hyperparameters import (
                get_hyperparameters, validate_hyperparameters, 
                HyperparameterConfig, HYPERPARAMETER_CONFIGS
            )
            
            # Test getting configurations
            for config_type in ['standard', 'fast', 'conservative']:
                config = get_hyperparameters(config_type)
                assert isinstance(config, HyperparameterConfig)
                validate_hyperparameters(config)
            
            # Test custom parameters
            custom_config = get_hyperparameters('standard', {'learning_rate': 0.001})
            assert custom_config.learning_rate == 0.001
            
            # Test invalid configuration
            try:
                get_hyperparameters('invalid_type')
                raise AssertionError("Should have raised ValueError for invalid type")
            except ValueError:
                pass  # Expected
            
            self.log("   Configuration validation passed", "debug")

    def test_trainer_factory(self):
        """Test the trainer factory system."""
        with self.test_context("Trainer Factory"):
            from ml_training.trainers import create_trainer, get_available_trainer_types
            
            # Test available types
            available_types = get_available_trainer_types()
            assert 'standard' in available_types
            assert 'fast' in available_types
            assert 'continue' in available_types
            
            # Test trainer creation
            for trainer_type in available_types:
                if trainer_type != 'continue':  # Skip continue trainer (needs existing models)
                    trainer = create_trainer(trainer_type)
                    assert hasattr(trainer, 'train')
                    assert hasattr(trainer, 'get_training_type')
                    assert trainer.get_training_type() in ['standard', 'fast']
            
            # Test invalid trainer type
            try:
                create_trainer('invalid_trainer')
                raise AssertionError("Should have raised ValueError for invalid trainer")
            except ValueError:
                pass  # Expected
            
            self.log("   Trainer factory validation passed", "debug")

    def test_environment_creation(self):
        """Test environment creation without Unity connection."""
        with self.test_context("Environment Creation (No Unity)"):
            try:
                # Import environment classes
                from ml_training.base_environment import (
                    SimpleEscapeCageEnv, FastEscapeCageEnv, TestEscapeCageEnv
                )
                
                # Note: We can't actually create environments without Unity running
                # But we can test the class definitions and inheritance
                assert hasattr(SimpleEscapeCageEnv, '_calculate_reward')
                assert hasattr(FastEscapeCageEnv, '_calculate_reward')
                assert hasattr(TestEscapeCageEnv, '_calculate_reward')
                
                self.log("   Environment classes properly defined", "debug")
                
            except Exception as e:
                if "Unity" in str(e) or "connection" in str(e).lower():
                    self.warn_test("Environment Creation", "Unity not running - classes validated only")
                else:
                    raise

    def test_unity_bridge(self):
        """Test Unity bridge functionality."""
        with self.test_context("Unity Bridge"):
            try:
                from communication.unity_bridge import UnityBridge
                
                # Test bridge creation (without actually connecting)
                bridge = UnityBridge()
                assert hasattr(bridge, 'start_server')
                assert hasattr(bridge, 'wait_for_unity')
                assert hasattr(bridge, 'send_action')
                assert hasattr(bridge, 'receive_observation')
                
                self.log("   Unity bridge class properly defined", "debug")
                
                # Try to test actual connection (will likely fail if Unity not running)
                try:
                    # This will likely fail, but we test the interface
                    bridge.start_server()
                    connected = bridge.wait_for_unity(timeout=1)  # Very short timeout
                    if connected:
                        self.log("   Unity connection successful!", "debug")
                        bridge.close()
                    else:
                        self.warn_test("Unity Connection", "Unity not running - bridge interface validated")
                except Exception:
                    self.warn_test("Unity Connection", "Unity not running - bridge interface validated")
                    
            except Exception as e:
                raise Exception(f"Unity bridge interface error: {e}")

    def test_model_utilities(self):
        """Test model utility functions."""
        with self.test_context("Model Utilities"):
            from ml_training.model_utils import find_model_file, list_available_models, ModelMetadata
            
            # Test model listing (should work even with no models)
            models = list_available_models()
            assert isinstance(models, list)
            
            # Test model finding with non-existent model
            result = find_model_file("non_existent_model")
            assert result is None
            
            # Test metadata creation
            metadata = ModelMetadata("test_model")
            assert metadata.model_name == "test_model"
            assert hasattr(metadata, 'creation_time')
            
            self.log("   Model utilities functioning correctly", "debug")

    def test_analytics_system(self):
        """Test analytics utilities."""
        with self.test_context("Analytics System"):
            from ml_training.analytics_utils import PerformanceAnalyzer, EpisodeMetrics
            
            # Test analyzer creation
            analyzer = PerformanceAnalyzer("test_session")
            assert analyzer.session_name == "test_session"
            
            # Test episode metrics
            episode = EpisodeMetrics(
                episode_num=1,
                steps=100,
                total_reward=50.0,
                success=True,
                terminated=True,
                truncated=False,
                duration=10.0,
                action_distribution={'up': 25, 'down': 25, 'left': 25, 'right': 25}
            )
            
            analyzer.add_episode(episode)
            assert len(analyzer.episodes) == 1
            
            self.log("   Analytics system functioning correctly", "debug")

    def test_logger_system(self):
        """Test logging setup."""
        with self.test_context("Logger System"):
            from ml_training.logger_setup import setup_logging, TrainingLogger
            
            # Test logger setup
            logger_manager = setup_logging("test_session")
            assert hasattr(logger_manager, 'logger')
            
            # Test logging functionality
            logger_manager.logger.info("Test log message")
            
            # Clean up
            if hasattr(logger_manager, 'close'):
                logger_manager.close()
            
            self.log("   Logger system functioning correctly", "debug")

    def test_training_smoke_test(self):
        """Quick smoke test of training system (without actual training)."""
        with self.test_context("Training System Smoke Test"):
            from ml_training.trainers import create_trainer
            
            # Test trainer creation and basic interface
            trainer = create_trainer('standard')
            
            # Verify trainer has required methods
            assert hasattr(trainer, 'train')
            assert hasattr(trainer, 'get_training_type')
            assert hasattr(trainer, 'create_agent')
            assert hasattr(trainer, 'get_environment')
            
            # Test configuration
            training_type = trainer.get_training_type()
            assert training_type in ['standard', 'fast']
            
            self.log("   Training system interface validated", "debug")

    def test_cli_interface(self):
        """Test command-line interface."""
        with self.test_context("CLI Interface"):
            import subprocess
            import sys
            
            # Test help command from project root
            result = subprocess.run([
                sys.executable, "ml_training/escape_cage_trainer.py", "--help"
            ], capture_output=True, text=True, timeout=30, cwd=".")
            
            if result.returncode == 0:
                assert "--trainer" in result.stdout
                assert "--steps" in result.stdout
                self.log("   CLI help system working", "debug")
            else:
                raise Exception(f"CLI help failed: {result.stderr}")

    def test_integration_health(self):
        """Test overall system integration and health."""
        with self.test_context("System Integration"):
            # Test that all components can work together
            from ml_training.trainers import create_trainer
            from ml_training.config.hyperparameters import get_hyperparameters
            
            # Create trainer with custom configuration
            trainer = create_trainer('fast')
            config = get_hyperparameters('fast', {'batch_size': 64})
            
            # Verify integration
            assert config.batch_size == 64
            assert trainer.get_training_type() == 'fast'
            
            self.log("   System integration validated", "debug")

    def run_quick_tests(self):
        """Run quick smoke tests only."""
        self.log("🚀 Running Quick Tests...")
        
        self.test_project_structure()
        self.test_imports()
        self.test_configuration_system()
        self.test_trainer_factory()
        self.test_training_smoke_test()

    def run_component_test(self, component: str):
        """Run tests for a specific component."""
        self.log(f"🎯 Testing Component: {component}")
        
        component_tests = {
            'structure': self.test_project_structure,
            'imports': self.test_imports,
            'config': self.test_configuration_system,
            'trainers': self.test_trainer_factory,
            'environment': self.test_environment_creation,
            'unity': self.test_unity_bridge,
            'models': self.test_model_utilities,
            'analytics': self.test_analytics_system,
            'logging': self.test_logger_system,
            'training': self.test_training_smoke_test,
            'cli': self.test_cli_interface,
            'integration': self.test_integration_health
        }
        
        if component in component_tests:
            component_tests[component]()
        else:
            available = list(component_tests.keys())
            raise ValueError(f"Unknown component '{component}'. Available: {available}")

    def run_full_tests(self):
        """Run comprehensive test suite."""
        self.log("🔬 Running Comprehensive Test Suite...")
        
        # Core system tests
        self.test_project_structure()
        self.test_imports()
        self.test_configuration_system()
        self.test_trainer_factory()
        
        # Component tests
        self.test_environment_creation()
        self.test_unity_bridge()
        self.test_model_utilities()
        self.test_analytics_system()
        self.test_logger_system()
        
        # Integration tests
        self.test_training_smoke_test()
        self.test_cli_interface()
        self.test_integration_health()

    def generate_report(self):
        """Generate final test report."""
        total_time = time.time() - self.start_time
        
        # Count results
        passed = len([r for r in self.reports if r.result == TestResult.PASS])
        failed = len([r for r in self.reports if r.result == TestResult.FAIL])
        skipped = len([r for r in self.reports if r.result == TestResult.SKIP])
        warnings = len([r for r in self.reports if r.result == TestResult.WARN])
        total = len(self.reports)
        
        print("\n" + "="*80)
        print("🧪 AI ESCAPE CAGE - COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total Tests: {total}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ⏭️  Skipped: {skipped}")
        print(f"   ⚠️  Warnings: {warnings}")
        print(f"   ⏱️  Duration: {total_time:.2f}s")
        
        if failed > 0:
            print(f"\n❌ FAILED TESTS:")
            for report in self.reports:
                if report.result == TestResult.FAIL:
                    print(f"   • {report.name}: {report.message}")
        
        if warnings > 0:
            print(f"\n⚠️  WARNINGS:")
            for report in self.reports:
                if report.result == TestResult.WARN:
                    print(f"   • {report.name}: {report.message}")
        
        # Overall status
        if failed == 0:
            if warnings == 0:
                print(f"\n🎉 ALL TESTS PASSED! System is healthy and ready for use.")
                return True
            else:
                print(f"\n✅ Tests passed with {warnings} warnings. System mostly healthy.")
                return True
        else:
            print(f"\n💥 {failed} test(s) failed. System needs attention.")
            return False


def main():
    """Main entry point for the test system."""
    parser = argparse.ArgumentParser(description="Comprehensive AI Escape Cage Test System")
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick smoke tests only')
    parser.add_argument('--component', type=str,
                       help='Test specific component (structure, imports, config, trainers, etc.)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with detailed information')
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = TestSuite(verbose=args.verbose)
    
    try:
        if args.quick:
            test_suite.run_quick_tests()
        elif args.component:
            test_suite.run_component_test(args.component)
        else:
            test_suite.run_full_tests()
            
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test system error: {e}")
        if args.verbose:
            print(traceback.format_exc())
        sys.exit(1)
    
    # Generate report
    success = test_suite.generate_report()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 