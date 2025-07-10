#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner

Runs all tests in the project with detailed reporting and error handling.
Provides summary statistics and failure analysis.
"""

import os
import sys
import unittest
import importlib.util
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback


class TestResult:
    """Container for test results with detailed information."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.duration = 0.0
        self.error_message = ""
        self.test_count = 0
        self.failure_count = 0
        self.details = {}


class ComprehensiveTestRunner:
    """
    Comprehensive test runner for the AI Escape Cage project.
    
    Features:
    - Automatic test discovery
    - Detailed error reporting
    - Summary statistics
    - Fallback testing methods
    - Module availability checking
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the test runner.
        
        Args:
            verbose: Whether to show detailed output
        """
        self.verbose = verbose
        self.test_results: Dict[str, TestResult] = {}
        self.project_root = Path(__file__).parent
        
        # Define test suites with their dependencies
        self.test_suites = {
            'analytics_utils': {
                'module': 'tests.test_analytics_utils',
                'function': 'run_analytics_tests',
                'description': 'Analytics and performance tracking utilities',
                'dependencies': ['numpy', 'matplotlib', 'seaborn']
            },
            'model_utils': {
                'module': 'tests.test_model_utils',
                'function': 'run_model_tests', 
                'description': 'Model management and validation utilities',
                'dependencies': ['pathlib', 'json']
            },
            'testing_utils': {
                'module': 'tests.test_testing_utils',
                'function': 'run_testing_tests',
                'description': 'Testing framework and utilities',
                'dependencies': ['unittest', 'datetime']
            },
            'communication': {
                'module': 'tests.test_communication',
                'function': 'run_communication_tests',
                'description': 'Unity-Python communication bridge',
                'dependencies': ['socket', 'json', 'threading']
            },
            'environments': {
                'module': 'tests.test_environments',
                'function': 'run_environment_tests',
                'description': 'Environment classes and functionality',
                'dependencies': ['unittest', 'tempfile', 'pathlib']
            }
        }
    
    def check_dependencies(self) -> Dict[str, bool]:
        """
        Check if all required dependencies are available.
        
        Returns:
            Dictionary mapping dependency names to availability status
        """
        dependencies = set()
        for suite_info in self.test_suites.values():
            dependencies.update(suite_info.get('dependencies', []))
        
        availability = {}
        for dep in dependencies:
            try:
                importlib.import_module(dep)
                availability[dep] = True
            except ImportError:
                availability[dep] = False
        
        return availability
    
    def discover_tests(self) -> List[str]:
        """
        Discover available test files in the tests directory.
        
        Returns:
            List of test file names found
        """
        tests_dir = self.project_root / "tests"
        
        if not tests_dir.exists():
            print("Tests directory not found!")
            return []
        
        test_files = []
        for file_path in tests_dir.glob("test_*.py"):
            if file_path.name != "__init__.py":
                test_files.append(file_path.stem)
        
        return test_files
    
    def import_test_module(self, module_name: str) -> Optional[Any]:
        """
        Safely import a test module.
        
        Args:
            module_name: Name of the module to import
            
        Returns:
            Imported module or None if import failed
        """
        try:
            # Add project root to Python path
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
            
            module = importlib.import_module(module_name)
            return module
        except ImportError as e:
            if self.verbose:
                print(f"Failed to import {module_name}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error importing {module_name}: {e}")
            return None
    
    def run_test_suite(self, suite_name: str, suite_info: Dict[str, Any]) -> TestResult:
        """
        Run a specific test suite.
        
        Args:
            suite_name: Name of the test suite
            suite_info: Information about the test suite
            
        Returns:
            TestResult object with results
        """
        result = TestResult(suite_name)
        start_time = time.time()
        
        try:
            print(f"\nRunning {suite_name} tests: {suite_info['description']}")
            
            # Import the test module
            module = self.import_test_module(suite_info['module'])
            if not module:
                result.error_message = f"Failed to import module {suite_info['module']}"
                return result
            
            # Get the test function
            test_function_name = suite_info.get('function')
            if test_function_name and hasattr(module, test_function_name):
                # Run the specific test function
                test_function = getattr(module, test_function_name)
                
                if self.verbose:
                    print(f"Calling {test_function_name}() directly...")
                
                try:
                    # Call the function and capture its output
                    test_result = test_function()
                    
                    if isinstance(test_result, bool):
                        result.passed = test_result
                    elif isinstance(test_result, dict):
                        result.passed = test_result.get('success', False)
                        result.test_count = test_result.get('tests_run', 0)
                        result.failure_count = test_result.get('failures', 0)
                        result.details = test_result
                    else:
                        result.passed = True  # Function completed without exception
                        
                except Exception as e:
                    result.error_message = str(e)
                    if self.verbose:
                        result.error_message += f"\n{traceback.format_exc()}"
                
            else:
                # Use unittest discovery as fallback
                if self.verbose:
                    print(f"Function {test_function_name} not found, using unittest discovery...")
                
                result = self._run_unittest_discovery(suite_name, suite_info['module'])
        
        except Exception as e:
            result.error_message = f"Unexpected error: {str(e)}"
            if self.verbose:
                result.error_message += f"\n{traceback.format_exc()}"
        
        result.duration = time.time() - start_time
        return result
    
    def _run_unittest_discovery(self, suite_name: str, module_name: str) -> TestResult:
        """
        Run tests using unittest discovery as a fallback method.
        
        Args:
            suite_name: Name of the test suite
            module_name: Name of the module
            
        Returns:
            TestResult object with results
        """
        result = TestResult(suite_name)
        
        try:
            print(f"Fallback: Running {module_name} with unittest discovery")
            
            # Import the module to make sure it exists
            module = self.import_test_module(module_name)
            if not module:
                result.error_message = f"Module {module_name} not available"
                return result
            
            # Create a test loader
            loader = unittest.TestLoader()
            
            # Load tests from the module
            try:
                suite = loader.loadTestsFromModule(module)
                
                # Run the tests
                runner = unittest.TextTestRunner(
                    verbosity=2 if self.verbose else 1,
                    stream=sys.stdout,
                    buffer=True
                )
                
                test_result = runner.run(suite)
                
                # Process results
                result.passed = test_result.wasSuccessful()
                result.test_count = test_result.testsRun
                result.failure_count = len(test_result.failures) + len(test_result.errors)
                
                if not result.passed:
                    error_details = []
                    for failure in test_result.failures:
                        error_details.append(f"FAIL: {failure[0]}\n{failure[1]}")
                    for error in test_result.errors:
                        error_details.append(f"ERROR: {error[0]}\n{error[1]}")
                    result.error_message = "\n".join(error_details)
                
            except Exception as e:
                result.error_message = f"Error running unittest discovery: {str(e)}"
                
        except Exception as e:
            result.error_message = f"Error in fallback testing: {str(e)}"
        
        return result
    
    def run_all_tests(self, specific_test: Optional[str] = None) -> Dict[str, TestResult]:
        """
        Run all available tests or a specific test.
        
        Args:
            specific_test: Name of specific test to run, or None for all tests
            
        Returns:
            Dictionary mapping test names to their results
        """
        print("AI Escape Cage Training System - Comprehensive Test Suite")
        print("=" * 60)
        
        # Display system information
        print(f"Python version: {sys.version}")
        print(f"Project root: {self.project_root}")
        
        # Check dependencies
        deps = self.check_dependencies()
        missing_deps = [dep for dep, available in deps.items() if not available]
        if missing_deps:
            print(f"Missing dependencies: {', '.join(missing_deps)}")
        
        # Determine which tests to run
        if specific_test:
            if specific_test in self.test_suites:
                print(f"Running specific test suite: {specific_test}")
                tests_to_run = {specific_test: self.test_suites[specific_test]}
            else:
                print(f"Test suite '{specific_test}' not found!")
                print(f"Available test suites: {', '.join(self.test_suites.keys())}")
                return {}
        else:
            print(f"Running all {len(self.test_suites)} test suites")
            tests_to_run = self.test_suites
        
        # Run each test suite
        for suite_name, suite_info in tests_to_run.items():
            try:
                # Check if required modules are available
                module = self.import_test_module(suite_info['module'])
                if not module:
                    print(f"\nRequired test module {suite_info['module']} not found!")
                    result = TestResult(suite_name)
                    result.error_message = f"Module {suite_info['module']} not available"
                    self.test_results[suite_name] = result
                    continue
                
                # Run the test suite
                result = self.run_test_suite(suite_name, suite_info)
                self.test_results[suite_name] = result
                
                # Display immediate results
                if result.passed:
                    print(f"PASSED: {suite_name} tests completed successfully in {result.duration:.2f}s")
                else:
                    print(f"FAILED: {suite_name} tests failed in {result.duration:.2f}s")
                    if result.error_message and self.verbose:
                        print(f"Error details: {result.error_message}")
                
            except Exception as e:
                print(f"Error running {suite_name}: {str(e)}")
                result = TestResult(suite_name)
                result.error_message = f"Unexpected error: {str(e)}"
                self.test_results[suite_name] = result
                
                # Try fallback method
                print(f"Attempting fallback test method for {suite_name}...")
                fallback_result = self._run_unittest_discovery(suite_name, suite_info['module'])
                if fallback_result.passed:
                    print(f"Fallback successful for {suite_name}")
                    self.test_results[suite_name] = fallback_result
        
        return self.test_results
    
    def print_summary(self):
        """Print a comprehensive summary of test results."""
        if not self.test_results:
            print("\nNo tests were run.")
            return
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.passed)
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(result.duration for result in self.test_results.values())
        
        # Overall result
        if failed_tests == 0:
            print("OVERALL RESULT: ALL TESTS PASSED!")
        else:
            print("OVERALL RESULT: SOME TESTS FAILED!")
        
        print(f"\nTotal execution time: {total_duration:.2f} seconds")
        print(f"Test suites run: {len(self.test_results)}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        
        # Individual test counts
        total_tests = sum(result.test_count for result in self.test_results.values() if result.test_count > 0)
        total_failures = sum(result.failure_count for result in self.test_results.values() if result.failure_count > 0)
        
        if total_tests > 0:
            print(f"Total individual tests: {total_tests}")
            print(f"Total failures: {total_failures}")
        
        # Detailed results
        print(f"\nDETAILED RESULTS:")
        for suite_name, result in self.test_results.items():
            status = "PASS" if result.passed else "FAIL"
            print(f"  {suite_name:<20} {status:<6} ({result.duration:.2f}s)")
            
            if not result.passed and result.error_message:
                # Show first line of error message
                first_line = result.error_message.split('\n')[0]
                print(f"    Error: {first_line}")
        
        # Recommendations
        if failed_tests > 0:
            print(f"\nRECOMMENDATIONS:")
            print("- Check the error details above for specific failures")
            print("- Ensure all dependencies are properly installed")
            print("- Verify that Unity is not running during environment tests")
            print("- Try running tests individually for more detailed output")


def main():
    """Main function for running comprehensive tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Test Suite Runner")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show detailed output")
    parser.add_argument("--test", "-t", type=str, 
                       help="Run specific test suite")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available test suites")
    
    args = parser.parse_args()
    
    runner = ComprehensiveTestRunner(verbose=args.verbose)
    
    if args.list:
        print("Available test suites:")
        for name, info in runner.test_suites.items():
            print(f"  {name:<20} - {info['description']}")
        return
    
    # Run tests
    try:
        results = runner.run_all_tests(specific_test=args.test)
        runner.print_summary()
        
        # Exit with appropriate code
        failed_count = sum(1 for result in results.values() if not result.passed)
        sys.exit(0 if failed_count == 0 else 1)
        
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 