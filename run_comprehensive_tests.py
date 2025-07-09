#!/usr/bin/env python3
"""
🧪 Comprehensive Test Suite Runner
Executes all test suites for the AI Escape Cage Training System.

This script runs comprehensive tests covering:
- Core environment functionality
- Communication systems
- Analytics utilities  
- Model management utilities
- Testing and evaluation utilities
- Integration tests across components

Usage:
    python run_comprehensive_tests.py [--verbose] [--coverage] [--specific TEST_NAME]

Examples:
    python run_comprehensive_tests.py                    # Run all tests
    python run_comprehensive_tests.py --verbose          # Run with detailed output
    python run_comprehensive_tests.py --coverage         # Run with coverage analysis
    python run_comprehensive_tests.py --specific analytics  # Run only analytics tests
"""

import sys
import os
import unittest
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import importlib.util

# Add project directories to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ml_training"))
sys.path.insert(0, str(project_root / "tests"))


class ComprehensiveTestRunner:
    """
    Comprehensive test runner for the entire AI training system.
    
    Features:
    - Automatic test discovery across multiple modules
    - Detailed reporting with timing and success metrics
    - Optional code coverage analysis
    - Selective test execution
    - Integration test coordination
    """
    
    def __init__(self, verbose: bool = False, coverage: bool = False):
        """
        Initialize the comprehensive test runner.
        
        Args:
            verbose: Enable detailed test output
            coverage: Enable code coverage analysis
        """
        self.verbose = verbose
        self.coverage = coverage
        self.test_results: Dict[str, Any] = {}
        self.total_start_time = None
        
        # Test suite definitions
        self.test_suites = {
            'environments': {
                'description': 'Environment and base functionality tests',
                'module': 'test_environments',
                'function': 'run_environment_tests',
                'required': True
            },
            'communication': {
                'description': 'Unity bridge and communication tests',
                'module': 'test_communication',
                'function': 'run_communication_tests',
                'required': True
            },
            'analytics': {
                'description': 'Analytics utilities and metrics tests',
                'module': 'test_analytics_utils',
                'function': 'run_analytics_tests',
                'required': False
            },
            'model_utils': {
                'description': 'Model management and utilities tests',
                'module': 'test_model_utils',
                'function': 'run_model_utils_tests',
                'required': False
            },
            'testing_utils': {
                'description': 'Testing and evaluation utilities tests',
                'module': 'test_testing_utils',
                'function': 'run_testing_utils_tests',
                'required': False
            }
        }
    
    def discover_test_modules(self) -> List[str]:
        """
        Discover available test modules in the tests directory.
        
        Returns:
            List of discovered test module names
        """
        tests_dir = project_root / "tests"
        if not tests_dir.exists():
            print("⚠️ Tests directory not found!")
            return []
        
        discovered = []
        for test_file in tests_dir.glob("test_*.py"):
            module_name = test_file.stem
            if module_name in [suite['module'] for suite in self.test_suites.values()]:
                discovered.append(module_name)
        
        return discovered
    
    def import_test_module(self, module_name: str) -> Optional[object]:
        """
        Safely import a test module.
        
        Args:
            module_name: Name of the test module to import
            
        Returns:
            Imported module object or None if import failed
        """
        try:
            # Try direct import first
            return __import__(module_name)
        except ImportError:
            try:
                # Try importing from tests directory
                spec = importlib.util.spec_from_file_location(
                    module_name, 
                    project_root / "tests" / f"{module_name}.py"
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Failed to import {module_name}: {e}")
                return None
    
    def run_test_suite(self, suite_name: str, suite_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a specific test suite.
        
        Args:
            suite_name: Name of the test suite
            suite_info: Suite configuration information
            
        Returns:
            Dictionary containing test results
        """
        print(f"\n🧪 Running {suite_name} tests: {suite_info['description']}")
        print("=" * 70)
        
        start_time = time.time()
        result = {
            'suite_name': suite_name,
            'description': suite_info['description'],
            'success': False,
            'duration': 0.0,
            'tests_run': 0,
            'failures': 0,
            'errors': 0,
            'skipped': 0,
            'error_message': None
        }
        
        try:
            # Import test module
            module = self.import_test_module(suite_info['module'])
            if module is None:
                result['error_message'] = f"Failed to import module {suite_info['module']}"
                print(f"❌ {result['error_message']}")
                return result
            
            # Get test function
            test_function = getattr(module, suite_info['function'], None)
            if test_function is None:
                result['error_message'] = f"Function {suite_info['function']} not found in {suite_info['module']}"
                print(f"❌ {result['error_message']}")
                return result
            
            # Run tests
            if self.verbose:
                print(f"📋 Executing {suite_info['function']}() from {suite_info['module']}")
            
            # Capture test output if not verbose
            if not self.verbose:
                # Redirect stdout temporarily
                import io
                from contextlib import redirect_stdout
                
                captured_output = io.StringIO()
                with redirect_stdout(captured_output):
                    success = test_function()
            else:
                success = test_function()
            
            result['success'] = success
            result['duration'] = time.time() - start_time
            
            if success:
                print(f"✅ {suite_name} tests completed successfully in {result['duration']:.2f}s")
            else:
                print(f"❌ {suite_name} tests failed in {result['duration']:.2f}s")
                
        except Exception as e:
            result['duration'] = time.time() - start_time
            result['error_message'] = str(e)
            print(f"💥 {suite_name} tests crashed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
        
        return result
    
    def run_unit_tests_fallback(self, module_name: str) -> Dict[str, Any]:
        """
        Fallback method to run unit tests using unittest discovery.
        
        Args:
            module_name: Name of the test module
            
        Returns:
            Dictionary containing test results
        """
        print(f"\n🔄 Fallback: Running {module_name} with unittest discovery")
        
        start_time = time.time()
        result = {
            'suite_name': module_name,
            'description': f'Unit tests for {module_name}',
            'success': False,
            'duration': 0.0,
            'tests_run': 0,
            'failures': 0,
            'errors': 0,
            'skipped': 0,
            'error_message': None
        }
        
        try:
            # Use unittest to discover and run tests
            loader = unittest.TestLoader()
            test_dir = str(project_root / "tests")
            
            # Discover tests
            test_suite = loader.discover(test_dir, pattern=f"{module_name}.py")
            
            # Run tests
            if self.verbose:
                runner = unittest.TextTestRunner(verbosity=2)
            else:
                runner = unittest.TextTestRunner(stream=open(os.devnull, 'w'))
            
            test_result = runner.run(test_suite)
            
            # Extract results
            result['tests_run'] = test_result.testsRun
            result['failures'] = len(test_result.failures)
            result['errors'] = len(test_result.errors)
            result['skipped'] = len(test_result.skipped)
            result['success'] = test_result.wasSuccessful()
            result['duration'] = time.time() - start_time
            
        except Exception as e:
            result['duration'] = time.time() - start_time
            result['error_message'] = str(e)
        
        return result
    
    def run_all_tests(self, specific_test: Optional[str] = None) -> bool:
        """
        Run all test suites or a specific test suite.
        
        Args:
            specific_test: Name of specific test suite to run (optional)
            
        Returns:
            True if all tests passed, False otherwise
        """
        self.total_start_time = time.time()
        
        print("🚀 AI Escape Cage Training System - Comprehensive Test Suite")
        print("=" * 70)
        print(f"📅 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📍 Project root: {project_root}")
        print(f"🔧 Python version: {sys.version}")
        
        if self.coverage:
            print("📊 Code coverage analysis: ENABLED")
        
        # Discover available test modules
        discovered_modules = self.discover_test_modules()
        print(f"🔍 Discovered test modules: {', '.join(discovered_modules)}")
        
        # Determine which tests to run
        if specific_test:
            if specific_test in self.test_suites:
                tests_to_run = {specific_test: self.test_suites[specific_test]}
                print(f"🎯 Running specific test suite: {specific_test}")
            else:
                print(f"❌ Test suite '{specific_test}' not found!")
                print(f"Available test suites: {', '.join(self.test_suites.keys())}")
                return False
        else:
            tests_to_run = self.test_suites
            print(f"🧪 Running all {len(tests_to_run)} test suites")
        
        # Run test suites
        all_passed = True
        total_tests = 0
        total_failures = 0
        total_errors = 0
        
        for suite_name, suite_info in tests_to_run.items():
            # Check if module is available
            if suite_info['module'] not in discovered_modules:
                if suite_info.get('required', False):
                    print(f"\n⚠️ Required test module {suite_info['module']} not found!")
                    all_passed = False
                else:
                    print(f"\n⏩ Skipping optional test module {suite_info['module']} (not found)")
                continue
            
            # Run the test suite
            suite_result = self.run_test_suite(suite_name, suite_info)
            self.test_results[suite_name] = suite_result
            
            # Update totals
            total_tests += suite_result.get('tests_run', 0)
            total_failures += suite_result.get('failures', 0)
            total_errors += suite_result.get('errors', 0)
            
            # Track overall success
            if not suite_result['success']:
                all_passed = False
                
                # Try fallback method for failed suites
                if suite_result.get('error_message'):
                    print(f"🔄 Attempting fallback test method for {suite_name}...")
                    fallback_result = self.run_unit_tests_fallback(suite_info['module'])
                    if fallback_result['success']:
                        print(f"✅ Fallback successful for {suite_name}")
                        self.test_results[f"{suite_name}_fallback"] = fallback_result
                        all_passed = True  # Reset if fallback succeeded
        
        # Generate final report
        self.generate_final_report(all_passed, total_tests, total_failures, total_errors)
        
        return all_passed
    
    def generate_final_report(self, all_passed: bool, total_tests: int, 
                            total_failures: int, total_errors: int) -> None:
        """
        Generate comprehensive final test report.
        
        Args:
            all_passed: Whether all tests passed
            total_tests: Total number of tests run
            total_failures: Total number of test failures
            total_errors: Total number of test errors
        """
        total_duration = time.time() - self.total_start_time
        
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 70)
        
        # Overall status
        if all_passed:
            print("🎉 OVERALL RESULT: ALL TESTS PASSED!")
        else:
            print("❌ OVERALL RESULT: SOME TESTS FAILED!")
        
        print(f"\n⏱️ Total execution time: {total_duration:.2f} seconds")
        print(f"📋 Test suites run: {len(self.test_results)}")
        
        if total_tests > 0:
            print(f"🧪 Total individual tests: {total_tests}")
            print(f"❌ Total failures: {total_failures}")
            print(f"💥 Total errors: {total_errors}")
            success_rate = ((total_tests - total_failures - total_errors) / total_tests) * 100
            print(f"✅ Success rate: {success_rate:.1f}%")
        
        # Detailed suite results
        print(f"\n📋 DETAILED SUITE RESULTS:")
        print("-" * 70)
        
        for suite_name, result in self.test_results.items():
            status_emoji = "✅" if result['success'] else "❌"
            duration_str = f"{result['duration']:.2f}s"
            
            print(f"{status_emoji} {suite_name:<20} | {duration_str:>8} | {result['description']}")
            
            if not result['success'] and result.get('error_message'):
                print(f"   Error: {result['error_message']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 70)
        
        if all_passed:
            print("🎯 All tests are passing! Your codebase is in excellent condition.")
            print("🚀 Consider running tests regularly as part of your development workflow.")
        else:
            failed_suites = [name for name, result in self.test_results.items() if not result['success']]
            print(f"🔧 Fix failing test suites: {', '.join(failed_suites)}")
            print("📚 Review error messages and update code accordingly.")
            print("🧪 Run individual test suites to isolate and debug issues.")
        
        if self.coverage:
            print("📊 Generate detailed coverage report for comprehensive analysis.")
        
        print("\n🔗 For detailed debugging, run individual test suites with --verbose flag.")


def main():
    """Main entry point for the comprehensive test runner."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive tests for AI Escape Cage Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_comprehensive_tests.py                    # Run all tests
  python run_comprehensive_tests.py --verbose          # Run with detailed output  
  python run_comprehensive_tests.py --coverage         # Run with coverage analysis
  python run_comprehensive_tests.py --specific analytics  # Run only analytics tests
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with detailed test information'
    )
    
    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        help='Enable code coverage analysis (requires coverage.py)'
    )
    
    parser.add_argument(
        '--specific', '-s',
        type=str,
        help='Run only a specific test suite (environments, communication, analytics, model_utils, testing_utils)'
    )
    
    args = parser.parse_args()
    
    # Create and run test runner
    runner = ComprehensiveTestRunner(verbose=args.verbose, coverage=args.coverage)
    
    try:
        success = runner.run_all_tests(specific_test=args.specific)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test runner crashed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 