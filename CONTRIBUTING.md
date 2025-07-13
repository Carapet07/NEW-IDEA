# Contributing to AI Escape Cage Training System

We welcome contributions from developers, researchers, and AI enthusiasts! This guide will help you get started with contributing to the project.

## Ways to Contribute

### **Game Development**
- Unity scene improvements and visual enhancements
- New environment layouts and puzzle mechanics
- Performance optimizations and graphics improvements
- Mobile platform support and cross-platform compatibility

### **AI/ML Enhancements**
- New reinforcement learning algorithms (A2C, SAC, TD3)
- Hyperparameter optimization and training improvements
- Curriculum learning and progressive difficulty systems
- Multi-agent environments and cooperative learning

### **Technical Improvements**
- Code optimization and performance enhancements
- Bug fixes and error handling improvements
- Documentation updates and API improvements
- Testing framework expansions and validation tools
- Cross-platform compatibility and deployment options

### **Research & Analysis**
- Performance benchmarking and comparative studies
- Behavioral analysis and interpretability research
- Transfer learning experiments and domain adaptation
- Publication-quality documentation and research papers

## Getting Started

### **1. Fork and Clone**
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/yourusername/escape-cage-ai.git
cd escape-cage-ai

# Add upstream remote
git remote add upstream https://github.com/original/escape-cage-ai.git
```

### **2. Development Setup**
```bash
# Create development environment
python -m venv escape_cage_dev
source escape_cage_dev/bin/activate  # On Windows: escape_cage_dev\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests to verify setup
python run_comprehensive_tests.py
```

### **3. Create Feature Branch**
```bash
# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Keep branch updated with upstream
git fetch upstream
git rebase upstream/main
```

## Code Standards

### **Python Code Style**
- **PEP 8**: Follow Python style guidelines strictly
- **Type Hints**: Use type annotations for all functions
- **Docstrings**: Google-style docstrings for all public functions
- **Import Organization**: Standard library, third-party, local imports
- **Line Length**: Maximum 88 characters (Black formatter default)

### **Example Code Structure**
```python
"""Module docstring describing purpose and usage."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from stable_baselines3 import PPO

from ml_training.base_environment import BaseEscapeCageEnv


class NewFeature:
    """Class for implementing new functionality.
    
    Args:
        param1: Description of parameter
        param2: Description of parameter
        
    Attributes:
        attribute1: Description of attribute
    """
    
    def __init__(self, param1: str, param2: int = 10):
        """Initialize new feature with parameters."""
        self.param1 = param1
        self.param2 = param2
    
    def process_data(self, data: List[float]) -> Dict[str, float]:
        """Process input data and return results.
        
        Args:
            data: List of float values to process
            
        Returns:
            Dictionary containing processed results
            
        Raises:
            ValueError: If data is empty or invalid
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        return {"mean": np.mean(data), "std": np.std(data)}
```

### **Unity C# Standards**
- **Unity Naming Conventions**: PascalCase for public, camelCase for private
- **Component Organization**: Single responsibility per script
- **Performance**: Object pooling for frequently created/destroyed objects
- **Documentation**: XML documentation comments for public methods

## Testing Guidelines

### **Required Tests**
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Performance Tests**: Verify training speed and memory usage
- **Communication Tests**: Test Unity-Python bridge reliability

### **Test Structure**
```python
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

class TestNewFeature(unittest.TestCase):
    """Test cases for NewFeature class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.feature = NewFeature("test", 20)
        self.test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    def test_process_data_valid_input(self):
        """Test process_data with valid input."""
        result = self.feature.process_data(self.test_data)
        
        self.assertIn("mean", result)
        self.assertIn("std", result)
        self.assertAlmostEqual(result["mean"], 3.0, places=2)
    
    def test_process_data_empty_input(self):
        """Test process_data raises ValueError for empty input."""
        with self.assertRaises(ValueError):
            self.feature.process_data([])
    
    @patch('numpy.mean')
    def test_process_data_with_mock(self, mock_mean):
        """Test process_data with mocked dependencies."""
        mock_mean.return_value = 10.0
        
        result = self.feature.process_data(self.test_data)
        
        mock_mean.assert_called_once_with(self.test_data)
        self.assertEqual(result["mean"], 10.0)

if __name__ == '__main__':
    unittest.main()
```

### **Running Tests**
```bash
# Run all tests
python run_comprehensive_tests.py

# Run specific test module
python -m unittest tests.test_new_feature

# Run with coverage
python run_comprehensive_tests.py --coverage

# Run performance benchmarks
python -m pytest tests/ --benchmark-only
```

## Submission Process

### **1. Development Workflow**
```bash
# Make your changes
git add .
git commit -m "Add new feature: brief description

Detailed description of changes made, including:
- What was added/modified/fixed
- Why the change was necessary  
- Any breaking changes or migration notes"

# Push to your fork
git push origin feature/your-feature-name
```

### **2. Pull Request Guidelines**
- **Clear Title**: Descriptive title summarizing the change
- **Detailed Description**: Explain what, why, and how
- **Tests**: Include tests for new functionality
- **Documentation**: Update relevant documentation
- **Backwards Compatibility**: Note any breaking changes

### **3. Code Review Process**
- **Automated Checks**: CI/CD pipeline runs tests and linting
- **Peer Review**: At least one maintainer reviews the code
- **Feedback Integration**: Address review comments promptly
- **Final Approval**: Maintainer approves and merges

## Areas Needing Contribution

### **High Priority**
- Performance optimization for training loops
- Cross-platform Unity build support
- Enhanced error handling in communication layer
- Documentation improvements and examples

### **Medium Priority**
- Additional RL algorithm implementations
- Visualization tools for training analysis
- Model comparison and benchmarking utilities
- Advanced environment configurations

### **Research Opportunities**
- Curriculum learning implementations
- Multi-agent environment support
- Transfer learning between environments
- Behavioral analysis and interpretability tools

## Getting Help

### **Communication Channels**
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Request Comments**: Code-specific discussions

### **Documentation Resources**
- **README.md**: Project overview and quick start
- **TECHNICAL.md**: Detailed technical documentation
- **API Documentation**: Generated from code docstrings

### **Development Environment Help**
- **setup_dev_environment.py**: Automated development setup
- **troubleshooting.md**: Common issues and solutions
- **example_contributions/**: Sample contributions for reference

## Recognition

### **Contributors**
- All contributors are recognized in CONTRIBUTORS.md
- Significant contributions receive special acknowledgment
- Academic contributors can be listed on research papers

### **Types of Recognition**
- **Code Contributors**: Listed with GitHub profile links
- **Research Contributors**: Academic affiliation and ORCID
- **Documentation Contributors**: Technical writing acknowledgment
- **Bug Reporters**: Recognition for valuable issue reports

**Happy coding!**

---

**Questions? Need help getting started?**
- Create an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing documentation for common solutions

[Back to Main README](README.md) • [Setup Guide](SETUP.md) • [Technical Docs](TECHNICAL.md) 