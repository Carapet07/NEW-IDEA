# 🤝 Contributing to AI Escape Cage

Thank you for your interest in contributing! This project welcomes contributions from developers of all skill levels.

---

## 🎯 **Ways to Contribute**

### **🎮 Game Development**
- **New Environments**: Create different puzzle scenarios
- **Visual Improvements**: Better graphics, animations, particle effects
- **UI/UX**: Training progress visualization, controls interface
- **Audio**: Sound effects, background music, audio feedback

### **🧠 AI/ML Enhancements**
- **Algorithm Comparison**: Implement A3C, DQN, SAC algorithms
- **Hyperparameter Tuning**: Optimize learning rates, network architectures
- **Advanced Features**: Curriculum learning, transfer learning
- **Performance**: Faster training, better sample efficiency

### **🔧 Technical Improvements**
- **Code Quality**: Refactoring, documentation, type hints
- **Testing**: Unit tests, integration tests, performance benchmarks
- **Platform Support**: Cross-platform compatibility improvements
- **Monitoring**: Training metrics, visualization tools

### **📚 Documentation & Education**
- **Tutorials**: Step-by-step guides for specific features
- **Examples**: Different environment configurations
- **Research**: Analysis of learning behaviors, strategy emergence
- **Outreach**: Blog posts, videos, educational materials

---

## 🚀 **Getting Started**

### **1. Development Setup**
```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/escape-cage-ai.git
cd escape-cage-ai

# Create development environment
python -m venv escape_cage_dev
source escape_cage_dev/bin/activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### **2. Project Structure**
```
escape-cage-ai/
├── communication/          # Unity-Python bridge
├── ml_training/            # AI training scripts
├── unity_project/          # Unity game files (if included)
├── tests/                  # Test suite
├── docs/                   # Documentation
├── examples/               # Example configurations
└── scripts/                # Utility scripts
```

### **3. Development Workflow**
```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# Write tests for new functionality
python -m pytest tests/

# Run code quality checks
black .
flake8 .
mypy .

# Commit and push
git add .
git commit -m "feat: your descriptive commit message"
git push origin feature/your-feature-name
```

---

## 🎨 **Contribution Ideas by Skill Level**

### **🌱 Beginner Friendly**
- **Add new object types** (switches, moving platforms)
- **Improve error messages** and user feedback
- **Create example configurations** for different scenarios
- **Write documentation** for existing features
- **Add unit tests** for utility functions

### **🌿 Intermediate**
- **Implement different RL algorithms** (DQN, A3C)
- **Create training visualization** dashboards
- **Add multi-agent support** for competitive/cooperative learning
- **Improve communication protocol** efficiency
- **Build automated testing** pipelines

### **🌳 Advanced**
- **Research novel RL techniques** for puzzle-solving
- **Implement curriculum learning** frameworks
- **Create sim-to-real transfer** capabilities
- **Develop hierarchical RL** approaches
- **Build distributed training** systems

---

## 📋 **Code Standards**

### **Python Code Style**
```python
# Use type hints
def calculate_reward(observation: Dict[str, Any]) -> float:
    """Calculate reward based on current observation.
    
    Args:
        observation: Current game state dictionary
        
    Returns:
        Calculated reward value
    """
    pass

# Use descriptive variable names
player_position = observation.get('player_position', (0, 0))
has_collected_key = observation.get('has_key', False)

# Follow PEP 8 conventions
MAX_TRAINING_EPISODES = 50000
DEFAULT_LEARNING_RATE = 0.0003
```

### **Unity C# Style**
```csharp
// Use PascalCase for public members
public class GameController : MonoBehaviour
{
    public GameObject Player;
    public float MovementSpeed = 3.0f;
    
    // Use camelCase for private members
    private bool hasKey = false;
    private Vector3 initialPosition;
    
    // Clear method documentation
    /// <summary>
    /// Processes an action received from the AI agent
    /// </summary>
    /// <param name="action">Action number (0-3)</param>
    private void ProcessAction(int action)
    {
        // Implementation here
    }
}
```

### **Documentation Standards**
- **Clear docstrings** for all public functions
- **Type hints** for function parameters and return values
- **Inline comments** for complex logic
- **README updates** for new features
- **Example usage** in docstrings

---

## 🧪 **Testing Guidelines**

### **Test Structure**
```python
# tests/test_environment.py
import pytest
from escape_cage.environment import EscapeCageEnv

class TestEscapeCageEnvironment:
    """Test suite for the escape cage environment."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.env = EscapeCageEnv()
    
    def test_observation_space(self):
        """Test observation space dimensions."""
        obs = self.env.reset()
        assert obs.shape == (7,)
        assert all(-10 <= x <= 10 for x in obs)
    
    def test_action_execution(self):
        """Test action execution."""
        self.env.reset()
        obs, reward, done, _, info = self.env.step(0)  # Move up
        assert isinstance(reward, float)
        assert isinstance(done, bool)
```

### **Test Categories**
- **Unit Tests**: Individual functions and classes
- **Integration Tests**: Component interactions
- **Performance Tests**: Training speed and efficiency
- **Regression Tests**: Ensure changes don't break existing functionality

---

## 📝 **Commit Message Format**

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
type(scope): description

[optional body]

[optional footer]
```

### **Examples**
```bash
feat(ai): add DQN algorithm implementation
fix(unity): resolve player boundary collision issue
docs(readme): update installation instructions
test(env): add unit tests for reward calculation
refactor(bridge): simplify communication protocol
```

### **Types**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

---

## 🔍 **Code Review Process**

### **Before Submitting PR**
- [ ] All tests pass (`python -m pytest`)
- [ ] Code follows style guidelines (`black .` and `flake8 .`)
- [ ] Type checking passes (`mypy .`)
- [ ] Documentation updated if needed
- [ ] Commit messages follow convention

### **PR Template**
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Screenshots (if applicable)
Include screenshots for UI changes.

## Additional Notes
Any additional context or considerations.
```

---

## 🌟 **Recognition**

### **Contributors**
All contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **Social media** shoutouts for major features

### **Contribution Types**
We value all types of contributions:
- **Code**: Features, bug fixes, optimizations
- **Documentation**: Tutorials, examples, improvements
- **Testing**: Writing tests, reporting bugs
- **Design**: UI/UX improvements, graphics
- **Community**: Helping others, discussions, feedback

---

## 🆘 **Getting Help**

### **Development Questions**
- **GitHub Discussions**: [Ask questions](https://github.com/yourusername/escape-cage-ai/discussions)
- **Discord/Slack**: Join our development community
- **Issues**: [Report bugs or request features](https://github.com/yourusername/escape-cage-ai/issues)

### **Learning Resources**
- **Reinforcement Learning**: [Spinning Up in Deep RL](https://spinningup.openai.com/)
- **Unity Development**: [Unity Learn](https://learn.unity.com/)
- **Python Best Practices**: [Real Python](https://realpython.com/)
- **Git Workflow**: [Atlassian Git Tutorials](https://www.atlassian.com/git/tutorials)

---

## 📄 **License**

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project. See [LICENSE](LICENSE) for details.

---

## 🙏 **Thank You**

Your contributions help make this project better for everyone! Whether you're fixing a typo, adding a feature, or helping other users, every contribution matters.

**Happy coding! 🚀**

---

<div align="center">

[⬅️ Back to Main README](README.md) • [🔧 Setup Guide](SETUP.md) • [🔬 Technical Docs](TECHNICAL.md)

</div> 