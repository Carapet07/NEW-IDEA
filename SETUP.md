# Complete Setup Guide - AI Escape Cage Training System

This guide walks you through setting up the AI Escape Cage training system from scratch. Follow these steps to get everything working properly.

## 📋 **Prerequisites Checklist**

### **Required Software**
- **Python 3.8+**: Download from [python.org](https://python.org)
- **Unity 2020.3 LTS or newer**: Download from [unity3d.com](https://unity3d.com)
- **Git**: For cloning the repository
- **Text Editor**: VS Code, PyCharm, or your preferred editor

### **System Requirements**
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: Optional but helpful for faster training
- **Storage**: 5GB free space
- **Network**: Open port 9999 for Unity-Python communication

## 🚀 **Installation Steps**

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/Carapet07/RL-Unity-Puzzle-Solver.git
cd RL-Unity-Puzzle-Solver
```

### **Step 2: Install Python Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt
```

**Core Dependencies Installed:**
- **TensorFlow 2.13+**: Deep learning framework
- **Stable-Baselines3 2.0+**: Reinforcement learning algorithms
- **Gymnasium**: Environment interface
- **NumPy, Pandas, Matplotlib**: Data science tools
- **Jupyter**: Interactive development
- **pytest**: Testing framework

### **Step 3: Verify Installation**
```bash
# Quick system health check
python test_system.py --quick
```

**Expected Output:**
```
INFO: 🚀 Running Quick Tests...
INFO: 🧪 Running: Project Structure
INFO:    ✅ PASS Project Structure (0.00s)
INFO: 🧪 Running: Python Imports
INFO:    ✅ PASS Python Imports (6.03s)
...

🎉 ALL TESTS PASSED! System is healthy and ready for use.
```

### **Step 4: Test Core Components**
```bash
# Test individual components
python test_system.py --component imports    # Test all imports
python test_system.py --component trainers   # Test trainer system
python test_system.py --component models     # Test model utilities
```

## 🎮 **Unity Setup**

### **Unity Project Setup**
1. **Open Unity Hub**
2. **Create New Project** or **Open Existing** escape cage project
3. **Load the escape cage scene**
4. **Press Play** to start the game environment
5. **Verify port 9999 is available** (check firewall settings)

### **Test Unity Connection**
```bash
# Test Unity bridge connection
python test_system.py --component unity
```

**If Unity is running and connected:**
```
✅ PASS Unity Bridge Connection
```

**If Unity is not running:**
```
⚠️  WARN Unity Bridge Connection: Unity not running (expected during setup)
```

## 🔧 **Advanced Setup Options**

### **Development Environment**
```bash
# Create isolated environment (recommended)
python -m venv escape_cage_env
source escape_cage_env/bin/activate  # On Windows: escape_cage_env\Scripts\activate

# Install in development mode
pip install -e .
```

### **Optional: Poetry Setup**
```bash
# If you prefer Poetry for dependency management
pip install poetry
poetry install
poetry shell
```

### **GPU Acceleration (Optional)**
```bash
# For NVIDIA GPUs (faster training)
pip install tensorflow-gpu
```

## ✅ **Verification Tests**

### **Complete System Test**
```bash
# Run comprehensive test suite
python test_system.py
```

**This tests:**
- ✅ Project structure and files
- ✅ Python imports and dependencies
- ✅ Configuration system
- ✅ Trainer factory and creation
- ✅ Environment classes
- ✅ Unity bridge connection
- ✅ Model utilities
- ✅ Analytics system
- ✅ Logging system
- ✅ Training system interface
- ✅ CLI commands
- ✅ System integration

### **Quick Training Test**
```bash
# Test training system (requires Unity)
python ml_training/escape_cage_trainer.py --trainer fast --steps 1000 --model test_setup
```

**Expected behavior:**
- Connects to Unity automatically
- Starts training for 1000 steps
- Saves model as `test_setup.zip`
- Shows progress updates

### **Model Management Test**
```bash
# Test model utilities
python ml_training/model_utils.py list
```

## 🛠️ **Troubleshooting**

### **Python Import Errors**
```bash
# Check Python version
python --version  # Should be 3.8+

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

# Verify Stable-Baselines3
python -c "import stable_baselines3 as sb3; print(f'SB3: {sb3.__version__}')"

# Test all critical imports
python test_system.py --component imports
```

### **Unity Connection Issues**
```bash
# Check if Unity is running
python test_system.py --component unity

# Common solutions:
# 1. Make sure Unity scene is loaded and playing
# 2. Check firewall isn't blocking port 9999
# 3. Restart Unity if connection fails
# 4. Only run one training session at a time
```

### **Permission/Path Issues**
```bash
# On Windows, you might need:
python -m pip install --user -r requirements.txt

# On macOS/Linux, you might need:
pip3 install -r requirements.txt
```

### **Dependency Conflicts**
```bash
# Create clean environment
python -m venv fresh_env
source fresh_env/bin/activate  # On Windows: fresh_env\Scripts\activate
pip install -r requirements.txt
```

## 📦 **Project Structure Overview**

After setup, your project should look like:
```
RL-Unity-Puzzle-Solver/
├── ml_training/
│   ├── escape_cage_trainer.py      # Main training script
│   ├── trainers/                   # Trainer classes
│   ├── config/                     # Configuration system
│   ├── base_environment.py         # Environment classes
│   ├── model_utils.py              # Model management
│   └── analytics_utils.py          # Performance analytics
├── communication/
│   └── unity_bridge.py             # Unity-Python bridge
├── models/                         # Trained AI models
├── logs/                           # Training logs
├── test_system.py                  # Comprehensive testing
├── requirements.txt                # Python dependencies
└── README.md                       # Quick start guide
```

## 🎯 **Next Steps**

### **1. Quick Success Test**
```bash
# Train your first AI (5-10 minutes)
python ml_training/escape_cage_trainer.py --trainer fast

# Test the trained model
python test_system.py --component models --verbose
```

### **2. Explore the System**
```bash
# See all available commands
python ml_training/escape_cage_trainer.py --help
python ml_training/model_utils.py --help
python test_system.py --help
```

### **3. Read Documentation**
- **README.md**: Quick start and basic commands
- **MODEL_USAGE.md**: Working with trained models
- **TESTING_GUIDE.md**: Comprehensive testing system
- **TECHNICAL.md**: System architecture details

## 🔍 **System Health Monitoring**

### **Regular Health Checks**
```bash
# Quick system check (run anytime)
python test_system.py --quick

# Full system validation
python test_system.py --verbose
```

### **Performance Monitoring**
```bash
# Check system resources
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().percent}%')"

# Monitor training logs
ls logs/
```

## 🎉 **Success Indicators**

You're ready to start training when:
- ✅ `python test_system.py --quick` passes all tests
- ✅ Unity connects successfully
- ✅ Training command starts without errors
- ✅ Models save to `models/` directory
- ✅ System responds to all CLI commands

## 📚 **Additional Resources**

### **Documentation**
- **README.md**: Main project overview
- **CONTRIBUTING.md**: Development guidelines
- **RESULTS.md**: Project achievements and metrics

### **Support**
- Check existing issues on GitHub
- Review troubleshooting sections in other docs
- Run `python test_system.py --verbose` for detailed diagnostics

**Ready to train your AI escape agent!** 🤖🚀

---

**Next:** Read [README.md](README.md) for quick start commands or [MODEL_USAGE.md](MODEL_USAGE.md) for working with trained models.
