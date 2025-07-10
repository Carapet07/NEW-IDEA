# Complete Setup Guide - AI Escape Cage

This guide walks you through setting up the AI Escape Cage training system from scratch. Follow these steps to get everything working properly.

## **Prerequisites Checklist**

### **Required Software**
- **Python 3.8+**: Download from python.org
- **Unity 2020.3 LTS or newer**: Download from unity3d.com
- **Git**: For cloning the repository
- **Text Editor**: VS Code, PyCharm, or your preferred editor

### **System Requirements**
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: Multi-core processor (4+ cores recommended)
- **GPU**: Optional but helpful for faster training
- **Storage**: 5GB free space
- **Network**: Open port 9999 for Unity-Python communication

### **Install Python Dependencies**
```bash
# Install core requirements
pip install tensorflow>=2.8.0
pip install stable-baselines3[extra]
pip install numpy matplotlib seaborn
pip install gymnasium

# Verify installation
python -c "import tensorflow as tf; import stable_baselines3; print('All imports successful!')"
```

### **Test Your Setup**
```bash
# Quick environment test
python -c "
import sys
print(f'Python version: {sys.version}')
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
import stable_baselines3
print('All imports successful!')
print('System ready for AI training!')
"
```

**Expected Output:**
```
Testing Python Environment...
All imports successful!
System ready for AI training!
```

**Ready to watch your AI learn? Let's go!**

[Back to Main README](README.md) • [View Results](RESULTS.md) 