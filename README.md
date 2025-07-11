# AI Escape Cage Training System

Train AI agents to escape virtual cages using reinforcement learning. **Mix and match any trainer with any environment** for maximum flexibility!

## 🚀 Quick Start for New Users

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Verify Setup**
```bash
python test_system.py --quick
```

### 3. **Your First AI (5-10 minutes)**
```bash
python ml_training/escape_cage_trainer.py --trainer fast
```

### 4. **Test Your Trained Model**
```bash
python test_system.py --component models
```

## 🎯 Understanding the System

The system has **2 key concepts** that you can mix and match:

### **Trainers** (Learning Strategy) 🧠
- **`standard`**: Stable, balanced learning (15-30 min, 85-95% success)
- **`fast`**: Quick, aggressive learning (5-10 min, 70-85% success)  
- **`continue`**: Improve existing models (variable time, builds on existing)

### **Environments** (Reward System) 🎮
- **`simple`**: Basic rewards (+100 escape, +10 key, -0.01 time)
- **`fast`**: Enhanced rewards (+200 escape, +50 key, progress bonuses)
- **`debug`**: Detailed logging and slow steps for training diagnostics

### **🧠 Smart Defaults - No `--environment` Needed!**
Each trainer automatically picks the best environment:
- **`--trainer standard`** → uses `simple` environment (stable learning)
- **`--trainer fast`** → uses `fast` environment (aggressive learning)  
- **`--trainer continue`** → uses `simple` environment (conservative improvement)

**Translation**: Just run `python ml_training/escape_cage_trainer.py --trainer fast` and it automatically uses the fast environment!

## 🔥 All Training Combinations (9 Total!)

### **Basic Command Structure**
```bash
python ml_training/escape_cage_trainer.py --trainer [TRAINER] --environment [ENV]
```

### **Quick Reference Table**

| Command | Learning | Rewards | Time | Success | Best For |
|---------|----------|---------|------|---------|----------|
| `--trainer fast` | Quick | Enhanced | 5-10 min | 70-85% | **Fastest results** |
| `--trainer fast --environment simple` | Quick | Basic | 5-10 min | 70-85% | **Rapid prototyping** |
| `--trainer fast --environment debug` | Quick | Basic + logs | 5-10 min | 70-85% | **Debug fast training** |
| `--trainer standard` | Stable | Basic | 15-30 min | 85-95% | **Production ready** |
| `--trainer standard --environment fast` | Stable | Enhanced | 15-30 min | 90-98% | **🏆 BEST PERFORMANCE** |
| `--trainer standard --environment debug` | Stable | Basic + logs | 15-30 min | 85-95% | **Debug stable training** |
| `--trainer continue` | Improve existing | Basic | Variable | Improves | **Improve existing** |
| `--trainer continue --environment fast` | Improve existing | Enhanced | Variable | Improves | **Upgrade existing** |
| `--trainer continue --environment debug` | Improve existing | Basic + logs | Variable | Improves | **Debug improvements** |

### **Most Common Examples**

```bash
# 🚀 FASTEST: Quick learning + enhanced rewards
python ml_training/escape_cage_trainer.py --trainer fast

# 🏆 BEST: Stable learning + enhanced rewards
python ml_training/escape_cage_trainer.py --trainer standard --environment fast

# 🔧 BASIC: Stable learning + basic rewards
python ml_training/escape_cage_trainer.py --trainer standard

# 📈 IMPROVE: Continue existing model with better rewards 
python ml_training/escape_cage_trainer.py --trainer continue --environment fast --model my_model

# 🐛 DEBUG: Any trainer with detailed logging
python ml_training/escape_cage_trainer.py --trainer standard --environment debug --steps 10000
```

## 🎓 New User Learning Path

### **Step 1: Quick Success (5 minutes)**
```bash
# Get immediate results to see the system working
python ml_training/escape_cage_trainer.py --trainer fast --steps 15000 --model my_first_ai
python test_system.py --component models
```

### **Step 2: Better Model (20 minutes)**  
```bash
# Train a high-quality model
python ml_training/escape_cage_trainer.py --trainer standard --environment fast --model my_best_ai
python test_system.py --component models --verbose
```

### **Step 3: Improve Existing (10 minutes)**
```bash
# Enhance your first model with better rewards
python ml_training/escape_cage_trainer.py --trainer continue --environment fast --model my_first_ai
```

### **Step 4: Compare Results**
```bash
# See how they all compare
python test_system.py --component models --verbose
```

## 📊 Environment Details

### **Simple Environment** (Conservative)
```python
# Basic reward structure - good for stable learning
When AI gets key: +10 points
When AI escapes: +100 points  
Each step: -0.01 penalty (encourages efficiency)
```

### **Fast Environment** (Aggressive) 
```python
# Enhanced reward structure - accelerates learning
When AI moves toward key: +1.0 progress reward
When AI gets close to key: +5.0 proximity bonus
When AI gets key: +50 points (5x more)
When AI moves toward exit: +2.0 progress reward  
When AI gets close to exit: +10.0 proximity bonus
When AI escapes: +200 points (2x more)
Each step: -0.1 penalty (10x more - encourages speed)
```

### **Debug Environment** (Development)
```python
# Same rewards as Simple BUT with extensive diagnostics:
Slower steps (0.2s vs 0.1s) for better observation
Detailed logging of all actions and decisions
Longer episodes (500 vs 200-500 steps) for thorough analysis
Perfect for debugging training problems
```

## ❓ When to Use `--environment`?

### **✅ Skip `--environment` (Use Smart Defaults)**
```bash
# Most common - just pick your trainer, environment is automatic:
python ml_training/escape_cage_trainer.py --trainer fast     # Uses 'fast' env
python ml_training/escape_cage_trainer.py --trainer standard # Uses 'simple' env
python ml_training/escape_cage_trainer.py --trainer continue # Uses 'simple' env
```

### **🔧 Use `--environment` (Override Defaults)**
```bash
# When you want different combinations:
python ml_training/escape_cage_trainer.py --trainer standard --environment fast   # Best performance
python ml_training/escape_cage_trainer.py --trainer fast --environment simple     # Fast learning, basic rewards
python ml_training/escape_cage_trainer.py --trainer standard --environment debug  # Troubleshoot training issues
```

## 🔧 Complete CLI Reference

### **Training Commands**

```bash
# Basic training (uses defaults)
python ml_training/escape_cage_trainer.py

# Specify trainer and environment
python ml_training/escape_cage_trainer.py --trainer standard --environment fast

# Custom training steps and model name
python ml_training/escape_cage_trainer.py --trainer fast --steps 25000 --model my_custom_ai

# Continue training existing model
python ml_training/escape_cage_trainer.py --trainer continue --model existing_model --continue-steps 30000

# Fine-tune existing model (conservative settings)
python ml_training/escape_cage_trainer.py --trainer continue --model existing_model --fine-tune --fine-steps 20000

# Continue training without backup (faster startup)
python ml_training/escape_cage_trainer.py --trainer continue --model existing_model --no-backup
```

### **Model Management Commands**

```bash
# List all models with details (size, date, success rate)
python ml_training/model_utils.py list

# Model operations
python ml_training/model_utils.py backup --model my_model # Backup specific model
python ml_training/model_utils.py delete --model my_model # Delete specific model
python ml_training/model_utils.py compare --model model1 --model2 model2 # Compare two models
```

### **Testing Commands**

```bash
# Quick system health check (5 tests, ~6 seconds)
python test_system.py --quick

# Full comprehensive test suite (12 tests, ~30 seconds)
python test_system.py

# Test specific components
python test_system.py --component structure    # Project file structure
python test_system.py --component imports      # Python imports and dependencies
python test_system.py --component config       # Configuration system
python test_system.py --component trainers     # Trainer creation and factory
python test_system.py --component environment  # Environment classes
python test_system.py --component unity        # Unity bridge connection
python test_system.py --component models       # Model utilities and management
python test_system.py --component analytics    # Analytics and performance tracking
python test_system.py --component logging      # Logging system
python test_system.py --component training     # Training system interface
python test_system.py --component cli          # Command-line interface
python test_system.py --component integration  # Overall system integration

# Verbose output for debugging
python test_system.py --verbose
python test_system.py --component models --verbose
```

## 🎮 Unity Setup

1. **Open Unity** with the escape cage scene
2. **Press Play** to start the game  
3. **Run any training command** - automatic connection on port 9999
4. **Watch your AI learn** in real-time!

## 🛠️ Troubleshooting

### **Connection Issues**
```bash
# Unity not responding?
# 1. Make sure Unity scene is loaded and playing
# 2. Check firewall isn't blocking port 9999  
# 3. Only run one training session at a time

# Test Unity connection specifically
python test_system.py --component unity
```

### **Model Issues**
```bash
# See what models you have
python ml_training/escape_cage_trainer.py --list-models

# Test model system
python test_system.py --component models

# Check models directory
ls models/
```

### **Training Problems**
```bash
# Use debug environment to see what's happening
python ml_training/escape_cage_trainer.py --trainer standard --environment debug --steps 10000

# Check system health
python test_system.py --quick

# Test training system interface
python test_system.py --component training
```

### **Performance Issues**
```bash
# Try better reward environment
python ml_training/escape_cage_trainer.py --trainer standard --environment fast

# Train longer with more steps
python ml_training/escape_cage_trainer.py --trainer standard --steps 100000

# Improve existing model instead of starting from scratch
python ml_training/escape_cage_trainer.py --trainer continue --environment fast --model my_model
```

## 📝 Quick Command Reference

### **Most Common Commands**
```bash
# Best for beginners (fast results)
python ml_training/escape_cage_trainer.py --trainer fast

# Best for production (high quality)  
python ml_training/escape_cage_trainer.py --trainer standard --environment fast

# Improve existing model
python ml_training/escape_cage_trainer.py --trainer continue --environment fast --model MODEL_NAME

# Test your models
python test_system.py --component models

# Check system health
python test_system.py --quick

# See all available models with details
python ml_training/model_utils.py list

# Get help for any command
python ml_training/escape_cage_trainer.py --help
python test_system.py --help
python ml_training/model_utils.py --help
```

### **Help & Documentation**
- Add `--help` to any command for detailed options
- See `SETUP.md` for detailed installation instructions
- See `TECHNICAL.md` for system architecture details
- See `CONTRIBUTING.md` for development guidelines

## 🎯 Success Tips

1. **Start with `--trainer fast`** for immediate gratification
2. **Use `--trainer standard --environment fast`** for best results  
3. **Always test your models** with `python test_system.py --component models`
4. **Improve existing models** with `--trainer continue --environment fast`
5. **Use `--environment debug`** when training isn't working as expected
6. **Run `python test_system.py --quick`** to verify system health
7. **Use `python ml_training/escape_cage_trainer.py --list-models`** to see available models

## 🔬 Understanding Training vs Testing

**Two Different Types of "Testing":**

### **Debug Environment** = Training Process Debugging
- **Purpose**: Debug problems *during training* (when the AI is learning)
- **Usage**: `--environment debug`
- **When**: Your AI isn't learning properly and you need detailed logs
- **Example**: "Why isn't my AI picking up the key? Let me see step-by-step what it's doing."

### **Test System** = Model Performance Testing  
- **Purpose**: Test *already trained* models to see how well they perform
- **Usage**: `python test_system.py --component models`
- **When**: After training is complete, to evaluate model performance
- **Example**: "My training finished, now let me see if the AI can consistently solve the puzzle."

Ready to train your escape cage AI! 🤖🏃‍♂️
