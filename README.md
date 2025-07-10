# AI Escape Cage Training System

Train AI agents to escape virtual cages using reinforcement learning. Simple setup, powerful results.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run tests to verify setup:**
   ```bash
   python run_comprehensive_tests.py
   ```

3. **Train your first AI (5-10 minutes):**
   ```bash
   python ml_training/escape_cage_trainer.py --trainer fast
   ```

4. **Test your trained model:**
   ```bash
   python ml_training/testing_utils.py test --model fast_trained_ai
   ```

## 📁 Project Structure

```
ml_training/
├── escape_cage_trainer.py    # All training methods (standard, fast, continue, improved)
├── model_utils.py            # Model management & organization
├── testing_utils.py          # Testing & performance analysis
├── base_environment.py       # Unity connection & environments
├── analytics_utils.py        # Performance analytics
└── logger_setup.py           # Logging setup

communication/
└── unity_bridge.py          # Unity-Python communication

tests/                       # Test suite
```

## 🎯 Essential Commands

### Training
```bash
# Quick training (5-10 min, 70-85% success rate)
python ml_training/escape_cage_trainer.py --trainer fast --steps 25000

# Standard training (15-30 min, 85-95% success rate)  
python ml_training/escape_cage_trainer.py --trainer standard --steps 50000

# Continue training existing model
python ml_training/escape_cage_trainer.py --trainer continue --model my_model

# Advanced training with analytics
python ml_training/escape_cage_trainer.py --trainer improved --steps 50000
```

### Testing & Analysis
```bash
# Test a model
python ml_training/testing_utils.py test --model trained_escape_ai --episodes 10

# Compare two models
python ml_training/testing_utils.py compare --model model1 --model2 model2

# Test multiple models
python ml_training/testing_utils.py batch --models model1 model2 model3

# List available models
python ml_training/testing_utils.py list
```

### Model Management
```bash
# Interactive model manager
python ml_training/model_utils.py interactive

# List all models
python ml_training/model_utils.py list --detailed

# Backup a model
python ml_training/model_utils.py backup --model important_model

# Compare models
python ml_training/model_utils.py compare --model model1 --model2 model2

# Clean up old models
python ml_training/model_utils.py cleanup --days 30
```

### Analytics
```bash
# Analyze training session
python ml_training/analytics_utils.py analyze --session_id latest

# Generate performance report
python ml_training/analytics_utils.py report --model my_model

# Export analytics data
python ml_training/analytics_utils.py export --format json
```

## 🎮 Unity Setup

1. **Open Unity** with the escape cage scene
2. **Press Play** to start the game
3. **Run any training script** - it will automatically connect
4. **Monitor progress** in the console

**Port**: Default 9999 (ensure it's not blocked by firewall)

## 🔧 Common Usage Patterns

### New User Workflow
```bash
# 1. Quick test to see if everything works
python ml_training/escape_cage_trainer.py --trainer fast --steps 10000

# 2. Test the quick model
python ml_training/testing_utils.py test --model fast_trained_ai

# 3. Train a better model
python ml_training/escape_cage_trainer.py --trainer standard

# 4. Compare models
python ml_training/testing_utils.py compare --model fast_trained_ai --model trained_escape_ai
```

### Model Development Cycle
```bash
# 1. Train baseline model
python ml_training/escape_cage_trainer.py --trainer standard --model baseline

# 2. Improve it
python ml_training/escape_cage_trainer.py --trainer continue --model baseline

# 3. Test performance
python ml_training/testing_utils.py test --model baseline_improved --episodes 20

# 4. Backup successful models
python ml_training/model_utils.py backup --model baseline_improved
```

### Performance Analysis
```bash
# 1. Run comprehensive test
python ml_training/testing_utils.py test --model my_model --episodes 50 --detailed

# 2. Analyze the session
python ml_training/analytics_utils.py analyze --model my_model

# 3. Generate report
python ml_training/analytics_utils.py report --model my_model --export
```

## 📊 Training Options Explained

| Trainer | Time | Success Rate | Best For |
|---------|------|--------------|----------|
| `fast` | 5-10 min | 70-85% | Quick testing, prototypes |
| `standard` | 15-30 min | 85-95% | Production models |
| `continue` | Variable | Improves existing | Refining models |
| `improved` | 20-40 min | 90-98% | Best performance |

## 🛠️ Troubleshooting

**Connection Issues:**
```bash
# Check if Unity is running and scene is loaded
# Ensure port 9999 is not blocked
# Only one training session can run at a time
```

**Model Not Found:**
```bash
# List available models
python ml_training/testing_utils.py list

# Check models directory
ls models/
```

**Poor Performance:**
```bash
# Try longer training
python ml_training/escape_cage_trainer.py --trainer standard --steps 100000

# Or continue training existing model
python ml_training/escape_cage_trainer.py --trainer continue --model my_model --continue-steps 50000
```

## 📝 Help & Documentation

- `--help` flag works with all scripts
- See `SETUP.md` for detailed installation
- See `MODEL_USAGE.md` for advanced model management
- See `TECHNICAL.md` for architecture details

## 🎯 Quick Reference

**Most Common Commands:**
```bash
# Train quickly
python ml_training/escape_cage_trainer.py --trainer fast

# Train properly  
python ml_training/escape_cage_trainer.py --trainer standard

# Test model
python ml_training/testing_utils.py test --model trained_escape_ai

# Manage models
python ml_training/model_utils.py interactive
```

Ready to train your escape cage AI! 🤖🏃‍♂️
