# Using Trained AI Models - Complete Guide

Learn how to work with your trained AI models using the current system. This guide covers testing, management, and troubleshooting for the AI Escape Cage project.

## 🚀 **Quick Start - Testing Your Models**

### **What Gets Saved During Training**
- **Model File**: `models/trained_escape_ai.zip` (complete AI neural network)
- **Metadata**: Performance statistics and training info (in `models/.model_metadata.json`)
- **Logs**: Training progress logs in `logs/` directory

### **Test Your Models**
```bash
# Test all your trained models
python test_system.py --component models

# Test with detailed output
python test_system.py --component models --verbose

# Quick system check including models
python test_system.py --quick
```

## 📊 **Model Management Commands**

### **List Available Models**
```bash
# Simple list with basic info
python ml_training/model_utils.py list
```

**Example Output:**
```
AI Model Collection
==================================================
Name                     Size      Modified         Success%
-----------------------------------------------------------------
trained_escape_ai        245.3K    01/15 14:30     N/A
fast_trained_ai          247.1K    01/16 09:15     N/A
my_custom_model          250.8K    01/17 11:45     N/A

Total models: 3
```

### **Model Operations**
```bash
# Create backup of important model
python ml_training/model_utils.py backup --model trained_escape_ai

# Compare two models
python ml_training/model_utils.py compare --model trained_escape_ai --model2 fast_trained_ai

# Delete unwanted model (with safety prompts)
python ml_training/model_utils.py delete --model old_model
```

## 🔄 **Continue Training Existing Models**

### **Basic Model Improvement**
```bash
# Continue training with more steps
python ml_training/escape_cage_trainer.py --trainer continue --model trained_escape_ai

# Continue with enhanced reward environment
python ml_training/escape_cage_trainer.py --trainer continue --environment fast --model trained_escape_ai
```

### **Advanced Continue Training Options**
```bash
# Continue for specific number of steps
python ml_training/escape_cage_trainer.py --trainer continue --model trained_escape_ai --continue-steps 50000

# Fine-tune with conservative settings
python ml_training/escape_cage_trainer.py --trainer continue --model trained_escape_ai --fine-tune --fine-steps 20000

# Continue without creating backup (faster startup)
python ml_training/escape_cage_trainer.py --trainer continue --model trained_escape_ai --no-backup
```

## 🎯 **Model Usage Scenarios**

### **Scenario 1: Daily Development**
```bash
# 1. Quick prototype training
python ml_training/escape_cage_trainer.py --trainer fast --model prototype_v1

# 2. Test the prototype
python test_system.py --component models --verbose

# 3. Improve promising models
python ml_training/escape_cage_trainer.py --trainer continue --environment fast --model prototype_v1

# 4. Backup successful models
python ml_training/model_utils.py backup --model prototype_v1
```

### **Scenario 2: Research & Experimentation**
```bash
# 1. Create baseline model
python ml_training/escape_cage_trainer.py --trainer standard --model baseline_v1

# 2. Create variants with different environments
python ml_training/escape_cage_trainer.py --trainer standard --environment fast --model baseline_v2
python ml_training/escape_cage_trainer.py --trainer fast --environment simple --model baseline_v3

# 3. Compare all variants
python ml_training/model_utils.py compare --model baseline_v1 --model2 baseline_v2
python ml_training/model_utils.py compare --model baseline_v2 --model2 baseline_v3

# 4. Test the best performer
python test_system.py --component models --verbose
```

### **Scenario 3: Production Deployment**
```bash
# 1. Train high-quality model
python ml_training/escape_cage_trainer.py --trainer standard --environment fast --model production_v1

# 2. Validate thoroughly
python test_system.py --component models --verbose

# 3. Create backup before deployment
python ml_training/model_utils.py backup --model production_v1

# 4. Test system integration
python test_system.py --verbose
```

## 📁 **File Organization**

### **Current Model Structure**
```
models/
├── trained_escape_ai.zip              # Default standard trainer model
├── fast_trained_ai.zip                # Default fast trainer model
├── my_custom_model.zip                # Custom named models
├── .model_metadata.json               # Metadata for all models
└── (your custom models...)

model_backups/
├── trained_escape_ai_backup_20240115_143022.zip
├── production_v1_backup_20240116_091530.zip
└── (timestamped backups...)
```

### **Best Practices**
- **Descriptive Names**: Use `escape_ai_v2_optimized` instead of `model1`
- **Version Control**: Create backups before major changes
- **Environment Tracking**: Note which environment was used for training
- **Performance Records**: Document success rates and training times

## 🛠️ **Troubleshooting**

### **Model Not Found Errors**
```bash
# Check what models exist
python ml_training/model_utils.py list

# Check models directory directly
ls models/

# Verify model file exists
ls models/your_model_name.zip
```

### **Model Testing Issues**
```bash
# Test the model system specifically
python test_system.py --component models

# Check Unity connection for testing
python test_system.py --component unity

# Full system health check
python test_system.py --quick
```

### **Continue Training Problems**
```bash
# Verify model exists before continuing
python ml_training/model_utils.py list

# Check system health
python test_system.py --quick

# Try with debug environment to see what's happening
python ml_training/escape_cage_trainer.py --trainer continue --environment debug --model your_model
```

### **Performance Issues**
```bash
# Check if Unity is running and connected
python test_system.py --component unity

# Verify model file isn't corrupted
python ml_training/model_utils.py compare --model your_model --model2 known_good_model

# Test with different environment
python ml_training/escape_cage_trainer.py --trainer continue --environment simple --model your_model
```

## 📈 **Performance Guidelines**

### **Training Success Indicators**
- **90%+ Success Rate**: Excellent, ready for production
- **80-89% Success Rate**: Good, suitable for most applications  
- **70-79% Success Rate**: Adequate, consider additional training
- **60-69% Success Rate**: Fair, needs improvement
- **<60% Success Rate**: Poor, requires significant retraining

### **Efficiency Metrics**
- **<30 steps average**: Excellent efficiency
- **30-50 steps**: Good performance
- **50-80 steps**: Acceptable, room for improvement
- **>80 steps**: Inefficient, needs optimization

### **Training Time Guidelines**
- **Fast Trainer**: 5-10 minutes, 70-85% success rate
- **Standard Trainer**: 15-30 minutes, 85-95% success rate
- **Continue Trainer**: Variable, builds on existing performance

## 🔍 **Advanced Model Analysis**

### **Model Comparison**
```bash
# Compare two models side by side
python ml_training/model_utils.py compare --model model1 --model2 model2
```

**Example Output:**
```
Comparing 'model1' vs 'model2'
================================================================
Metric          | Model 1              | Model 2
----------------------------------------------------------------
Size            | 245.3 KB             | 247.1 KB
Modified        | 2024-01-15 14:30     | 2024-01-16 09:15
Success Rate    | N/A                  | N/A
Training Steps  | Unknown              | Unknown

RECOMMENDATIONS:
→ Both models are similar in size
→ Model 2 is more recent
→ Use test_system.py --component models to evaluate performance
```

### **Model Backup Strategy**
```bash
# Create backup before risky operations
python ml_training/model_utils.py backup --model important_model

# Backup before continuing training
python ml_training/escape_cage_trainer.py --trainer continue --model important_model
# (backup is created automatically unless --no-backup is used)
```

## 🎯 **Success Tips**

1. **Always test first**: Use `python test_system.py --component models` before deploying
2. **Use descriptive names**: `escape_ai_production_v3` vs `model1`
3. **Create backups**: Before major changes or continue training
4. **Document performance**: Keep notes on which models work best
5. **Test different environments**: Try `--environment fast` for better performance
6. **Monitor training**: Use `--environment debug` when training isn't working
7. **Validate system health**: Run `python test_system.py --quick` regularly

## 📚 **Command Reference**

### **Testing Commands**
```bash
python test_system.py --component models                    # Test all models
python test_system.py --component models --verbose          # Detailed testing
python test_system.py --quick                              # Quick system check
```

### **Model Management**
```bash
python ml_training/model_utils.py list                     # List all models
python ml_training/model_utils.py backup --model NAME      # Backup model
python ml_training/model_utils.py delete --model NAME      # Delete model
python ml_training/model_utils.py compare --model A --model2 B  # Compare models
```

### **Continue Training**
```bash
python ml_training/escape_cage_trainer.py --trainer continue --model NAME
python ml_training/escape_cage_trainer.py --trainer continue --environment fast --model NAME
python ml_training/escape_cage_trainer.py --trainer continue --model NAME --fine-tune
```

### **Help Commands**
```bash
python ml_training/escape_cage_trainer.py --help           # Training help
python ml_training/model_utils.py --help                   # Model management help
python test_system.py --help                               # Testing help
```

## 🔗 **Related Documentation**

- **README.md**: Quick start and basic commands
- **SETUP.md**: Installation and dependencies
- **TESTING_GUIDE.md**: Comprehensive testing system
- **TECHNICAL.md**: System architecture details

Your AI models are ready to use! Train, test, and improve them with confidence. 🤖✨ 