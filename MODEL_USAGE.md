# Using Saved AI Models - Complete Guide

Learn how to load, test, and manage your trained AI models effectively. This guide covers everything from basic model usage to advanced model management techniques.

## **Quick Start - Using Your Trained Model**

### **What Gets Saved Automatically**
- **Model File**: `models/trained_escape_ai.zip` (complete AI brain)
- **Metadata**: Training statistics, performance metrics, creation date
- **Logs**: Detailed training progress and hyperparameters

## **Testing Your Saved Models**

### **Basic Testing**
```bash
python test_trained_ai.py
```

**What You'll See:**
```
Loading saved AI: trained_escape_ai
AI model 'trained_escape_ai' loaded successfully!
Watch the trained AI play! (Press Ctrl+C to stop)

Episode 1
AI grabbed the key! Strategic thinking!
AI ESCAPED! Smart AI!
Success! Reward: 109.2, Steps: 45

Episode 2
...
```

### **Advanced Testing Options**
```bash
# Test for 50 episodes with detailed analysis
python test_trained_ai.py --episodes 50 --detailed

# Test specific model
python test_trained_ai.py --model your_model_name

# Compare multiple models
python test_trained_ai.py --compare model1 model2 model3
```

## **Continue Training From Saved Models**

### **Basic Continuation**
```bash
python continue_training.py --model trained_escape_ai
```

### **Advanced Continuation**
```bash
# Continue with different learning rate
python continue_training.py --model trained_escape_ai --learning_rate 0.0001

# Continue for specific number of steps
python continue_training.py --model trained_escape_ai --steps 50000

# Fine-tune with different reward structure
python continue_training.py --model trained_escape_ai --environment fast
```

**When to Continue Training:**
- Model performance plateaued during initial training
- Want to fine-tune for specific scenarios
- Improve success rate from 80% to 90%+
- Adapt model to modified environment

**Results:**
- Faster convergence (builds on existing knowledge)
- Often achieves higher final performance
- Saves training time compared to starting from scratch

## **Model Management System**

### **Organize Your Models**
```bash
# Interactive model management
python model_manager.py interactive

# List all models with details
python model_manager.py list --detailed

# Organize loose files
python model_manager.py organize
```

### **Compare Models**
```bash
python model_manager.py compare --model1 trained_escape_ai --model2 trained_escape_ai_improved
```

**Example Output:**
```
Comparing 'trained_escape_ai' vs 'trained_escape_ai_improved'
================================================================
Metric          | Model 1              | Model 2
----------------------------------------------------------------
Size            | 245.3 KB             | 247.1 KB
Success Rate    | 78%                  | 89%
Steps           | 50000                | 75000
Modified        | 2024-01-15 14:30     | 2024-01-16 09:15

RECOMMENDATIONS:
→ trained_escape_ai_improved performs significantly better (89% vs 78%)
→ Recommend using trained_escape_ai_improved for production
```

## **Model Usage Scenarios**

### **Scenario 1: Daily Development**
1. **Quick Testing**: Use fast_training.py for 5-minute models
2. **Iteration**: Test ideas quickly, keep successful approaches
3. **Refinement**: Use continue_training.py to improve promising models
4. **Organization**: Use model_manager.py to keep workspace clean

### **Scenario 2: Research & Experimentation**
1. **Baseline**: Train standard model for comparison
2. **Variants**: Create multiple models with different parameters
3. **Analysis**: Use detailed testing to compare approaches
4. **Documentation**: Add metadata to track experiment parameters

### **Scenario 3: Demo Your AI**
1. **Best Model**: Identify highest-performing model
2. **Testing**: Verify consistent performance across multiple runs
3. **Backup**: Create backup before important demonstrations
4. **Showcase**: Use test_trained_ai.py for live demonstrations

## **File Organization**

### **Model Files Structure**
```
models/
├── trained_escape_ai.zip              # Your main model
├── trained_escape_ai_improved.zip     # Enhanced version
├── experimental_model_v1.zip          # Experimental variants
├── fast_prototype.zip                 # Quick prototypes
└── .model_metadata.json               # Metadata database
```

### **Backup System**
```
model_backups/
├── trained_escape_ai_backup_20240115_143022.zip
├── trained_escape_ai_backup_20240116_091530.zip
└── critical_model_backup_20240117_154500.zip
```

### **Best Practices**
- Use descriptive model names: `escape_ai_v2_optimized` vs `model1`
- Create backups before risky operations
- Keep metadata updated with training notes
- Archive old experimental models periodically
- Document successful hyperparameter combinations

## **Troubleshooting**

### **"Model not found" Error**
1. Check if file exists: `ls models/`
2. Verify exact filename (case-sensitive)
3. Use model_manager.py list to see available models
4. Check if model is in current directory vs models/ directory

### **Model Loads But Performs Poorly**
- Model may have been interrupted during training
- Try continue_training.py to improve performance
- Check if model was trained on same environment version
- Verify Unity environment is properly configured

### **Connection Issues**
- Ensure Unity is running and escape cage scene is loaded
- Check firewall settings for port 9999
- Restart both Unity and Python if connection fails
- Verify no other training sessions are running

### **Success Rate Guidelines**
- **90%+**: Excellent, ready for production/demo
- **80-89%**: Good, suitable for most applications
- **70-79%**: Adequate, consider additional training
- **60-69%**: Fair, needs improvement or different approach
- **<60%**: Poor, requires significant retraining or debugging

### **Speed Guidelines**
- **<30 steps average**: Excellent efficiency
- **30-50 steps**: Good performance
- **50-80 steps**: Acceptable but improvable
- **>80 steps**: Inefficient, needs optimization

## **Advanced Usage**

### **Model Versioning**
```bash
# Create version series
python escape_cage_trainer.py --save_name "escape_ai_v1"
python continue_training.py --model escape_ai_v1 --save_name "escape_ai_v2"
python continue_training.py --model escape_ai_v2 --save_name "escape_ai_v3"

# Compare version progression
python model_manager.py compare --model1 escape_ai_v1 --model2 escape_ai_v3
```

### **A/B Testing**
```bash
# Test multiple models simultaneously
python test_trained_ai.py --batch_test models/escape_ai_v1.zip models/escape_ai_v2.zip models/experimental.zip

# Statistical comparison
python analytics_utils.py --compare_models escape_ai_v1 escape_ai_v2 --episodes 100
```

### **Performance Profiling**
```bash
# Detailed performance analysis
python test_trained_ai.py --model your_model --episodes 100 --analyze

# Export results for external analysis
python analytics_utils.py --model your_model --export_csv results.csv
```

### **Custom Testing Scenarios**
```python
# Create custom test script
from test_trained_ai import load_model, test_model

model = load_model("your_model")
results = test_model(model, episodes=50, max_steps=200)
print(f"Custom test results: {results}")
```

## **Success! You Now Have**

- **Reusable AI Models** - No more training from scratch
- **Performance Testing** - Verify your AI works consistently  
- **Continuous Improvement** - Make your AI even better
- **Professional Organization** - Manage multiple model versions
- **Easy Demonstrations** - Show off your AI to others

**Your AI went from random movement to strategic problem-solving, and now you can use that intelligence over and over again!**

---

**Need More Help?**
- Check SETUP.md for installation issues
- See TECHNICAL.md for advanced configuration
- View RESULTS.md for performance expectations

**Ready to use your trained AI? Start testing!**

[Back to Main README](README.md) • [Setup Guide](SETUP.md) • [View Results](RESULTS.md) 