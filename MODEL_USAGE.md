# 🎮 Using Saved AI Models - Complete Guide

> **How to save, load, test, and manage your trained AI models**

---

## 🎯 **Quick Start - Using Your Trained Model**

After your AI finishes training, you'll have saved models that you can reuse without starting from scratch!

### **📋 What Gets Saved Automatically**
When training completes or you stop it:
- **`trained_escape_ai.zip`** - Your complete trained model
- **`escape_ai_partial.zip`** - Partial model if you stopped early

---

## 🧪 **Testing Your Saved Models**

### **👀 Watch Your AI Play (No Training)**
```bash
# Test your main trained model
python ml_training/test_trained_ai.py

# Test a specific model
python ml_training/test_trained_ai.py --model escape_ai_partial

# List all available models
python ml_training/test_trained_ai.py --list
```

**✅ What You'll See:**
```
🤖 Loading saved AI: trained_escape_ai
✅ AI model 'trained_escape_ai' loaded successfully!
🎮 Watch the trained AI play! (Press Ctrl+C to stop)

🎯 Episode 1
🗝️ AI grabbed the key! Strategic thinking!
🎉 AI ESCAPED! Smart AI!
✅ Success! Reward: 109.2, Steps: 45
📊 Success rate so far: 100.0% (1/1)
```

### **📊 Performance Testing**
Your trained AI should show:
- **Success Rate**: 70-90%+ (much better than random 0%)
- **Efficient Movement**: Direct path to key, then exit
- **Strategic Behavior**: Consistent "key first" strategy

---

## 🔄 **Continue Training From Saved Models**

### **📈 Train Your Model Further**
```bash
# Continue training for 25,000 more steps
python ml_training/continue_training.py

# Continue training a specific model
python ml_training/continue_training.py --model escape_ai_partial

# Train for different amount of steps
python ml_training/continue_training.py --steps 50000

# Fine-tune with conservative settings
python ml_training/continue_training.py --finetune
```

**🎯 When to Continue Training:**
- **Low Success Rate**: If your AI is below 70% success
- **Inconsistent Behavior**: Sometimes works, sometimes doesn't
- **Want Even Better Performance**: Push from 80% to 90%+
- **Test Different Settings**: Fine-tuning for stability

**✅ Results:**
- **`trained_escape_ai_improved.zip`** - Your enhanced model
- **Automatic Backup**: Original model preserved

---

## 📦 **Model Management System**

### **🗂️ Organize Your Models**
```bash
# List all your models with details
python ml_training/model_manager.py list

# Organize models into clean directory structure
python ml_training/model_manager.py organize

# Create backup of important model
python ml_training/model_manager.py backup --model trained_escape_ai --notes "Best performing model"
```

### **📊 Compare Model Performance**
```bash
# Compare two models side by side
python ml_training/model_manager.py compare --model trained_escape_ai --model2 trained_escape_ai_improved
```

**Example Output:**
```
⚖️ Comparing 'trained_escape_ai' vs 'trained_escape_ai_improved'
📏 Size:     245.2 KB  |  247.8 KB
📅 Modified: 2024-01-15 14:30  |  2024-01-15 16:45
🎯 Steps:    50000  |  75000
📊 Success:  82.5%  |  91.2%
```

### **📝 Add Model Information**
```bash
# Add performance information to your models
python ml_training/model_manager.py info --model trained_escape_ai --success-rate 85.2 --steps 50000 --notes "First successful training"
```

---

## 🎯 **Model Usage Scenarios**

### **🏆 Scenario 1: Perfect Your Best Model**
```bash
# 1. Test your current model
python ml_training/test_trained_ai.py

# 2. If success rate < 80%, continue training
python ml_training/continue_training.py --steps 25000

# 3. Test the improved version
python ml_training/test_trained_ai.py --model trained_escape_ai_improved
```

### **🔬 Scenario 2: Experiment with Different Settings**
```bash
# 1. Backup your best model
python ml_training/model_manager.py backup --model trained_escape_ai --notes "Baseline model"

# 2. Fine-tune with conservative settings
python ml_training/continue_training.py --finetune

# 3. Compare results
python ml_training/model_manager.py compare --model trained_escape_ai --model2 trained_escape_ai_finetuned
```

### **🎮 Scenario 3: Demo Your AI**
```bash
# Load and watch your best model play continuously
python ml_training/test_trained_ai.py --model trained_escape_ai_improved
```

**Perfect for:**
- Showing friends/colleagues
- Recording demo videos
- Verifying performance
- Understanding AI strategies

---

## 📁 **File Organization**

### **📋 Model Files Structure**
```
NEW-IDEA-main/
├── trained_escape_ai.zip              # Main trained model
├── escape_ai_partial.zip              # Partial training
├── trained_escape_ai_improved.zip     # Enhanced version
├── models/                             # Organized storage
│   ├── trained_escape_ai_backup_20240115_143000.zip
│   ├── experimental_model_v1.zip
│   └── model_metadata.json            # Performance data
└── ml_training/
    ├── escape_cage_trainer.py         # Original training
    ├── test_trained_ai.py             # Testing script  
    ├── continue_training.py           # Continue training
    └── model_manager.py               # Management tools
```

### **🗂️ Best Practices**
1. **Keep Your Best Model**: Always backup before experimenting
2. **Use Descriptive Names**: `model_high_success_rate.zip`
3. **Track Performance**: Record success rates and training steps
4. **Clean Up**: Remove old experimental models periodically

---

## 🔧 **Troubleshooting**

### **❌ "Model not found" Error**
```bash
# Check what models you have
python ml_training/test_trained_ai.py --list

# Or use model manager
python ml_training/model_manager.py list
```

### **❌ Model Loads But Performs Poorly**
- **Check Training Time**: Models need 20+ minutes of good training
- **Try Continued Training**: Add more steps with `continue_training.py`
- **Test Different Models**: If you have multiple, compare them

### **❌ Connection Issues**
- **Unity Must Be Running**: Start Unity and press Play before loading model
- **Same Environment**: Use same Unity scene setup as training
- **Port Conflicts**: Close other Python processes using port 9999

---

## 📊 **Performance Benchmarks**

### **🎯 Success Rate Guidelines**
- **0-20%**: Needs much more training (continue training)
- **20-50%**: Learning but not consistent (continue training)  
- **50-70%**: Good progress (fine-tune or continue)
- **70-85%**: Very good performance (ready for demos)
- **85%+**: Excellent performance (backup this model!)

### **⚡ Speed Guidelines**
- **Steps to Success**: 50-150 steps (efficient movement)
- **Episode Length**: 2-5 seconds per attempt
- **Key Discovery**: Should find key within 20-30 steps

---

## 🚀 **Advanced Usage**

### **🔄 Model Versioning**
```bash
# Create versions of your models
python ml_training/model_manager.py backup --model trained_escape_ai --notes "v1.0 - initial success"

# Continue training
python ml_training/continue_training.py --steps 25000

# Save as new version
python ml_training/model_manager.py backup --model trained_escape_ai_improved --notes "v1.1 - enhanced performance"
```

### **🎯 A/B Testing**
```bash
# Test multiple models and compare
python ml_training/test_trained_ai.py --model model_a
# Record success rate

python ml_training/test_trained_ai.py --model model_b  
# Record success rate

python ml_training/model_manager.py compare --model model_a --model2 model_b
```

### **🧹 Maintenance**
```bash
# Clean up old experimental models (keeps main models)
python ml_training/model_manager.py cleanup --days 7

# Organize scattered model files
python ml_training/model_manager.py organize
```

---

## 🎉 **Success! You Now Have**

✅ **Reusable AI Models** - No more training from scratch  
✅ **Performance Testing** - Verify your AI works consistently  
✅ **Continuous Improvement** - Make your AI even better  
✅ **Professional Organization** - Manage multiple model versions  
✅ **Easy Demonstrations** - Show off your AI to others  

**Your AI went from random movement to strategic problem-solving, and now you can use that intelligence over and over again!** 🧠🎮

---

<div align="center">

**🎮 Ready to use your trained AI? Start testing!**

[⬅️ Back to Main README](README.md) • [🔧 Setup Guide](SETUP.md) • [📊 View Results](RESULTS.md)

</div> 