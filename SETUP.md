# 🔧 Complete Setup Guide - AI Escape Cage

> **Step-by-step instructions to get your AI learning system running**

---

## 📋 **Prerequisites Checklist**

Before starting, make sure you have:
- [ ] **Python 3.8+** installed ([Download here](https://python.org))
- [ ] **Git** for cloning the repository ([Download here](https://git-scm.com))
- [ ] **4GB+ RAM** available
- [ ] **Stable internet** for downloading Unity and packages

---

## 🐍 **Part 1: Python Environment Setup**

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/escape-cage-ai.git
cd escape-cage-ai
```

### **Step 2: Create Virtual Environment**
```bash
# Create virtual environment
python -m venv escape_cage_env

# Activate it
# On macOS/Linux:
source escape_cage_env/bin/activate

# On Windows:
escape_cage_env\Scripts\activate
```

### **Step 3: Install Python Packages**
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python quick_test.py
```

**✅ Expected Output:**
```
🧪 Testing Python Environment...
✅ All imports successful!
✅ System ready for AI training!
```

---

## 🎮 **Part 2: Unity Setup**

### **Step 1: Download Unity**
1. Download **Unity Hub** from [unity3d.com](https://unity3d.com)
2. Install **Unity 2022.3 LTS** (free Personal license)
3. Make sure to include **Visual Studio** integration

### **Step 2: Create New Project**
1. Open Unity Hub
2. Click **"New Project"**
3. Select **"2D Core"** template  
4. Name: **"EscapeCageRL"**
5. Choose location and click **"Create"**

### **Step 3: Setup Game Objects**

#### **Create the Player**
1. **Hierarchy** → Right-click → **"2D Object"** → **"Sprites"** → **"Circle"**
2. Rename to **"Player"**
3. Set Position: **(-7, 0, 0)**
4. Set Scale: **(0.8, 0.8, 1)**
5. Change Color: **White**

**Add Components to Player:**
1. **Add Component** → **"Physics 2D"** → **"Rigidbody 2D"**
2. **Add Component** → **"Physics 2D"** → **"Circle Collider 2D"**
3. In Rigidbody2D: Set **Gravity Scale = 0**

#### **Create the Key**
1. **Hierarchy** → Right-click → **"2D Object"** → **"Sprites"** → **"Circle"**
2. Rename to **"Key"**  
3. Set Position: **(0, 3, 0)**
4. Set Scale: **(0.5, 0.5, 1)**
5. Change Color: **Yellow**

**Add Components to Key:**
1. **Add Component** → **"Physics 2D"** → **"Circle Collider 2D"**
2. In Circle Collider 2D: Check **"Is Trigger"**

#### **Create the Exit**
1. **Hierarchy** → Right-click → **"2D Object"** → **"Sprites"** → **"Square"**
2. Rename to **"Exit"**
3. Set Position: **(7, 0, 0)**  
4. Set Scale: **(1.2, 1.2, 1)**
5. Change Color: **Green**

**Add Components to Exit:**
1. **Add Component** → **"Physics 2D"** → **"Box Collider 2D"**
2. In Box Collider 2D: Check **"Is Trigger"**

### **Step 4: Add Communication Script**

1. Select **"Main Camera"** in Hierarchy
2. **Add Component** → **"New Script"**
3. Name: **"GameController"**
4. Double-click script to open in editor
5. Replace ALL content with this code:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Net;
using System.Text;
using System.Threading;

public class GameController : MonoBehaviour
{
    // Game Objects
    public GameObject player;
    public GameObject key;
    public GameObject exit;
    
    // Game State
    private bool hasKey = false;
    private bool gameWon = false;
    private Vector3 initialPlayerPos;
    private Vector3 initialKeyPos;
    
    // Communication
    private TcpListener tcpListener;
    private Thread tcpListenerThread;
    private TcpClient connectedTcpClient;
    private bool serverStarted = false;
    
    void Start()
    {
        // Find game objects
        player = GameObject.Find("Player");
        key = GameObject.Find("Key");
        exit = GameObject.Find("Exit");
        
        // Store initial positions
        initialPlayerPos = player.transform.position;
        initialKeyPos = key.transform.position;
        
        // Start server
        tcpListenerThread = new Thread(new ThreadStart(ListenForTcpClients));
        tcpListenerThread.IsBackground = true;
        tcpListenerThread.Start();
        
        Debug.Log("🚀 Unity: Waiting for Python AI connection...");
    }
    
    void ListenForTcpClients()
    {
        try
        {
            tcpListener = new TcpListener(IPAddress.Any, 9999);
            tcpListener.Start();
            Debug.Log("✅ Unity: Server started on port 9999");
            serverStarted = true;
            
            using (connectedTcpClient = tcpListener.AcceptTcpClient())
            {
                Debug.Log("🤖 Unity: Python AI connected!");
                
                using (NetworkStream stream = connectedTcpClient.GetStream())
                {
                    while (true)
                    {
                        try
                        {
                            // Send observation
                            string observation = GetObservation();
                            byte[] data = Encoding.UTF8.GetBytes(observation);
                            stream.Write(data, 0, data.Length);
                            
                            // Receive action
                            byte[] bytes = new byte[1024];
                            int bytesRead = stream.Read(bytes, 0, bytes.Length);
                            string response = Encoding.UTF8.GetString(bytes, 0, bytesRead);
                            
                            if (int.TryParse(response.Trim(), out int action))
                            {
                                ProcessAction(action);
                            }
                            
                            Thread.Sleep(100); // 10 FPS
                        }
                        catch (System.Exception e)
                        {
                            Debug.Log("Connection lost: " + e.Message);
                            break;
                        }
                    }
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.Log("Server error: " + e.Message);
        }
    }
    
    string GetObservation()
    {
        Vector3 playerPos = player.transform.position;
        Vector3 keyPos = key.transform.position;
        Vector3 exitPos = exit.transform.position;
        
        string obs = $"observation|{playerPos.x:F1}|{playerPos.y:F1}|{hasKey}|{keyPos.x:F1}|{keyPos.y:F1}|{exitPos.x:F1}|{exitPos.y:F1}|{gameWon}";
        return obs;
    }
    
    void ProcessAction(int action)
    {
        if (gameWon) return;
        
        Vector3 movement = Vector3.zero;
        float speed = 3.0f;
        
        switch (action)
        {
            case 0: movement = Vector3.up; break;      // Up
            case 1: movement = Vector3.down; break;    // Down  
            case 2: movement = Vector3.left; break;    // Left
            case 3: movement = Vector3.right; break;   // Right
        }
        
        // Apply movement
        player.transform.position += movement * speed * Time.fixedDeltaTime;
        
        // Keep player in bounds
        Vector3 pos = player.transform.position;
        pos.x = Mathf.Clamp(pos.x, -8f, 8f);
        pos.y = Mathf.Clamp(pos.y, -4f, 4f);
        player.transform.position = pos;
        
        // Check collisions
        CheckCollisions();
    }
    
    void CheckCollisions()
    {
        // Check key pickup
        if (!hasKey && Vector3.Distance(player.transform.position, key.transform.position) < 0.8f)
        {
            hasKey = true;
            key.SetActive(false);
            Debug.Log("🗝️ Unity: Key picked up!");
        }
        
        // Check exit (only if has key)
        if (hasKey && Vector3.Distance(player.transform.position, exit.transform.position) < 1.0f)
        {
            gameWon = true;
            Debug.Log("🎉 Unity: Level completed!");
            
            // Reset after short delay
            Invoke("ResetGame", 2.0f);
        }
    }
    
    void ResetGame()
    {
        hasKey = false;
        gameWon = false;
        player.transform.position = initialPlayerPos;
        key.transform.position = initialKeyPos;
        key.SetActive(true);
        Debug.Log("🔄 Unity: Game reset for next episode");
    }
    
    void OnApplicationQuit()
    {
        if (tcpListenerThread != null)
        {
            tcpListenerThread.Abort();
        }
        if (tcpListener != null)
        {
            tcpListener.Stop();
        }
    }
}
```

6. **Save** the script (Ctrl+S)
7. **Return to Unity** (it will compile automatically)

### **Step 5: Configure the Script**
1. Select **"Main Camera"** in Hierarchy
2. In **Inspector**, find **"Game Controller"** component
3. Drag **Player** from Hierarchy to **"Player"** field
4. Drag **Key** from Hierarchy to **"Key"** field  
5. Drag **Exit** from Hierarchy to **"Exit"** field

---

## 🚀 **Part 3: Testing & Training**

### **Step 1: Test Connection**
```bash
# Activate Python environment (if not already)
source escape_cage_env/bin/activate  # or escape_cage_env\Scripts\activate on Windows

# Start Python AI
python ml_training/escape_cage_trainer.py
```

**✅ Expected Output:**
```
🚀 Starting AI system...
⏳ Waiting for Unity connection...
```

### **Step 2: Start Unity**
1. In Unity, click **"Play"** button (▶️)
2. Check **Console** tab for messages

**✅ Expected Console Output:**
```
🚀 Unity: Waiting for Python AI connection...
✅ Unity: Server started on port 9999  
🤖 Unity: Python AI connected!
```

### **Step 3: Watch Training**
You should see:
- **Python Terminal**: Training progress, rewards, episode numbers
- **Unity Game View**: Player moving around (starts random, gets smarter)
- **Unity Console**: Connection status, key pickups, escapes

### **Training Timeline**
- **0-5 minutes**: Random exploration, AI learning environment
- **5-15 minutes**: Pattern recognition, moving toward key more often
- **15+ minutes**: Strategic behavior, consistent key → exit strategy

---

## 🔧 **Troubleshooting**

### **Python Issues**

**❌ "Module not found" errors**
```bash
# Reinstall packages
pip uninstall -y tensorflow keras stable-baselines3 gymnasium
pip install -r requirements.txt
```

**❌ "Port already in use"**
```bash
# Kill existing Python processes
pkill -f python
# Then restart training
```

### **Unity Issues**

**❌ "NullReferenceException"**
- Make sure all GameObjects are assigned in GameController component
- Check that Player, Key, Exit objects exist in Hierarchy

**❌ "Compilation errors"**  
- Close Unity completely
- Delete `Library` folder in Unity project
- Reopen Unity (it will rebuild)

**❌ "No connection to Python"**
- Check Windows Firewall (allow Unity and Python)
- Try disabling antivirus temporarily
- Restart both Unity and Python

### **Performance Issues**

**❌ "Training is very slow"**
```bash
# Reduce training frequency in escape_cage_trainer.py
# Change: time.sleep(0.1) to time.sleep(0.2)
```

**❌ "Unity freezing"**
- Reduce movement speed in Unity script  
- Change: `float speed = 3.0f;` to `float speed = 2.0f;`

---

## 🎯 **Verification Checklist**

Before reporting issues, verify:

### **Python Environment**
- [ ] Virtual environment activated
- [ ] All packages installed without errors
- [ ] `quick_test.py` runs successfully
- [ ] No firewall blocking Python

### **Unity Setup**
- [ ] Unity 2022.3 LTS installed
- [ ] Project created with 2D template
- [ ] All game objects created and positioned correctly
- [ ] GameController script added to Main Camera
- [ ] All object references assigned in inspector
- [ ] No compilation errors in Console

### **Training Process**
- [ ] Python shows "Waiting for Unity connection"
- [ ] Unity Play button pressed
- [ ] Unity Console shows "Python AI connected!"
- [ ] Player object moving in Unity Game View
- [ ] Training progress visible in Python terminal

---

## 🎉 **Success Indicators**

**🚀 You're ready when you see:**

**Python Terminal:**
```
✅ Unity connected! Training will start!
🧠 Creating AI agent...
🏋️ Starting AI training...
Episode 1: reward=8.4, length=200
Episode 2: reward=12.1, length=180
🗝️ AI found the key! +10 points
```

**Unity Console:**  
```
🤖 Unity: Python AI connected!
🗝️ Unity: Key picked up!
🎉 Unity: Level completed!
🔄 Unity: Game reset for next episode
```

**Unity Game View:**
- White circle (Player) moving around
- Yellow circle (Key) disappearing when touched
- Green square (Exit) that completes the level

---

## 🆘 **Getting Help**

If you're still stuck:

1. **Check the Issues**: [GitHub Issues](https://github.com/yourusername/escape-cage-ai/issues)
2. **Start a Discussion**: [GitHub Discussions](https://github.com/yourusername/escape-cage-ai/discussions)  
3. **Include in your help request**:
   - Operating system (Windows/Mac/Linux)
   - Python version (`python --version`)
   - Unity version
   - Complete error messages
   - Screenshots of any error dialogs

---

<div align="center">

**🎮 Ready to watch your AI learn? Let's go!**

[⬅️ Back to Main README](README.md) • [📊 View Results](RESULTS.md)

</div> 