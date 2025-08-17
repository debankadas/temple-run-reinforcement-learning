# Temple Run Data Collector

This tool helps you collect training data for the Temple Run AI classifier by capturing screenshots during gameplay and labeling them as "alive" or "dead" states.

## 🎯 Purpose

The classifier needs labeled examples to distinguish between:
- **Alive states**: When the character is running normally
- **Dead states**: When the character has crashed/died

## 📋 Prerequisites

```bash
pip install pynput pillow opencv-python numpy
```

## 🚀 How to Use

### 1. Setup Your Game
- Launch Temple Run on your device/emulator
- Ensure ADB is connected (same setup as training)
- Position the game window for optimal capture

### 2. Run the Data Collector

```bash
cd data_collector
python templerun_data_collector.py
```

### 3. Collect Data During Gameplay

**Controls:**
- **Press `A`**: Capture current frame as "alive" (character running)
- **Press `D`**: Capture current frame as "dead" (character crashed)
- **Press `ESC`**: Stop collecting and exit

**Best Practices:**
- Play the game manually while running the collector
- Press `A` frequently during normal running (every 1-2 seconds)
- Press `D` immediately when you see death/crash screens
- Collect diverse scenarios: different environments, lighting, obstacles

### 4. Data Collection Strategy

**For Alive States (`A` key):**
- Capture during normal running
- Include different environments (forest, cliffs, mines)
- Capture various lighting conditions
- Include different character positions (center, left, right lanes)
- Capture while jumping, sliding, turning

**For Dead States (`D` key):**
- Capture the exact moment of death/crash
- Include different types of deaths (hitting trees, falling, obstacles)
- Capture the death animation frames
- Include game over screens

## 📁 Output Structure

Data is saved in:
```
data_collector/
├── templerun_dataset/
│   ├── alive/
│   │   ├── alive_20250816_120530_123456.png
│   │   └── ...
│   ├── dead/
│   │   ├── dead_20250816_120545_789012.png
│   │   └── ...
│   └── dataset_info.json
└── templerun_data_collector.py
```

## 🎯 Collection Goals

**Minimum Recommended:**
- **Alive**: 200+ images (more is better)
- **Dead**: 100+ images
- **Balance**: Try to maintain roughly 2:1 ratio (alive:dead)

**Quality Tips:**
- Capture clear, unblurred frames
- Avoid duplicate/similar frames
- Include edge cases and challenging scenarios
- Capture from different game sessions

## 🔄 Improving the Model

After collecting new data:

1. **Retrain the Classifier:**
   ```bash
   cd temple_run
   python train.py --retrain-classifier
   ```

2. **Test the Updated Model:**
   - Run a few training episodes
   - Monitor classification accuracy in logs
   - Look for improved death detection

3. **Iterative Improvement:**
   - If model struggles with specific scenarios, collect more data for those cases
   - Pay attention to misclassified frames during training
   - Add challenging examples to improve robustness

## 📊 Dataset Statistics

The current dataset contains:
- **Alive images**: ~150 samples
- **Dead images**: ~80 samples
- **Total**: ~230 training examples

**To check your dataset:**
```bash
python -c "
import os
alive_count = len([f for f in os.listdir('templerun_dataset/alive') if f.endswith('.png')])
dead_count = len([f for f in os.listdir('templerun_dataset/dead') if f.endswith('.png')])
print(f'Alive: {alive_count}, Dead: {dead_count}, Total: {alive_count + dead_count}')
"
```

## 🛠️ Troubleshooting

**Issue: No frames being captured**
- Check ADB connection: `adb devices`
- Ensure Temple Run is in foreground
- Try restarting the data collector

**Issue: Wrong captures**
- Delete incorrect images from the dataset folders
- Re-run collection for better examples

**Issue: Poor classifier performance**
- Collect more diverse examples
- Ensure good lighting in captures
- Add more edge cases to dataset

## 🔄 Integration with Training

The collected data automatically integrates with the main training pipeline:
- Classifier training uses data from `data_collector/templerun_dataset/`
- No need to move files manually
- Just retrain the classifier after adding new data

## 📈 Advanced Tips

1. **Balanced Collection**: Aim for similar numbers in both categories
2. **Temporal Diversity**: Collect from different game sessions
3. **Environmental Variety**: Include all game environments
4. **Action Diversity**: Capture during different character actions
5. **Quality Control**: Review collected images periodically

Happy data collecting! 🎮📸