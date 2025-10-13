# Project Dora

**An Interactive Neural Network Learning Experience**

**Version 1.0** | Author: Zeyphrus @ [Inchworm Games](https://github.com/Andraszao/project-dora-ml)

Project Dora is a hands-on machine learning project that lets you draw digits and watch a neural network learn in real-time. Built in Godot Engine, it demonstrates fundamental deep learning concepts through immediate visual feedback and interactive training.

## What is This?

Dora is a digit recognition system that learns from the MNIST handwritten digit dataset. But unlike typical ML projects where training happens in a black box, Dora makes the learning process visible and interactive:

- **Draw a digit** → Dora predicts what it is
- **Correct Dora when wrong** → Watch the neural network update in real-time
- **See the math** → Console shows exact weight changes during learning
- **Track progress** → Progressive training targets push for ever-higher accuracy

This project is designed for learning. Every function is documented, the code prioritizes clarity over cleverness, and the architecture is intentionally straightforward.

## Key Features

### Interactive Learning
- Real-time feedback system where you can correct predictions
- Human-in-the-loop learning updates weights immediately based on user input
- Visual proof that learning is occurring (shows weight deltas)

### Progressive Training System ⭐
Dora doesn't just train - she gets better at learning as she succeeds!
- **Progressive Targets:** Accuracy goals that increase as she achieves them (80% → 84% → 88%...)
- **Achievement Rewards:** Learning rate boosts when targets are hit (momentum from success!)
- **Escalation Bonuses:** Bigger boosts when advancing to new difficulty levels
- **Milestone Saving:** Automatically saves weights at major achievements

This creates a "success breeds success" training dynamic - the better Dora does, the more confident (higher learning rate) she becomes!

### Robust Data Augmentation
Dora doesn't just memorize perfect digits - she learns to handle messy real-world input!
- **Rotation:** Random -15° to +15° tilts (handles rotated drawings)
- **Scaling:** Random 0.85x to 1.15x zoom (handles different sizes)
- **Gaussian Noise:** Light random noise added (handles sketchy strokes)
- **Progressive Noise Training:** Graduated noise levels from light to heavy
- **Partial Occlusion:** Random patches hidden (handles incomplete drawings)

With default settings, 1,000 samples become **~4,000 variations** - same digits in different conditions!

### Configurable Training Pipeline
- 67 export variables for tweaking every aspect of training
- Adam optimizer with momentum for faster convergence
- Early stopping to prevent overtraining
- Comprehensive training journey logs (JSON records of every session)

### Educational Focus
- Human-centric comments explaining the "why" behind the code
- 784→128→10 architecture (simple enough to understand, complex enough to work)
- Xavier weight initialization for stable gradients
- Backpropagation with gradient clipping
- Cross-entropy loss with softmax output

## What You'll Learn

Working with this project, you'll encounter:

1. **Neural Network Fundamentals**
   - Forward propagation through layers
   - Activation functions (ReLU, Softmax)
   - Weight initialization strategies
   - Backpropagation and gradient descent

2. **Training Techniques**
   - Mini-batch gradient descent
   - Adaptive learning rates (Adam optimizer)
   - Data augmentation for robustness
   - Progressive difficulty scaling
   - Early stopping strategies

3. **Real-World ML Engineering**
   - Training/validation workflows
   - Weight persistence across sessions
   - Online learning vs batch training
   - Performance optimization techniques

## Getting Started

### Prerequisites
- [Godot Engine 4.4+](https://godotengine.org/download/)
- The project already includes the MNIST dataset

### Quick Start
1. Clone this repository
2. Open the project in Godot Engine
3. Run the `Main.tscn` scene (F5)
4. Draw digits on the canvas and click "Test"
5. Give feedback to teach Dora when she's wrong

### First Time Setup
On first run, Dora will:
- Load the MNIST dataset (configurable sample size)
- Apply data augmentation if enabled
- Train for the configured number of epochs (default: 15)
- Save the trained weights for future sessions

**Training time:** ~30-60 minutes with default settings (1,000 samples → 3,900 after augmentation). Grab a coffee! ☕

**Subsequent runs:** Load instantly from saved weights - no waiting!

## Project Structure

```
project-dora-ml/
├── scenes/
│   └── Main.tscn                    # Main UI scene
├── scripts/
│   ├── DrawingCanvas.gd             # Canvas & training orchestration
│   ├── NeuralNetwork.gd             # Core network architecture
│   ├── NeuralNetworkTrainer.gd      # Backpropagation & optimization
│   ├── MNISTLoader.gd               # Dataset loading & augmentation
│   ├── MNISTWeights.gd              # Heuristic weight initialization
│   └── PatternClassifier.gd         # Pattern-based fallback classifier
├── training_data/                   # MNIST dataset (IDX format)
├── dora_trained_weights.dat         # Saved neural network weights
└── dora_training_journey.json       # Training session history
```

## Architecture Details

### Neural Network
- **Input Layer:** 784 neurons (28×28 pixel images)
- **Hidden Layer:** 128 neurons with ReLU activation
- **Output Layer:** 10 neurons with Softmax (digits 0-9)
- **Total Parameters:** ~101,000 trainable weights and biases

### Training Configuration
The default configuration balances learning speed with accuracy:
- **Training Samples:** 1,000 base images (configurable up to 60,000)
- **After Augmentation:** ~4,000 total variations (3× standard + noise training)
- **Epochs:** 15 with progressive targets (80% → 84% → 88%...)
- **Batch Size:** 100 (auto-optimized by CPU cores)
- **Learning Rate:** 0.002 starting, dynamically boosted on success
- **Augmentation Types:** Rotation, scaling, Gaussian noise, progressive noise, occlusion

Typical accuracy: **92-98%** on user drawings after default training.
Training time: **30-60 minutes** on modern CPUs (only happens once, then weights are saved and load instantly).

## Ways to Extend This

Here are some directions you could take this project:

### Beginner-Friendly Extensions
1. **Add more training data visualization** - Show misclassified examples
2. **Create a confusion matrix display** - See which digits get confused
3. **Add sound effects** - Audio feedback for correct/incorrect predictions
4. **Save/load drawing sessions** - Let users export their drawings

### Intermediate Challenges
1. **Implement different architectures** - Try 2 hidden layers, different sizes
2. **Add learning rate scheduling** - Decay learning rate over time
3. **Visualize the learned filters** - Show what features neurons detect
4. **Add a test/validation split** - Proper accuracy measurement
5. **Implement dropout** - Regularization technique to prevent overfitting

### Advanced Projects
1. **Convolutional Neural Network** - CNN architecture for better accuracy
2. **Transfer Learning** - Use a pre-trained model and fine-tune
3. **Uncertainty Estimation** - Monte Carlo dropout for confidence intervals
4. **Adversarial Examples** - Find inputs that fool the network
5. **Network Pruning** - Remove unnecessary weights for efficiency

### Research Directions
1. **Active Learning** - Intelligently choose which samples to learn from
2. **Meta-Learning** - Learn how to learn faster from corrections
3. **Explainable AI** - Visualize which pixels influence predictions
4. **Few-Shot Learning** - Learn new digits from very few examples

## Performance Tips

- **For faster training:** Increase batch size, enable speed optimizations
- **For better accuracy:** Increase training samples, enable data augmentation
- **For experimentation:** Enable verbose logging and prediction details
- **For demos:** Use reset_memory_on_start to train from scratch each time

## Technical Notes

### Why Godot Engine?
While Python/TensorFlow is the standard for ML, Godot offers unique advantages for learning:
- **Immediate visual feedback** - See results instantly, no separate plotting
- **Easy UI creation** - Built-in editor for interfaces
- **Cross-platform** - Runs on Windows, Mac, Linux, even web browsers
- **No external dependencies** - Everything in one project
- **Encourages understanding** - No "magic" libraries, see every step

### Performance Considerations
This is a CPU-only implementation in GDScript. Training 60,000 images takes longer than GPU-accelerated frameworks, but:
- The code is readable and educational
- Training happens once, then weights are saved
- Good enough for real-time prediction
- Shows that ML doesn't require specialized hardware for learning

### Code Philosophy
- **Clarity over performance** - Optimized for reading, not speed
- **Comments explain "why"** - Not just what the code does
- **No magic numbers** - Everything is configurable and documented
- **Avoid abstractions** - See the actual math, not hidden layers

## Contributing

This is a learning project! Improvements welcome, especially:
- Better documentation or explanations
- Additional training visualizations
- Performance optimizations that maintain clarity
- Example extensions that teach new concepts

## Documentation

- **README.md** (this file) - Getting started, features, extensions
- **LIMITATIONS.md** - Constraints, appropriate use cases, known issues

## License

MIT License - See [LICENSE](LICENSE) file for details.

This project is free to use, modify, and distribute. The MNIST dataset is publicly available under Creative Commons Attribution-Share Alike 3.0 license.

## Version History

**v1.0 (October 2025)**
- Initial public release
- Full MNIST training pipeline with 92-98% accuracy
- Progressive training with dynamic learning rate boosts
- Robust data augmentation (5 techniques)
- Human-in-the-loop learning from user corrections
- Comprehensive training journey logging

## Acknowledgments

- **MNIST Dataset:** Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- **Godot Engine:** The amazing Godot community
- **Inspiration:** All the learners who asked "but how does it really work?"

## Project Info

- **Author:** Zeyphrus @ Inchworm Games
- **GitHub:** https://github.com/Andraszao/project-dora-ml
- **Discord:** @zeyphrus
- **Version:** 1.0
- **License:** Educational use

---

**Made for learners, by a learner. Happy exploring! 🧠✨**
