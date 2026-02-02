# Project DORA - Documentation

**Version 1.0** | **Author:** Zeyphrus @ Inchworm Games  
**GitHub:** https://github.com/Andraszao/project-dora-ml

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Configuration Settings](#configuration-settings)
4. [Core Systems](#core-systems)
5. [Function Reference](#function-reference)
6. [Training Pipeline](#training-pipeline)
7. [Interactive Learning](#interactive-learning)
8. [Progressive Training System](#progressive-training-system)

---

## Overview

Project DORA is an interactive machine learning demonstration built in Godot that teaches a neural network to recognize handwritten digits. The system features:

- **Real-time learning** from user feedback
- **Progressive difficulty targets** that increase as Dora improves
- **Achievement-based learning rate boosts** ("success breeds success")
- **Comprehensive training journey logging** for analysis
- **Data augmentation** for improved generalization
- **MNIST dataset integration** (60,000 training samples)

### Key Innovation: Progressive Training

Unlike traditional ML systems with static accuracy targets, DORA dynamically increases its goals as it achieves success, creating a "gamified" training experience where achievement rewards accelerate learning.

---

## Architecture

### Neural Network
- **Input Layer:** 784 neurons (28×28 pixel grid)
- **Hidden Layer:** 128 neurons with ReLU activation
The ReLU (Rectified Linear Unit) activation function is a widely used activation function in neural networks. It is defined as a piecewise linear function that outputs the input directly if it is positive; otherwise, it outputs zero. This property introduces nonlinearity to the model, allowing it to learn complex patterns in the data.
- **Output Layer:** 10 neurons (digits 0-9) with softmax
The softmax function is a mathematical function that transforms a vector of real numbers into a vector of probabilities. Each element in the output is between 0 and 1, and the sum of all elements equals 1, making it useful for interpreting the outputs as probabilities in multi-class classification problems
- **Weight Initialization:** Xavier initialization for stable gradients
Xavier function, also known as Xavier initialization, is a technique used in deep learning to initialize the weights of neural networks. 

### Training Components
```
DrawingCanvas (Main Orchestrator)
    ├── NeuralNetwork (prediction & inference)
    ├── NeuralNetworkTrainer (backpropagation & optimization)
    ├── MNISTLoader (data loading & augmentation)
    └── PatternClassifier (fallback heuristic classifier)
```

---

## Configuration Settings

### Training Settings
| Setting | Default | Purpose |
|---------|---------|---------|
| `reset_memory_on_start` | false | Delete saved weights on startup |
| `force_fresh_training` | false | Retrain every time, ignore saved weights |
| `training_epochs` | 15 | Complete passes through dataset |
| `batch_size` | 100 | Samples per weight update |
| `training_samples` | 1000 | Total MNIST digits for training |

### Learning Settings
| Setting | Default | Purpose |
|---------|---------|---------|
| `enable_interactive_learning` | true | Allow user feedback learning |
| `online_learning_rate` | 0.0001 | Learning rate for feedback corrections |
| `show_weight_changes` | true | Display mathematical proof of learning |

### Optimizer Settings
| Setting | Default | Purpose |
|---------|---------|---------|
| `use_adam_optimizer` | true | Use Adam vs SGD |
| `adam_learning_rate` | 0.002 | Base learning rate |
| `adam_beta1` | 0.9 | Momentum decay (first moment) |
| `adam_beta2` | 0.999 | Momentum decay (second moment) |
| `adam_epsilon` | 1e-8 | Numerical stability constant |

### Progressive Training
| Setting | Default | Purpose |
|---------|---------|---------|
| `enable_progressive_targets` | true | Enable dynamic target system |
| `initial_target_accuracy` | 80.0% | Starting goal |
| `target_increment` | 4.0% | Accuracy increase per level |
| `max_target_accuracy` | 98.0% | Ultimate goal |
| `achievements_needed` | 2 | Consecutive epochs to advance |
| `achievement_lr_boost` | 1.05 | Learning rate boost per achievement (5%) |
| `escalation_lr_boost` | 1.10 | Learning rate boost for level-up (10%) |

---

## Core Systems

### 1. MNIST Data Pipeline
- Loads binary `.idx` files from training_data/
- Converts 28×28 grayscale images to normalized float arrays
- Applies optional augmentation (rotation, scaling, noise)
- Adds progressive noise training for robustness

### 2. Neural Network Training
- **Optimizer:** Adam with momentum and adaptive learning rates
- **Loss Function:** Cross-entropy for multi-class classification
- **Backpropagation:** Full gradient computation through hidden layer
- **Weight Persistence:** Saves trained weights to disk

### 3. Interactive Learning
- Real-time feedback from users ("Yes" / "No" / "Actually it's...")
- Online learning with small learning rate (0.0001) to prevent forgetting
- Weight change visualization for educational transparency
- Immediate persistence of learned corrections

### 4. Progressive Target System
- **Dynamic Goals:** Targets increase as Dora succeeds
- **Achievement Tracking:** Requires consecutive successes to advance
- **Learning Rate Boosts:** Rewards success with faster learning
- **Milestone Saving:** Backs up weights at major achievements

---

## Function Reference

### Initialization Functions

#### `_ready()`
**Purpose:** Main initialization and training orchestrator  
**Process:**
1. Handle memory reset if configured
2. Load MNIST dataset (with validation)
3. Apply data augmentation if enabled
4. Initialize or load neural network
5. Execute training pipeline or load saved weights
6. Setup canvas and UI connections

**Key Decision Points:**
- Should train fresh vs load weights?
- Enable progressive targets?
- Optimize batch size automatically?

---

#### `setup_canvas()`
**Purpose:** Initialize drawing surface and pixel data  
**Creates:**
- 280×280 display image for user drawing
- 28×28 pixel data array for ML processing
- Texture and StyleBox for panel rendering

---

### Drawing Functions

#### `_gui_input(event)`
**Purpose:** Process mouse/touch input for drawing  
**Handles:**
- Mouse button press/release (start/stop drawing)
- Mouse motion while drawing (continuous stroke)

---

#### `draw_at_position(pos: Vector2)`
**Purpose:** Paint circular brush stroke at position  
**Logic:**
- Validates position is within canvas bounds
- Draws circle using Pythagorean distance check
- Updates texture immediately for visual feedback
- Uses configurable `brush_size` and `brush_color`

---

#### `convert_to_ml_data()`
**Purpose:** Convert 280×280 canvas to 28×28 ML format  
**Steps:**
1. Scale down from display size to ML size
2. Average pixels in each block
3. Apply MNIST-style preprocessing
4. Store in normalized `pixel_data` array

---

#### `preprocess_like_mnist(raw_data: Array[float]) -> Array[float]`
**Purpose:** Advanced preprocessing for better recognition  
**Applies:**
- Bounding box detection
- Centering and aspect ratio preservation
- Contrast enhancement (multi-level thresholding)
- Optional Gaussian blur for anti-aliasing

**Returns:** 784-element array of normalized pixel values (0.0-1.0)

---

### Prediction Functions

#### `_on_test_pressed()`
**Purpose:** Process drawing and make prediction  
**Flow:**
1. Convert canvas to ML data
2. Run through neural network (or fallback classifier)
3. Display prediction with confidence percentage
4. Show feedback UI for interactive learning
5. Optionally show top-3 predictions

**Output Example:** "I think that's a 7! (94% confidence) 🎯"

---

### Interactive Learning Functions

#### `show_feedback_ui()` / `hide_feedback_ui()`
**Purpose:** Control visibility of feedback buttons  
**Buttons:**
- "Yes, that's right!" (positive reinforcement)
- "No, that's wrong" (prompt for correction)
- "Actually it's: [spinner]" (corrective learning)

---

#### `_on_feedback_yes()`
**Purpose:** Handle positive user feedback  
**Actions:**
1. Log correct prediction
2. Apply positive reinforcement to strengthen neural pathways
3. Increment learning counter
4. Save updated weights

---

#### `_on_feedback_no()`
**Purpose:** Handle negative feedback  
**Actions:**
1. Log incorrect prediction
2. Prompt user to specify correct digit
3. Update UI message for correction input

---

#### `_on_feedback_correct()`
**Purpose:** Process user correction  
**Process:**
1. Get correct digit from spinner input (0-9)
2. Validate input range
3. Apply corrective learning via backpropagation
4. Display weight changes if enabled
5. Save updated weights
6. Show confirmation message

---

#### `learn_from_feedback(input_data, correct_label, was_correct)`
**Purpose:** Core online learning implementation  
**Parameters:**
- `input_data: Array[float]` - 784 pixel values from drawing
- `correct_label: int` - True digit (0-9)
- `was_correct: bool` - Whether original prediction was right

**Algorithm:**
1. Validate inputs (size, range)
2. Capture neural network state (for proof of learning)
3. Initialize trainer with small online learning rate
4. Create one-hot target vector
5. Apply backpropagation (positive reinforcement or correction)
6. Measure and display weight changes
7. Persist updated weights

**Mathematical Proof:** Shows exact delta in weights and biases

---

### Training Pipeline Functions

#### `calculate_optimal_batch_size() -> int`
**Purpose:** Auto-optimize batch size for system  
**Considers:**
- CPU core count (scales with available cores)
- Memory safety (conservative limits)
- Dataset size (relative scaling)
- Fast training mode (50% boost if enabled)

**Returns:** Optimal batch size (50-250 typically)

---

#### `reset_dora_memory()`
**Purpose:** Delete saved weights for fresh start  
**Use Cases:**
- Fresh demos
- A/B testing
- Development iterations

---

### Progressive Training Functions

#### `_evaluate_progressive_target_per_epoch(epoch_accuracy, epoch_number) -> bool`
**Purpose:** Evaluate and update targets after each epoch  
**Logic:**
1. Initialize targets on first epoch
2. Check if current target was achieved
3. Increment achievement counter if successful
4. Apply achievement rewards (learning rate boost)
5. Escalate target if consecutive achievements met
6. Reset counter if target missed
7. Check for mastery (max target reached)

**Returns:** 
- `true` to continue training
- `false` to stop (mastery achieved)

**Output Example:**
```
📊 Epoch 5: 82.1% | Target: 80.0% | Achievements: 2/2
🎉 TARGET ACHIEVED! Achievement #2 towards next level
🚀 TARGET ESCALATED! New goal: 84.0% (was 80.0%)
```

---

#### `_get_current_learning_rate() -> float`
**Purpose:** Return current learning rate (including boosts)  
**Called by:** Trainer during backpropagation  
**Dynamic:** Returns boosted rate after achievements

---

#### `_apply_achievement_reward(epoch_number)`
**Purpose:** Apply training enhancements when targets hit  
**Actions:**
1. Boost learning rate by configurable factor (default 5%)
2. Log parameter change to training journey
3. Announce boost to console

**Philosophy:** "Success breeds success" - momentum from achievement

---

#### `_apply_escalation_bonus()`
**Purpose:** Special bonuses when targets escalate  
**Actions:**
1. Major learning rate boost (default 10%)
2. Log parameter change
3. Save milestone weights (if enabled)
4. Announce level-up

---

### Training Session Persistence

#### `save_training_session_state()`
**Purpose:** Persist comprehensive session state  
**Saves:**
- Progressive target state (current level, achievements)
- Dynamic parameters (earned learning rate boosts)
- Training statistics (epochs, accuracy, learning count)
- Achievement history
- Session metadata (timestamp, samples, augmentation)

**File:** `dora_training_session.dat` (JSON format)

---

#### `load_training_session_state()`
**Purpose:** Restore previous session state  
**Restores:**
- Target accuracy level
- Achievement progress
- Earned learning rate boosts (optional)
- Learning count from interactive feedback

**Configurability:** Can choose to restore boosted learning rate or start fresh

---

### Training Journey Logging

#### `initialize_training_journey_logging()`
**Purpose:** Start comprehensive training log  
**Creates:**
- Session number (auto-incremented)
- Session data structure with metadata
- Timestamp and configuration snapshot

---

#### `log_epoch_completion(epoch_number, accuracy, loss, target_accuracy)`
**Purpose:** Record detailed epoch metrics  
**Captures:**
- Epoch number and timestamp
- Accuracy and loss values
- Current target and achievement status
- Current learning rate
- Total samples trained

---

#### `log_achievement_milestone(milestone_type, details)`
**Purpose:** Record achievement events  
**Milestone Types:**
- `"target_achieved"` - Current target hit
- `"target_escalated"` - Level-up to new target
- `"max_target_reached"` - Mastery achieved

---

#### `log_parameter_change(parameter_name, old_value, new_value, reason)`
**Purpose:** Track dynamic hyperparameter changes  
**Records:**
- Parameter name (e.g., "adam_learning_rate")
- Old and new values
- Change ratio
- Reason (e.g., "achievement_reward", "escalation_bonus")

---

#### `finalize_training_session(final_accuracy, total_epochs)`
**Purpose:** Complete and save training session  
**Compiles:**
- Final results (accuracy, epochs, achievements)
- Session duration
- Success level categorization
- Complete journey log saved to JSON

---

#### `save_training_journey_log()` / `load_training_journey_log()`
**Purpose:** Persist/restore complete training history  
**File:** `dora_training_journey.json` (human-readable)  
**Contains:** All sessions with full metadata and progression

---

#### `get_training_journey_summary() -> String`
**Purpose:** Generate human-readable journey summary  
**Returns Example:**
```
Training Journey: 5 sessions, 96.2% best accuracy, 
75 total epochs, 12 achievements
```

---

#### `get_success_level(accuracy: float) -> String`
**Purpose:** Categorize training performance  
**Tiers:**
- `"exceptional"` - 98%+
- `"excellent"` - 95-98%
- `"good"` - 90-95%
- `"fair"` - 80-90%
- `"needs_improvement"` - <80%

---

### UI Control Functions

#### `_on_clear_pressed()`
**Purpose:** Clear canvas and reset UI  
**Actions:**
1. Reinitialize canvas to blank
2. Hide feedback UI
3. Reset prediction state

---

#### `_on_brush_size_changed(new_value: float)`
**Purpose:** Update brush size from slider  
**Updates:**
- Internal `brush_size` variable (clamped to min/max)
- Display label showing current size

---

### Utility Functions

#### `get_ml_data() -> Array[float]`
**Purpose:** Get current drawing as ML-ready data  
**Returns:** 784-element normalized pixel array

---

#### `apply_gaussian_blur(data, sigma) -> Array[float]`
**Purpose:** Apply 3×3 Gaussian blur for smoothing  
**Use:** Anti-aliasing (currently disabled for crispness)  
**Note:** `sigma` parameter reserved for future use

---

## Training Pipeline

### Standard Training Flow
```
1. Load MNIST data (configurable sample count)
2. Apply data augmentation (if enabled)
3. Add progressive noise training (if enabled)
4. Initialize neural network with Xavier weights
5. Configure optimizer (Adam or SGD)
6. Execute training epochs with batches
7. Save weights and session state
8. Initialize training journey logging
```

### Progressive Training Flow
```
For each epoch:
    1. Train on batches
    2. Evaluate accuracy on validation set
    3. Call _evaluate_progressive_target_per_epoch()
    4. Check if target achieved
    5. Apply achievement reward (boost learning rate)
    6. Check if ready to escalate target
    7. Apply escalation bonus (major boost)
    8. Log epoch details to journey
    9. Continue or stop based on mastery
```

---

## Interactive Learning

### User Feedback Flow
```
1. User draws digit on canvas
2. Clicks "Test My Drawing"
3. Dora makes prediction with confidence
4. Feedback UI appears with three options:
   
   Option A: "Yes, that's right!"
   → Positive reinforcement
   → Strengthen correct neural pathways
   → Increment learning counter
   
   Option B: "No, that's wrong"
   → Prompt for correction
   → Show "Actually it's:" input
   
   Option C: "Actually it's: [digit]"
   → Corrective learning
   → Adjust weights toward correct answer
   → Show mathematical proof of learning
   → Display weight/bias changes

5. Updated weights saved to disk
6. Dora remembers correction forever
```

### Online Learning Parameters
- **Learning Rate:** 0.0001 (100x smaller than batch training)
- **Purpose:** Prevent catastrophic forgetting
- **Method:** Single-sample backpropagation
- **Proof:** Displays exact numerical changes in weights/biases

---

## Progressive Training System

### Philosophy
Traditional ML systems train to a fixed target. DORA implements a "gamified" approach where:
- **Success is rewarded** with learning rate boosts
- **Targets escalate** as Dora improves
- **Momentum builds** through consecutive achievements
- **Maximum mastery** is the ultimate goal

### Example Progression
```
Session 1:
├─ Epoch 1: 75.2% | Target: 80.0% | Miss
├─ Epoch 2: 79.8% | Target: 80.0% | Miss  
├─ Epoch 3: 81.4% | Target: 80.0% | Achievement 1/2
│   └─ 🔥 Learning rate: 0.002 → 0.0021 (+5%)
├─ Epoch 4: 82.1% | Target: 80.0% | Achievement 2/2
│   ├─ 🔥 Learning rate: 0.0021 → 0.002205 (+5%)
│   └─ 🚀 TARGET ESCALATED! 80% → 84%
│       └─ 🔥 Learning rate: 0.002205 → 0.002426 (+10%)
├─ Epoch 5: 85.3% | Target: 84.0% | Achievement 1/2
...
└─ Final: 97.8% | Target: 98.0% | Mastery achieved!
```

### Achievement Mechanics
- **Consecutive requirement:** Must hit target multiple epochs in a row
- **Reward timing:** Boost applies to next epoch immediately
- **Escalation bonus:** Major boost when leveling up
- **Milestone saving:** Backs up weights at major achievements
- **Reset on failure:** Miss target = start achievement counter over

### Configurability
Every aspect is configurable:
- Initial target (50-95%)
- Increment per level (1-10%)
- Maximum target (90-99.9%)
- Achievements needed (1-5)
- Boost amounts (1.0-2.0x)
- Enable/disable entire system

---

## File Outputs

### Saved Files
| File | Purpose | Format |
|------|---------|--------|
| `dora_trained_weights.dat` | Neural network weights/biases | JSON |
| `dora_training_session.dat` | Session state and achievements | JSON |
| `dora_training_journey.json` | Complete training history | JSON (formatted) |
| `dora_milestone_weights.dat` | Backup at major achievements | JSON |

### Training Journey Structure
```json
{
  "total_sessions": 5,
  "journey_start_date": "2025-02-01 14:23:45",
  "last_updated": "2025-02-02 10:15:22",
  "sessions": [
    {
      "session_number": 1,
      "session_start_time": "2025-02-01 14:23:45",
      "initial_parameters": {
        "training_epochs": 15,
        "batch_size": 100,
        "adam_learning_rate": 0.002,
        "initial_target_accuracy": 80.0
      },
      "epochs": [
        {
          "epoch_number": 1,
          "accuracy": 81.2,
          "loss": 0.523,
          "target_accuracy": 80.0,
          "target_achieved": true
        }
      ],
      "achievements": [
        {
          "milestone_type": "target_achieved",
          "timestamp": "2025-02-01 14:25:12",
          "details": { "target": 80.0, "actual": 81.2 }
        }
      ],
      "parameter_changes": [
        {
          "parameter": "adam_learning_rate",
          "old_value": 0.002,
          "new_value": 0.0021,
          "reason": "achievement_reward"
        }
      ],
      "final_results": {
        "final_accuracy": 96.8,
        "total_epochs_completed": 15,
        "success_level": "excellent"
      }
    }
  ]
}
```

---

## Tips for Developers

### Training Configuration
- **Fast demos:** Use 500-1000 samples, 5-10 epochs
- **Quality training:** Use 5000-10000 samples, 15-25 epochs
- **Progressive learning:** Lower samples + more epochs = visible progression

### Debugging
- Enable `verbose_logging` for detailed console output
- Enable `show_prediction_details` to see top-3 predictions
- Enable `show_weight_changes` to verify learning occurs
- Check `visualize_first_digit` to confirm data loaded correctly

### Performance
- Enable `auto_optimize_batch_size` for system-appropriate settings
- Enable `enable_speed_optimizations` for faster training
- Disable journey logging if only measuring performance

### Learning Rate Strategy
- **Batch training:** Start with 0.002 (Adam default)
- **Online learning:** Use 0.0001 (100x smaller)
- **Achievement boosts:** 1.05x (5%) per achievement
- **Escalation boosts:** 1.10x (10%) on level-up

---

## Educational Value

Project DORA demonstrates:
1. **Real neural network learning** with mathematical proof
2. **Interactive ML** where humans teach AI in real-time
3. **Progressive difficulty** creating engaging learning progression
4. **Reward-based optimization** inspired by reinforcement learning
5. **Data augmentation** for generalization
6. **Hyperparameter sensitivity** through extensive configuration
7. **Training history tracking** for analysis and portfolio work

Perfect for:
- ML education and demonstrations
- Portfolio projects showing ML understanding
- Teaching backpropagation and gradient descent
- Exploring human-in-the-loop learning
- Game AI development skills

---

## Credits

**Author:** Zeyphrus (James) @ Inchworm Games  
**Framework:** Godot Engine 4.x  
**Dataset:** MNIST (Modified National Institute of Standards and Technology)  
**Optimization:** Adam (Adaptive Moment Estimation)

---

## License

See LICENSE file for details.
