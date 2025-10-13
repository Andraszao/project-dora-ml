# Project Dora - Known Limitations & Architecture Constraints

**Version 1.0** | Last Updated: October 2025

This document provides transparent information about Project Dora's limitations, constraints, and appropriate use cases. Understanding these helps set realistic expectations and guides future development.

## Intended Use Case

✅ **What Dora IS designed for:**
- Educational demonstrations of neural network fundamentals
- Interactive learning about ML concepts
- Small-scale digit recognition applications
- Prototyping and experimentation
- Understanding how neural networks actually work

❌ **What Dora is NOT designed for:**
- Production ML services at scale
- High-throughput real-time processing
- Mobile deployment (though Godot supports it)
- Critical systems requiring guaranteed accuracy
- Applications requiring sub-millisecond inference

## Performance Limitations

### Training Performance
**Limitation:** CPU-only training in GDScript
- **Impact:** Default settings (1,000 samples → 3,900 after augmentation, 15 epochs) takes approximately **30-60 minutes** depending on CPU. Full 60,000 samples would take many hours.
- **Reality check:** Plan for an hour of initial training time on first run
- **Mitigation:** Weights are saved and only trained once - subsequent runs load instantly. Coffee break during first training, then it's done forever.
- **When it matters:** If you need to retrain frequently or experiment with architecture changes
- **Alternative:** Reduce `training_samples` to 500 for faster experimentation

**Limitation:** Single-threaded batch processing
- **Impact:** Can't utilize all CPU cores during training
- **Mitigation:** Auto-optimized batch sizes help, speed optimizations enabled
- **When it matters:** Training on large datasets (>10,000 samples)

### Inference Performance
**Limitation:** Forward pass is unoptimized matrix multiplication in GDScript
- **Impact:** ~1-5ms per prediction on modern CPUs (fast enough for interactive drawing)
- **Reality:** Instant feedback for users - no perceptible delay
- **When it matters:** Real-time video processing (30+ fps) or high-frequency batch predictions
- **Note:** For drawing interface and interactive use, this is perfectly adequate

### Memory Usage
**Limitation:** Entire dataset loaded into memory during training
- **Impact:** Default 1,000 samples (→3,900 augmented) uses ~15-20MB. Full 60,000 samples would use ~250MB.
- **Reality:** Default settings are memory-friendly for all modern systems
- **Mitigation:** Use `training_samples` parameter to limit dataset size based on available RAM
- **When it matters:** Memory-constrained devices or very large custom datasets

## Architecture Limitations

### Fixed Architecture
**Limitation:** Hardcoded 784→128→10 network structure
- **Impact:** Can only process 28×28 grayscale images and classify 10 classes
- **Why:** Simplicity for learning - changing requires understanding the whole pipeline
- **Workaround:** Modify `input_size`, `hidden_size`, `output_size` (requires code changes)

### Single Hidden Layer
**Limitation:** Only one hidden layer (shallow network)
- **Impact:** Cannot learn extremely complex hierarchical features
- **Reality check:** For MNIST digits, this is sufficient (achieves 92-96% accuracy)
- **When to upgrade:** If extending to complex images (CIFAR-10, ImageNet)

### No Convolutional Layers
**Limitation:** Fully-connected layers don't exploit spatial structure
- **Impact:** Less parameter-efficient than CNNs, slightly lower accuracy ceiling
- **Trade-off:** Simpler to understand and implement
- **Typical accuracy:** Fully-connected ~96%, CNN ~99%+

## Data & Training Limitations

### MNIST-Specific Preprocessing
**Limitation:** Preprocessing pipeline optimized specifically for MNIST-style data
- **Impact:** Performance degrades on significantly different image types
- **Examples:** Color images, different sizes, non-centered objects
- **Mitigation:** The preprocessing is well-documented and can be modified

### Batch Training Only
**Limitation:** Initial training requires full dataset in memory
- **Impact:** Cannot stream data from disk during training
- **When it matters:** Custom datasets larger than available RAM
- **Partial mitigation:** Use smaller `training_samples` parameter

### Limited Regularization
**Limitation:** No dropout, L1/L2 regularization, or batch normalization
- **Impact:** May overfit on very small datasets or if overtrained
- **Mitigation:** Early stopping helps, data augmentation adds regularization effect
- **Reality:** For MNIST with augmentation, overfitting rarely occurs

## Security & Robustness Limitations

### Adversarial Examples
**Limitation:** No defense against adversarial inputs
- **Impact:** Carefully crafted images could fool the network
- **Reality:** For educational/demo use, this isn't a concern
- **When it matters:** Security-critical applications

### File System Access
**Limitation:** Saves weights to local filesystem with minimal validation
- **Impact:** Could potentially overwrite files or fail on permission errors
- **Mitigation:** All file operations use specific filenames (dora_*.dat)
- **Improvement needed:** Better permission/disk space checking

### No Rate Limiting
**Limitation:** No limits on interactive learning updates
- **Impact:** Rapid-fire corrections could theoretically destabilize weights
- **Reality:** Online learning rate is very small, making this unlikely
- **Mitigation:** Consider adding cooldown between corrections if needed

## Scalability Limitations

### Single Instance Only
**Limitation:** Not designed for distributed training or multi-user scenarios
- **Impact:** Can't pool learning across multiple users or devices
- **Use case alignment:** Perfect for single-user educational tool

### No Model Serving Infrastructure
**Limitation:** No REST API, no model deployment pipeline
- **Impact:** Can't easily integrate into web services or mobile apps
- **Reality:** Godot can export to web/mobile, but no serving infrastructure

### Manual Dataset Management
**Limitation:** MNIST data included in repo (large files)
- **Impact:** ~100MB repository size
- **Alternative:** Could download on first run, but adds complexity
- **Trade-off:** Convenience vs repo size - we chose convenience

## Accuracy & Capability Limitations

### Expected Accuracy
**Realistic expectations:**
- Training accuracy: 92-96% with default settings
- User drawing accuracy: 85-92% (depends on drawing quality)
- Best possible (this architecture): ~97-98%
- State-of-art (CNN): >99%

**Why the gap?**
- Fully-connected architecture vs convolutional
- Limited training samples (1,000 default vs 60,000 available)
- CPU training time constraints
- Educational simplicity prioritized over maximum accuracy

### Digit Recognition Only
**Limitation:** Only recognizes digits 0-9, nothing else
- **Cannot handle:** Letters, symbols, multiple digits, colored images
- **Extending:** Would require new dataset, modified architecture
- **Philosophy:** Do one thing well for learning purposes

## Development & Maintenance Limitations

### No Automated Testing
**Limitation:** No unit tests or integration tests
- **Impact:** Changes could introduce regressions unknowingly
- **Mitigation:** Well-documented code makes manual testing easier
- **Future:** Could add GDScript unit testing framework

### No Profiling Tools
**Limitation:** Limited performance profiling capabilities
- **Impact:** Hard to identify bottlenecks precisely
- **Available:** Godot profiler shows frame time, manual timing with Time class
- **Reality:** For this scale, profiling hasn't been necessary

### GDScript Performance Ceiling
**Limitation:** GDScript is interpreted, not compiled
- **Impact:** ~10-100x slower than C++ implementations
- **Trade-off:** Readability and cross-platform support vs raw speed
- **When it matters:** If you need 1000s of predictions per second

## Appropriate Use Cases

### ✅ Great For:
- Learning ML fundamentals
- Teaching neural networks in classrooms
- Prototyping digit recognition features
- Demonstrating ML concepts at conferences
- Personal projects and portfolios
- Understanding backpropagation viscerally

### ⚠️ Use With Caution:
- Mobile apps (works but not optimized)
- Real-time video processing (too slow)
- Large custom datasets (memory constraints)

### ❌ Not Suitable For:
- Production ML APIs serving millions of requests
- Safety-critical systems (medical, automotive)
- Applications requiring >99% accuracy guarantees
- High-security environments
- Distributed training clusters

## Future Improvement Paths

If you wanted to address these limitations, priority order:

**High Impact, Moderate Effort:**
1. Add unit tests for core functions
2. Implement model versioning and compatibility checks
3. Add comprehensive error handling and user feedback
4. Create automated benchmarking suite

**High Impact, High Effort:**
1. GPU acceleration using compute shaders
2. Convolutional layers for better accuracy
3. Model compression for mobile deployment
4. Data streaming for large datasets

**Lower Priority:**
1. REST API for model serving
2. Multi-user federated learning
3. Advanced regularization techniques
4. Automated hyperparameter tuning

## When to Move Beyond Dora

You should consider migrating to professional ML frameworks when:
- You need >99% accuracy (use CNNs in PyTorch/TensorFlow)
- Training time becomes a bottleneck (need GPU acceleration)
- You're deploying to production with SLA requirements
- You need features like distributed training or model versioning
- Dataset exceeds available memory

But for learning, teaching, and understanding neural networks deeply - Dora is intentionally simple and complete.

## Transparency Note

This project prioritizes **clarity over performance** and **education over optimization**. Every limitation listed here is a conscious trade-off in favor of making neural networks understandable. The goal isn't to compete with production frameworks - it's to teach how they work under the hood.

---

**Questions or concerns about limitations?** Open an issue on GitHub: https://github.com/Andraszao/project-dora-ml/issues

**Want to contribute improvements?** Pull requests welcome, especially for addressing the "High Impact, Moderate Effort" items above.

