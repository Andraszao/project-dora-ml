extends Panel
class_name DrawingCanvas

## ============================================================================
## PROJECT DORA - Main Canvas and Training Orchestrator
## Version 1.0 | Author: Zeyphrus @ Inchworm Games
## GitHub: https://github.com/Andraszao/project-dora-ml
## ============================================================================
##
## 🎛️ CONFIGURATION SETTINGS 🎛️
## These settings make Dora highly configurable for different use cases and demonstrations

@export_group("Training Settings")
## Deletes saved neural network weights on startup, forcing Dora to start with a completely blank slate.
## Useful for: Fresh demos, A/B testing, or showing learning progression from zero knowledge.
@export var reset_memory_on_start: bool = false

## Ignores any saved weights and retrains the neural network every time, even if weights exist.
## Useful for: Consistent demo experiences, testing training improvements, or development iterations.
@export var force_fresh_training: bool = false

## Number of complete passes through the training dataset during initial training.
## For progressive targets: More epochs with fewer samples = better learning progression visibility.
## Range: 10-25 recommended for progressive training, 3-8 for fast training.
@export_range(3, 50, 1) var training_epochs: int = 15

## Number of training samples processed together before updating weights during batch training.
## Smaller batches = more frequent updates. Larger batches = more stable gradients. Range: 10-200.
@export_range(10, 300, 10) var batch_size: int = 100

## Total number of MNIST digits used for initial training (max 60,000 available).
## For progressive targets: Fewer samples per epoch = gradual learning progression.
## Range: 500-2000 recommended for progressive training, 5000-10000 for fast training.
@export_range(100, 10000, 100) var training_samples: int = 1000

@export_group("Learning Settings") 
## Enables the interactive feedback system where users can correct Dora's predictions in real-time.
## When disabled, Dora only makes predictions without learning from user input.
@export var enable_interactive_learning: bool = true

## Learning rate for real-time weight updates when user provides feedback.
## Smaller values = gradual learning. Larger values = faster but potentially unstable learning.
## Range: 0.0001 to 0.001 recommended. Default 0.0001 = visible learning without forgetting batch training.
@export_range(0.00001, 0.001, 0.00001) var online_learning_rate: float = 0.0001

## Displays exact weight and bias changes in console when Dora learns from feedback.
## Provides mathematical proof that learning is actually occurring at the neural network level.
@export var show_weight_changes: bool = true

@export_group("Performance Settings")
## Automatically optimizes batch size based on CPU cores and available memory for faster training.
## Larger batches = faster training but may impact learning stability on some systems.
@export var auto_optimize_batch_size: bool = true
@export_range(0.1, 1.0, 0.1) var cpu_usage_limit: float = 0.7  ## CPU usage limit for batch size optimization (0.1-1.0)
@export var enable_fast_training_mode: bool = false  ## Enable aggressive optimizations for maximum training speed

@export_group("Optimizer Settings")
## Uses Adam optimizer instead of basic SGD for faster convergence and better performance.
## Adam adapts learning rates per parameter and typically achieves 2-5% higher accuracy.
@export var use_adam_optimizer: bool = true  ## Use Adam optimizer instead of basic SGD for faster convergence
@export_range(0.0001, 0.01, 0.0001) var adam_learning_rate: float = 0.002  ## Base learning rate for weight updates (0.0001-0.01)
@export_range(0.1, 0.999, 0.001) var adam_beta1: float = 0.9  ## Momentum decay rate for first moment estimates (0.9 recommended)
@export_range(0.9, 0.9999, 0.0001) var adam_beta2: float = 0.999  ## Momentum decay rate for second moment estimates (0.999 recommended)
@export_range(1e-10, 1e-6, 1e-9) var adam_epsilon: float = 1e-8  ## Small constant to prevent division by zero (1e-8 standard)

@export_group("Data Augmentation")
## Creates rotated, scaled, and noise-added variations of training data for better generalization.
## Helps Dora recognize digits drawn in different styles, angles, and sizes.
@export var enable_data_augmentation: bool = true
@export_range(1, 5, 1) var augmentation_factor: int = 3  ## Data multiplier: creates N variations per original image (1-5x)
@export_range(0.0, 45.0, 1.0) var max_rotation_degrees: float = 15.0  ## Maximum rotation angle for augmentation (0-45 degrees)
@export_range(0.5, 1.0, 0.05) var scale_range_min: float = 0.85  ## Minimum scale factor for augmentation (0.5-1.0)
@export_range(1.0, 2.0, 0.05) var scale_range_max: float = 1.15  ## Maximum scale factor for augmentation (1.0-2.0)
@export_range(0.0, 0.2, 0.01) var noise_level: float = 0.05  ## Gaussian noise level for robustness (0.0-0.2)

@export_group("Progressive Training")
## UNIQUE FEATURE: Dora gets better at learning as she succeeds!
## Dynamically increases target accuracy as Dora achieves previous goals, creating ever-higher standards.
## Learning rate BOOSTS activate when targets are hit - "success breeds success" momentum.
## NOTE: This applies to BATCH TRAINING only, not interactive feedback learning.
@export var enable_progressive_targets: bool = true
@export_range(50.0, 95.0, 1.0) var initial_target_accuracy: float = 80.0  ## Starting accuracy goal - realistic first challenge (50-95%)
@export_range(1.0, 10.0, 0.5) var target_increment: float = 4.0  ## Accuracy increase per level - larger = bigger jumps (1-10%)
@export_range(90.0, 99.9, 0.1) var max_target_accuracy: float = 98.0  ## Maximum achievable target - ultimate goal (90-99.9%)
@export_range(1, 5, 1) var achievements_needed: int = 2  ## Consecutive epochs needed to advance - consistency requirement (1-5)

@export_subgroup("Achievement Rewards")
## Training enhancements that activate when targets are achieved - success breeds success!
@export var enable_achievement_rewards: bool = true  ## Enable training boosts for achieving targets
@export_range(1.0, 2.0, 0.01) var achievement_lr_boost: float = 1.05  ## Learning rate boost per achievement (1.0-2.0)
@export_range(1.0, 2.0, 0.01) var escalation_lr_boost: float = 1.10  ## Learning rate boost for target escalation (1.0-2.0)
@export var save_milestone_weights: bool = true  ## Save weights at major milestones
@export var restore_earned_learning_rate: bool = false  ## Restore boosted learning rate from previous session

@export_group("Training Journey Logging")
## Comprehensive logging system that records every training session for analysis and portfolio documentation.
## Creates detailed records of Dora's learning progression across multiple training sessions.
@export var enable_training_journey_log: bool = true  ## Log detailed training progression for analysis
@export var log_epoch_details: bool = true  ## Record individual epoch performance and parameters
@export var log_achievement_milestones: bool = true  ## Record when targets are achieved and escalated
@export var log_parameter_changes: bool = true  ## Record learning rate boosts and other dynamic changes

@export_group("Noise Robustness Training")
## Trains Dora on progressively noisy versions of MNIST data for better real-world performance.
## Makes her robust to sketchy drawings, incomplete strokes, and various drawing styles.
@export var enable_noise_training: bool = true
@export var noise_training_ratio: float = 0.3  ## Fraction of training data to make noisy (0.0-1.0)
@export var max_training_noise: float = 0.2  ## Maximum noise level during training (0.0-0.5)
@export var noise_schedule_steps: int = 5  ## Number of progressive noise levels
@export var enable_partial_occlusion: bool = true  ## Train on partially hidden digits

@export_group("Training Optimization")
## Advanced training controls for achieving maximum performance and preventing overtraining.
## These settings help Dora reach peak performance efficiently.
@export var enable_early_stopping: bool = true  ## Stop training when accuracy plateaus
@export var early_stopping_patience: int = 2  ## Epochs to wait before stopping (if no improvement)
@export var target_accuracy_threshold: float = 99.0  ## Stop training when this accuracy is reached
@export var save_best_weights: bool = true  ## Keep the best performing weights during training
@export var enable_speed_optimizations: bool = true  ## Enable performance optimizations (reduce logging, cache calculations)
@export var progress_update_frequency: int = 20  ## How often to show batch progress (higher = faster, less verbose)

# Cellular Automata functionality moved to exploration folder

@export_group("User Interface")
## Controls the drawing experience and visual feedback for optimal user interaction.
## These settings affect how responsive and intuitive Dora feels to use.
@export var default_brush_size: int = 16  ## Starting brush size for drawing (pixels)
@export var min_brush_size: int = 4  ## Minimum allowed brush size
@export var max_brush_size: int = 40  ## Maximum allowed brush size
@export var brush_size_step: int = 2  ## How much brush size changes per slider tick
@export var show_confidence_threshold: float = 0.1  ## Minimum confidence to show prediction (0.0-1.0)

@export_group("Debug Settings")
## Enables detailed logging including training parameters, batch progress, and internal state.
## Useful for development and debugging, but creates verbose console output.
@export var verbose_logging: bool = false

## Shows the top 3 digit predictions with confidence percentages after each guess.
## Helps understand what Dora was "thinking" and how confident she was in alternatives.
@export var show_prediction_details: bool = true

## Displays ASCII art visualization of the first training digit during startup.
## Provides visual confirmation that MNIST data loaded correctly and shows data format.
@export var visualize_first_digit: bool = true

# AI systems for digit recognition
var pattern_classifier: PatternClassifier
var neural_network: NeuralNetwork

# Interactive learning variables
var current_prediction: int = -1
var current_confidence: float = 0.0
var current_pixel_data: Array[float] = []
var learning_count: int = 0

# Progressive training variables
var current_target_accuracy: float = 85.0
var target_achievements: int = 0
var epoch_accuracies: Array[float] = []

# Training journey logging system
var current_session_number: int = 1
var training_journey_log: Array[Dictionary] = []
var current_session_data: Dictionary = {}

# Drawing properties - initialized from export variables
var drawing = false
var brush_size: int  # Set from default_brush_size in _ready()
var brush_color = Color.WHITE

# Canvas data - 28x28 for ML processing
var canvas_size = Vector2(280, 280)  # Display size
var ml_size = Vector2(28, 28)        # ML processing size
var pixel_data: Array[float] = []    # Normalized pixel values for ML

# Drawing texture
var drawing_texture: ImageTexture
var drawing_image: Image

func _ready():
	"""
	Initialize Dora's neural network system with configurable training and learning settings.
	This function handles the complete ML pipeline: data loading, training, and setup.
	Also initializes UI elements from export variables for consistent configuration.
	"""
	
	# Initialize UI settings from export variables
	brush_size = default_brush_size
	
	# Step 1: Handle memory reset if configured
	# This ensures we can demonstrate learning from scratch or test different training approaches
	if reset_memory_on_start:
		reset_dora_memory()
		print("🗑️ [EDITOR SETTING] Memory reset enabled - starting fresh!")
	
	# Step 2: Load MNIST handwritten digit dataset
	# MNIST is the standard benchmark for digit recognition with 60,000 training samples
	print("🚀 Loading MNIST data and training neural network...")
	if verbose_logging:
		print("📊 [DEBUG] Training settings: %d samples, %d epochs, batch size %d" % [training_samples, training_epochs, batch_size])
	
	# Step 2.5: Optimize training parameters for this system
	if auto_optimize_batch_size:
		batch_size = calculate_optimal_batch_size()
	
	var mnist_data = MNISTLoader.load_training_batch(training_samples)
	
	if not mnist_data.has("images") or mnist_data["images"].size() == 0:
		print("❌ CRITICAL ERROR: Failed to load MNIST data!")
		print("   Please check that training_data/ folder contains:")
		print("   - train-images.idx3-ubyte")
		print("   - train-labels.idx1-ubyte")
		print("   Falling back to pattern classifier (limited accuracy)...")
		pattern_classifier = PatternClassifier.new()
		setup_canvas()
		return
	
	if mnist_data.has("images") and mnist_data["images"].size() > 0:
		print("🎉 MNIST data loaded successfully!")
		
		# Step 3: Data augmentation for improved generalization
		# Creates rotated, scaled, and noise-enhanced variations of training data
		if enable_data_augmentation and augmentation_factor > 1:
			var augmented_data = MNISTLoader.augment_training_data(
				mnist_data["images"], 
				mnist_data["labels"], 
				augmentation_factor
			)
			mnist_data["images"] = augmented_data["images"]
			mnist_data["labels"] = augmented_data["labels"]
			print("🔄 Data augmentation applied: ", augmentation_factor, "x multiplier")
		
		# Step 3.5: Progressive noise robustness training
		# Adds diffusion-inspired noise training for better real-world performance
		if enable_noise_training:
			var noise_enhanced_data = MNISTLoader.add_progressive_noise_training(
				mnist_data["images"],
				mnist_data["labels"],
				noise_training_ratio,
				max_training_noise,
				noise_schedule_steps
			)
			mnist_data["images"] = noise_enhanced_data["images"]
			mnist_data["labels"] = noise_enhanced_data["labels"]
			print("🌊 Noise robustness training applied!")
		
		# Step 4: Optional visualization for verification
		# Shows ASCII art of first training digit to confirm data loaded correctly
		if visualize_first_digit:
			MNISTLoader.visualize_digit(mnist_data["images"][0], mnist_data["labels"][0])
		
		# Step 4: Initialize the neural network architecture
		# Creates a 784->128->10 feedforward network with Xavier weight initialization
		neural_network = NeuralNetwork.new()
		
		# Step 5: Determine training strategy based on configuration
		# Either use existing weights (fast startup) or train fresh (reproducible results)
		var should_train = force_fresh_training or not neural_network.load_weights_from_file()
		
		if should_train:
			# Training path: Build Dora's knowledge from scratch
			if force_fresh_training:
				print("🆕 [EDITOR SETTING] Force fresh training enabled!")
			else:
				print("🆕 No previous memories found - training from scratch...")
			
			print("\n🏋️ Starting neural network training...")
			
			# Create trainer and load the dataset
			var trainer = NeuralNetworkTrainer.new(neural_network)
			trainer.load_training_data(mnist_data["images"], mnist_data["labels"])
			
			# Configure optimizer settings
			if use_adam_optimizer:
				# Pass callback so trainer can get dynamic learning rate during progressive training
				var lr_callback = Callable(self, "_get_current_learning_rate")
				trainer.enable_adam_optimizer(adam_learning_rate, adam_beta1, adam_beta2, adam_epsilon, lr_callback)
				print("🔧 Adam optimizer enabled with LR: %.4f (dynamic)" % adam_learning_rate)
			else:
				trainer.learning_rate = adam_learning_rate  # Use as base learning rate
				trainer.dynamic_lr_callback = Callable(self, "_get_current_learning_rate")
				print("🔧 Using SGD optimizer with LR: %.4f (dynamic)" % adam_learning_rate)
			
			# Configure early stopping if enabled
			if enable_early_stopping:
				trainer.enable_early_stopping(early_stopping_patience, target_accuracy_threshold)
				print("⏰ Early stopping configured!")
			
			# Configure speed optimizations if enabled
			if enable_speed_optimizations:
				trainer.enable_speed_optimizations(progress_update_frequency)
				print("🚀 Speed optimizations enabled!")
			
			# Execute training with configurable parameters
			# This is where Dora learns to recognize digits through backpropagation
			if enable_progressive_targets:
				# Create progressive target callback that evaluates after each epoch
				var progressive_callback = Callable(self, "_evaluate_progressive_target_per_epoch")
				trainer.train_multiple_epochs(training_epochs, batch_size, progressive_callback)
			else:
				trainer.train_multiple_epochs(training_epochs, batch_size)
			
			# Persist trained weights AND training session state for future sessions
			neural_network.save_weights_to_file()
			save_training_session_state()
			
			print("🎯 Training complete! Dora is now smart and will remember!")
		else:
			# Fast path: Load pre-existing knowledge
			load_training_session_state()
			print("🎉 Dora remembers her previous training!")
	
		# Initialize training journey logging
		if enable_training_journey_log:
			initialize_training_journey_logging()
			
			# Finalize training journey log if we just completed training
			if epoch_accuracies.size() > 0:
				var final_accuracy = epoch_accuracies.max()
				finalize_training_session(final_accuracy, epoch_accuracies.size())
	else:
		print("⚠️ MNIST data not loaded, using pattern classifier")
		# Initialize the pattern classifier
		pattern_classifier = PatternClassifier.new()
	
	# Ensure the canvas can receive keyboard input (regardless of training mode)
	grab_focus()
	set_focus_mode(Control.FOCUS_ALL)
	
	# Initialize the drawing surface
	setup_canvas()
	
	# Connect button signals
	var clear_button = get_node("../ButtonContainer/ClearButton")
	var test_button = get_node("../ButtonContainer/TestButton")
	var brush_slider = get_node("../BrushContainer/BrushSlider")
	var brush_value_label = get_node("../BrushContainer/BrushValue")
	
	# Connect feedback buttons
	var yes_button = get_node("../FeedbackContainer/FeedbackButtons/YesButton")
	var no_button = get_node("../FeedbackContainer/FeedbackButtons/NoButton")
	var correct_button = get_node("../FeedbackContainer/CorrectionContainer/CorrectButton")
	
	clear_button.pressed.connect(_on_clear_pressed)
	test_button.pressed.connect(_on_test_pressed)
	brush_slider.value_changed.connect(_on_brush_size_changed)
	yes_button.pressed.connect(_on_feedback_yes)
	no_button.pressed.connect(_on_feedback_no)
	correct_button.pressed.connect(_on_feedback_correct)
	
	# Cellular automata functionality moved to [future] exploration folder
	# All cellular automata variables and functions have been removed
	
	print("🎨 Drawing canvas ready! Touch to draw.")
	if neural_network:
		print("🧠 Trained neural network ready!")
	else:
		print("🧠 Pattern classifier ready!")

func setup_canvas():
	"""
	Initialize the drawing surface texture and pixel data array.
	Creates a blank canvas matching the panel's actual size for user drawing.
	"""
	# Use actual panel size (it might be different than expected 280x280)
	var actual_size = get_size()
	print("🖼️ Panel actual size: ", actual_size, " vs expected: ", canvas_size)
	
	# Create a blank image for drawing using actual size
	drawing_image = Image.create(int(actual_size.x), int(actual_size.y), false, Image.FORMAT_RGBA8)
	drawing_image.fill(Color.BLACK)  # Black background
	
	# Create texture from image
	drawing_texture = ImageTexture.new()
	drawing_texture.set_image(drawing_image)
	
	# Set as panel texture
	var style_box = StyleBoxTexture.new()
	style_box.texture = drawing_texture
	add_theme_stylebox_override("panel", style_box)
	
	# Initialize pixel data array
	pixel_data.clear()
	pixel_data.resize(28 * 28)
	pixel_data.fill(0.0)

func _gui_input(event):
	"""
	Process mouse and touch input for drawing on the canvas.
	Routes input events to the drawing handler.
	"""
	_handle_drawing_input(event)

func _handle_drawing_input(event):
	"""
	Handle mouse button presses and motion for drawing digits.
	Left mouse button draws, motion continues the stroke while pressed.
	"""
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			drawing = event.pressed
			if drawing:
				# Event position is already local to this control
				draw_at_position(event.position)
	
	elif event is InputEventMouseMotion and drawing:
		# Event position is already local to this control
		draw_at_position(event.position)

func draw_at_position(pos: Vector2):
	"""
	Draw a circular brush stroke at the specified position.
	Uses the current brush_size and brush_color to paint pixels on the canvas.
	"""
	# Debug: Show actual panel size vs expected size
	var actual_size = get_size()
	
	# Make sure position is within canvas bounds (use actual panel size)
	if pos.x < 0 or pos.y < 0 or pos.x >= actual_size.x or pos.y >= actual_size.y:
		if randf() < 0.1:  # Occasional debug
			print("🚫 Out of bounds: pos=", pos, " actual_size=", actual_size, " expected=", canvas_size)
		return
	
	# Draw a circle at the position
	for x in range(-brush_size/2, brush_size/2 + 1):
		for y in range(-brush_size/2, brush_size/2 + 1):
			var draw_x = int(pos.x) + x
			var draw_y = int(pos.y) + y
			
			# Check bounds and circular brush (use actual panel size)
			if (draw_x >= 0 and draw_x < actual_size.x and 
				draw_y >= 0 and draw_y < actual_size.y and
				x*x + y*y <= (brush_size/2) * (brush_size/2)):
				
				drawing_image.set_pixel(draw_x, draw_y, brush_color)
	
	# Update the texture
	drawing_texture.update(drawing_image)
	
	# Less verbose logging - only log occasionally
	if randf() < 0.1:  # Only 10% of the time
		print("✏️ Drawing at: ", pos, " (canvas: ", canvas_size, ")")

func _on_clear_pressed():
	"""
	Clear the canvas when the Clear button is pressed.
	Resets the drawing surface to blank and hides feedback UI.
	"""
	print("🧹 Clearing canvas...")
	
	setup_canvas()
	hide_feedback_ui()

func _on_brush_size_changed(new_value: float):
	"""
	Update brush size when the slider value changes.
	Updates both the internal brush_size and the displayed value label.
	"""
	# Clamp to valid range as safety check
	brush_size = int(clamp(new_value, min_brush_size, max_brush_size))
	var brush_value_label = get_node("../BrushContainer/BrushValue")
	brush_value_label.text = str(brush_size)
	print("🖌️ Brush size changed to: ", brush_size)

func show_feedback_ui():
	"""
	Display the feedback buttons after Dora makes a prediction.
	Only shows if interactive learning is enabled in settings.
	"""
	if enable_interactive_learning:
		var feedback_container = get_node("../FeedbackContainer")
		feedback_container.visible = true

func hide_feedback_ui():
	"""
	Hide the feedback buttons.
	Called when clearing canvas or after feedback is processed.
	"""
	var feedback_container = get_node("../FeedbackContainer")
	feedback_container.visible = false

func _on_feedback_yes():
	"""
	Handles positive user feedback when Dora's prediction was correct.
	Works for both drawing recognition and cellular growth evaluation.
	"""
	print("👍 Positive feedback! Dora was right about: ", current_prediction)
	learning_count += 1
	
	# Apply positive reinforcement to strengthen correct neural pathways
	if neural_network:
		learn_from_feedback(current_pixel_data, current_prediction, true)
	
	print("🧠 Dora learned from positive feedback! (Total lessons: %d)" % learning_count)
	
	hide_feedback_ui()

func _on_feedback_no():
	"""
	Handle "No" button press when Dora's prediction was incorrect.
	Prompts the user to specify what the correct digit actually was.
	"""
	print("👎 Negative feedback! Dora was wrong about: ", current_prediction)
	var prediction_label = get_node("../Prediction")
	prediction_label.text = "Oops! What digit was it really? Use 'Actually it's:' below 🤔"

func _on_feedback_correct():
	"""
	Handles corrective user feedback when Dora's prediction was wrong.
	
	This implements supervised learning correction - adjusting neural weights to prefer
	the correct answer over the incorrect prediction for similar input patterns.
	"""
	var correction_input = get_node("../FeedbackContainer/CorrectionContainer/CorrectionInput")
	var correct_digit = int(correction_input.value)
	
	# Validate input is in valid range
	if correct_digit < 0 or correct_digit > 9:
		print("❌ Invalid digit: %d. Must be 0-9." % correct_digit)
		var prediction_label = get_node("../Prediction")
		prediction_label.text = "Invalid digit! Please enter 0-9. 🤔"
		return
	
	print("📚 Correction received! It was actually: ", correct_digit, " (Dora guessed: ", current_prediction, ")")
	learning_count += 1
	
	# Apply corrective learning to adjust weights toward the correct answer
	if neural_network:
		learn_from_feedback(current_pixel_data, correct_digit, false)
	
	var prediction_label = get_node("../Prediction")
	prediction_label.text = "Thanks! I learned it was a %d. I'll remember that! 🎓" % correct_digit
	
	hide_feedback_ui()
	print("🎓 Dora learned from correction! (Total lessons: %d)" % learning_count)

func calculate_optimal_batch_size() -> int:
	"""
	Automatically calculate optimal batch size based on system capabilities.
	
	This function analyzes CPU cores and available memory to determine the best
	batch size for maximum training speed while maintaining learning stability.
	"""
	var cpu_cores = OS.get_processor_count()
	var safe_cores = max(1, int(cpu_cores * cpu_usage_limit))
	
	# Base batch size calculation
	var cpu_optimal = safe_cores * 25  # 25 samples per core (empirically good)
	var memory_safe = 150  # Conservative for most systems
	var dataset_relative = min(training_samples / 50, 200)  # Scale with dataset size
	
	# Choose the most conservative option
	var optimal_batch = min(cpu_optimal, memory_safe, dataset_relative)
	optimal_batch = max(optimal_batch, 50)  # Minimum batch size for stability
	
	if enable_fast_training_mode:
		optimal_batch = min(optimal_batch * 1.5, 250)  # 50% boost in fast mode
	
	print("🎯 Optimal batch size calculated:")
	print("   CPU cores: %d (using %d at %.0f%% limit)" % [cpu_cores, safe_cores, cpu_usage_limit * 100])
	print("   Recommended batch size: %d" % optimal_batch)
	print("   Fast training mode: %s" % ("ON" if enable_fast_training_mode else "OFF"))
	
	return int(optimal_batch)

func reset_dora_memory():
	"""
	Delete saved neural network weights to start fresh.
	Useful for demonstrations or testing different training approaches.
	"""
	var weights_file = "dora_trained_weights.dat"
	if FileAccess.file_exists(weights_file):
		DirAccess.remove_absolute(weights_file)
		print("🗑️ Deleted previous memories - Dora will start learning fresh!")
	else:
		print("🆕 No previous memories found - starting fresh!")

func learn_from_feedback(input_data: Array[float], correct_label: int, was_correct: bool):
	"""
	Implements real-time learning from user feedback using backpropagation.
	
	This is the core of Dora's interactive learning system. When a user provides feedback
	(either confirming a correct prediction or correcting a wrong one), this function
	immediately updates the neural network weights to incorporate that knowledge.
	
	Args:
		input_data: The 28x28 pixel data from user's drawing (784 float values 0.0-1.0)
		correct_label: The true digit (0-9) that the user drew
		was_correct: Whether Dora's original prediction was right (affects reinforcement type)
	"""
	if not neural_network:
		print("⚠️ No neural network available for learning")
		return
	
	# Validate inputs
	if input_data.size() != 784:
		print("❌ Invalid input data size: %d (expected 784)" % input_data.size())
		return
	
	if correct_label < 0 or correct_label > 9:
		print("❌ Invalid label: %d (must be 0-9)" % correct_label)
		return
	
	print("🔄 Teaching Dora... (was_correct: %s)" % was_correct)
	
	# Capture neural network state before learning for comparison
	# This allows us to mathematically prove that learning occurred
	var before_weights = neural_network.weights_hidden_output[0][correct_label]  # Sample weight
	var before_bias = neural_network.bias_output[correct_label]  # Sample bias
	
	# Initialize online learning trainer with configurable learning rate
	# Smaller learning rates prevent catastrophic forgetting of previous knowledge
	var trainer = NeuralNetworkTrainer.new(neural_network)
	
	# Configure optimizer for online learning (much smaller learning rate than batch training)
	if use_adam_optimizer:
		trainer.enable_adam_optimizer(online_learning_rate, adam_beta1, adam_beta2, adam_epsilon)
		if verbose_logging:
			print("   🔧 Online learning LR: %.6f (10x smaller than batch training)" % online_learning_rate)
	else:
		trainer.learning_rate = online_learning_rate
		if verbose_logging:
			print("   🔧 Online learning LR: %.6f" % online_learning_rate)
	
	# Create one-hot encoded target vector for the correct answer
	# Example: if correct_label=5, target=[0,0,0,0,0,1,0,0,0,0]
	var target: Array[float] = []
	for i in range(10):
		target.append(1.0 if i == correct_label else 0.0)
	
	# Get current network state for comparison
	var current_result = neural_network.predict(input_data)
	
	# Apply different learning strategies based on feedback type
	if was_correct:
		# Positive reinforcement: Strengthen the pathway that led to correct prediction
		# This increases confidence in similar patterns in the future
		trainer.backpropagate(input_data, correct_label, current_result.probabilities)
		print("✅ Reinforced correct prediction for digit: ", correct_label)
	else:
		# Corrective learning: Adjust weights to prefer the correct answer
		# This reduces likelihood of making the same mistake again
		trainer.backpropagate(input_data, correct_label, current_result.probabilities)
		print("🔧 Corrected prediction: %d → %d" % [current_prediction, correct_label])
	
	# Capture neural network state after learning to measure changes
	var after_weights = neural_network.weights_hidden_output[0][correct_label]
	var after_bias = neural_network.bias_output[correct_label]
	
	# Display mathematical proof of learning (optional for demos)
	# This shows exact numerical changes in the neural network's internal parameters
	if show_weight_changes:
		var weight_change = abs(after_weights - before_weights)
		var bias_change = abs(after_bias - before_bias)
		print("🧠 PROOF OF LEARNING:")
		print("   Weight[0][%d]: %.6f → %.6f (change: %.6f)" % [correct_label, before_weights, after_weights, weight_change])
		print("   Bias[%d]: %.6f → %.6f (change: %.6f)" % [correct_label, before_bias, after_bias, bias_change])
	
	# Persist the updated neural network to disk for future sessions
	# This ensures Dora remembers all user corrections across app restarts
	neural_network.save_weights_to_file()
	print("💾 Updated Dora's memory with new knowledge!")

func _on_test_pressed():
	"""
	Process the drawn image and make a prediction when Test button is pressed.
	Converts canvas to ML format and runs through neural network or pattern classifier.
	"""
	print("🔍 Processing drawing...")
	convert_to_ml_data()
	
	# Use trained neural network if available, otherwise pattern classifier
	var prediction_label = get_node("../Prediction")
	
	if neural_network:
		# Get prediction from trained neural network!
		var result = neural_network.predict(pixel_data)
		
		if result.prediction >= 0:
			# Store current prediction for feedback
			current_prediction = result.prediction
			current_confidence = result.confidence
			current_pixel_data = pixel_data.duplicate()
			
			var confidence_percent = int(result.confidence * 100)
			prediction_label.text = "I think that's a %d! (%d%% confidence) 🎯" % [result.prediction, confidence_percent]
			print("🎯 TRAINED NN Prediction: %d (%.1f%% confidence)" % [result.prediction, result.confidence * 100])
			
			# Show feedback options
			show_feedback_ui()
			
			# 🎛️ CONFIGURABLE PREDICTION DETAILS
			if show_prediction_details:
				var sorted_probs = []
				for i in range(10):
					sorted_probs.append({"digit": i, "prob": result.probabilities[i]})
				sorted_probs.sort_custom(func(a, b): return a.prob > b.prob)
				
				print("📊 Top 3 predictions from trained network:")
				for i in range(min(3, sorted_probs.size())):
					var item = sorted_probs[i]
					print("   %d: %.1f%%" % [item.digit, item.prob * 100])
		else:
			prediction_label.text = "Hmm, I'm not sure what that is... 🤔"
			hide_feedback_ui()
	else:
		# Fallback to pattern classifier
		var result = pattern_classifier.classify_digit(pixel_data)
		
		if result.prediction >= 0:
			var confidence_percent = int(result.confidence * 100)
			prediction_label.text = "I think that's a %d! (%d%% confidence) 🎯" % [result.prediction, confidence_percent]
			print("🎯 Pattern Classifier Prediction: %d (%.1f%% confidence)" % [result.prediction, result.confidence * 100])
			print("🧠 Reasoning: ", result.reasoning)
		else:
			prediction_label.text = "Hmm, I'm not sure what that is... 🤔"

func convert_to_ml_data():
	"""
	Convert the canvas drawing into 28x28 normalized pixel data for ML processing.
	Applies MNIST-style preprocessing including centering and contrast enhancement.
	"""
	pixel_data.clear()
	pixel_data.resize(28 * 28)
	
	# Use actual image size
	var actual_size = Vector2(drawing_image.get_width(), drawing_image.get_height())
	var scale_factor_x = actual_size.x / ml_size.x
	var scale_factor_y = actual_size.y / ml_size.y
	
	print("🔄 Converting image: ", actual_size, " → 28x28 (scale: ", scale_factor_x, "x", scale_factor_y, ")")
	
	# First pass: convert to 28x28 grayscale
	var raw_data: Array[float] = []
	for ml_y in range(int(ml_size.y)):
		for ml_x in range(int(ml_size.x)):
			var pixel_sum = 0.0
			var pixel_count = 0
			
			# Average the pixels in this block
			for y in range(int(ml_y * scale_factor_y), int((ml_y + 1) * scale_factor_y)):
				for x in range(int(ml_x * scale_factor_x), int((ml_x + 1) * scale_factor_x)):
					if x < actual_size.x and y < actual_size.y:
						var color = drawing_image.get_pixel(x, y)
						# Convert to grayscale (we want white pixels to be 1.0)
						pixel_sum += (color.r + color.g + color.b) / 3.0
						pixel_count += 1
			
			# Normalize and store
			var normalized_value = pixel_sum / pixel_count if pixel_count > 0 else 0.0
			raw_data.append(normalized_value)
	
	# MNIST-style preprocessing
	pixel_data = preprocess_like_mnist(raw_data)
	
	var non_zero_count = pixel_data.filter(func(x): return x > 0.0).size()
	var max_value = pixel_data.max()
	print("📊 Converted to 28x28 ML data. Non-zero pixels: %d/%d, Max value: %.3f" % [non_zero_count, pixel_data.size(), max_value])
	print("📊 First few pixels: ", pixel_data.slice(0, 10))
	print("📊 Center pixels (14,14): ", pixel_data[14*28 + 14], " nearby: ", pixel_data.slice(14*28 + 12, 14*28 + 17))

func preprocess_like_mnist(raw_data: Array[float]) -> Array[float]:
	"""
	Advanced MNIST-style preprocessing for better recognition accuracy
	Includes: bounding box detection, centering, aspect ratio preservation, 
	contrast enhancement, anti-aliasing, and morphological operations
	"""
	var processed: Array[float] = []
	
	# Step 1: Find the tight bounding box of the drawing
	var min_x = 28
	var max_x = -1
	var min_y = 28  
	var max_y = -1
	
	for y in range(28):
		for x in range(28):
			var pixel = raw_data[y * 28 + x]
			if pixel > 0.05:  # Lower threshold to catch lighter strokes
				min_x = min(min_x, x)
				max_x = max(max_x, x)
				min_y = min(min_y, y)
				max_y = max(max_y, y)
	
	# Handle empty drawings
	if max_x == -1:
		for i in range(784):
			processed.append(0.0)
		return processed
	
	print("🎯 Drawing bounds: (%d,%d) to (%d,%d)" % [min_x, min_y, max_x, max_y])
	
	# Step 2: Calculate optimal centering with aspect ratio preservation
	var width = max_x - min_x + 1
	var height = max_y - min_y + 1
	var max_dim = max(width, height)
	
	# MNIST digits typically use ~20x20 of the 28x28 space (leaving 4px border)
	var target_size = min(20, max_dim)  # Don't overshrink tiny drawings
	var scale_factor = float(target_size) / float(max_dim)
	
	# Center the scaled digit
	var center_x = 14  # Center of 28x28 grid
	var center_y = 14
	var scaled_width = int(width * scale_factor)
	var scaled_height = int(height * scale_factor)
	
	var offset_x = center_x - int(scaled_width / 2)
	var offset_y = center_y - int(scaled_height / 2)
	
	print("🎯 Scaling factor: %.3f, Target size: %dx%d, Offset: (%d,%d)" % [scale_factor, scaled_width, scaled_height, offset_x, offset_y])
	
	# Step 3: Apply advanced preprocessing
	for y in range(28):
		for x in range(28):
			var pixel_value = 0.0
			
			# Map back to original coordinates
			var orig_x = int((x - offset_x) / scale_factor) + min_x
			var orig_y = int((y - offset_y) / scale_factor) + min_y
			
			if (orig_x >= 0 and orig_x < 28 and orig_y >= 0 and orig_y < 28):
				pixel_value = raw_data[orig_y * 28 + orig_x]
				
				# Advanced contrast enhancement
				if pixel_value > 0.7:
					pixel_value = 1.0  # Full intensity for strong strokes
				elif pixel_value > 0.3:
					pixel_value = 0.7 + (pixel_value - 0.3) * 0.75  # Gradual enhancement
				elif pixel_value > 0.1:
					pixel_value = pixel_value * 2.0  # Boost medium values
				elif pixel_value > 0.05:
					pixel_value = 0.2  # Light gray for very faint marks
				else:
					pixel_value = 0.0  # Remove noise
			
			processed.append(pixel_value)
	
	# Step 4: Apply light smoothing to reduce aliasing (optional)
	if false:  # Disabled for now to maintain crispness
		processed = apply_gaussian_blur(processed, 0.5)
	
	return processed

func apply_gaussian_blur(data: Array[float], _sigma: float) -> Array[float]:
	"""
	Apply light Gaussian blur to reduce aliasing artifacts.
	Note: Currently uses fixed 3x3 Gaussian kernel, sigma parameter reserved for future use.
	"""
	var blurred: Array[float] = []
	var kernel_size = 3  # 3x3 kernel for light blur
	var kernel_offset = kernel_size / 2
	
	# Simple 3x3 Gaussian kernel
	var kernel = [
		[0.0625, 0.125, 0.0625],
		[0.125,  0.25,  0.125],
		[0.0625, 0.125, 0.0625]
	]
	
	for y in range(28):
		for x in range(28):
			var sum = 0.0
			var weight_sum = 0.0
			
			for ky in range(kernel_size):
				for kx in range(kernel_size):
					var sample_x = x + kx - kernel_offset
					var sample_y = y + ky - kernel_offset
					
					if sample_x >= 0 and sample_x < 28 and sample_y >= 0 and sample_y < 28:
						var weight = kernel[ky][kx]
						sum += data[sample_y * 28 + sample_x] * weight
						weight_sum += weight
			
			blurred.append(sum / weight_sum if weight_sum > 0 else 0.0)
	
	return blurred

func _evaluate_progressive_target_per_epoch(epoch_accuracy: float, epoch_number: int) -> bool:
	"""
	Evaluate progressive targets after each epoch during training.
	Returns true to continue training, false to stop early.
	
	This creates a dynamic training environment where targets increase as Dora improves,
	preventing stagnation and pushing toward excellence.
	"""
	
	# Initialize progressive targets on first epoch
	if epoch_number == 1:
		current_target_accuracy = initial_target_accuracy
		target_achievements = 0
		epoch_accuracies.clear()
		print("🎯 Progressive targets initialized! Starting goal: %.1f%%" % current_target_accuracy)
	
	# Store this epoch's accuracy
	epoch_accuracies.append(epoch_accuracy)
	
	# Check if current target was achieved
	var target_hit = epoch_accuracy >= current_target_accuracy
	if target_hit:
		target_achievements += 1
	
	# Print status with UPDATED count
	print("📊 Epoch %d: %.1f%% | Target: %.1f%% | Achievements: %d/%d" % 
		[epoch_number, epoch_accuracy, current_target_accuracy, target_achievements, achievements_needed])
	
	if target_hit:
		print("🎉 TARGET ACHIEVED! Achievement #%d towards next level" % target_achievements)
		
		# TRAINING ENHANCEMENT: Reward success with learning rate boost
		if enable_achievement_rewards:
			_apply_achievement_reward(epoch_number)
		
		# Check if ready to escalate target
		if target_achievements >= achievements_needed:
			var new_target = current_target_accuracy + target_increment
			
			if new_target <= max_target_accuracy:
				current_target_accuracy = new_target
				target_achievements = 0  # Reset counter
				print("🚀 TARGET ESCALATED! New goal: %.1f%% (was %.1f%%)" % [current_target_accuracy, current_target_accuracy - target_increment])
				print("🏆 Dora is becoming an expert! Next challenge awaits...")
				
				# TRAINING ENHANCEMENT: Escalation bonus
				if enable_achievement_rewards:
					_apply_escalation_bonus()
			else:
				print("👑 MASTERY ACHIEVED! Dora has reached maximum target accuracy: %.1f%%" % max_target_accuracy)
				print("🎓 Training goals completed - Dora is now a digit recognition expert!")
				return false  # Stop training - mastery achieved
	else:
		# Reset achievements on failure (need consecutive successes)
		if target_achievements > 0:
			target_achievements = 0
			print("📉 Target missed - achievement counter reset. Need %d consecutive hits!" % achievements_needed)
	
	# Show trend if we have enough data
	if epoch_accuracies.size() >= 3:
		var recent = epoch_accuracies.slice(-3)
		var trend = recent[-1] - recent[0]
		var trend_text = "improving ⬆" if trend > 0.5 else "stable ➡" if abs(trend) <= 0.5 else "declining ⬇"
		print("📈 3-epoch trend: %.1f%% (%s)" % [trend, trend_text])
	
	return true  # Continue training

func get_ml_data() -> Array[float]:
	"""
	Get the current drawing converted to ML-ready 28x28 pixel data.
	Converts and returns normalized pixel values suitable for neural network input.
	"""
	convert_to_ml_data()
	return pixel_data

func _get_current_learning_rate() -> float:
	"""
	Returns current learning rate for dynamic updates during training.
	Called by trainer during backpropagation to get latest LR (including boosts).
	"""
	return adam_learning_rate

func _apply_achievement_reward(_epoch_number: int):
	"""
	Apply training enhancements when accuracy targets are achieved.
	Boosts learning rate to capitalize on successful learning momentum.
	"""
	print("🔥 ACHIEVEMENT REWARD ACTIVATED!")
	
	# Boost the learning rate for next epoch
	var old_lr = adam_learning_rate
	adam_learning_rate *= achievement_lr_boost
	
	# Log the parameter change
	if enable_training_journey_log:
		log_parameter_change("adam_learning_rate", old_lr, adam_learning_rate, "achievement_reward")
	
	print("   📈 Learning rate boosted by %.1f%% → %.5f" % [(achievement_lr_boost - 1) * 100, adam_learning_rate])
	print("   💪 Dora feels confident and motivated to learn faster!")
	print("   ✅ Next backpropagation will use boosted rate!")

func _apply_escalation_bonus():
	"""
	Apply special training bonuses when targets escalate to new difficulty levels.
	Provides significant learning rate boost and saves milestone weights.
	"""
	print("🚀 ESCALATION BONUS ACTIVATED!")
	
	# Significant learning rate boost (configurable)
	var old_lr = adam_learning_rate
	adam_learning_rate *= escalation_lr_boost
	
	# Log the parameter change
	if enable_training_journey_log:
		log_parameter_change("adam_learning_rate", old_lr, adam_learning_rate, "escalation_bonus")
	
	print("   🎯 Major learning rate boost: %.5f (%.1f%% increase)" % [adam_learning_rate, (escalation_lr_boost - 1) * 100])
	print("   🏆 Dora is ready for greater challenges!")
	print("   ✅ Next backpropagation will use boosted rate!")
	
	# Save this successful state (configurable)
	if save_milestone_weights and neural_network:
		neural_network.save_weights_to_file("dora_milestone_weights.dat")
		print("   💾 Milestone weights saved for safety!")

# ================================
# TRAINING SESSION STATE PERSISTENCE
# ================================

func save_training_session_state():
	"""
	Save comprehensive training session state to disk.
	Persists progressive targets, achievements, and dynamic parameters for session continuity.
	"""
	print("💾 Saving training session state...")
	
	var session_data = {
		# Progressive target state
		"current_target_accuracy": current_target_accuracy,
		"target_achievements": target_achievements,
		"epoch_accuracies": epoch_accuracies,
		
		# Dynamic training parameters (earned through achievements)
		"final_adam_learning_rate": adam_learning_rate,
		"original_adam_learning_rate": 0.002,  # Could make this dynamic too
		
		# Training statistics
		"total_epochs_trained": epoch_accuracies.size(),
		"best_accuracy_achieved": epoch_accuracies.max() if epoch_accuracies.size() > 0 else 0.0,
		"learning_count": learning_count,
		
		# Achievement history
		"achievements_earned": target_achievements,
		"milestones_reached": current_target_accuracy - initial_target_accuracy,
		
		# Session metadata
		"training_completed": true,
		"session_timestamp": Time.get_datetime_string_from_system(),
		"training_samples_used": training_samples,
		"data_augmentation_factor": augmentation_factor
	}
	
	var file = FileAccess.open("dora_training_session.dat", FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(session_data))
		file.close()
		print("✅ Training session state saved!")
		print("   📊 Final LR: %.5f | Target: %.1f%% | Achievements: %d" % 
			[adam_learning_rate, current_target_accuracy, target_achievements])
	else:
		var error = FileAccess.get_open_error()
		print("❌ Failed to save training session state")
		print("   Error code: %d (check permissions and disk space)" % error)

func load_training_session_state():
	"""
	Load previous training session state from disk.
	Restores progressive targets, achievements, and earned learning rate boosts.
	"""
	var session_file = "dora_training_session.dat"
	
	if not FileAccess.file_exists(session_file):
		print("📝 No previous training session found - starting fresh")
		return false
	
	var file = FileAccess.open(session_file, FileAccess.READ)
	if not file:
		var error = FileAccess.get_open_error()
		print("❌ Failed to open training session file: ", session_file)
		print("   Error code: %d" % error)
		return false
	
	var json_string = file.get_as_text()
	file.close()
	
	var json = JSON.new()
	var parse_result = json.parse(json_string)
	
	if parse_result != OK:
		print("❌ Failed to parse training session file")
		return false
	
	var session_data = json.data
	
	if session_data.has("training_completed") and session_data["training_completed"]:
		# Restore progressive target state
		current_target_accuracy = session_data.get("current_target_accuracy", initial_target_accuracy)
		target_achievements = session_data.get("target_achievements", 0)
		var saved_accuracies = session_data.get("epoch_accuracies", [])
		epoch_accuracies.clear()
		for accuracy in saved_accuracies:
			epoch_accuracies.append(float(accuracy))
		
		# Restore earned learning rate (if user wants to continue with boosted rate)
		var saved_lr = session_data.get("final_adam_learning_rate", adam_learning_rate)
		var original_lr = session_data.get("original_adam_learning_rate", 0.002)
		
		# Show what was restored
		print("🎓 Training session state restored!")
		print("   📊 Target Level: %.1f%% | Achievements: %d/%d" % 
			[current_target_accuracy, target_achievements, achievements_needed])
		print("   📈 Earned Learning Rate: %.5f (was %.5f originally)" % [saved_lr, original_lr])
		print("   🏆 Best Accuracy: %.2f%% | Total Epochs: %d" % 
			[session_data.get("best_accuracy_achieved", 0), session_data.get("total_epochs_trained", 0)])
		print("   📚 Interactive Lessons: %d" % session_data.get("learning_count", 0))
		
		# Optionally restore the boosted learning rate
		# (User might want to start fresh or continue with earned boosts)
		if saved_lr > original_lr:
			if restore_earned_learning_rate:
				adam_learning_rate = saved_lr
				print("   🚀 RESTORED earned learning rate boost: %.5f!" % adam_learning_rate)
			else:
				print("   💡 Earned learning rate available: %.5f (enable 'restore_earned_learning_rate' to use)" % saved_lr)
		
		learning_count = session_data.get("learning_count", 0)
		return true
	else:
		print("⚠️ Training session file exists but incomplete")
		return false

# ============================================================================
# TRAINING JOURNEY LOGGING SYSTEM
# ============================================================================

func initialize_training_journey_logging():
	"""
	Initialize comprehensive training journey logging system.
	Creates detailed records of each training session for analysis and portfolio documentation.
	"""
	if not enable_training_journey_log:
		return
		
	print("📊 Initializing Training Journey Logging...")
	
	# Load existing journey log
	load_training_journey_log()
	
	# Determine next session number
	current_session_number = training_journey_log.size() + 1
	
	# Initialize current session data
	current_session_data = {
		"session_number": current_session_number,
		"session_start_time": Time.get_datetime_string_from_system(),
		"session_type": "recognition_training",
		"initial_parameters": get_current_training_parameters(),
		"epochs": [],
		"achievements": [],
		"parameter_changes": [],
		"final_results": {}
	}
	
	print("🔧 Session data structure initialized with all required arrays")
	
	print("📈 Training Session #%d initialized!" % current_session_number)

func get_current_training_parameters() -> Dictionary:
	"""
	Capture current training configuration for logging.
	Returns dictionary of all relevant training hyperparameters and settings.
	"""
	return {
		"training_epochs": training_epochs,
		"training_samples": training_samples,
		"batch_size": batch_size,
		"adam_learning_rate": adam_learning_rate,
		"adam_beta1": adam_beta1,
		"adam_beta2": adam_beta2,
		"initial_target_accuracy": initial_target_accuracy,
		"target_increment": target_increment,
		"max_target_accuracy": max_target_accuracy,
		"achievements_needed": achievements_needed,
		"achievement_lr_boost": achievement_lr_boost,
		"escalation_lr_boost": escalation_lr_boost,
		"augmentation_factor": augmentation_factor,
		"enable_progressive_targets": enable_progressive_targets,
		"enable_achievement_rewards": enable_achievement_rewards,
		"enable_cellular_automata": false
	}

func log_epoch_completion(epoch_number: int, accuracy: float, loss: float, target_accuracy: float):
	"""
	Log detailed metrics from a completed training epoch.
	Records accuracy, loss, targets, and timestamp for historical analysis.
	"""
	if not enable_training_journey_log or not log_epoch_details:
		return
		
	var epoch_data = {
		"epoch_number": epoch_number,
		"accuracy": accuracy,
		"loss": loss,
		"target_accuracy": target_accuracy,
		"target_achieved": accuracy >= target_accuracy,
		"learning_rate": adam_learning_rate,
		"timestamp": Time.get_datetime_string_from_system(),
		"samples_trained": training_samples * augmentation_factor
	}
	
	current_session_data.epochs.append(epoch_data)
	print("📊 Logged Epoch %d: %.2f%% accuracy (target: %.1f%%)" % [epoch_number, accuracy, target_accuracy])

func log_achievement_milestone(milestone_type: String, details: Dictionary):
	"""
	Log achievement milestones and target escalations.
	Tracks when targets are achieved, escalated, or maximum goals reached.
	"""
	if not enable_training_journey_log or not log_achievement_milestones:
		return
		
	var milestone_data = {
		"milestone_type": milestone_type,  # "target_achieved", "target_escalated", "max_target_reached"
		"timestamp": Time.get_datetime_string_from_system(),
		"details": details
	}
	
	# Safety check - ensure achievements array exists
	if not current_session_data.has("achievements"):
		current_session_data.achievements = []
	
	current_session_data.achievements.append(milestone_data)
	print("🏆 Logged Achievement: %s" % milestone_type)

func log_parameter_change(parameter_name: String, old_value: float, new_value: float, reason: String):
	"""
	Log dynamic parameter changes during training.
	Tracks learning rate adjustments and other hyperparameter modifications.
	"""
	if not enable_training_journey_log or not log_parameter_changes:
		return
		
	var change_data = {
		"parameter": parameter_name,
		"old_value": old_value,
		"new_value": new_value,
		"change_ratio": new_value / old_value if old_value > 0 else 0,
		"reason": reason,
		"timestamp": Time.get_datetime_string_from_system()
	}
	
	if not current_session_data.has("parameter_changes"):
		current_session_data.parameter_changes = []
	current_session_data.parameter_changes.append(change_data)
	print("⚙️ Logged Parameter Change: %s %.5f → %.5f (%s)" % [parameter_name, old_value, new_value, reason])

func finalize_training_session(final_accuracy: float, total_epochs: int):
	"""
	Finalize and save the current training session to journey log.
	Compiles final results and appends session to historical training data.
	"""
	if not enable_training_journey_log:
		return
	
	# Safety check - ensure current_session_data is properly initialized
	if not current_session_data.has("achievements"):
		print("⚠️ Session data missing achievements array - initializing")
		current_session_data.achievements = []
	
	if not current_session_data.has("parameter_changes"):
		current_session_data.parameter_changes = []
		
	current_session_data.final_results = {
		"final_accuracy": final_accuracy,
		"total_epochs_completed": total_epochs,
		"total_achievements": current_session_data.achievements.size(),
		"final_learning_rate": adam_learning_rate,
		"session_end_time": Time.get_datetime_string_from_system(),
		"session_duration_minutes": 0.0,  # Could calculate if needed
		"success_level": get_success_level(final_accuracy)
	}
	
	# Add completed session to journey log
	training_journey_log.append(current_session_data.duplicate(true))
	
	# Save to disk
	save_training_journey_log()
	
	print("💾 Training Session #%d finalized: %.2f%% final accuracy" % [current_session_number, final_accuracy])

func get_success_level(accuracy: float) -> String:
	"""
	Categorize training success based on final accuracy.
	Returns performance tier: exceptional, excellent, good, fair, or needs_improvement.
	"""
	if accuracy >= 98.0:
		return "exceptional"
	elif accuracy >= 95.0:
		return "excellent"
	elif accuracy >= 90.0:
		return "good"
	elif accuracy >= 80.0:
		return "fair"
	else:
		return "needs_improvement"

func save_training_journey_log():
	"""
	Save complete training journey to disk as JSON.
	Stores all historical training sessions with metadata and results.
	"""
	var journey_start = ""
	if training_journey_log.size() > 0:
		journey_start = training_journey_log[0].get("session_start_time", "")
	
	var journey_data = {
		"total_sessions": training_journey_log.size(),
		"journey_start_date": journey_start,
		"last_updated": Time.get_datetime_string_from_system(),
		"sessions": training_journey_log
	}
	
	# Store in project directory (next to executable) instead of hidden user folder
	var file = FileAccess.open("dora_training_journey.json", FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(journey_data, "\t"))
		file.close()
		print("💾 Training journey saved: %d sessions recorded" % training_journey_log.size())
	else:
		var error = FileAccess.get_open_error()
		print("❌ Failed to save training journey log")
		print("   Error code: %d (check write permissions)" % error)

func load_training_journey_log():
	"""
	Load existing training journey from disk.
	Restores historical training session data from previous runs.
	"""
	var journey_file = "dora_training_journey.json"
	if not FileAccess.file_exists(journey_file):
		training_journey_log = []
		return
		
	var file = FileAccess.open(journey_file, FileAccess.READ)
	if file:
		var json_string = file.get_as_text()
		file.close()
		
		var json = JSON.new()
		var parse_result = json.parse(json_string)
		
		if parse_result == OK:
			var journey_data = json.data
			var saved_sessions = journey_data.get("sessions", [])
			training_journey_log.clear()
			for session in saved_sessions:
				training_journey_log.append(session as Dictionary)
			print("📊 Loaded training journey: %d previous sessions" % training_journey_log.size())
		else:
			print("❌ Failed to parse training journey log")
			training_journey_log = []
	else:
		print("❌ Failed to load training journey log")
		training_journey_log = []

func get_training_journey_summary() -> String:
	"""
	Get a human-readable summary of the complete training journey.
	Returns formatted string with total sessions, best accuracy, and achievements.
	"""
	if training_journey_log.size() == 0:
		return "No training sessions recorded yet."
	
	var total_sessions = training_journey_log.size()
	var best_accuracy = 0.0
	var total_epochs = 0
	var total_achievements = 0
	
	for session in training_journey_log:
		var final_results = session.get("final_results", {})
		var session_accuracy = final_results.get("final_accuracy", 0.0)
		if session_accuracy > best_accuracy:
			best_accuracy = session_accuracy
		total_epochs += final_results.get("total_epochs_completed", 0)
		total_achievements += final_results.get("total_achievements", 0)
	
	return "Training Journey: %d sessions, %.1f%% best accuracy, %d total epochs, %d achievements" % [
		total_sessions, best_accuracy, total_epochs, total_achievements
	]

func _show_intelligent_feedback(category: String, target: int, predicted: int, confidence: float):
	"""
	Display intelligent feedback UI based on prediction confidence and accuracy.
	Tailors message based on success/failure and confidence level.
	"""
	var message = ""
	
	match category:
		"success_confident":
			message = "🎉 I successfully grew digit %d! I'm %.1f%% confident.\n\nDid I do well?" % [predicted, confidence * 100]
		"success_uncertain":
			message = "🤔 I think I grew digit %d correctly, but I'm only %.1f%% confident.\n\nHow does it look to you?" % [predicted, confidence * 100]
		"failure_confident":
			message = "😅 Oops! You asked for %d, but I'm %.1f%% sure I grew %d instead.\n\nCan you help me understand what went wrong?" % [target, confidence * 100, predicted]
		"failure_uncertain":
			message = "😕 I'm confused... You wanted %d, I think I made %d, but I'm only %.1f%% confident.\n\nWhat do you see?" % [target, predicted, confidence * 100]
	
	# Show the feedback UI with Dora's intelligent question
	var feedback_container = get_node("../FeedbackContainer")
	feedback_container.visible = true
	
	# Update the feedback message (try common label paths)
	var feedback_label = null
	if feedback_container.has_node("Label"):
		feedback_label = feedback_container.get_node("Label")
	elif feedback_container.has_node("FeedbackLabel"):
		feedback_label = feedback_container.get_node("FeedbackLabel")
	
	if feedback_label:
		feedback_label.text = message
	else:
		print("⚠️ Could not find feedback label in FeedbackContainer")
		print("📝 Dora's message: " + message)
