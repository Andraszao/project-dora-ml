extends RefCounted
class_name NeuralNetworkTrainer

## ============================================================================
## NEURAL NETWORK TRAINING SYSTEM
## Version 1.0 | Author: Zeyphrus @ Inchworm Games
## ============================================================================
##
## 🎓 NEW TO TRAINING NEURAL NETWORKS?
## This is where Dora actually LEARNS - backpropagation happens here!
## If you've read NeuralNetwork.gd (forward pass), this is the next step.
##
## Start with these functions:
## 1. train_epoch() - One complete pass through training data
## 2. backpropagate() - The learning algorithm itself
## 3. calculate_loss() - Measures how wrong predictions are
##
## ============================================================================
## KEY TERMS EXPLAINED
## ============================================================================
##
## 🎯 Backpropagation
## The algorithm that teaches the network by working backward from mistakes.
## Steps: 1) Make prediction, 2) Calculate error, 3) Figure out which weights caused it,
## 4) Adjust those weights. It's like tracing a bad outcome back to its causes.
##
## 📉 Gradient Descent
## The method for adjusting weights. "Gradient" = direction of steepest error increase.
## We go the opposite direction (descend) to reduce error. Learning rate controls step size.
##
## 🔄 SGD vs Adam Optimizer
## SGD (Stochastic Gradient Descent): Simple, fixed learning rate for all weights
## Adam (Adaptive Moment): Smart! Gives each weight its own learning rate based on history
## Adam usually trains 2-3x faster and reaches higher accuracy. It "remembers" momentum.
##
## 📦 Mini-Batch Training
## Instead of learning from one image at a time, we process batches (e.g., 100 images).
## Why? Smoother gradients, faster training, better use of CPU. Sweet spot: 50-200 images.
##
## 🎚️ Learning Rate
## How big of a step to take when adjusting weights. Too large = unstable, too small = slow.
## Think: Walking down a hill - big steps are fast but you might overshoot the bottom!
##
## 🛑 Early Stopping
## Stop training when accuracy plateaus or hits target, prevents wasting time and overtraining.
## Overtraining = memorizing training data instead of learning general patterns.
##
## ============================================================================
##
## Implements backpropagation with SGD or Adam optimization for training on MNIST data.
## Includes early stopping, speed optimizations, and progress tracking.

var network: NeuralNetwork
var training_data: Array = []
var training_labels: Array = []
var learning_rate: float = 0.001  # Base learning rate
var dynamic_lr_callback: Callable = Callable()  # Optional callback to get dynamic learning rate

# Adam optimizer variables
var use_adam: bool = false
var adam_beta1: float = 0.9
var adam_beta2: float = 0.999
var adam_epsilon: float = 1e-8
var adam_t: int = 0  # Time step for Adam

# Adam momentum and velocity matrices
var m_weights_input_hidden: Array = []
var v_weights_input_hidden: Array = []
var m_weights_hidden_output: Array = []
var v_weights_hidden_output: Array = []
var m_bias_hidden: Array[float] = []
var v_bias_hidden: Array[float] = []
var m_bias_output: Array[float] = []
var v_bias_output: Array[float] = []

func _init(neural_network: NeuralNetwork):
	"""
	Initialize the trainer with a neural network to train.
	Sets up the training pipeline ready for data loading and optimization.
	"""
	network = neural_network
	print("🏋️ Neural Network Trainer initialized!")

# Track the final accuracy from the last training session
var last_accuracy: float = 0.0

# Early stopping variables
var best_accuracy: float = 0.0
var epochs_without_improvement: int = 0
var early_stopping_enabled: bool = false
var early_stopping_patience: int = 2
var target_threshold: float = 99.0

# Speed optimization variables
var speed_optimizations_enabled: bool = false
var progress_update_frequency: int = 10
var adam_bias_correction_cache: Dictionary = {}  # Cache for pow() calculations

func enable_adam_optimizer(lr: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, lr_callback: Callable = Callable()):
	"""
	Enable Adam optimizer for adaptive learning rate training.
	Adam typically converges faster and achieves better accuracy than standard SGD.
	Parameters: lr (learning rate), beta1 (momentum decay), beta2 (velocity decay), epsilon (numerical stability).
	lr_callback: Optional function that returns current learning rate (for dynamic LR during training).
	"""
	use_adam = true
	learning_rate = lr
	dynamic_lr_callback = lr_callback
	adam_beta1 = beta1
	adam_beta2 = beta2
	adam_epsilon = epsilon
	adam_t = 0
	
	# Initialize Adam momentum and velocity matrices
	initialize_adam_matrices()
	print("🚀 Adam optimizer enabled! LR: %.4f, β1: %.3f, β2: %.3f" % [lr, beta1, beta2])

func enable_early_stopping(patience: int = 2, threshold: float = 99.0):
	"""
	Enable early stopping to prevent overtraining and save time.
	Stops training when accuracy plateaus or reaches target threshold.
	patience: Number of epochs without improvement before stopping.
	threshold: Target accuracy to stop at (e.g., 99.0 = 99%).
	"""
	early_stopping_enabled = true
	early_stopping_patience = patience
	target_threshold = threshold
	best_accuracy = 0.0
	epochs_without_improvement = 0
	print("⏰ Early stopping enabled! Patience: %d epochs, Target: %.1f%%" % [patience, threshold])

func enable_speed_optimizations(update_frequency: int = 20):
	"""
	Enable performance optimizations for faster training
	- Reduces logging frequency
	- Caches expensive calculations (pow functions)
	- Optimizes memory allocations
	"""
	speed_optimizations_enabled = true
	progress_update_frequency = update_frequency
	adam_bias_correction_cache.clear()
	print("🚀 Speed optimizations enabled! Update frequency: every %d batches" % update_frequency)

func initialize_adam_matrices():
	"""
	Initialize Adam momentum and velocity matrices to zero.
	Required before training with Adam optimizer to track gradient statistics.
	Creates separate matrices for each weight and bias parameter.
	"""
	# Input to hidden layer
	m_weights_input_hidden = []
	v_weights_input_hidden = []
	for i in range(network.input_size):
		var m_row = []
		var v_row = []
		for h in range(network.hidden_size):
			m_row.append(0.0)
			v_row.append(0.0)
		m_weights_input_hidden.append(m_row)
		v_weights_input_hidden.append(v_row)
	
	# Hidden to output layer
	m_weights_hidden_output = []
	v_weights_hidden_output = []
	for h in range(network.hidden_size):
		var m_row = []
		var v_row = []
		for o in range(network.output_size):
			m_row.append(0.0)
			v_row.append(0.0)
		m_weights_hidden_output.append(m_row)
		v_weights_hidden_output.append(v_row)
	
	# Biases
	m_bias_hidden = []
	v_bias_hidden = []
	m_bias_output = []
	v_bias_output = []
	for i in range(network.hidden_size):
		m_bias_hidden.append(0.0)
		v_bias_hidden.append(0.0)
	for i in range(network.output_size):
		m_bias_output.append(0.0)
		v_bias_output.append(0.0)

func load_training_data(images: Array, labels: Array):
	"""
	Load training images and labels into the trainer.
	images: Array of 784-element float arrays (28x28 normalized pixels).
	labels: Array of integers (0-9) for corresponding digit labels.
	"""
	training_data = images
	training_labels = labels
	print("📚 Training data loaded: ", training_data.size(), " samples")

func train_epoch(batch_size: int = 100) -> Dictionary:
	"""
	Train for one complete pass through the entire dataset.
	Processes data in batches using mini-batch gradient descent.
	Returns dictionary with accuracy, loss, and statistics for the epoch.
	"""
	print("🚀 Starting training epoch...")
	
	var total_loss = 0.0
	var correct_predictions = 0
	var batches_processed = 0
	
	# Shuffle the training data for better learning
	var indices = range(training_data.size())
	indices.shuffle()
	
	# Process in batches
	for i in range(0, training_data.size(), batch_size):
		var batch_end = min(i + batch_size, training_data.size())
		var batch_loss = 0.0
		var batch_correct = 0
		
		# Process each sample in the batch
		for j in range(i, batch_end):
			var idx = indices[j]
			var image = training_data[idx]
			var label = training_labels[idx]
			
			# Forward pass
			var result = network.predict(image)
			
			# Calculate loss (cross-entropy)
			var loss = calculate_loss(result.probabilities, label)
			batch_loss += loss
			
			# Check if prediction is correct
			if result.prediction == label:
				batch_correct += 1
			
			# Backward pass (update weights)
			backpropagate(image, label, result.probabilities)
		
		total_loss += batch_loss
		correct_predictions += batch_correct
		batches_processed += 1
		
		# Progress update based on optimization settings
		var update_freq = progress_update_frequency if speed_optimizations_enabled else 10
		if batches_processed % update_freq == 0:
			var batch_accuracy = float(batch_correct) / (batch_end - i) * 100.0
			print("   Batch ", batches_processed, ": Accuracy = %.1f%%" % batch_accuracy)
	
	# Calculate final metrics
	var accuracy = float(correct_predictions) / training_data.size() * 100.0
	var avg_loss = total_loss / training_data.size()
	
	print("✅ Epoch complete!")
	print("   📊 Accuracy: %.2f%%" % accuracy)
	print("   📉 Average Loss: %.4f" % avg_loss)
	
	return {
		"accuracy": accuracy,
		"loss": avg_loss,
		"correct": correct_predictions,
		"total": training_data.size()
	}

func calculate_loss(probabilities: Array[float], true_label: int) -> float:
	"""
	Calculate cross-entropy loss for a single prediction.
	Measures how far the predicted probability distribution is from the true label.
	Lower loss = better prediction. Returns scalar loss value.
	"""
	# Avoid log(0) by adding small epsilon
	var epsilon = 1e-15
	var true_prob = max(probabilities[true_label], epsilon)
	return -log(true_prob)

func backpropagate(input_data: Array[float], true_label: int, probabilities: Array[float]):
	"""
	Backpropagation algorithm - the core of neural network learning.
	Calculates gradients of loss with respect to all weights and biases.
	Updates network parameters to reduce prediction error.
	"""
	
	# Update learning rate from dynamic callback if provided
	if dynamic_lr_callback.is_valid():
		learning_rate = dynamic_lr_callback.call()
	
	if use_adam:
		adam_t += 1  # Increment Adam time step
	
	# Create target output (one-hot encoding)
	var target: Array[float] = []
	for i in range(10):
		target.append(1.0 if i == true_label else 0.0)
	
	# Calculate output layer errors
	var output_errors: Array[float] = []
	for i in range(10):
		output_errors.append(probabilities[i] - target[i])
	
	# Update output layer weights and biases
	update_output_layer(output_errors)
	
	# Calculate hidden layer errors
	var hidden_errors = calculate_hidden_errors(output_errors)
	
	# Update hidden layer weights and biases
	update_hidden_layer(hidden_errors, input_data)

func update_output_layer(output_errors: Array[float]):
	"""
	Update output layer weights and biases based on computed errors.
	Routes to either SGD or Adam update depending on optimizer selection.
	"""
	
	# Get hidden layer activations from last forward pass
	var hidden_activations = network.last_hidden_activations
	
	if use_adam:
		# Adam optimizer update for output layer
		update_output_layer_adam(output_errors, hidden_activations)
	else:
		# Standard SGD update for output layer
		update_output_layer_sgd(output_errors, hidden_activations)

func update_output_layer_sgd(output_errors: Array[float], hidden_activations: Array[float]):
	"""
	Stochastic Gradient Descent update for output layer.
	Simple but effective: subtract learning_rate * gradient from each weight.
	Includes gradient clipping to prevent instability.
	"""
	# Update weights: weight -= learning_rate * error * hidden_activation
	for h in range(network.hidden_size):
		for o in range(network.output_size):
			var weight_delta = learning_rate * output_errors[o] * hidden_activations[h]
			# Gradient clipping to prevent explosion
			weight_delta = clamp(weight_delta, -1.0, 1.0)
			network.weights_hidden_output[h][o] -= weight_delta
	
	# Update biases: bias -= learning_rate * error
	for o in range(network.output_size):
		var bias_delta = learning_rate * output_errors[o]
		bias_delta = clamp(bias_delta, -1.0, 1.0)
		network.bias_output[o] -= bias_delta

func update_output_layer_adam(output_errors: Array[float], hidden_activations: Array[float]):
	"""
	Adam optimizer update for output layer.
	Uses adaptive learning rates per-parameter based on gradient momentum and velocity.
	Typically converges faster than SGD and handles sparse gradients better.
	"""
	
	# Cache bias correction factors to avoid repeated pow() calculations
	var bias_correction_m: float
	var bias_correction_v: float
	
	if speed_optimizations_enabled and adam_bias_correction_cache.has(adam_t):
		var cached = adam_bias_correction_cache[adam_t]
		bias_correction_m = cached["m"]
		bias_correction_v = cached["v"]
	else:
		bias_correction_m = 1.0 - pow(adam_beta1, adam_t)
		bias_correction_v = 1.0 - pow(adam_beta2, adam_t)
		if speed_optimizations_enabled:
			adam_bias_correction_cache[adam_t] = {"m": bias_correction_m, "v": bias_correction_v}
	
	# Update weights with Adam
	for h in range(network.hidden_size):
			for o in range(network.output_size):
				var gradient = output_errors[o] * hidden_activations[h]
				
				# Update momentum and velocity
				m_weights_hidden_output[h][o] = adam_beta1 * m_weights_hidden_output[h][o] + (1.0 - adam_beta1) * gradient
				v_weights_hidden_output[h][o] = adam_beta2 * v_weights_hidden_output[h][o] + (1.0 - adam_beta2) * gradient * gradient
				
				# Bias correction (using cached values)
				var m_corrected = m_weights_hidden_output[h][o] / bias_correction_m
				var v_corrected = v_weights_hidden_output[h][o] / bias_correction_v
				
				# Update weight
				var weight_update = learning_rate * m_corrected / (sqrt(v_corrected) + adam_epsilon)
				weight_update = clamp(weight_update, -1.0, 1.0)  # Gradient clipping
				network.weights_hidden_output[h][o] -= weight_update
	
	# Update biases with Adam (using same cached bias correction)
	for o in range(network.output_size):
		var gradient = output_errors[o]
		
		# Update momentum and velocity
		m_bias_output[o] = adam_beta1 * m_bias_output[o] + (1.0 - adam_beta1) * gradient
		v_bias_output[o] = adam_beta2 * v_bias_output[o] + (1.0 - adam_beta2) * gradient * gradient
		
		# Bias correction (using cached values)
		var m_corrected = m_bias_output[o] / bias_correction_m
		var v_corrected = v_bias_output[o] / bias_correction_v
		
		# Update bias
		var bias_update = learning_rate * m_corrected / (sqrt(v_corrected) + adam_epsilon)
		bias_update = clamp(bias_update, -1.0, 1.0)  # Gradient clipping
		network.bias_output[o] -= bias_update

func calculate_hidden_errors(output_errors: Array[float]) -> Array[float]:
	"""
	Calculate error gradients for hidden layer using backpropagation.
	Propagates errors from output layer back through network connections.
	Applies derivative of ReLU activation function.
	"""
	var hidden_errors: Array[float] = []
	
	for h in range(network.hidden_size):
		var error = 0.0
		
		# Sum up weighted errors from output layer
		for o in range(network.output_size):
			error += output_errors[o] * network.weights_hidden_output[h][o]
		
		# Apply derivative of ReLU activation
		var hidden_activation = network.last_hidden_activations[h]
		var relu_derivative = 1.0 if hidden_activation > 0 else 0.0
		error *= relu_derivative
		
		hidden_errors.append(error)
	
	return hidden_errors

func update_hidden_layer(hidden_errors: Array[float], input_data: Array[float]):
	"""
	Update hidden layer weights and biases based on computed errors.
	Routes to either SGD or Adam update depending on optimizer selection.
	"""
	
	if use_adam:
		# Adam optimizer update for hidden layer
		update_hidden_layer_adam(hidden_errors, input_data)
	else:
		# Standard SGD update for hidden layer
		update_hidden_layer_sgd(hidden_errors, input_data)

func update_hidden_layer_sgd(hidden_errors: Array[float], input_data: Array[float]):
	"""
	Stochastic Gradient Descent update for hidden layer.
	Updates input-to-hidden weights and hidden biases.
	Includes gradient clipping for training stability.
	"""
	# Update weights: weight -= learning_rate * error * input
	for i in range(network.input_size):
		for h in range(network.hidden_size):
			var weight_delta = learning_rate * hidden_errors[h] * input_data[i]
			# Gradient clipping to prevent explosion
			weight_delta = clamp(weight_delta, -1.0, 1.0)
			network.weights_input_hidden[i][h] -= weight_delta
	
	# Update biases: bias -= learning_rate * error
	for h in range(network.hidden_size):
		var bias_delta = learning_rate * hidden_errors[h]
		bias_delta = clamp(bias_delta, -1.0, 1.0)
		network.bias_hidden[h] -= bias_delta

func update_hidden_layer_adam(hidden_errors: Array[float], input_data: Array[float]):
	"""
	Adam optimizer update for hidden layer.
	Updates input-to-hidden weights and hidden biases with adaptive learning rates.
	Uses cached bias correction to avoid redundant calculations.
	"""
	
	# Use cached bias correction factors (already calculated in output layer)
	var bias_correction_m: float
	var bias_correction_v: float
	
	if speed_optimizations_enabled and adam_bias_correction_cache.has(adam_t):
		var cached = adam_bias_correction_cache[adam_t]
		bias_correction_m = cached["m"]
		bias_correction_v = cached["v"]
	else:
		bias_correction_m = 1.0 - pow(adam_beta1, adam_t)
		bias_correction_v = 1.0 - pow(adam_beta2, adam_t)
		if speed_optimizations_enabled:
			adam_bias_correction_cache[adam_t] = {"m": bias_correction_m, "v": bias_correction_v}
	
	# Update weights with Adam
	for i in range(network.input_size):
		for h in range(network.hidden_size):
			var gradient = hidden_errors[h] * input_data[i]
			
			# Update momentum and velocity
			m_weights_input_hidden[i][h] = adam_beta1 * m_weights_input_hidden[i][h] + (1.0 - adam_beta1) * gradient
			v_weights_input_hidden[i][h] = adam_beta2 * v_weights_input_hidden[i][h] + (1.0 - adam_beta2) * gradient * gradient
			
			# Bias correction (using cached values)
			var m_corrected = m_weights_input_hidden[i][h] / bias_correction_m
			var v_corrected = v_weights_input_hidden[i][h] / bias_correction_v
			
			# Update weight
			var weight_update = learning_rate * m_corrected / (sqrt(v_corrected) + adam_epsilon)
			weight_update = clamp(weight_update, -1.0, 1.0)  # Gradient clipping
			network.weights_input_hidden[i][h] -= weight_update
	
	# Update biases with Adam (using same cached bias correction)
	for h in range(network.hidden_size):
		var gradient = hidden_errors[h]
		
		# Update momentum and velocity
		m_bias_hidden[h] = adam_beta1 * m_bias_hidden[h] + (1.0 - adam_beta1) * gradient
		v_bias_hidden[h] = adam_beta2 * v_bias_hidden[h] + (1.0 - adam_beta2) * gradient * gradient
		
		# Bias correction (using cached values)
		var m_corrected = m_bias_hidden[h] / bias_correction_m
		var v_corrected = v_bias_hidden[h] / bias_correction_v
		
		# Update bias
		var bias_update = learning_rate * m_corrected / (sqrt(v_corrected) + adam_epsilon)
		bias_update = clamp(bias_update, -1.0, 1.0)  # Gradient clipping
		network.bias_hidden[h] -= bias_update

func train_multiple_epochs(num_epochs: int = 5, batch_size: int = 100, progressive_callback: Callable = Callable()):
	"""
	Train the neural network for multiple complete passes through the dataset.
	Includes progress tracking, early stopping, and optional progressive target evaluation.
	progressive_callback: Optional function called after each epoch for dynamic training strategies.
	Returns array of training history dictionaries with accuracy and loss per epoch.
	"""
	print("🎯 Starting training: ", num_epochs, " epochs")
	
	var training_history = []
	
	for epoch in range(num_epochs):
		print("\n📚 EPOCH ", epoch + 1, "/", num_epochs)
		var epoch_result = train_epoch(batch_size)
		training_history.append(epoch_result)
		
		# Show progress
		print("🏆 Epoch ", epoch + 1, " Results:")
		print("   Accuracy: %.2f%% (%d/%d correct)" % [epoch_result.accuracy, epoch_result.correct, epoch_result.total])
		
		# Store current accuracy for progressive targets
		last_accuracy = epoch_result.accuracy
		
		# Call progressive target evaluation after each epoch if provided
		if progressive_callback.is_valid():
			var should_continue = progressive_callback.call(epoch_result.accuracy, epoch + 1)
			if not should_continue:
				print("🎯 Progressive targets indicate training should stop!")
				break
		
		# Early stopping logic
		if early_stopping_enabled:
			if epoch_result.accuracy > best_accuracy:
				best_accuracy = epoch_result.accuracy
				epochs_without_improvement = 0
				print("🎯 New best accuracy: %.2f%%!" % best_accuracy)
			else:
				epochs_without_improvement += 1
				print("📊 No improvement for %d/%d epochs" % [epochs_without_improvement, early_stopping_patience])
			
			# Check stopping conditions
			if epoch_result.accuracy >= target_threshold:
				print("🏆 TARGET ACCURACY REACHED! %.2f%% >= %.1f%%" % [epoch_result.accuracy, target_threshold])
				print("🎉 Training complete - target achieved!")
				break
			elif epochs_without_improvement >= early_stopping_patience:
				print("⏰ EARLY STOPPING: No improvement for %d epochs" % early_stopping_patience)
				print("🎯 Best accuracy achieved: %.2f%%" % best_accuracy)
				break
		else:
			# Legacy early stopping for excellent performance
			if epoch_result.accuracy > 95.0:
				print("🎉 Excellent accuracy reached! Stopping early.")
				break
	
	print("\n✅ Training complete!")
	
	# Show final progress
	if training_history.size() > 0:
		var final_result = training_history[-1]
		last_accuracy = final_result.accuracy  # Store for progressive targets
		print("🏆 Final Results:")
		print("   📊 Accuracy: %.2f%%" % final_result.accuracy)
		print("   🎯 Dora is now trained on real MNIST data!")
	
	return training_history
