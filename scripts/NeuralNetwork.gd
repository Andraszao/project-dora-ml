extends RefCounted
class_name NeuralNetwork

## ============================================================================
## NEURAL NETWORK CORE ARCHITECTURE
## Version 1.0 | Author: Zeyphrus @ Inchworm Games
## ============================================================================
##
## 🎓 NEW TO NEURAL NETWORKS?
## This file is the heart of Dora's learning system. If you're just starting:
## 1. Read the "What You'll Learn" section in README.md first
## 2. Start by reading the predict() function (line ~87) - that's the forward pass
## 3. The math happens in initialize_weights() and the activation functions
## 4. Training happens in NeuralNetworkTrainer.gd (backpropagation lives there)
##
## Key concepts you'll encounter here:
## • Forward Propagation: How input pixels become predictions (predict function)
## • Activation Functions: ReLU and Softmax transform neuron outputs
## • Weight Initialization: Xavier method prevents gradient problems
## • Network Architecture: Input → Hidden → Output layer structure
##
## ============================================================================
## KEY TERMS EXPLAINED
## ============================================================================
##
## 🔥 ReLU (Rectified Linear Unit)
## Think of it as a "keep positive numbers, zero out negatives" filter.
## Formula: relu(x) = max(0, x)
## Why? It lets the network learn complex patterns while being super fast to compute.
## Without it, stacking layers wouldn't add learning power. It's the secret sauce!
##
## 📊 Softmax
## Converts raw scores into probabilities that sum to 100%.
## Example: Raw outputs [2.3, 1.1, 0.5] → Probabilities [70%, 20%, 10%]
## Why? We want "I'm 85% sure it's a 3" not just "score of 2.8 for digit 3"
## It's how we get confidence levels for each digit prediction.
##
## ⚖️ Xavier Initialization
## A smart way to set initial random weights (not too big, not too small).
## Formula: random weights scaled by √(2/number_of_inputs)
## Why? Too large = gradients explode (learning fails), too small = gradients vanish
## Think of it as starting a pendulum with the "just right" push - not too hard, not too soft.
##
## 📐 Forward Propagation
## The journey of data through the network: Pixels → Hidden Layer → Predictions
## It's called "forward" because data flows one direction (no loops or cycles).
## This happens every time you click "Test" - it's how Dora makes a guess!
##
## 🔄 Backpropagation (happens in NeuralNetworkTrainer.gd)
## The "learning" part - adjusts weights based on mistakes.
## If Dora guesses wrong, backprop figures out which weights to tweak and by how much.
## It flows backward through the network (output → hidden → input), hence the name.
##
## ============================================================================
## 
## This is Dora's "brain" - a feedforward neural network designed specifically 
## for handwritten digit recognition. The architecture is carefully chosen to 
## balance accuracy, speed, and memory efficiency for real-time interaction.
##
## ARCHITECTURE DESIGN:
## 784 input neurons  → 28x28 pixel grid (MNIST standard)
## 128 hidden neurons → Sweet spot for digit complexity without overfitting
## 10 output neurons  → One for each digit (0-9) with softmax probability
##
## MATHEMATICAL FOUNDATION:
## Uses ReLU activation (hidden layer) and Softmax (output) with Xavier 
## initialization for stable gradient flow during backpropagation training.
##
## WANT TO LEARN MORE?
## • Neural network basics: See README.md "What You'll Learn" section
## • How training works: Check NeuralNetworkTrainer.gd for backpropagation
## • Change the architecture: Modify hidden_size below and experiment!
## • Visualize the weights: Try printing weights_input_hidden to see learned patterns

# Version information for model compatibility
const MODEL_VERSION = "1.0.0"
const COMPATIBLE_VERSIONS = ["1.0.0"]  # Versions this code can load

# Network layer sizes - optimized for digit recognition
var input_size = 784    # 28x28 pixel flattened array (standard MNIST format)
var hidden_size = 128   # Hidden layer size - balance between capacity and efficiency
var output_size = 10    # Output classes for digits 0-9

# Neural network parameters (weights and biases)
# These are the "learned knowledge" that gets saved/loaded between sessions
var weights_input_hidden: Array[Array] = []    # 784x128 matrix: input→hidden connections
var weights_hidden_output: Array[Array] = []   # 128x10 matrix: hidden→output connections
var bias_hidden: Array[float] = []             # 128 values: hidden layer biases
var bias_output: Array[float] = []             # 10 values: output layer biases

# Intermediate computation storage
# Needed for backpropagation algorithm to calculate gradients
var last_hidden_activations: Array[float] = [] # Stores hidden layer outputs for gradient calculation

func _init():
	"""
	Initialize the neural network with random weights.
	Sets up the complete architecture ready for training or loading saved weights.
	"""
	print("🧠 Initializing neural network...")
	initialize_weights()

func initialize_weights():
	"""
	Initialize neural network weights using Xavier/Glorot initialization
	
	WHY XAVIER INITIALIZATION?
	Random weights are crucial for breaking symmetry and enabling learning.
	Xavier initialization scales weights based on layer size to:
	- Prevent vanishing gradients (weights too small)
	- Prevent exploding gradients (weights too large)
	- Maintain stable signal flow through the network
	
	FORMULA: weight = random(-√(2/n), √(2/n)) where n = input layer size
	This ensures gradients stay in a healthy range during backpropagation.
	"""
	# Calculate Xavier scaling factors for each layer transition
	var xavier_input = sqrt(2.0 / float(input_size))    # Input→Hidden scaling
	var xavier_hidden = sqrt(2.0 / float(hidden_size))  # Hidden→Output scaling
	
	# Input to hidden weights [784 x 128]
	weights_input_hidden.clear()
	for i in range(input_size):
		var row: Array[float] = []
		for j in range(hidden_size):
			row.append(randf_range(-xavier_input, xavier_input))
		weights_input_hidden.append(row)
	
	# Hidden to output weights [128 x 10]
	weights_hidden_output.clear()
	for i in range(hidden_size):
		var row: Array[float] = []
		for j in range(output_size):
			row.append(randf_range(-xavier_hidden, xavier_hidden))
		weights_hidden_output.append(row)
	
	# Biases (start at zero)
	bias_hidden.clear()
	bias_output.clear()
	for i in range(hidden_size):
		bias_hidden.append(0.0)
	for i in range(output_size):
		bias_output.append(0.0)
	
	print("✅ Network initialized with Xavier weights (xavier_input=%.4f, xavier_hidden=%.4f)" % [xavier_input, xavier_hidden])

func predict(input_pixels: Array[float]) -> Dictionary:
	"""
	Run forward pass through network to make a prediction.
	Takes 784 normalized pixel values and returns predicted digit with confidence.
	Returns dict with keys: "prediction" (0-9), "confidence" (0.0-1.0), "probabilities" (array of 10 floats).
	"""
	if input_pixels.size() != input_size:
		print("❌ Error: Expected %d pixels, got %d" % [input_size, input_pixels.size()])
		return {"prediction": -1, "confidence": 0.0, "probabilities": []}
	
	# Forward pass to hidden layer
	var hidden_layer: Array[float] = []
	for h in range(hidden_size):
		var sum = bias_hidden[h]
		for i in range(input_size):
			sum += input_pixels[i] * weights_input_hidden[i][h]
		hidden_layer.append(relu(sum))
	
	# Store hidden activations for backpropagation
	last_hidden_activations = hidden_layer.duplicate()
	
	# Forward pass to output layer
	var output_layer: Array[float] = []
	for o in range(output_size):
		var sum = bias_output[o]
		for h in range(hidden_size):
			sum += hidden_layer[h] * weights_hidden_output[h][o]
		output_layer.append(sum)
	
	# Apply softmax to get probabilities
	var probabilities = softmax(output_layer)
	
	# Find the prediction (highest probability)
	var prediction = 0
	var max_prob = probabilities[0]
	for i in range(1, output_size):
		if probabilities[i] > max_prob:
			max_prob = probabilities[i]
			prediction = i
	
	return {
		"prediction": prediction,
		"confidence": max_prob,
		"probabilities": probabilities
	}

func relu(x: float) -> float:
	"""
	ReLU (Rectified Linear Unit) activation function.
	Returns x if positive, 0 otherwise. Prevents vanishing gradients in deep networks.
	"""
	return max(0.0, x)

func softmax(values: Array[float]) -> Array[float]:
	"""
	Softmax activation converts raw output scores to probabilities.
	Returns array of 10 values that sum to 1.0, representing confidence for each digit.
	"""
	var max_val = values.max()
	var exp_sum = 0.0
	var exp_values: Array[float] = []
	
	# Subtract max for numerical stability
	for val in values:
		var exp_val = exp(val - max_val)
		exp_values.append(exp_val)
		exp_sum += exp_val
	
	# Normalize
	var result: Array[float] = []
	for exp_val in exp_values:
		result.append(exp_val / exp_sum)
	
	return result

func load_pretrained_weights():
	"""
	Load pre-trained heuristic-based weights for digit recognition.
	Uses pattern-based initialization from MNISTWeights class.
	Note: These are not trained on real data, just educated guesses.
	"""
	print("🔄 Loading pre-trained MNIST weights...")
	
	var mnist_weights = MNISTWeights.get_pretrained_weights()
	
	# Replace our random weights with the pre-trained ones
	weights_input_hidden = mnist_weights["input_to_hidden"]
	weights_hidden_output = mnist_weights["hidden_to_output"]
	bias_hidden = mnist_weights["hidden_bias"]
	bias_output = mnist_weights["output_bias"]
	
	print("✅ MNIST weights loaded successfully!")
	print("📊 Network ready for digit recognition")

func save_weights_to_file(file_path: String = "dora_trained_weights.dat"):
	"""
	Save trained neural network weights to disk for persistence.
	Allows Dora to remember learned knowledge across sessions.
	Saves as JSON file with weights, biases, and metadata.
	"""
	print("💾 Saving Dora's memory to: ", file_path)
	
	var save_data = {
		"model_version": MODEL_VERSION,
		"architecture": {
			"input_size": input_size,
			"hidden_size": hidden_size,
			"output_size": output_size
		},
		"weights_input_hidden": weights_input_hidden,
		"weights_hidden_output": weights_hidden_output,
		"bias_hidden": bias_hidden,
		"bias_output": bias_output,
		"trained": true,
		"timestamp": Time.get_datetime_string_from_system()
	}
	
	var file = FileAccess.open(file_path, FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(save_data))
		file.close()
		print("✅ Dora's memory saved! She'll remember next time.")
		print("   📦 Model version: %s | Size: %d parameters" % [MODEL_VERSION, (input_size * hidden_size + hidden_size * output_size)])
	else:
		var error = FileAccess.get_open_error()
		print("❌ Failed to save weights to: ", file_path)
		print("   Error code: %d" % error)
		print("   Possible causes: Insufficient permissions, disk full, invalid path")

func load_weights_from_file(file_path: String = "dora_trained_weights.dat") -> bool:
	"""
	Load previously trained weights from disk.
	Restores Dora's learned knowledge from a previous session.
	Returns true if weights loaded successfully, false otherwise.
	"""
	print("🧠 Checking if Dora has memories at: ", file_path)
	
	if not FileAccess.file_exists(file_path):
		print("📝 No previous memories found - Dora will learn from scratch")
		return false
	
	var file = FileAccess.open(file_path, FileAccess.READ)
	if not file:
		var error = FileAccess.get_open_error()
		print("❌ Failed to open memory file: ", file_path)
		print("   Error code: %d" % error)
		return false
	
	var json_string = file.get_as_text()
	file.close()
	
	var json = JSON.new()
	var parse_result = json.parse(json_string)
	
	if parse_result != OK:
		print("❌ Failed to parse memory file")
		return false
	
	var save_data = json.data
	
	# Check model version compatibility
	var file_version = save_data.get("model_version", "0.0.0")
	if not file_version in COMPATIBLE_VERSIONS:
		print("⚠️ WARNING: Weight file version %s may not be compatible with current version %s" % [file_version, MODEL_VERSION])
		print("   Attempting to load anyway, but accuracy may be affected...")
	
	# Verify architecture matches
	if save_data.has("architecture"):
		var arch = save_data["architecture"]
		if arch.get("input_size", 784) != input_size or arch.get("hidden_size", 128) != hidden_size or arch.get("output_size", 10) != output_size:
			print("❌ ERROR: Saved model architecture doesn't match current network!")
			print("   Saved: %dx%dx%d | Current: %dx%dx%d" % [arch.get("input_size", 0), arch.get("hidden_size", 0), arch.get("output_size", 0), input_size, hidden_size, output_size])
			print("   Cannot load incompatible weights.")
			return false
	
	if save_data.has("trained") and save_data["trained"]:
		# Convert loaded arrays back to proper typed arrays
		weights_input_hidden = []
		for row in save_data["weights_input_hidden"]:
			var typed_row: Array[float] = []
			for val in row:
				typed_row.append(float(val))
			weights_input_hidden.append(typed_row)
		
		weights_hidden_output = []
		for row in save_data["weights_hidden_output"]:
			var typed_row: Array[float] = []
			for val in row:
				typed_row.append(float(val))
			weights_hidden_output.append(typed_row)
		
		bias_hidden = []
		for val in save_data["bias_hidden"]:
			bias_hidden.append(float(val))
		
		bias_output = []
		for val in save_data["bias_output"]:
			bias_output.append(float(val))
		
		print("🎉 Dora remembers! Loaded memories from: ", save_data.get("timestamp", "unknown time"))
		return true
	else:
		print("⚠️ Memory file exists but no training data found")
		return false

func _create_simple_pattern_weights():
	"""
	Create simple pattern-based weights for basic digit recognition.
	Uses heuristics to enhance certain pixel patterns for each digit.
	This is a fallback method; real training produces better results.
	"""
	print("🎯 Creating pattern-based weights...")
	
	# This is a simplified approach - in reality we'd use trained MNIST weights
	# For now, let's create weights that respond to basic patterns
	
	# Clear existing weights
	initialize_weights()
	
	# Enhance weights for certain pixel positions that are important for digits
	# This is a very basic heuristic approach
	for digit in range(10):
		_enhance_digit_pattern(digit)
	
	print("✅ Simple pattern weights created")

func _enhance_digit_pattern(digit: int):
	"""
	Enhance weights to favor specific patterns for a given digit.
	Applies heuristic weight boosts to pixels commonly associated with each digit shape.
	"""
	# This is a very simplified pattern recognition
	# Real training learns these patterns automatically from data
	
	match digit:
		0: # Circle pattern
			_enhance_edge_pixels()
		1: # Vertical line
			_enhance_center_vertical()
		2: # Horizontal lines
			_enhance_horizontal_lines()
		# Add more patterns as needed
		_:
			pass # Use random weights for other digits

func _enhance_edge_pixels():
	"""
	Enhance weights for pixels near the edges of the 28x28 grid.
	Useful for detecting circular digits like 0, 6, 8, and 9.
	"""
	for y in range(28):
		for x in range(28):
			if x == 0 or x == 27 or y == 0 or y == 27:
				var pixel_index = y * 28 + x
				for h in range(min(10, hidden_size)):  # Enhance first 10 hidden neurons
					weights_input_hidden[pixel_index][h] += 0.1

func _enhance_center_vertical():
	"""
	Enhance weights for vertical center columns.
	Primarily useful for recognizing digit 1 and the vertical stem of 4.
	"""
	for y in range(28):
		for x in range(12, 16):  # Center columns
			var pixel_index = y * 28 + x
			for h in range(10, min(20, hidden_size)):  # Different hidden neurons
				weights_input_hidden[pixel_index][h] += 0.1

func _enhance_horizontal_lines():
	"""
	Enhance weights for horizontal lines at top, middle, and bottom.
	Useful for digits with prominent horizontal features like 2, 3, 5, and 7.
	"""
	for y in [0, 13, 27]:  # Top, middle, bottom rows
		for x in range(28):
			var pixel_index = y * 28 + x
			for h in range(20, min(30, hidden_size)):  # Different hidden neurons
				weights_input_hidden[pixel_index][h] += 0.1
