extends RefCounted
class_name MNISTWeights

## Pre-trained Weight Generator
## Creates hand-crafted weights based on digit pattern heuristics.
## Used as a fallback or baseline before full training on MNIST data.

static func get_pretrained_weights() -> Dictionary:
	"""
	Generate pre-computed weights for basic digit recognition.
	Returns dictionary with weight matrices and bias vectors for neural network initialization.
	Note: These are heuristic-based, not trained on actual data.
	"""
	
	# Hand-crafted weights to recognize basic digit patterns
	# Real MNIST training learns better representations from 60,000 examples
	
	var weights = {
		"input_to_hidden": create_input_hidden_weights(),
		"hidden_to_output": create_hidden_output_weights(),
		"hidden_bias": create_hidden_bias(),
		"output_bias": create_output_bias()
	}
	
	print("📚 Loaded pre-trained MNIST-style weights")
	return weights

static func create_input_hidden_weights() -> Array[Array]:
	"""
	Create 784x128 input-to-hidden weight matrix with pattern-based initialization.
	Starts with random values, then enhances specific patterns for each digit type.
	Returns 2D array where weights[pixel_index][hidden_neuron] represents connection strength.
	"""
	var weights: Array[Array] = []
	
	# Initialize with small random values
	for i in range(784):  # 28x28 pixels
		var row: Array[float] = []
		for j in range(128):  # Hidden neurons
			row.append(randf_range(-0.1, 0.1))
		weights.append(row)
	
	# Enhance specific patterns for each digit type
	enhance_digit_patterns(weights)
	
	return weights

static func enhance_digit_patterns(weights: Array[Array]):
	"""
	Enhance weight matrix with digit-specific pattern detectors.
	Assigns different hidden neurons to detect features of each digit (0-9).
	Modifies weights in-place to favor patterns like edges, loops, and lines.
	"""
	
	# Digit 0: Enhance edge detection
	for y in range(28):
		for x in range(28):
			var pixel_idx = y * 28 + x
			# Edge pixels get higher weights for neurons 0-9 (digit 0 detectors)
			if is_edge_pixel(x, y):
				for h in range(10):
					weights[pixel_idx][h] += 0.3
	
	# Digit 1: Enhance vertical center line
	for y in range(28):
		for x in range(12, 16):  # Center columns
			var pixel_idx = y * 28 + x
			# Center vertical gets higher weights for neurons 10-19 (digit 1 detectors)
			for h in range(10, 20):
				weights[pixel_idx][h] += 0.4
	
	# Digit 2: Enhance horizontal lines
	enhance_horizontal_pattern(weights, [2, 14, 25], 20, 30)  # Top, middle, bottom
	
	# Digit 3: Enhance right side and curves
	for y in range(28):
		for x in range(18, 28):  # Right side
			var pixel_idx = y * 28 + x
			for h in range(30, 40):
				weights[pixel_idx][h] += 0.3
	
	# Digit 4: Enhance vertical line + horizontal crossbar
	enhance_vertical_pattern(weights, 40, 50)
	enhance_horizontal_pattern(weights, [14], 45, 50)  # Middle crossbar
	
	# Digit 5: Similar to 2 but different proportions
	enhance_horizontal_pattern(weights, [2, 14, 25], 50, 60)
	
	# Digit 6: Enhance bottom loop
	enhance_bottom_loop_pattern(weights, 60, 70)
	
	# Digit 7: Enhance top horizontal + diagonal
	enhance_horizontal_pattern(weights, [2], 70, 80)  # Top line
	enhance_diagonal_pattern(weights, 75, 80)
	
	# Digit 8: Enhance double loop patterns
	enhance_double_loop_pattern(weights, 80, 90)
	
	# Digit 9: Enhance top loop
	enhance_top_loop_pattern(weights, 90, 100)

static func is_edge_pixel(x: int, y: int) -> bool:
	"""
	Check if pixel is near the edge of the 28x28 grid.
	Useful for detecting circular digits like 0, 6, 8, and 9.
	"""
	return x < 4 or x > 23 or y < 4 or y > 23

static func enhance_horizontal_pattern(weights: Array[Array], rows: Array, start_neuron: int, end_neuron: int):
	"""
	Enhance weights for detecting horizontal lines at specific rows.
	Useful for digits with strong horizontal features (2, 5, 7).
	"""
	for row in rows:
		for x in range(28):
			var pixel_idx = row * 28 + x
			for h in range(start_neuron, min(end_neuron, 128)):
				weights[pixel_idx][h] += 0.25

static func enhance_vertical_pattern(weights: Array[Array], start_neuron: int, end_neuron: int):
	"""
	Enhance weights for detecting vertical center line.
	Primarily useful for recognizing digit 1 and the stem of 4.
	"""
	for y in range(28):
		for x in range(12, 16):  # Center columns
			var pixel_idx = y * 28 + x
			for h in range(start_neuron, min(end_neuron, 128)):
				weights[pixel_idx][h] += 0.3

static func enhance_diagonal_pattern(weights: Array[Array], start_neuron: int, end_neuron: int):
	"""
	Enhance weights for detecting diagonal lines.
	Specifically designed for digit 7's characteristic diagonal stroke.
	"""
	for i in range(28):
		var x = 20 - i  # Diagonal from top-right to bottom-left
		var y = i
		if x >= 0 and x < 28 and y >= 0 and y < 28:
			var pixel_idx = y * 28 + x
			for h in range(start_neuron, min(end_neuron, 128)):
				weights[pixel_idx][h] += 0.2

static func enhance_bottom_loop_pattern(weights: Array[Array], start_neuron: int, end_neuron: int):
	"""
	Enhance weights for detecting bottom loop patterns.
	Designed for digit 6 which has a prominent loop in the lower half.
	"""
	for y in range(14, 28):  # Bottom half
		for x in range(28):
			var pixel_idx = y * 28 + x
			# Enhance edge pixels more
			var weight_boost = 0.15 if is_edge_pixel(x, y - 14) else 0.05
			for h in range(start_neuron, min(end_neuron, 128)):
				weights[pixel_idx][h] += weight_boost

static func enhance_top_loop_pattern(weights: Array[Array], start_neuron: int, end_neuron: int):
	"""
	Enhance weights for detecting top loop patterns.
	Designed for digit 9 which has a prominent loop in the upper half.
	"""
	for y in range(0, 18):  # Top portion
		for x in range(28):
			var pixel_idx = y * 28 + x
			var weight_boost = 0.15 if is_edge_pixel(x, y) else 0.05
			for h in range(start_neuron, min(end_neuron, 128)):
				weights[pixel_idx][h] += weight_boost

static func enhance_double_loop_pattern(weights: Array[Array], start_neuron: int, end_neuron: int):
	"""
	Enhance weights for detecting double loop patterns.
	Designed for digit 8 which has two stacked loops (top and bottom).
	"""
	# Top loop
	for y in range(0, 14):
		for x in range(28):
			if is_edge_pixel(x, y):
				var pixel_idx = y * 28 + x
				for h in range(start_neuron, min(end_neuron, 128)):
					weights[pixel_idx][h] += 0.1
	
	# Bottom loop
	for y in range(14, 28):
		for x in range(28):
			if is_edge_pixel(x, y - 14):
				var pixel_idx = y * 28 + x
				for h in range(start_neuron, min(end_neuron, 128)):
					weights[pixel_idx][h] += 0.1

static func create_hidden_output_weights() -> Array[Array]:
	"""
	Create 128x10 hidden-to-output weight matrix.
	Connects hidden neurons to output digits (0-9) with grouped connections.
	Each group of ~13 hidden neurons primarily connects to one output digit.
	"""
	var weights: Array[Array] = []
	
	for i in range(128):
		var row: Array[float] = []
		for j in range(10):
			# Connect each group of hidden neurons to their corresponding digit
			var base_weight = randf_range(-0.1, 0.1)
			
			# Enhance connections: neurons 0-9 → digit 0, neurons 10-19 → digit 1, etc.
			var neuron_group = i / 13  # Roughly group neurons
			if neuron_group == j:
				base_weight += 0.5  # Strong positive connection
			elif abs(neuron_group - j) == 1:
				base_weight += 0.1  # Weak positive connection to neighbors
			
			row.append(base_weight)
		weights.append(row)
	
	return weights

static func create_hidden_bias() -> Array[float]:
	"""
	Create hidden layer bias vector with small random values.
	Biases help neurons activate even with zero input.
	"""
	var bias: Array[float] = []
	for i in range(128):
		bias.append(randf_range(-0.05, 0.05))
	return bias

static func create_output_bias() -> Array[float]:
	"""
	Create output layer bias vector with small random values.
	Provides baseline activation levels for each digit class (0-9).
	"""
	var bias: Array[float] = []
	for i in range(10):
		bias.append(randf_range(-0.1, 0.1))
	return bias
