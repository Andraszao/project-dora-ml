extends RefCounted
class_name MNISTLoader

## MNIST Dataset Loader
## Handles loading and preprocessing of the MNIST handwritten digit dataset.
## MNIST format specification: http://yann.lecun.com/exdb/mnist/

static func read_big_endian_32(file: FileAccess) -> int:
	"""
	Read a 32-bit integer from file in big-endian format.
	MNIST files use big-endian byte order, which needs conversion on most systems.
	"""
	var byte1 = file.get_8()
	var byte2 = file.get_8()  
	var byte3 = file.get_8()
	var byte4 = file.get_8()
	return (byte1 << 24) | (byte2 << 16) | (byte3 << 8) | byte4

static func load_mnist_images(file_path: String, max_samples: int = 60000) -> Array:
	"""
	Load MNIST images from idx3-ubyte binary format.
	Returns array of 784-element float arrays (28x28 images, normalized 0-1).
	max_samples limits how many images to load (default 60,000).
	"""
	print("📂 Loading MNIST images from: ", file_path)
	
	var file = FileAccess.open(file_path, FileAccess.READ)
	if not file:
		print("❌ Failed to open file: ", file_path)
		return []
	
	# Read header (MNIST uses big-endian format)
	var magic = read_big_endian_32(file)  # Should be 2051 for images
	var num_images = read_big_endian_32(file)
	var num_rows = read_big_endian_32(file)
	var num_cols = read_big_endian_32(file)
	
	print("📊 MNIST Images Header:")
	print("   Magic: ", magic, " (should be 2051)")
	print("   Images: ", num_images)
	print("   Size: ", num_rows, "x", num_cols)
	
	if magic != 2051:
		print("❌ Invalid magic number for images file")
		file.close()
		return []
	
	var images = []
	print("🔄 Loading images in batches...")
	for i in range(min(max_samples, num_images)):  # Use configurable max_samples
		var image_data: Array[float] = []
		
		# Read 28x28 pixels (784 bytes)
		for pixel in range(784):
			var byte_val = file.get_8()
			var normalized = float(byte_val) / 255.0  # Normalize to 0-1
			image_data.append(normalized)
		
		images.append(image_data)
		
		if i % 500 == 0:  # Log every 500 instead of 100 for speed
			print("   Loaded ", i + 1, "/", min(max_samples, num_images), " images...")
	
	file.close()
	print("✅ Loaded ", images.size(), " images successfully!")
	return images

static func load_mnist_labels(file_path: String, max_samples: int = 60000) -> Array:
	"""
	Load MNIST labels from idx1-ubyte binary format.
	Returns array of integers (0-9) corresponding to digit labels.
	max_samples limits how many labels to load (default 60,000).
	"""
	print("📂 Loading MNIST labels from: ", file_path)
	
	var file = FileAccess.open(file_path, FileAccess.READ)
	if not file:
		var error = FileAccess.get_open_error()
		print("❌ Failed to open file: ", file_path)
		print("   Error code: %d" % error)
		print("   Make sure MNIST data files are in res://training_data/ directory")
		return []
	
	# Read header (MNIST uses big-endian format)
	var magic = read_big_endian_32(file)  # Should be 2049 for labels
	var num_labels = read_big_endian_32(file)
	
	print("📊 MNIST Labels Header:")
	print("   Magic: ", magic, " (should be 2049)")
	print("   Labels: ", num_labels)
	
	if magic != 2049:
		print("❌ Invalid magic number for labels file")
		file.close()
		return []
	
	var labels = []
	
	print("🔄 Loading labels...")
	for i in range(min(max_samples, num_labels)):
		var label = file.get_8()  # Single byte 0-9
		labels.append(int(label))
		
		if i % 500 == 0:  # Log every 500 instead of 100 for speed
			print("   Loaded ", i + 1, "/", min(max_samples, num_labels), " labels...")
	
	file.close()
	print("✅ Loaded ", labels.size(), " labels successfully!")
	return labels

static func load_training_batch(batch_size: int = 1000) -> Dictionary:
	"""
	Load a batch of MNIST training data (images + labels).
	Returns dictionary with "images", "labels", and "count" keys.
	batch_size determines how many samples to load (default 1000).
	"""
	print("🎯 Loading MNIST training batch...")
	
	var images = load_mnist_images("res://training_data/train-images.idx3-ubyte", batch_size)
	var labels = load_mnist_labels("res://training_data/train-labels.idx1-ubyte", batch_size)
	
	if images.size() != labels.size():
		print("❌ Mismatch: ", images.size(), " images vs ", labels.size(), " labels")
		return {}
	
	print("✅ Training batch ready: ", images.size(), " samples (requested: ", batch_size, ")")
	
	# Show some sample data
	if images.size() > 0:
		print("📋 Sample data:")
		for i in range(min(5, images.size())):
			var image = images[i]
			var label = labels[i]
			var pixel_count = 0
			for pixel in image:
				if pixel > 0.1:
					pixel_count += 1
			print("   Sample ", i, ": Label=", label, ", Non-zero pixels=", pixel_count)
	
	return {
		"images": images,
		"labels": labels,
		"count": images.size()
	}

static func rotate_image(image_data: Array[float], angle_degrees: float) -> Array[float]:
	"""
	Rotate a 28x28 image by specified angle (clockwise).
	Uses nearest-neighbor interpolation for speed. Useful for data augmentation.
	angle_degrees: Rotation angle in degrees (positive = clockwise).
	"""
	var rotated: Array[float] = []
	var angle_rad = deg_to_rad(angle_degrees)
	var cos_a = cos(angle_rad)
	var sin_a = sin(angle_rad)
	var center = 13.5  # Center of 28x28 image
	
	for y in range(28):
		for x in range(28):
			# Translate to center, rotate, translate back
			var cx = x - center
			var cy = y - center
			var rx = cx * cos_a - cy * sin_a + center
			var ry = cx * sin_a + cy * cos_a + center
			
			# Sample from original image (nearest neighbor)
			var orig_x = int(round(rx))
			var orig_y = int(round(ry))
			
			if orig_x >= 0 and orig_x < 28 and orig_y >= 0 and orig_y < 28:
				rotated.append(image_data[orig_y * 28 + orig_x])
			else:
				rotated.append(0.0)  # Black background
	
	return rotated

static func add_noise(image_data: Array[float], noise_level: float = 0.1) -> Array[float]:
	"""
	Add Gaussian noise to image for robustness training.
	Helps network handle imperfect user drawings and varied input quality.
	noise_level: Standard deviation of noise (0.0-1.0, typical: 0.05-0.2).
	"""
	var noisy: Array[float] = []
	for pixel in image_data:
		var noise = randf_range(-noise_level, noise_level)
		var noisy_pixel = clamp(pixel + noise, 0.0, 1.0)
		noisy.append(noisy_pixel)
	return noisy

static func scale_image(image_data: Array[float], scale_factor: float) -> Array[float]:
	"""
	Scale image by specified factor while maintaining 28x28 output size.
	scale_factor >1.0 zooms in (crops), <1.0 zooms out (pads with black).
	Useful for augmentation to handle different digit sizes.
	"""
	var scaled: Array[float] = []
	var center = 13.5
	
	for y in range(28):
		for x in range(28):
			# Map back to original coordinates
			var orig_x = (x - center) / scale_factor + center
			var orig_y = (y - center) / scale_factor + center
			
			# Sample from original (nearest neighbor)
			var ix = int(round(orig_x))
			var iy = int(round(orig_y))
			
			if ix >= 0 and ix < 28 and iy >= 0 and iy < 28:
				scaled.append(image_data[iy * 28 + ix])
			else:
				scaled.append(0.0)  # Black background
	
	return scaled

static func augment_training_data(images: Array, labels: Array, augmentation_factor: int = 2) -> Dictionary:
	"""
	Create augmented training data with rotations, scaling, and noise.
	Multiplies dataset size by augmentation_factor to improve generalization.
	Each variation includes random rotation (-15° to +15°), scaling (0.85-1.15x), and light noise.
	Returns dictionary with augmented "images" and "labels" arrays.
	"""
	print("🔄 Augmenting training data (factor: ", augmentation_factor, ")...")
	
	var augmented_images = []
	var augmented_labels = []
	
	# Add original data first
	for i in range(images.size()):
		augmented_images.append(images[i])
		augmented_labels.append(labels[i])
	
	# Generate augmentations
	for factor in range(1, augmentation_factor):
		for i in range(images.size()):
			var original_image = images[i]
			var label = labels[i]
			
			# Random rotation (-15 to +15 degrees)
			var angle = randf_range(-15.0, 15.0)
			var rotated = rotate_image(original_image, angle)
			
			# Random scaling (0.85 to 1.15)
			var scale = randf_range(0.85, 1.15)
			var scaled = scale_image(rotated, scale)
			
			# Light noise (improves robustness)
			var noisy = add_noise(scaled, 0.05)
			
			augmented_images.append(noisy)
			augmented_labels.append(label)
	
	print("✅ Data augmentation complete: ", images.size(), " → ", augmented_images.size(), " samples")
	return {
		"images": augmented_images,
		"labels": augmented_labels
	}

static func add_progressive_noise_training(images: Array, labels: Array, noise_ratio: float = 0.3, max_noise: float = 0.2, steps: int = 5) -> Dictionary:
	"""
	Add progressively noisy versions of training data for robustness.
	Inspired by diffusion models - teaches network to handle degraded or incomplete inputs.
	noise_ratio: Fraction of dataset to augment with noise (0.3 = 30%).
	max_noise: Maximum noise intensity (0.2 = 20%).
	steps: Number of progressive noise levels from light to heavy.
	Returns enhanced dataset with original + noisy variations.
	"""
	print("🌊 Adding progressive noise training (ratio: %.1f%%, max_noise: %.1f%%, steps: %d)..." % [noise_ratio * 100, max_noise * 100, steps])
	
	var enhanced_images = []
	var enhanced_labels = []
	
	# Add original clean data
	for i in range(images.size()):
		enhanced_images.append(images[i])
		enhanced_labels.append(labels[i])
	
	# Calculate how many samples to make noisy
	var noisy_sample_count = int(images.size() * noise_ratio)
	var samples_per_step = noisy_sample_count / steps
	
	# Add progressively noisier versions
	for step in range(steps):
		var noise_level = (float(step + 1) / float(steps)) * max_noise
		
		for i in range(samples_per_step):
			var sample_idx = i % images.size()
			var original_image = images[sample_idx]
			var label = labels[sample_idx]
			
			# Add graduated noise
			var noisy_image = add_noise(original_image, noise_level)
			
			# Optional: Add partial occlusion (simulate incomplete drawings)
			if randf() < 0.3:  # 30% chance
				noisy_image = add_partial_occlusion(noisy_image)
			
			enhanced_images.append(noisy_image)
			enhanced_labels.append(label)
	
	print("✅ Noise robustness training added: ", images.size(), " → ", enhanced_images.size(), " samples")
	print("📊 Noise levels: %.1f%% to %.1f%% across %d steps" % [max_noise / steps * 100, max_noise * 100, steps])
	
	return {
		"images": enhanced_images,
		"labels": enhanced_labels
	}

static func add_partial_occlusion(image_data: Array[float]) -> Array[float]:
	"""
	Randomly occlude parts of digit to simulate incomplete drawings.
	Trains network to recognize partially hidden digits (important for real-world use).
	Occlusion patterns: random patches, horizontal stripes, or corner blocks.
	"""
	var occluded: Array[float] = []
	
	# Create random occlusion patterns
	var occlusion_type = randi() % 3
	
	for y in range(28):
		for x in range(28):
			var pixel = image_data[y * 28 + x]
			var should_occlude = false
			
			match occlusion_type:
				0:  # Random patches
					should_occlude = randf() < 0.1
				1:  # Horizontal stripes
					should_occlude = (y % 7) < 2
				2:  # Corner occlusion
					should_occlude = (x < 7 and y < 7) or (x > 20 and y > 20)
			
			if should_occlude:
				pixel = 0.0  # Make it background
			
			occluded.append(pixel)
	
	return occluded

static func visualize_digit(image_data: Array[float], label: int = -1):
	"""
	Print ASCII art visualization of a 28x28 digit to console.
	Useful for debugging and verifying that data loaded correctly.
	label: The actual digit label (0-9) to display with visualization.
	"""
	print("🖼️ Digit visualization (Label: ", label, "):")
	
	for row in range(28):
		var line = "   "
		for col in range(28):
			var pixel = image_data[row * 28 + col]
			if pixel > 0.5:
				line += "██"
			elif pixel > 0.2:
				line += "▓▓"
			elif pixel > 0.1:
				line += "░░"
			else:
				line += "  "
		print(line)
	print("")
