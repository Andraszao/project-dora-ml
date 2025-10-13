extends RefCounted
class_name PatternClassifier

## Pattern-Based Digit Classifier
## Simple heuristic-based digit recognition using geometric pattern analysis.
## Fallback classifier when neural network isn't available.

func classify_digit(pixels: Array[float]) -> Dictionary:
	"""
	Classify a 28x28 digit image using pattern matching heuristics.
	Analyzes geometric features like edges, loops, and lines to identify digits.
	Returns dict with "prediction", "confidence", "reasoning", and "all_scores".
	"""
	if pixels.size() != 784:
		return {"prediction": -1, "confidence": 0.0, "reasoning": "Invalid input"}
	
	# Calculate basic features
	var features = extract_features(pixels)
	
	# Score each digit based on pattern matching
	var scores = []
	scores.append(score_digit_0(features))
	scores.append(score_digit_1(features))
	scores.append(score_digit_2(features))
	scores.append(score_digit_3(features))
	scores.append(score_digit_4(features))
	scores.append(score_digit_5(features))
	scores.append(score_digit_6(features))
	scores.append(score_digit_7(features))
	scores.append(score_digit_8(features))
	scores.append(score_digit_9(features))
	
	# Find best match
	var best_digit = 0
	var best_score = scores[0]
	for i in range(1, 10):
		if scores[i] > best_score:
			best_score = scores[i]
			best_digit = i
	
	# Convert to confidence (0.0 to 1.0)
	var confidence = min(best_score / 100.0, 1.0)
	
	# Create reasoning
	var reasoning = "Best match: %d (score: %.1f)" % [best_digit, best_score]
	
	return {
		"prediction": best_digit,
		"confidence": confidence,
		"reasoning": reasoning,
		"all_scores": scores
	}

func extract_features(pixels: Array[float]) -> Dictionary:
	"""
	Extract geometric features from 28x28 pixel image for pattern matching.
	Computes intensity distributions (top/middle/bottom, left/center/right),
	density, center region fill, and vertical line strength.
	Returns dictionary with feature measurements.
	"""
	var features = {}
	
	# Basic stats
	var total_pixels = 0.0
	var non_zero_pixels = 0
	for pixel in pixels:
		total_pixels += pixel
		if pixel > 0.1:
			non_zero_pixels += 1
	
	features["total_intensity"] = total_pixels
	features["pixel_count"] = non_zero_pixels
	features["density"] = float(non_zero_pixels) / 784.0
	
	# Analyze vertical distribution (where are the pixels vertically?)
	var top_pixels = 0.0    # Rows 0-9
	var middle_pixels = 0.0 # Rows 10-17
	var bottom_pixels = 0.0 # Rows 18-27
	
	for row in range(28):
		for col in range(28):
			var pixel = pixels[row * 28 + col]
			if row < 10:
				top_pixels += pixel
			elif row < 18:
				middle_pixels += pixel
			else:
				bottom_pixels += pixel
	
	features["top_intensity"] = top_pixels
	features["middle_intensity"] = middle_pixels
	features["bottom_intensity"] = bottom_pixels
	
	# Analyze horizontal distribution
	var left_pixels = 0.0   # Cols 0-9
	var center_pixels = 0.0 # Cols 10-17
	var right_pixels = 0.0  # Cols 18-27
	
	for row in range(28):
		for col in range(28):
			var pixel = pixels[row * 28 + col]
			if col < 10:
				left_pixels += pixel
			elif col < 18:
				center_pixels += pixel
			else:
				right_pixels += pixel
	
	features["left_intensity"] = left_pixels
	features["center_intensity"] = center_pixels
	features["right_intensity"] = right_pixels
	
	# Center region analysis (important for 0, 6, 8, 9)
	var center_region = 0.0
	for row in range(10, 18):
		for col in range(10, 18):
			center_region += pixels[row * 28 + col]
	
	features["center_region"] = center_region
	
	# Vertical line detection (important for 1)
	var vertical_strength = 0.0
	for row in range(28):
		for col in range(12, 16):  # Center columns
			vertical_strength += pixels[row * 28 + col]
	
	features["vertical_strength"] = vertical_strength
	
	return features

# Individual digit scoring functions
func score_digit_0(features: Dictionary) -> float:
	"""
	Score how likely the features match digit 0.
	Looks for: balanced distribution on all sides, hollow center, closed loop shape.
	"""
	var score = 0.0
	
	# 0 should have balanced distribution on all sides (oval/circle)
	score += features["top_intensity"] * 0.25
	score += features["bottom_intensity"] * 0.25
	score += features["left_intensity"] * 0.25
	score += features["right_intensity"] * 0.25
	
	# Bonus for having pixels in all regions (closed shape)
	if (features["top_intensity"] > 5 and features["bottom_intensity"] > 5 and 
		features["left_intensity"] > 5 and features["right_intensity"] > 5):
		score += 30  # Big bonus for closed shape
	
	# Penalize too much center fill (should be hollow)
	if features["center_region"] > features["total_intensity"] * 0.4:
		score *= 0.7
	else:
		score += 10  # Bonus for being hollow
	
	# Penalize extreme vertical concentration (that's more like a 1)
	if features["center_intensity"] > (features["left_intensity"] + features["right_intensity"]) * 1.5:
		score *= 0.3  # Heavy penalty for vertical line
	
	return score

func score_digit_1(features: Dictionary) -> float:
	"""
	Score how likely the features match digit 1.
	Looks for: strong vertical center line, narrow width, minimal horizontal spread.
	"""
	var score = 0.0
	
	# 1 should be strongly vertical and narrow
	score += features["vertical_strength"] * 1.0
	score += features["center_intensity"] * 0.8
	
	# Big bonus for being narrow (center concentrated)
	if features["center_intensity"] > (features["left_intensity"] + features["right_intensity"]):
		score += 25  # Major bonus for central concentration
	
	# Penalize too much horizontal spread (that's not a 1)
	if features["left_intensity"] > features["center_intensity"] * 0.3:
		score *= 0.5
	if features["right_intensity"] > features["center_intensity"] * 0.3:
		score *= 0.5
	
	# Penalize if it's too closed/circular (that's a 0)
	if (features["top_intensity"] > 15 and features["bottom_intensity"] > 15 and 
		features["left_intensity"] > 10 and features["right_intensity"] > 10):
		score *= 0.2  # Heavy penalty for closed shapes
	
	return score

func score_digit_2(features: Dictionary) -> float:
	"""
	Score how likely the features match digit 2.
	Looks for: strong horizontal lines at top and bottom, moderate middle.
	"""
	var score = 0.0
	
	# 2 should have top and bottom horizontal lines
	score += features["top_intensity"] * 0.4
	score += features["bottom_intensity"] * 0.4
	score += features["middle_intensity"] * 0.2
	
	return score

func score_digit_3(features: Dictionary) -> float:
	"""
	Score how likely the features match digit 3.
	Looks for: strong right side, moderate left, top and bottom emphasis.
	"""
	var score = 0.0
	
	# 3 should have strong right side, moderate left
	score += features["right_intensity"] * 0.5
	score += features["top_intensity"] * 0.25
	score += features["bottom_intensity"] * 0.25
	
	return score

func score_digit_4(features: Dictionary) -> float:
	"""
	Score how likely the features match digit 4.
	Looks for: strong vertical line, horizontal crossbar in middle, left emphasis.
	"""
	var score = 0.0
	
	# 4 should have strong vertical line and horizontal line in middle
	score += features["vertical_strength"] * 0.4
	score += features["middle_intensity"] * 0.3
	score += features["left_intensity"] * 0.3
	
	return score

func score_digit_5(features: Dictionary) -> float:
	"""
	Score how likely the features match digit 5.
	Looks for: balanced top, middle, and bottom with slightly different proportions than 2.
	"""
	var score = 0.0
	
	# 5 similar to 2 but different proportions
	score += features["top_intensity"] * 0.3
	score += features["middle_intensity"] * 0.4
	score += features["bottom_intensity"] * 0.3
	
	return score

func score_digit_6(features: Dictionary) -> float:
	"""
	Score how likely the features match digit 6.
	Looks for: strong bottom half with closed loop, left emphasis, center region fill.
	"""
	var score = 0.0
	
	# 6 should have strong bottom half, closed loop
	score += features["bottom_intensity"] * 0.5
	score += features["left_intensity"] * 0.3
	score += features["center_region"] * 0.2
	
	return score

func score_digit_7(features: Dictionary) -> float:
	"""
	Score how likely the features match digit 7.
	Looks for: strong top horizontal line, right side emphasis, minimal bottom.
	"""
	var score = 0.0
	
	# 7 should be mostly top horizontal with diagonal
	score += features["top_intensity"] * 0.6
	score += features["right_intensity"] * 0.4
	
	# Penalize too much bottom
	if features["bottom_intensity"] > features["top_intensity"] * 0.3:
		score *= 0.6
	
	return score

func score_digit_8(features: Dictionary) -> float:
	"""
	Score how likely the features match digit 8.
	Looks for: balanced distribution everywhere (double loop), all regions active.
	"""
	var score = 0.0
	
	# 8 should have balanced distribution everywhere
	score += features["top_intensity"] * 0.25
	score += features["middle_intensity"] * 0.25
	score += features["bottom_intensity"] * 0.25
	score += features["center_region"] * 0.25
	
	return score

func score_digit_9(features: Dictionary) -> float:
	"""
	Score how likely the features match digit 9.
	Looks for: strong top half with loop, center region, right side, open bottom.
	"""
	var score = 0.0
	
	# 9 should have strong top half, open bottom
	score += features["top_intensity"] * 0.5
	score += features["center_region"] * 0.3
	score += features["right_intensity"] * 0.2
	
	return score
