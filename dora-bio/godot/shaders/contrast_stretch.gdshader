shader_type canvas_item;

// Displays a single-channel 32-bit float texture (Image.FORMAT_RF) as grayscale.
// The display window [low, high] is shared by the raw, baseline and DORA panels, so brightness
// differences between them are real and not an artifact of per-panel auto-contrast.
// This only affects display. The data itself is never quantized to 8 bits.

uniform float low = 0.0;
uniform float high = 1.0;
uniform float gamma : hint_range(0.2, 3.0) = 1.0;

void fragment() {
	float value = texture(TEXTURE, UV).r;
	float t = clamp((value - low) / max(high - low, 1e-6), 0.0, 1.0);
	COLOR = vec4(vec3(pow(t, 1.0 / gamma)), 1.0);
}
