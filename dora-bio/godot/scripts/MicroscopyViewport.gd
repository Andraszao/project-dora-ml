extends Control
## DORA v2.0-bio UI shell.
##
## This script only wires widgets to the C# DoraBioBridge. All ML state and math live in C#
## (DoraBioBridge.cs → DoraBio.Core). C# members keep their PascalCase names from GDScript,
## e.g. bridge.StartDenoise() and bridge.DenoiseFinished.connect(...).
##
## Node references are @export properties set in Main.tscn, not v1-style get_node("../...") paths,
## so moving nodes around in the editor doesn't break the wiring.

const SAMPLE_PATH := "res://samples/synthetic_cells.tif"
const STRETCH_SHADER := preload("res://shaders/contrast_stretch.gdshader")

@export var bridge: Node
@export var raw_view: TextureRect
@export var baseline_view: TextureRect
@export var dora_view: TextureRect
@export var raw_caption: Label
@export var baseline_caption: Label
@export var dora_caption: Label
@export var model_label: Label
@export var status_label: Label
@export var log_view: RichTextLabel
@export var load_sample_button: Button
@export var open_button: Button
@export var denoise_button: Button
@export var adapt_button: Button
@export var adapt_steps: SpinBox
@export var train_synthetic_button: Button
@export var train_folder_button: Button
@export var epochs_box: SpinBox
@export var cancel_button: Button
@export var save_button: Button
@export var gamma_slider: HSlider
@export var file_dialog: FileDialog
@export var folder_dialog: FileDialog

## One material shared by all three panels, so they always use the same display window.
var display_material := ShaderMaterial.new()

## Emoji go to the console but are stripped from the on-screen log: Godot's default font has no emoji
## glyphs, and machines without a system emoji font would show empty boxes.
var _emoji := RegEx.create_from_string("[\\x{1F000}-\\x{1FAFF}\\x{2300}-\\x{23FF}\\x{2600}-\\x{27BF}\\x{2B00}-\\x{2BFF}\\x{FE0F}]\\s?")


func _ready() -> void:
	display_material.shader = STRETCH_SHADER
	for view in [raw_view, baseline_view, dora_view]:
		view.material = display_material

	bridge.ModelChanged.connect(_on_model_changed)
	bridge.ImageLoaded.connect(_on_image_loaded)
	bridge.DenoiseFinished.connect(_on_denoise_finished)
	bridge.EpochCompleted.connect(_on_epoch_completed)
	bridge.TrainingFinished.connect(_on_training_finished)
	bridge.AdaptFinished.connect(_on_adapt_finished)
	bridge.BusyChanged.connect(_on_busy_changed)
	bridge.Log.connect(log_line)
	bridge.JobFailed.connect(_on_job_failed)

	load_sample_button.pressed.connect(load_path.bind(SAMPLE_PATH))
	open_button.pressed.connect(file_dialog.popup_centered_ratio.bind(0.7))
	file_dialog.file_selected.connect(load_path)
	denoise_button.pressed.connect(_on_denoise_pressed)
	adapt_button.pressed.connect(func(): bridge.StartAdapt(int(adapt_steps.value)))
	train_synthetic_button.pressed.connect(func(): bridge.StartSyntheticTraining(int(epochs_box.value), false))
	train_folder_button.pressed.connect(folder_dialog.popup_centered_ratio.bind(0.7))
	folder_dialog.dir_selected.connect(func(dir): bridge.StartTrainingOnFolder(dir, int(epochs_box.value), false))
	cancel_button.pressed.connect(func(): bridge.Cancel())
	save_button.pressed.connect(func(): bridge.SaveCheckpoint())
	gamma_slider.value_changed.connect(func(v): display_material.set_shader_parameter("gamma", v))

	# The bridge's _Ready ran before ours (children are ready first), so ask for the current state.
	_on_model_changed(bridge.DescribeModel())
	_on_busy_changed(false, "")
	log_line("🧬 DORA v2.0-bio ready. Load the sample, then Denoise.")


func load_path(path: String) -> void:
	if bridge.LoadImage(path):
		baseline_view.texture = null
		dora_view.texture = null
		baseline_caption.text = "Bilateral baseline"
		dora_caption.text = "DORA"
		_update_buttons(false)


func _on_denoise_pressed() -> void:
	bridge.StartDenoise()


func _on_model_changed(description: String) -> void:
	model_label.text = description
	model_label.tooltip_text = description
	model_label.mouse_filter = Control.MOUSE_FILTER_PASS


func _on_image_loaded(image: Image, low: float, high: float, description: String) -> void:
	raw_view.texture = ImageTexture.create_from_image(image)
	display_material.set_shader_parameter("low", low)
	display_material.set_shader_parameter("high", high)
	raw_caption.text = "Raw · %s" % description
	log_line("📂 %s (display window %.3f–%.3f)" % [description, low, high])


func _on_denoise_finished(denoised: Image, baseline: Image, noise_raw: float, noise_dora: float, noise_baseline: float, seconds: float) -> void:
	dora_view.texture = ImageTexture.create_from_image(denoised)
	baseline_view.texture = ImageTexture.create_from_image(baseline)
	baseline_caption.text = "Bilateral baseline · est. σ %.4f" % noise_baseline
	dora_caption.text = "DORA · est. σ %.4f (%.2fs)" % [noise_dora, seconds]
	log_line("✨ denoised in %.2fs · estimated noise σ: raw %.4f → bilateral %.4f, DORA %.4f" % [seconds, noise_raw, noise_baseline, noise_dora])


func _on_epoch_completed(epoch: int, val_loss: float, val_psnr: float, gain: float, lr: float, events: String) -> void:
	var psnr_text := "" if is_nan(val_psnr) else " · val PSNR %.2f dB (+%.2f)" % [val_psnr, gain]
	log_line("📊 epoch %d · val loss %.5f%s · lr %.2g  %s" % [epoch, val_loss, psnr_text, lr, events])


func _on_training_finished(summary: String) -> void:
	log_line("🏁 training %s. Weights saved to user://." % summary)
	if bridge.HasImage():
		bridge.StartDenoise()


func _on_adapt_finished(steps: int, first_loss: float, last_loss: float) -> void:
	log_line("🎯 adapted %d blind-spot steps · loss %.5f → %.5f (not saved yet)" % [steps, first_loss, last_loss])
	bridge.StartDenoise()


func _on_busy_changed(busy: bool, activity: String) -> void:
	status_label.text = ("Working: %s…" % activity) if busy else "Ready"
	_update_buttons(busy)


func _on_job_failed(message: String) -> void:
	log_line("[color=#ff7777]❌ %s[/color]" % message)


func _update_buttons(busy: bool) -> void:
	var has_image: bool = bridge.HasImage()
	for b in [load_sample_button, open_button, train_synthetic_button, train_folder_button, save_button]:
		b.disabled = busy
	denoise_button.disabled = busy or not has_image
	adapt_button.disabled = busy or not has_image
	cancel_button.disabled = not busy


func log_line(message: String) -> void:
	print_rich(message)
	log_view.append_text(_emoji.sub(message, "", true).strip_edges() + "\n")
